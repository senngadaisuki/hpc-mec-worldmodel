import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from models.attention import ContinuousPositionBias, Transformer


def min_max_normalize(tensor, dim=-1):
    min_vals = torch.amin(tensor, dim=dim, keepdim=True)
    max_vals = torch.amax(tensor, dim=dim, keepdim=True)
    normalized = (tensor - min_vals) / (max_vals - min_vals + 1e-8)
    return normalized


def estimate_theta(phases, periods, search_range=(-4, 4)):
    omegas = 2 * torch.pi / periods  # Calculate angular frequency ω_i
    psi = phases + torch.pi  # Adjust phase to [0, 2π)
    
    k_min, k_max = search_range
    num_cells = phases.size(-1)
    batch_size = phases.size(0)

    # Generate all possible k_i combinations as a tensor
    k_values = torch.arange(k_min, k_max + 1, device=phases.device)
    all_k_combinations = torch.cartesian_prod(*[k_values] * num_cells)  # (num_combinations, num_cells)
    
    # Expand dimensions for broadcasting
    all_k_combinations = all_k_combinations.unsqueeze(0).repeat(batch_size, 1, 1)  # (batch_size, num_combinations, num_cells)
    psi = psi.unsqueeze(1)  # (batch_size, 1, num_cells)
    omegas = omegas.unsqueeze(0).unsqueeze(1)  # (1, 1, num_cells)

    # Calculate θ for all k combinations
    theta_estimates = (psi + 2 * torch.pi * all_k_combinations) / omegas  # (batch_size, num_combinations, num_cells)
    theta_mean = theta_estimates.mean(dim=-1, keepdim=True)  # (batch_size, num_combinations, 1)

    # Calculate error
    error = ((theta_estimates - theta_mean) ** 2).sum(dim=-1)  # (batch_size, num_combinations)

    # Find the best θ with minimum error
    min_error, best_indices = torch.min(error, dim=1)  # (batch_size,)
    best_theta = theta_mean[torch.arange(batch_size), best_indices].squeeze(-1)  # (batch_size,)
    best_k = all_k_combinations[torch.arange(batch_size), best_indices]  # (batch_size, num_cells)

    # Normalize θ to [-π, π)
    theta_estimated = (best_theta + torch.pi) % (2 * torch.pi) - torch.pi

    return theta_estimated


class Grid_CANN1D(nn.Module):
    def __init__(self, L, neuron_num=100, k_mec=4., z_min=-torch.pi, z_max=torch.pi,
               tau=1., a_g=0.5, J0=4., dt=0.1, batch_first=True):
        super(Grid_CANN1D, self).__init__()

        # parameters
        self.L = L      # period
        self.omega = 2 * torch.pi / self.L
        self.tau = tau
        self.k = k_mec  # Global inhibition
        self.a = a_g  # Range of excitatory connections
        self.J0 = J0  # maximum connection value
        self.neuron_num = neuron_num
        self.z_min = z_min
        self.z_max = z_max
        self.z_range = z_max - z_min
        self.dt = dt

        # variables
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.x = torch.tensor(np.linspace(self.z_min, self.z_max, self.neuron_num, endpoint=False), device=self.device)  # The encoded feature values
        self.r = torch.zeros(neuron_num, device=self.device)
        self.u = torch.zeros(neuron_num, device=self.device)
        self.input = torch.zeros(neuron_num, device=self.device)
        self.center = torch.zeros(1, device=self.device)
        self.center_input = torch.zeros(1, device=self.device)
        # Connections
        self.conn_mat = self.make_conn()
        self.conn_fft = torch.fft.fft(self.conn_mat)

        self.batch_first = batch_first
    
    def xtopi(self, x):
        # x = [-np.pi, np.pi]
        phi = (x * self.omega) % (2 * torch.pi) - torch.pi
        return phi

    def period_bound(self, A):
        d = torch.remainder(A, self.z_range)
        d = torch.where(d > 0.5 * self.z_range, d - self.z_range, d)
        return d

    def make_conn(self):
        d = self.period_bound(torch.abs(self.x[0] - self.x))
        Jxx = self.J0 / (self.a * torch.sqrt(torch.tensor(2 * torch.pi))) * torch.exp(-0.5 * torch.square(d / self.a))
        return Jxx

    def get_input_pos(self, pos, input_strength=1.):
        assert pos.dim() == 2 and pos.size(1) == 1, "pos shape should be (batch_size, 1)"
        batch_size = pos.size(0)
        phase_input = self.xtopi(pos)
        d = self.period_bound(torch.abs(self.x.unsqueeze(0).expand(batch_size, -1) - phase_input))
        input = input_strength * torch.exp(-0.25 * torch.square(d / self.a))
        return input

    def reset_state(self):
        self.r = torch.zeros(self.neuron_num, device=self.device)
        self.u = torch.zeros(self.neuron_num, device=self.device)
        self.v = torch.zeros(self.neuron_num, device=self.device)
        self.input = torch.zeros(self.neuron_num, device=self.device)

    def get_center(self):
        exppos = torch.exp(1j * self.x)
        self.center = torch.angle(torch.sum(exppos * self.r, dim=-1)).float()
        self.center_input = torch.angle(torch.sum(exppos * self.input, dim=-1))

    def integration(self, x, v_mode=False):
        # Calculate recurrent input
        r_fft = torch.fft.fft(self.r, dim=-1)  # Specify the dimension for FFT
        Irec = torch.real(torch.fft.ifft(r_fft * self.conn_fft))  # Unsqueeze conn_fft to match batch size
        # v_theta input
        if v_mode:
            assert x.dim() == 2 and x.size(1) == 1, "In path integration, x shape should be (batch_size, 1)"
            # Compute v_phi in parallel
            v_phi = self.omega * torch.abs(x)
            batch_size, _ = x.size()
            shifts = torch.where(x > 0, 1, -1)
            rolled_u = torch.stack([
                torch.roll(self.u[i], shifts=shifts[i].item()) for i in range(batch_size)
            ])
            self.input = rolled_u * v_phi

        else:
            if x is not None:
                assert x.dim() == 2 and x.size(1) == self.neuron_num, f"In template matching, x shape should be (batch_size, {self.neuron_num})"
                self.input = x
            else:
                self.input = 0

        # Update neural state
        du = (-self.u + self.input + Irec) / self.tau * self.dt
        u = self.u + du
        self.u = torch.where(u > 0, u, 0)
        r1 = torch.square(self.u)
        r2 = 1.0 + self.k * torch.sum(r1, dim=-1)
        self.r = r1 / r2.unsqueeze(-1)
        # Calculate self position
        self.get_center()
        return self.u.float()
    
    def forward(self, x, u0=None, v_mode=False, last_time=100, blank_time=100, keep_trace=False):
        if u0 is not None:
            assert u0.dim() == 2 and u0.size(1) == self.neuron_num, f"u0 shape should be (batch_size, {self.neuron_num})"
            self.u = u0
            self.r = torch.square(self.u) / (1.0 + self.k * torch.sum(torch.square(self.u), dim=-1)).unsqueeze(-1)
        else:
            self.u = torch.zeros_like(x)
            self.r = torch.zeros_like(x)
        us = [] if keep_trace else None
        for _ in range(last_time):
            u = self.integration(x, v_mode=v_mode)
            if keep_trace:
                us.append(u)

        for _ in range(blank_time):
            u = self.integration(x=None)
            if keep_trace:
                us.append(u)
        
        if keep_trace:
            us = torch.stack(us)
            if self.batch_first:
                us = us.transpose(0, 1)

        return self.u.float(), min_max_normalize(self.r.float()), us.float() if keep_trace else None
     
    def to(self, device):
        self.device = device 
        self.x = self.x.to(device)
        self.r = self.r.to(device)
        self.u = self.u.to(device)
        self.input = self.input.to(device)
        self.center = self.center.to(device)
        self.center_input = self.center_input.to(device)

        self.conn_fft = self.conn_fft.to(device)
        return super().to(device)


class MEC_model(nn.Module):
    def __init__(self, module_lambda):
        super(MEC_model, self).__init__()
        self.module_lambda = module_lambda
        self.module_num = len(self.module_lambda)
        self.grid_neuron_num = 50
        self.group_cann = [Grid_CANN1D(L=self.module_lambda[i], neuron_num=self.grid_neuron_num) for i in range(self.module_num)]
    
    def forward(self, x, h0=None, v_mode=False, last_time=100, blank_time=100):
        # x: (batch_size, 1)
        us = []
        rs = []
        # path integration
        if v_mode:
            for i in range(self.module_num):
                x_subset = x
                h_0_subset = h0[:, self.grid_neuron_num*i:self.grid_neuron_num*(i+1)] if h0 is not None else None
                u, r, _ = self.group_cann[i](x_subset, h_0_subset, v_mode=True, last_time=last_time, blank_time=blank_time)
                us.append(u)
                rs.append(r)
        # template maching
        else:
            for i in range(self.module_num):
                x_subset = x[:, self.grid_neuron_num*i:self.grid_neuron_num*(i+1)]
                h_0_subset = h0[:, self.grid_neuron_num*i:self.grid_neuron_num*(i+1)] if h0 is not None else None
                u, r, _ = self.group_cann[i](x_subset, h_0_subset, v_mode=False, last_time=last_time, blank_time=blank_time)
                us.append(u)
                rs.append(r)
        us = torch.cat(us, dim=-1)
        rs = torch.cat(rs, dim=-1)
        return rs, us

    def get_group_input_pos(self, x, input_strength=1.):
        # x: (batch_size, 1)
        group_input = []  
        for i in range(self.module_num):
            group_input.append(self.group_cann[i].get_input_pos(x, input_strength=input_strength))  # (batch_size, module_lambda)
        x = torch.cat(group_input, dim=-1)
        return x    # (batch_size, sum module_lambda)

    def get_group_u_state(self):
        outputs = []
        for i in range(self.module_num):
            outputs.append(self.group_cann[i].u)
        return torch.cat(outputs, dim=-1).float()

    def get_group_readout(self):
        outputs = []
        for i in range(self.module_num):
            outputs.append(self.group_cann[i].center.unsqueeze(-1))
        return torch.cat(outputs, dim=-1).float()

    def to(self, device):
        self.device = device
        for i in range(self.module_num):
            self.group_cann[i] = self.group_cann[i].to(device)
        return super().to(device)


class CANNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hidden_depth):
        super(CANNEncoder, self).__init__()
        
        # Initial projection to intermediate dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Hidden layers with skip connections
        self.hidden_layers = nn.ModuleList()
        for i in range(hidden_depth):
            self.hidden_layers.append(nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ))
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        # Initial projection
        h = self.input_proj(x)
        
        # Apply hidden layers with residual connections
        for layer in self.hidden_layers:
            h = h + layer(h)
        
        # Final projection
        out = self.output_proj(h)
        
        return out
    
    
class CANNDecoder(nn.Module):
    def __init__(self, n_factors, hidden_dim, output_dim, hidden_depth):
        super(CANNDecoder, self).__init__()
        
        # Initial projection to hidden dimension
        self.input_proj = nn.Linear(n_factors, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        # Hidden layers with skip connections
        self.hidden_layers = nn.ModuleList()
        for i in range(hidden_depth):
            self.hidden_layers.append(nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ))
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, z):
        # Ensure input shape is correct
        z_flat = z.view(z.size(0), z.size(1), -1)
        
        # Initial projection
        h = self.input_proj(z_flat)
        h = self.input_norm(h)
        h = F.relu(h)
        
        # Apply hidden layers with residual connections
        for i, layer in enumerate(self.hidden_layers):
            # Start with smaller residuals to stabilize training
            if i < 2:  
                h = h + 0.3 * layer(h)  # Reduced impact for first layers
            else:
                h = h + layer(h)
        
        # Final projection
        out = self.output_proj(h)
        
        return out
    

class MEC_Spatial_Temporal(nn.Module):
    def __init__(self, MEC_hidden_size, patch_size, spatial_depth=1, temporal_depth=1):
        super(MEC_Spatial_Temporal, self).__init__()
        self.MEC_hidden_size = MEC_hidden_size
        self.patch_size = patch_size
        self.spatial_depth = spatial_depth
        self.temporal_depth = temporal_depth
        dim_head = 64
        heads = 8
        enc_spatial_transformer_kwargs = dict(
            dim = self.MEC_hidden_size // (self.patch_size ** 2),
            dim_head = dim_head,
            heads = heads,
            attn_dropout = 0.,
            ff_dropout = 0.1,
            peg = True,
            peg_causal = True,
        )

        enc_temporal_transformer_kwargs = dict(
            dim = self.MEC_hidden_size // (self.patch_size ** 2),
            dim_head = dim_head,
            heads = heads,
            attn_dropout = 0.,
            ff_dropout = 0.1,
            causal = True,
            peg = True,
            peg_causal = True,
        )

        self.spatial_rel_pos_bias = ContinuousPositionBias(dim = self.MEC_hidden_size // (self.patch_size ** 2), heads = heads)
        self.enc_spatial_transformer = Transformer(depth = self.spatial_depth, **enc_spatial_transformer_kwargs)
        self.enc_temporal_transformer = Transformer(depth = self.temporal_depth, **enc_temporal_transformer_kwargs)
        
    
    def forward(self, x):
        b, t, h, w, d = x.shape
        
        # Process with spatial transformer
        tokens = rearrange(x, 'b t h w d -> (b t) (h w) d')
        attn_bias = self.spatial_rel_pos_bias(h, w, device=tokens.device)
        tokens = self.enc_spatial_transformer(
            tokens, 
            attn_bias=attn_bias, 
            video_shape=(b, t, h, w)
        )
        tokens = rearrange(tokens, '(b t) (h w) d -> b t h w d', b=b, t=t, h=h, w=w)
        
        # Process with temporal transformer
        tokens = rearrange(tokens, 'b t h w d -> (b h w) t d')
        tokens = self.enc_temporal_transformer(tokens, video_shape=(b, t, h, w))
        tokens = rearrange(tokens, '(b h w) t d -> b t h w d', b=b, t=t, h=h, w=w)

        return tokens
    
    def to(self, device):
        self.spatial_rel_pos_bias = self.spatial_rel_pos_bias.to(device)
        self.enc_spatial_transformer = self.enc_spatial_transformer.to(device)
        self.enc_temporal_transformer = self.enc_temporal_transformer.to(device)
        return super().to(device)
