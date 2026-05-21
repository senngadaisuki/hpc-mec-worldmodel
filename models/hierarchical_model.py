# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from models.inverse_model import Inverse_model
from models.forward_model import GatedResidualForward
from models.HPC_network import HPC_Spatial_Temporal
from models.MEC_network import CANNEncoder, CANNDecoder, MEC_Spatial_Temporal

from einops import rearrange
from einops.layers.torch import Rearrange

def off_diag(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def off_diag_cov_loss(x: torch.Tensor) -> torch.Tensor:
    cov = torch.cov(einops.rearrange(x, "... E -> E (...)"))
    return off_diag(cov).square().mean()

def wrap_interval(x, lower=-torch.pi, upper=torch.pi):
    L = upper - lower
    return torch.remainder(x - lower, L) + lower


HPC_DIM = 512
MEC_DIM = 256
ACTION_DIM = 128

class HPC_model(nn.Module):
    def __init__(self, batch_first=True):
        super(HPC_model, self).__init__()
        self.channels = 32
        self.H = 16
        self.W = 16
        # self.seq_len = 16
        self.patch_size = 4
        self.feature_dim = self.channels * self.H * self.W
        self.HPC_hidden_size = HPC_DIM * self.patch_size**2
        self.spatial_depth = 4
        self.temporal_depth = 4

        patch_height = self.H // self.patch_size
        patch_width = self.W // self.patch_size
        hidden_dim_per_patch = self.HPC_hidden_size // (self.patch_size**2)

        # HPC model
        self.HPC_inf_model = HPC_Spatial_Temporal(self.HPC_hidden_size, self.patch_size, self.spatial_depth, self.temporal_depth)

        self.frame2hidden = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t h w (c p1 p2)', c=self.channels, p1=patch_height, p2=patch_width),
            nn.LayerNorm(self.channels * patch_height * patch_width),
            nn.Linear(self.channels * patch_height * patch_width, self.HPC_hidden_size // (self.patch_size ** 2)),
            nn.ReLU(inplace=True),
            nn.Linear(self.HPC_hidden_size // (self.patch_size ** 2), self.HPC_hidden_size // (self.patch_size ** 2)),
            nn.LayerNorm(self.HPC_hidden_size // (self.patch_size ** 2))
        )

        self.hidden2frame = nn.Sequential(
            nn.Linear(self.HPC_hidden_size // (self.patch_size ** 2), self.HPC_hidden_size // (self.patch_size ** 2)),
            nn.ReLU(inplace=True),
            nn.Linear(self.HPC_hidden_size // (self.patch_size ** 2), self.channels * patch_height * patch_width),
            Rearrange('b t h w (c p1 p2) -> b t c (h p1) (w p2)', p1=patch_height, p2=patch_width)
        )

        self.batch_first = batch_first

    def forward(self, x):
        tokens_all = self.frame2hidden(x)
        p_inf = self.HPC_inf_model(tokens_all)
        s_inf = self.hidden2frame(p_inf)  # [b, t, c, h, w]

        return {
            'x': x,
            's_inf': s_inf,
            'p_inf': p_inf,
        }

    def _mse_loss(self, s_actual, s_pred):
        mse = nn.MSELoss()
        s_actual = s_actual
        s_pred = s_pred
        return mse(s_pred, s_actual)

    def _covariance_reg_loss(self, obs_enc: torch.Tensor):
        return off_diag_cov_loss(obs_enc)
    
    def _var_loss(self, Z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
        Z = Z - Z.mean(dim=0)
        std_z = torch.sqrt(Z.var(dim=0) + eps)
        return F.relu(gamma - std_z).mean()
    
    def _time_variance_loss(self, Z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
        Z = Z - Z.mean(dim=1, keepdim=True)
        std_z = torch.sqrt(Z.var(dim=1) + eps)
        return F.relu(gamma - std_z).mean()

    # Calculate total loss as weighted sum of individual losses
    def total_loss(self, res):
        loss_s_inf = self._mse_loss(res['x'], res['s_inf'])
        
        p_inf = rearrange(res['p_inf'], 'b t h w d -> b t (h w d)')
        loss_covariance = self._covariance_reg_loss(p_inf)# + self._covariance_reg_loss(g_inf)
        loss_var = self._var_loss(rearrange(res['p_inf'], 'b t h w d -> b (t h w d)'), gamma=0.5)

        losses_dict = {
            'sensory_inf': loss_s_inf,
            'covariance': loss_covariance,
            'variance': loss_var,
        }

        weights = {
            'sensory_inf': 5.0,
            'covariance': 0.05,  # 0.05
            'variance': 0.05,
        }

        # Calculate total weighted loss
        total_loss = sum([weights[k] * v for k, v in losses_dict.items()])
        return total_loss, losses_dict

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        """
        num_params = sum(p.numel() for p in self.parameters())
        return num_params
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')  # Kaiming initialization
            if module.bias is not None:
                nn.init.zeros_(module.bias)  # Initialize bias to 0
        elif isinstance(module, nn.ConvTranspose2d):
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')  # Kaiming initialization
            if module.bias is not None:
                nn.init.zeros_(module.bias)  # Initialize bias to 0
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)  # Orthogonal initialization for weights
                elif 'bias' in name:
                    nn.init.zeros_(param)  # Initialize bias to 0
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def to(self, device):
        super().to(device)
        self.HPC_inf_model.to(device)
        self.frame2hidden.to(device)
        self.hidden2frame.to(device)

        return self
    
    def configure_optimizers(self, weight_decay, lr, betas, T_max):
        # Configure main optimizer with AdamW
        world_optimizer = torch.optim.AdamW(self.parameters(), lr=lr['world_lr'], betas=betas, weight_decay=weight_decay)
        world_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(world_optimizer, T_max=T_max, eta_min=1e-5)

        return world_optimizer, world_scheduler



class MEC_model(nn.Module):
    def __init__(self, batch_first=True):
        super(MEC_model, self).__init__()
        self.patch_size = 4
        self.grid_dim = MEC_DIM * self.patch_size**2
        self.HPC_hidden_size = HPC_DIM * self.patch_size**2
        self.spatial_depth = 4
        self.temporal_depth = 4

        # MEC model
        self.MEC_inf_model = MEC_Spatial_Temporal(self.grid_dim, self.patch_size, self.spatial_depth, self.temporal_depth)

        self.hidden2grid = CANNEncoder(
            input_dim=self.HPC_hidden_size,
            hidden_dim=self.HPC_hidden_size,
            output_dim=self.grid_dim,
            hidden_depth=4
        )

        self.grid2hidden = CANNDecoder(
            n_factors=self.grid_dim,
            hidden_dim=self.HPC_hidden_size,
            output_dim=self.HPC_hidden_size,
            hidden_depth=4
        )

        # Initialize batch_first flag
        self.batch_first = batch_first

    def forward(self, p_inf):
        g_inf = self.hidden2grid(rearrange(p_inf, 'b t h w d -> b t (h w d)'))    # g_inf: [b t 4*4*256] -> [b t 128]
        g_inf = self.MEC_inf_model(rearrange(g_inf, 'b t (h w d) -> b t h w d', h=self.patch_size, w=self.patch_size))
        g_inf = rearrange(g_inf, 'b t h w d -> b t (h w d)', h=self.patch_size, w=self.patch_size)
        p_int = rearrange(self.grid2hidden(g_inf), 'b t (h w d) -> b t h w d', h = self.patch_size, w = self.patch_size)
        
        return {
            'p_inf': p_inf,
            'p_int': p_int,
            'g_inf': g_inf,
        }

    def _mse_loss(self, s_actual, s_pred):
        mse = nn.MSELoss()
        s_actual = s_actual
        s_pred = s_pred
        return mse(s_pred, s_actual)
    
    def _cosine_similarity_loss(self, s_actual, s_pred):
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(s_pred, s_actual, dim=-1)
        # Define loss: ideally, the similarity should be 1, so loss = 1 - cos_sim
        loss = (1 - cos_sim).mean()
        return loss
    
    def _covariance_reg_loss(self, obs_enc: torch.Tensor):
        return off_diag_cov_loss(obs_enc)
    
    def _var_loss(self, Z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
        Z = Z - Z.mean(dim=0)
        std_z = torch.sqrt(Z.var(dim=0) + eps)
        return F.relu(gamma - std_z).mean()
    
    def _time_variance_loss(self, Z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
        Z = Z - Z.mean(dim=1, keepdim=True)
        std_z = torch.sqrt(Z.var(dim=1) + eps)
        return F.relu(gamma - std_z).mean()

     # Calculate total loss as weighted sum of individual losses
    def total_loss(self, res):
        p_inf = rearrange(res['p_inf'], 'b t h w d -> b t (h w d)')
        p_int = rearrange(res['p_int'], 'b t h w d -> b t (h w d)')
        loss_p_inf_mse = self._mse_loss(p_inf, p_int)

        loss_p_inf_cos = self._cosine_similarity_loss(p_inf, p_int)
    
        # g_inf = torch.atan2(res['g_inf'][..., 1], res['g_inf'][..., 0])  # [B, T, N, 2] -> [B, T, N]
        g_inf = res['g_inf']
        loss_covariance = self._covariance_reg_loss(g_inf)
        loss_var = self._var_loss(rearrange(g_inf, 'b t d -> b (t d)'))
        loss_time_var = self._time_variance_loss(g_inf, gamma=0.1)

        losses_dict = {
            'place_inf_mse': loss_p_inf_mse,
            'place_inf_cos': loss_p_inf_cos,
            'covariance': loss_covariance,
            'variance': loss_var,
            'temporal_variance': loss_time_var,
        }

        weights = {
            'place_inf_mse': 1.,  # 0.05
            'place_inf_cos': 1.,
            'covariance': 0.05,  # 0.05
            'variance': 0.05,
            'temporal_variance': 0.05,
        }

        # Calculate total weighted loss
        total_loss = sum([weights[k] * v for k, v in losses_dict.items()])
        return total_loss, losses_dict
    
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        """
        num_params = sum(p.numel() for p in self.parameters())
        return num_params
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')  # Kaiming initialization
            if module.bias is not None:
                nn.init.zeros_(module.bias)  # Initialize bias to 0
        elif isinstance(module, nn.ConvTranspose2d):
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')  # Kaiming initialization
            if module.bias is not None:
                nn.init.zeros_(module.bias)  # Initialize bias to 0
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)  # Orthogonal initialization for weights
                elif 'bias' in name:
                    nn.init.zeros_(param)  # Initialize bias to 0
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def to(self, device):
        super().to(device)
        self.MEC_inf_model.to(device)
        self.hidden2grid.to(device)
        self.grid2hidden.to(device)
        return self
    
    def configure_optimizers(self, weight_decay, lr, betas, T_max):
        # Configure main optimizer with AdamW
        world_optimizer = torch.optim.AdamW(self.parameters(), lr=lr['world_lr'], betas=betas, weight_decay=weight_decay)
        world_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(world_optimizer, T_max=T_max, eta_min=1e-5)

        return world_optimizer, world_scheduler
    

class JointHPCMEC(nn.Module):
    def __init__(self, HPC_model, MEC_model):
        super(JointHPCMEC, self).__init__()
        self.HPC_model = HPC_model
        self.MEC_model = MEC_model
        
    def forward(self, x):
        # HPC forward pass
        hpc_result = self.HPC_model(x)
        p_inf = hpc_result['p_inf']
        
        # MEC forward pass
        mec_result = self.MEC_model(p_inf)
        
        # Add end-to-end reconstruction
        p_int = mec_result['p_int']
        s_recon = self.HPC_model.hidden2frame(p_int)
        
        return {
            'x': x,
            's_inf': hpc_result['s_inf'],
            's_recon': s_recon,  # end-to-end reconstruction
            'p_inf': p_inf,
            'p_int': p_int,
            'g_inf': mec_result['g_inf']
        }
        

    def smoothing_loss(self, g_inf):
        """
        Encourage that delta_g is smooth over time.
        """
        delta_g = g_inf[:, 1:] - g_inf[:, :-1]         # (b, t-1, d)
        delta_delta_g = delta_g[:, 1:] - delta_g[:, :-1]   # (b, t-2, d)
        loss = (delta_delta_g**2).mean()
        return loss


    def total_loss(self, res):
        # Direct sensory reconstruction loss
        loss_s_inf = self.HPC_model._mse_loss(res['x'], res['s_inf'])
        
        # Place cell reconstruction loss
        p_inf = rearrange(res['p_inf'], 'b t h w d -> b t (h w d)')
        p_int = rearrange(res['p_int'], 'b t h w d -> b t (h w d)')
        loss_p_inf = self.MEC_model._mse_loss(p_inf, p_int)
        
        # End-to-end reconstruction loss
        loss_s_recon = self.HPC_model._mse_loss(res['x'], res['s_recon'])
        
        # Regularization (apply lighter regularization)
        loss_covariance = self.MEC_model._covariance_reg_loss(res['g_inf']) + self.HPC_model._covariance_reg_loss(res['p_inf'])
        loss_var = self.MEC_model._var_loss(rearrange(res['g_inf'], 'b t d -> b (t d)'), gamma=0.5) + self.HPC_model._var_loss(rearrange(res['p_inf'], 'b t h w d -> b (t h w d)'), gamma=0.5)

        loss_smoothing = self.smoothing_loss(res['g_inf'])

        losses_dict = {
            'sensory_inf': loss_s_inf,
            'place_inf': loss_p_inf,
            'sensory_recon': loss_s_recon,
            'covariance': loss_covariance,
            'variance': loss_var,
            'smoothing': loss_smoothing
        }
        
        weights = {
            'sensory_inf': 5.0,
            'place_inf': .22,
            'sensory_recon': 5.0,  # End-to-end weight
            'covariance': 0.01,    # Reduced regularization
            'variance': 0.01,
            'smoothing': 0.01
        }
        
        total_loss = sum([weights[k] * v for k, v in losses_dict.items()])
        return total_loss, losses_dict
    
    def configure_optimizers(self, weight_decay, lr, betas, T_max):
        # Configure main optimizer with AdamW
        world_optimizer = torch.optim.AdamW(self.parameters(), lr=lr['world_lr'], betas=betas, weight_decay=weight_decay)
        world_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(world_optimizer, T_max=T_max, eta_min=1e-5)

        return world_optimizer, world_scheduler
    

class Inverse_World_model(nn.Module):
    def __init__(self, JointHPCMEC_model, batch_first=True):
        super(Inverse_World_model, self).__init__()

        self.HPC_model = JointHPCMEC_model.HPC_model
        self.MEC_model = JointHPCMEC_model.MEC_model

        self.channels = 32
        self.H = 16
        self.W = 16
        self.patch_size = 4
        self.feature_dim = self.channels * self.H * self.W
        self.action_dim = ACTION_DIM
        self.grid_dim = MEC_DIM * self.patch_size**2
        self.HPC_hidden_size = HPC_DIM * self.patch_size**2
        self.spatial_depth = 4
        self.temporal_depth = 4

        self.period = 1

        self.Inverse_model = Inverse_model(input_dim=MEC_DIM, action_dim=self.action_dim, hidden_dim=MEC_DIM//2)
        self.forward_dynamics = GatedResidualForward(g_dim=MEC_DIM, z_dim=self.action_dim, hidden_dim=MEC_DIM, depth=4, patch_size=self.patch_size)

        # Initialize batch_first flag
        self.batch_first = batch_first
    
    def forward(self, x, eta=None):
        batch_size, seq_len, *_ = x.size()

        hpc_result = self.HPC_model(x)
        p_inf = hpc_result['p_inf']
        s_inf = hpc_result['s_inf']
        
        # MEC forward pass
        mec_result = self.MEC_model(p_inf)
        g_inf = mec_result['g_inf']
        p_int = mec_result['p_int']
        s_recon = self.HPC_model.hidden2frame(p_int)

        g_inf_prev = g_inf[:, :-1].detach() # [b, t-1, g]
        g_inf_next = g_inf[:, 1:].detach()    # [b, t-1, g]
        a_low = self.Inverse_model(g_inf_prev, g_inf_next)

        # Generation Loop, autoregression
        generated_frames = []
        p_gen_list = [] 
        g_gen_list = []
        last_g_gen = g_inf[:, 0:1].detach()

        # Generate frames autoregressively
        for i in range(1, seq_len):
            if self.period is not None and i % self.period == 0:
                last_g = g_inf[:, i-1:i]
            else:
                last_g = last_g_gen
            next_g_gen = last_g + self.forward_dynamics(last_g, a_low[:, i-1:i])    # a_low_t, g_int_t -> g_gen_t+1
            next_p_gen = rearrange(self.MEC_model.grid2hidden(next_g_gen), 'b t (h w d) -> b t h w d', h = self.patch_size, w = self.patch_size)
            next_s_gen = self.HPC_model.hidden2frame(next_p_gen)

            generated_frames.append(next_s_gen)
            p_gen_list.append(next_p_gen)
            g_gen_list.append(next_g_gen)

            # Autoregressive update
            last_g_gen = next_g_gen

        # Concatenate all generated frames
        s_gen = torch.cat(generated_frames, dim=1) 
        p_gen = torch.cat(p_gen_list, dim=1)
        g_gen = torch.cat(g_gen_list, dim=1) 

        return {
            'x': x,
            's_inf': s_inf,
            's_gen': s_gen,
            's_recon': s_recon,
            'p_inf': p_inf,
            'p_gen': p_gen,
            'p_int': p_int,
            'g_inf': g_inf,
            'g_gen': g_gen,
            'a_low': a_low,
        }

    def _predict(self, x, period: int | None = 1):
        """Shared rollout used by `one_step` / `autoregressive`.

        `period` controls how often the ground-truth latent is fed back:
          - period=1    : feed back every step (fully teacher-forced -> one-step)
          - period=None : never feed back (fully autoregressive)
          - period=N    : feed back the ground-truth latent every N steps
        """
        batch_size, seq_len, *_ = x.size()

        hpc_result = self.HPC_model(x)
        p_inf = hpc_result['p_inf']
        s_inf = hpc_result['s_inf']

        mec_result = self.MEC_model(p_inf)
        g_inf = mec_result['g_inf']
        p_int = mec_result['p_int']
        s_recon = self.HPC_model.hidden2frame(p_int)

        g_inf_prev = g_inf[:, :-1].detach() # [b, t-1, g]
        g_inf_next = g_inf[:, 1:].detach()    # [b, t-1, g]
        a_low = self.Inverse_model(g_inf_prev, g_inf_next)

        # Generation Loop, autoregression
        generated_frames = []
        p_gen_list = []
        g_gen_list = []
        last_g_gen = g_inf[:, 0:1].detach()

        for i in range(1, seq_len):
            if period is not None and i % period == 0:
                last_g = g_inf[:, i-1:i]      # feedback: ground-truth latent
            else:
                last_g = last_g_gen           # use the model's own prediction
            next_g_gen = last_g + self.forward_dynamics(last_g, a_low[:, i-1:i])    # a_low_t, g_int_t -> g_gen_t+1
            next_p_gen = rearrange(self.MEC_model.grid2hidden(next_g_gen), 'b t (h w d) -> b t h w d', h = self.patch_size, w = self.patch_size)
            next_s_gen = self.HPC_model.hidden2frame(next_p_gen)

            generated_frames.append(next_s_gen)
            p_gen_list.append(next_p_gen)
            g_gen_list.append(next_g_gen)

            # Autoregressive update
            last_g_gen = next_g_gen

        s_gen = torch.cat(generated_frames, dim=1)
        p_gen = torch.cat(p_gen_list, dim=1)
        g_gen = torch.cat(g_gen_list, dim=1)

        return {
            'x': x,
            's_inf': s_inf,
            's_gen': s_gen,
            's_recon': s_recon,
            'p_inf': p_inf,
            'p_gen': p_gen,
            'p_int': p_int,
            'g_inf': g_inf,
            'g_gen': g_gen,
            'a_low': a_low,
        }

    def one_step(self, x):
        """One-step prediction: every frame is predicted from the
        ground-truth previous latent (fully teacher-forced)."""
        return self._predict(x, period=1)

    def autoregressive(self, x, feedback_period=None):
        """Autoregressive rollout from the first frame.

        feedback_period=None : pure autoregression (no feedback after frame 0).
        feedback_period=N    : re-inject the ground-truth latent every N steps.
        """
        return self._predict(x, period=feedback_period)

    def transfer(self, x_t, x, period: int | None = 1):
        """Transfer source transitions onto target content.

        `period` controls how often the target ground-truth latent is fed back:
          - period=1    : feed back every step (teacher-forced transfer)
          - period=None : never feed back (fully autoregressive transfer)
          - period=N    : feed back the target latent every N steps
        """
        batch_size, seq_len, *_ = x.size()

        hpc_result = self.HPC_model(x)
        p_inf = hpc_result['p_inf']
        mec_result = self.MEC_model(p_inf)
        g_inf = mec_result['g_inf']

        hpc_result = self.HPC_model(x_t)
        p_inf_t = hpc_result['p_inf']
        mec_result = self.MEC_model(p_inf_t)
        g_inf_t = mec_result['g_inf']

        g_inf_prev = g_inf_t[:, :-1].detach() # [b, t-1, g]
        g_inf_next = g_inf_t[:, 1:].detach()    # [b, t-1, g]
        a_low_t = self.Inverse_model(g_inf_prev, g_inf_next)

         # Generation Loop, autoregression
        generated_frames = []
        p_gen_list = [] 
        g_gen_list = []
        last_g_gen = g_inf[:, 0:1].detach()

        # Generate frames with optional target feedback.
        for i in range(1, seq_len):
            if period is not None and i % period == 0:
                last_g = g_inf[:, i-1:i]
            else:
                last_g = last_g_gen
            next_g_gen = last_g + self.forward_dynamics(last_g, a_low_t[:, i-1:i])    # a_low_t, g_int_t -> g_gen_t+1
            next_p_gen = rearrange(self.MEC_model.grid2hidden(next_g_gen), 'b t (h w d) -> b t h w d', h = self.patch_size, w = self.patch_size)
            next_s_gen = self.HPC_model.hidden2frame(next_p_gen)

        #    Store the generated outputs
            generated_frames.append(next_s_gen)
            p_gen_list.append(next_p_gen)
            g_gen_list.append(next_g_gen)

            # Autoregressive update
            last_g_gen = next_g_gen

        # Concatenate all generated frames
        s_gen = torch.cat(generated_frames, dim=1) 
        p_gen = torch.cat(p_gen_list, dim=1)
        g_gen = torch.cat(g_gen_list, dim=1) 

        return {
            'x': x,
            's_gen': s_gen,
            'p_inf': p_inf,
            'p_gen': p_gen,
            'g_inf': g_inf,
            'g_gen': g_gen,
            'a_low_t': a_low_t,
        }

    def reuse_z(self, x, period=1):
        batch_size, seq_len, *_ = x.size()

        hpc_result = self.HPC_model(x)
        p_inf = hpc_result['p_inf']
        mec_result = self.MEC_model(p_inf)
        g_inf = mec_result['g_inf']

        g_inf_prev = g_inf[:, 0:1].detach()    # [b, 1, g]
        g_inf_next = g_inf[:, 1:2].detach()    # [b, 1, g]
        z = self.Inverse_model(g_inf_prev, g_inf_next)

        generated_frames = []
        p_gen_list = [] 
        g_gen_list = []
        last_g_gen = g_inf[:, 0:1]
        for i in range(1, seq_len):
            if i % period == 0:
                last_g = g_inf[:, i-1:i]
            else:
                last_g = last_g_gen
            next_g_gen = last_g + self.forward_dynamics(last_g, z)    # a_low_t, g_int_t -> g_gen_t+1
            next_p_gen = rearrange(self.MEC_model.grid2hidden(next_g_gen), 'b t (h w d) -> b t h w d', h = self.patch_size, w = self.patch_size)
            next_s_gen = self.HPC_model.hidden2frame(next_p_gen)

        #    Store the generated outputs
            generated_frames.append(next_s_gen)
            p_gen_list.append(next_p_gen)
            g_gen_list.append(next_g_gen)

            # Autoregressive update
            last_g_gen = next_g_gen

        # Concatenate all generated frames
        s_gen = torch.cat(generated_frames, dim=1) 
        p_gen = torch.cat(p_gen_list, dim=1)
        g_gen = torch.cat(g_gen_list, dim=1) 

        return {
            'x': x,
            's_gen': s_gen,
            'p_gen': p_gen,
            'g_gen': g_gen,
        }
        
    def _mse_loss(self, s_actual, s_pred):
        mse = nn.MSELoss()
        s_actual = s_actual
        s_pred = s_pred
        return mse(s_pred, s_actual)
        
    def _action_loss(self, a_low, g_inf):
        """
        Align predicted Δg (via cann_mlp from [g_t, a_t]) 
        with true Δg = g_{t+1} - g_t extracted from visual branch.
        """
        # true delta_g: [B, T-1, G]
        delta_true = (g_inf[:, 1:] - g_inf[:, :-1])
        # predict delta via your MLP
        delta_pred = self.forward_dynamics(g_inf, a_low)

        loss = F.mse_loss(delta_pred, delta_true)
        return loss

    def _g_alignment_loss(self, g_inf, g_gen):
        mse = F.mse_loss(g_inf, g_gen)
        cosine = 1 - F.cosine_similarity(g_inf, g_gen, dim=-1).mean()
        loss = mse + cosine * 0.5
        return loss
    
    def _cycle_consistency_loss(self, g_inf, g_gen, a_low):
        g_prev = g_inf[:, :-1].detach()
        g_next = g_gen[:, :]
        a_pred = self.Inverse_model(g_prev, g_next)
        # Calculate the cycle consistency loss
        loss = F.mse_loss(a_low, a_pred)
        return loss
    
    def _contrastive_loss(self, g_inf, g_gen):
        cos_sim = F.cosine_similarity(g_inf, g_gen, dim=-1).mean()
        return cos_sim
    
    def _covariance_reg_loss(self, obs_enc: torch.Tensor):
        return off_diag_cov_loss(obs_enc)
    
    def _var_loss(self, Z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
        Z = Z - Z.mean(dim=0)
        std_z = torch.sqrt(Z.var(dim=0) + eps)
        return F.relu(gamma - std_z).mean()
    
    def _time_variance_loss(self, Z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
        Z = Z - Z.mean(dim=1, keepdim=True)
        std_z = torch.sqrt(Z.var(dim=1) + eps)
        return F.relu(gamma - std_z).mean()

    # Calculate total loss as weighted sum of individual losses
    def total_loss(self, res):
         # 1. Sensory Losses
        loss_s_inf = self._mse_loss(res['x'], res['s_inf'])
        loss_s_recon = self._mse_loss(res['x'], res['s_recon'])
        loss_s_gen = self._mse_loss(res['x'][:, 1:], res['s_gen'])

        # 2. Place Cell Losses
        p_inf = rearrange(res['p_inf'], 'b t h w d -> b t (h w d)')
        p_int = rearrange(res['p_int'], 'b t h w d -> b t (h w d)')
        p_gen = rearrange(res['p_gen'], 'b t h w d -> b t (h w d)')
        loss_p_inf = self.MEC_model._mse_loss(p_inf, p_int)
        loss_p_gen = self._mse_loss(p_inf[:, 1:].detach(), p_gen[:, :]) # Target p_inf should likely be detached if used

        # 3. Grid Cell Alignment Loss
        loss_g_inf_gen = self._g_alignment_loss(res['g_inf'][:, 1:].detach(), res['g_gen'][:, :]) # Compares target g_{t+1} with generated g_{t+1}
       
        # 4. Contrastive Loss
        loss_contrastive = self._contrastive_loss(res['g_inf'][:, :-1], res['g_gen'])

        # 5. Action Loss
        delta_g_true = res['g_inf'][:, 1:].detach() - res['g_inf'][:, :-1].detach()
        delta_g_pred = self.forward_dynamics(res['g_inf'][:, :-1], res['a_low']) # Predict delta_g from g_t and a_t
        loss_action = self._g_alignment_loss(delta_g_true, delta_g_pred) # Compare true delta_g with predicted delta_g

        # 5. Regularization
        loss_covariance = self.MEC_model._covariance_reg_loss(res['g_inf']) + self.HPC_model._covariance_reg_loss(res['p_inf'])
        loss_var = self.MEC_model._var_loss(rearrange(res['g_inf'], 'b t d -> b (t d)'), gamma=0.2) + self.HPC_model._var_loss(rearrange(res['p_inf'], 'b t h w d -> b (t h w d)'), gamma=0.2)

        losses_dict = {
            'sensory_inf': loss_s_inf,
            'sensory_gen': loss_s_gen,
            'sensory_recon': loss_s_recon,
            'place_inf': loss_p_inf,
            'place_gen': loss_p_gen,
            'grid': loss_g_inf_gen,
            'action': loss_action,
            'cycle': loss_contrastive,
            'covariance': loss_covariance,
            'variance': loss_var,
        }

        # Focus more on action learning initially
        weights = {
            'sensory_inf': 5., #5.,
            'sensory_gen': 3., #3.,
            'sensory_recon': 5., #5., 
            'place_inf': 2.,
            'place_gen': 1.,
            'grid': 5.,
            'action': 1.,  # Increase action loss importance
            'cycle': 1,
            'covariance': 0.05,
            'variance': 0.05,
        }

        # Calculate total weighted loss
        total_loss = sum([weights[k] * v for k, v in losses_dict.items()])
        return total_loss, losses_dict
    
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        """
        num_params = sum(p.numel() for p in self.parameters())
        return num_params
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')  # Kaiming initialization
            if module.bias is not None:
                nn.init.zeros_(module.bias)  # Initialize bias to 0
        elif isinstance(module, nn.ConvTranspose2d):
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')  # Kaiming initialization
            if module.bias is not None:
                nn.init.zeros_(module.bias)  # Initialize bias to 0
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)  # Orthogonal initialization for weights
                elif 'bias' in name:
                    nn.init.zeros_(param)  # Initialize bias to 0
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def to(self, device):
        super().to(device)
        self.Inverse_model.to(device)
        self.forward_dynamics.to(device)
        self.HPC_model.to(device)
        self.MEC_model.to(device)

        return self

    def configure_optimizers(self, weight_decay, lr, betas, T_max, finetune_world_model=False):
        inverse_params = list(self.Inverse_model.parameters())
        dynamics_params = list(self.forward_dynamics.parameters())
        world_params = list(self.HPC_model.parameters()) + list(self.MEC_model.parameters())

        for param in inverse_params + dynamics_params:
            param.requires_grad = True
        for param in world_params:
            param.requires_grad = finetune_world_model

        params_to_train = [
            {'params': inverse_params + dynamics_params, 'lr': lr['inverse_lr']},
        ]
        if finetune_world_model:
            params_to_train.append({'params': world_params, 'lr': lr['world_lr']})

        optimizer = torch.optim.AdamW(
            params_to_train,
            lr=lr['inverse_lr'],
            betas=betas,
            weight_decay=weight_decay
        )

        min_group_lr = min(group['lr'] for group in params_to_train)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=min_group_lr * 0.1
        )

        return optimizer, scheduler
