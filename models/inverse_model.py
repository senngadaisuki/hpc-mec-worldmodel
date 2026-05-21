# -*- coding: utf-8 -*-
import torch.nn as nn
from einops import rearrange

class Inverse_model(nn.Module):
    def __init__(self, input_dim=512, action_dim=256, hidden_dim=256, hidden_depth=2):
        super().__init__()

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
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, g_prev, g_next):
        """
        g_prev: [B, T-1, D]
        g_next: [B, T-1, D]
        """
        g_prev = rearrange(g_prev, 'b t (h w c) -> b t h w c', h=4, w=4)   # [B, T-1, H, W, C]
        g_next = rearrange(g_next, 'b t (h w c) -> b t h w c', h=4, w=4)   # [B, T-1, H, W, C]

        # Temporal difference
        x = g_next - g_prev  # [B, T-1, H, W, C]

        # Initial projection
        h = self.input_proj(x)
        
        # Apply hidden layers with residual connections
        for layer in self.hidden_layers:
            h = h + layer(h)
        
        # Final projection
        action = self.output_proj(h)
        return action