# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from einops import rearrange

class MLPResidualBlock(nn.Module):
    def __init__(self, dim, ff_mult=4, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_mult * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.ff(self.norm(x))
        return x
    
class GatedResidualForward(nn.Module):
    def __init__(self, g_dim=4096, z_dim=512, hidden_dim=4096, depth=4, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.init_proj = nn.Linear(g_dim + z_dim, g_dim)

        self.blocks = nn.ModuleList([
            MLPResidualBlock(dim=g_dim, ff_mult=4) for _ in range(depth)
        ])

        self.delta_proj = nn.Sequential(
            nn.LayerNorm(g_dim),
            nn.Linear(g_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, g_dim),
        )

    def forward(self, g, z):
        g = rearrange(g, 'b t (h w d) -> b t h w d', h=self.patch_size, w=self.patch_size)
        x = torch.cat([g, z], dim=-1)
        x = self.init_proj(x)

        for block in self.blocks:
            x = x + block(x)  # Residual connection

        delta_g = self.delta_proj(x)
        delta_g = rearrange(delta_g, 'b t h w d -> b t (h w d)')
        return delta_g
