import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import gymnasium as gym

class Actor(nn.Module):
    def __init__(self, action_dim_dip, action_dim_eco, action_dim_dist, action_dim_target, board_size=64):
        super().__init__()
        self.in_channels = 4
        self.eco_dim = int(action_dim_eco)  # e.g., 6
        
        # ... CNN and FC layers ...
        self.conv_block = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.fc_common = nn.Linear((64 * board_size) + 8, 256)

        # SINGLE HEAD for Economy: Must be (eco_dim * 2)
        self.economy_head = nn.Linear(256, self.eco_dim * 2) 
        
        self.diplomacy_head = nn.Linear(256, action_dim_dip)
        self.distribution_head = nn.Linear(256, action_dim_dist)
        self.target_head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, board_state, global_stats):
        batch_size = board_state.shape[0]
        features = self.conv_block(board_state)
        
        flat_spatial = features.reshape(batch_size, -1)
        combined = torch.cat([flat_spatial, global_stats], dim=1)
        common = F.relu(self.fc_common(combined))

        # --- THE VECTOR SPLIT ---
        eco_logits_all = self.economy_head(common)
        
        # We split eco_logits_all (size 12) into two chunks of size 6
        # The result is a tuple: (Tensor[batch, 6], Tensor[batch, 6])
        soldiers_logits, mines_logits = torch.split(eco_logits_all, self.eco_dim, dim=1)

        dip_logits = self.diplomacy_head(common)
        dist_logits = self.distribution_head(common)
        target_logits = self.target_head(features).reshape(batch_size, -1)

        return dip_logits, (soldiers_logits, mines_logits), dist_logits, target_logits

class Critic(nn.Module):
    def __init__(self, board_size: int = 64):
        super().__init__()
        self.in_channels = 4
        
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # (16 channels * 64 tiles) + 8 global stats
        self.value_head = nn.Sequential(
            nn.Linear((16 * board_size) + 8, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, board_state: torch.Tensor, global_stats: torch.Tensor) -> torch.Tensor:
        spatial_features = self.conv(board_state)
        combined = torch.cat([spatial_features, global_stats], dim=1)
        return self.value_head(combined)




class StrategyActorCritic(nn.Module):
    def __init__(
        self, 
        action_dim_dip: int, 
        # Updated: This now represents the size of each MultiDiscrete choice (e.g., 6)
        action_dim_eco: int, 
        action_dim_dist: int, 
        action_dim_target: int, 
        board_size: int = 64, 
        gamma: float = 0.99
    ):
        super().__init__()
        
        # 1. The Actor now needs to know it has two economy heads
        # We pass action_dim_eco which the Actor class will use for both Soldiers and Mines
        self.actor = Actor(
            action_dim_dip, 
            action_dim_eco, 
            action_dim_dist, 
            action_dim_target, 
            board_size
        )
        
        # 2. The Critic remains the same
        self.critic = Critic(board_size)
        
        self.gamma = gamma

    def forward(self, board_state: torch.Tensor, global_stats: torch.Tensor):
        # The internal Actor.forward() should return:
        # dip, (eco_sol, eco_mine), dist, target
        dip, eco, dist, target = self.actor(board_state, global_stats)
        value = self.critic(board_state, global_stats)
        
        return dip, eco, dist, target, value

    def configure_optimizers(self):
        # Using a slightly higher LR or making it configurable is often helpful
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
