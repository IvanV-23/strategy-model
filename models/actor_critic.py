import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import gymnasium as gym

class Actor(nn.Module):
    def __init__(self, action_dim_dip: int, action_dim_eco: int, action_dim_dist: int, action_dim_target: int, board_size: int = 64):
        super().__init__()
        # 1. New input channels (Owner, Status, Soldiers, Mask)
        self.in_channels = 4 
        self.board_size = board_size
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # This head picks the tile to target. 
        # We output action_dim_target to stay consistent with the gym space.
        self.target_head = nn.Conv2d(64, 1, kernel_size=1)
        
        # 2. Updated dimensions: (64 channels * board_size) + 8 global stats
        self.fc_common = nn.Linear((64 * board_size) + 8, 256)
        
        self.diplomacy_head = nn.Linear(256, action_dim_dip)
        self.economy_head = nn.Linear(256, action_dim_eco)
        self.distribution_head = nn.Linear(256, action_dim_dist)

    def forward(self, board_state: torch.Tensor, global_stats: torch.Tensor):
        batch_size = board_state.shape[0]
        
        # Pass through CNN: Output (batch, 64, 8, 8)
        features = self.conv_block(board_state) 
        
        # Target logits: (batch, 1, 8, 8) -> flattened to (batch, 64)
        target_logits = self.target_head(features).reshape(batch_size, -1)
        
        # Flatten spatial features and combine with scalar resources
        flat_spatial = features.reshape(batch_size, -1)
        combined = torch.cat([flat_spatial, global_stats], dim=1)
        
        common = F.relu(self.fc_common(combined))
        
        return (self.diplomacy_head(common), 
                self.economy_head(common), 
                self.distribution_head(common),
                target_logits)

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
        action_dim_eco: int, 
        action_dim_dist: int, 
        action_dim_target: int, # Added 4th action head dimension
        board_size: int = 64, 
        gamma: float = 0.99
    ):
        super().__init__()
        # 1. The Actor handles all 4 policy heads
        self.actor = Actor(
            action_dim_dip, 
            action_dim_eco, 
            action_dim_dist, 
            action_dim_target, 
            board_size
        )
        
        # 2. The Critic handles the state-value estimation
        self.critic = Critic(board_size)
        
        self.gamma = gamma

    def forward(self, board_state: torch.Tensor, global_stats: torch.Tensor):
        """
        board_state: (Batch, 4, 8, 8)
        global_stats: (Batch, 8)
        """
        # Unpack the 4 actor outputs (Logits for each action type)
        dip, eco, dist, target = self.actor(board_state, global_stats)
        
        # Get the 1 critic output (Scalar value)
        value = self.critic(board_state, global_stats)
        
        return dip, eco, dist, target, value

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer