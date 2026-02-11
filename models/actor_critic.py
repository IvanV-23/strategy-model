import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Actor(nn.Module):
    def __init__(self, action_dim_dip, action_dim_eco, action_dim_dist, action_dim_target, board_size=64):
        super().__init__()
        self.in_channels = 5
        self.eco_dim = int(action_dim_eco)
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # UPDATE: Changed from +8 to +10 for the new stats vector
        self.fc_common = nn.Linear((64 * board_size) + 10, 256)

        # SINGLE HEAD for Economy: size is (eco_dim * 2) -> [Soldiers, Mines]
        self.economy_head = nn.Linear(256, self.eco_dim * 2) 
        
        self.diplomacy_head = nn.Linear(256, action_dim_dip)
        self.distribution_head = nn.Linear(256, action_dim_dist)
        self.target_head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, board_state, global_stats, target_mask=None, build_mask=None):
        batch_size = board_state.shape[0]
        features = self.conv_block(board_state)
        
        flat_spatial = features.reshape(batch_size, -1)
        combined = torch.cat([flat_spatial, global_stats], dim=1)
        common = F.relu(self.fc_common(combined))

        # --- ECONOMY HEAD ---
        eco_logits_all = self.economy_head(common)
        soldiers_logits, mines_logits = torch.split(eco_logits_all, self.eco_dim, dim=1)

        if build_mask is not None:
            # Use reshape to be safe, or just ensure build_mask is already (Batch, 6)
            # The ~ operator flips the boolean mask for masked_fill
            mines_logits = mines_logits.masked_fill(~build_mask.view(mines_logits.shape).bool(), -1e9)

        # --- DIPLOMACY & DISTRIBUTION ---
        dip_logits = self.diplomacy_head(common)
        dist_logits = self.distribution_head(common)

        # --- TARGET HEAD ---
        target_logits = self.target_head(features).reshape(batch_size, -1)
        if target_mask is not None:
            target_logits = target_logits.masked_fill(~target_mask.bool(), -1e9)

        return dip_logits, (soldiers_logits, mines_logits), dist_logits, target_logits

class Critic(nn.Module):
    def __init__(self, board_size: int = 64):
        super().__init__()
        self.in_channels = 5
        
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # UPDATE: Changed from +8 to +10 to match the new ReplayBuffer stats
        self.value_head = nn.Sequential(
            nn.Linear((16 * board_size) + 10, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, board_state: torch.Tensor, global_stats: torch.Tensor) -> torch.Tensor:
        spatial_features = self.conv(board_state)
        combined = torch.cat([spatial_features, global_stats], dim=1)
        return self.value_head(combined)

class StrategyActorCritic(nn.Module):
    def __init__(self, action_dim_dip, action_dim_eco, action_dim_dist, action_dim_target, board_size=64, gamma=0.99):
        super().__init__()
        self.actor = Actor(action_dim_dip, action_dim_eco, action_dim_dist, action_dim_target, board_size)
        self.critic = Critic(board_size)
        self.gamma = gamma

    def forward(self, board_state, global_stats, target_mask=None, build_mask=None):
        # Pass masks down to the actor for valid action filtering
        dip, eco, dist, target = self.actor(board_state, global_stats, target_mask, build_mask)
        value = self.critic(board_state, global_stats)
        return dip, eco, dist, target, value

    def configure_optimizers(self, lr=1e-4):
        return torch.optim.Adam(self.parameters(), lr=lr)
