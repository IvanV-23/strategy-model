import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import gymnasium as gym

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim_dip: int, action_dim_eco: int, action_dim_dist: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.diplomacy_head = nn.Linear(128, action_dim_dip)
        self.economy_head = nn.Linear(128, action_dim_eco)
        self.distribution_head = nn.Linear(128, action_dim_dist) 

    def forward(self, state: torch.Tensor):
        x = self.backbone(state)
        return (self.diplomacy_head(x), 
                self.economy_head(x), 
                self.distribution_head(x))

class Critic(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.value_head = nn.Linear(128, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.backbone(state)
        state_value = self.value_head(x)
        return state_value


class StrategyActorCritic(pl.LightningModule):
    def __init__(self, state_dim: int, action_dim_diplomacy: int, action_dim_economy: int, action_dim_dist: int, gamma: float = 0.99):
        super().__init__()
        # Pass the 3rd dimension to the Actor
        self.actor = Actor(state_dim, action_dim_diplomacy, action_dim_economy, action_dim_dist)
        self.critic = Critic(state_dim)
        self.gamma = gamma

        self.save_hyperparameters()

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Actor now returns 3 sets of logits
        diplomacy_logits, economy_logits, dist_logits = self.actor(state)
        state_value = self.critic(state)
        return diplomacy_logits, economy_logits, dist_logits, state_value

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
