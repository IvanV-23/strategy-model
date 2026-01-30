import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import gymnasium as gym

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim_diplomacy: int, action_dim_economy: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.diplomacy_head = nn.Linear(128, action_dim_diplomacy)
        self.economy_head = nn.Linear(128, action_dim_economy)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.backbone(state)
        diplomacy_logits = self.diplomacy_head(x)
        economy_logits = self.economy_head(x)
        return diplomacy_logits, economy_logits

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
    def __init__(self, state_dim: int, action_dim_diplomacy: int, action_dim_economy: int, gamma: float = 0.99):
        super().__init__()
        self.actor = Actor(state_dim, action_dim_diplomacy, action_dim_economy)
        self.critic = Critic(state_dim)
        self.gamma = gamma

        self.save_hyperparameters()

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        diplomacy_logits, economy_logits = self.actor(state)
        state_value = self.critic(state)
        return diplomacy_logits, economy_logits, state_value


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
