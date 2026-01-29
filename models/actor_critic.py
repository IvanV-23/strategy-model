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
    def __init__(self, env: gym.Env):
        super().__init__()
        # Determine state and action dimensions from the environment
        # Assuming observation space is a Dict of Box and Discrete
        state_dim = 0
        for space in env.observation_space.spaces.values():
            if isinstance(space, gym.spaces.Box):
                state_dim += int(np.prod(space.shape))
            elif isinstance(space, gym.spaces.Discrete):
                state_dim += 1 # For discrete, we'll treat it as a single feature

        action_dim_diplomacy = env.action_space["diplomacy"].n
        action_dim_economy = env.action_space["economy"].n

        self.actor = Actor(state_dim, action_dim_diplomacy, action_dim_economy)
        self.critic = Critic(state_dim)

        self.save_hyperparameters()

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        diplomacy_logits, economy_logits = self.actor(state)
        state_value = self.critic(state)
        return diplomacy_logits, economy_logits, state_value

    def training_step(self, batch, batch_idx):
        # This is a placeholder for PPO training logic.
        # In a real PPO implementation, you would calculate policy_loss and value_loss
        # and combine them.
        states, actions_diplomacy, actions_economy, advantages, returns = batch

        # Actor loss (placeholder)
        diplomacy_logits, economy_logits = self.actor(states)
        
        dist_diplomacy = torch.distributions.Categorical(logits=diplomacy_logits)
        dist_economy = torch.distributions.Categorical(logits=economy_logits)

        log_prob_diplomacy = dist_diplomacy.log_prob(actions_diplomacy)
        log_prob_economy = dist_economy.log_prob(actions_economy)
        
        # This is a simplified actor loss, PPO uses ratio and clipping
        actor_loss = -(log_prob_diplomacy * advantages).mean() - (log_prob_economy * advantages).mean()

        # Critic loss (placeholder)
        state_values = self.critic(states).squeeze(-1)
        critic_loss = F.mse_loss(state_values, returns)

        total_loss = actor_loss + 0.5 * critic_loss # Example weighting

        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("actor_loss", actor_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("critic_loss", critic_loss, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
