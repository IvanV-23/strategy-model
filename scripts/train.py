import gymnasium as gym
import torch
from torch.utils.data import DataLoader, IterableDataset
from typing import Iterator, Tuple
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import numpy as np

import sys
import os

# Add the parent directory to sys.path to allow importing local modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from enviroment.strategy_env import StrategyEnv
from models.actor_critic import StrategyActorCritic
from data.replay_buffer import ReplayBuffer

class RLDataset(IterableDataset):
    def __init__(self, buffer: ReplayBuffer, batch_size: int):
        super().__init__()
        self.buffer = buffer
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[Tuple]:
        while True:
            yield self.buffer.sample(self.batch_size)

class StrategyLightningModule(pl.LightningModule):
    def __init__(self, env: gym.Env, buffer: ReplayBuffer, lr: float, collect_steps: int, gamma: float = 0.99, batch_size: int = 64):
        super().__init__()
        self.env = env
        self.buffer = buffer
        self.lr = lr
        self.collect_steps = collect_steps
        self.gamma = gamma
        self.batch_size = batch_size

        self.model = StrategyActorCritic(
            action_dim_dip=self.env.action_space["diplomacy"].n,
            action_dim_eco=self.env.action_space["economy"].nvec[0],
            action_dim_dist=self.env.action_space["distribution"].n,
            action_dim_target=self.env.action_space["target_tile"].n,
            board_size=64, 
            gamma=self.gamma
        )
        
        # We need info for the mask
        self.obs, self.info = self.env.reset()
        self.total_reward = 0
        self.episode_count = 0
        
        self.save_hyperparameters(ignore=['env', 'buffer', 'model'])

    def _process_obs(self, obs_batch):
            # Handle single observation (from env.reset/step) vs batch (from ReplayBuffer)
            if isinstance(obs_batch, list) and len(obs_batch) == 1:
                obs_batch = obs_batch[0]

            boards = torch.as_tensor(obs_batch["board_state"], dtype=torch.float32).view(-1, 4, 8, 8)
            p_res = torch.as_tensor(obs_batch["player_resources"], dtype=torch.float32).view(-1, 4)
            
            # CHANGE THIS: .view(-1, 3) instead of (-1, 4)
            o_res = torch.as_tensor(obs_batch["opponent_resources"], dtype=torch.float32).view(-1, 3)
            
            turn = torch.as_tensor(obs_batch["turn_number"], dtype=torch.float32).view(-1, 1)

            # Total stats: 4 (player) + 3 (opp) + 1 (turn) = 8
            stats = torch.cat([p_res, o_res, turn], dim=-1)
            return boards.to(self.device), stats.to(self.device)

    def on_train_epoch_start(self):
        for _ in range(self.collect_steps):
            board_tensor, stats_tensor = self._process_obs(self.obs)

            with torch.no_grad():
                dip_logits, eco_logits, dist_logits, target_logits, _ = self.forward(board_tensor, stats_tensor)
            
            # --- FEATURE: Apply Action Masking during collection ---
            mask = self.info.get("action_mask", None)
            if mask is not None:
                mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
                target_logits[0, ~mask_tensor] = -1e10 

            sol_logits, mine_logits = eco_logits

            # Sample actions
            action = {
                            "diplomacy": torch.distributions.Categorical(logits=dip_logits).sample().item(),
                            "economy": [
                                torch.distributions.Categorical(logits=sol_logits).sample().item(),
                                torch.distributions.Categorical(logits=mine_logits).sample().item()
                            ],
                            "distribution": torch.distributions.Categorical(logits=dist_logits).sample().item(),
                            "target_tile": torch.distributions.Categorical(logits=target_logits).sample().item()
                        }

            next_obs, reward, terminated, truncated, next_info = self.env.step(action)
            self.total_reward += reward
            
            # Store experience (Ensure your buffer can store the 'mask' if you want off-policy masking)
            self.buffer.add(
                state=self.obs,
                action=action, 
                reward=reward, 
                next_state=next_obs,
                terminated=terminated, 
                truncated=truncated,
                mask=mask # Pass mask here
            )

            self.obs = next_obs
            self.info = next_info
            if terminated or truncated:
                self.obs, self.info = self.env.reset()
                self.log("episode/total_reward", self.total_reward, on_step=False, on_epoch=True)
                self.total_reward = 0
                self.episode_count += 1

    def training_step(self, batch, batch_idx):
        # 1. Unpack batch 
        (states, actions_dip, actions_eco, actions_dist, 
            actions_target, rewards, next_states, terminals, truncated, masks) = [
                t.squeeze(0) if isinstance(t, torch.Tensor) else t for t in batch
            ]

        # 2. Pre-process observations
        boards, stats = self._process_obs(states)
        next_boards, next_stats = self._process_obs(next_states)

        # 3. Forward Pass
        # Updated: model now returns a tuple for the economy head
        dip_logits, (sol_logits, mine_logits), dist_logits, target_logits, current_values = self(boards, stats)
        current_values = current_values.squeeze(-1)

        # --- FEATURE: Safe Action Masking ---
        masks = masks.view(target_logits.shape).bool() 
        target_logits = target_logits.masked_fill(~masks, -1e10)

        # 4. Calculate Target Values (Bellman Equation)
        with torch.no_grad():
            *_, next_values = self(next_boards, next_stats)
            next_values = next_values.squeeze(-1)
            mask_done = (~terminals).float()
            returns = (rewards / 10.0) + (self.gamma * next_values * mask_done)

        # 5. Advantage Calculation
        advantages = (returns - current_values).detach()
        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 6. Policy Distributions
        dist_dip = torch.distributions.Categorical(logits=dip_logits)
        dist_sol = torch.distributions.Categorical(logits=sol_logits)
        dist_min = torch.distributions.Categorical(logits=mine_logits)
        dist_dist = torch.distributions.Categorical(logits=dist_logits)
        dist_target = torch.distributions.Categorical(logits=target_logits)

        # --- 7. Multi-Head Actor Loss (FORCED FLATTENING) ---
        
        # Ensure actions are the correct shape [BatchSize] and type (Long)
        # We use .reshape(-1) to guarantee a 1D tensor for log_prob
        a_dip = actions_dip.long().reshape(-1)
        a_dist = actions_dist.long().reshape(-1)
        a_target = actions_target.long().reshape(-1)

        # Economy: actions_eco is [BatchSize, 2]
        # If the buffer gives [1, 2], reshape(-1) would give [2], which is WRONG.
        # We must explicitly take the columns.
        if actions_eco.dim() == 1: # Handle single-sample case [2] -> [1, 2]
            actions_eco = actions_eco.unsqueeze(0)
            
        a_sol = actions_eco[:, 0].long().reshape(-1)
        a_min = actions_eco[:, 1].long().reshape(-1)

        # Calculate individual log probabilities
        log_prob_dip = dist_dip.log_prob(a_dip)
        log_prob_sol = dist_sol.log_prob(a_sol)
        log_prob_min = dist_min.log_prob(a_min)
        log_prob_dist = dist_dist.log_prob(a_dist)
        log_prob_target = dist_target.log_prob(a_target)

        # Losses
        loss_dip = -(log_prob_dip * adv).mean()
        loss_eco = -((log_prob_sol + log_prob_min) * adv).mean()
        loss_dist = -(log_prob_dist * adv).mean()
        loss_target = -(log_prob_target * adv).mean()

        actor_loss = loss_dip + loss_eco + loss_dist + loss_target

        # 8. Critic Loss
        critic_loss = torch.nn.functional.mse_loss(current_values, returns)

        # 9. Entropy
        ent_dip = dist_dip.entropy().mean()
        ent_sol = dist_sol.entropy().mean()
        ent_min = dist_min.entropy().mean()
        ent_dist = dist_dist.entropy().mean()
        ent_target = dist_target.entropy().mean()
        total_entropy = ent_dip + ent_sol + ent_min + ent_dist + ent_target

        # 10. Total Loss
        # loss = Actor + 0.5*Critic - 0.01*Entropy
        total_loss = actor_loss + (0.5 * critic_loss) - (0.01 * total_entropy)

        # --- LOGGING ---
        self.log_dict({
            "loss/total": total_loss,
            "loss/actor": actor_loss,
            "loss/critic": critic_loss,
            "loss/head_eco": loss_eco,
            "stats/entropy": total_entropy,
            "stats/value_mean": current_values.mean(),
            "stats/advantage_abs": adv.abs().mean()
        }, prog_bar=True)

        return total_loss

    def forward(self, board_state, global_stats):
        return self.model(board_state, global_stats)
    
    def configure_optimizers(self):
        return self.model.configure_optimizers()

    def train_dataloader(self):
        return DataLoader(RLDataset(self.buffer, self.batch_size), batch_size=1)

def train_agent_lightning():
    BATCH_SIZE = 64
    BUFFER_CAPACITY = 10000
    COLLECT_STEPS_PER_EPOCH = 1000
    TRAIN_EPOCHS = 100
    LEARNING_RATE = 1e-4

    env = gym.make("StrategyProblem-v0")
    # Make sure your ReplayBuffer handles the 'mask' key in .add() and returns it in .sample()
    replay_buffer = ReplayBuffer(BUFFER_CAPACITY, env.observation_space, env.action_space)

    lightning_module = StrategyLightningModule(
        env=env, buffer=replay_buffer, lr=LEARNING_RATE,
        collect_steps=COLLECT_STEPS_PER_EPOCH, batch_size=BATCH_SIZE
    )

    trainer = pl.Trainer(
        max_epochs=TRAIN_EPOCHS,
        limit_train_batches=100, 
        callbacks=[EarlyStopping(monitor="episode/total_reward", patience=20, mode="max")],
        gradient_clip_val=0.5,
        logger=MLFlowLogger(experiment_name="Strategy_Masked", tracking_uri="file:./ml-runs"),
    )

    trainer.fit(lightning_module)
    torch.save(lightning_module.model.state_dict(), "trained_model.pth")

if __name__ == "__main__":
    train_agent_lightning()