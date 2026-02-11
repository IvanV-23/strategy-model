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

# Add the parent directory to sys.path
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
        
        self.obs, self.info = self.env.reset()
        self.total_reward = 0
        self.episode_count = 0
        
        self.save_hyperparameters(ignore=['env', 'buffer', 'model'])

    def _process_obs(self, obs_batch, info_batch=None):
            """
            Converts observation and optional info masks into correctly shaped tensors.
            """
            # --- Handle Observations ---
            if "board_stats" in obs_batch:  # Single observation from env.reset() or env.step()
                boards = torch.as_tensor(obs_batch["board_state"], dtype=torch.float32, device=self.device).view(-1, 5, 8, 8)
                p_res = torch.as_tensor(obs_batch["player_resources"], dtype=torch.float32, device=self.device).view(-1, 4)
                o_res = torch.as_tensor(obs_batch["opponent_resources"], dtype=torch.float32, device=self.device).view(-1, 3)
                m_stats = torch.as_tensor(obs_batch["board_stats"], dtype=torch.float32, device=self.device).view(-1, 2)
                turn = torch.as_tensor(obs_batch["turn_number"], dtype=torch.float32, device=self.device).view(-1, 1)
                stats = torch.cat([p_res, o_res, m_stats, turn], dim=-1)
            else:  # Batch from ReplayBuffer
                boards = obs_batch["board_state"].to(self.device)
                stats = obs_batch["full_stats"].to(self.device)

            # --- Handle Masks ---


            t_mask, b_mask = None, None
            if info_batch is not None:
                # Handle single info dict vs batch of info dicts
                if isinstance(info_batch, dict):
                    # Single step collection
                    t_mask = torch.as_tensor(info_batch.get("action_mask", np.ones(64)), dtype=torch.bool, device=self.device).view(-1, 64)
                    b_mask = torch.as_tensor(info_batch.get("build_mask", np.ones(6)), dtype=torch.bool, device=self.device).view(-1, 6)
                else:
                    # If masks were already batched by the ReplayBuffer/DataLoader
                    t_mask = info_batch[0].to(self.device).view(-1, 64) # m_target
                    b_mask = info_batch[1].to(self.device).view(-1, 6)  # m_build

            if boards.dim() == 5 and boards.size(0) == 1:
                boards = boards.squeeze(0)
                stats = stats.squeeze(0)
                t_mask = t_mask.squeeze(0)
                b_mask = b_mask.squeeze(0)

            return boards, stats, t_mask, b_mask

    def on_train_start(self):
        """Fill the buffer with initial steps before training begins."""
        print(f"Pre-filling buffer with {self.batch_size} steps...")
        while len(self.buffer) < self.batch_size:
            self.on_train_epoch_start() # Reuse your collection logic

    def on_train_epoch_start(self):
            for _ in range(self.collect_steps):
                # Process obs and masks together
                board_t, stats_t, t_mask, b_mask = self._process_obs(self.obs, self.info)

                with torch.no_grad():
                    dip_logits, eco_logits, dist_logits, target_logits, _ = self.model(
                        board_t, stats_t, target_mask=t_mask, build_mask=b_mask
                    )
                
                sol_logits, mine_logits = eco_logits

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
            
            # Store experience with the full info dict so buffer can extract both masks
            self.buffer.add(
                state=self.obs,
                action=action, 
                reward=reward, 
                next_state=next_obs,
                terminated=terminated, 
                truncated=truncated,
                info=self.info # Changed from mask=mask
            )

            self.obs = next_obs
            self.info = next_info
            if terminated or truncated:
                self.obs, self.info = self.env.reset()
                self.log("episode/total_reward", self.total_reward, on_step=False, on_epoch=True)
                self.total_reward = 0
                self.episode_count += 1

    def training_step(self, batch, batch_idx):
        # 1. Unpack the full batch from your ReplayBuffer
        (states, a_dip, a_eco, a_dist, a_target, rewards, 
        next_states, terminals, truncated, m_target, m_build) = batch
        # SQUEEZE EVERYTHING: Remove the extra dimension [1, 64, ...] -> [64, ...]
        if rewards.dim() == 2: # If boards are 5D, everything else likely has a stray dim too
            a_dip = a_dip.squeeze(0)
            a_eco = a_eco.squeeze(0)
            a_dist = a_dist.squeeze(0)
            a_target = a_target.squeeze(0)
            rewards = rewards.squeeze(0)
            terminals = terminals.squeeze(0)
            # (Apply squeeze to masks and next_states as well if needed)

        # 2. Update this line to unpack all 4 values returned by _process_obs
        # We pass (m_target, m_build) as the second argument
        boards, stats, t_mask, b_mask = self._process_obs(states, (m_target, m_build))
        
        # 3. Use these processed masks in your model forward pass
        dip_logits, (sol_logits, mine_logits), dist_logits, target_logits, current_values = self.model(
            boards, stats, target_mask=t_mask, build_mask=b_mask
        )

        # 4. Target Values
        with torch.no_grad():
            # We don't strictly need masks for the next_values in simple A2C, but it's cleaner
            *_, next_values = self.model(boards, stats)
            next_values = next_values.squeeze(-1)
            mask_done = (~terminals).float()
            returns = (rewards / 10.0) + (self.gamma * next_values * mask_done)

        # 5. Advantage
        advantages = (returns - current_values).detach()
        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 6. Policy Distributions
        dist_dip = torch.distributions.Categorical(logits=dip_logits)
        dist_sol = torch.distributions.Categorical(logits=sol_logits)
        dist_min = torch.distributions.Categorical(logits=mine_logits)
        dist_dist = torch.distributions.Categorical(logits=dist_logits)
        dist_target = torch.distributions.Categorical(logits=target_logits)

        # 7. Multi-Head Actor Loss
        # Extract actions
        if a_eco.dim() == 1: a_eco = a_eco.unsqueeze(0)
        
        # Calculate Log Probs
        log_prob_dip = dist_dip.log_prob(a_dip.long().reshape(-1))
        log_prob_sol = dist_sol.log_prob(a_eco[:, 0].long().reshape(-1))
        log_prob_min = dist_min.log_prob(a_eco[:, 1].long().reshape(-1))
        log_prob_dist = dist_dist.log_prob(a_dist.long().reshape(-1))
        log_prob_target = dist_target.log_prob(a_target.long().reshape(-1))

        # Loss components
        loss_dip = -(log_prob_dip * adv).mean()
        loss_eco = -((log_prob_sol + log_prob_min) * adv).mean()
        loss_dist = -(log_prob_dist * adv).mean()
        loss_target = -(log_prob_target * adv).mean()

        actor_loss = loss_dip + loss_eco + loss_dist + loss_target
        critic_loss = torch.nn.functional.mse_loss(current_values.view(-1), returns.view(-1))

        # Entropy for exploration
        total_entropy = (dist_dip.entropy() + dist_sol.entropy() + 
                         dist_min.entropy() + dist_dist.entropy() + 
                         dist_target.entropy()).mean()

        total_loss = actor_loss + (0.5 * critic_loss) - (0.01 * total_entropy)

        self.log_dict({
            "loss/total": total_loss,
            "loss/actor": actor_loss,
            "loss/critic": critic_loss,
            "stats/entropy": total_entropy,
            "stats/advantage": adv.mean()
        }, prog_bar=True)

        return total_loss

    def forward(self, board_state, global_stats):
        return self.model(board_state, global_stats)
    
    def configure_optimizers(self):
        return self.model.configure_optimizers()

    def train_dataloader(self):
        return DataLoader(RLDataset(self.buffer, self.batch_size), batch_size=None)

def train_agent_lightning():
    BATCH_SIZE = 64
    BUFFER_CAPACITY = 10000
    COLLECT_STEPS_PER_EPOCH = 1000
    TRAIN_EPOCHS = 500
    LEARNING_RATE = 1e-4
    torch.set_float32_matmul_precision('high')

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
        callbacks=[EarlyStopping(
                                monitor='loss/total', # Change this from 'episode/total_reward'
                                min_delta=0.00,
                                patience=20,
                                verbose=True,
                                mode='min'
                                )],
        gradient_clip_val=0.5,
        logger=MLFlowLogger(experiment_name="Strategy_Masked", tracking_uri="file:./ml-runs"),
    )

    trainer.fit(lightning_module)
    torch.save(lightning_module.model.state_dict(), "trained_model.pth")

if __name__ == "__main__":
    train_agent_lightning()