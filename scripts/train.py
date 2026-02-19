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
    def __init__(self, env: gym.Env, buffer: ReplayBuffer, lr: float, collect_steps: int, gamma: float = 0.99, batch_size: int = 64, lam=0.95):
        super().__init__()
        self.env = env
        self.buffer = buffer
        self.lr = lr
        self.collect_steps = collect_steps
        self.gamma = gamma
        self.batch_size = batch_size
        self.lam = lam
        self.entropy_coeff = 0.1  # Start much higher
        self.entropy_decay = 0.995 # Slowly reduce exploration as it gets better
        self.min_entropy = 0.01
        self.value_loss_coeff = 0.05  # Reduced from 0.5 to balance high Critic loss
        self.reward_scaling_factor = 1e-3 # Scale rewards down (e.g., -100,000 becomes -100)

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
                m_stats = torch.as_tensor(obs_batch["board_stats"], dtype=torch.float32, device=self.device).view(-1, 6)
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
                    b_mask = torch.as_tensor(
                                                info_batch.get("build_mask", np.ones(7)), 
                                                dtype=torch.bool, 
                                                device=self.device
                                            ).view(-1, 7) 
                else:
                    # If masks were already batched by the ReplayBuffer/DataLoader
                    t_mask = info_batch[0].to(self.device).view(-1, 64) # m_target
                    b_mask = info_batch[1].to(self.device).view(-1, 7)  # m_build

            if boards.dim() == 5 and boards.size(0) == 1:
                boards = boards.squeeze(0)
                stats = stats.squeeze(0)
                t_mask = t_mask.squeeze(0)
                b_mask = b_mask.squeeze(0)

            # --- Log ---
            # We take the mean across the batch dimension (dim 0) 
            # and then select the specific resource index.

            # Assuming index 0 is Gold, 1 is Wood, 2 is Soldiers
            if "player_resources" in obs_batch:
                # Handle batch from ReplayBuffer (2D) vs Single Step (1D)
                res = torch.as_tensor(obs_batch["player_resources"], dtype=torch.float32)
                if res.dim() == 1:
                    self.log("game_stats/player_gold", res[0])
                    self.log("game_stats/player_wood", res[1])
                    self.log("game_stats/player_soldiers", res[2])
                    self.log("game_stats/player_mines", res[3])
                else:
                    # Log the average across the current training batch
                    self.log("game_stats/player_gold", res[:, 0].mean())
                    self.log("game_stats/player_wood", res[:, 1].mean())
                    self.log("game_stats/player_soldiers", res[:, 2].mean())
                    self.log("game_stats/player_mines", res[:, 3].mean())
            
            if "board_stats" in obs_batch:
                stats_raw = obs_batch["board_stats"]
                # If stats_raw is a batch, take the mean
                if stats_raw.ndim > 1:
                    self.log("game_stats/mines", stats_raw[:, 1].mean())
                    self.log("game_stats/gold_income", stats_raw[:, 2].mean())
                    self.log("game_stats/wood_income", stats_raw[:, 3].mean())
                    self.log("game_stats/trade_routes", stats_raw[:, 4].mean())
                else:
                    self.log("game_stats/mines", stats_raw[1])
                    self.log("game_stats/gold_income", stats_raw[2])
                    self.log("game_stats/wood_income", stats_raw[3])
                    self.log("game_stats/trade_routes", stats_raw[4])

            return boards, stats, t_mask, b_mask

    def _compute_gae(self, trajectory, start_idx):
            rewards = np.array([e['reward'] for e in trajectory]) * self.reward_scaling_factor
            values = np.array([e['value'] for e in trajectory])
            dones = np.array([e['done'] for e in trajectory])
            
            # Get V(s_next) for the very last step
            with torch.no_grad():
                next_board, next_stats, _, _ = self._process_obs(self.obs)
                _, _, _, _, last_value = self.model(next_board, next_stats)
                next_value = last_value.item()

            advantages = np.zeros_like(rewards)
            last_gae_lam = 0
            
            for t in reversed(range(self.collect_steps)):
                if t == self.collect_steps - 1:
                    next_non_terminal = 1.0 - dones[t]
                    next_val = next_value
                else:
                    next_non_terminal = 1.0 - dones[t]
                    next_val = values[t + 1]
                
                # TD Error: delta = r + gamma * V(s_next) - V(s)
                delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
                
                # GAE: A = delta + gamma * lambda * A_next
                last_gae_lam = delta + self.gamma * self.lam * next_non_terminal * last_gae_lam
                advantages[t] = last_gae_lam
                
            returns = advantages + values
            self.buffer.store_gae(start_idx, advantages, returns)


    def on_train_start(self):
            """Fill the buffer with initial steps before training begins."""
            print(f"Pre-filling buffer with {self.batch_size} steps...")
            
            while self.buffer.size < self.batch_size:
                # Process current observation
                board_t, stats_t, t_mask, b_mask = self._process_obs(self.obs, self.info)

                with torch.no_grad():
                    dip_logits, eco_logits, dist_logits, target_logits, _ = self.model(
                        board_t, stats_t, target_mask=t_mask, build_mask=b_mask
                    )
                
                sol_logits, mine_logits, trade_logits = eco_logits

                # Sample Actions (Random-ish exploration start)
                action = {
                    "diplomacy": torch.distributions.Categorical(logits=dip_logits).sample().item(),
                    "economy": [
                        torch.distributions.Categorical(logits=sol_logits).sample().item(),
                        torch.distributions.Categorical(logits=mine_logits).sample().item(),
                        torch.distributions.Categorical(logits=trade_logits).sample().item()
                    ],
                    "distribution": torch.distributions.Categorical(logits=dist_logits).sample().item(),
                    "target_tile": torch.distributions.Categorical(logits=target_logits).sample().item()
                }

                next_obs, reward, terminated, truncated, next_info = self.env.step(action)
                
                self.buffer.add(
                    state=self.obs,
                    action=action, 
                    reward=reward, 
                    next_state=next_obs,
                    terminated=terminated, 
                    truncated=truncated,
                    info=self.info
                )

                self.obs = next_obs
                self.info = next_info
                
                if terminated or truncated:
                    self.obs, self.info = self.env.reset()

    def on_train_epoch_start(self):
            # We track the start index in the buffer to update it later with GAE
            start_idx = self.buffer.idx
            trajectory_data = []

            for _ in range(self.collect_steps):
                # 1. Process current observation
                board_t, stats_t, t_mask, b_mask = self._process_obs(self.obs, self.info)

                # 2. Policy Inference (No Gradient)
                with torch.no_grad():
                    dip_logits, eco_logits, dist_logits, target_logits, value = self.model(
                        board_t, stats_t, target_mask=t_mask, build_mask=b_mask
                    )
                
                sol_logits, mine_logits, trade_logits = eco_logits

                # 3. Sample Actions
                action = {
                    "diplomacy": torch.distributions.Categorical(logits=dip_logits).sample().item(),
                    "economy": [
                        torch.distributions.Categorical(logits=sol_logits).sample().item(),
                        torch.distributions.Categorical(logits=mine_logits).sample().item(),
                        torch.distributions.Categorical(logits=trade_logits).sample().item()
                    ],
                    "distribution": torch.distributions.Categorical(logits=dist_logits).sample().item(),
                    "target_tile": torch.distributions.Categorical(logits=target_logits).sample().item()
                }

                # 4. Step Environment
                next_obs, reward, terminated, truncated, next_info = self.env.step(action)
                self.total_reward += reward

                # 5. Record metadata for GAE calculation
                trajectory_data.append({
                    'reward': reward,
                    'value': value.item(),
                    'done': terminated or truncated
                })
                
                # 6. Store in Buffer
                self.buffer.add(
                    state=self.obs,
                    action=action, 
                    reward=reward, 
                    next_state=next_obs,
                    terminated=terminated, 
                    truncated=truncated,
                    info=self.info 
                )

                # 7. Update pointers
                self.obs = next_obs
                self.info = next_info

                if terminated or truncated:
                    self.obs, self.info = self.env.reset()
                    self.log("episode/total_reward", self.total_reward, on_step=False, on_epoch=True)
                    self.total_reward = 0
                    self.episode_count += 1

            # --- IMPORTANT: Compute GAE only AFTER the collection loop is done ---
            self._compute_gae(trajectory_data, start_idx)

    def training_step(self, batch, batch_idx):
            # 1. Unpack the full batch (now including pre-calculated GAE data)
            (
                states, a_dip, a_eco, a_dist, a_target, rewards, 
                next_states, terminals, truncated, m_target, m_build,
                pre_adv, pre_returns  # These are the GAE values from the buffer
            ) = batch

            opt = self.optimizers()
            current_lr = opt.param_groups[0]['lr']
            self.log("stats/learning_rate",
                            current_lr,
                            on_step=True,
                            on_epoch=False, 
                            prog_bar=True)

            # SQUEEZE: Remove the extra dimension [1, Batch, ...] -> [Batch, ...]
            if rewards.dim() == 2:
                a_dip = a_dip.squeeze(0)
                a_eco = a_eco.squeeze(0)
                a_dist = a_dist.squeeze(0)
                a_target = a_target.squeeze(0)
                rewards = rewards.squeeze(0)
                terminals = terminals.squeeze(0)
                pre_adv = pre_adv.squeeze(0)
                pre_returns = pre_returns.squeeze(0)

            # 2. Process observations and masks
            boards, stats, t_mask, b_mask = self._process_obs(states, (m_target, m_build))
            
            # 3. Get current policy logits and state values
            # We don't need next_states here anymore because GAE was computed during collection!
            dip_logits, (sol_logits, mine_logits, trade_logits), dist_logits, target_logits, current_values = self.model(
                boards, stats, target_mask=t_mask, build_mask=b_mask
            )


            self.log("logits/trade_route_avg", trade_logits.mean())
            self.log("logits/trade_route_max", trade_logits.max())
            
            # 4. Advantage Normalization
            # Using pre-calculated advantages from GAE
            adv = (pre_adv - pre_adv.mean()) / (pre_adv.std() + 1e-8)

            # 5. Policy Distributions (Actor)
            dist_dip = torch.distributions.Categorical(logits=dip_logits)
            dist_sol = torch.distributions.Categorical(logits=sol_logits)
            dist_min = torch.distributions.Categorical(logits=mine_logits)
            dist_dist = torch.distributions.Categorical(logits=dist_logits)
            dist_target = torch.distributions.Categorical(logits=target_logits)
            dist_trade = torch.distributions.Categorical(logits=trade_logits)

            # 6. Multi-Head Actor Loss
            if a_eco.dim() == 1: a_eco = a_eco.unsqueeze(0)
            
            log_prob_dip = dist_dip.log_prob(a_dip.long().reshape(-1))
            log_prob_sol = dist_sol.log_prob(a_eco[:, 0].long().reshape(-1))
            log_prob_min = dist_min.log_prob(a_eco[:, 1].long().reshape(-1))
            log_prob_tra = dist_trade.log_prob(a_eco[:, 2].long().reshape(-1)) 
            log_prob_dist = dist_dist.log_prob(a_dist.long().reshape(-1))
            log_prob_target = dist_target.log_prob(a_target.long().reshape(-1))

            # Actor losses (Advantage tells us if the action was better than average)
            loss_dip = -(log_prob_dip * adv).mean()
            loss_eco = -((log_prob_sol + log_prob_min + log_prob_tra) * adv).mean()
            loss_dist = -(log_prob_dist * adv).mean()
            loss_target = -(log_prob_target * adv).mean()

            actor_loss = loss_dip + loss_eco + loss_dist + loss_target

            # 7. Critic Loss
            # The Critic tries to predict the GAE returns
            critic_loss = torch.nn.functional.huber_loss(
                current_values.view(-1), 
                pre_returns.view(-1), 
                delta=1.0 # Adjust delta based on scaled reward range /(previous 10)
            )
            # 8. Entropy for Exploration
            total_entropy = (dist_dip.entropy() + dist_sol.entropy() + 
                            dist_min.entropy() + dist_trade.entropy() + 
                            dist_dist.entropy() + dist_target.entropy()).mean()

            # Final Weighted Loss
            total_loss = actor_loss + (self.value_loss_coeff * critic_loss) - (self.entropy_coeff * total_entropy)

            # 9. GAE & Training Metrics Logging
            # 'explained_variance' tells you how good the Critic is at its job
            y_pred = current_values.view(-1).detach().cpu().numpy()
            y_true = pre_returns.view(-1).detach().cpu().numpy()
            var_y = np.var(y_true)
            explained_var = 1.0 - np.var(y_true - y_pred) / (var_y + 1e-8) if var_y > 0 else 0

            self.log_dict({
                "loss/total": total_loss,
                "loss/actor": actor_loss,
                "loss/critic": critic_loss,
                "stats/entropy": total_entropy,
                "stats/explained_variance": explained_var,
                "stats/mean_advantage": pre_adv.mean(),
                "stats/mean_returns": pre_returns.mean(),
                "stats/value_mean": current_values.mean()
            }, prog_bar=True)

            return total_loss

    def on_train_epoch_end(self):
        # Slowly decay entropy to allow convergence later, but keep a minimum floor
        self.entropy_coeff = max(self.min_entropy, self.entropy_coeff * self.entropy_decay)
        self.log("stats/entropy_coeff", self.entropy_coeff)

    def forward(self, board_state, global_stats):
        return self.model(board_state, global_stats)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        # mode='min' because we want to reduce LR when the LOSS stops decreasing
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5,      # Cut LR in half when hitting a plateau
            patience=10,     # How many epochs/steps to wait before cutting
            verbose=True     # Helpful to see the "Reducing learning rate..." message in console
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "loss/total", # MUST match the string in self.log("loss/total", ...)
                "interval": "epoch",     # or "step" depending on how often you log
                "frequency": 1
            }
        }

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
                                patience=100,
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
