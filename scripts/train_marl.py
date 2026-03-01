import gymnasium as gym
import torch
import torch.nn.functional as F
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
from models.marl_critic import MARL_Strategy
from data.replay_buffer import ReplayBuffer

class RLDataset(IterableDataset):
    def __init__(self, buffer: ReplayBuffer, batch_size: int):
        super().__init__()
        self.buffer = buffer
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[Tuple]:
        while True:
            if self.buffer.size >= self.batch_size:
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
        
        # Hyperparameters
        self.entropy_coeff = 0.1
        self.entropy_decay = 0.998 # Slower decay
        self.min_entropy = 0.05    # Higher minimum entropy to maintain exploration
        self.value_loss_coeff = 0.5
        self.reward_scaling_factor = 1e-1

        # MARL Model initialization
        self.model = MARL_Strategy(in_channels=5, stats_dim=20)
        
        # Optimize model if not on Windows (torch.compile is better supported on Linux)
        if hasattr(torch, "compile") and sys.platform != "win32":
            try:
                self.model = torch.compile(self.model)
            except Exception:
                pass
        
        self.obs, self.info = self.env.reset()
        self.total_reward = 0
        self.save_hyperparameters(ignore=['env', 'buffer', 'model'])

    @torch.no_grad()
    def _process_obs(self, obs_batch, info_batch=None):
        device = self.device
        if "board_stats" in obs_batch: # Single step collection
            boards = torch.from_numpy(obs_batch["board_state"]).to(device, dtype=torch.float32, non_blocking=True).unsqueeze(0) / 255.0
            
            # Efficiently combine stats
            stats_np = np.concatenate([
                obs_batch["player_resources"],
                obs_batch["opponent_resources"],
                obs_batch["board_stats"],
                [obs_batch["turn_number"]]
            ]).astype(np.float32)
            stats = torch.from_numpy(stats_np).to(device, dtype=torch.float32, non_blocking=True).unsqueeze(0) / 1000.0
        else: # Batch from ReplayBuffer
            boards = obs_batch["board_state"].to(device, dtype=torch.float32, non_blocking=True) / 255.0
            stats = obs_batch["full_stats"].to(device, dtype=torch.float32, non_blocking=True) / 1000.0

        t_mask, b_mask = None, None
        if info_batch is not None:
            if isinstance(info_batch, dict):
                t_mask = torch.from_numpy(info_batch.get("action_mask")).to(device, dtype=torch.bool, non_blocking=True).view(1, -1)
                b_mask = torch.from_numpy(info_batch.get("build_mask")).to(device, dtype=torch.bool, non_blocking=True).view(1, -1)
            else:
                t_mask = info_batch[0].to(device, dtype=torch.bool, non_blocking=True)
                b_mask = info_batch[1].to(device, dtype=torch.bool, non_blocking=True)

        return boards, stats, t_mask, b_mask

    def _compute_gae(self, trajectory, last_obs, last_info, start_idx):
        rewards = np.array([e['reward'] for e in trajectory]) * self.reward_scaling_factor
        values = np.array([e['value'] for e in trajectory])
        dones = np.array([e['done'] for e in trajectory])
        
        with torch.no_grad():
            b, s, tm, bm = self._process_obs(last_obs, last_info)
            res = self.model(b, s, tm, bm)
            next_value = res["value"].item()

        advantages = np.zeros_like(rewards)
        last_gae_lam = 0
        for t in reversed(range(len(trajectory))):
            next_non_terminal = 1.0 - dones[t]
            next_val = next_value if t == len(trajectory) - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            last_gae_lam = delta + self.gamma * self.lam * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam
                
        returns = advantages + values
        self.buffer.store_gae(start_idx, advantages, returns)

    def _prefill_buffer(self):
        if self.buffer.size >= self.batch_size:
            return

        print(f"Pre-filling buffer with {self.batch_size} steps...")
        while self.buffer.size < self.batch_size:
            board_t, stats_t, t_mask, b_mask = self._process_obs(self.obs, self.info)
            with torch.no_grad():
                res = self.model(board_t, stats_t, target_mask=t_mask, build_mask=b_mask)
            
            eco_l = res["eco"]
            mil_l = res["mil"]
            dip_l = res["dip"]

            action = {
                "diplomacy": torch.distributions.Categorical(logits=dip_l).sample().item(),
                "economy": [
                    torch.distributions.Categorical(logits=eco_l[0]).sample().item(),
                    torch.distributions.Categorical(logits=eco_l[1]).sample().item(),
                    torch.distributions.Categorical(logits=eco_l[2]).sample().item(),
                    torch.distributions.Categorical(logits=eco_l[3]).sample().item()
                ],
                "distribution": torch.distributions.Categorical(logits=mil_l[0]).sample().item(),
                "target_tile": torch.distributions.Categorical(logits=mil_l[1]).sample().item()
            }

            next_obs, reward, terminated, truncated, next_info = self.env.step(action)
            self.buffer.add(self.obs, action, reward, next_obs, terminated, truncated, self.info)
            self.obs, self.info = next_obs, next_info
            if terminated or truncated:
                self.obs, self.info = self.env.reset()

    def on_fit_start(self):
        self._prefill_buffer()

    def on_train_epoch_start(self):
        trajectory_data = []
        start_idx = self.buffer.idx
        for _ in range(self.collect_steps):
            board_t, stats_t, t_mask, b_mask = self._process_obs(self.obs, self.info)
            with torch.no_grad():
                res = self.model(board_t, stats_t, target_mask=t_mask, build_mask=b_mask)
            
            eco_l = res["eco"]
            mil_l = res["mil"]
            dip_l = res["dip"]

            action = {
                "diplomacy": torch.distributions.Categorical(logits=dip_l).sample().item(),
                "economy": [
                    torch.distributions.Categorical(logits=eco_l[0]).sample().item(),
                    torch.distributions.Categorical(logits=eco_l[1]).sample().item(),
                    torch.distributions.Categorical(logits=eco_l[2]).sample().item(),
                    torch.distributions.Categorical(logits=eco_l[3]).sample().item()
                ],
                "distribution": torch.distributions.Categorical(logits=mil_l[0]).sample().item(),
                "target_tile": torch.distributions.Categorical(logits=mil_l[1]).sample().item()
            }

            next_obs, reward, terminated, truncated, next_info = self.env.step(action)
            self.total_reward += reward

            trajectory_data.append({'reward': reward, 'value': res["value"].item(), 'done': terminated or truncated})
            self.buffer.add(self.obs, action, reward, next_obs, terminated, truncated, self.info)

            self.obs, self.info = next_obs, next_info
            if terminated or truncated:
                self.log("episode/total_reward", self.total_reward, prog_bar=True)
                self.total_reward = 0
                self.obs, self.info = self.env.reset()

        self._compute_gae(trajectory_data, self.obs, self.info, start_idx)

    def training_step(self, batch, batch_idx):
        (states, a_dip, a_eco, a_dist, a_target, rewards, _, _, _, m_target, m_build, pre_adv, pre_returns) = batch

        boards, stats, t_mask, b_mask = self._process_obs(states, (m_target, m_build))
        res = self.model(boards, stats, target_mask=t_mask, build_mask=b_mask)

        res_gold = states["player_resources"][:, 0].mean()
        self.log("game_stats/player_gold", res_gold, on_step=False, on_epoch=True)

        adv = (pre_adv - pre_adv.mean()) / (pre_adv.std() + 1e-8)

        # Vectorized distribution operations
        d_dip = torch.distributions.Categorical(logits=res["dip"])
        d_sol = torch.distributions.Categorical(logits=res["eco"][0])
        d_min = torch.distributions.Categorical(logits=res["eco"][1])
        d_tra = torch.distributions.Categorical(logits=res["eco"][2])
        d_wh  = torch.distributions.Categorical(logits=res["eco"][3])
        d_mil = torch.distributions.Categorical(logits=res["mil"][0])
        d_tar = torch.distributions.Categorical(logits=res["mil"][1])

        lp = (d_dip.log_prob(a_dip) + 
              d_sol.log_prob(a_eco[:, 0]) +
              d_min.log_prob(a_eco[:, 1]) + 
              d_tra.log_prob(a_eco[:, 2]) + 
              d_wh.log_prob(a_eco[:, 3]) + 
              d_mil.log_prob(a_dist) + 
              d_tar.log_prob(a_target))

        actor_loss = -(lp * adv).mean()
        critic_loss = F.huber_loss(res["value"].view(-1), pre_returns.view(-1), delta=1.0)
        
        entropy = (d_dip.entropy() + d_sol.entropy() + d_min.entropy() + 
                   d_tra.entropy() + d_wh.entropy() + d_mil.entropy() + 
                   d_tar.entropy()).mean()

        total_loss = actor_loss + (self.value_loss_coeff * critic_loss) - (self.entropy_coeff * entropy)

        self.log_dict({
            "loss/total": total_loss,
            "loss/actor": actor_loss,
            "loss/critic": critic_loss,
            "stats/entropy": entropy
        }, on_step=False, on_epoch=True, prog_bar=True)

        return total_loss

    def on_train_epoch_end(self):
        self.entropy_coeff = max(self.min_entropy, self.entropy_coeff * self.entropy_decay)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        return DataLoader(
            RLDataset(self.buffer, self.batch_size), 
            batch_size=None,
            pin_memory=(self.device.type == "cuda")
        )

def train_agent_lightning():
    # Performance setup
    BATCH_SIZE, BUFFER_CAP, COLLECT_STEPS, EPOCHS, LR = 128, 10000, 1024, 1000, 5e-5
    torch.set_float32_matmul_precision('high')

    env = StrategyEnv()
    replay_buffer = ReplayBuffer(BUFFER_CAP, env.observation_space, env.action_space)

    model = StrategyLightningModule(env, replay_buffer, LR, COLLECT_STEPS, batch_size=BATCH_SIZE)

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        gradient_clip_val=1.0,
        precision="16-mixed", 
        logger=MLFlowLogger(experiment_name="MARL_Strategy_Optimized", tracking_uri="file:./ml-runs"),
        callbacks=[EarlyStopping(monitor='loss/total', patience=100, mode='min')],
        log_every_n_steps=10,
        limit_train_batches=10 # Optimize updates per collection
    )

    trainer.fit(model)
    torch.save(model.model.state_dict(), "marl_strategy_optimized.pth")

if __name__ == "__main__":
    train_agent_lightning()
