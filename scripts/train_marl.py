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
from models.marl_critic import MARL_Strategy, SoldierAgent
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

class SoldierSandbox:
    """Trains the Tactical SoldierAgent to reach goals in a simplified environment."""
    def __init__(self):
        self.model = SoldierAgent(in_channels=8, goal_dim=16)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def train(self, steps=5000):
        print("Training Tactical SoldierAgent in sandbox...")
        for i in range(steps):
            # Synthetic local view (5x5x8) and goal
            local_view = torch.randn(32, 8, 5, 5)
            goal = torch.randn(32, 16)
            
            # Target action: simply move towards "highest goal value" (Dummy heuristic for pre-training)
            target = torch.randint(0, 5, (32,))
            
            logits = self.model(local_view, goal)
            loss = F.cross_entropy(logits, target)
            
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            if i % 1000 == 0: print(f"Sandbox Step {i}, Loss: {loss.item():.4f}")
        
        # Freeze model
        for p in self.model.parameters(): p.requires_grad = False
        print("Tactical SoldierAgent trained and frozen.")
        return self.model

class StrategyLightningModule(pl.LightningModule):
    def __init__(self, env: gym.Env, buffer: ReplayBuffer, lr: float, collect_steps: int, soldier_model: SoldierAgent, batch_size: int = 64):
        super().__init__()
        self.env = env
        self.buffer = buffer
        self.lr = lr
        self.collect_steps = collect_steps
        self.batch_size = batch_size
        self.gamma = 0.99
        self.lam = 0.95
        
        self.model = MARL_Strategy(in_channels=8, stats_dim=27)
        self.soldier_agent = soldier_model # Frozen Tactical Agent
        
        self.current_goal = None
        self.goal_persistence = 10 # Change goal every 10 turns
        
        self.obs, self.info = self.env.reset()
        self.total_reward = 0

    @torch.no_grad()
    def _process_obs(self, obs_batch, info_batch=None, log=True):
        device = self.device
        if "board_stats" in obs_batch:
            boards = torch.from_numpy(obs_batch["board_state"]).to(device, dtype=torch.float32).unsqueeze(0) / 255.0
            stats_np = np.concatenate([obs_batch["player_resources"], obs_batch["opponent_resources"], obs_batch["board_stats"], [obs_batch["turn_number"]]]).astype(np.float32)
            stats = torch.from_numpy(stats_np).to(device, dtype=torch.float32).unsqueeze(0) / 1000.0
        else:
            boards = obs_batch["board_state"].to(device, dtype=torch.float32) / 255.0
            stats = obs_batch["full_stats"].to(device, dtype=torch.float32) / 1000.0

        t_mask, b_mask = None, None
        if info_batch is not None:
            if isinstance(info_batch, dict):
                t_mask = torch.from_numpy(info_batch.get("action_mask")).to(device, dtype=torch.bool).view(1, -1)
                b_mask = torch.from_numpy(info_batch.get("build_mask")).to(device, dtype=torch.bool).view(1, -1)
            else:
                t_mask, b_mask = info_batch[0].to(device, dtype=torch.bool), info_batch[1].to(device, dtype=torch.bool)
        else:
            t_mask, b_mask = torch.ones((1, 256), device=device, dtype=torch.bool), torch.ones((1, 8), device=device, dtype=torch.bool)
        
        if log and "player_resources" in obs_batch:
            res = torch.as_tensor(obs_batch["player_resources"], dtype=torch.float32)
            if res.dim() == 1:
                self.log("game_stats/player_gold", res[0])
                self.log("game_stats/player_wood", res[1])
                self.log("game_stats/player_workers", res[2])
                self.log("game_stats/player_mines", res[3])
                self.log("game_stats/player_food", res[4])
                self.log("game_stats/gold_capacity", res[5])
                self.log("game_stats/wood_capacity", res[6])
                self.log("game_stats/worker_capacity", res[7])
                self.log("game_stats/food_capacity", res[8])
            else:
                self.log("game_stats/player_gold", res[:, 0].mean())
                self.log("game_stats/player_wood", res[:, 1].mean())
                self.log("game_stats/player_workers", res[:, 2].mean())
                self.log("game_stats/player_mines", res[:, 3].mean())
                self.log("game_stats/player_food", res[:, 4].mean())
                self.log("game_stats/gold_capacity", res[:, 5].mean())  
                self.log("game_stats/wood_capacity", res[:, 6].mean())
                self.log("game_stats/worker_capacity", res[:, 7].mean())
                self.log("game_stats/food_capacity", res[:, 8].mean())
        
        # In batch mode, board_stats info is in 'mine_stats'
        stats_raw = obs_batch.get("board_stats") if "board_stats" in obs_batch else obs_batch.get("mine_stats")
        if log and stats_raw is not None:
            stats_raw = torch.as_tensor(stats_raw, dtype=torch.float32)
            if stats_raw.dim() > 1:
                self.log("game_stats/mines", stats_raw[:, 1].mean())
                self.log("game_stats/gold_income", stats_raw[:, 2].mean())
                self.log("game_stats/wood_income", stats_raw[:, 3].mean())
                self.log("game_stats/trade_routes", stats_raw[:, 4].mean())
                self.log("game_stats/p1_tiles", stats_raw[:, 5].mean())
                self.log("game_stats/p2_tiles", stats_raw[:, 6].mean())
                self.log("game_stats/potential_trade_routes", stats_raw[:, 7].mean())
                self.log("game_stats/crop_fields", stats_raw[:, 8].mean())
                self.log("game_stats/net_income", stats_raw[:, 9].mean())
                self.log("game_stats/lost_gold", stats_raw[:, 10].mean())
                self.log("game_stats/lost_wood", stats_raw[:, 11].mean())
                self.log("game_stats/lost_food", stats_raw[:, 12].mean())
                self.log("game_stats/defeated_workers", stats_raw[:, 13].mean())
            else:
                self.log("game_stats/mines", stats_raw[1])
                self.log("game_stats/gold_income", stats_raw[2])
                self.log("game_stats/wood_income", stats_raw[3])
                self.log("game_stats/trade_routes", stats_raw[4])
                self.log("game_stats/p1_tiles", stats_raw[5])
                self.log("game_stats/p2_tiles", stats_raw[6])
                self.log("game_stats/potential_trade_routes", stats_raw[7])
                self.log("game_stats/crop_fields", stats_raw[8])
                self.log("game_stats/net_income", stats_raw[9])
                self.log("game_stats/lost_gold", stats_raw[10])
                self.log("game_stats/lost_wood", stats_raw[11])
                self.log("game_stats/lost_food", stats_raw[12])
                self.log("game_stats/defeated_workers", stats_raw[13])

        return boards, stats, t_mask, b_mask

    def on_train_epoch_start(self):
        trajectory_data = []
        start_idx = self.buffer.idx
        for _ in range(self.collect_steps):
            board_t, stats_t, t_mask, b_mask = self._process_obs(self.obs, self.info)
            with torch.no_grad():
                res = self.model(board_t, stats_t, target_mask=t_mask, build_mask=b_mask)
            
            if self.env.current_turn % self.goal_persistence == 0:
                self.current_goal = res["goal"]

            action = {
                "diplomacy": torch.distributions.Categorical(logits=res["dip"]).sample().item(),
                "economy": [torch.distributions.Categorical(logits=l).sample().item() for l in res["eco"]],
                "target_tile": torch.distributions.Categorical(logits=res["mil"]).sample().item()
            }

            next_obs, reward, terminated, truncated, next_info = self.env.step(action, self.soldier_agent, self.current_goal)
            
            self.total_reward += reward
            trajectory_data.append({'reward': reward, 'value': res["value"].item(), 'done': terminated or truncated})
            self.buffer.add(self.obs, action, reward, next_obs, terminated, truncated, self.info)
            self.obs, self.info = next_obs, next_info
            if terminated or truncated:
                self.log("episode/total_reward", self.total_reward, prog_bar=True); self.total_reward = 0
                self.obs, self.info = self.env.reset()
        
        self._compute_gae(trajectory_data, self.obs, self.info, start_idx)

    def _compute_gae(self, trajectory, last_obs, last_info, start_idx):
        rewards = np.array([e['reward'] for e in trajectory]) * 0.1
        rewards = np.clip(rewards, -5.0, 5.0)
        values = np.array([e['value'] for e in trajectory])
        dones = np.array([e['done'] for e in trajectory])
        
        with torch.no_grad():
            b, s, tm, bm = self._process_obs(last_obs, last_info, log=False)
            res = self.model(b, s, tm, bm)
            next_value = res["value"].item()

        advantages = np.zeros_like(rewards)
        last_gae_lam = 0
        for t in reversed(range(len(trajectory))):
            next_non_terminal = 1.0 - dones[t]
            next_val = next_value if t == len(trajectory) - 1 else values[t + 1]
            delta = rewards[t] + 0.99 * next_val * next_non_terminal - values[t]
            last_gae_lam = delta + 0.99 * 0.95 * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam
                
        returns = advantages + values
        self.buffer.store_gae(start_idx, advantages, returns)

    def _prefill_buffer(self):
        if self.buffer.size >= self.batch_size: return
        print(f"Pre-filling buffer...")
        while self.buffer.size < self.batch_size:
            board_t, stats_t, t_mask, b_mask = self._process_obs(self.obs, self.info, log=False)
            with torch.no_grad():
                res = self.model(board_t, stats_t, target_mask=t_mask, build_mask=b_mask)
            eco_l = res["eco"]
            action = {
                "diplomacy": torch.distributions.Categorical(logits=res["dip"]).sample().item(),
                "economy": [torch.distributions.Categorical(logits=l).sample().item() for l in eco_l],
                "target_tile": torch.distributions.Categorical(logits=res["mil"]).sample().item()
            }
            next_obs, reward, terminated, truncated, next_info = self.env.step(action, self.soldier_agent, res["goal"])
            self.buffer.add(self.obs, action, reward, next_obs, terminated, truncated, self.info)
            self.obs, self.info = next_obs, next_info
            if terminated or truncated: self.obs, self.info = self.env.reset()

    def on_fit_start(self): self._prefill_buffer()

    def training_step(self, batch, batch_idx):
        (states, a_dip, a_eco, a_target, rewards, _, t_d, t_c, m_target, m_build, pre_adv, pre_returns) = batch
        boards, stats, t_mask, b_mask = self._process_obs(states, (m_target, m_build))
        res = self.model(boards, stats, target_mask=t_mask, build_mask=b_mask)
        
        def stab(l): return torch.clamp(torch.nan_to_num(l, nan=0.0, posinf=20.0, neginf=-20.0), -20, 20)
        res["dip"], res["mil"] = stab(res["dip"]), stab(res["mil"])
        res["eco"] = [stab(l) for l in res["eco"]]
        
        adv = (pre_adv - pre_adv.mean()) / (pre_adv.std() + 1e-8)
        d_dip, d_tar = torch.distributions.Categorical(logits=res["dip"]), torch.distributions.Categorical(logits=res["mil"])
        lp = d_dip.log_prob(a_dip) + d_tar.log_prob(a_target)
        for i, l in enumerate(res["eco"]): lp += torch.distributions.Categorical(logits=l).log_prob(a_eco[:, i])

        actor_loss = -(lp * adv).mean()
        critic_loss = F.huber_loss(res["value"].view(-1), pre_returns.view(-1), delta=1.0)
        total_loss = actor_loss + 0.1 * critic_loss
        self.log("loss/total", total_loss, prog_bar=True)
        return total_loss

    def configure_optimizers(self): return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def train_dataloader(self):
        return DataLoader(RLDataset(self.buffer, self.batch_size), batch_size=None)

def train_agent_lightning():
    BATCH_SIZE, BUFFER_CAP, COLLECT_STEPS, EPOCHS, LR = 128, 10000, 1024, 1000, 3e-5

    sandbox = SoldierSandbox()
    soldier_model = sandbox.train()
    
    env = StrategyEnv()

    replay_buffer = ReplayBuffer(BUFFER_CAP, env.observation_space, env.action_space)

    model = StrategyLightningModule(env, replay_buffer, LR, COLLECT_STEPS, soldier_model, batch_size=BATCH_SIZE)
    
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        gradient_clip_val=0.5,
        precision="32", 
        logger=MLFlowLogger(experiment_name="MARL_Strategy_Optimized", tracking_uri="file:./ml-runs"),
        callbacks=[EarlyStopping(monitor='loss/total', patience=20, mode='min')],
        log_every_n_steps=10,
        limit_train_batches=100 
    )

    trainer.fit(model)
    torch.save(model.model.state_dict(), "marl_strategy_optimized.pth")


if __name__ == "__main__":
    train_agent_lightning()
