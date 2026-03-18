import gymnasium as gym
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from typing import Iterator, Tuple, Dict, Any
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import EarlyStopping
import numpy as np

from enviroment.strategy_env import StrategyEnv
from data.replay_buffer import ReplayBuffer
from models.marl_critic import MARL_Strategy, SoldierAgent

class SoldierSandbox:
    """
    Simulates a tactical environment to pre-train the SoldierAgent.
    """
    def __init__(self):
        self.model = SoldierAgent(in_channels=8, goal_dim=16)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def train(self, steps=5000):
        print("Training Tactical SoldierAgent in sandbox...")
        for i in range(steps):
            # Synthetic local view (5x5x8) and goal
            local_view = torch.randn(32, 8, 5, 5)
            goal = torch.randn(32, 16)
            
            # Slot 2: Aggression Mode (randomly on/off)
            goal[:, 2] = (torch.rand(32) > 0.5).float()

            # Hybrid Goal: First 2 slots are relative (dr, dc)
            target = []
            for b in range(32):
                dr, dc = goal[b, 0].item(), goal[b, 1].item()
                if abs(dr) > abs(dc):
                    action = 2 if dr > 0 else 1 # South if dr > 0, else North
                elif abs(dc) > 0:
                    action = 3 if dc > 0 else 4 # East if dc > 0, else West
                else:
                    action = 0 # Stay
                target.append(action)
            
            target = torch.tensor(target)
            logits = self.model(local_view, goal)
            loss = F.cross_entropy(logits, target)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if i % 1000 == 0:
                print(f"Sandbox Step {i}, Loss: {loss.item():.4f}")
        return self.model

class RLDataset(IterableDataset):
    def __init__(self, buffer: ReplayBuffer, batch_size: int):
        self.buffer = buffer
        self.batch_size = batch_size

    def __iter__(self) -> Iterator:
        while True:
            yield self.buffer.sample(self.batch_size)

class StrategyLightningModule(pl.LightningModule):
    def __init__(self, env: StrategyEnv, replay_buffer: ReplayBuffer, lr: float, collect_steps: int, soldier_model: SoldierAgent, batch_size: int = 64):
        super().__init__()
        self.env = env
        self.buffer = replay_buffer
        self.lr = lr
        self.collect_steps = collect_steps
        self.batch_size = batch_size
        self.model = MARL_Strategy(in_channels=8, stats_dim=28)
        self.soldier_agent = soldier_model # Initially unfreezing for warm-up
        for p in self.soldier_agent.parameters(): 
            p.requires_grad = True
        
        self.obs, self.info = self.env.reset()
        self.total_reward = 0
        self.goal_persistence = 5
        self.register_buffer("current_goal", torch.zeros((1, 16)))

    def _process_obs(self, obs, info, log=True):
        device = self.device
        if isinstance(obs, dict):
            # Check if it's a batch or a single observation
            is_batch = obs["board_state"].ndim == 4
            
            if is_batch:
                boards = obs["board_state"].to(device)
                stats = obs["full_stats"].to(device)
                stats_raw = obs["board_stats"]
                
                # Info is a tuple of masks in training_step: (m_target, m_build, m_fort_mask)
                t_mask, b_mask, f_mask = info
                t_mask = t_mask.to(device)
                b_mask = b_mask.to(device)
                f_mask = f_mask.to(device)
            else:
                boards = torch.as_tensor(obs["board_state"], dtype=torch.float32).unsqueeze(0).to(device)
                # Combine all stats parts: player_res (9) + opponent_res (3) + board_stats (15) + turn (1) = 28
                s_np = np.concatenate([obs["player_resources"], obs["opponent_resources"], obs["board_stats"], [obs["turn_number"]]]).astype(np.float32)
                stats = torch.as_tensor(s_np, dtype=torch.float32).unsqueeze(0).to(device)
                t_mask = torch.as_tensor(info["action_mask"], dtype=torch.bool).unsqueeze(0).to(device)
                b_mask = torch.as_tensor(info["build_mask"], dtype=torch.bool).unsqueeze(0).to(device)
                f_mask = torch.as_tensor(info["fortify_target_mask"], dtype=torch.bool).unsqueeze(0).to(device)
                stats_raw = obs["board_stats"]
        else:
            # This case might be used if obs is a list of dicts
            boards = torch.as_tensor(np.stack([o["board_state"] for o in obs]), dtype=torch.float32).to(device)
            
            stats_list = []
            for o in obs:
                s_np = np.concatenate([o["player_resources"], o["opponent_resources"], o["board_stats"], [o["turn_number"]]]).astype(np.float32)
                stats_list.append(s_np)
            stats = torch.as_tensor(np.stack(stats_list), dtype=torch.float32).to(device)

            t_mask = torch.as_tensor(np.stack([i["action_mask"] for i in info]), dtype=torch.bool).to(device)
            b_mask = torch.as_tensor(np.stack([i["build_mask"] for i in info]), dtype=torch.bool).to(device)
            f_mask = torch.as_tensor(np.stack([i["fortify_target_mask"] for i in info]), dtype=torch.bool).to(device)
            stats_raw = np.stack([o["board_stats"] for o in obs])

        if log and stats_raw is not None:
            stats_raw = torch.as_tensor(stats_raw, dtype=torch.float32)
            if stats_raw.dim() > 1:
                self.log("game_stats/player_soldiers", stats_raw[:, 14].mean())
            else:
                self.log("game_stats/player_soldiers", stats_raw[14])

        return boards, stats, t_mask, b_mask, f_mask

    def on_train_epoch_start(self):
        trajectory_data = []
        start_idx = self.buffer.idx
        for _ in range(self.collect_steps):
            board_t, stats_t, t_mask, b_mask, f_mask = self._process_obs(self.obs, self.info)
            with torch.no_grad():
                res = self.model(board_t, stats_t, target_mask=t_mask, build_mask=b_mask, fortify_target_mask=f_mask)
            
            action = {
                "diplomacy": torch.distributions.Categorical(logits=res["dip"]).sample().item(),
                "economy": [torch.distributions.Categorical(logits=l).sample().item() for l in res["eco"]],
                "target_tile": torch.distributions.Categorical(logits=res["mil"]).sample().item(),
                "soldier_target_tile": torch.distributions.Categorical(logits=res["mil_soldier"]).sample().item(),
                "fortify_tile": torch.distributions.Categorical(logits=res["mil_fortify"]).sample().item()
            }

            # Aggression Mode: If soldier target is an ENEMY, slot 2 = 1.0
            if self.env.current_turn % self.goal_persistence == 0:
                target_idx = action["soldier_target_tile"]
                tr, tc = target_idx // 16, target_idx % 16
                is_enemy = self.obs["board_state"][0, tr, tc] == 2
                self.current_goal = res["goal"].clone()
                self.current_goal[0, 2] = 1.0 if is_enemy else 0.0

            next_obs, reward, terminated, truncated, next_info = self.env.step(action, self.soldier_agent, self.current_goal)
            
            if "soldier_accuracy" in next_info:
                closer, total = next_info["soldier_accuracy"]
                if total > 0:
                    self.log("tactical/soldier_accuracy", closer / total, prog_bar=True)
                    # Strategic Alignment Reward: Bonus for moving closer to target
                    reward += (closer / total) * 0.5

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
            b, s, tm, bm, fm = self._process_obs(last_obs, last_info, log=False)
            res = self.model(b, s, tm, bm, fm)
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
            board_t, stats_t, t_mask, b_mask, f_mask = self._process_obs(self.obs, self.info, log=False)
            with torch.no_grad():
                res = self.model(board_t, stats_t, target_mask=t_mask, build_mask=b_mask, fortify_target_mask=f_mask)
            
            # Update goal from model output
            self.current_goal = res["goal"].clone()

            action = {
                "diplomacy": torch.distributions.Categorical(logits=res["dip"]).sample().item(),
                "economy": [torch.distributions.Categorical(logits=l).sample().item() for l in res["eco"]],
                "target_tile": torch.distributions.Categorical(logits=res["mil"]).sample().item(),
                "soldier_target_tile": torch.distributions.Categorical(logits=res["mil_soldier"]).sample().item(),
                "fortify_tile": torch.distributions.Categorical(logits=res["mil_fortify"]).sample().item()
            }
            next_obs, reward, terminated, truncated, next_info = self.env.step(action, self.soldier_agent, self.current_goal)
            self.buffer.add(self.obs, action, reward, next_obs, terminated, truncated, self.info)
            self.obs, self.info = next_obs, next_info
            if terminated or truncated: self.obs, self.info = self.env.reset()

    def on_fit_start(self): self._prefill_buffer()

    def training_step(self, batch, batch_idx):
        (states, a_dip, a_eco, a_target, a_sol_target, a_fort, rewards, _, t_d, t_c, m_target, m_build, m_fort_mask, pre_adv, pre_returns) = batch
        boards, stats, t_mask, b_mask, f_mask = self._process_obs(states, (m_target, m_build, m_fort_mask))
        res = self.model(boards, stats, target_mask=t_mask, build_mask=b_mask, fortify_target_mask=f_mask)
        
        def stab(l): return torch.clamp(torch.nan_to_num(l, nan=0.0, posinf=20.0, neginf=-20.0), -20, 20)
        res["dip"], res["mil"], res["mil_soldier"], res["mil_fortify"] = stab(res["dip"]), stab(res["mil"]), stab(res["mil_soldier"]), stab(res["mil_fortify"])
        res["eco"] = [stab(l) for l in res["eco"]]
        
        adv = (pre_adv - pre_adv.mean()) / (pre_adv.std() + 1e-8)
        d_dip = torch.distributions.Categorical(logits=res["dip"])
        d_tar = torch.distributions.Categorical(logits=res["mil"])
        d_sol = torch.distributions.Categorical(logits=res["mil_soldier"])
        d_fort = torch.distributions.Categorical(logits=res["mil_fortify"])
        
        entropy = d_dip.entropy().mean() + d_tar.entropy().mean() + d_sol.entropy().mean() + d_fort.entropy().mean()
        lp = d_dip.log_prob(a_dip) + d_tar.log_prob(a_target) + d_sol.log_prob(a_sol_target) + d_fort.log_prob(a_fort)
        for i, l in enumerate(res["eco"]): 
            dist = torch.distributions.Categorical(logits=l)
            lp += dist.log_prob(a_eco[:, i])
            entropy += dist.entropy().mean()

        actor_loss = -(lp * adv).mean() - 0.01 * entropy
        critic_loss = F.huber_loss(res["value"].view(-1), pre_returns.view(-1), delta=1.0)
        total_loss = actor_loss + 0.1 * critic_loss
        self.log("loss/total", total_loss, prog_bar=True)
        return total_loss

    def configure_optimizers(self): return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def train_dataloader(self):
        return DataLoader(RLDataset(self.buffer, self.batch_size), batch_size=None)

def train_agent_lightning():
    torch.set_float32_matmul_precision('medium')
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
