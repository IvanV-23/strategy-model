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
    """
    IterableDataset for Reinforcement Learning.
    Samples from a ReplayBuffer.
    """
    def __init__(self, buffer: ReplayBuffer, batch_size: int):
        super().__init__()
        self.buffer = buffer
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[Tuple]:
        """
        Yields a batch of experiences from the replay buffer.
        """
        while True:
            yield self.buffer.sample(self.batch_size)

class StrategyLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for training the StrategyActorCritic model.
    """
    def __init__(self, env: gym.Env, buffer: ReplayBuffer, lr: float, collect_steps: int, gamma: float = 0.99, batch_size: int = 64):
        super().__init__()
        self.env = env
        self.buffer = buffer
        self.lr = lr
        self.collect_steps = collect_steps
        self.gamma = gamma
        self.batch_size = batch_size

        self.model = StrategyActorCritic(
            state_dim=self.buffer.state_dim,
            action_dim_diplomacy=self.env.action_space["diplomacy"].n,
            action_dim_economy=self.env.action_space["economy"].n,
            gamma=self.gamma
        )
        
        self.obs, _ = self.env.reset()
        self.total_reward = 0
        self.episode_count = 0

    def on_train_epoch_start(self):
        """
        Called at the beginning of each training epoch.
        Collects new experience and adds it to the replay buffer.
        """
        for _ in range(self.collect_steps):
            state_tensor = torch.tensor(self.buffer._flatten_observation(self.obs), dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                diplomacy_logits, economy_logits, _ = self.model(state_tensor)
            
            prob_diplomacy = torch.distributions.Categorical(logits=diplomacy_logits)
            prob_economy = torch.distributions.Categorical(logits=economy_logits)

            action_diplomacy = prob_diplomacy.sample().item()
            action_economy = prob_economy.sample().item()
            
            action = {"diplomacy": action_diplomacy, "economy": action_economy}

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            self.total_reward += reward
            
            self.buffer.add(
                state=self.obs,
                action=action, 
                reward=reward, 
                next_state=next_obs,
                terminated=terminated, 
                truncated=truncated
            )

            self.obs = next_obs
            if terminated or truncated:
                self.obs, _ = self.env.reset()
                self.log("episode/total_reward", self.total_reward, on_step=False, on_epoch=True)
                self.total_reward = 0
                self.episode_count += 1

    def training_step(self, batch, batch_idx):
        # 1. Unpack batch
        states, actions_dip, actions_eco, rewards, next_states, terminals, _ = batch

        # 2. Forward pass (calling the underlying nn.Module)
        # Note: Use self.model(...) if you wrapped your ActorCritic inside this class
        dip_logits, eco_logits, current_values = self.model(states) 
        current_values = current_values.squeeze(-1)
        
        # 3. Scale rewards & GAE logic
        scaled_rewards = rewards / 100.0
        with torch.no_grad():
            _, _, next_values = self.model(next_states)
            next_values = next_values.squeeze(-1)
            returns = scaled_rewards + (self.gamma * next_values * (~terminals))
            advantages = returns - current_values

        # 4. Calculate Losses
        dist_dip = torch.distributions.Categorical(logits=dip_logits)
        dist_eco = torch.distributions.Categorical(logits=eco_logits)
        
        actor_loss = -(dist_dip.log_prob(actions_dip) * advantages.detach()).mean() - \
                    (dist_eco.log_prob(actions_eco) * advantages.detach()).mean()
        
        critic_loss = torch.nn.functional.mse_loss(current_values, returns)
        entropy_loss = dist_dip.entropy().mean() + dist_eco.entropy().mean()

        total_loss = actor_loss + (0.5 * critic_loss) - (0.01 * entropy_loss)

        # 5. LOGGING (This will now work because it's in the top-level module!)
        self.log("train/total_loss", total_loss, prog_bar=True)
        self.log("train/reward", rewards.mean(), prog_bar=True)
        
        return total_loss

    def configure_optimizers(self):
        return self.model.configure_optimizers()

    def train_dataloader(self):
        return DataLoader(
            RLDataset(self.buffer, self.batch_size),
            batch_size=1, # The dataset already returns a batch
        )

def train_agent_lightning():
    # Hyperparameters
    BATCH_SIZE = 64
    BUFFER_CAPACITY = 1000
    COLLECT_STEPS_PER_EPOCH = 200
    TRAIN_EPOCHS = 100
    LEARNING_RATE = 1e-4

    print(torch.cuda.is_available())

    torch.set_float32_matmul_precision('high')
    
    # 1. Environment and Buffer
    env = gym.make("StrategyProblem-v0")
    replay_buffer = ReplayBuffer(BUFFER_CAPACITY, env.observation_space, env.action_space)

    # 2. Lightning Module
    lightning_module = StrategyLightningModule(
        env=env,
        buffer=replay_buffer,
        lr=LEARNING_RATE,
        collect_steps=COLLECT_STEPS_PER_EPOCH,
        batch_size=BATCH_SIZE
    )
    # 2.1 Early Stopping Callback
    early_stop_callback = EarlyStopping(
    monitor="train/reward",   # Must match exactly what you put in self.log()
    min_delta=1.0,            # Minimum improvement to be considered "better"
    patience=20,              # How many epochs/checks to wait before giving up
    verbose=True,
    mode="max"                # We want to MAXIMIZE reward
)
    
    # 3. MLFlow Logger
    mlflow_logger = MLFlowLogger(experiment_name="StrategyProblem_RL", tracking_uri="file:./ml-runs")

    # 4. Trainer
    trainer = pl.Trainer(
        max_epochs=TRAIN_EPOCHS,
        limit_train_batches=100,
        callbacks=[early_stop_callback],
        gradient_clip_val=0.5,
        logger=mlflow_logger,
        log_every_n_steps=1,
        enable_progress_bar=True,
    )

    # 5. Initial experience collection
    print("Starting initial experience collection...")
    obs, _ = env.reset()
    for _ in range(BUFFER_CAPACITY // 10): # Fill 10% of buffer
        state_tensor = torch.tensor(replay_buffer._flatten_observation(obs), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            diplomacy_logits, economy_logits, _ = lightning_module.model(state_tensor)
        
        prob_diplomacy = torch.distributions.Categorical(logits=diplomacy_logits)
        prob_economy = torch.distributions.Categorical(logits=economy_logits)

        action_diplomacy = prob_diplomacy.sample().item()
        action_economy = prob_economy.sample().item()
        action = {"diplomacy": action_diplomacy, "economy": action_economy}

        next_obs, reward, terminated, truncated, _ = env.step(action)
        
        replay_buffer.add(
            state=obs,
            action=action, 
            reward=reward, 
            next_state=next_obs,
            terminated=terminated, 
            truncated=truncated
        )

        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
    print(f"Buffer size after initial collection: {len(replay_buffer)}")

    # 6. Train the model
    trainer.fit(lightning_module)

    # 7. Save the model
    model_path = "trained_model.pth"
    torch.save(lightning_module.model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_agent_lightning()
