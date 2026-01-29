import gymnasium as gym
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger

import sys
import os

# Add the parent directory to sys.path to allow importing local modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from enviroment.strategy_env import StrategyEnv # Ensure environment is registered
from models.actor_critic import StrategyActorCritic
from data.replay_buffer import ReplayBuffer

def collect_experience(env: gym.Env, model: StrategyActorCritic, buffer: ReplayBuffer, num_steps: int):
    """Collects experience from the environment using the current policy."""
    obs, _ = env.reset() # `obs` is the dictionary observation from the environment

    for _ in range(num_steps):
        # Convert dictionary observation to flattened tensor for model input
        state_tensor = torch.tensor(buffer._flatten_observation(obs), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            diplomacy_logits, economy_logits, _ = model(state_tensor)
        
        prob_diplomacy = torch.distributions.Categorical(logits=diplomacy_logits)
        prob_economy = torch.distributions.Categorical(logits=economy_logits)

        action_diplomacy = prob_diplomacy.sample().item()
        action_economy = prob_economy.sample().item()
        
        action = {"diplomacy": action_diplomacy, "economy": action_economy}

        next_obs, reward, terminated, truncated, _ = env.step(action) # `next_obs` is dictionary
        
        buffer.add(
            state=obs, # Pass the dictionary observation
            action=action, 
            reward=reward, 
            next_state=next_obs, # Pass the dictionary observation
            terminated=terminated, 
            truncated=truncated
        )

        obs = next_obs # Update current observation to next observation
        if terminated or truncated:
            obs, _ = env.reset() # Reset returns dictionary observation



def train_agent():
    # Hyperparameters
    BATCH_SIZE = 64
    BUFFER_CAPACITY = 10000
    COLLECT_STEPS_PER_EPOCH = 200
    TRAIN_EPOCHS = 50
    LEARNING_RATE = 1e-4

    # 1. Environment
    env = gym.make("StrategyProblem-v0")

    # 2. Replay Buffer
    replay_buffer = ReplayBuffer(BUFFER_CAPACITY, env.observation_space, env.action_space)

    # 3. Model
    model = StrategyActorCritic(
        state_dim=replay_buffer.state_dim,
        action_dim_diplomacy=env.action_space["diplomacy"].n,
        action_dim_economy=env.action_space["economy"].n
    )

    # 4. PyTorch Lightning Trainer setup (placeholder for MLFlow)
    # mlflow_logger = MLFlowLogger(experiment_name="StrategyProblem_RL", tracking_uri="file:./logs")
    trainer = pl.Trainer(
        max_epochs=TRAIN_EPOCHS, 
        # logger=mlflow_logger, # Uncomment to enable MLFlow
        log_every_n_steps=1,
        enable_progress_bar=True
    )

    print("Starting initial experience collection...")
    # Initial experience collection to fill the buffer a bit
    collect_experience(env, model, replay_buffer, COLLECT_STEPS_PER_EPOCH * 5)
    print(f"Buffer size after initial collection: {len(replay_buffer)}")

    # Custom training loop for RL
    for epoch in range(TRAIN_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{TRAIN_EPOCHS} ---")
        # Collect more experience
        collect_experience(env, model, replay_buffer, COLLECT_STEPS_PER_EPOCH)
        
        # Sample from buffer and train
        if len(replay_buffer) >= BATCH_SIZE:
            # For simplicity, we'll manually get a batch and pass it to training_step
            # In a full PL integration, you'd use a DataLoader
            states, actions_diplomacy, actions_economy, rewards, _, _, _ = replay_buffer.sample(BATCH_SIZE)
            
            # Placeholder for calculating advantages and returns for PPO
            # In a real PPO, this would involve value estimates and GAE
            advantages = torch.randn(BATCH_SIZE) # Random for now
            returns = torch.randn(BATCH_SIZE) # Random for now
            
            batch = (states, actions_diplomacy, actions_economy, advantages, returns)
            
            # Manually call training_step (for demonstration; normally Trainer does this)
            # This part needs proper integration with LightningDataModule or manual optimization
            
            # To properly train with Lightning Trainer, we need a DataLoader.
            # For now, we'll just demonstrate calling training_step if we were doing manual optimization.
            # A full PPO implementation would involve more sophisticated data loading and `on_train_epoch_end` hooks.
            
            # Dummy training step execution for conceptual demonstration
            model.optimizer = model.configure_optimizers() # Re-configure optimizer if not done by trainer
            model.optimizer.zero_grad()
            loss = model.training_step(batch, 0)
            if loss is not None:
                loss.backward()
                model.optimizer.step()
                print(f"Epoch {epoch+1} - Loss: {loss.item():.4f}")
            
        else:
            print("Replay buffer not yet full enough for training batch.")

    env.close()
    print("Training finished.")

    # --- Save the trained model ---
    model_path = "trained_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_agent()
