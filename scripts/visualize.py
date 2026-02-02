import gymnasium as gym
import torch
import time
import sys
import os
import pygame


# Add the parent directory to sys.path to allow importing local modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from enviroment.strategy_env import StrategyEnv
from models.actor_critic import StrategyActorCritic
from data.replay_buffer import ReplayBuffer

def visualize_agent():
    """
    Loads a trained agent and visualizes it playing in the environment.
    """
    # 1. Environment
    env = gym.make("StrategyProblem-v0", render_mode="human")

    # 2. Replay Buffer
    buffer = ReplayBuffer(1, env.observation_space, env.action_space)

    # 3. Model
    model = StrategyActorCritic(
        state_dim=buffer.state_dim,
        action_dim_diplomacy=env.action_space["diplomacy"].n,
        action_dim_economy=env.action_space["economy"].n,
        action_dim_dist=env.action_space["distribution"].n
    )
    
    model_path = "trained_model.pth"
    if os.path.exists(model_path):
        print(f"Loading trained model from {model_path}")
        # Added weights_only=True to resolve the FutureWarning
        model.load_state_dict(torch.load(model_path, weights_only=True))
    else:
        print(f"Warning: Could not find {model_path}. Using an untrained model.")
    
    model.eval() 

    print("Starting visualization...")
    obs, _ = env.reset()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Convert observation to tensor
        state_tensor = torch.tensor(buffer._flatten_observation(obs), dtype=torch.float32).unsqueeze(0)
        
        # Get actions from model
        with torch.no_grad():
            # FIXED: Unpack 4 values instead of 3
            diplomacy_logits, economy_logits, dist_logits, _ = model(state_tensor)
        
        # Create distributions for all action heads
        prob_diplomacy = torch.distributions.Categorical(logits=diplomacy_logits)
        prob_economy = torch.distributions.Categorical(logits=economy_logits)
        prob_dist = torch.distributions.Categorical(logits=dist_logits)

        # Sample actions
        action_diplomacy = prob_diplomacy.sample().item()
        action_economy = prob_economy.sample().item()
        action_dist = prob_dist.sample().item()
        
        # FIXED: Include the 'distribution' key in the action dictionary
        action = {
            "diplomacy": action_diplomacy, 
            "economy": action_economy,
            "distribution": action_dist
        }

        # Step the environment
        obs, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            print("Episode finished. Resetting...")
            obs, _ = env.reset()
            time.sleep(2) 
            
    env.close()
    print("Visualization finished.")

if __name__ == "__main__":
    visualize_agent()
