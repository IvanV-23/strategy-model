import gymnasium as gym
import torch
import time
import sys
import os
import pygame
import numpy as np

# Add the parent directory to sys.path to allow importing local modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from enviroment.strategy_env import StrategyEnv
from models.actor_critic import StrategyActorCritic

def visualize_agent():
    # 1. Environment Setup
    env = gym.make("StrategyProblem-v0", render_mode="human")

    # 2. Model Initialization
    model = StrategyActorCritic(
        action_dim_dip=env.action_space["diplomacy"].n,
        action_dim_eco=env.action_space["economy"].n,
        action_dim_dist=env.action_space["distribution"].n,
        action_dim_target=env.action_space["target_tile"].n,
        board_size=64 
    )
    
    # 3. Load Weights
    model_path = "trained_model.pth"
    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}")
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location="cpu"))
    else:
        print("Warning: No model found. Visualizing random agent with masking.")
    
    model.eval() 

    # Initial reset to get first observation and info
    obs, info = env.reset()
    running = True
    
    print("Starting loop. Close the Pygame window to stop.")
    while running:
        # Prevent Pygame from hanging
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 4. Prepare inputs
        with torch.no_grad():
            # Board State: [1, 4, 8, 8]
            board_tensor = torch.as_tensor(obs["board_state"], dtype=torch.float32).unsqueeze(0)
            
            # Global Stats: [1, 9] (Adjusted to match your model's expected input)
            p_res = torch.as_tensor(obs["player_resources"], dtype=torch.float32).flatten()
            o_res = torch.as_tensor(obs["opponent_resources"], dtype=torch.float32).flatten()
            turn = torch.as_tensor(obs["turn_number"], dtype=torch.float32).reshape(1)
            stats_tensor = torch.cat([p_res, o_res, turn]).unsqueeze(0)

            # 5. Model Inference
            dip_l, eco_l, dist_l, tar_l, _ = model(board_tensor, stats_tensor)
        
            # 6. Apply Action Masking
            # We use a very large negative number so these actions aren't picked by argmax
            mask = info.get("action_mask", None)
            if mask is not None:
                mask_tensor = torch.as_tensor(mask, dtype=torch.bool)
                tar_l[0, ~mask_tensor] = -1e10  # Mask out invalid tiles

            # 7. Action Selection (Greedy/Argmax for evaluation)
            action = {
                "diplomacy": torch.argmax(dip_l, dim=1).item(),
                "economy": torch.argmax(eco_l, dim=1).item(),
                "distribution": torch.argmax(dist_l, dim=1).item(),
                "target_tile": torch.argmax(tar_l, dim=1).item()
            }

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(f"Episode Finished. Reward: {reward:.2f}. Resetting...")
            obs, info = env.reset()
            time.sleep(1) # Pause to let the user see the final state
            
    env.close()

if __name__ == "__main__":
    visualize_agent()