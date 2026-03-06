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
from models.marl_critic import MARL_Strategy

def visualize_marl_agent():
    # 1. Environment Setup
    env = StrategyEnv(render_mode="human")

    # 2. Model Initialization
    # MARL model with the same dimensions used in train_marl.py
    model = MARL_Strategy(in_channels=5, stats_dim=27)
    
    # 3. Load Weights
    model_path = "marl_strategy_optimized.pth"
    if not os.path.exists(model_path):
        model_path = "marl_strategy_model.pth"

    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}")
        state_dict = torch.load(model_path, weights_only=True, map_location="cpu")
        # Handle potential 'model.' prefix if saved from Lightning
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=True)
    else:
        print(f"Warning: No model found at '{model_path}'. Visualizing random agent.")
    
    model.eval() 

    obs, info = env.reset()
    running = True
    
    print("Starting MARL visualization loop. Close the Pygame window to stop.")
    while running:
        # Prevent Pygame from hanging
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 4. Prepare inputs
        with torch.no_grad():
            board_tensor = torch.from_numpy(obs["board_state"]).to(torch.float32).unsqueeze(0) / 255.0
            
            # Match the _process_obs logic from train_marl.py
            stats_np = np.concatenate([
                obs["player_resources"],
                obs["opponent_resources"],
                obs["board_stats"],
                [obs["turn_number"]]
            ]).astype(np.float32)
            stats_tensor = torch.from_numpy(stats_np).to(torch.float32).unsqueeze(0) / 1000.0

            # Masking tensors
            b_mask = info.get("build_mask", np.ones(8))
            b_mask_tensor = torch.from_numpy(b_mask).to(torch.bool).unsqueeze(0)
            
            t_mask = info.get("action_mask", np.ones(64))
            t_mask_tensor = torch.from_numpy(t_mask).to(torch.bool).unsqueeze(0)

            # 5. Model Inference
            res = model(
                board_tensor, 
                stats_tensor, 
                target_mask=t_mask_tensor,
                build_mask=b_mask_tensor
            )
            
            eco_l = res["eco"] # (workers, mines, trade, warehouse, crop)
            mil_l = res["mil"] # (dist, target)
            dip_l = res["dip"]

            # 7. Action Selection (Greedy/Argmax)
            action = {
                "diplomacy": torch.argmax(dip_l, dim=1).item(),
                "economy": [
                    torch.argmax(eco_l[0], dim=1).item(),
                    torch.argmax(eco_l[1], dim=1).item(),
                    torch.argmax(eco_l[2], dim=1).item(),
                    torch.argmax(eco_l[3], dim=1).item(),
                    torch.argmax(eco_l[4], dim=1).item()
                ],
                "distribution": torch.argmax(mil_l[0], dim=1).item(),
                "target_tile": torch.argmax(mil_l[1], dim=1).item()
            }

        # 8. Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Slow down for visibility
        time.sleep(0.2) 

        if terminated or truncated:
            print(f"Episode Finished. Resetting...")
            obs, info = env.reset()
            time.sleep(1.0) 
            
    env.close()

if __name__ == "__main__":
    visualize_marl_agent()
