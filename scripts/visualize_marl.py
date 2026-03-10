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
from models.marl_critic import MARL_Strategy, SoldierAgent

def visualize_marl_agent():
    # 1. Environment Setup
    env = StrategyEnv(render_mode="human")

    # 2. Model Initialization
    model = MARL_Strategy(in_channels=8, stats_dim=28)
    soldier_model = SoldierAgent(in_channels=8, goal_dim=16) # Tactical Agent
    
    # 3. Load Weights
    model_path = "marl_strategy_optimized.pth"
    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}")
        state_dict = torch.load(model_path, weights_only=True, map_location="cpu")
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'): new_state_dict[k[6:]] = v
            else: new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=True)
    
    model.eval()
    soldier_model.eval()

    obs, info = env.reset()
    current_goal = None
    running = True
    
    use_cpp = os.getenv("USE_CPP_RENDER") == "1"
    print("Starting HRL visualization loop.")

    while running:
        if not use_cpp:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False

        with torch.no_grad():
            board_t = torch.from_numpy(obs["board_state"]).to(torch.float32).unsqueeze(0) / 255.0
            stats_np = np.concatenate([obs["player_resources"], obs["opponent_resources"], obs["board_stats"], [obs["turn_number"]]]).astype(np.float32)
            stats_t = torch.from_numpy(stats_np).to(torch.float32).unsqueeze(0) / 1000.0

            b_mask = info.get("build_mask", np.ones(8))
            b_mask_t = torch.from_numpy(b_mask).to(torch.bool).unsqueeze(0)
            t_mask_t = torch.from_numpy(info.get("action_mask", np.ones(256))).to(torch.bool).unsqueeze(0)

            res = model(board_t, stats_t, target_mask=t_mask_t, build_mask=b_mask_t)
            
            # HRL: Update goal every 10 turns
            if env.current_turn % 10 == 0:
                current_goal = res["goal"]

            eco_l = res["eco"]
            action = {
                "diplomacy": torch.argmax(res["dip"], dim=1).item(),
                "economy": [torch.argmax(l, dim=1).item() for l in eco_l],
                "target_tile": torch.argmax(res["mil"], dim=1).item()
            }

        # 8. Step environment with HRL parameters
        obs, reward, terminated, truncated, info = env.step(action, soldier_model, current_goal)

        time.sleep(0.2) 
        if terminated or truncated:
            print(f"Episode Finished. Resetting...")
            obs, info = env.reset()
            time.sleep(1.0) 
            
    env.close()

if __name__ == "__main__":
    visualize_marl_agent()
