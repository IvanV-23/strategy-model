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
    eco_dims = env.action_space["economy"].nvec 

    model = StrategyActorCritic(
        action_dim_dip=env.action_space["diplomacy"].n,
        action_dim_eco=eco_dims[0],        # Usually 11 if 0-10 soldiers
        action_dim_dist=env.action_space["distribution"].n,
        action_dim_target=env.action_space["target_tile"].n,
        board_size=64 
    )
    
    # 3. Load Weights
    # Look for the Lightning checkpoint or the exported .pth
    model_path = "trained_model.pth"
    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}")
        # Note: If you saved via PyTorch Lightning, the keys might have 'model.' prefix
        state_dict = torch.load(model_path, weights_only=True, map_location="cpu")
        model.load_state_dict(state_dict)
    else:
        print("Warning: No model found at 'trained_model.pth'. Visualizing random agent.")
    
    model.eval() 

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
            b_stats = torch.as_tensor(obs["board_stats"], dtype=torch.float32).flatten()

            # Global Stats: Combine resources and turn into [1, 9]
            p_res = torch.as_tensor(obs["player_resources"], dtype=torch.float32).flatten()
            o_res = torch.as_tensor(obs["opponent_resources"], dtype=torch.float32).flatten()
            turn = torch.as_tensor([obs["turn_number"]], dtype=torch.float32)
            stats_tensor = torch.cat([p_res, o_res, turn, b_stats]).unsqueeze(0)

            # 5. Model Inference
            # eco_logits is a tuple: (sol_logits, mine_logits)
            dip_l, (sol_l, mine_l), dist_l, tar_l, _ = model(board_tensor, stats_tensor)
        
            # 6. Apply Action Masking
            mask = info.get("action_mask", None)
            if mask is not None:
                mask_tensor = torch.as_tensor(mask, dtype=torch.bool)
                # Ensure mask is applied to the batch dimension [1, 64]
                tar_l = tar_l.masked_fill(~mask_tensor.unsqueeze(0), -1e10)

            # 7. Action Selection (Greedy/Argmax)
            # Economy is handled by taking the max of both heads and combining into a list
            action = {
                "diplomacy": torch.argmax(dip_l, dim=1).item(),
                "economy": np.array([
                    torch.argmax(sol_l, dim=1).item(),
                    torch.argmax(mine_l, dim=1).item()
                ], dtype=np.int64),
                "distribution": torch.argmax(dist_l, dim=1).item(),
                "target_tile": torch.argmax(tar_l, dim=1).item()
            }

        # 8. Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Slow down the visualization so humans can actually see the moves
        time.sleep(0.1) 

        if terminated or truncated:
            print(f"Episode Finished. Reward: {reward:.2f}. Resetting...")
            obs, info = env.reset()
            time.sleep(1.0) 
            
    env.close()

if __name__ == "__main__":
    visualize_agent()

