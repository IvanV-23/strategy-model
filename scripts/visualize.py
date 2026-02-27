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
        model.load_state_dict(state_dict, strict=False)
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
            board_tensor = torch.as_tensor(obs["board_state"], dtype=torch.float32).unsqueeze(0)
            
            # --- UPDATE: Verify your stats concatenation matches training_step ---
            p_res = torch.as_tensor(obs["player_resources"], dtype=torch.float32).flatten()
            o_res = torch.as_tensor(obs["opponent_resources"], dtype=torch.float32).flatten()
            m_stats = torch.as_tensor(obs["board_stats"], dtype=torch.float32).flatten()
            turn = torch.as_tensor([obs["turn_number"]], dtype=torch.float32)
            
            # Ensure this order matches your _process_obs exactly!
            stats_tensor = torch.cat([p_res, o_res, m_stats, turn]).unsqueeze(0)

            # --- UPDATE: Handle Build Mask ---
            b_mask = info.get("build_mask", np.ones(8)) # Size 8 now
            b_mask_tensor = torch.as_tensor(b_mask, dtype=torch.bool).unsqueeze(0)

            # 5. Model Inference (Pass build_mask to the actor for internal masking)
            dip_l, eco_logits, dist_l, tar_l, _ = model(
                board_tensor, 
                stats_tensor, 
                build_mask=b_mask_tensor
            )
            
            # UPDATE: Unpack 4 components
            sol_l, mine_l, trade_l, wh_l = eco_logits 
        
            # 6. Apply Target Masking
            t_mask = info.get("action_mask", None)
            if t_mask is not None:
                t_mask_tensor = torch.as_tensor(t_mask, dtype=torch.bool)
                tar_l = tar_l.masked_fill(~t_mask_tensor.unsqueeze(0), -1e10)

            # 7. Action Selection (Greedy/Argmax)
            action = {
                "diplomacy": torch.argmax(dip_l, dim=1).item(),
                "economy": np.array([
                    torch.argmax(sol_l, dim=1).item(),
                    torch.argmax(mine_l, dim=1).item(),
                    torch.argmax(trade_l, dim=1).item(),
                    torch.argmax(wh_l, dim=1).item() # NEW: Warehouse action
                ], dtype=np.int64),
                "distribution": torch.argmax(dist_l, dim=1).item(),
                "target_tile": torch.argmax(tar_l, dim=1).item()
            }

        # 8. Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        time.sleep(0.1) 

        if terminated or truncated:
            print(f"Episode Finished. Resetting...")
            obs, info = env.reset()
            time.sleep(1.0) 
            
    env.close()

if __name__ == "__main__":
    visualize_agent()

