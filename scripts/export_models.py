#!/usr/bin/env python3
"""
Export trained PyTorch models to TorchScript for C++ inference.
Run this after training to generate .pt files for the C++ engine.

Usage:
    python scripts/export_models.py
"""

import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.marl_critic import MARL_Strategy, SoldierAgent


def export_for_cpp(output_dir="cpp_render/models"):
    os.makedirs(output_dir, exist_ok=True)
    
    manager_path = "marl_strategy_optimized.pth"
    soldier_path = "soldier_agent_trained.pth"
    
    if not os.path.exists(manager_path):
        print(f"ERROR: Manager weights not found: {manager_path}")
        print("Train the model first with: python scripts/train_marl.py")
        return False
        
    if not os.path.exists(soldier_path):
        print(f"WARNING: Soldier weights not found: {soldier_path}")
        print("Soldier agent will use random actions.")
    
    print("=" * 60)
    print("Exporting models to TorchScript for C++ inference")
    print("=" * 60)
    
    # --- Manager Model ---
    print("\n[1/2] Exporting Manager Model (MARL_Strategy)...")
    manager = MARL_Strategy(in_channels=8, stats_dim=28)
    manager.load_state_dict(torch.load(manager_path, weights_only=True, map_location="cpu"))
    manager.eval()
    
    board_ex = torch.zeros(1, 8, 16, 16)
    stats_ex = torch.zeros(1, 28)
    action_mask_ex = torch.ones(1, 256, dtype=torch.bool)
    build_mask_ex = torch.ones(1, 8, dtype=torch.bool)
    fortify_mask_ex = torch.ones(1, 256, dtype=torch.bool)
    
    try:
        print("  Attempting torch.jit.script (supports dynamic control flow)...")
        scripted_manager = torch.jit.script(manager)
    except Exception as e:
        print(f"  script failed: {e}")
        print("  Falling back to torch.jit.trace...")
        scripted_manager = torch.jit.trace(
            manager,
            example_inputs=(board_ex, stats_ex, action_mask_ex, build_mask_ex, fortify_mask_ex)
        )
    
    manager_out_path = os.path.join(output_dir, "manager.pt")
    scripted_manager.save(manager_out_path)
    print(f"  Saved: {manager_out_path}")
    
    # --- Soldier Model ---
    print("\n[2/2] Exporting Soldier Model (SoldierAgent)...")
    soldier = SoldierAgent(in_channels=8, goal_dim=16)
    
    if os.path.exists(soldier_path):
        soldier.load_state_dict(torch.load(soldier_path, weights_only=True, map_location="cpu"))
        print(f"  Loaded weights from: {soldier_path}")
    else:
        print(f"  WARNING: No weights found, using random initialization")
    
    soldier.eval()
    
    board_s = torch.zeros(1, 8, 16, 16)
    goal_s = torch.zeros(1, 16)
    mask_s = torch.ones(1, 256, dtype=torch.bool)
    
    try:
        print("  Attempting torch.jit.script...")
        scripted_soldier = torch.jit.script(soldier)
    except Exception as e:
        print(f"  script failed: {e}")
        print("  Falling back to torch.jit.trace...")
        scripted_soldier = torch.jit.trace(soldier, example_inputs=(board_s, goal_s, mask_s))
    
    soldier_out_path = os.path.join(output_dir, "soldier.pt")
    scripted_soldier.save(soldier_out_path)
    print(f"  Saved: {soldier_out_path}")
    
    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)
    print(f"\nModels exported to: {output_dir}/")
    print("  - manager.pt: Strategy/Policy network")
    print("  - soldier.pt: Soldier tactical network")
    print("\nRebuild C++ with LibTorch to use these models.")
    
    return True


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "cpp_render/models"
    success = export_for_cpp(output_dir)
    sys.exit(0 if success else 1)
