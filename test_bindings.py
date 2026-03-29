#!/usr/bin/env python3
"""Test script to verify pybind11 bindings work correctly."""

import sys

try:
    import strategy_engine as engine
    print("SUCCESS: strategy_engine module loaded")
except ImportError as e:
    print(f"FAIL: Could not import strategy_engine: {e}")
    print("\nBuild instructions:")
    print("  cd cpp_render/build")
    print("  cmake .. -DCMAKE_BUILD_TYPE=Release")
    print("  cmake --build . --config Release")
    print("  cp strategy_engine*.so <project_root>/enviroment/")
    sys.exit(1)

def test_board_factory():
    print("\n=== Testing BoardFactory ===")
    state = engine.BoardFactory.create_default(16, 16, 0, 0, 15, 15)
    assert len(state.board) == 16
    assert len(state.board[0]) == 16
    print(f"Board created: {len(state.board)}x{len(state.board[0])}")
    print(f"Turn: {state.turn}")
    print(f"Player resources: gold={state.p_res.gold}, wood={state.p_res.wood}")
    print("PASS: BoardFactory.create_default")

def test_game_state():
    print("\n=== Testing GameState ===")
    state = engine.GameState()
    state.turn = 5
    state.p_res.gold = 100.0
    state.p_res.wood = 50.0
    assert state.turn == 5
    assert state.p_res.gold == 100.0
    print("PASS: GameState construction and field access")

def test_action_bundle():
    print("\n=== Testing ActionBundle ===")
    bundle = engine.ActionBundle()
    bundle.diplomacy = 1
    bundle.economy = [1, 2, 3, 0, 1, 0]
    bundle.target_tile = 42
    bundle.soldier_target = 100
    bundle.fortify_tile = 7
    assert bundle.diplomacy == 1
    assert len(bundle.economy) == 6
    print(f"Diplomacy: {bundle.diplomacy}")
    print(f"Economy: {bundle.economy}")
    print("PASS: ActionBundle")

def test_turn_manager():
    print("\n=== Testing TurnManager ===")
    manager = engine.TurnManager()
    state = engine.BoardFactory.create_default(16, 16, 0, 0, 15, 15)
    
    p1 = engine.ActionBundle()
    p1.diplomacy = 0
    p1.economy = [0, 0, 0, 0, 0, 0]
    
    p2 = engine.ActionBundle()
    p2.diplomacy = 1
    p2.economy = [0, 0, 0, 0, 0, 0]
    
    prev = engine.GameState()
    prev.turn = 0
    
    manager.step(state, p1, p2)
    assert state.turn == 1
    
    is_term = manager.is_terminal(state)
    print(f"Terminal: {is_term}")
    
    reward = manager.compute_reward(prev, state, 1)
    print(f"Reward: {reward}")
    print("PASS: TurnManager.step, is_terminal, compute_reward")

def test_build_action():
    print("\n=== Testing BuildAction enum ===")
    cost = engine.ResourceSystem.get_cost(engine.BuildAction.Sawmill)
    print(f"Sawmill cost: gold={cost.gold}, wood={cost.wood}")
    assert cost.gold == 100.0
    assert cost.wood == 50.0
    print("PASS: BuildAction enum and get_cost")

def test_tile_status():
    print("\n=== Testing TileStatus enum ===")
    assert engine.TileStatus.Base == engine.TileStatus.Base
    assert engine.TileStatus.Sawmill == engine.TileStatus.Sawmill
    print(f"TileStatus.Base: {engine.TileStatus.Base}")
    print(f"TileStatus.Sawmill: {engine.TileStatus.Sawmill}")
    print("PASS: TileStatus enum")

if __name__ == "__main__":
    print("=" * 50)
    print("Testing pybind11 strategy_engine bindings")
    print("=" * 50)
    
    test_board_factory()
    test_game_state()
    test_action_bundle()
    test_turn_manager()
    test_build_action()
    test_tile_status()
    
    print("\n" + "=" * 50)
    print("ALL TESTS PASSED!")
    print("=" * 50)
