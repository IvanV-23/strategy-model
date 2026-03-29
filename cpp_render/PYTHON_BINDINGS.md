# Phase 7: pybind11 Python Bindings

## Overview

The C++ game engine is now exposed to Python via pybind11, eliminating the TCP socket protocol.

## Files

- `src/python_bindings.cpp` - pybind11 module exposing all C++ engine classes
- `enviroment/native_strategy_env.py` - Gymnasium environment wrapper around native module
- `test_bindings.py` - Test script to verify bindings work
- `build_python.sh` - Build script for the Python module

## Exposed Classes

| C++ Class | Python Access |
|-----------|---------------|
| `GameState` | `engine.GameState` |
| `TurnManager` | `engine.TurnManager` |
| `BoardFactory` | `engine.BoardFactory` |
| `TileStatus` | `engine.TileStatus` |
| `Tile` | `engine.Tile` |
| `Resources` | `engine.Resources` |
| `ActionBundle` | `engine.ActionBundle` |
| `BuildAction` | `engine.BuildAction` |
| `ShotEvent` | `engine.ShotEvent` |

## Building

```bash
cd cpp_render/build
cmake .. -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build . --config Release
cp strategy_engine*.so ../../enviroment/
```

Or use the build script:
```bash
cd cpp_render
chmod +x build_python.sh
./build_python.sh
cp build/python/strategy_engine*.so ../enviroment/
```

## Usage

```python
from enviroment.native_strategy_env import NativeStrategyEnv

env = NativeStrategyEnv(render_mode="human")
obs, info = env.reset()

action = {
    "diplomacy": 0,
    "economy": [0, 0, 0, 0, 0, 0],
    "target_tile": 0,
    "soldier_target_tile": 0,
    "fortify_tile": 0,
}

obs, reward, terminated, truncated, info = env.step(action)
env.close()
```

## Direct Engine Usage

```python
import strategy_engine as engine

# Create game state
state = engine.BoardFactory.create_default(16, 16, 0, 0, 15, 15)

# Create action bundles
p1 = engine.ActionBundle()
p1.diplomacy = 0
p1.economy = [1, 0, 0, 0, 0, 0]

p2 = engine.ActionBundle()
p2.diplomacy = 1

# Step the game
manager = engine.TurnManager()
manager.step(state, p1, p2)

# Check terminal
is_terminal = manager.is_terminal(state)

# Compute reward
reward = manager.compute_reward(prev_state, state, player_id=1)
```

## Rendering

The C++ binary (`strategy_renderer`) can still run standalone with its own SDL event loop for visual debugging. The pybind11 module provides `ShotEvent` data for Python-side rendering.
