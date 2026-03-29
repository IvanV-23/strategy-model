# Standalone C++ Game with AI Inference

This directory contains the complete standalone game that runs with AI models directly in C++ using LibTorch.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    strategy_game.exe                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐│
│  │  GameState  │  │ TurnManager │  │  InferenceEngine    ││
│  │  (16x16)    │  │  (Rules)    │  │  (LibTorch .pt)     ││
│  └─────────────┘  └─────────────┘  └─────────────────────┘│
│                                                    │        │
│  ┌──────────────────────────────────────────────┐ │        │
│  │              GameRenderer (SDL2)             │ │        │
│  │  ┌──────┐ ┌───────┐ ┌────────┐ ┌─────────┐  │ │        │
│  │  │Tile  │ │Building│ │Unit    │ │Overlay  │  │ │        │
│  │  │Pass  │ │Pass   │ │Pass    │ │(UI/Anim)│  │ │        │
│  │  └──────┘ └───────┘ └────────┘ └─────────┘  │ │        │
│  └──────────────────────────────────────────────┘ │        │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ models/manager.pt   │  (TorchScript)
│ models/soldier.pt   │  (TorchScript)
└─────────────────┘
```

## Files

- `src/main_standalone.cpp` - Standalone game loop with AI
- `src/inference_engine.cpp` - LibTorch inference wrapper
- `include/inference_engine.hpp` - Inference engine interface
- `models/` - Directory for exported TorchScript models

## Build Instructions

### Prerequisites (Windows)

1. **Visual Studio 2022** with C++ desktop development
2. **vcpkg** for dependency management
3. **LibTorch** (via vcpkg or direct download)
4. **SDL2** and **SDL2_ttf** (via vcpkg)

### Step 1: Install vcpkg and dependencies

```powershell
# Clone vcpkg
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat

# Install dependencies
vcpkg install sdl2:x64-windows
vcpkg install sdl2-ttf:x64-windows
vcpkg install nlohmann-json:x64-windows
vcpkg install torch-cpp:x64-windows  # or torch:x64-windows
```

### Step 2: Build the game

```powershell
cd cpp_render
mkdir build
cd build

# Configure with LibTorch support
cmake .. -G "Visual Studio 17 2022" `
    -DCMAKE_TOOLCHAIN_FILE="C:/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake" `
    -DVCPKG_MANIFEST_MODE=ON

# Build the AI-enabled game
cmake --build . --config Release --target strategy_game
```

### Step 3: Export AI Models

```powershell
# From project root (Python environment with PyTorch)
python scripts/export_models.py
```

This creates:
- `cpp_render/models/manager.pt` - Strategy network
- `cpp_render/models/soldier.pt` - Soldier tactical network

### Step 4: Run

```powershell
# From cpp_render/build/Release
./strategy_game.exe

# Or from project root
./cpp_render/build/Release/strategy_game.exe
```

## Controls

| Key | Action |
|-----|--------|
| Arrow Keys | Pan camera |
| +/- | Zoom in/out |
| Space | Pause/Resume |
| ESC | Quit |

## Model Export

After training, export models for C++:

```python
from scripts.export_models import export_for_cpp
export_for_cpp("cpp_render/models")
```

## Fallback

If LibTorch is not available, the game builds without AI inference (`strategy_renderer`). If models aren't loaded, rule-based actions are used instead.

## Troubleshooting

### "LibTorch not found"
- Set `CMAKE_PREFIX_PATH` to your LibTorch installation
- Or install via vcpkg: `vcpkg install torch-cpp`

### "models/manager.pt not found"
- Run `python scripts/export_models.py` first
- Models must be in `cpp_render/models/` relative to the executable

### "SDL2 not found"
- Install via vcpkg: `vcpkg install sdl2 sdl2-ttf`
