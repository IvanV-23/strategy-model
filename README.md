# Strategy RL: Deep Reinforcement Learning Warfare

A custom **Gymnasium** environment and **PyTorch** implementation of a strategic conquest game. The project features a dual-resource economy, military management, and a grid-based territory system where an RL agent competes against a scripted opponent.

---

## 🎮 Core Features

* **Custom Environment:** A multi-layered `StrategyEnv` built on Gymnasium standards involving diplomacy, economy, and board conquest.
* **Dual-Head Architecture:** Uses a Multi-Head Neural Network to handle separate action branches for **Diplomacy** (Move, Build, Attack) and **Economy** (Gather, Create Units).
* **Grid-Based Expansion:** A neighbor-aware territory mechanic where winning battles allows the player to capture adjacent tiles and eventually the enemy base.
* **Multi-Agent Learning:** Supports Multi-Agent Reinforcement Learning (MARL) with shared critics.
* **C++ Renderer:** High-performance C++ rendering engine for real-time visualization.
* **Trained Models:** Pre-trained model checkpoints available (trained_model.pth, soldier_agent_trained.pth, marl_strategy_optimized.pth).

---

## 🕹️ Game Mechanics

### 1. Resources & Economy
The agent manages four primary resource indices:
-   **Gold:** Used for building construction and worker maintenance.
-   **Wood:** Primary material used for building construction.
-   **Workers:** Required to win battles and expand territory.
-   **Buildings:** Increases passive gold income per turn.

### 2. Action Space
The agent chooses two actions per turn (Multi-Discrete):
-   **Diplomatic Branch:**
    -   `0`: Idle (Wait).
    -   `1`: Build (Consumes wood, increases gold income).
    -   `2`: Attack (Compares army strength vs enemy defense to capture tiles).
-   **Economic Branch:**
    -   `0`: Gather (Increases Wood).
    -   `1`: Create Units (Consumes gold, increases Workers).

### 3. Territory & Combat
-   **Expansion:** Territory grows from a starting "Anchor" tile.
-   **Battle Logic:** Success depends on your army size exceeding the opponent's defense (Base Defense + Workers).
-   **Victory Condition:** Capturing the enemy's starting base tile ends the episode with a massive reward.

---

## 📂 Project Structure

```text
├── scripts/
│   ├── train.py           # Standard training loop
│   ├── train_marl.py      # Multi-Agent RL training
│   ├── visualize.py       # Python visualization
│   └── visualize_marl.py  # MARL visualization
├── enviroment/
│   ├── strategy_env.py    # Top-level Gymnasium wrapper
│   ├── reward_env.py      # Reward shaping
│   ├── enviroment_blocks/ # Block-based environment
│   ├── enviroment_branches/ # Branched action space
│   ├── buildings/         # Building mechanics
│   └── enviroment_render/ # Python renderer
├── models/
│   ├── actor_critic.py    # Actor-Critic network
│   ├── marl_critic.py     # Multi-Agent critic
│   ├── agents/           # Agent implementations
│   └── heads/            # Policy/value heads
├── cpp_render/           # C++ rendering engine
│   ├── main.cpp          # C++ renderer
│   ├── include/          # Headers
│   └── CMakeLists.txt    # Build config
├── trained_model.pth              # Standard trained model
├── soldier_agent_trained.pth      # Soldier agent model
└── marl_strategy_optimized.pth     # MARL optimized model
```
