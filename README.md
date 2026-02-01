# Strategy RL: Deep Reinforcement Learning Warfare

A custom **Gymnasium** environment and **PyTorch** implementation of a strategic conquest game. The project features a dual-resource economy, military management, and a grid-based territory system where an RL agent competes against a scripted opponent.

---

## 🎮 Core Features

* **Custom Environment:** A multi-layered `StrategyEnv` built on Gymnasium standards involving diplomacy, economy, and board conquest.
* **Dual-Head Architecture:** Uses a Multi-Head Neural Network to handle separate action branches for **Diplomacy** (Move, Build, Attack) and **Economy** (Gather, Create Units).
* **Grid-Based Expansion:** A neighbor-aware territory mechanic where winning battles allows the player to capture adjacent tiles and eventually the enemy base.
* **Real-time Visualization:** Pygame-based renderer showing territory control (Blue vs. Red), resource counts, and unit buildings.

---

## 🕹️ Game Mechanics

### 1. Resources & Economy
The agent manages four primary resource indices:
-   **Gold:** Used for building construction and soldier maintenance.
-   **Wood:** Primary material used for building construction.
-   **Soldiers:** Required to win battles and expand territory.
-   **Buildings:** Increases passive gold income per turn.

### 2. Action Space
The agent chooses two actions per turn (Multi-Discrete):
-   **Diplomatic Branch:**
    -   `0`: Idle (Wait).
    -   `1`: Build (Consumes wood, increases gold income).
    -   `2`: Attack (Compares army strength vs enemy defense to capture tiles).
-   **Economic Branch:**
    -   `0`: Gather (Increases Wood).
    -   `1`: Create Units (Consumes gold, increases Soldiers).

### 3. Territory & Combat
-   **Expansion:** Territory grows from a starting "Anchor" tile.
-   **Battle Logic:** Success depends on your army size exceeding the opponent's defense (Base Defense + Soldiers).
-   **Victory Condition:** Capturing the enemy's starting base tile ends the episode with a massive reward.

---

## 📂 Project Structure

```text
├── main.py              # Training loop and entry point
├── visualization.py     # Pygame rendering engine and UI
├── environment/
│   ├── strategy_env.py  # Top-level Gymnasium wrapper
│   ├── board_env.py     # Grid logic and territory flood-fill
│   └── player_env.py    # Resource management and combat logic
└── models/
    └── agent.py         # PyTorch Policy/Value neural network
