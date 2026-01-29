# Project Context: StrategyProblem

## Project Description
This project aims to develop a Multi-Agent Reinforcement Learning (MARL) agent capable of playing a strategic game. The agent must manage two distinct decision-making branches:

Diplomacy: [Trade, Pass, Attack]

Economy: [Invest in Buildings, Create Units, Idle]

## Tech Stack
- **Framework:** PyTorch Lightning
- **Reinforcement Learning (Enviroment):** Gymnasium
- **Experiment Tracking:** MLFlow (logging metrics, parameters, and model versions).
- **Domain:** Reinforcement Learning (Actor-Critic)
- **Architecture:** Multi-Head Actor-Critic (PPO preferred).

## Coding Standards
- Use type hints in all Python functions.
- Prefer modular architectures (separate model, trainer, and wrappers).
- Always include logging for loss and rewards.

## Project Structure
- `models/`:  Actor-Critic architectures & LightningModules
- `data/`: Replay buffers and local checkpointing
- `enviroment/`: Gymnasium environment logic & observation wrappers
- `scripts/`: Training and Evaluation entry points
- `logs/`: MLFlow local tracking URI