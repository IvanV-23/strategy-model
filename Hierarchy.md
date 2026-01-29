strategyproblem/
│
├── data/                  # Saved models and training logs
├── env/
│   ├── __init__.py
│   └── game_env.py        # Your custom Game Logic (Gymnasium interface)
├── models/
│   ├── __init__.py
│   ├── actor_critic.py    # The Neural Network architecture (The "Brain")
│   └── lightning_agent.py # The LightningModule (Training logic)
├── config.yaml            # Hyperparameters (learning rate, batch size)
├── train.py               # The entry point to start training
└── evaluate.py            # Script to watch your agent play after training