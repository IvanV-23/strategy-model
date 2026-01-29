import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any

class StrategyEnv(gym.Env):
    """
    A Multi-Agent Reinforcement Learning environment for a strategic game.
    The agent manages two distinct decision-making branches: Diplomacy and Economy.
    """
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self):
        super().__init__()
        # Define the action space for Diplomacy and Economy
        # Diplomacy: 0: Trade, 1: Pass, 2: Attack
        # Economy: 0: Invest in Buildings, 1: Create Units, 2: Idle
        self.diplomacy_action_space = spaces.Discrete(3)
        self.economy_action_space = spaces.Discrete(3)

        # Combined action space (Tuple of discrete actions)
        self.action_space = spaces.Dict({
            "diplomacy": self.diplomacy_action_space,
            "economy": self.economy_action_space,
        })

        # Define the observation space
        # For now, a simple example: player resources (e.g., gold, wood) and opponent strength
        # This will need to be significantly expanded for a real game
        self.observation_space = spaces.Dict({
            "player_resources": spaces.Box(low=0, high=1000, shape=(2,), dtype=np.int32), # e.g., [gold, wood]
            "opponent_strength": spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32),
            "turn_number": spaces.Discrete(100), # Max 100 turns
        })

        self.current_turn = 0
        self.player_resources = np.array([100, 50], dtype=np.int32) # Initial resources
        self.opponent_strength = np.array([50], dtype=np.int32) # Initial opponent strength

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Helper to get current observation."""
        return {
            "player_resources": self.player_resources,
            "opponent_strength": self.opponent_strength,
            "turn_number": np.array(self.current_turn, dtype=np.int32),
        }

    def _get_info(self) -> Dict[str, Any]:
        """Helper to get current info."""
        return {
            "player_gold": self.player_resources[0],
            "player_wood": self.player_resources[1],
            "opponent_strength": self.opponent_strength[0],
            "current_turn": self.current_turn,
        }

    def reset(self, seed: int = None, options: Dict = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_turn = 0
        self.player_resources = np.array([100, 50], dtype=np.int32)
        self.opponent_strength = np.array([50], dtype=np.int32)

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        # Unpack actions
        diplomacy_action = action["diplomacy"]
        economy_action = action["economy"]

        reward = 0.0
        terminated = False
        truncated = False

        # --- Apply Diplomacy Actions ---
        if diplomacy_action == 0:  # Trade
            # Placeholder for trading logic: gain some resources
            self.player_resources[0] += np.random.randint(5, 15)
            self.player_resources[1] += np.random.randint(2, 8)
            reward += 1.0
        elif diplomacy_action == 1:  # Pass
            reward -= 0.1 # Small penalty for passing
        elif diplomacy_action == 2:  # Attack
            # Placeholder for attack logic: simplified outcome
            if self.player_resources[0] > self.opponent_strength[0]: # If player has more gold than opponent strength
                self.opponent_strength[0] -= np.random.randint(5, 15)
                reward += 5.0
            else:
                self.player_resources[0] -= np.random.randint(5, 10)
                reward -= 2.0

        # --- Apply Economy Actions ---
        if economy_action == 0:  # Invest in Buildings
            # Placeholder for building logic: consume resources, gain future benefit (implicit in reward)
            if self.player_resources[0] >= 20:
                self.player_resources[0] -= 20
                reward += 2.0 # Reward for investing
            else:
                reward -= 0.5 # Penalty for trying to build without resources
        elif economy_action == 1:  # Create Units
            # Placeholder for unit creation logic: consume resources, increase strength
            if self.player_resources[1] >= 10:
                self.player_resources[1] -= 10
                self.player_resources[0] += 5 # Units might generate some gold
                reward += 1.5
            else:
                reward -= 0.5
        elif economy_action == 2:  # Idle
            self.player_resources[0] += 1 # Small passive gold gain for idling
            reward += 0.05


        # Update turn number
        self.current_turn += 1

        # Check for termination conditions
        if self.opponent_strength[0] <= 0:
            reward += 10.0 # Big reward for defeating opponent
            terminated = True
        elif self.player_resources[0] <= 0 and self.player_resources[1] <= 0 and self.current_turn > 10: # Player runs out of resources
            reward -= 5.0 # Penalty for losing all resources
            terminated = True
        elif self.current_turn >= 100:
            truncated = True # Episode ends after 100 turns

        # Ensure resources don't go below zero
        self.player_resources = np.maximum(self.player_resources, 0)
        self.opponent_strength = np.maximum(self.opponent_strength, 0)

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        print(f"--- Turn {self.current_turn} ---")
        print(f"Player Resources: Gold={self.player_resources[0]}, Wood={self.player_resources[1]}")
        print(f"Opponent Strength: {self.opponent_strength[0]}")
        print("--------------------")

    def close(self):
        pass
