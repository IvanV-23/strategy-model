import numpy as np
import torch
from typing import Dict, List, Tuple
import gymnasium as gym

class ReplayBuffer:
    def __init__(self, capacity: int, observation_space: gym.spaces.Dict, action_space: gym.spaces.Dict):
        self.capacity = capacity
        self.idx = 0
        self.size = 0
        
        # Board: (Capacity, Channels, Height, Width)
        self.board_states = np.zeros((capacity, 4, 8, 8), dtype=np.float32)
        self.next_board_states = np.zeros((capacity, 4, 8, 8), dtype=np.float32)
        
        # Stats: player(4) + opp(3) + turn(1) = 8
        self.stats = np.zeros((capacity, 8), dtype=np.float32)
        self.next_stats = np.zeros((capacity, 8), dtype=np.float32)

        # Actions
        self.actions_diplomacy = np.zeros((capacity,), dtype=np.int64)
        self.actions_economy = np.zeros((capacity,), dtype=np.int64)
        self.actions_distribution = np.zeros((capacity,), dtype=np.int64)
        self.actions_target = np.zeros((capacity,), dtype=np.int64)
        
        # Masking Feature: Store the 8x8 grid mask
        self.masks = np.ones((capacity, 64), dtype=np.bool_)

        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.terminated = np.zeros((capacity,), dtype=np.bool_)
        self.truncated = np.zeros((capacity,), dtype=np.bool_)

    def _extract_stats(self, obs: Dict) -> np.ndarray:
        return np.concatenate([
            obs["player_resources"],   
            obs["opponent_resources"], 
            [obs["turn_number"]]       
        ]).astype(np.float32)

    def add(self, state: Dict, action: Dict, reward: float, next_state: Dict, terminated: bool, truncated: bool, mask: np.ndarray = None):
        self.board_states[self.idx] = state["board_state"]
        self.next_board_states[self.idx] = next_state["board_state"]
        
        self.stats[self.idx] = self._extract_stats(state)
        self.next_stats[self.idx] = self._extract_stats(next_state)

        self.actions_diplomacy[self.idx] = action["diplomacy"]
        self.actions_economy[self.idx] = action["economy"]
        self.actions_distribution[self.idx] = action["distribution"]
        self.actions_target[self.idx] = action["target_tile"]
        
        # Store mask if provided, else default to all True
        if mask is not None:
            self.masks[self.idx] = mask.flatten()
        else:
            self.masks[self.idx] = True

        self.rewards[self.idx] = reward
        self.terminated[self.idx] = terminated
        self.truncated[self.idx] = truncated

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple:
        idxs = np.random.choice(self.size, size=batch_size, replace=False)
        
        states = {
            "board_state": torch.from_numpy(self.board_states[idxs]),
            "player_resources": torch.from_numpy(self.stats[idxs, 0:4]),
            "opponent_resources": torch.from_numpy(self.stats[idxs, 4:7]),
            "turn_number": torch.from_numpy(self.stats[idxs, 7:8])
        }
        
        next_states = {
            "board_state": torch.from_numpy(self.next_board_states[idxs]),
            "player_resources": torch.from_numpy(self.next_stats[idxs, 0:4]),
            "opponent_resources": torch.from_numpy(self.next_stats[idxs, 4:7]),
            "turn_number": torch.from_numpy(self.next_stats[idxs, 7:8])
        }

        return (
            states, 
            torch.from_numpy(self.actions_diplomacy[idxs]),
            torch.from_numpy(self.actions_economy[idxs]),
            torch.from_numpy(self.actions_distribution[idxs]),
            torch.from_numpy(self.actions_target[idxs]),
            torch.from_numpy(self.rewards[idxs]),
            next_states,
            torch.from_numpy(self.terminated[idxs]),
            torch.from_numpy(self.truncated[idxs]),
            torch.from_numpy(self.masks[idxs]) # Return mask
        )

    def clear(self):
        self.idx = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size
