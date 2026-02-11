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
        self.board_states = np.zeros((capacity, 5, 8, 8), dtype=np.float32)
        self.next_board_states = np.zeros((capacity, 5, 8, 8), dtype=np.float32)
        
        # Stats: player(4) + opp(3) + mine_count(1) + mine_capacity(1) + turn(1) = 10
        # Increased to 10 to be safe and scalable
        self.stats_dim = 10 
        self.stats = np.zeros((capacity, self.stats_dim), dtype=np.float32)
        self.next_stats = np.zeros((capacity, self.stats_dim), dtype=np.float32)

        self.actions_diplomacy = np.zeros((capacity,), dtype=np.int64)
        self.actions_economy = np.zeros((capacity, 2), dtype=np.int64) 
        self.actions_distribution = np.zeros((capacity,), dtype=np.int64)
        self.actions_target = np.zeros((capacity,), dtype=np.int64)
        
        # Masking: We now store two masks: Target (64) and Build (6)
        self.masks_target = np.ones((capacity, 64), dtype=np.bool_)
        self.masks_build = np.ones((capacity, 6), dtype=np.bool_)

        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.terminated = np.zeros((capacity,), dtype=np.bool_)
        self.truncated = np.zeros((capacity,), dtype=np.bool_)

    def _extract_stats(self, obs: Dict) -> np.ndarray:
        # Extract mine stats from the new board_stats key
        mine_count = obs["board_stats"][0]
        mine_cap = obs["board_stats"][1]
        
        return np.concatenate([
            obs["player_resources"],    # [0,1,2,3]
            obs["opponent_resources"],  # [4,5,6]
            [mine_count],               # [7]
            [mine_cap],                 # [8]
            [obs["turn_number"]]        # [9]
        ]).astype(np.float32)

    def add(self, state: Dict, action: Dict, reward: float, next_state: Dict, 
            terminated: bool, truncated: bool, info: Dict = None):
        
        self.board_states[self.idx] = state["board_state"]
        self.next_board_states[self.idx] = next_state["board_state"]
        
        self.stats[self.idx] = self._extract_stats(state)
        self.next_stats[self.idx] = self._extract_stats(next_state)

        self.actions_diplomacy[self.idx] = action["diplomacy"]
        self.actions_economy[self.idx] = action["economy"] 
        self.actions_distribution[self.idx] = action["distribution"]
        self.actions_target[self.idx] = action["target_tile"]
        
        # Store both masks from the info dict
        if info:
            self.masks_target[self.idx] = info.get("action_mask", np.ones(64)).flatten()
            self.masks_build[self.idx] = info.get("build_mask", np.ones(6)).flatten()

        self.rewards[self.idx] = reward
        self.terminated[self.idx] = terminated
        self.truncated[self.idx] = truncated

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple:
        idxs = np.random.choice(self.size, size=batch_size, replace=False)
        
        def get_dict(stats_array):
            return {
                "board_state": torch.from_numpy(self.board_states[idxs]),
                "player_resources": torch.from_numpy(stats_array[idxs, 0:4]),
                "opponent_resources": torch.from_numpy(stats_array[idxs, 4:7]),
                "mine_stats": torch.from_numpy(stats_array[idxs, 7:9]), # Count and Cap
                "turn_number": torch.from_numpy(stats_array[idxs, 9:10]),
                # Combined stats for the Actor's FC layer (now size 10)
                "full_stats": torch.from_numpy(stats_array[idxs]) 
            }

        return (
            get_dict(self.stats), 
            torch.from_numpy(self.actions_diplomacy[idxs]),
            torch.from_numpy(self.actions_economy[idxs]),
            torch.from_numpy(self.actions_distribution[idxs]),
            torch.from_numpy(self.actions_target[idxs]),
            torch.from_numpy(self.rewards[idxs]),
            get_dict(self.next_stats),
            torch.from_numpy(self.terminated[idxs]),
            torch.from_numpy(self.truncated[idxs]),
            torch.from_numpy(self.masks_target[idxs]),
            torch.from_numpy(self.masks_build[idxs])
        )
    def clear(self):
        self.idx = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size