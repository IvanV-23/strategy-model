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
        
        # Full Stats: = 15
        # Increased to 10 to be safe and scalable
        self.stats_dim = 15
        self.stats = np.zeros((capacity, self.stats_dim), dtype=np.float32)
        self.next_stats = np.zeros((capacity, self.stats_dim), dtype=np.float32)

        self.actions_diplomacy = np.zeros((capacity,), dtype=np.int64)
        self.actions_economy = np.zeros((capacity, 3), dtype=np.int64) 
        self.actions_distribution = np.zeros((capacity,), dtype=np.int64)
        self.actions_target = np.zeros((capacity,), dtype=np.int64)
        
        # Masking: We now store two masks: Target (64) and Build (7)
        self.masks_target = np.ones((capacity, 64), dtype=np.bool_)
        self.masks_build = np.ones((capacity, 7), dtype=np.bool_)

        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.terminated = np.zeros((capacity,), dtype=np.bool_)
        self.truncated = np.zeros((capacity,), dtype=np.bool_)

        # Add for GAE
        self.advantages = np.zeros((capacity,), dtype=np.float32)
        self.returns = np.zeros((capacity,), dtype=np.float32)
    
    def _extract_stats(self, obs: Dict) -> np.ndarray:
            # Extract all 4 stats from the new board_stats key
            mine_count = obs["board_stats"][0]
            mine_cap   = obs["board_stats"][1]
            gold_inc   = obs["board_stats"][2] 
            wood_inc   = obs["board_stats"][3] 
            trade_routes_count = obs["board_stats"][4]
            owned_tiles = obs["board_stats"][5]
            net_income = obs["board_stats"][6]
            
            return np.concatenate([
                obs["player_resources"],    # [0,1,2,3]
                obs["opponent_resources"],  # [4,5,6]
                [mine_count],               # [7]
                [mine_cap],                 # [8]
                [gold_inc],                 # [9]  <- New
                [wood_inc],                 # [10] <- New
                [trade_routes_count],
                [owned_tiles],
                [net_income],
                [obs["turn_number"]]        # [11] <- Moves to index 11
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
                    
            def get_dict(stats_array, board_array):
                return {
                    "board_state": torch.from_numpy(board_array[idxs]),
                    "player_resources": torch.from_numpy(stats_array[idxs, 0:4]),
                    "opponent_resources": torch.from_numpy(stats_array[idxs, 4:7]),
                    "mine_stats": torch.from_numpy(stats_array[idxs, 7:11]), # Updated 7:9 to 7:11 (includes income)
                    "turn_number": torch.from_numpy(stats_array[idxs, 11:12]), # Updated 9:10 to 11:12
                    "full_stats": torch.from_numpy(stats_array[idxs]) 
                }

            return (
                get_dict(self.stats, self.board_states), 
                torch.from_numpy(self.actions_diplomacy[idxs]),
                torch.from_numpy(self.actions_economy[idxs]),
                torch.from_numpy(self.actions_distribution[idxs]),
                torch.from_numpy(self.actions_target[idxs]),
                torch.from_numpy(self.rewards[idxs]),
                get_dict(self.next_stats, self.next_board_states),
                torch.from_numpy(self.terminated[idxs]),
                torch.from_numpy(self.truncated[idxs]),
                torch.from_numpy(self.masks_target[idxs]),
                torch.from_numpy(self.masks_build[idxs]),
                torch.from_numpy(self.advantages[idxs]),
                torch.from_numpy(self.returns[idxs])
            )
    
    def store_gae(self, start_idx: int, advantages: np.ndarray, returns: np.ndarray):
        indices = np.arange(start_idx, start_idx + len(advantages)) % self.capacity
        self.advantages[indices] = advantages
        self.returns[indices] = returns

    def clear(self):
        self.idx = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size