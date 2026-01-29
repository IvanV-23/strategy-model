import numpy as np
import torch
from typing import Dict, List, Tuple
import gymnasium as gym

class ReplayBuffer:
    def __init__(self, capacity: int, observation_space: gym.spaces.Dict, action_space: gym.spaces.Dict):
        self.capacity = capacity
        self.idx = 0
        self.size = 0

        self.state_dim = 0
        for space in observation_space.spaces.values():
            if isinstance(space, gym.spaces.Box):
                self.state_dim += int(np.prod(space.shape))
            elif isinstance(space, gym.spaces.Discrete):
                self.state_dim += 1
        
        self.states = np.zeros((capacity, self.state_dim), dtype=np.float32)
        self.actions_diplomacy = np.zeros((capacity, ), dtype=np.int64)
        self.actions_economy = np.zeros((capacity, ), dtype=np.int64)
        self.rewards = np.zeros((capacity, ), dtype=np.float32)
        self.next_states = np.zeros((capacity, self.state_dim), dtype=np.float32)
        self.terminated = np.zeros((capacity, ), dtype=np.bool_)
        self.truncated = np.zeros((capacity, ), dtype=np.bool_)
        
        # Store the keys for consistent order when flattening
        self._obs_keys = sorted(observation_space.spaces.keys())

    def add(self, state: Dict, action: Dict, reward: float, next_state: Dict, terminated: bool, truncated: bool):
        flat_state = self._flatten_observation(state)
        flat_next_state = self._flatten_observation(next_state)

        self.states[self.idx] = flat_state
        self.actions_diplomacy[self.idx] = action["diplomacy"]
        self.actions_economy[self.idx] = action["economy"]
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = flat_next_state
        self.terminated[self.idx] = terminated
        self.truncated[self.idx] = truncated

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        idxs = np.random.choice(self.size, size=batch_size, replace=False)
        
        states = torch.tensor(self.states[idxs], dtype=torch.float32)
        actions_diplomacy = torch.tensor(self.actions_diplomacy[idxs], dtype=torch.long)
        actions_economy = torch.tensor(self.actions_economy[idxs], dtype=torch.long)
        rewards = torch.tensor(self.rewards[idxs], dtype=torch.float32)
        next_states = torch.tensor(self.next_states[idxs], dtype=torch.float32)
        terminated = torch.tensor(self.terminated[idxs], dtype=torch.bool)
        truncated = torch.tensor(self.truncated[idxs], dtype=torch.bool)

        return states, actions_diplomacy, actions_economy, rewards, next_states, terminated, truncated

    def _flatten_observation(self, obs: Dict) -> np.ndarray:
        """Flattens a dictionary observation into a single numpy array."""
        flat_obs = []
        for key in self._obs_keys:
            value = obs[key]
            if isinstance(value, np.ndarray):
                flat_obs.extend(value.flatten())
            else: # Handles scalar values like int, float, or np.int32
                flat_obs.append(value)
        return np.array(flat_obs, dtype=np.float32)

    def __len__(self) -> int:
        return self.size
