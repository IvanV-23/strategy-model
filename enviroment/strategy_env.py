import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any
import pygame


class StrategyEnv(gym.Env):
    """
    A Multi-Agent Reinforcement Learning environment for a strategic game.
    The agent manages two distinct decision-making branches: Diplomacy and Economy.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: str = None):
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
        self.observation_space = spaces.Dict({
            "player_resources": spaces.Box(low=0, high=1000, shape=(2,), dtype=np.int32),
            "opponent_strength": spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32),
            "turn_number": spaces.Discrete(100),
        })

        self.current_turn = 0
        self.player_resources = np.array([100, 50], dtype=np.int32)
        self.opponent_strength = np.array([50], dtype=np.int32)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None
        self.screen_width = 800
        self.screen_height = 600

    def _get_obs(self) -> Dict[str, np.ndarray]:
        return {
            "player_resources": self.player_resources,
            "opponent_strength": self.opponent_strength,
            "turn_number": self.current_turn,
        }

    def _get_info(self) -> Dict[str, Any]:
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

        if self.render_mode == "human":
            self.render()

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        diplomacy_action = action["diplomacy"]
        economy_action = action["economy"]
        reward = 0.0
        terminated = False
        truncated = False

        if diplomacy_action == 0:
            self.player_resources[0] += np.random.randint(5, 15)
            self.player_resources[1] += np.random.randint(2, 8)
            reward += 1.0
        elif diplomacy_action == 1:
            reward -= 0.1
        elif diplomacy_action == 2:
            if self.player_resources[0] > self.opponent_strength[0]:
                self.opponent_strength[0] -= np.random.randint(5, 15)
                reward += 5.0
            else:
                self.player_resources[0] -= np.random.randint(5, 10)
                reward -= 2.0

        if economy_action == 0:
            if self.player_resources[0] >= 20:
                self.player_resources[0] -= 20
                reward += 2.0
            else:
                reward -= 0.5
        elif economy_action == 1:
            if self.player_resources[1] >= 10:
                self.player_resources[1] -= 10
                self.player_resources[0] += 5
                reward += 1.5
            else:
                reward -= 0.5
        elif economy_action == 2:
            self.player_resources[0] += 1
            reward += 0.05

        self.current_turn += 1

        if self.opponent_strength[0] <= 0:
            reward += 10.0
            terminated = True
        elif self.player_resources[0] <= 0 and self.player_resources[1] <= 0 and self.current_turn > 10:
            reward -= 5.0
            terminated = True
        elif self.current_turn >= 100:
            truncated = True

        self.player_resources[:] = np.maximum(self.player_resources, 0)
        self.opponent_strength[:] = np.maximum(self.opponent_strength, 0)

        if self.render_mode == "human":
            self.render()

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None or pygame is None:
            return

        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.set_caption("Strategy Game")
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)

        self.screen.fill((20, 20, 20)) # Dark background

        # Colors
        GOLD_COLOR = (255, 215, 0)
        WOOD_COLOR = (139, 69, 19)
        STRENGTH_COLOR = (255, 0, 0)
        TEXT_COLOR = (255, 255, 255)

        # --- Display Player Resources ---
        gold_text = self.font.render(f"Gold: {self.player_resources[0]}", True, GOLD_COLOR)
        wood_text = self.font.render(f"Wood: {self.player_resources[1]}", True, WOOD_COLOR)
        self.screen.blit(gold_text, (50, 50))
        self.screen.blit(wood_text, (50, 90))

        # --- Display Opponent Strength ---
        strength_text = self.font.render(f"Opponent Strength: {self.opponent_strength[0]}", True, STRENGTH_COLOR)
        self.screen.blit(strength_text, (self.screen_width - 350, 50))

        # --- Display Turn Number ---
        turn_text = self.font.render(f"Turn: {self.current_turn}", True, TEXT_COLOR)
        self.screen.blit(turn_text, (self.screen_width // 2 - 50, 20))
        
        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )


    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None
            self.font = None
