from typing import Tuple, Dict, Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from enviroment.enviroment_blocks.opponent_env import OpponentEnv
from enviroment.enviroment_blocks.player_env import PlayerEnv
from enviroment.enviroment_render.strategy_renderer import StrategyRenderer
from enviroment.enviroment_blocks.board_env import BoardEnv

class StrategyEnv(gym.Env):
    """
    A Multi-Agent Reinforcement Learning environment for a strategic game.
    The agent manages two distinct decision-making branches: Diplomacy and Economy.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: str = None):
        super().__init__()

        #Initialize enviroment blocks
        self.opponent_env = OpponentEnv()
        self.player_env = PlayerEnv()
        self.board_env = BoardEnv()

        #Initialize renderer
        # 1. Define dimensions FIRST
        self.screen_width = 800
        self.screen_height = 600
        self.render_mode = render_mode
        self.metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
        self.renderer = StrategyRenderer(self.screen_width, self.screen_height, self.metadata)

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
            "player_resources": spaces.Box(low=0, high=1000, shape=(4,), dtype=np.int32),
            "opponent_resources": spaces.Box(low=0, high=1000, shape=(3,), dtype=np.int32), 
            "turn_number": spaces.Discrete(100),
        })

        self.current_turn = 0
        #self.player_resources = np.array([100, 50, 0], dtype=np.int32)
        #self.opponent_strength = np.array([50], dtype=np.int32)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None
        self.screen_width = 800
        self.screen_height = 600

    def _get_obs(self) -> Dict[str, np.ndarray]:
        return {
            "player_resources": self.player_env.resources,
            "opponent_resources": self.opponent_env.resources, # Returns the (3,) array
            "turn_number": self.current_turn,
        }

    def _get_info(self) -> Dict[str, Any]:
        return {
            "player_gold": self.player_env.resources[0],
            "player_wood": self.player_env.resources[1],
            "opponent_strength": self.opponent_env.resources[2],
            "current_turn": self.current_turn,
        }
    def _action_extraction(self, action:dict):
        self.diplomacy_action = action["diplomacy"]
        self.economy_action = action["economy"]
        reward = 0.0
        terminated = False
        truncated = False
        return reward, terminated, truncated

    def _store_actions_for_rendering(self, action:dict):
        self.last_diplomacy_choice = action["diplomacy"]
        self.last_economy_choice = action["economy"]
        self.dip_labels = ["Trade", "Pass", "Attack"]
        self.eco_labels = ["Invest", "Create Units", "Idle"]
        self.last_dip_str = self.dip_labels[self.last_diplomacy_choice]
        self.last_eco_str = self.eco_labels[self.last_economy_choice]

    def reset(self, seed: int = None, options: Dict = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_turn = 0
        self.player_env.reset()
        
        self.opponent_env.reset()

        self.board_env.reset()


        if self.render_mode == "human":
            self.render()

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:

            
            # 0. Action Extraction
            reward,terminated, truncated = self._action_extraction(action)


            # 1. Resource calculation 
            reward += self.player_env.resource_calculation()
            

            # 2. Store Actions for Rendering
            self._store_actions_for_rendering(action)

            # 3. --- DIPLOMACY BRANCH ---

            if self.diplomacy_action == 0: # Trade
                reward += self.player_env.trade()

            elif self.diplomacy_action == 1: # Pass
                pass

            elif self.diplomacy_action == 2: # Attack (Modified for Soldiers)
                attack_reward, updated_opponent_status, battle_victory = self.player_env.attack(
                    opponent_status={
                        "gold": self.opponent_env.resources[0],
                        "wood": self.opponent_env.resources[1],
                        "soldiers": self.opponent_env.resources[2],
                        "defense": 10 + self.opponent_env.resources[2]
                    }
                )
                reward += attack_reward
                self.opponent_env.resources[0] = updated_opponent_status["gold"]
                self.opponent_env.resources[1] = updated_opponent_status["wood"]
                self.opponent_env.resources[2] = updated_opponent_status["soldiers"]
                if battle_victory:
                    self.board_env.claim_adjacent_tile(owner_id=1)

            # 4. --- ECONOMY BRANCH ---
            if self.economy_action == 0: # Invest (Gold -> Wood)
                reward += self.player_env.invest()
                    
            elif self.economy_action == 1: # Create Units (Wood -> Soldiers)
                reward += self.player_env.create_units()
                    
            elif self.economy_action == 2: # Idle
                reward += self.player_env.build_gold_getter()

            # 5. Turn and Resource Management
            self.current_turn += 1
            
            # 6. Turn and Opponent Status
            reward -= self.opponent_env.score_calculation() + self.current_turn * 0.1 # Penalty for opponent strength

            # 7. Opponent Action
            opp_reward, updated_player_status, opp_truncated, battle_victory = self.opponent_env.action_step(
                current_turn=self.current_turn, 
                player_status={
                    "gold": self.player_env.resources[0],
                    "wood": self.player_env.resources[1],
                    "soldiers": self.player_env.resources[2],
                    "defense": 10 + self.player_env.resources[2]    
                }
            )
            reward += opp_reward
            self.player_env.resources[0] = updated_player_status["gold"]
            self.player_env.resources[1] = updated_player_status["wood"]
            self.player_env.resources[2] = updated_player_status["soldiers"]
            if opp_truncated:
                truncated = True
                print(f"Player defeated! {reward} reward.")
            if battle_victory:
                self.board_env.claim_adjacent_tile(owner_id=2)


            # 8. Termination Logic
            if self.opponent_env.resources[2] <= 0:
                reward += 100.0 # Victory reward
                terminated = True
                print(f"Opponent defeated! {reward} reward.")

            elif self.player_env.resources[0] <= 0 and self.player_env.resources[1] <= 0:
                # Bankrupt condition
                reward -= self.opponent_env.resources[2] * 2
                terminated = True
                print(f"Player bankrupt! {reward} reward.")

            # 9. Rendering and Return
            if self.render_mode == "human":
                self.render()

            observation = self._get_obs()
            info = self._get_info()

            return observation, reward, terminated, truncated, info

    def render(self):
        # 1. Guard clause
        if self.render_mode is None:
            return


        # Pack current state into a dictionary
        state_data = {
            'p_res': self.player_env.resources,
            'o_res': self.opponent_env.resources,
            'turn': self.current_turn,
            "board": self.board_env.get_tile_data(),
            'dip_act': getattr(self, 'last_dip_str', "None"),
            'eco_act': getattr(self, 'last_eco_str', "None")
    }

        # Call the external renderer
        return self.renderer.render_frame(self.render_mode, state_data)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None
            self.font = None
