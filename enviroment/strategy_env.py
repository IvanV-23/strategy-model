from typing import Tuple, Dict, Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from enviroment.reward_env import RewardEnv
from enviroment.enviroment_blocks.resources_manager_env import ResourceManager
from enviroment.enviroment_blocks.opponent_env import OpponentEnv
from enviroment.enviroment_blocks.player_env import PlayerEnv
import os
from enviroment.enviroment_render.strategy_renderer import StrategyRenderer
from enviroment.enviroment_render.cpp_bridge import CppRendererBridge
from enviroment.enviroment_blocks.board_env import BoardEnv
from enviroment.enviroment_blocks.stats_env import StatsEnv
from enviroment.enviroment_branches.diplomacy_env import DiplomacyEnv
from enviroment.enviroment_branches.economy_env import EconomyEnv
class StrategyEnv(gym.Env):
    """
    A Multi-Agent Reinforcement Learning environment for a strategic game.
    The agent manages two distinct decision-making branches: Diplomacy and Economy.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: str = None):
        super().__init__()

        self.history = ["Game Started"]

        #Initialize enviroment blocks
        self.opponent_env = OpponentEnv()
        self.player_env = PlayerEnv()
        self.board_env = BoardEnv()
        self.stats_env = StatsEnv(board_env=self.board_env,player_env=self.player_env)
        self.reward_env = RewardEnv(board_env=self.board_env,player_env=self.player_env)
        self.resource_manager = ResourceManager(board_env=self.board_env, player_env=self.player_env)


        #Initiliaze enviroment branches
        self.diplomacy_branch = DiplomacyEnv(board_env=self.board_env,player_env=self.player_env,opponent_env=self.opponent_env)
        self.economy_branch = EconomyEnv(board_env=self.board_env,player_env=self.player_env,opponent_env=self.opponent_env)
        
        #Initialize renderer
        self.screen_width = 800
        self.screen_height = 1000
        self.render_mode = render_mode
        self.metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

        if os.getenv("USE_CPP_RENDER") == "1" and render_mode == "human":
            print("Attempting to use C++ renderer...")
            try:
                self.renderer = CppRendererBridge(self.screen_width, self.screen_height, self.metadata)
                if self.renderer.proc is None: # Fallback if C++ renderer fails to start
                    print("C++ renderer failed to start. Falling back to Pygame renderer.")
                    self.renderer = StrategyRenderer(self.screen_width, self.screen_height, self.metadata)
            except Exception as e:
                print(f"Error initializing C++ renderer: {e}. Falling back to Pygame.")
                self.renderer = StrategyRenderer(self.screen_width, self.screen_height, self.metadata)
        else:
            self.renderer = StrategyRenderer(self.screen_width, self.screen_height, self.metadata)

        # Define the action space for Diplomacy and Economy
        # Diplomacy: 0: Trade, 1: Pass, 2: Attack

        ## 0 means "Do nothing" for that specific unit
        self.diplomacy_action_space = spaces.Discrete(3)
        self.economy_action_space = spaces.MultiDiscrete([10, 6, 6, 2, 2])  # [workers, mines, trade, warehouse, crop_field]
        self.distribution_action_space = spaces.Discrete(5)
        self.target_action_space = spaces.Discrete(256)  # 16x16 board
        
        # Combined action space (Tuple of discrete actions)
        self.action_space = spaces.Dict({
            "diplomacy": self.diplomacy_action_space,
            "economy": self.economy_action_space,
            "distribution": self.distribution_action_space,
            "target_tile": self.target_action_space,
        })

        # Define the observation space
        self.observation_space = spaces.Dict({
            "player_resources": spaces.Box(low=0, high=1000, shape=(9,), dtype=np.int32),
            "opponent_resources": spaces.Box(low=0, high=1000, shape=(3,), dtype=np.int32), 
            "turn_number": spaces.Discrete(1000),
            "board_state": spaces.Box(low=0, high=255, shape=(5, 16, 16), dtype=np.int32),
            "board_stats": spaces.Box(low=0, high=1000, shape=(14,), dtype=np.float32),
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
            "player_resources": np.append(self.player_env.resources, self.player_env._capacity),
            "opponent_resources": self.opponent_env.resources, # Returns the (3,) array
            "turn_number": self.current_turn,
            "board_state": self.board_env.full_board_state(),  # Returns a (5, 16, 16) array
            "board_stats": self.stats_env.get_game_stats(
                lost_gold=getattr(self, 'lost_gold', 0),
                lost_wood=getattr(self, 'lost_wood', 0),
                lost_food=getattr(self, 'lost_food', 0),
                defeated_workers=getattr(self, 'defeated_workers', 0)
            )
        }

    def _get_info(self) -> Dict[str, Any]:
        return {
            "player_gold": self.player_env.resources[0],
            "player_wood": self.player_env.resources[1],
            "opponent_strength": self.opponent_env.resources[2],
            "current_turn": self.current_turn,
            "action_mask": self.board_env.get_action_mask(player_id=1),
            "build_mask": self.board_env.get_build_mask(player_id=1, player_gold=self.player_env.resources[0], player_wood=self.player_env.resources[1])
        }

    def _action_extraction(self, action:dict):
        #Diplo Head 
        self.diplomacy_action = action["diplomacy"]

        #Economy Head
        # Directly extract counts
        # action["economy"] is now an array like [3, 0]
        self.workers_to_build = action["economy"][0]
        self.mines_to_build = action["economy"][1]
        self.trade_route_action = action["economy"][2]
        self.warehouse_action = action["economy"][3]
        self.crop_field_action = action["economy"][4]
        
        # Distribution Head
        self.distribution_action = action["distribution"]

        # Target Head
        target_idx = action["target_tile"]
        self.target_row = target_idx // self.board_env.cols
        self.target_col = target_idx % self.board_env.cols


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

    def _store_actions_for_rendering(self, action: dict):
        self.last_diplomacy_choice = action["diplomacy"]
        eco_act = action["economy"]
        
        # Display as "Build: 2S, 1M"
        self.last_eco_str = f"Build: {eco_act[0]}S, {eco_act[1]}M"
        
        self.dip_labels = ["Trade", "Pass", "Attack"]
        self.last_dip_str = self.dip_labels[self.last_diplomacy_choice]

    def reset(self, seed: int = None, options: Dict = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_turn = 0
        self.player_env.reset()
        
        self.opponent_env.reset()

        self.board_env.reset()

        self.lost_gold = 0
        self.lost_wood = 0
        self.lost_food = 0
        self.defeated_workers = 0


        if self.render_mode == "human":
            self.render()

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:

            
            # 0. Action Extraction
            reward,terminated, truncated = self._action_extraction(action)
            # 0.1 Workers redistribution
            p1_workers = self.player_env.resources[2]
            p2_workers = self.opponent_env.resources[2]

            self.board_env.redistribute_workers(owner_id=1, total_workers=p1_workers, style=self.distribution_action)
            self.board_env.redistribute_workers(owner_id=2, total_workers=p2_workers, style=0)
            

            # 1. Resource calculation 
            player_resources_calculation_result = self.resource_manager.collect_resources(player_id=1)
            self.lost_gold = player_resources_calculation_result.get("lost_gold", 0)
            self.lost_wood = player_resources_calculation_result.get("lost_wood", 0)
            self.defeated_workers = 0

            
            
            # 2. Store Actions for Rendering
            self._store_actions_for_rendering(action)

            # 3. --- DIPLOMACY BRANCH ---

            action_result = self.diplomacy_branch.execute_diplomacy(diplomacy_action=self.diplomacy_action,
                                                                    target_col=self.target_col,
                                                                    target_row=self.target_row,
                                                                    current_turn=self.current_turn
                                                                    )

            truncated = action_result["truncated"]
            reward += action_result["reward"]
            self.defeated_workers += action_result.get("defeated_workers", 0)
            self.history.append(action_result["history"])

            # 4. --- ECONOMY BRANCH ---

            economy_action_result = self.economy_branch.execute_economy_action(new_workers=self.workers_to_build,
                                                                               new_mines=self.mines_to_build,
                                                                               trade_route_action=self.trade_route_action,
                                                                               warehouse_action=self.warehouse_action,
                                                                               crop_field_action=self.crop_field_action
                                                                               )

            reward += economy_action_result["reward"]

            # Food Consumption
            food_reward = self.player_env.process_food_consumption()
            reward += food_reward


            # 5. Turn and Resource Management
            self.current_turn += 1
            
            # 6. Turn and Opponent Status
            # reward -= self.opponent_env.score_calculation() + self.current_turn * 0.1 # Penalty for opponent strength

            # 7. Opponent Action
            # First,redistribute opponent workers to the board so the attack power is correct
            self.board_env.redistribute_workers(owner_id=2, total_workers=self.opponent_env.resources[2], style=0)

            # Get opponent's intent
            intent_to_attack = self.opponent_env.action_step(self.current_turn, self.board_env.get_owned_tiles(owner_id=2))

            if intent_to_attack:
                # The Board determines if the attack succeeds based on worker proximity
                battle_victory, base_captured, prev_owner, defeated_by_opponent, mine_captured = self.board_env.claim_adjacent_tile(owner_id=2)
                
                if battle_victory:
                    print("Opponent captured a tile!")
                    # P2 Gains resources for winning
                    self.opponent_env.resources[0] += 5
                    self.opponent_env.resources[1] += 2
                    
                    # Deduct the actual workers lost on the tile from Player 1's total pool
                    if prev_owner == 1:
                        self.player_env.resources[2] = int(max(0, self.player_env.resources[2] - defeated_by_opponent))
        
                    if mine_captured:
                        print("Mine captured")
                        reward -= 0.1
                        self.player_env.resources[3] -= 1
                    if base_captured:
                        reward -= 100.0
                        truncated = True
                        print(f"Player DEFEATED at turn {self.current_turn}!")
                else:
                    # Attack failed (Opponent wasn't strong enough or no adjacent tiles)
                    # Small penalty to opponent resources for the failed campaign
                    self.opponent_env.resources[2] = max(0, self.opponent_env.resources[2] - self.opponent_env.resources[2]*0.1)
                    reward += 0.1

            
            self.player_env.resources[3] = int (self.board_env.p1_buildings_manager.get_mine_count(player_id=1))

            if self.player_env.resources[0] <= 0:
                # Bankrupt condition
                #reward -= 0.01 * abs(self.player_env.resources[0])  # Small penalty for running out of gold        
                reward -= 5
                print(f"Player bankrupt! {reward} reward.")

            # 9. Rendering and Return
            if self.render_mode == "human":
                self.render()

            self.board_env.update_board()

            observation = self._get_obs()
            info = self._get_info()

            reward += self.reward_env.calculate_player_resources_reward(game_turn = self.current_turn,
                                                                        model_observations=observation,
                                                                        player_resources_result=player_resources_calculation_result)

            return observation, reward, terminated, truncated, info

    def render(self):
        # 1. Guard clause
        if self.render_mode is None:
            return


        # Pack current state into a dictionary
        state_data = {
            'p_res': self.player_env.resources,
            'o_res': self.opponent_env.resources,
            'p_gen': (self.player_env.gold_net_income,
                      self.player_env.wood_raw_income,
                      0,
                      self.board_env.collect_food_income(player_id=1)),
            'o_gen': (1 + self.board_env.get_owned_tiles(owner_id=2) * 2,
                      1 + self.board_env.get_owned_tiles(owner_id=2),
                      0),
            'turn': self.current_turn,
            "board": self.board_env.get_tile_data(),
            'dip_act': getattr(self, 'last_dip_str', "None"),
            'eco_act': getattr(self, 'last_eco_str', "None"),
            "p1_routes": self.board_env.p1_trade_manager.active_routes,
            "o1_routes": [], 
            "p1_base": self.board_env.p1_trade_manager.base_coords,
            "o1_base": (self.board_env.rows - 1, self.board_env.cols - 1),
            'history': self.history[-5:], 
            "p1_capacity": self.player_env._capacity,
            "p_warehouses": self.board_env.p1_buildings_manager.get_mines_by_owner_and_level(player_id=1, mine_level=2),
            "p_crops": self.board_env.p1_buildings_manager.get_crop_field_count(player_id=1)
            
        }

        # Call the external renderer
        return self.renderer.render_frame(self.render_mode, state_data)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None
            self.font = None
