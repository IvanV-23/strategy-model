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

        self.history = ["Game Started"]

        #Initialize enviroment blocks
        self.opponent_env = OpponentEnv()
        self.player_env = PlayerEnv()
        self.board_env = BoardEnv()

        #Initialize renderer
        # 1. Define dimensions FIRST
        self.screen_width = 800
        self.screen_height = 1000
        self.render_mode = render_mode
        self.metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
        self.renderer = StrategyRenderer(self.screen_width, self.screen_height, self.metadata)

        # Define the action space for Diplomacy and Economy
        # Diplomacy: 0: Trade, 1: Pass, 2: Attack
        # Economy: [0] Soldiers to build (0-5), [1] Mines to build (0-5)
        ## 0 means "Do nothing" for that specific unit
        self.diplomacy_action_space = spaces.Discrete(3)
        self.economy_action_space = spaces.MultiDiscrete([6, 6])
        self.distribution_action_space = spaces.Discrete(5)
        self.target_action_space = spaces.Discrete(64)  # Assuming a 8x8 board
        
        # Combined action space (Tuple of discrete actions)
        self.action_space = spaces.Dict({
            "diplomacy": self.diplomacy_action_space,
            "economy": self.economy_action_space,
            "distribution": self.distribution_action_space,
            "target_tile": self.target_action_space,
        })

        # Define the observation space
        self.observation_space = spaces.Dict({
            "player_resources": spaces.Box(low=0, high=1000, shape=(4,), dtype=np.int32),
            "opponent_resources": spaces.Box(low=0, high=1000, shape=(3,), dtype=np.int32), 
            "turn_number": spaces.Discrete(1000),
            "board_state": spaces.Box(low=0, high=255, shape=(5, 8, 8), dtype=np.int32),
            "board_stats": spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
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
            "board_state": self.board_env.full_board_state(),  # Returns a (5, 8, 8) array
            "board_stats": self.board_env.get_board_state_and_stats()["stats"]  # New: Get additional board stats for the critic
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
        self.soldiers_to_build = action["economy"][0]
        self.mines_to_build = action["economy"][1]

        # Distribution Head
        self.distribution_action = action["distribution"]

        # Target Head
        target_idx = action["target_tile"]
        self.target_row = target_idx // 8
        self.target_col = target_idx % 8


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


        if self.render_mode == "human":
            self.render()

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:

            
            # 0. Action Extraction
            reward,terminated, truncated = self._action_extraction(action)
            # 0.1 Soldiers redistribution

            p1_soldiers = self.player_env.resources[2] 
            p2_soldiers = self.opponent_env.resources[2]

            self.board_env.redistribute_soldiers(owner_id=1, total_soldiers=p1_soldiers, style=self.distribution_action)
            self.board_env.redistribute_soldiers(owner_id=2, total_soldiers=p2_soldiers, style=0)

            # 1. Resource calculation 

            reward += self.player_env.resource_calculation(owned_tiles=self.board_env.get_owned_tiles(owner_id=1),
                                                           wood_income=self.board_env.collect_income(player_id=1)
                                                           )
            

            # 2. Store Actions for Rendering
            self._store_actions_for_rendering(action)

            # 3. --- DIPLOMACY BRANCH ---

            if self.diplomacy_action == 0: # Trade
                reward += self.player_env.trade()

            elif self.diplomacy_action == 1: # Pass
                reward -= self.board_env.get_owned_tiles(owner_id=2)*0.5
            
            elif self.diplomacy_action == 2: # Attack (Modified for Soldiers)
                print(f"Player attack ")
                # 1. Ask the board to resolve combat based on spatial soldier distribution
                # Note: claim_adjacent_tile now handles 'Attack Power > Defense' internally
                victory, base_captured, prev_owner, reason, defeated_soldiers = self.board_env.claim_target_tile(1, (self.target_row, self.target_col))
                
                res_msg = "VICTORY" if victory else "FAILED"
                self.history.append(f"Attack on ({self.target_row},{self.target_col}): {res_msg}") # <--- ADD THIS
                
                # 2. Update Player 1 resources based on result
                attack_reward = self.player_env.process_battle_consequences(victory, base_captured, prev_owner, reason, self.board_env.get_owned_tiles(owner_id=1))
                reward += attack_reward

                # 3. Update Opponent resources if they lost
                if victory:
                    # If P1 won, P2 loses soldiers and resources
                    self.opponent_env.resources[0] = max(0, self.opponent_env.resources[0] - 50)
                    self.opponent_env.resources[1] = max(0, self.opponent_env.resources[1] - 25)
                    
                    # Soldiers are already handled by redistribution in the next step, 
                    # but let's reduce their pool directly for the loss:
                    self.opponent_env.resources[2] = max(0, self.opponent_env.resources[2] - defeated_soldiers)
                    reward += defeated_soldiers * 0.01  

                # 4. Handle game termination
                if base_captured:
                    truncated = True
                    print(f"Opponent defeated! Total Reward: {reward}")

            # 4. --- ECONOMY BRANCH ---


            eco_reward = self.player_env.process_economy(
                num_soldiers=self.soldiers_to_build, 
                num_mines=0
            )
            reward += eco_reward

            costs = [(0,0), (50,0), (100,20), (150,50), (200,80), (500,200)]
            gold_cost, wood_cost = costs[self.mines_to_build]

            # 2. Execute build if player has resources
            if self.player_env.resources[0] >= gold_cost and self.player_env.resources[1] >= wood_cost:
                success, msg = self.board_env.build_mine(1, self.mines_to_build)              
                if success and self.mines_to_build > 0:
                    self.player_env.resources[0] -= gold_cost
                    self.player_env.resources[1] -= wood_cost
                    reward += 0.5 * self.mines_to_build  # Reward for successful construction
                    self.player_env.resources[3] += 1 * self.mines_to_build  # Each mine increases wood income
                    print(f"Built {self.mines_to_build} mines. Reward: {reward}")
                if not success:
                    print(f"Failed to build mines: {msg}")
                    reward -= 0.1
            # 5. Turn and Resource Management
            self.current_turn += 1
            
            # 6. Turn and Opponent Status
            reward -= self.opponent_env.score_calculation() + self.current_turn * 0.1 # Penalty for opponent strength

            # 7. Opponent Action
            # First,redistribute opponent soldiers to the board so the attack power is correct
            self.board_env.redistribute_soldiers(owner_id=2, total_soldiers=self.opponent_env.resources[2], style=0)

            # Get opponent's intent
            intent_to_attack = self.opponent_env.action_step(self.current_turn, self.board_env.get_owned_tiles(owner_id=2))

            if intent_to_attack:
                # The Board determines if the attack succeeds based on soldier proximity
                battle_victory, base_captured, prev_owner, defeated_soldiers = self.board_env.claim_adjacent_tile(owner_id=2)
                
                if battle_victory:
                    print("Opponent captured a tile!")
                    # P2 Gains resources for winning
                    self.opponent_env.resources[0] += 5
                    self.opponent_env.resources[1] += 2
                    
                    # P1 Loses resources for losing a tile
                    #self.player_env.resources[0] = max(0, self.player_env.resources[0] - 50)
                    #self.player_env.resources[1] = max(0, self.player_env.resources[1] - 25)
          
                    #if prev_owner == 1:
                        #if self.board_env.get_owned_tiles(owner_id=2) > self.board_env.get_owned_tiles(owner_id=1):
                            #reward -= 0.02  # Penalty for losing a tile to opponent
                    
                    if base_captured:
                        reward -= 100.0
                        truncated = True
                        print(f"Player DEFEATED at turn {self.current_turn}!")
                else:
                    # Attack failed (Opponent wasn't strong enough or no adjacent tiles)
                    # Small penalty to opponent resources for the failed campaign
                    self.opponent_env.resources[2] = max(0, self.opponent_env.resources[2] - self.opponent_env.resources[2]*0.1)

            


            if self.player_env.resources[0] <= 0:
                # Bankrupt condition
                reward -= 0.01 * self.player_env.resources[0]  # Small penalty for running out of gold        
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
            'p_gen': (1 + self.board_env.get_owned_tiles(owner_id=1)*2,
                      self.board_env.collect_income(player_id=1)+ + self.player_env.resources[3] * 2,
                      0),
            'o_gen': (1 + self.board_env.get_owned_tiles(owner_id=2) * 2,
                      1 + self.board_env.get_owned_tiles(owner_id=2),
                      0),
            'turn': self.current_turn,
            "board": self.board_env.get_tile_data(),
            'dip_act': getattr(self, 'last_dip_str', "None"),
            'eco_act': getattr(self, 'last_eco_str', "None"),
            'history': self.history[-5:],  # Show last 5 actions in history
            
        }

        # Call the external renderer
        return self.renderer.render_frame(self.render_mode, state_data)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None
            self.font = None
