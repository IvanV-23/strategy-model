from enviroment.enviroment_blocks.opponent_env import OpponentEnv
from enviroment.enviroment_blocks.player_env import PlayerEnv
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

        #Initialize enviroment blocks
        self.opponent_env = OpponentEnv()
        self.player_env = PlayerEnv()


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
            "player_resources": spaces.Box(low=0, high=1000, shape=(3,), dtype=np.int32),
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

    def reset(self, seed: int = None, options: Dict = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_turn = 0
        self.player_env.reset()
        
        self.opponent_env.reset()


        if self.render_mode == "human":
            self.render()

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:

            
            # 0. Action Extraction
            self.diplomacy_action = action["diplomacy"]
            self.economy_action = action["economy"]
            reward = 0.0
            terminated = False
            truncated = False

            # 1. Resource calculation 
            self.player_env.resources[0] += 5  # Passive gold income each turn
            self.player_env.resources[0] -= self.player_env.resources[2]*0.01
    
            #Resource rewards
            reward += self.player_env.resources[2]*0.1
            reward += self.player_env.resources[0]*0.5
            

            # 2. Store Actions for Rendering
            self.last_diplomacy_choice = action["diplomacy"]
            self.last_economy_choice = action["economy"]
            self.dip_labels = ["Trade", "Pass", "Attack"]
            self.eco_labels = ["Invest", "Create Units", "Idle"]
            self.last_dip_str = self.dip_labels[self.last_diplomacy_choice]
            self.last_eco_str = self.eco_labels[self.last_economy_choice]

            # 3. --- DIPLOMACY BRANCH ---
            if self.diplomacy_action == 0: # Trade
                reward += self.player_env.trade()
            elif self.diplomacy_action == 1: # Pass
                pass
            elif self.diplomacy_action == 2: # Attack (Modified for Soldiers)
                attack_reward, updated_opponent_status = self.player_env.attack(
                    opponent_status={
                        "gold": self.opponent_env.resources[0],
                        "wood": self.opponent_env.resources[1],
                        "soldiers": self.opponent_env.resources[2]
                    }
                )
                reward += attack_reward
                self.opponent_env.resources[0] = updated_opponent_status["gold"]
                self.opponent_env.resources[1] = updated_opponent_status["wood"]
                self.opponent_env.resources[2] = updated_opponent_status["soldiers"]

            # 4. --- ECONOMY BRANCH ---
            if self.economy_action == 0: # Invest (Gold -> Wood)
                reward += self.player_env.invest()
                    
            elif self.economy_action == 1: # Create Units (Wood -> Soldiers)
                reward += self.player_env.create_units()
                    
            elif self.economy_action == 2: # Idle
                pass

            # 5. Turn and Resource Management
            self.current_turn += 1
            
            # 6. Opponent Status
            reward -= self.opponent_env.resources[2] * 0.1 # Penalty for opponent strength

            # 7. Opponent Action
            opp_reward, updated_player_status, opp_truncated = self.opponent_env.action_step(
                current_turn=self.current_turn, 
                player_status={
                    "gold": self.player_env.resources[0],
                    "wood": self.player_env.resources[1],
                    "soldiers": self.player_env.resources[2]    
                }
            )
            reward += opp_reward
            self.player_env.resources[0] = updated_player_status["gold"]
            self.player_env.resources[1] = updated_player_status["wood"]
            self.player_env.resources[2] = updated_player_status["soldiers"]
            if opp_truncated:
                truncated = True
                print(f"Player defeated! {reward} reward.")


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

            # 2. Initialize Pygame window only once
            if self.screen is None and self.render_mode == "human":
                pygame.init()
                pygame.display.set_caption("Strategy Game - AI Agent")
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
                self.clock = pygame.time.Clock()
                self.font = pygame.font.Font(None, 32) # Slightly smaller for more info

            # 3. Draw Background
            self.screen.fill((20, 20, 25)) # Deep dark blue-grey

            # Colors
            GOLD_COLOR = (255, 215, 0)
            WOOD_COLOR = (139, 69, 19)
            STRENGTH_COLOR = (255, 80, 80) # Softer red
            TEXT_COLOR = (240, 240, 240)
            ACTION_COLOR = (0, 255, 255)
            SOLDIER_COLOR = (200, 200, 200)

            # --- 4. PLAYER SECTION (LEFT) ---
            p_res = self.player_env.resources
            player_header = self.font.render("--- PLAYER ---", True, TEXT_COLOR)
            self.screen.blit(player_header, (50, 50))
            
            self.screen.blit(self.font.render(f"Gold: {p_res[0]}", True, GOLD_COLOR), (50, 90))
            self.screen.blit(self.font.render(f"Wood: {p_res[1]}", True, WOOD_COLOR), (50, 130))
            self.screen.blit(self.font.render(f"Soldiers: {p_res[2]}", True, SOLDIER_COLOR), (50, 170))

            # --- 5. OPPONENT SECTION (RIGHT) ---
            # Accessing the new array from the Block
            o_res = self.opponent_env.resources 
            opp_header = self.font.render("--- OPPONENT ---", True, STRENGTH_COLOR)
            self.screen.blit(opp_header, (self.screen_width - 300, 50))

            self.screen.blit(self.font.render(f"Gold: {o_res[0]}", True, GOLD_COLOR), (self.screen_width - 300, 90))
            self.screen.blit(self.font.render(f"Wood: {o_res[1]}", True, WOOD_COLOR), (self.screen_width - 300, 130))
            self.screen.blit(self.font.render(f"Soldiers: {o_res[2]}", True, STRENGTH_COLOR), (self.screen_width - 300, 170))

            # 6. Turn Number (Center Top)
            turn_text = self.font.render(f"TURN: {self.current_turn}", True, TEXT_COLOR)
            self.screen.blit(turn_text, (self.screen_width // 2 - 50, 20))

            # 7. Last Actions (Bottom Center)
            dip_act = getattr(self, 'last_dip_str', "None")
            eco_act = getattr(self, 'last_eco_str', "None")
            action_str = f"DECISIONS >> Diplomacy: {dip_act} | Economy: {eco_act}"
            last_actions_text = self.font.render(action_str, True, ACTION_COLOR)
            # Centering the action text at the bottom
            text_rect = last_actions_text.get_rect(center=(self.screen_width // 2, self.screen_height - 50))
            self.screen.blit(last_actions_text, text_rect)

            # 8. Output handling
            if self.render_mode == "human":
                pygame.display.flip()
                self.clock.tick(self.metadata.get("render_fps", 4))
            elif self.render_mode == "rgb_array":
                return np.transpose(
                    np.array(pygame.surfarray.pixels3d(self.screen)), 
                    axes=(1, 0, 2)
                )


    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None
            self.font = None
