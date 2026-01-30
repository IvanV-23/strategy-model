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
            "player_resources": spaces.Box(low=0, high=1000, shape=(3,), dtype=np.int32),
            "opponent_strength": spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32),
            "turn_number": spaces.Discrete(100),
        })

        self.current_turn = 0
        self.player_resources = np.array([100, 50, 0], dtype=np.int32)
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
        self.player_resources = np.array([100, 50, 5], dtype=np.int32)
        self.opponent_strength = np.array([50], dtype=np.int32)

        if self.render_mode == "human":
            self.render()

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
            initial_opponent_strength = self.opponent_strength[0]
            


            # 0. Action Extraction
            self.diplomacy_action = action["diplomacy"]
            self.economy_action = action["economy"]
            reward = 0.0
            terminated = False
            truncated = False

            # 1. Resource calculation 
            self.player_resources[0] += 5  # Passive gold income each turn
            self.player_resources[0] -= self.player_resources[2]*0.01
    
            #Resource rewards
            reward += self.player_resources[2]*0.1
            reward += self.player_resources[0]*0.5
            

            # 2. Store Actions for Rendering
            self.last_diplomacy_choice = action["diplomacy"]
            self.last_economy_choice = action["economy"]
            self.dip_labels = ["Trade", "Pass", "Attack"]
            self.eco_labels = ["Invest", "Create Units", "Idle"]
            self.last_dip_str = self.dip_labels[self.last_diplomacy_choice]
            self.last_eco_str = self.eco_labels[self.last_economy_choice]

            # 3. --- DIPLOMACY BRANCH ---
            if self.diplomacy_action == 0: # Trade
                # Gain Gold (index 0) and Wood (index 1)
                self.player_resources[0] += np.random.randint(5, 15)
                self.player_resources[1] += np.random.randint(2, 8)
                reward += 1.0
            elif self.diplomacy_action == 1: # Pass
                reward -= 0.1
            elif self.diplomacy_action == 2: # Attack (Modified for Soldiers)
                army_size = self.player_resources[2] # Index 2 is Soldiers
                
                #if army_size > self.opponent_strength[0]:
                if army_size > 5:
                    # Success: Damage opponent
                    self.opponent_strength[0] -= np.random.randint(10, 20)
                    # Some soldiers are lost in battle
                    self.player_resources[2] -= np.random.randint(1, 4)
                    # Gain resources from victory
                    self.player_resources[0] += np.random.randint(10, 20)
                    self.player_resources[1] += np.random.randint(5, 15) 
                    reward += 5.0
                else:
                    # Failure: Army wiped out, lose gold penalty
                    self.player_resources[2] = 0
                    self.player_resources[0] -= np.random.randint(5, 10)
                    reward -= 2.0

            # 4. --- ECONOMY BRANCH ---
            if self.economy_action == 0: # Invest (Gold -> Wood)
                if self.player_resources[0] >= 20:
                    self.player_resources[0] -= 20
                    self.player_resources[1] += 10 
                    reward += 2.0
                else:
                    reward -= 0.5 # Penalty for insufficient funds
                    
            elif self.economy_action == 1: # Create Units (Wood -> Soldiers)
                if self.player_resources[1] >= 10:
                    self.player_resources[1] -= 10
                    self.player_resources[2] += 5 
                    reward += 1.5
                else:
                    reward -= 0.5 # Penalty for insufficient wood
                    
            elif self.economy_action == 2: # Idle
                pass
                #self.player_resources[0] += 1 # Passive gold gain
                #reward += 0.05

            # 5. Turn and Resource Management
            self.current_turn += 1
            
            # Clip resources at 0 to prevent negative values
            self.player_resources[:] = np.maximum(self.player_resources, 0)
            self.opponent_strength[:] = np.maximum(self.opponent_strength, 0)

            # 6. Opponent Status
            reward -= self.opponent_strength[0] * 0.1 # Penalty for opponent strength

            # 7. Opponent Action
            if self.current_turn % 10 == 0:
                # Opponent attacks every 10 turns
                attack_strength = self.opponent_strength[0] * (self.current_turn / 100) # Increasing strength over time
                if self.player_resources[2] >= attack_strength:
                    # Successfully defend
                    self.player_resources[2] -= attack_strength // 2 # Lose some soldiers
                    self.opponent_strength[0] -= self.player_resources[2] // 2 # Opponent weakens
                    reward += 5.0       
                else:
                    if self.player_resources[2] == 0:
                        reward -= self.opponent_strength[0] * 2
                        truncated = True

                        print(f"Player defeated! {reward} reward.")
                        # Failed to defend
                    self.player_resources[0] -= 10 # Lose gold
                    self.player_resources[1] -= 5  # Lose wood
                    self.player_resources[2] = 0   # Lose all soldiers
                    reward -= 10.0

            # 8. Termination Logic
            if self.opponent_strength[0] <= 0:
                reward += 100.0 # Victory reward
                terminated = True
                print(f"Opponent defeated! {reward} reward.")

            elif self.player_resources[0] <= 0 and self.player_resources[1] <= 0:
                # Bankrupt condition
                reward -= self.opponent_strength[0] * 2
                terminated = True
                print(f"Player bankrupt! {reward} reward.")

            #elif self.current_turn >= 100:
            #    reward -= 100.0
            #    truncated = True
            #    print("Maximum turns reached. Game over.")

            # 9. Rendering and Return
            if self.render_mode == "human":
                self.render()

            observation = self._get_obs()
            info = self._get_info()

            return observation, reward, terminated, truncated, info

    def render(self):

        # 1. Guard clause: do nothing if rendering is disabled
        if self.render_mode is None:
            return

        # 2. Initialize Pygame window only once
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.set_caption("Strategy Game - AI Agent")
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)

        # 3. Draw Background
        self.screen.fill((20, 20, 20)) 

        # Colors
        GOLD_COLOR = (255, 215, 0)
        WOOD_COLOR = (139, 69, 19)
        STRENGTH_COLOR = (255, 0, 0)
        TEXT_COLOR = (255, 255, 255)
        ACTION_COLOR = (0, 255, 255) # Cyan for actions
        SOLDIER_COLOR = (192, 192, 192) # Silver

        # 4. Display Player Resources (Top Left)
        gold_text = self.font.render(f"Gold: {self.player_resources[0]}", True, GOLD_COLOR)
        wood_text = self.font.render(f"Wood: {self.player_resources[1]}", True, WOOD_COLOR)
        soldier_text = self.font.render(f"Soldiers: {self.player_resources[2]}", True, SOLDIER_COLOR)
        self.screen.blit(gold_text, (50, 50))
        self.screen.blit(wood_text, (50, 90))
        self.screen.blit(soldier_text, (50, 130))

        # 5. Display Opponent Strength (Top Right)
        strength_str = f"Opponent Strength: {self.opponent_strength[0]}"
        strength_text = self.font.render(strength_str, True, STRENGTH_COLOR)
        self.screen.blit(strength_text, (self.screen_width - 350, 50))

        # 6. Display Turn Number (Center Top)
        turn_text = self.font.render(f"Turn: {self.current_turn}", True, TEXT_COLOR)
        self.screen.blit(turn_text, (self.screen_width // 2 - 50, 20))

        # 7. Display Last Actions (Bottom Left) ---
        # We use getattr in case the variables aren't initialized yet
        dip_act = getattr(self, 'last_dip_str', "None")
        eco_act = getattr(self, 'last_eco_str', "None")
        
        action_str = f"Last Action -> Diplomacy: {dip_act} | Economy: {eco_act}"
        last_actions_text = self.font.render(action_str, True, ACTION_COLOR)
        self.screen.blit(last_actions_text, (50, self.screen_height - 60))

        # 8. Output handling
        if self.render_mode == "human":
            pygame.display.flip()
            # Cap the frame rate
            self.clock.tick(self.metadata.get("render_fps", 30))
        
        elif self.render_mode == "rgb_array":
            # Convert pygame surface to a numpy array for recording/ML use
            import numpy as np
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
