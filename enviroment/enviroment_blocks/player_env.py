import numpy as np

class PlayerEnv:
    def __init__(self):
        self.reset()

    def reset(self, gold=100, wood=50, soldiers=5, gold_buildings=0):
        """Initializes player resources."""
        # Index 0: Gold, 1: Wood, 2: Soldiers, 3: Gold Buildings
        self._resources = np.array([gold, wood, soldiers, gold_buildings], dtype=np.int32)
        return self._resources
    def resource_calculation(self, board_grid):
            reward = 0.0
            # Get owned tiles
            player_tiles_count = np.sum(board_grid[:, :, 0] == 1)
            opponent_tiles_count = np.sum(board_grid[:, :, 0] == 2)

            #Gold calculation
            self._resources[0] += 1 
            #self._resources[0] += self._resources[3] * 2         
            self._resources[0] += player_tiles_count * 2  

            #self._resources[0] -= self._resources[2]*0.01

            #Wood calculation

            self._resources[1] += self._resources[3] * 2  
            print(f"DEBUG: Player has {player_tiles_count} tiles, Opponent has {opponent_tiles_count} tiles.")
    
            #Resource rewards
            
            if opponent_tiles_count > player_tiles_count:
                reward -= 0.1 * (opponent_tiles_count - player_tiles_count)
                return reward
            
            reward =  self._resources[2]*0.1 + self._resources[3]*0.1 + (player_tiles_count) 
            
            return reward
    def trade(self):
        """Performs a trade action to gain resources."""
        reward = 0.0
        if self._resources[0] >= 5:
            self._resources[0] -= 5
        self._resources[1] += 10
        reward += 0.0
        return reward
    
    def attack(self, opponent_status:dict):
        """DEPRECATED. Performs an attack action."""
        reward = 0.0
        army_size = self._resources[2] # Index 2 is Soldiers
        battle_victory = False
        
        #if army_size > self.opponent_strength[0]:
        if army_size > opponent_status["defense"]:
            # Some soldiers are lost in battle
            #self._resources[2] -= np.random.randint(1, 4)
            # Success: Damage opponent
            opponent_status["soldiers"] -= self._resources[2]

            # Gain resources from victory
            self._resources[0] += 100
            self._resources[1] += 50

            # Opponent lost resources on defeat 
            opponent_status["gold"] -= 100
            opponent_status["wood"] -= 50

            battle_victory = True
            reward += 10.0
        else:
            # Failure: Army wiped out, lose gold penalty
            self._resources[2] = self._resources[2] - 1
            reward -= 2.0
        return reward, opponent_status, battle_victory
    
    def process_battle_consequences(self, victory, base_captured, previous_owner, reason, player_tiles):
            """
            Updates internal resources based on targeted battle results and failure reasons.
            previous_owner: 0 for neutral, 2 for opponent (assuming current player is 1)
            reason: "success", "out_of_bounds", "already_owned", "not_adjacent", "insufficient_force"
            """
            reward = 0.0
            
            if victory:
                if previous_owner == 0:
                    # Rewards for expanding into neutral territory
                    self._resources[0] += 20
                    self._resources[1] += 10
                    reward += 0.5 * player_tiles
                    print("Captured neutral tile.")
                else:
                    # Rewards for successfully raiding the opponent
                    self._resources[0] += 60
                    self._resources[1] += 30
                    reward += 1.5 * player_tiles
                    print(f"Captured enemy tile from player {previous_owner}!")

                if base_captured:
                    reward += 10.0  # Massive game-winning bonus
                    
            else:
                print(f"Player attack failed! Reason: {reason}")
                # --- Logic for handling failures based on the reason ---
                if reason == "not_adjacent":
                    # The agent tried to "teleport" or jump across the map
                    reward -= 0.2
                    
                elif reason == "already_owned":
                    # Waste of an action attacking its own territory
                    reward -= 0.1
                    
                elif reason == "out_of_bounds":
                    # Trying to click off the map
                    reward -= 0.5  # Heavy penalty for invalid input logic
                    
                elif reason == "insufficient_force":
                    # Valid move, just not strong enough. 
                    # Small penalty to encourage building up power first.
                    reward -= 0.05
                    # You might also lose a tiny bit of power for a failed siege
                    self._resources[2] = max(0, self._resources[2] - 1) 

            return reward

    def invest(self):
        """Invests gold to gain wood."""
        reward = 0.0
        if self._resources[0] >= 20:
            self._resources[0] -= 20
            self._resources[1] += 10 
            reward += 0.01
        else:
            reward -= 0.5 # Penalty for insufficient funds
        return reward
    def create_units(self):
        """Creates soldiers using wood"""
        reward = 0.0
        if self._resources[1] >= 10:
            self._resources[1] -= 10
            self._resources[2] += 5 
            reward += 0.1
        else:
            reward -= 0.1 # Penalty for insufficient wood
        return reward
    def build_gold_getter(self) -> float:
        reward = 0.0
        cost_gold = 50
        cost_wood = 00
        if self.resources[0] >= cost_gold and self.resources[1] >= cost_wood:
            self.resources[0] -= cost_gold
            self.resources[1] -= cost_wood
            self.resources[3] += 1 # Add building
            return 0.1 # Positive reward for successful investment
        else:
            reward -= 0.5 
            return reward # Penalty for trying to build without resources
    @property
    def resources(self) -> np.ndarray:
        return self._resources

    @resources.setter
    def resources(self, value):
        self._resources = np.maximum(value, 0)
