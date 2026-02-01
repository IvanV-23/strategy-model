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
            self._resources[0] += self._resources[3] * 2         
            self._resources[0] += player_tiles_count * 1   

            self._resources[0] -= self._resources[2]*0.01

            print(f"DEBUG: Player has {player_tiles_count} tiles, Opponent has {opponent_tiles_count} tiles.")
    
            #Resource rewards
            
            if opponent_tiles_count > player_tiles_count:
                reward -= 5.0
                return reward
            
            reward =  self._resources[0]*0.01 + self._resources[3]*0.1 +(player_tiles_count*2) - self._resources[1]*0.01 - (opponent_tiles_count*5)
            
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
        """Performs an attack action."""
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
    
    def invest(self):
        """Invests gold to gain wood."""
        reward = 0.0
        if self._resources[0] >= 20:
            self._resources[0] -= 20
            self._resources[1] += 10 
            reward += 2.0
        else:
            reward -= 0.5 # Penalty for insufficient funds
        return reward
    def create_units(self):
        """Creates soldiers using wood"""
        reward = 0.0
        if self._resources[1] >= 10:
            self._resources[1] -= 10
            self._resources[2] += 5 
            reward += 1.5
        else:
            reward -= 0.5 # Penalty for insufficient wood
        return reward
    def build_gold_getter(self) -> float:
        cost_gold = 50
        cost_wood = 20
        if self.resources[0] >= cost_gold and self.resources[1] >= cost_wood:
            self.resources[0] -= cost_gold
            self.resources[1] -= cost_wood
            self.resources[3] += 1 # Add building
            return 10.0 # Positive reward for successful investment
        else:
            return -5.0 # Penalty for trying to build without resources
    @property
    def resources(self) -> np.ndarray:
        return self._resources

    @resources.setter
    def resources(self, value):
        self._resources = np.maximum(value, 0)
