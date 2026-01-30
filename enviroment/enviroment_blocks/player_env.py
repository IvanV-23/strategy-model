import numpy as np

class PlayerEnv:
    def __init__(self):
        self.reset()

    def reset(self, gold=100, wood=50, soldiers=5):
        """Initializes player resources."""
        # Index 0: Gold, 1: Wood, 2: Soldiers
        self._resources = np.array([gold, wood, soldiers], dtype=np.int32)
        return self._resources
    
    def trade(self):
        """Performs a trade action to gain resources."""
        reward = 0.0

        self._resources[0] += np.random.randint(5, 15)
        self._resources[1] += np.random.randint(2, 8)
        reward += 1.0
        return reward
    
    def attack(self, opponent_status:dict):
        """Performs an attack action."""
        reward = 0.0
        army_size = self._resources[2] # Index 2 is Soldiers
        
        #if army_size > self.opponent_strength[0]:
        if army_size > opponent_status["soldiers"]:
            # Some soldiers are lost in battle
            self._resources[2] -= np.random.randint(1, 4)
            # Success: Damage opponent
            opponent_status["soldiers"] -= self._resources[2]

            # Gain resources from victory
            self._resources[0] += np.random.randint(10, 20)
            self._resources[1] += np.random.randint(5, 15)

            # Opponent lost resources on defeat 
            opponent_status["gold"] -= np.random.randint(10, 20)
            opponent_status["wood"] += np.random.randint(5, 15)

            reward += 5.0
        else:
            # Failure: Army wiped out, lose gold penalty
            self._resources[2] = 0
            reward -= 2.0
        return reward, opponent_status
    
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
    @property
    def resources(self) -> np.ndarray:
        return self._resources

    @resources.setter
    def resources(self, value):
        self._resources = np.maximum(value, 0)