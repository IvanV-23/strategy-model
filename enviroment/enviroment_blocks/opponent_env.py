import numpy as np

class OpponentEnv:
    def __init__(self):
        # Index 0: Gold, 1: Wood, 2: Soldiers
        self._resources = np.array([100, 50, 10], dtype=np.int32)
    
    def reset(self, gold=100, wood=50, soldiers=10):
            """Restores the opponent to starting resources."""
            # Index 0: Gold, 1: Wood, 2: Soldiers
            self._resources = np.array([gold, wood, soldiers], dtype=np.int32)
            return self.resources

    def action_step(self, current_turn: int, owned_tiles: int):
            """ 
            Decides opponent strategy. 
            Returns: (intent_to_attack, updated_resources)
            """
            # Passive resource income
            self._resources[0] += 1 + owned_tiles * 2
            self._resources[1] += 1 + owned_tiles 
            
            # 1. Economic Logic: Build soldiers if possible every 5 turns
            if current_turn % 1 == 0:
                can_afford_gold = self._resources[0] // 20
                can_afford_wood = self._resources[1] // 10
            
                num_to_build = int(min(can_afford_gold, can_afford_wood))

                if num_to_build > 0:
                    self._resources[0] -= 20 * num_to_build
                    self._resources[1] -= 10 * num_to_build
                    self._resources[2] += num_to_build  # Gain a batch of soldiers
            
            # 2. Strategic Intent: Decide to attack every 10 turns
            intent_to_attack = False
            if current_turn % 10 == 0 and self._resources[2] > 0:
                intent_to_attack = True

            if self._resources[0] > 1000:
                 #trade
                self._resources[0] -= 10
                self._resources[1] += 20

            # Ensure resources stay non-negative
            self._resources = np.maximum(self._resources, 0)
            
            return intent_to_attack
    
    def score_calculation(self):
            #Score based on resources
            score = self._resources[0]*0.5 + self._resources[2]*0.2
            return score
    @property
    def resources(self) -> np.ndarray:
        return self._resources

    @resources.setter
    def resources(self, value):
        self._resources = np.maximum(value, 0)

    @property
    def strength(self) -> int:
        """Still return soldiers as 'strength' for legacy logic."""
        return int(self._resources[2])
