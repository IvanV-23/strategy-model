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

    def action_step(self, current_turn: int, player_status: dict):
        """ Takes an action for the opponent. """
        reward = 0.0
        truncated = False

        # Update resources
        self._resources[0] += 5
        self._resources[1] += 1
        self._resources[2] += 0

        if current_turn % 5 == 0:
            # Opponent builds soldiers every 5 turns
            if self._resources[0] >= 10 and self._resources[1] >= 5:
                self._resources[0] -= 10  # Spend gold
                self._resources[1] -= 5   # Spend wood
                self._resources[2] += 2  # Gain soldiers

        # Ensure resources do not go negative
        self._resources = np.maximum(self._resources, 0)

        if current_turn % 10 == 0:
            # Opponent attacks every 10 turns
            attack_strength = self._resources[2] * (current_turn / 100) # Increasing strength over time
            if player_status["soldiers"] >= attack_strength:
                # Successfully defend
                player_status["soldiers"] -= attack_strength // 2 # Lose some soldiers
                self._resources[2] -= player_status["soldiers"] // 2 # Opponent weakens
                reward += 5.0       
            else:
                if player_status["soldiers"] == 0:
                    reward -= (player_status["gold"] + player_status["wood"]) * 0.1
                    truncated = True

                    
                # Failed to defend
                player_status["gold"] -= 10 # Lose gold
                player_status["wood"] -= 5  # Lose wood
                player_status["soldiers"] = 0   # Lose all soldiers
                reward -= 10.0

        return reward, player_status, truncated

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
