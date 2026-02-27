import numpy as np



class PlayerEnv:
    def __init__(self):
        self.reset()
        self.SOLDIER_COST_GOLD = 10
        self.SOLDIER_COST_WOOD = 5
        self.MINE_COST_GOLD = 50
        self.MINE_COST_WOOD = 20
        #######Per turn incomes##################
        self.gold_net_income = 0
        self.gold_raw_income = 0
        self.wood_raw_income = 0
        #######Player capacity###################
        

    def reset(self, gold=100, wood=50, soldiers=5, gold_buildings=0):
        """Initializes player resources."""
        # Index 0: Gold, 1: Wood, 2: Soldiers, 3: Gold Buildings
        self._resources = np.array([gold, wood, soldiers, gold_buildings], dtype=np.int32)
        self._capacity = np.array([500, 500, 1000], dtype=np.int32)
        return self._resources

    
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
                    reward += 1 + player_tiles
                    print("Captured neutral tile.")
                else:
                    # Rewards for successfully raiding the opponent
                    self._resources[0] += 60
                    self._resources[1] += 30
                    reward += 1.5 * player_tiles
                    print(f"Captured enemy tile from player {previous_owner}!")

                if base_captured:
                    reward += 10.0 * player_tiles  # Massive game-winning bonus
                    
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
                    reward -= 0.01
                    # You might also lose a tiny bit of power for a failed siege
                    #self._resources[2] = max(0, self._resources[2] - 1) 

            return reward

    
    def process_economy(self, num_soldiers: int, num_mines: int) -> float:
        """
        Processes the construction of soldiers and mines based on model counts.
        num_soldiers: Count from MultiDiscrete[0]
        num_mines: Count from MultiDiscrete[1]
        """
        reward = 0.0
        
        # 1. Calculate Total Costs
        total_gold_needed = (num_soldiers * self.SOLDIER_COST_GOLD) + \
                            (num_mines * self.MINE_COST_GOLD)
        
        total_wood_needed = (num_soldiers * self.SOLDIER_COST_WOOD) + \
                            (num_mines * self.MINE_COST_WOOD)

        # 2. Check Affordability
        if num_soldiers == 0 and num_mines == 0:
            return 0.0  # Idle turn, no reward/penalty

        if self._resources[0] >= total_gold_needed and self._resources[1] >= total_wood_needed:
            # Success: Deduct and Build
            self._resources[0] -= total_gold_needed
            self._resources[1] -= total_wood_needed
            
            self._resources[2] += num_soldiers  # Index 2: Soldiers
            self._resources[3] += num_mines     # Index 3: Gold Buildings (Mines)
            
            # Small positive reward for successful production
            reward += (num_soldiers * 0.1) + (num_mines * 0.1)
            print(f"ECONOMY: Built {num_soldiers} soldiers and {num_mines} mines.")
            
        else:
            # Failure: Insufficient funds
            # Penalty scales with how much they overspent to discourage "hallucinating" money
            reward -= 0.5 
            print("ECONOMY: Insufficient resources to fulfill build order!")
            
        return reward
    
    @property
    def resources(self) -> np.ndarray:
        return self._resources

    @resources.setter
    def resources(self, value):
        self._resources = np.maximum(value, 0)

    @property
    def capacity(self) -> np.ndarray:
        return self._capacity
    
    @capacity.setter
    def capacity(self, value):
        self._capacity = np.maximum(value, 0)
