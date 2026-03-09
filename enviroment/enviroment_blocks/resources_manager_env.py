import numpy as np

from enviroment.enviroment_blocks import player_env
from enviroment.enviroment_blocks import board_env

class ResourceManager:
    def __init__(self, board_env:board_env.BoardEnv, player_env:player_env.PlayerEnv):
        self.board_env = board_env
        self.player_env = player_env

    def _get_gold_income(self, player_id):
        """
            Calculate gold income based on owned tiles and trade routes.
        """
        owned_tiles = self.board_env.get_owned_tiles(owner_id=player_id)
        trade_route_income = self.board_env.collect_gold_income(player_id=player_id)
        workers_taxes = self.player_env._resources[2] * 0.001  # 0.001 gold per worker as tax
        
        return 1 + owned_tiles + trade_route_income + workers_taxes
    
    def _get_gold_expenses(self):
        """
            Calculate gold expenses based on workers and soldiers.
        """
        #worker_upkeep = self.player_env._resources[2] * 0.005
        # Soldier maintenance: 0.1 gold per soldier
        soldier_count = self.board_env.get_soldier_count(player_id=1)
        soldier_upkeep = soldier_count 
        return soldier_upkeep

    def _get_wood_income(self, player_id):

        tiles_wood_income = self.board_env.collect_wood_income(player_id=player_id)

        # Mine Lvl 1 (status 1) gives 2, Mine Lvl 2 (status 4) gives 4
        mines_l1 = self.board_env.p1_buildings_manager.get_building_count_by_type(player_id, 1)
        mines_l2 = self.board_env.p1_buildings_manager.get_building_count_by_type(player_id, 4)
        
        mines_wood_income = mines_l1 * 2 + mines_l2 * 4
        
        return 1 + tiles_wood_income + mines_wood_income


    def collect_resources(self, player_id):
        """
        Collects resources from all owned tiles and updates the player's resource pool.
        """
        lost_gold = 0
        lost_wood = 0
        lost_food = 0

        gold_income = self._get_gold_income(player_id=player_id)
        gold_expenses = self._get_gold_expenses()
        self.player_env.gold_net_income = gold_income - gold_expenses

        current_gold = self.player_env._resources[0]
        potential_total_gold = current_gold + self.player_env.gold_net_income
        
        # --- SOLDIER MAINTENANCE LOGIC ---
        if potential_total_gold < 0:
            # Player is bankrupt and can't pay maintenance
            shortfall = abs(potential_total_gold)
            # Each soldier dying "saves" 0.1 gold maintenance this turn? 
            # Or just remove soldiers because they couldn't be paid.
            # Let's say for every 0.1 gold short, 1 soldier dies.
            soldiers_to_die = int(shortfall / 0.1) + 1
            actual_soldiers = self.board_env.get_soldier_count(player_id=1)
            soldiers_to_die = min(actual_soldiers, soldiers_to_die)
            
            if soldiers_to_die > 0:
                print(f"BANKRUPTCY: {soldiers_to_die} soldiers died due to lack of gold!")
                self.board_env.remove_soldiers(player_id=1, count=soldiers_to_die)
            
            # After some soldiers die, we still clip gold to 0
            self.player_env._resources[0] = 0
        else:
            if potential_total_gold > self.player_env.capacity[0]:
                lost_gold = potential_total_gold - self.player_env.capacity[0]
                self.player_env._resources[0] = self.player_env.capacity[0]
            else:
                self.player_env._resources[0] = potential_total_gold

        self.player_env.wood_raw_income = self._get_wood_income(player_id=player_id)

        total_wood = self.player_env._resources[1] + self.player_env.wood_raw_income
        if total_wood > self.player_env.capacity[1]:
            lost_wood = total_wood - self.player_env.capacity[1]
            self.player_env._resources[1] = self.player_env.capacity[1]
        else:
            self.player_env._resources[1] = total_wood

        # Food Collection
        crops_l1 = self.board_env.p1_buildings_manager.get_building_count_by_type(player_id, 5)
        crops_l2 = self.board_env.p1_buildings_manager.get_building_count_by_type(player_id, 7)
        
        self.player_env.food_raw_income =  (crops_l1 * 500) + (crops_l2 * 1000)
        
        total_food = self.player_env._resources[4] + self.player_env.food_raw_income

        self.player_env.food_net_income = self.player_env.food_raw_income - self.player_env._resources[2] * 0.5

        if total_food > self.player_env.capacity[3]:
            lost_food = total_food - self.player_env.capacity[3]
            self.player_env._resources[4] = self.player_env.capacity[3]
        else:
            self.player_env._resources[4] = total_food

        return {"status":"ok", "lost_gold": lost_gold, "lost_wood": lost_wood, "lost_food": lost_food, "lost_soldiers": soldiers_to_die if potential_total_gold < 0 else 0}
