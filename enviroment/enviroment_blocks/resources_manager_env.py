
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
        
        return 1 + owned_tiles + trade_route_income
    
    def _get_gold_expenses(self):
        """
            Calculate gold expenses based on soldiers and other upkeep costs.
        """
        soldier_upkeep = self.player_env._resources[2] * 0.005
        return soldier_upkeep

    def _get_wood_income(self, player_id):

        tiles_wood_income = self.board_env.collect_wood_income(player_id=player_id)

        mines_wood_income = self.board_env.p1_buildings_manager.get_mines_by_owner_and_level(player_id=1, mine_level=1) * 2 + self.board_env.p1_buildings_manager.get_mines_by_owner_and_level(player_id=1, mine_level=2) * 4
        
        return 1 + tiles_wood_income + mines_wood_income

    def collect_resources(self, player_id):
        """
        Collects resources from all owned tiles and updates the player's resource pool.
        """
        lost_gold = 0
        lost_wood = 0

        gold_income = self._get_gold_income(player_id=player_id)
        gold_expenses = self._get_gold_expenses()
        self.player_env.gold_net_income = gold_income - gold_expenses

        #Add reosources while there is capacity, otherwise calculate lost resources
        self.player_env._resources[0] = min(self.player_env._resources[0]+self.player_env.gold_net_income,
                                        self.player_env.capacity[0])
        
        if self.player_env._resources[0]+self.player_env.gold_net_income > self.player_env.capacity[0]:
            lost_gold = self.player_env._resources[0]+self.player_env.gold_net_income - self.player_env.capacity[0]
            print(f"Lost Gold: {lost_gold}")

        self.player_env.wood_raw_income = self._get_wood_income(player_id=player_id)

        if self.player_env._resources[1]+self.player_env.wood_raw_income > self.player_env.capacity[1]:
            lost_wood = self.player_env._resources[1]+self.player_env.wood_raw_income - self.player_env.capacity[1]
            print(f"Lost Wood: {lost_wood}")

        self.player_env._resources[1] = min(self.player_env._resources[1]+self.player_env.wood_raw_income, self.player_env.capacity[1])

        return {"status":"ok", "lost_gold": lost_gold, "lost_wood": lost_wood}
