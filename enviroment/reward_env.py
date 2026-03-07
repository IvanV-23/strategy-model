import numpy as np

from  enviroment.enviroment_blocks import player_env
from  enviroment.enviroment_blocks import board_env

class RewardEnv:
    def __init__(self, player_env: player_env.PlayerEnv, board_env: board_env.BoardEnv):
        self.player = player_env
        self.board = board_env
            
    def calculate_player_resources_reward(self, game_turn=0, model_observations=None, player_resources_result=None):
        #Turn status
        owned_tiles=self.board.get_owned_tiles(owner_id=1)
        trade_routes=self.board.p1_trade_manager.active_routes
        
        # Building counts
        mines_lv1 = self.board.p1_buildings_manager.get_mines_by_owner_and_level(player_id=1, mine_level=1)
        mines_lv2 = self.board.p1_buildings_manager.get_mines_by_owner_and_level(player_id=1, mine_level=4)
        warehouses = self.board.p1_buildings_manager.get_mines_by_owner_and_level(player_id=1, mine_level=2)
        

        reward =  (self.player._resources[2]*0.2 + self.player._resources[3]*0.2 + len(trade_routes)*2 + owned_tiles*0.5)
        
        #Income rewards

        reward += self.player.gold_raw_income*0.3
        #reward += self.player.gold_net_income*0.1
        reward += self.player.wood_raw_income*0.3
        reward += self.player.food_raw_income*0.3

        # Additional rewards for buildings
        reward += mines_lv1 * 0.5 + mines_lv2 * 1.5 + warehouses * 0.8

        potencial_mines = model_observations["board_stats"][6]

        #reward -= potencial_mines*0.02

        potencial_trade_routes = model_observations["board_stats"][7]

        #reward -= potencial_trade_routes*0.02

        reward += self.player.capacity[0]*0.02
        reward += self.player.capacity[1]*0.02

        #reward -= player_resources_result["lost_gold"]*0.001
        #reward -= player_resources_result["lost_wood"]*0.001

        

        return reward

               
