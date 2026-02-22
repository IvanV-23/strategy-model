import numpy as np

from  enviroment.enviroment_blocks import player_env
from  enviroment.enviroment_blocks import board_env

class RewardEnv:
    def __init__(self, player_env: player_env.PlayerEnv, board_env: board_env.BoardEnv):
        self.player = player_env
        self.board = board_env
            
    def calculate_player_resources_reward(self, game_turn=0):
        #Turn status
        owned_tiles=self.board.get_owned_tiles(owner_id=1)
        trade_routes=self.board.p1_trade_manager.active_routes
        

        reward =  (self.player._resources[2]*0.2 + self.player._resources[3]*0.2 + self.player.gold_raw_income*0.3 + self.player.wood_raw_income*0.3 + self.player.gold_net_income*0.1 + len(trade_routes)*2 + owned_tiles*0.5 - (self.player.resources[1]*0.02*game_turn/1000))
        
        return reward

               
