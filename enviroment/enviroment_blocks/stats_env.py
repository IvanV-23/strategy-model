import numpy as np

from  enviroment.enviroment_blocks import player_env
from  enviroment.enviroment_blocks import board_env

class StatsEnv:
    def __init__(self, player_env: player_env.PlayerEnv, board_env: board_env.BoardEnv):
        self.player = player_env
        self.board = board_env
    
    def get_game_stats(self)->dict:
        
        board_stats = self.board.get_board_state_and_stats()["stats"]

        global_stats = np.append(board_stats, self.player.gold_net_income).astype(np.float32)

        return global_stats
         

       
