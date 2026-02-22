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
         
    def calculate_player_resources(self):
        #Turn status
        owned_tiles=self.board.get_owned_tiles(owner_id=1)
        wood_production = self.board.collect_wood_income(player_id=1)
        routes_gold_income = self.board.collect_gold_income(player_id=1)


        #Gold calculation
        gold_income = 1 + owned_tiles + routes_gold_income
        gold_expenses = self.player._resources[2] * 0.005
        self.player.gold_net_income = gold_income - gold_expenses
        self.player._resources[0] += self.player.gold_net_income

        
        print(f"Gold expenses {gold_expenses}")
        print(f"Gold gold_income {gold_income}")
        
        #Wood calculation
        self.player.wood_raw_income = 1 + wood_production + self.player._resources[3] * 2

        self.player._resources[1] +=  self.player.wood_raw_income

        return {"status":"ok"}
       