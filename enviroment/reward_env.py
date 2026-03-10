import numpy as np

from  enviroment.enviroment_blocks import player_env
from  enviroment.enviroment_blocks import board_env

class RewardEnv:
    def __init__(self, player_env: player_env.PlayerEnv, board_env: board_env.BoardEnv):
        self.player = player_env
        self.board = board_env
            
    def calculate_player_resources_reward(self, game_turn=0, model_observations=None, player_resources_result=None):
        # 1. Tile & Infrastructure Status
        owned_tiles = self.board.get_owned_tiles(owner_id=1)
        trade_routes = self.board.p1_trade_manager.active_routes
        
        # 2. Building counts
        mines_lv1 = self.board.p1_buildings_manager.get_mines_by_owner_and_level(player_id=1, mine_level=1)
        mines_lv2 = self.board.p1_buildings_manager.get_mines_by_owner_and_level(player_id=1, mine_level=4)
        warehouses = self.board.p1_buildings_manager.get_mines_by_owner_and_level(player_id=1, mine_level=2)
        
        # --- RESOURCE CLIPPING ---
        # Clip incomes to a very safe maximum for reward calculation
        gold_inc = np.clip(self.player.gold_raw_income, 0, 10000)
        wood_inc = np.clip(self.player.wood_raw_income, 0, 10000)
        food_inc = np.clip(self.player.food_raw_income, 0, 10000)

        # --- SCALED DOWN REWARDS ---
        defeated_workers = player_resources_result.get("defeated_workers", 0)
        reward = (self.player._resources[3] * 0.05 + len(trade_routes) * 0.5 + owned_tiles * 0.1 + defeated_workers * 0.05)
        
        # --- LOGARITHMIC INCOME REWARDS ---
        reward += np.log1p(gold_inc) * 0.5
        reward += np.log1p(wood_inc) * 0.5
        reward += np.log1p(food_inc) * 0.5

        # 3. Building Milestone Rewards
        reward += mines_lv1 * 0.2 + mines_lv2 * 0.5 + warehouses * 0.2

        # 4. Capacity Rewards
        # Dramatically reduced from 0.02 to 0.001 to prevent warehouse-spam reward explosion
        reward += self.player.capacity[0] * 0.001
        reward += self.player.capacity[1] * 0.001

        # --- FINAL STABILITY CLIP ---
        # Ensures no single turn can provide more than 20 reward points,
        # keeping the total episode reward within a range the neural network can learn from.
        return np.clip(reward, -20.0, 20.0)
