import numpy as np

from  enviroment.enviroment_blocks import player_env
from  enviroment.enviroment_blocks import board_env

class RewardEnv:
    def __init__(self, player_env: player_env.PlayerEnv, board_env: board_env.BoardEnv):
        self.player = player_env
        self.board = board_env
            
    def calculate_player_resources_reward(self, game_turn=0, model_observations=None, player_resources_result=None):
        # 1. Economic Stability (Small constant for staying alive)
        reward = 0.1

        # 2. Territorial Control (Current reward)
        p1_tiles = self.board.get_owned_tiles(owner_id=1)
        p2_tiles = self.board.get_owned_tiles(owner_id=2)
        reward += (p1_tiles - p2_tiles) * 0.05
        
        # 3. MILITARY INCENTIVE (The "Hunting" reward)
        # Reward for defeating enemy workers (Economic Sabotage)
        defeated_workers = player_resources_result.get("defeated_workers", 0)
        reward += defeated_workers * 0.5

        # Reward for damage to enemy units/buildings
        damage_dealt = player_resources_result.get("damage_dealt_to_opponent", 0.0)
        reward += damage_dealt * 0.01 # Scaled down from 2.0 because damage can be large (200 per soldier)

        # 4. DISINCENTIVE (The "Logistics" cost)
        # Penalty for losing your own units (Prevents Kamikaze)
        lost_soldiers = player_resources_result.get("lost_soldiers", 0)
        reward -= lost_soldiers * 0.3

        # --- Legacy / Economic Rewards ---
        trade_routes = self.board.p1_trade_manager.active_routes
        mines_lv1 = self.board.p1_buildings_manager.get_mines_by_owner_and_level(player_id=1, mine_level=1)
        mines_lv2 = self.board.p1_buildings_manager.get_mines_by_owner_and_level(player_id=1, mine_level=4)
        
        gold_inc = np.clip(self.player.gold_raw_income, 0, 10000)
        wood_inc = np.clip(self.player.wood_raw_income, 0, 10000)
        food_inc = np.clip(self.player.food_raw_income, 0, 10000)

        reward += len(trade_routes) * 0.1
        reward += np.log1p(gold_inc) * 0.1
        reward += np.log1p(wood_inc) * 0.1
        reward += np.log1p(food_inc) * 0.1
        reward += mines_lv1 * 0.05 + mines_lv2 * 0.1

        # --- FINAL STABILITY CLIP ---
        return np.clip(reward, -20.0, 20.0)
