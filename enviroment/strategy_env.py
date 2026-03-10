from typing import Tuple, Dict, Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from enviroment.reward_env import RewardEnv
from enviroment.enviroment_blocks.resources_manager_env import ResourceManager
from enviroment.enviroment_blocks.opponent_env import OpponentEnv
from enviroment.enviroment_blocks.player_env import PlayerEnv
import os
from enviroment.enviroment_render.strategy_renderer import StrategyRenderer
from enviroment.enviroment_render.cpp_bridge import CppRendererBridge
from enviroment.enviroment_blocks.board_env import BoardEnv
from enviroment.enviroment_blocks.stats_env import StatsEnv
from enviroment.enviroment_branches.diplomacy_env import DiplomacyEnv
from enviroment.enviroment_branches.economy_env import EconomyEnv

class StrategyEnv(gym.Env):
    """
    A Multi-Agent Reinforcement Learning environment for a strategic game.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: str = None):
        super().__init__()

        self.history = ["Game Started"]

        #Initialize blocks
        self.opponent_env = OpponentEnv()
        self.player_env = PlayerEnv()
        self.board_env = BoardEnv()
        self.stats_env = StatsEnv(board_env=self.board_env, player_env=self.player_env)
        self.reward_env = RewardEnv(board_env=self.board_env, player_env=self.player_env)
        self.resource_manager = ResourceManager(board_env=self.board_env, player_env=self.player_env)

        #Initialize branches
        self.diplomacy_branch = DiplomacyEnv(board_env=self.board_env, player_env=self.player_env, opponent_env=self.opponent_env)
        self.economy_branch = EconomyEnv(board_env=self.board_env, player_env=self.player_env, opponent_env=self.opponent_env)
        
        #Initialize renderer
        self.screen_width = 800
        self.screen_height = 1000
        self.render_mode = render_mode

        if os.getenv("USE_CPP_RENDER") == "1" and render_mode == "human":
            print("Attempting to use C++ renderer...")
            try:
                self.renderer = CppRendererBridge(self.screen_width, self.screen_height, self.metadata)
                if self.renderer.proc is None:
                    self.renderer = StrategyRenderer(self.screen_width, self.screen_height, self.metadata)
            except Exception:
                self.renderer = StrategyRenderer(self.screen_width, self.screen_height, self.metadata)
        else:
            self.renderer = StrategyRenderer(self.screen_width, self.screen_height, self.metadata)

        # Action space: Economy now has 6 heads [Soldiers, Mines, Trade, Warehouse, Crops, Fortify]
        self.diplomacy_action_space = spaces.Discrete(3)
        self.economy_action_space = spaces.MultiDiscrete([10, 3, 2, 3, 3, 2])
        self.target_action_space = spaces.Discrete(256)
        
        self.action_space = spaces.Dict({
            "diplomacy": self.diplomacy_action_space,
            "economy": self.economy_action_space,
            "target_tile": self.target_action_space,
        })

        # Observation space: 8 channels
        self.observation_space = spaces.Dict({
            "player_resources": spaces.Box(low=0, high=1000, shape=(9,), dtype=np.float64),
            "opponent_resources": spaces.Box(low=0, high=1000, shape=(3,), dtype=np.int32), 
            "turn_number": spaces.Discrete(1000),
            "board_state": spaces.Box(low=0, high=255, shape=(8, 16, 16), dtype=np.int32),
            "board_stats": spaces.Box(low=0, high=1000, shape=(15,), dtype=np.float32),
        })

        self.current_turn = 0
        self.max_turns = 1000
        self.screen = None
        self.clock = None

    def _get_obs(self) -> Dict[str, np.ndarray]:
        return {
            "player_resources": np.append(self.player_env.resources, self.player_env.capacity),
            "opponent_resources": self.opponent_env.resources,
            "turn_number": self.current_turn,
            "board_state": self.board_env.full_board_state(),
            "board_stats": self.stats_env.get_game_stats(
                lost_gold=getattr(self, 'lost_gold', 0),
                lost_wood=getattr(self, 'lost_wood', 0),
                lost_food=getattr(self, 'lost_food', 0),
                defeated_workers=getattr(self, 'defeated_workers', 0),
                player_soldiers=self.board_env.get_soldier_count(player_id=1)
            )
        }

    def _get_info(self) -> Dict[str, Any]:
        return {
            "action_mask": self.board_env.get_action_mask(player_id=1),
            "build_mask": self.board_env.get_build_mask(player_id=1, player_gold=self.player_env.resources[0], player_wood=self.player_env.resources[1])
        }

    def _action_extraction(self, action: dict):
        self.diplomacy_action = action["diplomacy"]
        eco = action["economy"]
        self.soldiers_to_recruit = eco[0]
        self.mines_action = eco[1]
        self.trade_action = eco[2]
        self.warehouse_action = eco[3]
        self.crops_action = eco[4]
        self.fortify_action = eco[5]

        target_idx = action["target_tile"]
        self.target_row = target_idx // 16
        self.target_col = target_idx % 16

        return 0.0, False, False

    def reset(self, seed: int = None, options: Dict = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_turn = 0
        self.player_env.reset()
        self.opponent_env.reset()
        self.board_env.reset()
        self.history = ["Game Started"]
        if self.render_mode == "human": self.render()
        return self._get_obs(), self._get_info()

    def step(self, action: Dict[str, int], soldier_agent=None, p1_goal=None) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        reward, terminated, truncated = self._action_extraction(action)
        self.defeated_workers = 0

        # 0.1 Workers redistribution
        if self.board_env.redistribute_workers(owner_id=1, total_workers=self.player_env.resources[2]):
            reward -= 50.0; terminated = True
        self.board_env.redistribute_workers(owner_id=2, total_workers=self.opponent_env.resources[2])
        if terminated or truncated: return self._finalize_step(reward, terminated, truncated)

        # 0.2 SOLDIER LOGIC
        if self.current_turn > 0 and self.current_turn % 100 == 0:
            self.board_env.spawn_soldiers(player_id=2, count=1)
        
        terminated_soldiers, sol_defeated_workers = self.board_env.move_soldiers(soldier_agent, p1_goal)
        if terminated_soldiers: terminated = True
        self.defeated_workers += sol_defeated_workers

        # 0.3 WORKER GROWTH
        if self.current_turn > 0 and self.current_turn % 5 == 0:
            self.board_env.grow_workers(p1_food_gen=self.player_env.food_raw_income)

        # Sync
        self.player_env.resources[2] = float(np.sum(self.board_env.grid[self.board_env.grid[:,:,0]==1, 2]))
        self.opponent_env.resources[2] = float(np.sum(self.board_env.grid[self.board_env.grid[:,:,0]==2, 2]))

        # 1. Resources
        calc = self.resource_manager.collect_resources(player_id=1)
        self.lost_gold = calc.get("lost_gold", 0)
        self.lost_wood = calc.get("lost_wood", 0)
        self.lost_food = calc.get("lost_food", 0)
        self.lost_soldiers = calc.get("lost_soldiers", 0)
        
        # 2. Render State Storage
        self.last_dip_str = ["Trade", "Pass", "Attack"][self.diplomacy_action]
        self.last_eco_str = f"Recruit {self.soldiers_to_recruit}S"

        # 3. Diplomacy
        res = self.diplomacy_branch.execute_diplomacy(self.diplomacy_action, self.target_row, self.target_col, self.current_turn)
        if res.get("terminated"): terminated = True
        reward += res["reward"]
        self.history.append(res["history"])
        self.defeated_workers += res.get("defeated_workers", 0)
        
        if terminated or truncated: return self._finalize_step(reward, terminated, truncated, calc)

        # 4. Economy
        eco_res = self.economy_branch.execute_economy_action(
            self.soldiers_to_recruit, self.mines_action, self.trade_action,
            self.warehouse_action, self.crops_action, self.fortify_action
        )
        reward += eco_res["reward"]
        reward += self.player_env.process_food_consumption()

        # 5. Turn Management
        self.current_turn += 1
        if self.current_turn >= self.max_turns: truncated = True

        # 7. Opponent
        self.board_env.redistribute_workers(owner_id=2, total_workers=self.opponent_env.resources[2])
        if self.opponent_env.action_step(self.current_turn, self.board_env.get_owned_tiles(2)):
            victory, base_t, prev, _, mine = self.board_env.claim_adjacent_tile(2)
            if victory:
                self.opponent_env.resources[0] += 5
                if prev == 1: self.player_env.resources[2] = float(max(0, self.player_env.resources[2]-1))
                if base_t: reward -= 100.0; terminated = True
            else:
                self.opponent_env.resources[2] *= 0.9; reward += 0.1

        return self._finalize_step(reward, terminated, truncated, calc)

    def _finalize_step(self, reward, terminated, truncated, calc=None):
        if self.render_mode == "human": self.render()
        self.board_env.update_board()
        obs = self._get_obs()
        if calc:
            calc["defeated_workers"] = getattr(self, 'defeated_workers', 0)
            reward += self.reward_env.calculate_player_resources_reward(self.current_turn, obs, calc)
        return obs, float(np.clip(reward, -50.0, 50.0)), terminated, truncated, self._get_info()

    def render(self):
        if not self.render_mode: return
        state = {
            'p_res': self.player_env.resources, 'o_res': self.opponent_env.resources,
            'p_gen': (self.player_env.gold_net_income, self.player_env.wood_raw_income, 0, self.player_env.food_raw_income),
            'o_gen': (1, 1, 0, 0), 'turn': self.current_turn, 'board': self.board_env.get_tile_data(),
            'dip_act': getattr(self, 'last_dip_str', "None"), 'eco_act': getattr(self, 'last_eco_str', "None"),
            'history': self.history[-5:], 'p1_capacity': self.player_env.capacity,
            'p1_routes': self.board_env.p1_trade_manager.active_routes, 'p1_base': (0,0),
            'o1_routes': [], 'o1_base': (15,15),
            'p_warehouses': self.board_env.p1_buildings_manager.get_building_count_by_type(1, 2) + self.board_env.p1_buildings_manager.get_building_count_by_type(1, 6),
            'p_crops': self.board_env.p1_buildings_manager.get_building_count_by_type(1, 5) + self.board_env.p1_buildings_manager.get_building_count_by_type(1, 7)
        }
        return self.renderer.render_frame(self.render_mode, state)

    def close(self):
        if self.screen: pygame.quit()
