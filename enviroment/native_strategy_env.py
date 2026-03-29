import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional

try:
    import strategy_engine as engine
    NATIVE_ENGINE_AVAILABLE = True
except ImportError:
    NATIVE_ENGINE_AVAILABLE = False
    engine = None

def create_native_env(render_mode: str = None) -> 'NativeStrategyEnv':
    """Factory function to create a native strategy environment."""
    return NativeStrategyEnv(render_mode=render_mode)

class NativeStrategyEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: str = None):
        super().__init__()

        if not NATIVE_ENGINE_AVAILABLE:
            raise RuntimeError("strategy_engine native module not available. Build with pybind11.")

        self._manager = engine.TurnManager()
        self._state = engine.BoardFactory.create_default(16, 16, 0, 0, 15, 15)
        self._render_mode = render_mode
        self._current_turn = 0
        self._max_turns = 1000
        self._shot_events = []

        self.diplomacy_action_space = spaces.Discrete(3)
        self.economy_action_space = spaces.MultiDiscrete([10, 3, 2, 3, 3, 2])
        self.target_action_space = spaces.Discrete(256)
        self.soldier_target_action_space = spaces.Discrete(256)
        self.fortify_target_action_space = spaces.Discrete(256)

        self.action_space = spaces.Dict({
            "diplomacy": self.diplomacy_action_space,
            "economy": self.economy_action_space,
            "target_tile": self.target_action_space,
            "soldier_target_tile": self.soldier_target_action_space,
            "fortify_tile": self.fortify_target_action_space,
        })

        self.observation_space = spaces.Dict({
            "player_resources": spaces.Box(low=0, high=1000, shape=(9,), dtype=np.float64),
            "opponent_resources": spaces.Box(low=0, high=1000, shape=(3,), dtype=np.int32),
            "turn_number": spaces.Discrete(1000),
            "board_state": spaces.Box(low=0, high=255, shape=(8, 16, 16), dtype=np.int32),
            "board_stats": spaces.Box(low=0, high=1000, shape=(15,), dtype=np.float32),
        })

    def _extract_obs(self) -> Dict[str, np.ndarray]:
        p_res = self._state.p_res

        board_state = np.zeros((8, 16, 16), dtype=np.int32)
        for r in range(len(self._state.board)):
            for c in range(len(self._state.board[r])):
                tile = self._state.board[r][c]
                board_state[0, r, c] = tile.owner
                board_state[1, r, c] = int(tile.status)
                board_state[2, r, c] = tile.wood
                board_state[3, r, c] = tile.fortified
                board_state[4, r, c] = tile.soldiers
                board_state[5, r, c] = tile.p2_soldiers

        if self._state.target_tile:
            tr, tc = self._state.target_tile
            board_state[6, tr, tc] = 1

        if self._state.soldier_target_tile:
            sr, sc = self._state.soldier_target_tile
            board_state[7, sr, sc] = 1

        opponent_res = self._compute_opponent_resources()

        return {
            "player_resources": np.array([p_res.gold, p_res.wood, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
            "opponent_resources": np.array(opponent_res, dtype=np.int32),
            "turn_number": self._current_turn,
            "board_state": board_state,
            "board_stats": np.zeros(15, dtype=np.float32),
        }

    def _compute_opponent_resources(self) -> list:
        gold = 0.0
        wood = 0.0
        workers = 0.0
        for row in self._state.board:
            for tile in row:
                if tile.owner == 2:
                    if tile.status == engine.TileStatus.Warehouse_P2:
                        gold += 5
                    elif tile.status == engine.TileStatus.Crops_P2:
                        gold += 3
                    if tile.status == engine.TileStatus.Sawmill_P2:
                        wood += 10
                    workers += tile.wood
        return [gold, wood, workers]

    def _get_info(self) -> Dict[str, Any]:
        return {
            "action_mask": np.ones(256, dtype=np.int32),
            "build_mask": self._get_build_mask(),
            "fortify_target_mask": np.ones(256, dtype=np.int32),
            "shot_events": self._shot_events,
        }

    def _get_build_mask(self) -> np.ndarray:
        mask = np.ones(4, dtype=np.int32)
        p_res = self._state.p_res
        if p_res.gold < 100 or p_res.wood < 50:
            mask[0] = 0
        if p_res.gold < 60 or p_res.wood < 40:
            mask[1] = 0
        if p_res.gold < 40 or p_res.wood < 20:
            mask[2] = 0
        if p_res.gold < 30 or p_res.wood < 15:
            mask[3] = 0
        return mask

    def reset(self, seed: Optional[int] = None, options: Dict = None) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)
        self._state = engine.BoardFactory.create_default(16, 16, 0, 0, 15, 15)
        self._current_turn = 0
        self._shot_events = []
        return self._extract_obs(), self._get_info()

    def step(self, action: Dict) -> Tuple[Dict, float, bool, bool, Dict]:
        p1_bundle = engine.ActionBundle()
        p1_bundle.diplomacy = int(action["diplomacy"])
        p1_bundle.economy = [int(x) for x in action["economy"]]
        p1_bundle.target_tile = int(action["target_tile"])
        p1_bundle.soldier_target = int(action["soldier_target_tile"])
        p1_bundle.fortify_tile = int(action["fortify_tile"])

        p2_bundle = engine.ActionBundle()
        p2_bundle.diplomacy = 1
        p2_bundle.economy = [0, 0, 0, 0, 0, 0]

        prev_state = self._state

        self._manager.step(self._state, p1_bundle, p2_bundle)
        self._current_turn += 1

        self._shot_events = [
            {"from": list(se.from_coord), "to": list(se.to_coord)}
            for se in self._state.shot_events
        ]

        terminated = self._manager.is_terminal(self._state)
        truncated = self._current_turn >= self._max_turns

        reward = self._manager.compute_reward(prev_state, self._state, 1)

        return self._extract_obs(), float(reward), terminated, truncated, self._get_info()

    def render(self):
        if not self._render_mode:
            return None
        state = {
            'p_res': [self._state.p_res.gold, self._state.p_res.wood, 0.0],
            'o_res': self._compute_opponent_resources(),
            'turn': self._current_turn,
            'board': [
                [
                    {
                        "owner": tile.owner,
                        "status": int(tile.status),
                        "wood": tile.wood,
                        "fortified": tile.fortified,
                        "soldiers": tile.soldiers,
                        "p2_soldiers": tile.p2_soldiers
                    }
                    for tile in row
                ]
                for row in self._state.board
            ],
            'shot_events': self._shot_events,
        }
        return state

    def close(self):
        pass
