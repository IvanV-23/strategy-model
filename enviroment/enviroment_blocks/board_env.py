import numpy as np
import torch

from enviroment.enviroment_blocks.board_blocks.trade_route_manager_env import TradeRouteManager
from enviroment.enviroment_blocks.board_blocks.buildings_manager_env import BuildingsManager

class BoardEnv:
    def __init__(self, rows=16, cols=16):
        self.rows = rows
        self.cols = cols
        # 0: Owner, 1: Status, 2: Workers, 3: Wood, 4: P1 Sol, 5: P2 Sol, 6: Fort
        self.grid = np.zeros((rows, cols, 7), dtype=np.int32)
        self.p1_trade_manager = TradeRouteManager(base_coords=(0, 0))
        self.p1_buildings_manager = BuildingsManager(self.grid, rows, cols)

    def reset(self):
        resource_layer = self.grid[:, :, 3].copy() 
        self.grid = np.zeros((self.rows, self.cols, 7), dtype=np.int32)
        self.grid[:, :, 3] = resource_layer
        self.grid[0, 0, :3] = [1, 3, 10]
        self.grid[0, 0, 6] = 2 # Default Fortification Level 2
        self.grid[self.rows-1, self.cols-1, :3] = [2, 3, 10]
        self.grid[self.rows-1, self.cols-1, 6] = 2 # Default Fortification Level 2
        if np.sum(resource_layer) == 0: self._generate_resources()
        self.p1_trade_manager.active_routes = []
        return self.grid

    def update_board(self):
        self.p1_buildings_manager.update_board(self.grid, self.rows, self.cols)

    def spawn_soldiers(self, player_id, count):
        base_coords = (0, 0) if player_id == 1 else (self.rows - 1, self.cols - 1)
        channel = 4 if player_id == 1 else 5
        self.grid[base_coords[0], base_coords[1], channel] += count

    def move_soldiers(self, soldier_agent=None, p1_goal=None, target_tile=None):
        """
        Moves soldiers. 
        If soldier_agent and p1_goal are provided, P1 uses the HRL model.
        target_tile: (row, col) coordinates of the strategic target.
        """
        game_terminated = False
        defeated_workers = 0
        damage_dealt = 0.0
        self.lost_soldiers_combat = 0
        moved_closer = 0
        total_moves = 0
        p1_base, p2_base = (0, 0), (self.rows - 1, self.cols - 1)
        
        # 1. Process P1 Soldiers (Tactical Agent)
        next_p1 = np.zeros((self.rows, self.cols), dtype=np.int32)
        p1_indices = np.argwhere(self.grid[:, :, 4] > 0)
        
        full_obs = self.full_board_state() # (8, rows, cols)
        
        for r, c in p1_indices:
            count = self.grid[r, c, 4]
            nr, nc = r, c
            
            if soldier_agent is not None and p1_goal is not None:
                total_moves += 1
                # Hybrid Goal: Inject relative spatial intent into the first 2 slots
                current_p1_goal = p1_goal.clone()
                if target_tile is not None:
                    tr, tc = target_tile
                    # Relative Transformation: (target_xy - soldier_xy) / board_size
                    current_p1_goal[0, 0] = float(tr - r) / 20.0
                    current_p1_goal[0, 1] = float(tc - c) / 20.0

                local_view = self._get_local_view(full_obs, r, c, size=5)
                local_t = torch.from_numpy(local_view).float().unsqueeze(0).to(p1_goal.device)
                with torch.no_grad():
                    logits = soldier_agent(local_t, current_p1_goal)
                    action = torch.argmax(logits, dim=1).item()
                moves = [(0,0), (-1,0), (1,0), (0,1), (0,-1)]
                dr, dc = moves[action]
                if 0 <= r+dr < self.rows and 0 <= c+dc < self.cols:
                    nr, nc = r+dr, c+dc
                
                if target_tile is not None:
                    tr, tc = target_tile
                    old_dist = abs(tr - r) + abs(tc - c)
                    new_dist = abs(tr - nr) + abs(tc - nc)
                    if new_dist < old_dist:
                        moved_closer += 1
            else:
                neighbors = self._get_neighbors(r, c)
                if neighbors:
                    nr, nc = neighbors[np.random.randint(len(neighbors))]

            # Resolve P1 combat at (nr, nc)
            target_owner = self.grid[nr, nc, 0]
            if target_owner == 2:
                attack = count * 200
                defense = self.grid[nr, nc, 2] + (20 if self.grid[nr, nc, 1] > 0 else 0) + (self.grid[nr, nc, 5] * 200) + 100 + (100 if self.grid[nr, nc, 6] == 1 else 0)
                if attack > defense:
                    if (nr, nc) == p2_base: game_terminated = True
                    defeated_workers += int(self.grid[nr, nc, 2])
                    # Damage Dealt: Soldiers (200 each) + Buildings (Status > 0) + Fortifications (Status 6)
                    damage_dealt += (self.grid[nr, nc, 5] * 200.0)
                    if self.grid[nr, nc, 1] > 0: damage_dealt += 100.0
                    if self.grid[nr, nc, 6] > 0: damage_dealt += 100.0 * self.grid[nr, nc, 6]

                    self.grid[nr, nc, 0] = 0; self.grid[nr, nc, 1] = 0; self.grid[nr, nc, 2] = 0
                    self.grid[nr, nc, 5] = 0; self.grid[nr, nc, 6] = 0
                    next_p1[nr, nc] += count
            else:
                p2_count = self.grid[nr, nc, 5]
                if p2_count > 0:
                    if count * 200 > p2_count * 200:
                        damage_dealt += (p2_count * 200.0)
                        self.grid[nr, nc, 5] = 0; next_p1[nr, nc] += count
                else:
                    next_p1[nr, nc] += count

        # 2. Process P2 Soldiers (Random)
        next_p2 = np.zeros((self.rows, self.cols), dtype=np.int32)
        for r, c in np.argwhere(self.grid[:, :, 5] > 0):
            count = self.grid[r, c, 5]
            neighbors = self._get_neighbors(r, c)
            if not neighbors: next_p2[r, c] += count; continue
            nr, nc = neighbors[np.random.randint(len(neighbors))]
            
            if self.grid[nr, nc, 0] == 1:
                attack = count * 200
                defense = self.grid[nr, nc, 2] + (20 if self.grid[nr, nc, 1] > 0 else 0) + (next_p1[nr, nc] * 200) + 100 + (100 if self.grid[nr, nc, 6] == 1 else 0)
                if attack > defense:
                    if (nr, nc) == p1_base: game_terminated = True
                    self.lost_soldiers_combat += int(next_p1[nr, nc])
                    self.grid[nr, nc, 0] = 0; self.grid[nr, nc, 1] = 0; self.grid[nr, nc, 2] = 0
                    self.grid[nr, nc, 6] = 0; next_p1[nr, nc] = 0
                    next_p2[nr, nc] += count
            else:
                p1_count = next_p1[nr, nc]
                if p1_count > 0:
                    if count * 200 > p1_count * 200:
                        self.lost_soldiers_combat += int(next_p1[nr, nc])
                        next_p1[nr, nc] = 0; next_p2[nr, nc] += count
                    else:
                        next_p2[nr, nc] += count

        self.grid[:, :, 4] = next_p1
        self.grid[:, :, 5] = next_p2
        return game_terminated, defeated_workers, (moved_closer, total_moves), damage_dealt

    def _get_local_view(self, full_obs, r, c, size=5):
        pad = size // 2
        padded = np.pad(full_obs, ((0,0), (pad,pad), (pad,pad)), mode='constant')
        return padded[:, r:r+size, c:c+size]

    def grow_workers(self, p1_food_gen=None):
        WORKER_LIMIT, BASE_LIMIT = 50, 500
        p1_base, p2_base = (0, 0), (self.rows-1, self.cols-1)
        p1_tiles = np.argwhere(self.grid[:, :, 0] == 1)
        total_p1 = np.sum(self.grid[self.grid[:, :, 0] == 1, 2])
        
        if p1_food_gen is not None and p1_food_gen > 0:
            mult = np.clip(1.0 + (p1_food_gen / (total_p1 + 50)), 1.0, 2.0)
        else:
            mult = 1.0 # NO FOOD = NO GROWTH
            
        for r, c in p1_tiles:
            self.grid[r, c, 2] = int(self.grid[r, c, 2] * mult)
            self.grid[r, c, 2] = min(self.grid[r, c, 2], BASE_LIMIT if (r, c) == p1_base else WORKER_LIMIT)
        for r, c in np.argwhere(self.grid[:, :, 0] == 2):
            self.grid[r, c, 2] *= 2
            self.grid[r, c, 2] = min(self.grid[r, c, 2], BASE_LIMIT if (r, c) == p2_base else WORKER_LIMIT)
        return np.log1p(total_p1) * 0.1

    def fortify_tile(self, player_id, target_coords=None):
        if target_coords is not None:
            r, c = target_coords
            if 0 <= r < self.rows and 0 <= c < self.cols:
                if self.grid[r, c, 0] == player_id and self.grid[r, c, 6] < 2:
                    self.grid[r, c, 6] += 1
                    return True, f"at_{r}_{c}_lvl_{self.grid[r,c,6]}"
            return False, "invalid_target"

        owned = np.argwhere((self.grid[:, :, 0] == player_id) & (self.grid[:, :, 6] < 2))
        if not len(owned): return False, "none"
        base = np.array([0, 0] if player_id == 1 else [self.rows-1, self.cols-1])
        dists = [abs(r-base[0]) + abs(c-base[1]) for r, c in owned]
        candidates = [owned[i] for i, d in enumerate(dists) if d == min(dists)]
        r, c = candidates[np.random.randint(len(candidates))]
        self.grid[r, c, 6] += 1
        return True, f"at_{r}_{c}_lvl_{self.grid[r,c,6]}"

    def process_fortification_defense(self):
        """
        Level 2 fortifications shoot adjacent enemy soldiers.
        """
        shot_events = []
        # Find all Level 2 fortifications
        for r, c in np.argwhere(self.grid[:, :, 6] == 2):
            owner = self.grid[r, c, 0]
            enemy_channel = 5 if owner == 1 else 4
            for nr, nc in self._get_neighbors(r, c):
                if self.grid[nr, nc, enemy_channel] > 0:
                    self.grid[nr, nc, enemy_channel] -= 1
                    shot_events.append({"from": (r, c), "to": (nr, nc)})
                    if enemy_channel == 4: # P1 lost a soldier
                         if hasattr(self, 'lost_soldiers_combat'):
                             self.lost_soldiers_combat += 1
        return shot_events

    def get_tile_data(self):
        return [[{"owner": int(self.grid[r,c,0]), "status": int(self.grid[r,c,1]), "workers": float(self.grid[r,c,2]), "wood": int(self.grid[r,c,3]), "soldiers": int(self.grid[r,c,4]), "p2_soldiers": int(self.grid[r,c,5]), "fortified": int(self.grid[r,c,6])} for c in range(self.cols)] for r in range(self.rows)]

    def _get_neighbors(self, r, c):
        res = []
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            if 0 <= r+dr < self.rows and 0 <= c+dc < self.cols: res.append((r+dr, c+dc))
        return res

    def claim_adjacent_tile(self, owner_id):
        enemy_base = (self.rows-1, self.cols-1) if owner_id == 1 else (0,0)
        attacks = {}
        for r, c in np.argwhere(self.grid[:, :, 0] == owner_id):
            force = self.grid[r, c, 2]
            for nr, nc in self._get_neighbors(r, c):
                if self.grid[nr, nc, 0] != owner_id:
                    attacks[(nr, nc)] = attacks.get((nr, nc), 0) + force
        valid = []
        for (tr, tc), pwr in attacks.items():
            dfn = self.grid[tr, tc, 2] + (100 if self.grid[tr, tc, 0] != 0 else 0) + (20 if self.grid[tr, tc, 1] > 0 else 0) + (100 if self.grid[tr, tc, 6] == 1 else 0) + (50 if (tr, tc) in [(0,0), (self.rows-1, self.cols-1)] else 0)
            if self.grid[tr, tc, 0] == 0 or pwr > dfn: valid.append((tr, tc))
        if valid:
            tr, tc = valid[np.random.randint(len(valid))]
            prev = int(self.grid[tr, tc, 0])
            self.grid[tr, tc, 0], self.grid[tr, tc, 2], self.grid[tr, tc, 6] = int(owner_id), 1, 0
            return True, (tr, tc) == enemy_base, prev, 0, (prev != 0 and self.grid[tr, tc, 1] != 0)
        return False, False, None, 0, False

    def redistribute_workers(self, owner_id, total_workers):
        total_workers = int(total_workers)
        owned = np.argwhere(self.grid[:, :, 0] == owner_id)
        if not len(owned): return True
        if total_workers < len(owned):
            base = np.array([0,0] if owner_id == 1 else [self.rows-1, self.cols-1])
            dists = [np.linalg.norm(np.array(c) - base) for c in owned]
            keep = np.argsort(dists)[:total_workers]
            for i in range(len(owned)):
                if i not in keep: r, c = owned[i]; self.grid[r,c,0] = 0; self.grid[r,c,2] = 0; self.grid[r,c,6] = 0
            owned = owned[keep]
        if not len(owned): return True
        probs = np.ones(len(owned)) / len(owned)
        rem = total_workers - len(owned)
        base_c = (0,0) if owner_id == 1 else (self.rows-1, self.cols-1)
        if rem > 0:
            alloc = np.random.multinomial(rem, probs)
            excess = 0
            for i, (r, c) in enumerate(owned):
                tot = 1 + alloc[i]
                if (r, c) != base_c and tot > 50: excess += (tot-50); self.grid[r,c,2] = 50
                else: self.grid[r,c,2] = tot
            self.grid[base_c[0], base_c[1], 2] += excess
        else:
            for r, c in owned: self.grid[r, c, 2] = 1
        return False

    def claim_target_tile(self, owner_id, target):
        tr, tc = target
        if not (0 <= tr < self.rows and 0 <= tc < self.cols) or self.grid[tr, tc, 0] == owner_id or int(self.grid[tr, tc, 0]) != 0: return False, False, 0, "fail", 0
        if not any(self.grid[nr, nc, 0] == owner_id for nr, nc in self._get_neighbors(tr, tc)): return False, False, 0, "fail", 0
        attack = sum(self.grid[nr, nc, 2] for nr, nc in self._get_neighbors(tr, tc) if self.grid[nr, nc, 0] == owner_id)
        if attack > int(self.grid[tr, tc, 2]):
            self.grid[tr, tc, 0], self.grid[tr, tc, 2] = int(owner_id), 1
            return True, False, 0, "success", int(self.grid[tr, tc, 2])
        return False, False, 0, "fail", 0

    def get_soldier_count(self, player_id): return int(np.sum(self.grid[:, :, 4 if player_id == 1 else 5]))
    def remove_soldiers(self, player_id, count):
        ch = 4 if player_id == 1 else 5
        coords = np.argwhere(self.grid[:, :, ch] > 0)
        if not len(coords): return
        base = np.array([0,0] if player_id == 1 else [self.rows-1, self.cols-1])
        coords = sorted(coords, key=lambda x: np.linalg.norm(x - base), reverse=True)
        rem = 0
        for r, c in coords:
            if rem >= count: break
            take = min(self.grid[r, c, ch], count - rem)
            self.grid[r, c, ch] -= take; rem += take

    def get_owned_tiles(self, owner_id): return int(np.sum(self.grid[:, :, 0] == owner_id))
    
    def full_board_state(self) -> np.ndarray:
        mask = np.zeros((self.rows, self.cols), dtype=np.int32)
        for r, c in np.argwhere(self.grid[:, :, 0] == 1):
            for nr, nc in self._get_neighbors(r, c):
                if self.grid[nr, nc, 0] != 1: mask[nr, nc] = 1
        # 8 channels
        combined = np.concatenate([self.grid[:, :, :3], mask[..., np.newaxis], self.grid[:, :, 3:]], axis=2)
        return combined.transpose(2, 0, 1).astype(np.int32)

    def get_board_state_and_stats(self)-> dict:
        gs = np.array([self.p1_buildings_manager.get_resource_tile_count(1), self.p1_buildings_manager.get_mine_count(1), self.collect_gold_income(1), self.collect_wood_income(1), len(self.p1_trade_manager.active_routes), self.get_owned_tiles(owner_id=1), self.get_owned_tiles(owner_id=2), (int(self.p1_buildings_manager.get_mine_count(1)) - len(self.p1_trade_manager.active_routes)), self.p1_buildings_manager.get_crop_field_count(1)], dtype=np.float32)
        return {"visual": self.full_board_state(), "stats": gs}
    
    def get_action_mask(self, player_id):
        mask = np.zeros(self.rows * self.cols, dtype=bool)
        for r, c in np.argwhere(self.grid[:, :, 0] == player_id):
            for nr, nc in self._get_neighbors(r, c):
                # Allow selecting empty tiles (owner 0) OR enemy tiles (owner 2)
                if self.grid[nr, nc, 0] in [0, 2]: mask[nr * self.cols + nc] = True
        return mask

    def get_fortify_target_mask(self, player_id):
        mask = np.zeros(self.rows * self.cols, dtype=bool)
        # Any owned tile that is Level 0 or Level 1 is a valid target to select/upgrade
        for r, c in np.argwhere((self.grid[:, :, 0] == player_id) & (self.grid[:, :, 6] < 2)):
            mask[r * self.cols + c] = True
        return mask

    def get_build_mask(self, player_id, player_gold, player_wood, pending_fortify_tile=None):
        mask = np.zeros(8, dtype=bool)
        m_cnt = self.p1_buildings_manager.get_mine_count(player_id)
        res = self.p1_buildings_manager.get_resource_tile_count(player_id)
        space = np.any((self.grid[:, :, 0] == player_id) & (self.grid[:, :, 1] == 0) & (self.grid[:, :, 3] > 0))
        if m_cnt < res and space and player_gold >= 50: mask[0] = True 
        if np.any((self.grid[:, :, 0] == player_id) & (self.grid[:, :, 1] == 1)) and player_gold >= 100 and player_wood >= 50: mask[1] = True 
        if self.p1_trade_manager:
            mines = np.argwhere((self.grid[:, :, 0] == player_id) & ((self.grid[:, :, 1] == 1) | (self.grid[:, :, 1] == 4)))
            if any(tuple(pos) not in self.p1_trade_manager.active_routes for pos in mines): mask[2] = True 
        owned = np.argwhere(self.grid[:, :, 0] == player_id)
        sites = [tuple(c) for c in owned if self.grid[c[0], c[1], 1] == 0]
        has_m = any(any(self.grid[nr, nc, 1] in [1, 4] for nr, nc in self._get_neighbors(r, c)) for r, c in sites)
        if player_gold >= 30 and player_wood >= 20 and has_m: mask[3] = True 
        if np.any((self.grid[:, :, 0] == player_id) & (self.grid[:, :, 1] == 2)) and player_gold >= 60 and player_wood >= 40: mask[4] = True 
        has_wb = any(any(self.grid[nr, nc, 1] in [2, 3, 6] for nr, nc in self._get_neighbors(r, c)) for r, c in sites)
        if player_gold >= 20 and player_wood >= 10 and has_wb: mask[5] = True 
        if np.any((self.grid[:, :, 0] == player_id) & (self.grid[:, :, 1] == 5)) and player_gold >= 40 and player_wood >= 20: mask[6] = True 
        
        # Fortify: only if we have a pending target and it's still valid
        if player_gold >= 50 and player_wood >= 50:
            if pending_fortify_tile is not None:
                r, c = pending_fortify_tile
                if self.grid[r, c, 0] == player_id and self.grid[r, c, 6] < 2:
                    mask[7] = True
            else:
                # Fallback if no pending (though the agent should pick one)
                if np.any((self.grid[:, :, 0] == player_id) & (self.grid[:, :, 6] < 2)):
                    mask[7] = True
        return mask

    def collect_wood_income(self, player_id): return int(np.sum(self.grid[self.grid[:, :, 0] == player_id, 3]))
    def collect_food_income(self, player_id): return self.p1_buildings_manager.get_building_count_by_type(player_id, 5) * 10 + self.p1_buildings_manager.get_building_count_by_type(player_id, 7) * 20
    def create_trade_route(self, player_id):
        self.p1_trade_manager.validate_routes(self.grid, player_id)
        mines = np.argwhere((self.grid[:, :, 0] == player_id) & ((self.grid[:, :, 1] == 1) | (self.grid[:, :, 1] == 4)))
        eligible = [tuple(pos) for pos in mines if tuple(pos) not in self.p1_trade_manager.active_routes]
        if not eligible: return False, "none"
        base = self.p1_trade_manager.base_coords
        best = max(eligible, key=lambda p: abs(p[0]-base[0]) + abs(p[1]-base[1]))
        return self.p1_trade_manager.create_route(best), "success"
    def collect_gold_income(self, player_id):
        if not self.p1_trade_manager: return 0.0
        self.p1_trade_manager.validate_routes(self.grid, player_id) 
        return float(self.p1_trade_manager.calculate_income(self.grid) * 2)

    def _generate_resources(self):
        for r in range(self.rows):
            for c in range(self.cols): self.grid[r, c, 3] = np.random.randint(5, 11) if np.random.random() > 0.8 else 0
