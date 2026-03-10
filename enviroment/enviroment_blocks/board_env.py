import numpy as np

from enviroment.enviroment_blocks.board_blocks.trade_route_manager_env import TradeRouteManager
from enviroment.enviroment_blocks.board_blocks.buildings_manager_env import BuildingsManager

class BoardEnv:
    def __init__(self, rows=16, cols=16):
        # Board dimensions
        self.rows = rows
        self.cols = cols
        
        # Index 0: Owner (0=None, 1=P1, 2=P2)
        # Index 1: Status (Building types)
        # Index 2: Workers
        # Index 3: Resources (Wood)
        # Index 4: P1 Soldiers
        # Index 5: P2 Soldiers
        # Index 6: Fortification (0 or 1) (NEW)
        self.grid = np.zeros((rows, cols, 7), dtype=np.int32)

        # Managers
        self.p1_trade_manager = TradeRouteManager(base_coords=(0, 0))
        self.p1_buildings_manager = BuildingsManager(self.grid, rows, cols)

    def reset(self):
        resource_layer = self.grid[:, :, 3].copy() 
        self.grid = np.zeros((self.rows, self.cols, 7), dtype=np.int32)
        self.grid[:, :, 3] = resource_layer
        
        # Set starting bases
        self.grid[0, 0, :3] = [1, 3, 10]
        self.grid[self.rows-1, self.cols-1, :3] = [2, 3, 10]

        if np.sum(resource_layer) == 0:
            self._generate_resources()

        self.p1_trade_manager.active_routes = []
        return self.grid

    def update_board(self):
        self.p1_buildings_manager.update_board(self.grid, self.rows, self.cols)

    def spawn_soldiers(self, player_id, count):
        base_coords = (0, 0) if player_id == 1 else (self.rows - 1, self.cols - 1)
        channel = 4 if player_id == 1 else 5
        self.grid[base_coords[0], base_coords[1], channel] += count

    def move_soldiers(self):
        """
        Moves P1 and P2 soldiers randomly. Resolves combat.
        Returns True if ANY base is captured.
        """
        game_terminated = False
        p1_base = (0, 0)
        p2_base = (self.rows - 1, self.cols - 1)
        
        # 1. Process P1 Soldiers
        next_p1_soldiers = np.zeros((self.rows, self.cols), dtype=np.int32)
        p1_indices = np.argwhere(self.grid[:, :, 4] > 0)
        for r, c in p1_indices:
            count = self.grid[r, c, 4]
            neighbors = self._get_neighbors(r, c)
            if not neighbors: next_p1_soldiers[r, c] += count; continue
            
            idx = np.random.randint(len(neighbors))
            nr, nc = neighbors[idx]
            
            target_owner = self.grid[nr, nc, 0]
            if target_owner == 2: # Attacking P2
                # Force = Soldiers * 200 + Workers + Building + P2 Soldiers + Fortification Bonus
                attack = count * 200
                fort_bonus = 100 if self.grid[nr, nc, 6] == 1 else 0
                defense = self.grid[nr, nc, 2] + (20 if self.grid[nr, nc, 1] > 0 else 0) + (self.grid[nr, nc, 5] * 200) + 100 + fort_bonus
                
                if attack > defense:
                    if (nr, nc) == p2_base: game_terminated = True; print("P1 SOLDIERS CAPTURED THE ENEMY BASE!")
                    self.grid[nr, nc, 0] = 0
                    self.grid[nr, nc, 1] = 0
                    self.grid[nr, nc, 2] = 0
                    self.grid[nr, nc, 5] = 0 
                    self.grid[nr, nc, 6] = 0 # Fortification destroyed
                    next_p1_soldiers[nr, nc] += count
            else:
                p2_count = self.grid[nr, nc, 5]
                if p2_count > 0:
                    if count * 200 > p2_count * 200:
                        self.grid[nr, nc, 5] = 0
                        next_p1_soldiers[nr, nc] += count
                else:
                    next_p1_soldiers[nr, nc] += count
        
        # 2. Process P2 Soldiers
        next_p2_soldiers = np.zeros((self.rows, self.cols), dtype=np.int32)
        p2_indices = np.argwhere(self.grid[:, :, 5] > 0)
        for r, c in p2_indices:
            count = self.grid[r, c, 5]
            neighbors = self._get_neighbors(r, c)
            if not neighbors: next_p2_soldiers[r, c] += count; continue
            
            idx = np.random.randint(len(neighbors))
            nr, nc = neighbors[idx]
            
            target_owner = self.grid[nr, nc, 0]
            if target_owner == 1: # Attacking P1
                attack = count * 200
                fort_bonus = 100 if self.grid[nr, nc, 6] == 1 else 0
                defense = self.grid[nr, nc, 2] + (20 if self.grid[nr, nc, 1] > 0 else 0) + (next_p1_soldiers[nr, nc] * 200) + 100 + fort_bonus
                
                if attack > defense:
                    if (nr, nc) == p1_base: game_terminated = True; print("P2 SOLDIERS CAPTURED THE PLAYER BASE!")
                    self.grid[nr, nc, 0] = 0
                    self.grid[nr, nc, 1] = 0
                    self.grid[nr, nc, 2] = 0
                    self.grid[nr, nc, 6] = 0 # Fortification destroyed
                    next_p1_soldiers[nr, nc] = 0 
                    next_p2_soldiers[nr, nc] += count
            else:
                p1_count = next_p1_soldiers[nr, nc]
                if p1_count > 0:
                    if count * 200 > p1_count * 200:
                        next_p1_soldiers[nr, nc] = 0
                        next_p2_soldiers[nr, nc] += count
                else:
                    next_p2_soldiers[nr, nc] += count

        self.grid[:, :, 4] = next_p1_soldiers
        self.grid[:, :, 5] = next_p2_soldiers
        return game_terminated

    def grow_workers(self, p1_food_gen=None):
        WORKER_LIMIT = 50
        BASE_LIMIT = 500
        p1_base = (0, 0)
        p2_base = (self.rows - 1, self.cols - 1)
        
        p1_tiles = np.argwhere(self.grid[:, :, 0] == 1)
        if p1_food_gen is not None:
            total_p1_workers = np.sum(self.grid[self.grid[:, :, 0] == 1, 2])
            growth_multiplier = 1.02 + (p1_food_gen / (total_p1_workers + 50))
            growth_multiplier = np.clip(growth_multiplier, 1.0, 2.0)
        else:
            growth_multiplier = 2.0

        for r, c in p1_tiles:
            self.grid[r, c, 2] = int(self.grid[r, c, 2] * growth_multiplier)
            if (r, c) != p1_base: self.grid[r, c, 2] = min(self.grid[r, c, 2], WORKER_LIMIT)
            else: self.grid[r, c, 2] = min(self.grid[r, c, 2], BASE_LIMIT)
        
        p2_tiles = np.argwhere(self.grid[:, :, 0] == 2)
        for r, c in p2_tiles:
            self.grid[r, c, 2] *= 2
            if (r, c) != p2_base: self.grid[r, c, 2] = min(self.grid[r, c, 2], WORKER_LIMIT)
            else: self.grid[r, c, 2] = min(self.grid[r, c, 2], BASE_LIMIT)
        
        return np.log1p(np.sum(self.grid[self.grid[:, :, 0] == 1, 2])) * 0.1

    def fortify_tile(self, player_id):
        """
        Fortifies the closest available tile owned by the player to their base.
        Returns (success, message)
        """
        owned_indices = np.argwhere((self.grid[:, :, 0] == player_id) & (self.grid[:, :, 6] == 0))
        if len(owned_indices) == 0:
            return False, "no_tiles_available_to_fortify"
        
        base = np.array([0, 0] if player_id == 1 else [self.rows - 1, self.cols - 1])
        # Calculate Manhattan distances
        distances = [abs(r - base[0]) + abs(c - base[1]) for r, c in owned_indices]
        min_dist = min(distances)
        
        # Get all tiles at the minimum distance
        candidates = [owned_indices[i] for i, d in enumerate(distances) if d == min_dist]
        
        # Pick one randomly
        target = candidates[np.random.randint(len(candidates))]
        r, c = target
        self.grid[r, c, 6] = 1 # Fortified
        return True, f"fortified_tile_at_{r}_{c}"

    def get_tile_data(self):
        tile_data = []
        for r in range(self.rows):
            row_list = []
            for c in range(self.cols):
                row_list.append({
                    "owner": int(self.grid[r, c, 0]),
                    "status": int(self.grid[r, c, 1]), 
                    "workers": float(self.grid[r, c, 2]),
                    "wood": int(self.grid[r, c, 3]),
                    "soldiers": int(self.grid[r, c, 4]),
                    "p2_soldiers": int(self.grid[r, c, 5]),
                    "fortified": int(self.grid[r, c, 6]) # NEW
                })
            tile_data.append(row_list)
        return tile_data

    def _get_neighbors(self, r, c):
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                neighbors.append((nr, nc))
        return neighbors

    def claim_adjacent_tile(self, owner_id):
        rows, cols = self.grid.shape[0], self.grid.shape[1]
        terminated = False
        enemy_base = (rows - 1, cols - 1) if int(owner_id) == 1 else (0, 0)
        
        candidate_attacks = {} 
        owned_coords = np.argwhere(self.grid[:, :, 0] == owner_id)
        for r, c in owned_coords:
            attacking_force = self.grid[r, c, 2]
            for nr, nc in self._get_neighbors(r, c):
                if self.grid[nr, nc, 0] != owner_id:
                    if (nr, nc) not in candidate_attacks: candidate_attacks[(nr, nc)] = 0
                    candidate_attacks[(nr, nc)] += attacking_force

        if not candidate_attacks: return False, False, None, 0, False

        valid_conquests = []
        for (tr, tc), attack_power in candidate_attacks.items():
            target_owner = self.grid[tr, tc, 0]
            target_defense = self.grid[tr, tc, 2]
            if target_owner != 0:
                target_defense += 100 
                if self.grid[tr, tc, 1] > 0: target_defense += 20 
                if self.grid[tr, tc, 6] == 1: target_defense += 50 # Fortification bonus
            if (tr, tc) == (0, 0) or (tr, tc) == (rows - 1, cols - 1): target_defense += 50
            if target_owner == 0 or attack_power > target_defense: valid_conquests.append(((tr, tc), attack_power))

        if valid_conquests:
            idx = np.random.choice(len(valid_conquests))
            (target_r, target_c), _ = valid_conquests[idx]
            previous_owner = int(self.grid[target_r, target_c, 0])
            mine_was_lost = (previous_owner != 0 and int(self.grid[target_r, target_c, 1]) != 0)
            if (target_r, target_c) == enemy_base: terminated = True
            self.grid[target_r, target_c, 0] = int(owner_id)
            self.grid[target_r, target_c, 2] = 1 
            self.grid[target_r, target_c, 6] = 0 # Destroy fortification on capture
            return True, terminated, previous_owner, 0, mine_was_lost
        return False, False, None, 0, False

    def redistribute_workers(self, owner_id, total_workers, style=0):
        total_workers = int(total_workers)
        owned_indices = np.argwhere(self.grid[:, :, 0] == owner_id)
        num_tiles = len(owned_indices)
        if num_tiles == 0: return True 

        if total_workers < num_tiles:
            base = (0, 0) if owner_id == 1 else (self.rows - 1, self.cols - 1)
            distances = [np.linalg.norm(np.array(coord) - np.array(base)) for coord in owned_indices]
            keep_indices = np.argsort(distances)[:total_workers]
            for i in range(num_tiles):
                if i not in keep_indices:
                    r, c = owned_indices[i]; self.grid[r, c, 0] = 0; self.grid[r, c, 2] = 0; self.grid[r, c, 6] = 0
            owned_indices = owned_indices[keep_indices]
            num_tiles = len(owned_indices)
        if num_tiles == 0: return True 

        weights = np.ones(num_tiles) 
        probs = weights / np.sum(weights)
        remaining = total_workers - num_tiles
        base_coords = (0, 0) if owner_id == 1 else (self.rows - 1, self.cols - 1)
        
        if remaining > 0:
            extra = np.random.multinomial(remaining, probs)
            excess = 0
            for i, (r, c) in enumerate(owned_indices):
                total = 1 + extra[i]
                if (r, c) != base_coords and total > 50:
                    excess += (total - 50); self.grid[r, c, 2] = 50
                else: self.grid[r, c, 2] = total
            self.grid[base_coords[0], base_coords[1], 2] += excess
        else:
            for r, c in owned_indices: self.grid[r, c, 2] = 1
        return False

    def claim_target_tile(self, owner_id: int, target_coords: tuple) -> tuple[bool, bool, int, str, int]:
        tr, tc = target_coords
        if not (0 <= tr < self.rows and 0 <= tc < self.cols): return False, False, 0, "out_of_bounds", 0
        if self.grid[tr, tc, 0] == owner_id: return False, False, 0, "already_owned", 0
        if int(self.grid[tr, tc, 0]) != 0: return False, False, int(self.grid[tr, tc, 0]), "not_neutral", 0

        is_adjacent = any(self.grid[nr, nc, 0] == owner_id for nr, nc in self._get_neighbors(tr, tc))
        if not is_adjacent: return False, False, 0, "not_adjacent", 0

        total_attack = sum(self.grid[nr, nc, 2] for nr, nc in self._get_neighbors(tr, tc) if self.grid[nr, nc, 0] == owner_id)
        if total_attack > int(self.grid[tr, tc, 2]):
            self.grid[tr, tc, 0] = int(owner_id); self.grid[tr, tc, 2] = 1 
            return True, False, 0, "success", int(self.grid[tr, tc, 2])
        return False, False, 0, "insufficient_force", 0

    def get_soldier_count(self, player_id):
        return int(np.sum(self.grid[:, :, 4 if player_id == 1 else 5]))

    def remove_soldiers(self, player_id, count):
        channel = 4 if player_id == 1 else 5
        coords = np.argwhere(self.grid[:, :, channel] > 0)
        if len(coords) == 0: return
        base = np.array([0, 0] if player_id == 1 else [self.rows-1, self.cols-1])
        coords = sorted(coords, key=lambda x: np.linalg.norm(x - base), reverse=True)
        removed = 0
        for r, c in coords:
            if removed >= count: break
            to_rem = min(self.grid[r, c, channel], count - removed)
            self.grid[r, c, channel] -= to_rem; removed += to_rem

    def get_owned_tiles(self, owner_id): return int(np.sum(self.grid[:, :, 0] == owner_id))
    
    def full_board_state(self) -> np.ndarray:
        mask = np.zeros((self.rows, self.cols), dtype=np.int32)
        player_tiles = np.argwhere(self.grid[:, :, 0] == 1)
        for r, c in player_tiles:
            for nr, nc in self._get_neighbors(r, c):
                if self.grid[nr, nc, 0] != 1: mask[nr, nc] = 1

        # Return 8 channels: Owner, Status, Workers, Mask, Resource, P1 Soldiers, P2 Soldiers, Fortification
        combined = np.concatenate([
            self.grid[:, :, :3], 
            mask[..., np.newaxis], 
            self.grid[:, :, 3:4],
            self.grid[:, :, 4:5],
            self.grid[:, :, 5:6],
            self.grid[:, :, 6:7]
        ], axis=2)
        return combined.transpose(2, 0, 1).astype(np.int32)

    def get_board_state_and_stats(self)-> dict:
        global_stats = np.array([
            self.p1_buildings_manager.get_resource_tile_count(player_id=1), 
            self.p1_buildings_manager.get_mine_count(player_id=1),          
            self.collect_gold_income(player_id=1),                         
            self.collect_wood_income(player_id=1),                         
            len(self.p1_trade_manager.active_routes),                      
            self.get_owned_tiles(owner_id=1),                              
            self.get_owned_tiles(owner_id=2),                              
            (int(self.p1_buildings_manager.get_mine_count(player_id=1)) - len(self.p1_trade_manager.active_routes)),
            self.p1_buildings_manager.get_crop_field_count(player_id=1)    
        ], dtype=np.float32)
        return {"visual": self.full_board_state(), "stats": global_stats}
    
    def get_action_mask(self, player_id):
        mask = np.zeros(self.rows * self.cols, dtype=bool)
        owned = np.argwhere(self.grid[:, :, 0] == player_id)
        for r, c in owned:
            for nr, nc in self._get_neighbors(r, c):
                if self.grid[nr, nc, 0] == 0: mask[nr * self.cols + nc] = True
        return mask

    def get_build_mask(self, player_id: int, player_gold: int, player_wood: int) -> np.ndarray:
        # Mask: [MineL1, MineL2, Trade, WH_L1, WH_L2, CropL1, CropL2, Fortify]
        mask = np.zeros(8, dtype=bool)
        cur_mines = self.p1_buildings_manager.get_mine_count(player_id)
        res_tiles = self.p1_buildings_manager.get_resource_tile_count(player_id)
        has_space = np.any((self.grid[:, :, 0] == player_id) & (self.grid[:, :, 1] == 0) & (self.grid[:, :, 3] > 0))
        if cur_mines < res_tiles and has_space and player_gold >= 50: mask[0] = True 
        if np.any((self.grid[:, :, 0] == player_id) & (self.grid[:, :, 1] == 1)) and player_gold >= 100 and player_wood >= 50: mask[1] = True 
        manager = self.p1_trade_manager if player_id == 1 else None 
        if manager:
            mines = np.argwhere((self.grid[:, :, 0] == player_id) & ((self.grid[:, :, 1] == 1) | (self.grid[:, :, 1] == 4)))
            if any(tuple(pos) not in manager.active_routes for pos in mines): mask[2] = True 
        owned = np.argwhere(self.grid[:, :, 0] == player_id)
        sites = [tuple(c) for c in owned if self.grid[c[0], c[1], 1] == 0]
        has_m_n = any(any(self.grid[nr, nc, 1] in [1, 4] for nr, nc in self._get_neighbors(r, c)) for r, c in sites)
        if player_gold >= 30 and player_wood >= 20 and has_m_n: mask[3] = True 
        if np.any((self.grid[:, :, 0] == player_id) & (self.grid[:, :, 1] == 2)) and player_gold >= 60 and player_wood >= 40: mask[4] = True 
        has_wb_n = any(any(self.grid[nr, nc, 1] in [2, 3, 6] for nr, nc in self._get_neighbors(r, c)) for r, c in sites)
        if player_gold >= 20 and player_wood >= 10 and has_wb_n: mask[5] = True 
        if np.any((self.grid[:, :, 0] == player_id) & (self.grid[:, :, 1] == 5)) and player_gold >= 40 and player_wood >= 20: mask[6] = True 
        
        # Fortify: Cost 50 Gold, 50 Wood
        has_unfortified = np.any((self.grid[:, :, 0] == player_id) & (self.grid[:, :, 6] == 0))
        if player_gold >= 50 and player_wood >= 50 and has_unfortified:
            mask[7] = True

        return mask

    def collect_wood_income(self, player_id): return int(np.sum(self.grid[self.grid[:, :, 0] == player_id, 3]))
    def collect_food_income(self, player_id): return self.p1_buildings_manager.get_building_count_by_type(player_id, 5) * 10 + self.p1_buildings_manager.get_building_count_by_type(player_id, 7) * 20
    def create_trade_route(self, player_id: int) -> tuple[bool, str]:
        manager = self.p1_trade_manager if player_id == 1 else None
        manager.validate_routes(self.grid, player_id)
        mines = np.argwhere((self.grid[:, :, 0] == player_id) & ((self.grid[:, :, 1] == 1) | (self.grid[:, :, 1] == 4)))
        eligible = [tuple(pos) for pos in mines if tuple(pos) not in manager.active_routes]
        if not eligible: return False, "no_mines"
        base_r, base_c = manager.base_coords
        furthest = max(eligible, key=lambda p: abs(p[0]-base_r) + abs(p[1]-base_c))
        if manager.create_route(furthest): return True, "success"
        return False, "fail"
    def collect_gold_income(self, player_id):
        manager = self.p1_trade_manager if player_id == 1 else None
        if not manager: return 0.0
        manager.validate_routes(self.grid, player_id) 
        return float(manager.calculate_income(self.grid) * 2)

    def _generate_resources(self):
        for r in range(self.rows):
            for c in range(self.cols):
                if np.random.random() > 0.8: self.grid[r, c, 3] = np.random.randint(5, 11)
                else: self.grid[r, c, 3] = 0
