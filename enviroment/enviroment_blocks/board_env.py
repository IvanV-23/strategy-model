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
        # Index 4: Soldiers
        self.grid = np.zeros((rows, cols, 5), dtype=np.int32)

        # Managers
        self.p1_trade_manager = TradeRouteManager(base_coords=(0, 0))
        self.p1_buildings_manager = BuildingsManager(self.grid, rows, cols)

    def reset(self):
        # Preserve resource distribution if it exists, or generate new
        resource_layer = self.grid[:, :, 3].copy() 
        self.grid = np.zeros((self.rows, self.cols, 5), dtype=np.int32)
        self.grid[:, :, 3] = resource_layer
        
        # Set starting bases
        # [Owner, Status=Base, Workers]
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
        self.grid[base_coords[0], base_coords[1], 4] += count

    def move_soldiers(self):
        """
        Moves soldiers randomly. Returns True if opponent base is captured.
        """
        terminated = False
        p2_base = (self.rows - 1, self.cols - 1)
        next_soldier_grid = np.zeros((self.rows, self.cols), dtype=np.int32)
        soldier_indices = np.argwhere(self.grid[:, :, 4] > 0)
        
        for r, c in soldier_indices:
            count = self.grid[r, c, 4]
            neighbors = self._get_neighbors(r, c)
            if not neighbors:
                next_soldier_grid[r, c] += count
                continue

            idx = np.random.randint(len(neighbors))
            nr, nc = neighbors[idx]
            target_owner = self.grid[nr, nc, 0]
            
            if target_owner == 2: # P1 Soldiers attacking P2
                attack_power = count * 200
                defense_power = self.grid[nr, nc, 2] # Workers
                if self.grid[nr, nc, 1] > 0: defense_power += 20 # Building
                
                # Base or Tile bonus
                defense_power += 100

                if attack_power > defense_power:
                    # Success: Wipe tile
                    if (nr, nc) == p2_base:
                        terminated = True
                        print("SOLDIERS CAPTURED THE ENEMY BASE!")

                    self.grid[nr, nc, 0] = 0
                    self.grid[nr, nc, 1] = 0
                    self.grid[nr, nc, 2] = 0
                    next_soldier_grid[nr, nc] += count 
                else:
                    # Failure: Soldiers perish
                    pass 
            else:
                next_soldier_grid[nr, nc] += count

        self.grid[:, :, 4] = next_soldier_grid
        return terminated

    def grow_workers(self, p1_food_gen=None):
        WORKER_LIMIT = 50
        BASE_LIMIT = 500 
        p1_base = (0, 0)
        p2_base = (self.rows - 1, self.cols - 1)
        
        # P1 Growth
        p1_tiles = np.argwhere(self.grid[:, :, 0] == 1)
        if p1_food_gen is not None:
            total_p1_workers = np.sum(self.grid[self.grid[:, :, 0] == 1, 2])
            growth_multiplier = 1.02 + (p1_food_gen / (total_p1_workers + 50))
            growth_multiplier = np.clip(growth_multiplier, 1.0, 2.0)
        else:
            growth_multiplier = 2.0

        for r, c in p1_tiles:
            self.grid[r, c, 2] = int(self.grid[r, c, 2] * growth_multiplier)
            if (r, c) != p1_base:
                self.grid[r, c, 2] = min(self.grid[r, c, 2], WORKER_LIMIT)
            else:
                self.grid[r, c, 2] = min(self.grid[r, c, 2], BASE_LIMIT)
        
        # P2 Growth
        p2_tiles = np.argwhere(self.grid[:, :, 0] == 2)
        for r, c in p2_tiles:
            self.grid[r, c, 2] *= 2
            if (r, c) != p2_base:
                self.grid[r, c, 2] = min(self.grid[r, c, 2], WORKER_LIMIT)
            else:
                self.grid[r, c, 2] = min(self.grid[r, c, 2], BASE_LIMIT)
        
        return np.log1p(total_p1_workers) * 0.1

    def get_tile_data(self):
        tile_data = []
        for r in range(self.rows):
            row_list = []
            for c in range(self.cols):
                row_list.append({
                    "owner": int(self.grid[r, c, 0]),
                    "status": int(self.grid[r, c, 1]), 
                    "workers": int(self.grid[r, c, 2]),
                    "wood": int(self.grid[r, c, 3]),
                    "soldiers": int(self.grid[r, c, 4])
                })
            tile_data.append(row_list)
        return tile_data

    def _generate_resources(self):
        for r in range(self.rows):
            for c in range(self.cols):
                if np.random.random() > 0.8:
                    self.grid[r, c, 3] = np.random.randint(5, 11)
                else:
                    self.grid[r, c, 3] = 0

    def _get_neighbors(self, r, c):
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                neighbors.append((nr, nc))
        return neighbors

    def claim_adjacent_tile(self, owner_id):
        """Used by the Opponent (Simple AI)"""
        rows, cols = self.grid.shape[0], self.grid.shape[1]
        terminated = False
        enemy_base = (rows - 1, cols - 1) if int(owner_id) == 1 else (0, 0)
        previous_owner = None 
        mine_was_lost = False 

        # 1. Find all potential tiles to attack
        candidate_attacks = {} 
        owned_coords = np.argwhere(self.grid[:, :, 0] == owner_id)
        
        for r, c in owned_coords:
            attacking_force = self.grid[r, c, 2]
            for nr, nc in self._get_neighbors(r, c):
                if self.grid[nr, nc, 0] != owner_id:
                    if (nr, nc) not in candidate_attacks:
                        candidate_attacks[(nr, nc)] = 0
                    candidate_attacks[(nr, nc)] += attacking_force

        if not candidate_attacks:
            return False, False, None, 0, False

        # 2. Filter candidates by battle logic
        valid_conquests = []
        for (tr, tc), attack_power in candidate_attacks.items():
            target_owner = self.grid[tr, tc, 0]
            target_defense = self.grid[tr, tc, 2]

            if target_owner != 0:
                target_defense += 100 
                if self.grid[tr, tc, 1] > 0:
                    target_defense += 20 

            if (tr, tc) == (0, 0) or (tr, tc) == (rows - 1, cols - 1):
                target_defense += 50

            if target_owner == 0 or attack_power > target_defense:
                valid_conquests.append(((tr, tc), attack_power))

        # 3. Execute a random valid conquest
        if valid_conquests:
            idx = np.random.choice(len(valid_conquests))
            (target_r, target_c), final_attack_power = valid_conquests[idx]
            
            previous_owner = int(self.grid[target_r, target_c, 0])
            target_structure = int(self.grid[target_r, target_c, 1])
            defeated_workers = int(self.grid[target_r, target_c, 2])
            
            if previous_owner != 0 and target_structure != 0:
                mine_was_lost = True
            
            if (target_r, target_c) == enemy_base:
                terminated = True
            
            self.grid[target_r, target_c, 0] = int(owner_id)
            self.grid[target_r, target_c, 2] = 1 
            
            return True, terminated, previous_owner, defeated_workers, mine_was_lost
        
        return False, False, None, 0, False

    def redistribute_workers(self, owner_id, total_workers, style=0):
        total_workers = int(total_workers)
        # 1. Find all tiles currently owned by this player
        owned_indices = np.argwhere(self.grid[:, :, 0] == owner_id)
        num_tiles = len(owned_indices)

        if num_tiles == 0: return True # Defeat

        # 2. Rule: Every tile needs at least 1 worker
        if total_workers < num_tiles:
            base = (0, 0) if owner_id == 1 else (self.rows - 1, self.cols - 1)
            distances = [np.linalg.norm(np.array(coord) - np.array(base)) for coord in owned_indices]
            keep_indices = np.argsort(distances)[:total_workers]
            
            for i in range(num_tiles):
                if i not in keep_indices:
                    r, c = owned_indices[i]
                    self.grid[r, c, 0] = 0 
                    self.grid[r, c, 2] = 0 
            
            owned_indices = owned_indices[keep_indices]
            num_tiles = len(owned_indices)

        if num_tiles == 0: return True 

        # 3. Strategic Weighting
        weights = np.ones(num_tiles) 

        # 4. Final Allocation
        probs = weights / np.sum(weights)
        remaining_workers = total_workers - num_tiles
        WORKER_LIMIT = 50
        base_coords = (0, 0) if owner_id == 1 else (self.rows - 1, self.cols - 1)
        excess_workers = 0
        
        if remaining_workers > 0:
            extra_allocations = np.random.multinomial(remaining_workers, probs)
            for i, (r, c) in enumerate(owned_indices):
                total_on_tile = 1 + extra_allocations[i]
                if (r, c) != base_coords and total_on_tile > WORKER_LIMIT:
                    excess_workers += (total_on_tile - WORKER_LIMIT)
                    self.grid[r, c, 2] = WORKER_LIMIT
                else:
                    self.grid[r, c, 2] = total_on_tile
            self.grid[base_coords[0], base_coords[1], 2] += excess_workers
        else:
            for r, c in owned_indices:
                self.grid[r, c, 2] = 1
        
        return False

    def claim_target_tile(self, owner_id: int, target_coords: tuple) -> tuple[bool, bool, int, str, int]:
        tr, tc = target_coords
        rows, cols = self.grid.shape[0], self.grid.shape[1]
        p1_base, p2_base = (0, 0), (rows - 1, cols - 1)
        enemy_base = p2_base if int(owner_id) == 1 else p1_base
        
        if not (0 <= tr < rows and 0 <= tc < cols):
            return False, False, 0, "out_of_bounds", 0
        if self.grid[tr, tc, 0] == owner_id:
            return False, False, 0, "already_owned", 0

        # Workers only conquer Neutral
        target_owner = int(self.grid[tr, tc, 0])
        if target_owner != 0:
            return False, False, target_owner, "workers_cannot_attack_occupied", 0

        # Adjacency
        is_adjacent = False
        total_attacking_force = 0
        for nr, nc in self._get_neighbors(tr, tc):
            if self.grid[nr, nc, 0] == owner_id:
                is_adjacent = True
                total_attacking_force += self.grid[nr, nc, 2]

        if not is_adjacent:
            return False, False, 0, "not_adjacent", 0

        # Battle
        actual_workers = int(self.grid[tr, tc, 2])
        if total_attacking_force > actual_workers:
            base_captured = (tr, tc) == enemy_base
            self.grid[tr, tc, 0] = int(owner_id)
            self.grid[tr, tc, 2] = 1 
            return True, base_captured, target_owner, "success", actual_workers
        
        return False, False, target_owner, "insufficient_force", 0

    def get_soldier_count(self, player_id):
        if player_id == 1: return int(np.sum(self.grid[:, :, 4]))
        return 0

    def remove_soldiers(self, player_id, count):
        if player_id != 1: return
        soldier_coords = np.argwhere(self.grid[:, :, 4] > 0)
        if len(soldier_coords) == 0: return
        
        base = np.array([0, 0])
        soldier_coords = sorted(soldier_coords, key=lambda x: np.linalg.norm(x - base), reverse=True)
        
        removed = 0
        for r, c in soldier_coords:
            if removed >= count: break
            on_tile = self.grid[r, c, 4]
            to_remove = min(on_tile, count - removed)
            self.grid[r, c, 4] -= to_remove
            removed += to_remove

    def get_owned_tiles(self, owner_id):
        return int(np.sum(self.grid[:, :, 0] == owner_id))
    
    def get_tile_ownership(self):
        return self.grid[:, :, 0]
    
    def full_board_state(self) -> np.ndarray:
        rows, cols = self.grid.shape[0], self.grid.shape[1]
        mask = np.zeros((rows, cols), dtype=np.int32)
        owner_channel = self.grid[:, :, 0]
        player_tiles = np.argwhere(owner_channel == 1)
        
        for r, c in player_tiles:
            for nr, nc in self._get_neighbors(r, c):
                if owner_channel[nr, nc] != 1:
                    mask[nr, nc] = 1

        core_layers = self.grid[:, :, :3] 
        resource_layer = self.grid[:, :, 3:4]
        soldier_layer = self.grid[:, :, 4:]
        
        combined = np.concatenate([
            core_layers, 
            mask[..., np.newaxis], 
            resource_layer,
            soldier_layer
        ], axis=2)
        
        return combined.transpose(2, 0, 1).astype(np.int32)

    def get_potencial_trade_routes(self) -> int:
        return int(self.p1_buildings_manager.get_mine_count(player_id=1)) - len(self.p1_trade_manager.active_routes)

    def get_board_state_and_stats(self)-> dict:
        spatial_board = self.full_board_state() 
        global_stats = np.array([
            self.p1_buildings_manager.get_resource_tile_count(player_id=1), 
            self.p1_buildings_manager.get_mine_count(player_id=1),          
            self.collect_gold_income(player_id=1),                         
            self.collect_wood_income(player_id=1),                         
            len(self.p1_trade_manager.active_routes),                      
            self.get_owned_tiles(owner_id=1),                              
            self.get_owned_tiles(owner_id=2),                              
            self.get_potencial_trade_routes(),                             
            self.p1_buildings_manager.get_crop_field_count(player_id=1)    
        ], dtype=np.float32)
        
        return {"visual": spatial_board, "stats": global_stats}
    
    def get_action_mask(self, player_id):
        mask = np.zeros(self.rows * self.cols, dtype=bool)
        owned_coords = np.argwhere(self.grid[:, :, 0] == player_id)
        for r, c in owned_coords:
            for nr, nc in self._get_neighbors(r, c):
                if self.grid[nr, nc, 0] == 0:
                    mask[nr * self.cols + nc] = True
        return mask

    def get_build_mask(self, player_id: int, player_gold: int, player_wood: int) -> np.ndarray:
        mask = np.zeros(7, dtype=bool)
        current_mines = self.p1_buildings_manager.get_mine_count(player_id)
        resource_tiles = self.p1_buildings_manager.get_resource_tile_count(player_id)
        has_mine_space = np.any((self.grid[:, :, 0] == player_id) & (self.grid[:, :, 1] == 0) & (self.grid[:, :, 3] > 0))
        
        if current_mines < resource_tiles and has_mine_space and player_gold >= 50:
            mask[0] = True 
        has_l1_mines = np.any((self.grid[:, :, 0] == player_id) & (self.grid[:, :, 1] == 1))
        if has_l1_mines and player_gold >= 100 and player_wood >= 50:
            mask[1] = True 

        manager = self.p1_trade_manager if player_id == 1 else None 
        if manager:
            all_mines = np.argwhere((self.grid[:, :, 0] == player_id) & ((self.grid[:, :, 1] == 1) | (self.grid[:, :, 1] == 4)))
            eligible_mines = [tuple(pos) for pos in all_mines if tuple(pos) not in manager.active_routes]
            if len(eligible_mines) > 0:
                mask[2] = True 

        owned_coords = np.argwhere(self.grid[:, :, 0] == player_id)
        potential_sites = [tuple(c) for c in owned_coords if self.grid[c[0], c[1], 1] == 0]
        
        has_mine_neighbor = False
        for r, c in potential_sites:
            for nr, nc in self._get_neighbors(r, c):
                if self.grid[nr, nc, 1] in [1, 4]: 
                    has_mine_neighbor = True; break
            if has_mine_neighbor: break
        if player_gold >= 30 and player_wood >= 20 and has_mine_neighbor:
            mask[3] = True 

        has_l1_wh = np.any((self.grid[:, :, 0] == player_id) & (self.grid[:, :, 1] == 2))
        if has_l1_wh and player_gold >= 60 and player_wood >= 40:
            mask[4] = True 

        has_wh_base_neighbor = False
        for r, c in potential_sites:
            for nr, nc in self._get_neighbors(r, c):
                if self.grid[nr, nc, 1] in [2, 3, 6]: 
                    has_wh_base_neighbor = True; break
            if has_wh_base_neighbor: break
        if player_gold >= 20 and player_wood >= 10 and has_wh_base_neighbor:
            mask[5] = True 

        has_l1_crop = np.any((self.grid[:, :, 0] == player_id) & (self.grid[:, :, 1] == 5))
        if has_l1_crop and player_gold >= 40 and player_wood >= 20:
            mask[6] = True 
        return mask

    def collect_wood_income(self, player_id):
        owned_mask = (self.grid[:, :, 0] == player_id)
        return int(np.sum(self.grid[owned_mask, 3]))

    def collect_food_income(self, player_id):
        return self.p1_buildings_manager.get_building_count_by_type(player_id, 5) * 10 + \
               self.p1_buildings_manager.get_building_count_by_type(player_id, 7) * 20

    def create_trade_route(self, player_id: int) -> tuple[bool, str]:
        manager = self.p1_trade_manager if player_id == 1 else None
        manager.validate_routes(self.grid, player_id)
        all_mines = np.argwhere((self.grid[:, :, 0] == player_id) & ((self.grid[:, :, 1] == 1) | (self.grid[:, :, 1] == 4)))
        eligible_mines = [tuple(pos) for pos in all_mines if tuple(pos) not in manager.active_routes]
        if not eligible_mines: return False, "no_mines"
        base_r, base_c = manager.base_coords
        furthest = max(eligible_mines, key=lambda p: abs(p[0]-base_r) + abs(p[1]-base_c))
        if manager.create_route(furthest): return True, "success"
        return False, "fail"

    def collect_gold_income(self, player_id):
        manager = self.p1_trade_manager if player_id == 1 else None
        if not manager: return 0.0
        manager.validate_routes(self.grid, player_id) 
        return float(manager.calculate_income(self.grid) * 2)
