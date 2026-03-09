import numpy as np

class BuildingsManager:

    def __init__(self, grid, rows, cols):
        self.grid = grid
        self.rows = rows
        self.cols = cols

    def update_board(self, new_grid, rows, cols):
        self.grid = new_grid
        self.rows = rows
        self.cols = cols

    def _get_neighbors(self, r, c):
            """Helper to get valid adjacent coordinates"""
            neighbors = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    neighbors.append((nr, nc))
            return neighbors

    def get_mine_count(self, player_id):
        """
            Returns the number of Mines (Status 1 or 4) currently owned by the player.
        """
        owned_mask = (self.grid[:, :, 0] == player_id)
        is_mine_mask = (self.grid[:, :, 1] == 1) | (self.grid[:, :, 1] == 4)
        
        player_mines = np.logical_and(owned_mask, is_mine_mask)
        return np.sum(player_mines)

    def build_mine(self, player_id, mine_type):
            """
            Attempts to build a specific mine type on a resource-rich tile.
            """
            # 1. Action 0 is "Do Nothing"
            if mine_type == 0:
                return True, "skipped"

            # 2. Global Limit Check
            current_mines = self.get_mine_count(player_id)
            resource_tiles = self.get_resource_tile_count(player_id)
            if current_mines >= resource_tiles:
                return False, f"limit_reached_{current_mines}/{resource_tiles}"

            # 3. Find a valid tile for the player
            valid_tiles = np.argwhere(
                (self.grid[:, :, 0] == player_id) & 
                (self.grid[:, :, 1] == 0) & 
                (self.grid[:, :, 3] > 0)
            )

            if len(valid_tiles) == 0:
                return False, "no_valid_resource_tiles_available"

            # 4. Strategy: Pick the tile with the HIGHEST resource value
            best_idx = np.argmax([self.grid[tr, tc, 3] for tr, tc in valid_tiles])
            r, c = valid_tiles[best_idx]

            # 5. Build Mine Lvl 1
            self.grid[r, c, 1] = 1 
            
            return True, f"built_mine_at_{r}_{c}"

    def get_resource_tile_count(self, player_id: int) -> int:
        owned_mask = (self.grid[:, :, 0] == player_id)
        has_resource_mask = (self.grid[:, :, 3] > 0)
        valid_spots = np.logical_and(owned_mask, has_resource_mask)
        return int(np.sum(valid_spots))

    def get_potential_mines(self) -> int:
        return int(np.sum(self.grid[:, :, 3] > 0)) - self.get_mine_count(player_id=1)

    def build_warehouse(self, player_id):
        owned_coords = np.argwhere(self.grid[:, :, 0] == player_id)
        potential_sites = [tuple(coord) for coord in owned_coords if self.grid[coord[0], coord[1], 1] == 0]
        
        valid_sites = []
        for r, c in potential_sites:
            for nr, nc in self._get_neighbors(r, c):
                if self.grid[nr, nc, 1] == 1 or self.grid[nr, nc, 1] == 4: # Neighbor has a mine
                    valid_sites.append((r, c))
                    break
        
        if not valid_sites:
            return False, "no_valid_adjacent_tiles"

        target_r, target_c = valid_sites[0]
        self.grid[target_r, target_c, 1] = 2  # Status 2 = Warehouse Lvl 1
        return True, f"warehouse_built_at_{target_r}_{target_c}"

    def build_crop_field(self, player_id):
        owned_coords = np.argwhere(self.grid[:, :, 0] == player_id)
        potential_sites = [tuple(coord) for coord in owned_coords if self.grid[coord[0], coord[1], 1] == 0]
        
        valid_sites = []
        for r, c in potential_sites:
            for nr, nc in self._get_neighbors(r, c):
                # Adjacent to warehouse or base
                if self.grid[nr, nc, 1] in [2, 3, 6]: 
                    valid_sites.append((r, c))
                    break
        
        if not valid_sites:
            return False, "no_valid_adjacent_warehouses"

        target_r, target_c = valid_sites[0]
        self.grid[target_r, target_c, 1] = 5  # Status 5 = Crop Field Lvl 1
        return True, f"crop_field_built_at_{target_r}_{target_c}"

    def get_crop_field_count(self, player_id):
        owned_mask = (self.grid[:, :, 0] == player_id)
        is_crop_mask = (self.grid[:, :, 1] == 5) | (self.grid[:, :, 1] == 7)
        return int(np.sum(np.logical_and(owned_mask, is_crop_mask)))

    def update_mine(self, player_id):
        owned_mines = np.argwhere(
            (self.grid[:, :, 0] == player_id) & 
            (self.grid[:, :, 1] == 1)
        )
        if len(owned_mines) == 0:
            return False, "no_mines_available_to_update"
        r, c = owned_mines[0]
        self.grid[r, c, 1] = 4 # Status 4 = Mine Lvl 2
        return True, f"mine_updated_at_{r}_{c}"

    def update_warehouse(self, player_id):
        owned_warehouses = np.argwhere(
            (self.grid[:, :, 0] == player_id) & 
            (self.grid[:, :, 1] == 2)
        )
        if len(owned_warehouses) == 0:
            return False, "no_warehouses_available_to_update"
        r, c = owned_warehouses[0]
        self.grid[r, c, 1] = 6 # Status 6 = Warehouse Lvl 2
        return True, f"warehouse_updated_at_{r}_{c}"

    def update_crop_field(self, player_id):
        owned_crops = np.argwhere(
            (self.grid[:, :, 0] == player_id) & 
            (self.grid[:, :, 1] == 5)
        )
        if len(owned_crops) == 0:
            return False, "no_crops_available_to_update"
        r, c = owned_crops[0]
        self.grid[r, c, 1] = 7 # Status 7 = Crop Field Lvl 2
        return True, f"crop_updated_at_{r}_{c}"

    def get_mines_by_owner_and_level(self, player_id, mine_level):
        mines = np.argwhere(
            (self.grid[:, :, 0] == player_id) & 
            (self.grid[:, :, 1] == mine_level)
        )
        return int(mines.shape[0])
    
    def get_building_count_by_type(self, player_id, status_code):
        owned_mask = (self.grid[:, :, 0] == player_id)
        status_mask = (self.grid[:, :, 1] == status_code)
        return int(np.sum(np.logical_and(owned_mask, status_mask)))
