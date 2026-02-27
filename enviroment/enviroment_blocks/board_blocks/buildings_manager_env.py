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
            Returns the number of Mines (Status 2) currently owned by the player.
        """
        owned_mask = (self.grid[:, :, 0] == player_id)
        is_mine_mask = (self.grid[:, :, 1] == 1) # Assuming 2 is the status for Mine
        
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
            # UPDATED: We now check Owner (L0), No Building (L1), AND Resource Value > 0 (L3)
            valid_tiles = np.argwhere(
                (self.grid[:, :, 0] == player_id) & 
                (self.grid[:, :, 1] == 0) & 
                (self.grid[:, :, 3] > 0)
            )

            if len(valid_tiles) == 0:
                # This is a critical fallback in case the mask and logic get out of sync
                return False, "no_valid_resource_tiles_available"

            # 4. Strategy: Which resource tile to pick?
            # Option A: Pick the FIRST available (current behavior)
            # r, c = valid_tiles[0]
            
            # Option B: Pick the tile with the HIGHEST resource value (Better for the Agent)
            best_idx = np.argmax([self.grid[tr, tc, 3] for tr, tc in valid_tiles])
            r, c = valid_tiles[best_idx]

            # 5. Build the specific Mine Type
            self.grid[r, c, 1] = mine_type 
            
            return True, f"built_type_{mine_type}_at_{r}_{c}"

    def get_resource_tile_count(self, player_id: int) -> int:
        """
            Returns the count of tiles owned by the player that have a 
            resource value greater than 0 in Layer 3.
        """
        # Filter for tiles owned by this player (Layer 0)
        owned_mask = (self.grid[:, :, 0] == player_id)
        
        # Filter for tiles that actually have resources generated (Layer 3)
        has_resource_mask = (self.grid[:, :, 3] > 0)
        
        # The count of valid buildable spots currently owned
        valid_spots = np.logical_and(owned_mask, has_resource_mask)
        return int(np.sum(valid_spots))

    def get_potential_mines(self) -> int:
        """
        Returns the count of all tiles that have resources (potential mine locations).
        These are tiles where grid[:, :, 3] > 0 (resource value > 0).
        """
        return int(np.sum(self.grid[:, :, 3] > 0)) - self.get_mine_count(player_id=1)

    def build_warehouse(self, player_id):
        """
            Finds an empty tile owned by player_id that is adjacent to 
            an existing building (status > 0) and builds a warehouse.
        """
        # 1. Find all tiles owned by the player
        owned_coords = np.argwhere(self.grid[:, :, 0] == player_id)
        
        # 2. Identify potential build sites: Owned, but no building (status == 0)
        potential_sites = [tuple(coord) for coord in owned_coords if self.grid[coord[0], coord[1], 1] == 0]
        
        valid_sites = []
        for r, c in potential_sites:
            # Check neighbors for any existing building
            for nr, nc in self._get_neighbors(r, c):
                if self.grid[nr, nc, 1] == 1: # Neighbor has a building
                    valid_sites.append((r, c))
                    break
        
        if not valid_sites:
            return False, "no_valid_adjacent_tiles"

        # 3. Build at the first valid site (or use a strategy like picking the one closest to base)
        target_r, target_c = valid_sites[0]
        self.grid[target_r, target_c, 1] = 2  # Status 2 = Warehouse
        return True, f"warehouse_built_at_{target_r}_{target_c}"

    def update_mine(self, player_id):
        """
            Automatically selects an existing mine owned by player_id and updates it to a new mine type.
            Returns (success: bool, message: str)
        """
        # 1. Find all mines owned by the player
        owned_mines = np.argwhere(
            (self.grid[:, :, 0] == player_id) & 
            (self.grid[:, :, 1] == 1)
        )
        
        # 2. Check if there are any mines available to update
        if len(owned_mines) == 0:
            return False, "no_mines_available_to_update"
        
        # 3. Select the first mine (or use a strategy like highest resource value)
        r, c = owned_mines[0]
        
        # 4. Update the mine type
        self.grid[r, c, 1] = 4
        return True, f"mine_updated_to_level_2_at_{r}_{c}"

    def get_mines_by_owner_and_level(self, player_id, mine_level):
        """
            Returns the count of mines owned by player_id with the specified mine_level.
            
            Args:
                player_id: The owner of the mines
                mine_level: The level/type of the mine to filter by
            
            Returns:
                int: Number of matching mines
        """
        mines = np.argwhere(
            (self.grid[:, :, 0] == player_id) & 
            (self.grid[:, :, 1] == mine_level)
        )
        return int(np.sum(mines.shape[0]))
