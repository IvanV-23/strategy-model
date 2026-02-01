import numpy as np

class BoardEnv:
    def __init__(self, rows=8, cols=8):
        self.rows = rows
        self.cols = cols
        # Index 0: Owner, Index 1: Status
        self.grid = np.zeros((rows, cols, 2), dtype=np.int32)

    def reset(self):
        self.grid = np.zeros((self.rows, self.cols, 2), dtype=np.int32)
        # Optional: Set starting positions
        self.grid[0, 0] = [1, 1] # Player starts with a building at top-left
        self.grid[self.rows-1, self.cols-1] = [2, 1] # Opponent at bottom-right
        return self.grid

    def get_tile_data(self):
        """Returns a format the Renderer can easily digest"""
        board_state = []
        for r in range(self.rows):
            row_data = []
            for c in range(self.cols):
                row_data.append({
                    "owner": self.grid[r, c, 0],
                    "status": self.grid[r, c, 1]
                })
            board_state.append(row_data)
        return board_state

    def claim_adjacent_tile(self, owner_id):
            rows, cols = self.grid.shape[0], self.grid.shape[1]
            candidate_tiles = []
            truncated = False
            enemy_base = (rows - 1, cols - 1) if owner_id == 1 else (0, 0)

            # 1. Get all tiles currently owned by this player
            owned_coords = []
            for r in range(self.rows):
                for c in range(self.cols):
                    if int(self.grid[r, c, 0]) == int(owner_id):
                        owned_coords.append((r, c))
            #print(f"DEBUG: Player {owner_id} owns {len(owned_coords)} tiles.")

            # 2. Find valid neighbors to conquer
            for r, c in owned_coords:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        # ONLY capture if it's NOT already ours
                        if self.grid[nr, nc, 0] != owner_id:
                            candidate_tiles.append((nr, nc))

            unique_candidates = list(set(candidate_tiles))
            #print(f"DEBUG: Found {len(unique_candidates)} potential tiles to capture.") # LOG 2
            
            if unique_candidates:
                # OPTIONAL: Instead of pure random, target the enemy base direction
                # For now, we'll stay random but ensure the update is forced
                idx = np.random.choice(len(unique_candidates))
                target_r, target_c = unique_candidates[idx]
                
                if (target_r, target_c) == enemy_base:
                    truncated = True
                
                # FORCE UPDATE: Ensure the grid value is exactly the owner_id
                self.grid[target_r, target_c, 0] = int(owner_id)
                #print(f"DEBUG: Player {owner_id} captured tile at ({target_r}, {target_c})") 
                return True, truncated
            
            return False, False
