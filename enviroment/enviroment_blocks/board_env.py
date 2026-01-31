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
            """
            Finds tiles that are currently unowned (0) AND adjacent 
            to a tile already owned by owner_id.
            """
            rows, cols = self.grid.shape[0], self.grid.shape[1]
            candidate_tiles = []

            # 1. Find all tiles currently owned by this player/opponent
            owned_coords = np.argwhere(self.grid[:, :, 0] == owner_id)

            # 2. Check neighbors (Up, Down, Left, Right) for each owned tile
            for r, c in owned_coords:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    
                    # Check bounds and if the neighbor is unowned (0)
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if self.grid[nr, nc, 0] == 0:
                            candidate_tiles.append((nr, nc))

            # 3. Remove duplicates and pick one
            unique_candidates = list(set(candidate_tiles))

            if unique_candidates:
                # Pick a random valid neighbor to expand into
                idx = np.random.choice(len(unique_candidates))
                target_r, target_c = unique_candidates[idx]
                
                self.grid[target_r, target_c, 0] = owner_id
                return True
            
            return False
