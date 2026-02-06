import numpy as np

class BoardEnv:
    def __init__(self, rows=8, cols=8):
        self.rows = rows
        self.cols = cols
        self.grid = np.zeros((rows, cols, 3), dtype=np.int32)
        # Index 0: Owner (0=None, 1=P1, 2=P2)
        # Index 1: Status (e.g., 1=Building/Base)
        # Index 2: Soldiers (number of units on this specific tile)
        self.grid = np.zeros((rows, cols, 3), dtype=np.int32)

    def reset(self):
        self.grid = np.zeros((self.rows, self.cols, 3), dtype=np.int32)
        # Optional: Set starting positions
        self.grid[0, 0] = [1, 1, 10]  # Player starts at top-left with 10 soldiers
        self.grid[self.rows-1, self.cols-1] = [2, 1, 10] # Opponent at bottom-right
        return self.grid

    def get_tile_data(self):
            board_state = []
            for r in range(self.rows):
                row_data = []
                for c in range(self.cols):
                    row_data.append({
                        "owner": int(self.grid[r, c, 0]),
                        "status": int(self.grid[r, c, 1]),
                        "soldiers": int(self.grid[r, c, 2])
                    })
                board_state.append(row_data)
            return board_state

    def _get_neighbors(self, r, c):
            """Helper to get valid adjacent coordinates"""
            neighbors = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    neighbors.append((nr, nc))
            return neighbors

    def claim_adjacent_tile(self, owner_id):
            rows, cols = self.grid.shape[0], self.grid.shape[1]
            truncated = False
            enemy_base = (rows - 1, cols - 1) if int(owner_id) == 1 else (0, 0)
            previous_owner = None  # Default if no conquest happens

            # 1. Find all potential tiles to attack
            candidate_attacks = {} # {(r, c): total_attacking_soldiers}
            owned_coords = np.argwhere(self.grid[:, :, 0] == owner_id)
            
            for r, c in owned_coords:
                attacking_force = self.grid[r, c, 2]
                for nr, nc in self._get_neighbors(r, c):
                    if self.grid[nr, nc, 0] != owner_id:
                        if (nr, nc) not in candidate_attacks:
                            candidate_attacks[(nr, nc)] = 0
                        candidate_attacks[(nr, nc)] += attacking_force

            if not candidate_attacks:
                return False, False, None

            # 2. Filter candidates by battle logic
            valid_conquests = []
            for (tr, tc), attack_power in candidate_attacks.items():
                target_owner = self.grid[tr, tc, 0]
                target_defense = self.grid[tr, tc, 2]

                if target_owner == 0 or attack_power > target_defense:
                    valid_conquests.append((tr, tc))

            # 3. Execute a random valid conquest
            if valid_conquests:
                idx = np.random.choice(len(valid_conquests))
                target_r, target_c = valid_conquests[idx]
                
                # --- CAPTURE PREVIOUS OWNER HERE ---
                previous_owner = int(self.grid[target_r, target_c, 0])
                
                if (target_r, target_c) == enemy_base:
                    truncated = True
                
                # Conquer the tile
                self.grid[target_r, target_c, 0] = int(owner_id)
                self.grid[target_r, target_c, 2] = 1 
                
                return True, truncated, previous_owner
            
            return False, False, None

    def redistribute_soldiers(self, owner_id, total_soldiers, style=0):
        """
        Distributes total_soldiers based on a strategic style:
            0: Balanced (Equal)
            1: Frontline (Mass units at the borders for attack)
            2: Defensive (Mass units near the base)
            3: Random (Exploration for the RL agent)
        """
        # 1. Find all tiles currently owned by this player
        owned_indices = np.argwhere(self.grid[:, :, 0] == owner_id)
        num_tiles = len(owned_indices)

        if num_tiles == 0:
            return

        # 2. Rule: Every tile needs at least 1 soldier to remain 'owned'
        # This prevents "teleporting" your whole army and leaving land empty
        if total_soldiers < num_tiles:
            base = (0, 0) if owner_id == 1 else (self.rows - 1, self.cols - 1)
            distances = [np.linalg.norm(np.array(coord) - np.array(base)) for coord in owned_indices]
            keep_indices = np.argsort(distances)[:total_soldiers]
            
            for i in range(num_tiles):
                if i not in keep_indices:
                    r, c = owned_indices[i]
                    self.grid[r, c, 0] = 0 
                    self.grid[r, c, 2] = 0 
            
            owned_indices = owned_indices[keep_indices]
            num_tiles = len(owned_indices)

        if num_tiles == 0: return

        # 3. Strategic Weighting
        # We assign a 'weight' to each tile based on the chosen strategy
        weights = np.ones(num_tiles) # Default: Equal weight

        if style == 1: # FRONTLINE: Prioritize tiles adjacent to enemies or empty space
            for i, (r, c) in enumerate(owned_indices):
                # Check neighbors
                is_frontier = False
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        if self.grid[nr, nc, 0] != owner_id:
                            is_frontier = True
                            break
                weights[i] = 5.0 if is_frontier else 1.0 # Frontline tiles get 5x more troops

        elif style == 2: # DEFENSIVE: Prioritize tiles closest to the base
            base = (0, 0) if owner_id == 1 else (self.rows - 1, self.cols - 1)
            for i, coord in enumerate(owned_indices):
                dist = np.linalg.norm(np.array(coord) - np.array(base))
                weights[i] = 1.0 / (dist + 1) # Closer to base = higher weight

        elif style == 3: # RANDOM/CHAOTIC: RL agent uses this to find new "shapes"
            weights = np.random.dirichlet(np.ones(num_tiles), size=1)[0] * num_tiles

        # 4. Final Allocation based on weights
        # Convert weights to a probability distribution
        probs = weights / np.sum(weights)
        
        # Reserve 1 soldier per tile first to maintain ownership
        remaining_soldiers = total_soldiers - num_tiles
        
        if remaining_soldiers > 0:
            # Distribute the "extra" soldiers based on the calculated probabilities
            extra_allocations = np.random.multinomial(remaining_soldiers, probs)
            for i, (r, c) in enumerate(owned_indices):
                self.grid[r, c, 2] = 1 + extra_allocations[i]
        else:
            # Just 1 soldier per tile if we are at the limit
            for r, c in owned_indices:
                self.grid[r, c, 2] = 1

    def claim_target_tile(self, owner_id: int, target_coords: tuple) -> tuple[bool, bool, int, str]:
        """
        Attempts to capture a specific tile.
        Returns: (battle_victory, base_captured, previous_owner, reason)
        """
        tr, tc = target_coords
        rows, cols = self.grid.shape[0], self.grid.shape[1]
        enemy_base = (rows - 1, cols - 1) if int(owner_id) == 1 else (0, 0)
        
        # 1. Out of Bounds
        if not (0 <= tr < rows and 0 <= tc < cols):
            return False, False, 0, "out_of_bounds"

        # 2. Self-Attack
        if self.grid[tr, tc, 0] == owner_id:
            return False, False, 0, "already_owned"

        # 3. Adjacency Check
        is_adjacent = False
        total_attacking_force = 0
        
        for nr, nc in self._get_neighbors(tr, tc):
            if self.grid[nr, nc, 0] == owner_id:
                is_adjacent = True
                total_attacking_force += self.grid[nr, nc, 2]

        if not is_adjacent:
            return False, False, 0, "not_adjacent"

        # 4. Battle Logic
        target_owner = int(self.grid[tr, tc, 0])
        target_defense = self.grid[tr, tc, 2]
        previous_owner = target_owner

        if target_owner == 0 or total_attacking_force > target_defense:
            base_captured = (tr, tc) == enemy_base
            self.grid[tr, tc, 0] = int(owner_id)
            self.grid[tr, tc, 2] = 1 
            return True, base_captured, previous_owner, "success"
        
        # 5. Failed Battle
        return False, False, previous_owner, "insufficient_force"

    def get_owned_tiles(self, owner_id):
            return np.sum(self.grid[:, :, 0] == owner_id)
    def get_tile_ownership(self):
            return self.grid[:, :, 0]
    
    def full_board_state(self) -> np.ndarray:
            """
            Returns a 4-channel representation of the board:
            0: Owner ID
            1: Status (Buildings)
            2: Soldiers
            3: Adjacency Mask (1.0 if adjacent to Player 1 territory and not owned by P1)
            """
            # Current grid is (8, 8, 3) -> [Owner, Status, Soldiers]
            # Let's create the 4th channel: Adjacency Mask
            rows, cols = self.grid.shape[0], self.grid.shape[1]
            mask = np.zeros((rows, cols), dtype=np.int32)
            
            owner_channel = self.grid[:, :, 0]
            player_tiles = np.argwhere(owner_channel == 1)
            
            for r, c in player_tiles:
                for nr, nc in self._get_neighbors(r, c):
                    # If neighbor is not already owned by Player 1
                    if owner_channel[nr, nc] != 1:
                        mask[nr, nc] = 1

            # Combine the original 3 channels with the new mask
            # We use axis=2 because the original grid is (8, 8, 3)
            combined = np.concatenate([self.grid, mask[..., np.newaxis]], axis=2)
            
            # Transpose from (Rows, Cols, Channels) to (Channels, Rows, Cols)
            # This makes it (4, 8, 8)
            return combined.transpose(2, 0, 1).astype(np.int32)


    def get_action_mask(self, player_id):
        """
        Returns a 1D boolean array of size 64.
        True = Valid target for attack.
        False = Invalid target (already owned or not reachable).
        """
        # 1. Initialize a flat mask of 64 zeros (8x8)
        mask = np.zeros(self.rows * self.cols, dtype=bool)
        
        # 2. Find all tiles owned by the player
        owned_coords = np.argwhere(self.grid[:, :, 0] == player_id)
        
        # 3. For every owned tile, check its neighbors
        for r, c in owned_coords:
            for nr, nc in self._get_neighbors(r, c):
                # A neighbor is a valid target if NOT owned by the current player
                if self.grid[nr, nc, 0] != player_id:
                    # Convert 2D (nr, nc) to 1D index
                    flat_idx = nr * self.cols + nc
                    mask[flat_idx] = True
                    
        return mask
