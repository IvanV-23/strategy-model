import numpy as np

from enviroment.enviroment_blocks.board_blocks.trade_route_manager_env import TradeRouteManager

class BoardEnv:
    def __init__(self, rows=8, cols=8):
        #Board
        self.rows = rows
        self.cols = cols
        
        
        # Index 0: Owner (0=None, 1=P1, 2=P2)
        # Index 1: Status (e.g., 1=Building/Base)
        # Index 2: Soldiers (number of units on this specific tile)
        # Index 3: Resource Value (The static value of the tile)
        self.grid = np.zeros((rows, cols, 4), dtype=np.int32)

        #Trade routes
        self.p1_trade_manager = TradeRouteManager(base_coords=(0, 0))

    def reset(self):
            # Reset everything 

            resource_layer = self.grid[:, :, 3].copy() 
            self.grid = np.zeros((self.rows, self.cols, 4), dtype=np.int32)
            self.grid[:, :, 3] = resource_layer
            
            # Set starting positions
            self.grid[0, 0, :3] = [1, 1, 10]
            self.grid[self.rows-1, self.cols-1, :3] = [2, 1, 10]

            #Generate new resources
            self._generate_resources()

            #REset trade routes
            self.p1_trade_manager.active_routes = []

            return self.grid

    def get_tile_data(self):
        """
        Converts the internal numpy grid into a format the renderer understands.
        grid[:,:,0] = Owner
        grid[:,:,1] = Status (Building Type 0-5)
        grid[:,:,2] = Soldiers
        grid[:,:,3] = Wood Resources
        """
        tile_data = []
        for r in range(self.rows):
            row_list = []
            for c in range(self.cols):
                row_list.append({
                    "owner": int(self.grid[r, c, 0]),
                    "status": int(self.grid[r, c, 1]), # This is the Mine level!
                    "soldiers": int(self.grid[r, c, 2]),
                    "wood": int(self.grid[r, c, 3])
                })
            tile_data.append(row_list)
        return tile_data

    def _generate_resources(self):
            """Randomly distributes resource values across the board at initialization."""
            # Example: 20% of tiles have high resources (5-10), others have 1
            for r in range(self.rows):
                for c in range(self.cols):
                    if np.random.random() > 0.8:
                        self.grid[r, c, 3] = np.random.randint(5, 11)
                    else:
                        self.grid[r, c, 3] = 0

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
        previous_owner = None 
        mine_was_lost = False # Initialize the new flag

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
            return False, False, None, 0, False

        # 2. Filter candidates by battle logic
        valid_conquests = []
        for (tr, tc), attack_power in candidate_attacks.items():
            target_owner = self.grid[tr, tc, 0]
            target_defense = self.grid[tr, tc, 2]

            # --- ADDED: Mine Defense Bonus ---
            # Check if a mine (status > 0) is built on this tile
            if self.grid[tr, tc, 1] > 0:
                target_defense += 50

            # Apply base defense bonus if applicable (consistent with claim_target_tile)
            if (tr, tc) == (0, 0) or (tr, tc) == (rows - 1, cols - 1):
                target_defense += 50

            if target_owner == 0 or attack_power > target_defense:
                valid_conquests.append(((tr, tc), attack_power))

        # 3. Execute a random valid conquest
        if valid_conquests:
            # Pick a random successful battle
            idx = np.random.choice(len(valid_conquests))
            (target_r, target_c), final_attack_power = valid_conquests[idx]
            
            # --- CAPTURE STATE BEFORE OVERWRITING ---
            previous_owner = int(self.grid[target_r, target_c, 0])
            target_structure = int(self.grid[target_r, target_c, 1])
            
            # Check if a mine was lost (assuming Structure ID 2 is a Mine)
            if previous_owner != 0 and target_structure != 0:
                mine_was_lost = True
            
            if (target_r, target_c) == enemy_base:
                truncated = True
            
            # Conquer the tile
            self.grid[target_r, target_c, 0] = int(owner_id)
            self.grid[target_r, target_c, 2] = 1 # Occupy with 1 soldier
            
            # Return the new 5th variable: mine_was_lost
            return True, truncated, previous_owner, final_attack_power, mine_was_lost
        
        return False, False, None, 0, False

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

    def claim_target_tile(self, owner_id: int, target_coords: tuple) -> tuple[bool, bool, int, str, int]:
                tr, tc = target_coords
                rows, cols = self.grid.shape[0], self.grid.shape[1]
                
                # Define base locations
                p1_base = (0, 0)
                p2_base = (rows - 1, cols - 1)
                enemy_base = p2_base if int(owner_id) == 1 else p1_base
                
                # 1. Validation Checks
                if not (0 <= tr < rows and 0 <= tc < cols):
                    return False, False, 0, "out_of_bounds", 0
                if self.grid[tr, tc, 0] == owner_id:
                    return False, False, 0, "already_owned", 0

                # 2. Adjacency & Attacking Force Calculation
                is_adjacent = False
                total_attacking_force = 0
                for nr, nc in self._get_neighbors(tr, tc):
                    if self.grid[nr, nc, 0] == owner_id:
                        is_adjacent = True
                        total_attacking_force += self.grid[nr, nc, 2]

                if not is_adjacent:
                    return False, False, 0, "not_adjacent", 0

                # 3. Defense Logic
                target_owner = int(self.grid[tr, tc, 0])
                # We track the actual soldiers on the tile for the "defeated" count
                actual_soldiers = int(self.grid[tr, tc, 2])
                target_defense = actual_soldiers

                
                # Apply mechanical bonus for bases
                if (tr, tc) == p1_base or (tr, tc) == p2_base:
                    target_defense += 50

                # 4. Battle Logic
                if target_owner == 0 or total_attacking_force > target_defense:
                    base_captured = (tr, tc) == enemy_base
                    
                    # The "defeated" count is the number of soldiers that were there
                    # If target_owner was 0, actual_soldiers is likely 0 anyway.
                    defeated_count = actual_soldiers
                    
                    # Update Grid
                    self.grid[tr, tc, 0] = int(owner_id)
                    self.grid[tr, tc, 2] = 1 # Occupy with 1 soldier
                    
                    return True, base_captured, target_owner, "success", defeated_count
                
                # If the attack fails, 0 enemy soldiers were removed from the board
                return False, False, target_owner, "insufficient_force", 0

    def get_owned_tiles(self, owner_id):
            return np.sum(self.grid[:, :, 0] == owner_id)
    def get_tile_ownership(self):
            return self.grid[:, :, 0]
    
    def full_board_state(self) -> np.ndarray:
            """
            Returns a 5-channel representation of the board:
            0: Owner ID
            1: Status (Buildings)
            2: Soldiers
            3: Adjacency Mask (Maintained at index 3 for script compatibility)
            4: Resources (Added at the end)
            """
            rows, cols = self.grid.shape[0], self.grid.shape[1]
            
            # 1. Generate the Adjacency Mask
            mask = np.zeros((rows, cols), dtype=np.int32)
            owner_channel = self.grid[:, :, 0]
            player_tiles = np.argwhere(owner_channel == 1)
            
            for r, c in player_tiles:
                for nr, nc in self._get_neighbors(r, c):
                    if owner_channel[nr, nc] != 1:
                        mask[nr, nc] = 1

            # 2. Extract layers from self.grid (which is [Owner, Status, Soldiers, Resources])
            core_layers = self.grid[:, :, :3]  # Owner, Status, Soldiers
            resource_layer = self.grid[:, :, 3:] # Resources
            
            # 3. Reconstruct with Mask at Index 3
            # Resulting order: [Owner(0), Status(1), Soldiers(2), Mask(3), Resources(4)]
            combined = np.concatenate([
                core_layers, 
                mask[..., np.newaxis], 
                resource_layer
            ], axis=2)
            
            # 4. Transpose to (5, 8, 8)
            return combined.transpose(2, 0, 1).astype(np.int32)

    def get_board_state_and_stats(self)-> dict:
        # The spatial board as you already have it (5, 8, 8)
        spatial_board = self.full_board_state() 
        
        # The global stats vector (Scalable!)
        global_stats = np.array([
            self.get_resource_tile_count(player_id=1),
            self.get_mine_count(player_id=1),
            self.collect_gold_income(player_id=1)+ self.get_owned_tiles(owner_id=1),
            self.collect_wood_income(player_id=1),
            len(self.p1_trade_manager.active_routes),
            self.get_owned_tiles(owner_id=1)
            # Future-proofing: add more here easily
            # self.get_gold_balance(player_id),
            # self.get_tech_level(player_id),
        ], dtype=np.float32)
        
        return {
            "visual": spatial_board,
            "stats": global_stats
        }
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

    def get_build_mask(self, player_id: int, player_gold: int, player_wood: int) -> np.ndarray:
            """
            Returns a mask of size 7:
            Indices 0-5: Build Mine (Type 0: None, 1-5: Levels)
            Index 6: Create Trade Route
            """
            # Initialize mask for 6 mine actions + 1 trade route action = 7
            mask = np.zeros(7, dtype=bool)
            
            # --- MINE LOGIC (Indices 0-5) ---
            mask[0] = True # "Do Nothing" is always valid for mines
            
            current_mines = self.get_mine_count(player_id)
            resource_tiles = self.get_resource_tile_count(player_id)
            
            # Check if we have room and space for more mines
            has_space = np.any((self.grid[:, :, 0] == player_id) & (self.grid[:, :, 1] == 0) & (self.grid[:, :, 3] > 0))
            
            if current_mines < resource_tiles and has_space:
                mine_costs = [0, 50] # Costs for types 0-1
                for i in range(1, 2):
                    if player_gold >= mine_costs[i]:
                        mask[i] = True

            # --- TRADE ROUTE LOGIC (Index 6) ---
            manager = self.p1_trade_manager if player_id == 1 else None # Add P2 logic if needed
            if manager:
                # Rule: Must have a mine that isn't already connected
                all_mines = np.argwhere((self.grid[:, :, 0] == player_id) & (self.grid[:, :, 1] > 0))
                
                # Check if any mine position is not in active_routes
                eligible_mines = [
                    tuple(pos) for pos in all_mines 
                    if tuple(pos) not in manager.active_routes and tuple(pos) != manager.base_coords
                ]
                
                # If there's at least one unconnected mine and player can afford it (e.g., costs 30 gold)
                route_cost = 0
                if len(eligible_mines) > 0 and player_gold >= route_cost:
                    mask[6] = True
            
            return mask

    def collect_wood_income(self, player_id):
        """
        Calculates and returns the total resources owned by a player.
        Call this at the start of every turn.
        """
        # Find tiles where Index 0 matches player_id
        owned_mask = (self.grid[:, :, 0] == player_id)
        # Sum the resource values (Index 3) of those tiles
        total_income = np.sum(self.grid[owned_mask, 3])
        return total_income

#------ Additional methods for resource management, posibly a new class in the future ------

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

    def get_mine_count(self, player_id):
        """
        Returns the number of Mines (Status 2) currently owned by the player.
        """
        owned_mask = (self.grid[:, :, 0] == player_id)
        is_mine_mask = (self.grid[:, :, 1] != 0) # Assuming 2 is the status for Mine
        
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

#------- methods for trade route management-----------

    def create_trade_route(self, player_id: int) -> tuple[bool, str]:
            """
            Automatically selects an eligible mine and establishes a trade route to the base.
            Validation:
            1. Checks if player has mines without existing routes.
            2. Checks if there is "room" for more routes (Routes < Mines).
            """
            manager = self.p1_trade_manager if player_id == 1 else None
            
            # 1. Clean up the manager first (remove routes to lost/destroyed mines)
            manager.validate_routes(self.grid, player_id)

            # 2. Find all current mines owned by the player
            # We assume grid[r, c, 1] > 0 means a mine is present
            all_mines = np.argwhere(
                (self.grid[:, :, 0] == player_id) & 
                (self.grid[:, :, 1] > 0)
            )

            # 3. Filter for mines that do NOT have a route yet
            eligible_mines = [
                (tuple(pos)) for pos in all_mines 
                if tuple(pos) not in manager.active_routes and tuple(pos) != manager.base_coords
            ]

            if not eligible_mines:
                return False, "no_unconnected_mines_available"

            # 4. Strategy: Which mine to connect?
            # We pick the furthest mine to maximize the Manhattan distance bonus immediately.
            base_r, base_c = manager.base_coords
            furthest_mine = max(
                eligible_mines, 
                key=lambda pos: abs(pos[0] - base_r) + abs(pos[1] - base_c)
            )

            # 5. Establish the route
            success = manager.create_route(furthest_mine)
            
            if success:
                return True, f"route_created_to_{furthest_mine}"
            return False, "failed_to_create_route"

    def collect_gold_income(self, player_id):
            # 1. Base Tile Income
            owned_mask = (self.grid[:, :, 0] == player_id)
            #tile_income = np.sum(self.grid[owned_mask, 3])
            tile_income = 0
            
            # 2. Trade Manager Income
            manager = self.p1_trade_manager if player_id == 1 else None
            manager.validate_routes(self.grid, player_id) # Clean up lost mines first
            trade_income = manager.calculate_income(self.grid) * 2
            print(f"Trade route income {trade_income}")
            
            return tile_income + trade_income
