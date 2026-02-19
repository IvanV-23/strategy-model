import numpy as np

class TradeRouteManager:
    def __init__(self, base_coords: tuple):
        self.base_coords = base_coords
        self.active_routes = []  # List of tuples: (r, c) of the mine

    def create_route(self, mine_coords: tuple) -> bool:
        """Adds a route if it doesn't already exist."""
        if mine_coords not in self.active_routes:
            self.active_routes.append(mine_coords)
            return True
        return False

    def calculate_income(self, grid: np.ndarray) -> float:
        """Calculates income based on Manhattan distance for active routes."""
        total_bonus = 0.0
        br, bc = self.base_coords
        
        for (mr, mc) in self.active_routes:
            # Distance bonus: |r1-r2| + |c1-c2|
            distance = abs(mr - br) + abs(mc - bc)
            mine_level = grid[mr, mc, 1]
            total_bonus += (mine_level * distance) 
            
        return total_bonus

    def validate_routes(self, grid: np.ndarray, owner_id: int):
        """Removes routes if the mine is destroyed or lost to an enemy."""
        self.active_routes = [
            (r, c) for (r, c) in self.active_routes 
            if grid[r, c, 0] == owner_id and grid[r, c, 1] > 0
        ]
