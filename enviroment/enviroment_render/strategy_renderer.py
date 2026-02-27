from cmath import rect
import pygame
import numpy as np
import math

class StrategyRenderer:
    def __init__(self, width, height, metadata):
        self.width = width
        self.height = height
        self.metadata = metadata
        self.screen = None
        self.clock = None
        self.font = None
        
        # Colors
        self.COLORS = {
            "bg": (20, 20, 25),
            "gold": (255, 215, 0),
            "gold_shadow": (184, 134, 11),
            "wood": (139, 69, 19),
            "wood_light": (160, 82, 45),
            "opp_soldiers": (255, 80, 80),
            "text": (240, 240, 240),
            "action": (0, 255, 255),
            "player_soldiers": (200, 200, 200),
            "helmet_dark": (100, 100, 100),
            "mine": (200, 150, 255),
            "grid": (50, 50, 60),
            "mine_icon": (0, 255, 100),
            "warehouse": (210, 180, 140),
            "warehouse_roof": (100, 50, 30),
            "base": (200, 200, 200),        # Light stone/grey
            "base_accent": (50, 50, 50),    # Dark trim
            "base_flag": (255, 255, 0)      # Yellow flag/emblem
        }

    def _init_pygame(self, render_mode):
        if self.screen is None:
            pygame.init()
            if render_mode == "human":
                pygame.display.set_caption("Strategy Game - AI Agent")
                self.screen = pygame.display.set_mode((self.width, self.height))
            else:
                self.screen = pygame.Surface((self.width, self.height))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 32)

    # --- ICON DRAWING METHODS ---

    def _draw_coin(self, x, y, radius=10):
        pygame.draw.circle(self.screen, self.COLORS["gold_shadow"], (x, y), radius)
        pygame.draw.circle(self.screen, self.COLORS["gold"], (x, y), radius - 2)
        pygame.draw.circle(self.screen, (255, 255, 200), (x - 3, y - 3), radius // 4)

    def _draw_wood(self, x, y, scale=1.0):
        # Scale allows for smaller icons on the tiles
        w, h = int(24 * scale), int(12 * scale)
        log_rect = pygame.Rect(x - w//2, y - h//2, w, h)
        pygame.draw.rect(self.screen, self.COLORS["wood"], log_rect)
        pygame.draw.ellipse(self.screen, self.COLORS["wood_light"], [x + w//4, y - h//2, w//3, h])
        pygame.draw.line(self.screen, self.COLORS["wood_light"], (x - w//3, y - h//4), (x + w//12, y - h//4), 1)

    def _draw_helmet(self, x, y, color):
        pygame.draw.arc(self.screen, color, [x - 10, y - 10, 20, 20], 0, 3.14, 10)
        pygame.draw.rect(self.screen, color, [x - 10, y, 20, 8])
        pygame.draw.line(self.screen, self.COLORS["bg"], (x - 6, y + 3), (x + 6, y + 3), 2)

    def _draw_mine_icon(self, x, y):
        pts = [(x, y - 10), (x + 8, y - 2), (x + 5, y + 8), (x - 5, y + 8), (x - 8, y - 2)]
        pygame.draw.polygon(self.screen, (120, 50, 180), pts)
        pygame.draw.polygon(self.screen, self.COLORS["mine"], pts, 2)
        pygame.draw.line(self.screen, (255, 255, 255), (x - 2, y - 4), (x + 1, y - 6), 1)

    def _draw_trade_routes(self, routes, base_coords, start_x, start_y, cell_size, color):
            """Draws lines from the base to each active mine."""
            if not routes:
                return

            for mine_r, mine_c in routes:
                # Calculate pixel centers for the start (Base) and end (Mine)
                # We add cell_size // 2 to target the center of the tile
                start_pos = (
                    start_x + base_coords[1] * cell_size + cell_size // 2,
                    start_y + base_coords[0] * cell_size + cell_size // 2
                )
                end_pos = (
                    start_x + mine_c * cell_size + cell_size // 2,
                    start_y + mine_r * cell_size + cell_size // 2
                )

                # Create a pulsing effect for the "supply flow"
                pulse = (math.sin(pygame.time.get_ticks() * 0.005) + 1) / 2
                line_width = int(2 + pulse * 2)
                
                # Draw the main line
                pygame.draw.line(self.screen, color, start_pos, end_pos, line_width)
                
                # Draw a small circle at the mine to show it's "connected"
                pygame.draw.circle(self.screen, color, end_pos, 4)

    def _draw_warehouse_icon(self, x, y):
        # Draw the main body of the shed
        body_rect = pygame.Rect(x - 8, y - 2, 16, 10)
        pygame.draw.rect(self.screen, self.COLORS["warehouse"], body_rect)
        pygame.draw.rect(self.screen, (0, 0, 0), body_rect, 1) # Outline
        
        # Draw the gabled roof (Triangle)
        roof_pts = [(x - 11, y - 2), (x, y - 10), (x + 11, y - 2)]
        pygame.draw.polygon(self.screen, self.COLORS["warehouse_roof"], roof_pts)
        pygame.draw.polygon(self.screen, (0, 0, 0), roof_pts, 1) # Outline
        
        # Small door
        door_rect = pygame.Rect(x - 2, y + 2, 4, 6)
        pygame.draw.rect(self.screen, (60, 40, 20), door_rect)
        
    def _draw_progress_bar(self, x, y, width, height, current, maximum, color):
        # Background (Empty bar)
        bg_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, (40, 40, 50), bg_rect, border_radius=2)
        
        # Fill (Current progress)
        fill_percent = min(1.0, current / maximum) if maximum > 0 else 0
        if fill_percent > 0:
            fill_rect = pygame.Rect(x, y, int(width * fill_percent), height)
            pygame.draw.rect(self.screen, color, fill_rect, border_radius=2)
        
        # Border
        pygame.draw.rect(self.screen, (100, 100, 100), bg_rect, 1, border_radius=2)
   
    def _draw_base_icon(self, x, y):
        # Main Tower Body
        body_rect = pygame.Rect(x - 12, y - 5, 24, 15)
        pygame.draw.rect(self.screen, self.COLORS["base"], body_rect)
        pygame.draw.rect(self.screen, self.COLORS["base_accent"], body_rect, 1)

        # Battlements (The "teeth" on top)
        for i in range(-12, 12, 6):
            battlement = pygame.Rect(x + i, y - 9, 4, 4)
            pygame.draw.rect(self.screen, self.COLORS["base"], battlement)
            pygame.draw.rect(self.screen, self.COLORS["base_accent"], battlement, 1)

        # Central Gate
        gate_rect = pygame.Rect(x - 4, y + 2, 8, 8)
        pygame.draw.rect(self.screen, (20, 20, 20), gate_rect)
        
        # Small Flag/Banner
        pygame.draw.line(self.screen, (200, 200, 200), (x + 8, y - 14), (x + 8, y - 9), 2)
        pygame.draw.polygon(self.screen, self.COLORS["base_flag"], [(x + 8, y - 14), (x + 16, y - 11), (x + 8, y - 8)])
    
    def _draw_mine_lvl2_icon(self, x, y):
        # Draw a wooden support structure (scaffolding) behind the mine
        pygame.draw.rect(self.screen, (80, 40, 0), (x - 10, y + 2, 20, 8)) # Base beam
        pygame.draw.line(self.screen, (80, 40, 0), (x - 8, y + 2), (x - 8, y - 8), 2) # Left pillar
        pygame.draw.line(self.screen, (80, 40, 0), (x + 8, y + 2), (x + 8, y - 8), 2) # Right pillar

        # Draw the main crystal (slightly larger than lvl 1)
        pts = [(x, y - 12), (x + 10, y - 2), (x + 6, y + 8), (x - 6, y + 8), (x - 10, y - 2)]
        pygame.draw.polygon(self.screen, (150, 80, 255), pts) # Darker purple fill
        pygame.draw.polygon(self.screen, (220, 180, 255), pts, 2) # Bright highlight border
        
        # Add a second smaller "glow" crystal next to it
        small_pts = [(x+4, y+2), (x+9, y+4), (x+7, y+9), (x+2, y+7)]
        pygame.draw.polygon(self.screen, self.COLORS["mine_icon"], small_pts)

    # --- RENDER LOGIC ---

    def draw_board(self, board_data=None, routes=None, bases=None, target_index=None, rows=8, cols=8, cell_size=45):
            """
            Complete Board Drawing Method with Layered Rendering:
            1. Background & Ownership
            2. Trade Route Connections
            3. Resources, Buildings, and Soldiers
            4. Grid & Action Highlight
            """
            if board_data is not None:
                rows, cols = len(board_data), len(board_data[0])
            
            start_x = (self.width - (cols * cell_size)) // 2
            start_y = (self.height - (rows * cell_size)) // 2
            small_font = pygame.font.Font(None, 20)
            tiny_font = pygame.font.Font(None, 16)

            # --- LAYER 1: TILE BACKGROUNDS ---
            for r in range(rows):
                for c in range(cols):
                    rect = pygame.Rect(start_x + c * cell_size, start_y + r * cell_size, cell_size, cell_size)
                    color = self.COLORS["bg"]
                    
                    if board_data is not None:
                        tile = board_data[r][c]
                        if tile["owner"] == 1: 
                            color = (30, 60, 120)    # Blue-ish for Player
                        elif tile["owner"] == 2: 
                            color = (100, 30, 30)  # Red-ish for Opponent
                    
                    pygame.draw.rect(self.screen, color, rect)

            # --- LAYER 2: TRADE ROUTES ---
            # We draw these after backgrounds so they are visible, 
            # but before icons so they don't cover text.
            if routes is not None and bases is not None:
                p1_routes, p2_routes = routes
                p1_base, p2_base = bases
                
                # Draw Player 1 Routes (Cyan)
                if p1_routes:
                    self._draw_trade_routes(p1_routes, p1_base, start_x, start_y, cell_size, (0, 255, 255))
                # Draw Player 2 Routes (Orange)
                if p2_routes:
                    self._draw_trade_routes(p2_routes, p2_base, start_x, start_y, cell_size, (255, 120, 0))

            # --- LAYER 3: RESOURCES, BUILDINGS, & UNITS ---
            for r in range(rows):
                for c in range(cols):
                    if board_data is None: continue
                    
                    tile = board_data[r][c]
                    rect = pygame.Rect(start_x + c * cell_size, start_y + r * cell_size, cell_size, cell_size)

                    # 1. Wood Resource Icons
                    wood_count = tile.get("wood", 0)
                    if wood_count > 0:
                        self._draw_wood(rect.x + 12, rect.y + 10, scale=0.5)
                        wood_txt = tiny_font.render(str(wood_count), True, self.COLORS["wood_light"])
                        self.screen.blit(wood_txt, (rect.x + 22, rect.y + 4))

                    # 2. Buildings (Mines Type 1-5)
                    status = tile.get("status", 0)
                    if status == 1: # Mine
                        self._draw_mine_icon(rect.centerx, rect.centery)
                        level_txt = tiny_font.render("M", True, (255, 255, 255))
                        self.screen.blit(level_txt, (rect.centerx + 8, rect.centery + 2))
                    elif status == 2: # Warehouse
                        self._draw_warehouse_icon(rect.centerx, rect.centery)
                    elif status == 3: # Base (NEW)
                        self._draw_base_icon(rect.centerx, rect.centery)
                    elif status == 4: # Mine Lvl 2 (NEW)
                        self._draw_mine_lvl2_icon(rect.centerx, rect.centery)
                        level_txt = tiny_font.render("2", True, (0, 255, 100)) # Green '2' for high tier
                        self.screen.blit(level_txt, (rect.centerx + 8, rect.centery + 2))        
                    # 3. Soldiers
                    if tile["soldiers"] > 0:
                        txt_col = (255, 255, 255) if tile["owner"] != 0 else (120, 120, 130)
                        soldier_surf = small_font.render(str(tile["soldiers"]), True, txt_col)
                        text_rect = soldier_surf.get_rect(midbottom=(rect.centerx, rect.bottom - 2))
                        self.screen.blit(soldier_surf, text_rect)

                    # 4. Grid Lines
                    pygame.draw.rect(self.screen, self.COLORS["grid"], rect, 1)

            # --- LAYER 4: SELECTION HIGHLIGHT ---
            if target_index is not None:
                tr, tc = target_index // cols, target_index % cols
                t_rect = pygame.Rect(start_x + tc * cell_size, start_y + tr * cell_size, cell_size, cell_size)
                # Draw a thick neon border for the current action target
                pygame.draw.rect(self.screen, self.COLORS["action"], t_rect, 3)

    def draw_resource_icons(self, count, start_x, start_y):
        for i in range(int(count)):
            x_pos = start_x + ((i % 5) * 25)
            y_pos = start_y + ((i // 5) * 25)
            self._draw_mine_icon(x_pos, y_pos)

    def render_frame(self, render_mode, state_data):
        self._init_pygame(render_mode)
        if render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT: pygame.quit()

        self.screen.fill(self.COLORS["bg"])
        
        # --- FONTS & COLORS ---
        gen_font = pygame.font.Font(None, 24)
        gen_col = (100, 255, 100)
        
        # --- DATA EXTRACTION ---
        p_res = state_data['p_res']
        p_cap = state_data.get('p1_capacity', [999, 999, 999])
        o_res = state_data['o_res']
        o_cap = state_data.get('o1_capacity', p_cap) 
        p_gen = state_data.get('p_gen', [0, 0, 0])
        o_gen = state_data.get('o_gen', [0, 0, 0])

        # --- LAYOUT ---
        box_width, box_height, padding = 220, 180, 60
        p_box_rect = pygame.Rect(padding, 40, box_width, box_height)
        o_box_rect = pygame.Rect(self.width - box_width - padding, 40, box_width, box_height)

        # Draw Boxes
        pygame.draw.rect(self.screen, (35, 35, 45), p_box_rect, border_radius=15)
        pygame.draw.rect(self.screen, (60, 60, 80), p_box_rect, 2, border_radius=15)
        pygame.draw.rect(self.screen, (45, 30, 30), o_box_rect, border_radius=15)
        pygame.draw.rect(self.screen, (100, 50, 50), o_box_rect, 2, border_radius=15)

        # --- RENDER PLAYER & OPPONENT RESOURCES ---
        # Unified loop for both sides to keep code clean
        sides = [
            ("PLAYER", p_box_rect, p_res, p_cap, p_gen, (100, 200, 255)),
            ("OPPONENT", o_box_rect, o_res, o_cap, o_gen, self.COLORS["opp_soldiers"])
        ]

        for label, box, res, cap, gen, title_col in sides:
            self.screen.blit(self.font.render(label, True, title_col), (box.x + 20, 50))
            
            rows = [
                ("gold", res[0], cap[0], gen[0], 95),
                ("wood", res[1], cap[1], gen[1], 130),
                ("soldiers", res[2], cap[2], gen[2], 165)
            ]

            for type_key, val, maximum, income, y_pos in rows:
                # Draw Icon
                if type_key == "gold": self._draw_coin(box.x + 30, y_pos)
                elif type_key == "wood": self._draw_wood(box.x + 30, y_pos)
                else: self._draw_helmet(box.x + 30, y_pos, title_col)
                
                # Draw Text
                res_text = f"{val}/{maximum}"
                text_col = (255, 50, 50) if val >= maximum else self.COLORS.get(type_key, title_col)
                if type_key == "soldiers": text_col = title_col # Use side-specific color for soldiers
                
                self.screen.blit(self.font.render(res_text, True, text_col), (box.x + 55, y_pos - 15))
                self.screen.blit(gen_font.render(f"+{income}", True, gen_col), (box.x + 175, y_pos - 10))
                
                # Draw Progress Bar under the text
                bar_color = text_col
                self._draw_progress_bar(box.x + 55, y_pos + 8, 110, 5, val, maximum, bar_color)

        # --- INFRASTRUCTURE ---
        # Mines
        mine_label = gen_font.render("MINES", True, self.COLORS["mine"])
        self.screen.blit(mine_label, (p_box_rect.x + 10, p_box_rect.bottom + 15))
        self.draw_resource_icons(p_res[3], p_box_rect.x + 15, p_box_rect.bottom + 40)

        # Warehouses
        wh_label = gen_font.render("WAREHOUSES", True, self.COLORS["warehouse"])
        self.screen.blit(wh_label, (p_box_rect.x + 10, p_box_rect.bottom + 85))
        p_warehouses = state_data.get('p_warehouses', 0) 
        for i in range(int(p_warehouses)):
            wx = p_box_rect.x + 20 + ((i % 5) * 25)
            wy = p_box_rect.bottom + 110 + ((i // 5) * 25)
            self._draw_warehouse_icon(wx, wy)

        # --- BOARD ---
        self.draw_board(
            board_data=state_data.get('board'),
            target_index=state_data.get('target_act'),
            routes=(state_data.get('p1_routes'), state_data.get('o1_routes')),
            bases=(state_data.get('p1_base'), state_data.get('o1_base'))
        )
        
        # --- LOGS ---
        log_box = pygame.Rect(self.width - 250, self.height - 120, 230, 100)
        pygame.draw.rect(self.screen, (20, 20, 30), log_box, border_radius=10)
        pygame.draw.rect(self.screen, (50, 50, 70), log_box, 1, border_radius=10)
        for i, msg in enumerate(state_data.get('history', [])[-3:]):
            msg_surf = pygame.font.Font(None, 22).render(f"> {msg}", True, (200, 200, 200))
            self.screen.blit(msg_surf, (log_box.x + 10, log_box.y + 30 + (i * 22)))

        # --- HEADER ---
        turn_text = self.font.render(f"TURN: {state_data['turn']}", True, self.COLORS["text"])
        self.screen.blit(turn_text, (self.width // 2 - 50, 20))

        if render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata.get("render_fps", 4))
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))
  