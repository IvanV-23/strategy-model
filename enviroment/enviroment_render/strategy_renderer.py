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
            "mine_icon": (0, 255, 100)
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
                    if 1 <= status <= 5:
                        self._draw_mine_icon(rect.centerx, rect.centery)
                        level_txt = tiny_font.render(str(int(status)), True, (255, 255, 255))
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
            gen_col = (100, 255, 100)  # Light green for generation
            
            # --- LAYOUT ---
            box_width, box_height, padding = 220, 180, 60
            p_box_rect = pygame.Rect(padding, 40, box_width, box_height)
            o_box_rect = pygame.Rect(self.width - box_width - padding, 40, box_width, box_height)

            # Draw Player/Opponent Info Boxes
            pygame.draw.rect(self.screen, (35, 35, 45), p_box_rect, border_radius=15)
            pygame.draw.rect(self.screen, (60, 60, 80), p_box_rect, 2, border_radius=15)
            pygame.draw.rect(self.screen, (45, 30, 30), o_box_rect, border_radius=15)
            pygame.draw.rect(self.screen, (100, 50, 50), o_box_rect, 2, border_radius=15)

            # --- PLAYER RESOURCES ---
            p_res = state_data['p_res']
            p_gen = state_data.get('p_gen', [0, 0, 0]) # Default to 0 if not provided
            
            self.screen.blit(self.font.render("PLAYER", True, (100, 200, 255)), (p_box_rect.x + 20, 50))
            
            # Gold
            self._draw_coin(p_box_rect.x + 30, 95)
            self.screen.blit(self.font.render(f": {p_res[0]}", True, self.COLORS["gold"]), (p_box_rect.x + 55, 83))
            self.screen.blit(gen_font.render(f"+{p_gen[0]}", True, gen_col), (p_box_rect.x + 160, 88))
            
            # Wood
            self._draw_wood(p_box_rect.x + 30, 130)
            self.screen.blit(self.font.render(f": {p_res[1]}", True, self.COLORS["wood"]), (p_box_rect.x + 55, 118))
            self.screen.blit(gen_font.render(f"+{p_gen[1]}", True, gen_col), (p_box_rect.x + 160, 123))
            
            # Soldiers
            self._draw_helmet(p_box_rect.x + 30, 165, self.COLORS["player_soldiers"])
            self.screen.blit(self.font.render(f": {p_res[2]}", True, self.COLORS["player_soldiers"]), (p_box_rect.x + 55, 153))
            self.screen.blit(gen_font.render(f"+{p_gen[2]}", True, gen_col), (p_box_rect.x + 160, 158))
            
            # --- INFRASTRUCTURE ---
            mine_area_rect = pygame.Rect(p_box_rect.x - 5, p_box_rect.bottom + 10, box_width, 80)
            if state_data.get('eco_act') == 1:
                pulse = (math.sin(pygame.time.get_ticks() * 0.01) + 1) / 2
                pygame.draw.rect(self.screen, (0, 80 * pulse, 30 * pulse), mine_area_rect, border_radius=12)
                pygame.draw.rect(self.screen, self.COLORS["mine_icon"], mine_area_rect, 2, border_radius=12)

            mine_label = pygame.font.Font(None, 24).render("INFRASTRUCTURE: MINES", True, self.COLORS["mine"])
            self.screen.blit(mine_label, (p_box_rect.x, p_box_rect.bottom + 15))
            self.draw_resource_icons(p_res[3], p_box_rect.x + 10, p_box_rect.bottom + 45)

            # --- OPPONENT RESOURCES ---
            o_res = state_data['o_res']
            o_gen = state_data.get('o_gen', [0, 0, 0])
            
            self.screen.blit(self.font.render("OPPONENT", True, self.COLORS["opp_soldiers"]), (o_box_rect.x + 20, 50))
            
            # Gold
            self._draw_coin(o_box_rect.x + 30, 95)
            self.screen.blit(self.font.render(f": {o_res[0]}", True, self.COLORS["gold"]), (o_box_rect.x + 55, 83))
            self.screen.blit(gen_font.render(f"+{o_gen[0]}", True, gen_col), (o_box_rect.x + 160, 88))
            
            # Wood
            self._draw_wood(o_box_rect.x + 30, 130)
            self.screen.blit(self.font.render(f": {o_res[1]}", True, self.COLORS["wood"]), (o_box_rect.x + 55, 118))
            self.screen.blit(gen_font.render(f"+{o_gen[1]}", True, gen_col), (o_box_rect.x + 160, 123))
            
            # Soldiers
            self._draw_helmet(o_box_rect.x + 30, 165, self.COLORS["opp_soldiers"])
            self.screen.blit(self.font.render(f": {o_res[2]}", True, self.COLORS["opp_soldiers"]), (o_box_rect.x + 55, 153))
            self.screen.blit(gen_font.render(f"+{o_gen[2]}", True, gen_col), (o_box_rect.x + 160, 158))

            # --- BOARD & LOGS ---
            #self.draw_board(board_data=state_data.get('board'), target_index=state_data.get('target_act'))
            
            self.draw_board(
            board_data=state_data.get('board'),
            target_index=state_data.get('target_act'),
            routes=(state_data.get('p1_routes'), state_data.get('o1_routes')),
            bases=(state_data.get('p1_base'), state_data.get('o1_base'))
            )
            
            log_box = pygame.Rect(self.width - 250, self.height - 120, 230, 100)
            pygame.draw.rect(self.screen, (20, 20, 30), log_box, border_radius=10)
            pygame.draw.rect(self.screen, (50, 50, 70), log_box, 1, border_radius=10)
            history = state_data.get('history', ["Game started..."])
            for i, msg in enumerate(history[-3:]):
                msg_surf = pygame.font.Font(None, 22).render(f"> {msg}", True, (200, 200, 200))
                self.screen.blit(msg_surf, (log_box.x + 10, log_box.y + 30 + (i * 22)))

            # Header
            turn_text = self.font.render(f"TURN: {state_data['turn']}", True, self.COLORS["text"])
            self.screen.blit(turn_text, (self.width // 2 - 50, 20))

            if render_mode == "human":
                pygame.display.flip()
                self.clock.tick(self.metadata.get("render_fps", 4))
            else:
                return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))
            