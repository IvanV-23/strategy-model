import pygame
import numpy as np

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

    def _draw_wood(self, x, y):
        # Draw a log shape (rectangle + ellipse ends)
        log_rect = pygame.Rect(x - 12, y - 6, 24, 12)
        pygame.draw.rect(self.screen, self.COLORS["wood"], log_rect)
        pygame.draw.ellipse(self.screen, self.COLORS["wood_light"], [x + 6, y - 6, 8, 12])
        # Add a few "bark" lines
        pygame.draw.line(self.screen, self.COLORS["wood_light"], (x - 8, y - 2), (x + 2, y - 2), 1)

    def _draw_helmet(self, x, y, color):
        # Draw a simple soldier helmet/shield icon
        # Top dome
        pygame.draw.arc(self.screen, color, [x - 10, y - 10, 20, 20], 0, 3.14, 10)
        # Bottom face guard
        pygame.draw.rect(self.screen, color, [x - 10, y, 20, 8])
        # Eye slit
        pygame.draw.line(self.screen, self.COLORS["bg"], (x - 6, y + 3), (x + 6, y + 3), 2)

    def _draw_mine_icon(self, x, y):
            # Draw a jagged purple crystal/mine shape
            pts = [
                (x, y - 10), (x + 8, y - 2), (x + 5, y + 8), 
                (x - 5, y + 8), (x - 8, y - 2)
            ]
            # Darker purple base
            pygame.draw.polygon(self.screen, (120, 50, 180), pts)
            # Lighter purple highlight
            pygame.draw.polygon(self.screen, self.COLORS["mine"], pts, 2)
            # Shine
            pygame.draw.line(self.screen, (255, 255, 255), (x - 2, y - 4), (x + 1, y - 6), 1)

    # --- RENDER LOGIC ---

    def draw_board(self, board_data=None, target_index=None, rows=8, cols=8, cell_size=45):
            if board_data is not None:
                rows, cols = len(board_data), len(board_data[0])
            
            start_x = (self.width - (cols * cell_size)) // 2
            start_y = (self.height - (rows * cell_size)) // 2
            small_font = pygame.font.Font(None, 22)

            for r in range(rows):
                for c in range(cols):
                    rect = pygame.Rect(start_x + c * cell_size, start_y + r * cell_size, cell_size, cell_size)
                    color = self.COLORS["bg"]
                    
                    if board_data is not None:
                        tile = board_data[r][c]
                        if tile["owner"] == 1: color = (30, 60, 120)
                        elif tile["owner"] == 2: color = (100, 30, 30)
                        
                        pygame.draw.rect(self.screen, color, rect)
                        if tile.get("status") == 1:
                            pygame.draw.rect(self.screen, (255, 255, 255), rect.inflate(-30, -30))
                        if tile["soldiers"] > 0:
                            txt_col = (255, 255, 255) if tile["owner"] != 0 else (100, 100, 100)
                            self.screen.blit(small_font.render(str(tile["soldiers"]), True, txt_col), 
                                            small_font.render(str(tile["soldiers"]), True, txt_col).get_rect(center=rect.center))
                    
                    pygame.draw.rect(self.screen, (60, 60, 70), rect, 1)

            if target_index is not None:
                tr, tc = target_index // cols, target_index % cols
                t_rect = pygame.Rect(start_x + tc * cell_size, start_y + tr * cell_size, cell_size, cell_size)
                pygame.draw.rect(self.screen, (0, 255, 255), t_rect, 3)
                L = 12
                pygame.draw.line(self.screen, (255, 255, 255), t_rect.topleft, (t_rect.x + L, t_rect.y), 2)
                pygame.draw.line(self.screen, (255, 255, 255), t_rect.topleft, (t_rect.x, t_rect.y + L), 2)
                pygame.draw.line(self.screen, (255, 255, 255), t_rect.bottomright, (t_rect.right - L, t_rect.bottom), 2)
                pygame.draw.line(self.screen, (255, 255, 255), t_rect.bottomright, (t_rect.right, t_rect.bottom - L), 2)

    def draw_resource_icons(self, count, start_x, start_y):
            for i in range(int(count)):
                # Create a small grid of icons (5 per row)
                x_pos = start_x + ((i % 5) * 25)
                y_pos = start_y + ((i // 5) * 25)
                self._draw_mine_icon(x_pos, y_pos)

    def render_frame(self, render_mode, state_data):
            self._init_pygame(render_mode)
            if render_mode == "human":
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: pygame.quit()

            self.screen.fill(self.COLORS["bg"])
            
            # --- DIMENSIONS & LAYOUT ---
            box_width, box_height, padding = 220, 180, 60
            p_box_rect = pygame.Rect(padding, 40, box_width, box_height)
            o_box_rect = pygame.Rect(self.width - box_width - padding, 40, box_width, box_height)

            # --- DRAW ENCAPSULATED BOXES ---
            # Player
            pygame.draw.rect(self.screen, (35, 35, 45), p_box_rect, border_radius=15)
            pygame.draw.rect(self.screen, (60, 60, 80), p_box_rect, 2, border_radius=15)
            # Opponent
            pygame.draw.rect(self.screen, (45, 30, 30), o_box_rect, border_radius=15)
            pygame.draw.rect(self.screen, (100, 50, 50), o_box_rect, 2, border_radius=15)

            # --- PLAYER SIDE CONTENT ---
            p_res = state_data['p_res']
            self.screen.blit(self.font.render("PLAYER", True, (100, 200, 255)), (p_box_rect.x + 20, 50))
            self._draw_coin(p_box_rect.x + 30, 95)
            self.screen.blit(self.font.render(f": {p_res[0]}", True, self.COLORS["gold"]), (p_box_rect.x + 55, 83))
            self._draw_wood(p_box_rect.x + 30, 130)
            self.screen.blit(self.font.render(f": {p_res[1]}", True, self.COLORS["wood"]), (p_box_rect.x + 55, 118))
            self._draw_helmet(p_box_rect.x + 30, 165, self.COLORS["player_soldiers"])
            self.screen.blit(self.font.render(f": {p_res[2]}", True, self.COLORS["player_soldiers"]), (p_box_rect.x + 55, 153))
            
            # --- INFRASTRUCTURE: MINES ---
            mine_area_rect = pygame.Rect(p_box_rect.x - 5, p_box_rect.bottom + 10, box_width, 80)
            if state_data.get('eco_act') == 1:
                import math
                pulse = (math.sin(pygame.time.get_ticks() * 0.01) + 1) / 2
                pygame.draw.rect(self.screen, (0, 80 * pulse, 30 * pulse), mine_area_rect, border_radius=12)
                pygame.draw.rect(self.screen, self.COLORS["mine_icon"], mine_area_rect, 2, border_radius=12)

            mine_label = pygame.font.Font(None, 24).render("INFRASTRUCTURE: MINES", True, self.COLORS["mine"])
            self.screen.blit(mine_label, (p_box_rect.x, p_box_rect.bottom + 15))
            self.draw_resource_icons(p_res[3], p_box_rect.x + 10, p_box_rect.bottom + 45)

            # --- OPPONENT SIDE CONTENT ---
            o_res = state_data['o_res']
            self.screen.blit(self.font.render("OPPONENT", True, self.COLORS["opp_soldiers"]), (o_box_rect.x + 20, 50))
            self._draw_coin(o_box_rect.x + 30, 95)
            self.screen.blit(self.font.render(f": {o_res[0]}", True, self.COLORS["gold"]), (o_box_rect.x + 55, 83))
            self._draw_wood(o_box_rect.x + 30, 130)
            self.screen.blit(self.font.render(f": {o_res[1]}", True, self.COLORS["wood"]), (o_box_rect.x + 55, 118))
            self._draw_helmet(o_box_rect.x + 30, 165, self.COLORS["opp_soldiers"])
            self.screen.blit(self.font.render(f": {o_res[2]}", True, self.COLORS["opp_soldiers"]), (o_box_rect.x + 55, 153))

            # --- MAIN BOARD ---
            self.draw_board(board_data=state_data.get('board'), target_index=state_data.get('target_act'))
            
            # --- HISTORY LOG (Bottom Right) ---
            log_box = pygame.Rect(self.width - 250, self.height - 120, 230, 100)
            pygame.draw.rect(self.screen, (20, 20, 30), log_box, border_radius=10)
            pygame.draw.rect(self.screen, (50, 50, 70), log_box, 1, border_radius=10)
            
            log_title = pygame.font.Font(None, 20).render("RECENT EVENTS", True, (150, 150, 150))
            self.screen.blit(log_title, (log_box.x + 10, log_box.y + 5))
            
            # Display last 3 messages from state_data['history']
            history = state_data.get('history', ["Game started...", "", ""])
            for i, msg in enumerate(history[-3:]):
                msg_surf = pygame.font.Font(None, 22).render(f"> {msg}", True, (200, 200, 200))
                self.screen.blit(msg_surf, (log_box.x + 10, log_box.y + 30 + (i * 22)))

            # --- FOOTER ---
            turn_text = self.font.render(f"TURN: {state_data['turn']}", True, self.COLORS["text"])
            self.screen.blit(turn_text, (self.width // 2 - 50, 20))

            if render_mode == "human":
                pygame.display.flip()
                self.clock.tick(self.metadata.get("render_fps", 4))
            else:
                return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))
    