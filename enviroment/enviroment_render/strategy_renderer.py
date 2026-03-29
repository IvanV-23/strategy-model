from cmath import rect
import pygame
import numpy as np
import math

BOARD_ROWS = 16  
BOARD_COLS = 16
CELL_SIZE = 40   
UI_PANEL_HEIGHT = 220 
UI_PANEL_WIDTH = 250  

class StrategyRenderer:
    def __init__(self, width, height, metadata):
        self.width = (BOARD_COLS * CELL_SIZE) + (UI_PANEL_WIDTH * 2)
        self.height = (BOARD_ROWS * CELL_SIZE) + UI_PANEL_HEIGHT
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
            "opp_workers": (255, 80, 80),
            "text": (240, 240, 240),
            "action": (0, 255, 255),
            "player_workers": (200, 200, 200),
            "helmet_dark": (100, 100, 100),
            "mine": (200, 150, 255),
            "grid": (50, 50, 60),
            "mine_icon": (0, 255, 100),
            "warehouse": (210, 180, 140),
            "warehouse_roof": (100, 50, 30),
            "base": (200, 200, 200),        
            "base_accent": (50, 50, 50),    
            "base_flag": (255, 255, 0),      
            "food": (150, 255, 100),
            "crop_field": (100, 180, 50),
            "fortified": (0, 100, 255), # Blue for Level 1
            "fortified_lvl2": (255, 0, 255), # Purple for Level 2
            "shot": (255, 255, 0) # Yellow for shots
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

    def _draw_coin(self, x, y, radius=10):
        pygame.draw.circle(self.screen, self.COLORS["gold_shadow"], (x, y), radius)
        pygame.draw.circle(self.screen, self.COLORS["gold"], (x, y), radius - 2)

    def _draw_wood(self, x, y, scale=1.0):
        w, h = int(24 * scale), int(12 * scale)
        log_rect = pygame.Rect(x - w//2, y - h//2, w, h)
        pygame.draw.rect(self.screen, self.COLORS["wood"], log_rect)

    def _draw_helmet(self, x, y, color):
        pygame.draw.rect(self.screen, color, [x - 10, y, 20, 8])

    def _draw_mine_icon(self, x, y):
        pts = [(x, y - 10), (x + 8, y - 2), (x + 5, y + 8), (x - 5, y + 8), (x - 8, y - 2)]
        pygame.draw.polygon(self.screen, (120, 50, 180), pts)

    def _draw_warehouse_icon(self, x, y):
        pygame.draw.rect(self.screen, self.COLORS["warehouse"], pygame.Rect(x - 8, y - 2, 16, 10))

    def _draw_warehouse_lvl2_icon(self, x, y):
        pygame.draw.rect(self.screen, (150, 150, 160), pygame.Rect(x - 10, y - 4, 20, 14))

    def _draw_base_icon(self, x, y):
        pygame.draw.rect(self.screen, self.COLORS["base"], pygame.Rect(x - 12, y - 5, 24, 15))

    def _draw_mine_lvl2_icon(self, x, y):
        pts = [(x, y - 12), (x + 10, y - 2), (x + 6, y + 8), (x - 6, y + 8), (x - 10, y - 2)]
        pygame.draw.polygon(self.screen, (150, 80, 255), pts)

    def _draw_crop_field_icon(self, x, y):
        for i in range(-8, 9, 4):
            pygame.draw.line(self.screen, self.COLORS["food"], (x + i, y + 8), (x + i, y - 4), 2)

    def _draw_crop_field_lvl2_icon(self, x, y):
        for i in range(-10, 11, 4):
            pygame.draw.line(self.screen, (50, 255, 50), (x + i, y + 8), (x + i, y - 6), 3)

    def _draw_food_icon(self, x, y):
        pygame.draw.line(self.screen, self.COLORS["food"], (x, y + 8), (x, y - 8), 3)

    def _draw_shot(self, start_pos, end_pos):
        pygame.draw.line(self.screen, self.COLORS["shot"], start_pos, end_pos, 2)
        pygame.draw.circle(self.screen, self.COLORS["shot"], end_pos, 4)

    def _draw_soldier(self, x, y, count, color=(255, 50, 50)):
        radius = 12
        pygame.draw.circle(self.screen, color, (x, y), radius)
        pygame.draw.circle(self.screen, (255, 255, 255), (x, y), radius, 1)
        if count > 1:
            sf = pygame.font.Font(None, 18)
            txt = sf.render(str(int(count)), True, (255, 255, 255))
            self.screen.blit(txt, txt.get_rect(center=(x, y)))

    def draw_board(self, board_data=None, routes=None, bases=None, target_index=None, shot_events=None):
        rows, cols = BOARD_ROWS, BOARD_COLS
        cell_size = CELL_SIZE
        if board_data: rows, cols = len(board_data), len(board_data[0])
        start_x = (self.width - (cols * cell_size)) // 2
        start_y = (self.height - (rows * cell_size)) // 2
        small_font = pygame.font.Font(None, 20)
        tiny_font = pygame.font.Font(None, 16)

        for r in range(rows):
            for c in range(cols):
                rect = pygame.Rect(start_x + c * cell_size, start_y + r * cell_size, cell_size, cell_size)
                color = self.COLORS["bg"]
                if board_data:
                    tile = board_data[r][c]
                    tile_type = tile.get("tile_type", 0)
                    if tile_type == 1:  # Water
                        color = (50, 100, 200)
                    elif tile["owner"] == 1: color = (30, 60, 120)
                    elif tile["owner"] == 2: color = (100, 30, 30)
                pygame.draw.rect(self.screen, color, rect)

        if routes and bases:
            p1_routes, p2_routes = routes; p1_base, p2_base = bases
            if p1_routes:
                for mr, mc in p1_routes:
                    pygame.draw.line(self.screen, (0, 255, 255), (start_x + p1_base[1]*cell_size + 20, start_y + p1_base[0]*cell_size + 20), (start_x + mc*cell_size + 20, start_y + mr*cell_size + 20), 2)

        for r in range(rows):
            for c in range(cols):
                if not board_data: continue
                tile = board_data[r][c]
                rect = pygame.Rect(start_x + c * cell_size, start_y + r * cell_size, cell_size, cell_size)
                
                # Fortification highlight
                f_lvl = tile.get("fortified", 0)
                if f_lvl == 1:
                    pygame.draw.rect(self.screen, self.COLORS["fortified"], rect, 3)
                elif f_lvl == 2:
                    pygame.draw.rect(self.screen, self.COLORS["fortified_lvl2"], rect, 4)

                status = tile.get("status", 0)
                if status == 1: self._draw_mine_icon(rect.centerx, rect.centery)
                elif status == 2: self._draw_warehouse_icon(rect.centerx, rect.centery)
                elif status == 3: self._draw_base_icon(rect.centerx, rect.centery)
                elif status == 4: self._draw_mine_lvl2_icon(rect.centerx, rect.centery)
                elif status == 5: self._draw_crop_field_icon(rect.centerx, rect.centery)
                elif status == 6: self._draw_warehouse_lvl2_icon(rect.centerx, rect.centery)
                elif status == 7: self._draw_crop_field_lvl2_icon(rect.centerx, rect.centery)

                if tile["workers"] > 0:
                    txt = small_font.render(str(int(tile["workers"])), True, (255, 255, 255))
                    self.screen.blit(txt, txt.get_rect(midbottom=(rect.centerx, rect.bottom - 2)))

                if tile.get("soldiers", 0) > 0:
                    self._draw_soldier(rect.centerx, rect.centery, tile["soldiers"], (255, 50, 50))
                if tile.get("p2_soldiers", 0) > 0:
                    self._draw_soldier(rect.centerx, rect.centery, tile["p2_soldiers"], (255, 120, 0))

                pygame.draw.rect(self.screen, self.COLORS["grid"], rect, 1)

        if shot_events:
            for event in shot_events:
                fr, fc = event["from"]
                tr, tc = event["to"]
                start = (start_x + fc * cell_size + 20, start_y + fr * cell_size + 20)
                end = (start_x + tc * cell_size + 20, start_y + tr * cell_size + 20)
                self._draw_shot(start, end)

    def render_frame(self, render_mode, state_data):
        self._init_pygame(render_mode)
        if render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT: pygame.quit()

        self.screen.fill(self.COLORS["bg"])
        p_res, o_res = state_data['p_res'], state_data['o_res']
        self.draw_board(board_data=state_data.get('board'), 
                        routes=(state_data.get('p1_routes'), []), 
                        bases=((0,0), (15,15)),
                        shot_events=state_data.get('shot_events'))
        
        if render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata.get("render_fps", 4))
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))
