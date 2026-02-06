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

    # --- RENDER LOGIC ---

    def draw_board(self, board_data=None, rows=8, cols=8, cell_size=45):
        if board_data is not None:
            rows = len(board_data)
            cols = len(board_data[0])
        
        board_width = cols * cell_size
        board_height = rows * cell_size
        start_x = (self.width - board_width) // 2
        start_y = (self.height - board_height) // 2

        small_font = pygame.font.Font(None, 22)

        for r in range(rows):
            for c in range(cols):
                rect = pygame.Rect(start_x + c * cell_size, start_y + r * cell_size, cell_size, cell_size)
                color = self.COLORS["bg"]
                
                if board_data is not None:
                    tile = board_data[r][c]
                    if tile["owner"] == 1: color = (0, 100, 255)
                    elif tile["owner"] == 2: color = (120, 40, 40)
                    
                    pygame.draw.rect(self.screen, color, rect)
                    if tile["status"] == 1:
                        pygame.draw.rect(self.screen, (255, 255, 255), rect.inflate(-24, -24))

                    if tile["soldiers"] > 0:
                        txt_col = (255, 255, 255) if tile["owner"] != 0 else (100, 100, 100)
                        soldier_text = small_font.render(str(tile["soldiers"]), True, txt_col)
                        self.screen.blit(soldier_text, soldier_text.get_rect(center=rect.center))
                else:
                    pygame.draw.rect(self.screen, color, rect)

                pygame.draw.rect(self.screen, (60, 60, 70), rect, 1)

    def draw_resource_icons(self, count, start_x, start_y, char, color):
        for i in range(int(count)):
            icon = self.font.render(char, True, color)
            x_pos = start_x + ((i % 10) * 20)
            y_offset = start_y + (25 * ((start_x + (i * 20)) // (self.width // 2 - 50)))
            self.screen.blit(icon, (x_pos, y_offset))

    def render_frame(self, render_mode, state_data):
        self._init_pygame(render_mode)
        if render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT: pygame.quit()

        self.screen.fill(self.COLORS["bg"])
        self.draw_board(board_data=state_data.get('board'))

        # --- PLAYER SIDE ---
        p_res = state_data['p_res']
        self.screen.blit(self.font.render("--- PLAYER ---", True, self.COLORS["text"]), (50, 50))
        
        # Gold
        self._draw_coin(62, 102, radius=10)
        self.screen.blit(self.font.render(f": {p_res[0]}", True, self.COLORS["gold"]), (85, 90))
        
        # Wood
        self._draw_wood(62, 142)
        self.screen.blit(self.font.render(f": {p_res[1]}", True, self.COLORS["wood"]), (85, 130))
        
        # Soldiers
        self._draw_helmet(62, 182, self.COLORS["player_soldiers"])
        self.screen.blit(self.font.render(f": {p_res[2]}", True, self.COLORS["player_soldiers"]), (85, 170))
        
        # Mines
        self.screen.blit(self.font.render("Mines: ", True, self.COLORS["mine"]), (50, 210))
        self.draw_resource_icons(p_res[3], 130, 210, "M", self.COLORS["mine_icon"])

        # --- OPPONENT SIDE ---
        o_res = state_data['o_res']
        opp_x = self.width - 180
        self.screen.blit(self.font.render("--- OPPONENT ---", True, self.COLORS["opp_soldiers"]), (opp_x, 50))
        
        # Gold
        self._draw_coin(opp_x + 12, 102, radius=10)
        self.screen.blit(self.font.render(f": {o_res[0]}", True, self.COLORS["gold"]), (opp_x + 35, 90))
        
        # Wood
        self._draw_wood(opp_x + 12, 142)
        self.screen.blit(self.font.render(f": {o_res[1]}", True, self.COLORS["wood"]), (opp_x + 35, 130))
        
        # Soldiers
        self._draw_helmet(opp_x + 12, 182, self.COLORS["opp_soldiers"])
        self.screen.blit(self.font.render(f": {o_res[2]}", True, self.COLORS["opp_soldiers"]), (opp_x + 35, 170))

        # --- FOOTER ---
        turn_text = self.font.render(f"TURN: {state_data['turn']}", True, self.COLORS["text"])
        self.screen.blit(turn_text, (self.width // 2 - 50, 20))

        action_str = f"DECISIONS >> Dip: {state_data['dip_act']} | Eco: {state_data['eco_act']}"
        act_text = self.font.render(action_str, True, self.COLORS["action"])
        self.screen.blit(act_text, act_text.get_rect(center=(self.width // 2, self.height - 50)))

        if render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata.get("render_fps", 4))
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))