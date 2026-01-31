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
            "wood": (139, 69, 19),
            "opp_soldiers": (255, 80, 80),
            "text": (240, 240, 240),
            "action": (0, 255, 255),
            "player_soldiers": (200, 200, 200),
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

    def draw_board(self, board_data=None, rows=8, cols=8, cell_size=45):
            # 1. Use the provided rows/cols if board_data is None
            if board_data is not None:
                rows = len(board_data)
                cols = len(board_data[0])
                
            board_width = cols * cell_size
            board_height = rows * cell_size
            start_x = (self.width - board_width) // 2
            start_y = (self.height - board_height) // 2

            for r in range(rows):
                for c in range(cols):
                    rect = pygame.Rect(start_x + c * cell_size, start_y + r * cell_size, cell_size, cell_size)
                    
                    # Default background
                    color = self.COLORS["bg"]
                    
                    # 2. Only try to access board_data if it actually exists
                    if board_data is not None:
                        tile = board_data[r][c]
                        if tile["owner"] == 1: 
                            color = (40, 40, 80)  # Player Blue
                        elif tile["owner"] == 2: 
                            color = (80, 40, 40)  # Opponent Red
                        
                        pygame.draw.rect(self.screen, color, rect)
                        
                        if tile["status"] == 1:
                            building_rect = rect.inflate(-20, -20)
                            pygame.draw.rect(self.screen, (255, 255, 255), building_rect)
                    else:
                        # Just draw the empty grid square
                        pygame.draw.rect(self.screen, color, rect)

                    # 3. Always draw the grid border
                    pygame.draw.rect(self.screen, (60, 60, 70), rect, 1)
    
    def draw_resource_icons(self, count, start_x, start_y, char, color):
        """Helper to draw a row of icons like M, M, M"""
        for i in range(int(count)):
            icon = self.font.render(char, True, color)
            x_pos = start_x + ((i % 10) * 20)
            y_offset = start_y + (25 * ( (start_x + (i * 20)) // (self.width // 2 - 50)))
            self.screen.blit(icon, (x_pos, y_offset))

    def render_frame(self, render_mode, state_data):

        self._init_pygame(render_mode)

        if render_mode == "human":
            # This pumps the event loop so Windows/Linux knows the app is alive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()


        self.screen.fill(self.COLORS["bg"])

        # 2. Draw Board (Centered)
        self.draw_board(
            board_data=state_data.get('board'), 
            rows=8, 
            cols=8, 
            cell_size=45
        )

        # Player Info
        p_res = state_data['p_res']
        self.screen.blit(self.font.render("--- PLAYER ---", True, self.COLORS["text"]), (50, 50))
        self.screen.blit(self.font.render(f"Gold: {p_res[0]}", True, self.COLORS["gold"]), (50, 90))
        self.screen.blit(self.font.render(f"Wood: {p_res[1]}", True, self.COLORS["wood"]), (50, 130))
        self.screen.blit(self.font.render(f"Soldiers: {p_res[2]}", True, self.COLORS["player_soldiers"]), (50, 170))
        
        # Mine Icons
        self.screen.blit(self.font.render("Mines: ", True, self.COLORS["mine"]), (50, 210))
        self.draw_resource_icons(p_res[3], 130, 210, "M", self.COLORS["mine_icon"])

        # Opponent Info
        o_res = state_data['o_res']
        right_offset = 180 
        opp_x = self.width - right_offset

        self.screen.blit(self.font.render("--- OPPONENT ---", True, self.COLORS["opp_soldiers"]), (opp_x, 50))
        self.screen.blit(self.font.render(f"Gold: {o_res[0]}", True, self.COLORS["gold"]), (opp_x, 90))
        self.screen.blit(self.font.render(f"Wood: {o_res[1]}", True, self.COLORS["wood"]), (opp_x, 130))
        self.screen.blit(self.font.render(f"Soldiers: {o_res[2]}", True, self.COLORS["opp_soldiers"]), (opp_x, 170))

        # Turn & Actions
        turn_text = self.font.render(f"TURN: {state_data['turn']}", True, self.COLORS["text"])
        self.screen.blit(turn_text, (self.width // 2 - 50, 20))

        action_str = f"DECISIONS >> Diplomacy: {state_data['dip_act']} | Economy: {state_data['eco_act']}"
        act_text = self.font.render(action_str, True, self.COLORS["action"])
        self.screen.blit(act_text, act_text.get_rect(center=(self.width // 2, self.height - 50)))

        if render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata.get("render_fps", 4))
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))
