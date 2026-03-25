#include <iostream>
#include <vector>
#include <thread>
#include <cmath>
#include <SDL.h>
#include <SDL_ttf.h>
#include <nlohmann/json.hpp>

#include "camera.hpp"
#include "graphics_utils.hpp"
#include "state_manager.hpp"
#include "network.hpp"

#include "renderers/tile_renderer.hpp"
#include "renderers/wall_renderer.hpp"
#include "renderers/tree_renderer.hpp"
#include "renderers/crop_renderer.hpp"
#include "renderers/sawmill_renderer.hpp"
#include "renderers/warehouse_renderer.hpp"
#include "renderers/building_renderer.hpp"
#include "renderers/tower_renderer.hpp"
#include "renderers/unit_renderer.hpp"

GameStateManager global_state;

class GameRenderer {
public:
    GameRenderer(int w, int h) : width(w), height(h), cam(w, h) {
        SDL_Init(SDL_INIT_VIDEO);
        TTF_Init();
        window = SDL_CreateWindow("Strategy 3D Isometric", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, w, h, SDL_WINDOW_SHOWN);
        renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
        font = TTF_OpenFont("C:/Windows/Fonts/arial.ttf", 18);
    }

    ~GameRenderer() {
        if (font) TTF_CloseFont(font);
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        TTF_Quit();
        SDL_Quit();
    }

    void handle_input() {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) running = false;
            if (e.type == SDL_KEYDOWN) {
                switch (e.key.keysym.sym) {
                    case SDLK_UP: cam.offsetY += 20; break;
                    case SDLK_DOWN: cam.offsetY -= 20; break;
                    case SDLK_LEFT: cam.offsetX += 20; break;
                    case SDLK_RIGHT: cam.offsetX -= 20; break;
                    case SDLK_KP_PLUS: case SDLK_EQUALS: cam.zoom *= 1.1f; break;
                    case SDLK_KP_MINUS: case SDLK_MINUS: cam.zoom /= 1.1f; break;
                }
            }
        }
    }

    void render() {
        SDL_SetRenderDrawColor(renderer, 15, 15, 20, 255);
        SDL_RenderClear(renderer);

        json state = global_state.get_data();
        if (state.is_null()) {
            draw_text("Waiting for connection on port 8080...", width / 2 - 150, height / 2, {100, 100, 120, 255});
            SDL_RenderPresent(renderer);
            return;
        }

        if (state.contains("board")) {
            const auto& board = state["board"];
            int rows = (int)board.size();
            int cols = (int)board[0].size();

            // PASS 1: GROUND TILES
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < cols; ++c) {
                    tile_renderer.render(renderer, cam, board, r, c);
                }
            }

            // PASS 2: OBJECTS (Back-to-Front)
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < cols; ++c) {
                    wall_renderer.render(renderer, cam, board, r, c);
                    tree_renderer.render(renderer, cam, board, r, c);
                    crop_renderer.render(renderer, cam, board, r, c);
                    sawmill_renderer.render(renderer, cam, board, r, c);
                    warehouse_renderer.render(renderer, cam, board, r, c);
                    tower_renderer.render(renderer, cam, board, r, c);
                    building_renderer.render(renderer, cam, board, r, c);
                    unit_renderer.render(renderer, cam, board, r, c);
                }
            }
        }

        // PASS 3: ANIMATIONS (Overlay)
        if (state.contains("target_tile")) {
            int tr = state["target_tile"][0];
            int tc = state["target_tile"][1];
            Point2D p = cam.project((float)tc + 0.5f, (float)tr + 0.5f, 5.0f);
            SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
            SDL_SetRenderDrawColor(renderer, 255, 255, 0, 80); 
            Graphics::fill_circle(renderer, p.x, p.y, (int)(12 * cam.zoom));
            SDL_SetRenderDrawColor(renderer, 255, 255, 255, 150);
            SDL_RenderDrawLine(renderer, p.x - 8, p.y, p.x + 8, p.y);
            SDL_RenderDrawLine(renderer, p.x, p.y - 8, p.x, p.y + 8);
        }

        if (state.contains("soldier_target_tile")) {
            int str = state["soldier_target_tile"][0];
            int stc = state["soldier_target_tile"][1];
            Point2D p = cam.project((float)stc + 0.5f, (float)str + 0.5f, 10.0f);
            int turn = state.contains("turn") ? state["turn"].get<int>() : 0;
            int pulse = (int)(sin(turn * 0.5f) * 5 + 15);
            SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
            SDL_SetRenderDrawColor(renderer, 255, 0, 0, 150); 
            Graphics::fill_circle(renderer, p.x, p.y, (int)(pulse * cam.zoom / 2));
            SDL_SetRenderDrawColor(renderer, 255, 50, 50, 255);
            SDL_RenderDrawLine(renderer, p.x - 15, p.y, p.x + 15, p.y);
            SDL_RenderDrawLine(renderer, p.x, p.y - 15, p.x, p.y + 15);
        }

        if (state.contains("shot_events")) {
            for (const auto& shot : state["shot_events"]) {
                if (shot.contains("from") && shot.contains("to")) {
                    Point2D p_from = cam.project(shot["from"][1].get<float>() + 0.5f, shot["from"][0].get<float>() + 0.5f, 20.0f);
                    Point2D p_to = cam.project(shot["to"][1].get<float>() + 0.5f, shot["to"][0].get<float>() + 0.5f, 10.0f);
                    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 200); 
                    SDL_RenderDrawLine(renderer, p_from.x, p_from.y, p_to.x, p_to.y);
                    Graphics::fill_circle(renderer, p_to.x, p_to.y, (int)(4 * cam.zoom));
                    Graphics::fill_circle(renderer, p_from.x, p_from.y, (int)(3 * cam.zoom));
                }
            }
        }

        draw_ui(state);
        SDL_RenderPresent(renderer);
    }

    bool is_running() const { return running; }

private:
    int width, height;
    bool running = true;
    IsoCamera cam;
    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;
    TTF_Font* font = nullptr;

    TileRenderer tile_renderer;
    WallRenderer wall_renderer;
    TreeRenderer tree_renderer;
    CropRenderer crop_renderer;
    SawmillRenderer sawmill_renderer;
    WarehouseRenderer warehouse_renderer;
    BuildingRenderer building_renderer;
    TowerRenderer tower_renderer;
    UnitRenderer unit_renderer;

    void draw_text(const std::string& text, int x, int y, SDL_Color col) {
        if (!font) return;
        SDL_Surface* s = TTF_RenderText_Blended(font, text.c_str(), col);
        SDL_Texture* t = SDL_CreateTextureFromSurface(renderer, s);
        SDL_Rect dst = {x, y, s->w, s->h};
        SDL_RenderCopy(renderer, t, NULL, &dst);
        SDL_FreeSurface(s);
        SDL_DestroyTexture(t);
    }

    void draw_ui(const json& state) {
        SDL_Rect ui_bg = {0, 0, width, 60};
        SDL_SetRenderDrawColor(renderer, 30, 30, 40, 200);
        SDL_RenderFillRect(renderer, &ui_bg);

        if (state.contains("p_res")) {
            const auto& res = state["p_res"];
            draw_text("GOLD: " + std::to_string((int)res[0].get<double>()), 20, 20, {255, 215, 0, 255});
            draw_text("WOOD: " + std::to_string((int)res[1].get<double>()), 150, 20, {150, 100, 50, 255});
        }

        // Count soldiers from board data
        int p1_sols = 0, p2_sols = 0;
        if (state.contains("board")) {
            for (const auto& row : state["board"]) {
                for (const auto& tile : row) {
                    if (tile.contains("soldiers")) p1_sols += tile["soldiers"].get<int>();
                    if (tile.contains("p2_soldiers")) p2_sols += tile["p2_soldiers"].get<int>();
                }
            }
        }
        draw_text("P1 SOL: " + std::to_string(p1_sols), 300, 20, {255, 50, 50, 255});
        draw_text("P2 SOL: " + std::to_string(p2_sols), 450, 20, {255, 120, 0, 255});

        if (state.contains("turn")) {
            draw_text("TURN: " + std::to_string(state["turn"].get<int>()), width / 2 + 100, 20, {255, 255, 255, 255});
        }
        
        draw_text("Arrows: Pan | +/-: Zoom", width - 220, 20, {180, 180, 180, 255});
    }
};

int main(int argc, char* argv[]) {
    GameRenderer gr(1200, 800);
    std::thread st(server_thread);
    st.detach();
    
    while (gr.is_running()) {
        gr.handle_input();
        gr.render();
        SDL_Delay(16);
    }
    return 0;
}
