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
#include "board_factory.hpp"
#include "turn_manager.hpp"
#include "inference_engine.hpp"

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
        window = SDL_CreateWindow("Strategy 3D Isometric - AI Game", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, w, h, SDL_WINDOW_SHOWN);
        renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
        font = TTF_OpenFont("C:/Windows/Fonts/arial.ttf", 18);
        if (!font) font = TTF_OpenFont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18);
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
                    case SDLK_SPACE: paused = !paused; break;
                    case SDLK_ESCAPE: running = false; break;
                }
            }
        }
    }

    void render_game_state(const GameState& state, int turn) {
        SDL_SetRenderDrawColor(renderer, 15, 15, 20, 255);
        SDL_RenderClear(renderer);

        if (state.board.empty()) {
            draw_text("No game state", width / 2 - 50, height / 2, {100, 100, 120, 255});
            SDL_RenderPresent(renderer);
            return;
        }

        const auto& board = state.board;
        int rows = (int)board.size();
        int cols = rows > 0 ? (int)board[0].size() : 0;

        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                tile_renderer.render(renderer, cam, board, r, c);
            }
        }

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

        if (state.target_tile.has_value()) {
            auto [tr, tc] = state.target_tile.value();
            Point2D p = cam.project((float)tc + 0.5f, (float)tr + 0.5f, 5.0f);
            SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
            SDL_SetRenderDrawColor(renderer, 255, 255, 0, 80);
            Graphics::fill_circle(renderer, p.x, p.y, (int)(12 * cam.zoom));
            SDL_SetRenderDrawColor(renderer, 255, 255, 255, 150);
            SDL_RenderDrawLine(renderer, p.x - 8, p.y, p.x + 8, p.y);
            SDL_RenderDrawLine(renderer, p.x, p.y - 8, p.x, p.y + 8);
        }

        if (state.soldier_target_tile.has_value()) {
            auto [str, stc] = state.soldier_target_tile.value();
            Point2D p = cam.project((float)stc + 0.5f, (float)str + 0.5f, 10.0f);
            int pulse = (int)(sin(turn * 0.5f) * 5 + 15);
            SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
            SDL_SetRenderDrawColor(renderer, 255, 0, 0, 150);
            Graphics::fill_circle(renderer, p.x, p.y, (int)(pulse * cam.zoom / 2));
            SDL_SetRenderDrawColor(renderer, 255, 50, 50, 255);
            SDL_RenderDrawLine(renderer, p.x - 15, p.y, p.x + 15, p.y);
            SDL_RenderDrawLine(renderer, p.x, p.y - 15, p.x, p.y + 15);
        }

        for (const auto& shot : state.shot_events) {
            Point2D p_from = cam.project(shot.from.second + 0.5f, shot.from.first + 0.5f, 20.0f);
            Point2D p_to = cam.project(shot.to.second + 0.5f, shot.to.first + 0.5f, 10.0f);
            SDL_SetRenderDrawColor(renderer, 255, 255, 255, 200);
            SDL_RenderDrawLine(renderer, p_from.x, p_from.y, p_to.x, p_to.y);
            Graphics::fill_circle(renderer, p_to.x, p_to.y, (int)(4 * cam.zoom));
            Graphics::fill_circle(renderer, p_from.x, p_from.y, (int)(3 * cam.zoom));
        }

        draw_ui(state, turn);
        SDL_RenderPresent(renderer);
    }

    bool is_running() const { return running; }
    bool is_paused() const { return paused; }
    void set_paused(bool p) { paused = p; }

private:
    int width, height;
    bool running = true;
    bool paused = false;
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
        if (!s) return;
        SDL_Texture* t = SDL_CreateTextureFromSurface(renderer, s);
        SDL_Rect dst = {x, y, s->w, s->h};
        SDL_RenderCopy(renderer, t, NULL, &dst);
        SDL_FreeSurface(s);
        SDL_DestroyTexture(t);
    }

    void draw_ui(const GameState& state, int turn) {
        SDL_Rect ui_bg = {0, 0, width, 60};
        SDL_SetRenderDrawColor(renderer, 30, 30, 40, 200);
        SDL_RenderFillRect(renderer, &ui_bg);

        draw_text("GOLD: " + std::to_string((int)state.p_res.gold), 20, 20, {255, 215, 0, 255});
        draw_text("WOOD: " + std::to_string((int)state.p_res.wood), 150, 20, {150, 100, 50, 255});

        int p1_sols = 0, p2_sols = 0;
        for (const auto& row : state.board) {
            for (const auto& tile : row) {
                if (tile.owner == 1) p1_sols += tile.soldiers;
                if (tile.owner == 2) p2_sols += tile.p2_soldiers;
            }
        }
        draw_text("P1 SOL: " + std::to_string(p1_sols), 300, 20, {255, 50, 50, 255});
        draw_text("P2 SOL: " + std::to_string(p2_sols), 450, 20, {255, 120, 0, 255});
        draw_text("TURN: " + std::to_string(turn), width / 2 + 100, 20, {255, 255, 255, 255});

        std::string pause_text = paused ? "PAUSED (SPACE)" : "SPACE: Pause | ESC: Quit";
        draw_text(pause_text, width - 280, 20, {180, 180, 180, 255});
    }
};

class StandaloneAIGame {
public:
    StandaloneAIGame() : renderer(1280, 720), turn_manager() {
        state = BoardFactory::create_default(16, 16, 0, 0, 15, 15);
        
        std::string manager_path = "models/manager.pt";
        std::string soldier_path = "models/soldier.pt";
        
        ai.load(manager_path, soldier_path);
        
        if (!ai.is_loaded()) {
            std::cout << "WARNING: AI models not loaded. Game will use rule-based actions." << std::endl;
        } else {
            std::cout << "AI models loaded successfully!" << std::endl;
        }
    }

    void run() {
        std::cout << "Starting standalone AI game..." << std::endl;
        std::cout << "Controls:" << std::endl;
        std::cout << "  Arrows: Pan camera" << std::endl;
        std::cout << "  +/-: Zoom" << std::endl;
        std::cout << "  Space: Pause/Resume" << std::endl;
        std::cout << "  ESC: Quit" << std::endl;
        
        Uint32 last_step_time = SDL_GetTicks();
        const Uint32 step_interval = 500;

        while (renderer.is_running()) {
            renderer.handle_input();

            if (!renderer.is_paused()) {
                Uint32 current_time = SDL_GetTicks();
                if (current_time - last_step_time >= step_interval) {
                    step_game();
                    last_step_time = current_time;
                }
            }

            renderer.render_game_state(state, current_turn);
            SDL_Delay(16);
        }
        
        std::cout << "Game finished after " << current_turn << " turns." << std::endl;
    }

private:
    GameRenderer renderer;
    TurnManager turn_manager;
    InferenceEngine ai;
    GameState state;
    int current_turn = 0;

    void step_game() {
        auto obs = ObservationBuilder::build(state);
        
        TurnManager::ActionBundle p1_action;
        if (ai.is_loaded()) {
            CppActionBundle ai_action = ai.select_action(obs);
            p1_action.diplomacy = ai_action.diplomacy;
            p1_action.economy = ai_action.economy;
            p1_action.target_tile = ai_action.target_tile;
            p1_action.soldier_target = ai_action.soldier_target;
            p1_action.fortify_tile = ai_action.fortify_tile;
        } else {
            p1_action = get_rule_based_action();
        }

        TurnManager::ActionBundle p2_action = get_p2_action();

        turn_manager.step(state, p1_action, p2_action);
        
        if (state.target_tile.has_value()) {
            auto [tr, tc] = state.target_tile.value();
        }
        
        current_turn++;

        if (turn_manager.is_terminal(state)) {
            std::cout << "Game ended! Turn " << current_turn << std::endl;
            renderer.set_paused(true);
        }

        if (current_turn >= 1000) {
            std::cout << "Max turns reached!" << std::endl;
            renderer.set_paused(true);
        }
    }

    TurnManager::ActionBundle get_rule_based_action() {
        TurnManager::ActionBundle action;
        action.diplomacy = 1;
        action.economy = {0, 0, 0, 0, 0, 0};
        action.target_tile = 0;
        action.soldier_target = 0;
        action.fortify_tile = 0;
        
        if (state.p_res.gold >= 100 && state.p_res.wood >= 50) {
            action.economy[0] = 1;
        }
        
        return action;
    }

    TurnManager::ActionBundle get_p2_action() {
        TurnManager::ActionBundle action;
        action.diplomacy = 1;
        action.economy = {0, 0, 0, 0, 0, 0};
        
        if (current_turn % 10 == 0 && state.p_res.gold >= 50) {
            action.economy[0] = 1;
        }
        
        return action;
    }
};

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "  Strategy Game - Standalone AI Edition" << std::endl;
    std::cout << "========================================" << std::endl;
    
    StandaloneAIGame game;
    game.run();
    
    return 0;
}
