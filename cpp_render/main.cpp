#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <cmath>
#include <SDL.h>
#include <SDL_ttf.h>
#include <winsock2.h>
#include <ws2tcpip.h>

#include <nlohmann/json.hpp>

#pragma comment(lib, "ws2_32.lib")

using json = nlohmann::json;

struct GameState {
    json data;
    std::mutex mtx;
    bool updated = false;
};

GameState global_state;

class Renderer {
public:
    Renderer(int width, int height) : width(width), height(height) {
        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            std::cerr << "SDL could not initialize! SDL_Error: " << SDL_GetError() << std::endl;
        }
        if (TTF_Init() == -1) {
            std::cerr << "SDL_ttf could not initialize! TTF_Error: " << TTF_GetError() << std::endl;
        }
        window = SDL_CreateWindow("Strategy Game - C++", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, width, height, SDL_WINDOW_SHOWN);
        if (!window) {
            std::cerr << "Window could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        }
        renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
        if (!renderer) {
            std::cerr << "Renderer could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        }
        
        const char* font_paths[] = {
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "arial.ttf"
        };
        for (const char* path : font_paths) {
            font = TTF_OpenFont(path, 18);
            if (font) break;
        }
        
        if (!font) {
            std::cerr << "Failed to load font! TTF_Error: " << TTF_GetError() << std::endl;
        }

        small_font = TTF_OpenFont("C:/Windows/Fonts/arial.ttf", 14);
        if (!small_font) small_font = font;
    }

    ~Renderer() {
        if (font && font != small_font) TTF_CloseFont(font);
        if (small_font) TTF_CloseFont(small_font);
        if (renderer) SDL_DestroyRenderer(renderer);
        if (window) SDL_DestroyWindow(window);
        TTF_Quit();
        SDL_Quit();
    }
    
    void render_frame(const json& state) {
        if (state.is_null()) return;

        SDL_SetRenderDrawColor(renderer, 20, 20, 25, 255);
        SDL_RenderClear(renderer);

        draw_ui(state);
        
        if (state.contains("board")) {
            const auto& board = state["board"];
            int rows = (int)board.size();
            int cols = (int)board[0].size();
            int cell_size = 40;
            int start_x = (width - (cols * cell_size)) / 2;
            int start_y = 200;

            draw_board_backgrounds(state, start_x, start_y, rows, cols, cell_size);
            draw_routes(state, start_x, start_y, cell_size);
            draw_board_content(state, start_x, start_y, rows, cols, cell_size);
            draw_shots(state, start_x, start_y, cell_size);
        }

        SDL_RenderPresent(renderer);
    }

private:
    int width, height;
    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;
    TTF_Font* font = nullptr;
    TTF_Font* small_font = nullptr;

    void draw_board_backgrounds(const json& state, int start_x, int start_y, int rows, int cols, int cell_size) {
        const auto& board = state["board"];
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                SDL_Rect rect = {start_x + c * cell_size, start_y + r * cell_size, cell_size, cell_size};
                int owner = board[r][c]["owner"];
                
                if (owner == 1) SDL_SetRenderDrawColor(renderer, 30, 60, 120, 255);
                else if (owner == 2) SDL_SetRenderDrawColor(renderer, 100, 30, 30, 255);
                else SDL_SetRenderDrawColor(renderer, 40, 40, 45, 255);
                
                SDL_RenderFillRect(renderer, &rect);
            }
        }
    }

    void draw_routes(const json& state, int start_x, int start_y, int cell_size) {
        if (state.contains("p1_routes") && state.contains("p1_base")) {
            auto base = state["p1_base"];
            int bx = start_x + base[1].get<int>() * cell_size + cell_size / 2;
            int by = start_y + base[0].get<int>() * cell_size + cell_size / 2;
            SDL_SetRenderDrawColor(renderer, 0, 255, 255, 255); 
            for (auto& route : state["p1_routes"]) {
                int mx = start_x + route[1].get<int>() * cell_size + cell_size / 2;
                int my = start_y + route[0].get<int>() * cell_size + cell_size / 2;
                SDL_RenderDrawLine(renderer, bx, by, mx, my);
                SDL_RenderDrawLine(renderer, bx+1, by, mx+1, my);
                SDL_RenderDrawLine(renderer, bx, by+1, mx, my+1);
                draw_circle(mx, my, 4);
            }
        }
    }

    void draw_board_content(const json& state, int start_x, int start_y, int rows, int cols, int cell_size) {
        const auto& board = state["board"];
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                SDL_Rect rect = {start_x + c * cell_size, start_y + r * cell_size, cell_size, cell_size};
                const auto& tile = board[r][c];

                // --- Fortification Highlight ---
                if (tile.contains("fortified")) {
                    int f_lvl = tile["fortified"].get<int>();
                    if (f_lvl == 1) {
                        SDL_SetRenderDrawColor(renderer, 0, 100, 255, 255); // Blue
                        for(int i=0; i<3; ++i) {
                            SDL_Rect border = {rect.x+i, rect.y+i, rect.w-2*i, rect.h-2*i};
                            SDL_RenderDrawRect(renderer, &border);
                        }
                    } else if (f_lvl == 2) {
                        SDL_SetRenderDrawColor(renderer, 255, 0, 255, 255); // Purple
                        for(int i=0; i<4; ++i) {
                            SDL_Rect border = {rect.x+i, rect.y+i, rect.w-2*i, rect.h-2*i};
                            SDL_RenderDrawRect(renderer, &border);
                        }
                    }
                }

                SDL_SetRenderDrawColor(renderer, 60, 60, 70, 255);
                SDL_RenderDrawRect(renderer, &rect);

                if (tile.contains("wood") && tile["wood"].get<int>() > 0) {
                    draw_wood_icon(rect.x + 5, rect.y + 5);
                }

                int status = tile["status"];
                if (status == 1) draw_mine_icon(rect.x + cell_size / 2, rect.y + cell_size / 2, 1);
                else if (status == 2) draw_warehouse_icon(rect.x + cell_size / 2, rect.y + cell_size / 2, 1);
                else if (status == 3) draw_base_icon(rect.x + cell_size / 2, rect.y + cell_size / 2);
                else if (status == 4) draw_mine_icon(rect.x + cell_size / 2, rect.y + cell_size / 2, 2);
                else if (status == 5) draw_crop_field_icon(rect.x + cell_size / 2, rect.y + cell_size / 2, 1);
                else if (status == 6) draw_warehouse_icon(rect.x + cell_size / 2, rect.y + cell_size / 2, 2);
                else if (status == 7) draw_crop_field_icon(rect.x + cell_size / 2, rect.y + cell_size / 2, 2);

                if (tile.contains("workers") && tile["workers"].get<double>() > 0) {
                     draw_text(std::to_string((int)tile["workers"].get<double>()), rect.x + 15, rect.y + 22, {255, 255, 255, 255}, true);
                }

                if (tile.contains("soldiers") && tile["soldiers"].get<int>() > 0) {
                    draw_soldier_icon(rect.x + cell_size / 2, rect.y + cell_size / 2, tile["soldiers"].get<int>(), {255, 50, 50, 255});
                }
                
                if (tile.contains("p2_soldiers") && tile["p2_soldiers"].get<int>() > 0) {
                    draw_soldier_icon(rect.x + cell_size / 2, rect.y + cell_size / 2, tile["p2_soldiers"].get<int>(), {255, 120, 0, 255});
                }
            }
        }
    }

    void draw_ui(const json& state) {
        SDL_Rect p_box = {20, 20, 250, 150};
        SDL_SetRenderDrawColor(renderer, 35, 35, 45, 255);
        SDL_RenderFillRect(renderer, &p_box);
        SDL_SetRenderDrawColor(renderer, 60, 60, 80, 255);
        SDL_RenderDrawRect(renderer, &p_box);
        draw_text("PLAYER", p_box.x + 15, p_box.y + 10, {100, 200, 255, 255});
        
        if(state.contains("p_res") && state.contains("p_gen")) {
            const auto& res = state["p_res"];
            const auto& gen = state["p_gen"];
            draw_text("Gold: " + std::to_string((int)res[0].get<double>()) + " (+" + std::to_string((int)gen[0].get<double>()) + ")", p_box.x + 15, p_box.y + 40, {255, 215, 0, 255});
            draw_text("Wood: " + std::to_string((int)res[1].get<double>()) + " (+" + std::to_string((int)gen[1].get<double>()) + ")", p_box.x + 15, p_box.y + 65, {139, 69, 19, 255});
            draw_text("Workers: " + std::to_string((int)res[2].get<double>()), p_box.x + 15, p_box.y + 90, {200, 200, 200, 255});
            if (res.size() > 4 && gen.size() > 3)
                draw_text("Food: " + std::to_string((int)res[4].get<double>()) + " (+" + std::to_string((int)gen[3].get<double>()) + ")", p_box.x + 15, p_box.y + 115, {150, 255, 100, 255});
        }

        SDL_Rect o_box = {width - 270, 20, 250, 150};
        SDL_SetRenderDrawColor(renderer, 45, 30, 30, 255);
        SDL_RenderFillRect(renderer, &o_box);
        SDL_SetRenderDrawColor(renderer, 100, 50, 50, 255);
        SDL_RenderDrawRect(renderer, &o_box);
        draw_text("OPPONENT", o_box.x + 15, o_box.y + 10, {255, 80, 80, 255});

        if(state.contains("o_res") && state.contains("o_gen")) {
            const auto& res = state["o_res"];
            const auto& gen = state["o_gen"];
            draw_text("Gold: " + std::to_string((int)res[0].get<double>()) + " (+" + std::to_string((int)gen[0].get<double>()) + ")", o_box.x + 15, o_box.y + 40, {255, 215, 0, 255});
            draw_text("Wood: " + std::to_string((int)res[1].get<double>()) + " (+" + std::to_string((int)gen[1].get<double>()) + ")", o_box.x + 15, o_box.y + 65, {139, 69, 19, 255});
            draw_text("Workers: " + std::to_string((int)res[2].get<double>()), o_box.x + 15, o_box.y + 90, {200, 200, 200, 255});
        }

        if (state.contains("turn")) {
            draw_text("Turn: " + std::to_string(state["turn"].get<int>()), width / 2 - 40, 20, {255, 255, 255, 255});
        }
    }

    void draw_shots(const json& state, int start_x, int start_y, int cell_size) {
        if (state.contains("shot_events")) {
            SDL_SetRenderDrawColor(renderer, 255, 255, 0, 255); // Yellow
            for (auto& event : state["shot_events"]) {
                auto from = event["from"];
                auto to = event["to"];
                int fx = start_x + from[1].get<int>() * cell_size + cell_size / 2;
                int fy = start_y + from[0].get<int>() * cell_size + cell_size / 2;
                int tx = start_x + to[1].get<int>() * cell_size + cell_size / 2;
                int ty = start_y + to[0].get<int>() * cell_size + cell_size / 2;
                
                SDL_RenderDrawLine(renderer, fx, fy, tx, ty);
                draw_circle(tx, ty, 4);
                fill_circle(tx, ty, 2);
            }
        }
    }
    
    void draw_mine_icon(int x, int y, int level) {
        if (level == 1) SDL_SetRenderDrawColor(renderer, 150, 80, 255, 255);
        else SDL_SetRenderDrawColor(renderer, 0, 255, 100, 255); 
        SDL_Rect r = {x - 6, y - 6, 12, 12};
        SDL_RenderFillRect(renderer, &r);
        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
        SDL_RenderDrawRect(renderer, &r);
    }

    void draw_warehouse_icon(int x, int y, int level) {
        if (level == 1) SDL_SetRenderDrawColor(renderer, 210, 180, 140, 255);
        else SDL_SetRenderDrawColor(renderer, 150, 150, 160, 255); 
        SDL_Rect body = {x - 8, y - 2, 16, 10};
        SDL_RenderFillRect(renderer, &body);
        if (level == 1) SDL_SetRenderDrawColor(renderer, 100, 50, 30, 255);
        else SDL_SetRenderDrawColor(renderer, 80, 40, 30, 255);
        SDL_Rect roof = {x - 10, y - 6, 20, 4};
        SDL_RenderFillRect(renderer, &roof);
    }

    void draw_base_icon(int x, int y) {
        SDL_SetRenderDrawColor(renderer, 200, 200, 200, 255);
        SDL_Rect body = {x - 10, y - 5, 20, 15};
        SDL_RenderFillRect(renderer, &body);
        SDL_SetRenderDrawColor(renderer, 255, 255, 0, 255);
        SDL_Rect flag = {x + 2, y - 12, 6, 4};
        SDL_RenderFillRect(renderer, &flag);
    }

    void draw_crop_field_icon(int x, int y, int level) {
        if (level == 1) SDL_SetRenderDrawColor(renderer, 100, 180, 50, 255);
        else SDL_SetRenderDrawColor(renderer, 50, 200, 50, 255); 
        for(int i=-6; i<=6; i+=4) {
            SDL_Rect stalk = {x + i, y - 6, 2, 12};
            SDL_RenderFillRect(renderer, &stalk);
        }
    }

    void draw_wood_icon(int x, int y) {
        SDL_SetRenderDrawColor(renderer, 139, 69, 19, 255);
        SDL_Rect log = {x, y, 12, 6};
        SDL_RenderFillRect(renderer, &log);
    }

    void draw_soldier_icon(int x, int y, int count, SDL_Color color) {
        SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, color.a);
        fill_circle(x, y, 12);
        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
        draw_circle(x, y, 12);
        if (count > 1) {
            draw_text(std::to_string(count), x - 4, y - 6, {255, 255, 255, 255}, true);
        }
    }

    void fill_circle(int x, int y, int radius) {
        for (int w = 0; w < radius * 2; w++) {
            for (int h = 0; h < radius * 2; h++) {
                int dx = radius - w;
                int dy = radius - h;
                if ((dx*dx + dy*dy) <= (radius * radius)) {
                    SDL_RenderDrawPoint(renderer, x + dx, y + dy);
                }
            }
        }
    }

    void draw_circle(int x, int y, int radius) {
        const int32_t diameter = (radius * 2);
        int32_t cx = (radius - 1);
        int32_t cy = 0;
        int32_t tx = 1;
        int32_t ty = 1;
        int32_t error = (tx - diameter);

        while (cx >= cy) {
            SDL_RenderDrawPoint(renderer, x + cx, y - cy);
            SDL_RenderDrawPoint(renderer, x + cx, y + cy);
            SDL_RenderDrawPoint(renderer, x - cx, y - cy);
            SDL_RenderDrawPoint(renderer, x - cx, y + cy);
            SDL_RenderDrawPoint(renderer, x + cy, y - cx);
            SDL_RenderDrawPoint(renderer, x + cy, y + cx);
            SDL_RenderDrawPoint(renderer, x - cy, y - cx);
            SDL_RenderDrawPoint(renderer, x - cy, y + cx);

            if (error <= 0) {
                cy++;
                error += ty;
                ty += 2;
            }
            if (error > 0) {
                cx--;
                tx += 2;
                error += (tx - diameter);
            }
        }
    }
    
    void draw_text(const std::string& text, int x, int y, SDL_Color color, bool use_small_font = false) {
        TTF_Font* f = use_small_font ? small_font : font;
        if (!f) return;
        SDL_Surface* surface = TTF_RenderText_Blended(f, text.c_str(), color);
        if (!surface) return;
        SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);
        SDL_Rect dstRect = {x, y, surface->w, surface->h};
        SDL_RenderCopy(renderer, texture, NULL, &dstRect);
        SDL_FreeSurface(surface);
        SDL_DestroyTexture(texture);
    }
};

void server_thread() {
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) return;

    SOCKET ListenSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    sockaddr_in service;
    service.sin_family = AF_INET;
    service.sin_addr.s_addr = inet_addr("127.0.0.1");
    service.sin_port = htons(8080);
    
    if (bind(ListenSocket, (SOCKADDR*)&service, sizeof(service)) == SOCKET_ERROR) {
        closesocket(ListenSocket);
        WSACleanup();
        return;
    }
    listen(ListenSocket, SOMAXCONN);
    
    while (true) {
        SOCKET AcceptSocket = accept(ListenSocket, NULL, NULL);
        if (AcceptSocket == INVALID_SOCKET) break;

        std::string accumulated_data;
        char buffer[4096];
        while (true) {
            int bytesReceived = recv(AcceptSocket, buffer, sizeof(buffer), 0);
            if (bytesReceived <= 0) break;
            accumulated_data.append(buffer, bytesReceived);
            size_t pos;
            while ((pos = accumulated_data.find('\n')) != std::string::npos) {
                std::string line = accumulated_data.substr(0, pos);
                accumulated_data.erase(0, pos + 1);
                if (line.empty()) continue;
                try {
                    json j = json::parse(line);
                    std::lock_guard<std::mutex> lock(global_state.mtx);
                    global_state.data = j;
                    global_state.updated = true;
                } catch (...) {}
            }
        }
        closesocket(AcceptSocket);
    }
    closesocket(ListenSocket);
    WSACleanup();
}

int main(int argc, char* argv[]) {
    Renderer r(800, 900);
    std::thread s(server_thread);
    s.detach();
    
    bool running = true;
    while(running) {
        SDL_Event e;
        while(SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) running = false;
        }

        {
            std::lock_guard<std::mutex> lock(global_state.mtx);
            r.render_frame(global_state.data);
            global_state.updated = false;
        }

        SDL_Delay(16); 
    }
    return 0;
}
