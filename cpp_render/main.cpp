#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <cmath>
#include <algorithm>
#include <SDL.h>
#include <SDL_ttf.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <nlohmann/json.hpp>

#pragma comment(lib, "ws2_32.lib")

using json = nlohmann::json;

// --- Math & Types ---
struct Point3D {
    float x, y, z;
};

struct Point2D {
    int x, y;
};

class IsoCamera {
public:
    int screenWidth, screenHeight;
    float zoom = 1.0f;
    int offsetX = 0, offsetY = 0;
    float tileW = 64.0f;
    float tileH = 32.0f;

    IsoCamera(int w, int h) : screenWidth(w), screenHeight(h) {
        offsetX = w / 2;
        offsetY = h / 4;
    }

    Point2D project(float x, float y, float z = 0.0f) const {
        // Isometric projection: 
        // screenX = (mapX - mapY) * (tileW / 2)
        // screenY = (mapX + mapY) * (tileH / 2) - mapZ
        int sx = static_cast<int>((x - y) * (tileW / 2.0f) * zoom) + offsetX;
        int sy = static_cast<int>((x + y) * (tileH / 2.0f) * zoom) - static_cast<int>(z * zoom) + offsetY;
        return {sx, sy};
    }
};

// --- Render Helpers ---
namespace Graphics {
    void draw_iso_diamond(SDL_Renderer* renderer, const IsoCamera& cam, float x, float y, float size, SDL_Color color) {
        Point2D p1 = cam.project(x, y);
        Point2D p2 = cam.project(x + size, y);
        Point2D p3 = cam.project(x + size, y + size);
        Point2D p4 = cam.project(x, y + size);

        SDL_Vertex vertices[4];
        for (int i = 0; i < 4; ++i) vertices[i].color = color;
        
        vertices[0].position = {(float)p1.x, (float)p1.y};
        vertices[1].position = {(float)p2.x, (float)p2.y};
        vertices[2].position = {(float)p3.x, (float)p3.y};
        vertices[3].position = {(float)p4.x, (float)p4.y};

        int indices[] = {0, 1, 2, 0, 2, 3};
        SDL_RenderGeometry(renderer, nullptr, vertices, 4, indices, 6);
        
        // Border
        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 30);
        SDL_RenderDrawLine(renderer, p1.x, p1.y, p2.x, p2.y);
        SDL_RenderDrawLine(renderer, p2.x, p2.y, p3.x, p3.y);
        SDL_RenderDrawLine(renderer, p3.x, p3.y, p4.x, p4.y);
        SDL_RenderDrawLine(renderer, p4.x, p4.y, p1.x, p1.y);
    }

    void draw_iso_box(SDL_Renderer* renderer, const IsoCamera& cam, float x, float y, float w, float d, float h, SDL_Color color, float z_start = 0.0f) {
        // Bottom Face
        Point2D b1 = cam.project(x, y, z_start);
        Point2D b2 = cam.project(x + w, y, z_start);
        Point2D b3 = cam.project(x + w, y + d, z_start);
        Point2D b4 = cam.project(x, y + d, z_start);

        // Top Face
        Point2D t1 = cam.project(x, y, z_start + h);
        Point2D t2 = cam.project(x + w, y, z_start + h);
        Point2D t3 = cam.project(x + w, y + d, z_start + h);
        Point2D t4 = cam.project(x, y + d, z_start + h);

        auto draw_quad = [&](Point2D p1, Point2D p2, Point2D p3, Point2D p4, SDL_Color col) {
            SDL_Vertex v[4];
            for (int i = 0; i < 4; ++i) v[i].color = col;
            v[0].position = {(float)p1.x, (float)p1.y};
            v[1].position = {(float)p2.x, (float)p2.y};
            v[2].position = {(float)p3.x, (float)p3.y};
            v[3].position = {(float)p4.x, (float)p4.y};
            int ind[] = {0, 1, 2, 0, 2, 3};
            SDL_RenderGeometry(renderer, nullptr, v, 4, ind, 6);
        };

        SDL_Color side1 = { (Uint8)(color.r * 0.8), (Uint8)(color.g * 0.8), (Uint8)(color.b * 0.8), color.a };
        SDL_Color side2 = { (Uint8)(color.r * 0.6), (Uint8)(color.g * 0.6), (Uint8)(color.b * 0.6), color.a };

        draw_quad(b1, b2, t2, t1, side1); // Side 1
        draw_quad(b2, b3, t3, t2, side2); // Side 2
        draw_quad(t1, t2, t3, t4, color); // Top

        // Outlines
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 50);
        SDL_RenderDrawLine(renderer, t1.x, t1.y, t2.x, t2.y);
        SDL_RenderDrawLine(renderer, t2.x, t2.y, t3.x, t3.y);
        SDL_RenderDrawLine(renderer, t3.x, t3.y, t4.x, t4.y);
        SDL_RenderDrawLine(renderer, t4.x, t4.y, t1.x, t1.y);
        SDL_RenderDrawLine(renderer, b1.x, b1.y, t1.x, t1.y);
        SDL_RenderDrawLine(renderer, b2.x, b2.y, t2.x, t2.y);
        SDL_RenderDrawLine(renderer, b3.x, b3.y, t3.x, t3.y);
    }

    static void fill_circle(SDL_Renderer* renderer, int x, int y, int radius) {
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

}

// --- Entity Classes ---
class BaseRenderer {
public:
    virtual void render(SDL_Renderer* renderer, const IsoCamera& cam, const json& tile, int r, int c) = 0;
};

class TileRenderer : public BaseRenderer {
public:
    void render(SDL_Renderer* renderer, const IsoCamera& cam, const json& tile, int r, int c) override {
        int owner = tile["owner"];
        SDL_Color col = {40, 40, 45, 255};
        if (owner == 1) col = {30, 60, 120, 255};
        else if (owner == 2) col = {100, 30, 30, 255};
        
        Graphics::draw_iso_diamond(renderer, cam, (float)c, (float)r, 1.0f, col);
    }
};

class BuildingRenderer : public BaseRenderer {
public:
    void render(SDL_Renderer* renderer, const IsoCamera& cam, const json& tile, int r, int c) override {
        int status = tile["status"];
        // Status 1, 4 (Sawmill) and 5, 7 (Crops) are handled by dedicated renderers
        if (status == 0 || status == 1 || status == 4 || status == 5 || status == 7) return; 

        SDL_Color col = {200, 200, 200, 255};
        float height = 0.5f;

        if (status == 2 || status == 6) { // Warehouses
            col = {210, 180, 140, 255};
            height = 0.6f;
        } else if (status == 3) { // Base
            col = {255, 215, 0, 255};
            height = 0.8f;
        }

        Graphics::draw_iso_box(renderer, cam, (float)c + 0.2f, (float)r + 0.2f, 0.6f, 0.6f, height * cam.tileW, col);
    }
};

class CropRenderer : public BaseRenderer {
public:
    void render(SDL_Renderer* renderer, const IsoCamera& cam, const json& tile, int r, int c) override {
        int status = tile["status"];
        if (status != 5 && status != 7) return; // Only render for crop statuses

        float z = cam.zoom;
        SDL_Color soilCol = {101, 67, 33, 255}; // Dark Earth
        SDL_Color plantCol = {100, 200, 50, 255}; // Bright Crop Green

        // 1. Draw the Tilled Soil Base
        Graphics::draw_iso_box(renderer, cam, (float)c + 0.1f, (float)r + 0.1f, 0.8f, 0.8f, 0.05f * cam.tileW, soilCol);

        // 2. Draw the Furrows (Rows) and Tufts
        for (float row_off = 0.25f; row_off <= 0.75f; row_off += 0.25f) {
            for (float col_off = 0.2f; col_off <= 0.8f; col_off += 0.2f) {
                Graphics::draw_iso_box(renderer, cam, 
                    (float)c + col_off, 
                    (float)r + row_off, 
                    0.1f, 0.1f, 
                    0.15f * cam.tileW, 
                    plantCol, 
                    0.05f * cam.tileW); 
            }
        }
    }
};

class SawmillRenderer : public BaseRenderer {
public:
    void render(SDL_Renderer* renderer, const IsoCamera& cam, const json& tile, int r, int c) override {
        int status = tile["status"];
        if (status != 1 && status != 4) return; 

        Point2D p = cam.project(c + 0.5f, r + 0.5f);
        float z = cam.zoom;

        // --- 2. THE MAIN CABIN (Brown Cube) ---
        SDL_Color buildingCol = {139, 69, 19, 255}; // Saddle Brown
        float cabinHeight = 0.6f * cam.tileW;
        Graphics::draw_iso_box(renderer, cam, (float)c + 0.2f, (float)r + 0.2f, 0.3f, 0.3f, cabinHeight, buildingCol);

        // --- 3. THE SLANTED ROOF (Slate Grey) ---
        SDL_Color roofCol = {80, 80, 90, 255};
        Graphics::draw_iso_box(renderer, cam, (float)c + 0.15f, (float)r + 0.15f, 0.7f, 0.7f, 0.1f * cam.tileW, roofCol, cabinHeight); 

        // --- 4. THE LOG PILE ---
        SDL_Color logCol = {101, 67, 33, 255};
        for(int i = 0; i < 3; ++i) {
            SDL_Rect log = { p.x + (int)(10*z), p.y - (int)((5 + i*4)*z), (int)(15*z), (int)(3*z) };
            SDL_SetRenderDrawColor(renderer, logCol.r, logCol.g, logCol.b, 255);
            SDL_RenderFillRect(renderer, &log);
        }
    }
};

class TreeRenderer : public BaseRenderer {
public:
    void render(SDL_Renderer* renderer, const IsoCamera& cam, const json& tile, int r, int c) override {
        if (!tile.contains("wood") || tile["wood"].get<int>() <= 0) return;

        // Don't draw a tree if there's a sawmill (status 1 or 4) on this tile
        int status = tile["status"];
        if (status == 1 || status == 4) return;

        // Use the camera projection to stay locked to the grid
        Point2D p = cam.project(c + 0.5f, r + 0.5f);
        float z = cam.zoom;

        // 1. Drop Shadow (Grounded)
        SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 60); 
        Graphics::fill_circle(renderer, p.x, p.y, (int)(10 * z));

        // 2. Trunk (Brown Log) - Scaled by Zoom
        SDL_SetRenderDrawColor(renderer, 101, 67, 33, 255);
        int trunkW = (int)(6 * z);
        int trunkH = (int)(25 * z);
        SDL_Rect trunk = { p.x - trunkW / 2, p.y - trunkH, trunkW, trunkH };
        SDL_RenderFillRect(renderer, &trunk);

        // 3. Rounded Crown (3 Layers for "3D" depth) - Scaled by Zoom
        int crownY = p.y - (int)(35 * z);
        
        // Bottom layer
        SDL_SetRenderDrawColor(renderer, 20, 100, 20, 255);
        Graphics::fill_circle(renderer, p.x, crownY, (int)(18 * z));
        
        // Middle layer
        SDL_SetRenderDrawColor(renderer, 34, 139, 34, 255);
        Graphics::fill_circle(renderer, p.x, crownY - (int)(4 * z), (int)(14 * z));
        
        // Top highlight
        SDL_SetRenderDrawColor(renderer, 60, 200, 60, 255);
        Graphics::fill_circle(renderer, p.x - (int)(3 * z), crownY - (int)(7 * z), (int)(7 * z));
    }
};

class WallRenderer : public BaseRenderer {
public:
    void render(SDL_Renderer* renderer, const IsoCamera& cam, const json& tile, int r, int c) override {
        if (!tile.contains("fortified")) return;
        int f_lvl = tile["fortified"];
        if (f_lvl <= 0) return;

        SDL_Color wallCol = (f_lvl == 1) ? SDL_Color{160, 120, 60, 255} : SDL_Color{140, 140, 150, 255};
        float wallH = (f_lvl == 1) ? 0.2f : 0.4f;
        float wallT = (f_lvl == 1) ? 0.1f : 0.15f; // Thickness

        // North Wall
        Graphics::draw_iso_box(renderer, cam, (float)c, (float)r, 1.0f, wallT, wallH * cam.tileW, wallCol);
        // West Wall
        Graphics::draw_iso_box(renderer, cam, (float)c, (float)r, wallT, 1.0f, wallH * cam.tileW, wallCol);
        // South Wall
        Graphics::draw_iso_box(renderer, cam, (float)c, (float)r + 1.0f - wallT, 1.0f, wallT, wallH * cam.tileW, wallCol);
        // East Wall
        Graphics::draw_iso_box(renderer, cam, (float)c + 1.0f - wallT, (float)r, wallT, 1.0f, wallH * cam.tileW, wallCol);
    }
};

class UnitRenderer : public BaseRenderer {
public:
    void render(SDL_Renderer* renderer, const IsoCamera& cam, const json& tile, int r, int c) override {
        // Player 1 Soldiers (Red) - Positioned near the "back" corner of the tile
        if (tile.contains("soldiers") && tile["soldiers"].get<int>() > 0) {
            draw_unit(renderer, cam, r, c, tile["soldiers"].get<int>(), {255, 50, 50, 255}, -0.35f);
        }
        // Player 2 Soldiers (Orange) - Positioned near the "front" corner of the tile
        if (tile.contains("p2_soldiers") && tile["p2_soldiers"].get<int>() > 0) {
            draw_unit(renderer, cam, r, c, tile["p2_soldiers"].get<int>(), {255, 120, 0, 255}, 0.35f);
        }
    }

private:
    void draw_unit(SDL_Renderer* renderer, const IsoCamera& cam, int r, int c, int count, SDL_Color col, float offset) {
        float z_factor = cam.zoom;
        // Base of the unit on the ground (z=0)
        // Using the offset for both x and y to push them to opposite corners
        Point2D p = cam.project(c + 0.5f + offset, r + 0.5f + offset, 0.0f);
        
        int unit_radius = (int)(7 * z_factor);
        int unit_height = (int)(25 * z_factor); // Taller soldiers

        SDL_SetRenderDrawColor(renderer, col.r, col.g, col.b, col.a);
        
        // Draw 3D cylinder
        for (int i = 0; i < unit_height; ++i) {
            Graphics::fill_circle(renderer, p.x, p.y - i, unit_radius);
        }
        
        // Head highlight (White)
        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
        Graphics::fill_circle(renderer, p.x, p.y - unit_height, unit_radius);
    }
};

// --- Main Application ---
class GameStateManager {
public:
    json data;
    std::mutex mtx;
    bool updated = false;

    void update(const json& j) {
        std::lock_guard<std::mutex> lock(mtx);
        data = j;
        updated = true;
    }

    json get_data() {
        std::lock_guard<std::mutex> lock(mtx);
        return data;
    }
};

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
        json state = global_state.get_data();
        if (state.is_null()) return;

        SDL_SetRenderDrawColor(renderer, 15, 15, 20, 255);
        SDL_RenderClear(renderer);

        if (state.contains("board")) {
            const auto& board = state["board"];
            int rows = (int)board.size();
            int cols = (int)board[0].size();

            // Depth Sorting: Render from back (0,0) to front (max, max)
            // Nested r, c loops provide the correct isometric Back-to-Front order.
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < cols; ++c) {
                    tile_renderer.render(renderer, cam, board[r][c], r, c);
                    wall_renderer.render(renderer, cam, board[r][c], r, c);
                    tree_renderer.render(renderer, cam, board[r][c], r, c);
                    crop_renderer.render(renderer, cam, board[r][c], r, c);
                    sawmill_renderer.render(renderer, cam, board[r][c], r, c);
                    building_renderer.render(renderer, cam, board[r][c], r, c);
                    unit_renderer.render(renderer, cam, board[r][c], r, c);
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
    BuildingRenderer building_renderer;
    UnitRenderer unit_renderer;

    void draw_ui(const json& state) {
        auto draw_text = [&](const std::string& text, int x, int y, SDL_Color col) {
            if (!font) return;
            SDL_Surface* s = TTF_RenderText_Blended(font, text.c_str(), col);
            SDL_Texture* t = SDL_CreateTextureFromSurface(renderer, s);
            SDL_Rect dst = {x, y, s->w, s->h};
            SDL_RenderCopy(renderer, t, NULL, &dst);
            SDL_FreeSurface(s);
            SDL_DestroyTexture(t);
        };

        SDL_Rect ui_bg = {0, 0, width, 60};
        SDL_SetRenderDrawColor(renderer, 30, 30, 40, 200);
        SDL_RenderFillRect(renderer, &ui_bg);

        if (state.contains("p_res")) {
            const auto& res = state["p_res"];
            draw_text("GOLD: " + std::to_string((int)res[0].get<double>()), 20, 20, {255, 215, 0, 255});
            draw_text("WOOD: " + std::to_string((int)res[1].get<double>()), 150, 20, {150, 100, 50, 255});
        }
        if (state.contains("turn")) {
            draw_text("TURN: " + std::to_string(state["turn"].get<int>()), width / 2 - 40, 20, {255, 255, 255, 255});
        }
        
        draw_text("Arrows: Pan | +/-: Zoom", width - 220, 20, {180, 180, 180, 255});
    }
};

void server_thread() {
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);
    SOCKET ListenSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    sockaddr_in service;
    service.sin_family = AF_INET;
    service.sin_addr.s_addr = inet_addr("127.0.0.1");
    service.sin_port = htons(8080);
    bind(ListenSocket, (SOCKADDR*)&service, sizeof(service));
    listen(ListenSocket, SOMAXCONN);
    
    while (true) {
        SOCKET AcceptSocket = accept(ListenSocket, NULL, NULL);
        if (AcceptSocket == INVALID_SOCKET) break;
        std::string buffer;
        char chunk[4096];
        while (true) {
            int bytes = recv(AcceptSocket, chunk, sizeof(chunk), 0);
            if (bytes <= 0) break;
            buffer.append(chunk, bytes);
            size_t pos;
            while ((pos = buffer.find('\n')) != std::string::npos) {
                try {
                    global_state.update(json::parse(buffer.substr(0, pos)));
                } catch (...) {}
                buffer.erase(0, pos + 1);
            }
        }
        closesocket(AcceptSocket);
    }
    closesocket(ListenSocket);
    WSACleanup();
}

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
