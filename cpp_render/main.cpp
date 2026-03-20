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

    void draw_iso_box_stacked(SDL_Renderer* renderer, const IsoCamera& cam, float x, float y, float w, float d, float h, SDL_Color color, float z_start) {
            // We manually offset the projection by z_start since we can't change the original function
            // This is a "wrapper" that simulates the z-axis offset
            draw_iso_box(renderer, cam, x, y, w, d, h, color, z_start); 
        }

    void draw_complete_building(SDL_Renderer* renderer, const IsoCamera& cam, float x, float y, float w, float d, float wallH, float roofH, SDL_Color wallCol, SDL_Color roofCol) {
        // Draw the four walls using your existing box logic (if it exists) 
        // or a simple local implementation to ensure it works.
        
        // 1. Walls
        draw_iso_box(renderer, cam, x, y, w, d, wallH, wallCol, 0.0f);

        // 2. The Roof (Inclined/Pitched effect)
        // We draw two slabs that meet in the middle
        float midW = w / 2.0f;
        
        // Left pitch
        draw_iso_box(renderer, cam, x - 0.05f, y - 0.05f, midW + 0.05f, d + 0.1f, roofH, roofCol, wallH);
        // Right pitch
        draw_iso_box(renderer, cam, x + midW, y - 0.05f, midW + 0.05f, d + 0.1f, roofH, roofCol, wallH);
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
    virtual void render(SDL_Renderer* renderer, const IsoCamera& cam, const json& board, int r, int c) = 0;
};

class TileRenderer : public BaseRenderer {
public:
    void render(SDL_Renderer* renderer, const IsoCamera& cam, const json& board, int r, int c) override {
        const auto& tile = board[r][c];
        int owner = tile["owner"];
        SDL_Color col = {40, 40, 45, 255};
        if (owner == 1) col = {30, 60, 120, 255};
        else if (owner == 2) col = {100, 30, 30, 255};
        
        Graphics::draw_iso_diamond(renderer, cam, (float)c, (float)r, 1.0f, col);
    }
};

class BuildingRenderer : public BaseRenderer {
public:
    void render(SDL_Renderer* renderer, const IsoCamera& cam, const json& board, int r, int c) override {
        const auto& tile = board[r][c];
        int status = tile["status"];

        if (status == 0 || status == 1 || status == 4 || status == 5 ||
            status == 7 || status == 2 || status == 6 || status == 3) return;

        SDL_Color col = {200, 200, 200, 255};
        float height = 0.5f;


        Graphics::draw_iso_box(renderer, cam, (float)c + 0.2f, (float)r + 0.2f, 0.6f, 0.6f, height * cam.tileW, col);
    }
};

class TowerRenderer : public BaseRenderer {
public:
    void render(SDL_Renderer* renderer, const IsoCamera& cam, const json& board, int r, int c) override {
        const auto& tile = board[r][c];
        int status = tile["status"];
        if (status != 3) return; // Only render on "Base" tiles (status==3)

        draw_tower(renderer, cam, (float)c, (float)r);
    }

private:
    void draw_tower(SDL_Renderer* renderer, const IsoCamera& cam, float cx, float cy) {
        float z = cam.zoom;

        // ---- Color Palette (stone-like) ----
        SDL_Color stone_light = {190, 185, 175, 255};  // top face
        SDL_Color stone_mid   = {155, 150, 140, 255};  // front face (auto-darkened by draw_iso_box)
        SDL_Color stone_dark  = {110, 105,  95, 255};  // dark side
        SDL_Color merlon_col  = {170, 165, 155, 255};  // battlements
        SDL_Color door_col    = { 45,  35,  25, 255};  // dark doorway
        SDL_Color shadow_col  = { 80,  75,  68, 255};  // buttress shadow

        float tW = cam.tileW;

        // =============================
        // 1. MAIN TOWER BODY
        // =============================
        float tower_x  = cx + 0.15f;
        float tower_y  = cy + 0.15f;
        float tower_w  = 0.70f;
        float tower_d  = 0.70f;
        float tower_h  = 1.8f * tW;   // tall tower

        Graphics::draw_iso_box(renderer, cam,
            tower_x, tower_y, tower_w, tower_d, tower_h, stone_mid, 0.0f);

        // =============================
        // 2. BUTTRESSES (side supports)
        // Two on the front-left face, two on front-right
        // =============================
        float butt_w = 0.10f, butt_d = 0.10f, butt_h = tower_h * 0.55f;

        // Left face buttresses
        Graphics::draw_iso_box(renderer, cam,
            tower_x - butt_w, tower_y + 0.15f,
            butt_w, butt_d, butt_h, shadow_col, 0.0f);
        Graphics::draw_iso_box(renderer, cam,
            tower_x - butt_w, tower_y + 0.45f,
            butt_w, butt_d, butt_h, shadow_col, 0.0f);

        // Right face buttresses
        Graphics::draw_iso_box(renderer, cam,
            tower_x + tower_w, tower_y + 0.15f,
            butt_w, butt_d, butt_h, shadow_col, 0.0f);
        Graphics::draw_iso_box(renderer, cam,
            tower_x + tower_w, tower_y + 0.45f,
            butt_w, butt_d, butt_h, shadow_col, 0.0f);

        // =============================
        // 3. DOOR RECESS
        // A dark thin box slightly inset on the front face
        // =============================
        float door_w = 0.14f, door_d = 0.04f, door_h = 0.22f * tW;
        Graphics::draw_iso_box(renderer, cam,
            tower_x + (tower_w - door_w) * 0.5f,
            tower_y - door_d,           // slightly in front
            door_w, door_d, door_h,
            door_col, 0.0f);

        // Door frame (slightly wider, lighter)
        SDL_Color frame_col = {140, 130, 115, 255};
        Graphics::draw_iso_box(renderer, cam,
            tower_x + (tower_w - door_w) * 0.5f - 0.02f,
            tower_y - door_d * 0.5f,
            door_w + 0.04f, door_d * 0.5f, door_h + 0.03f * tW,
            frame_col, 0.0f);

        // =============================
        // 4. PARAPET / WALL WALK
        // Slightly wider slab just below the battlements
        // =============================
        float parapet_h = 0.08f * tW;
        Graphics::draw_iso_box(renderer, cam,
            tower_x - 0.04f, tower_y - 0.04f,
            tower_w + 0.08f, tower_d + 0.08f,
            parapet_h, stone_light, tower_h);

        // =============================
        // 5. BATTLEMENTS (Merlons)
        // Row of square teeth around the top
        // =============================
        float merlon_w = 0.10f, merlon_d = 0.10f, merlon_h = 0.14f * tW;
        float merlon_gap = 0.16f;  // spacing between merlons
        float merlon_z = tower_h + parapet_h;

        // Front-left edge merlons (along X axis)
        for (float mx = tower_x - 0.04f; mx < tower_x + tower_w; mx += merlon_gap) {
            Graphics::draw_iso_box(renderer, cam,
                mx, tower_y - 0.04f,
                merlon_w, merlon_d, merlon_h,
                merlon_col, merlon_z);
        }
        // Front-right edge merlons (along Y axis)
        for (float my = tower_y - 0.04f; my < tower_y + tower_d; my += merlon_gap) {
            Graphics::draw_iso_box(renderer, cam,
                tower_x + tower_w - 0.04f, my,
                merlon_d, merlon_w, merlon_h,
                merlon_col, merlon_z);
        }

        // =============================
        // 6. STONE BLOCK TEXTURE LINES
        // Horizontal lines across both visible faces
        // =============================
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 35);
        SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
        int num_courses = 7;
        for (int i = 1; i <= num_courses; ++i) {
            float stone_z = tower_h * ((float)i / (num_courses + 1));

            // Left face stone line
            Point2D sl1 = cam.project(tower_x,           tower_y,           stone_z);
            Point2D sl2 = cam.project(tower_x,           tower_y + tower_d, stone_z);
            SDL_RenderDrawLine(renderer, sl1.x, sl1.y, sl2.x, sl2.y);

            // Right face stone line
            Point2D sr1 = cam.project(tower_x + tower_w, tower_y,           stone_z);
            Point2D sr2 = cam.project(tower_x + tower_w, tower_y + tower_d, stone_z);
            SDL_RenderDrawLine(renderer, sr1.x, sr1.y, sr2.x, sr2.y);
        }

        // =============================
        // 7. FLAG / BANNER on top
        // =============================
        float flag_z = merlon_z + merlon_h;
        Point2D flag_base = cam.project(tower_x + tower_w * 0.5f, tower_y + tower_d * 0.5f, flag_z);

        // Pole
        SDL_SetRenderDrawColor(renderer, 200, 200, 200, 255);
        SDL_RenderDrawLine(renderer,
            flag_base.x, flag_base.y,
            flag_base.x, flag_base.y - (int)(22 * z));

        // Banner (team color — gold for base)
        SDL_Color banner = {255, 200, 0, 255};
        SDL_Rect b1 = { flag_base.x, flag_base.y - (int)(22*z), (int)(10*z), (int)(7*z) };
        SDL_SetRenderDrawColor(renderer, banner.r, banner.g, banner.b, 255);
        SDL_RenderFillRect(renderer, &b1);
        SDL_SetRenderDrawColor(renderer, 200, 150, 0, 255);
        SDL_RenderDrawRect(renderer, &b1);
    }
};

class WarehouseRenderer : public BaseRenderer {
public:
    void render(SDL_Renderer* renderer, const IsoCamera& cam, const json& board, int r, int c) override {
        const auto& tile = board[r][c];
        int status = tile["status"];
        if (status != 2 && status != 6) return; 

        SDL_Color beige = {210, 180, 140, 255};
        SDL_Color slate = {60, 60, 70, 255};
        
        Graphics::draw_complete_building(
            renderer, cam, 
            (float)c + 0.2f, (float)r + 0.2f, 
            0.6f, 0.6f, 
            0.4f * cam.tileW, 0.1f * cam.tileW, 
            beige, slate
        );

        SDL_Color crateCol = {139, 69, 19, 255};
        Graphics::draw_iso_box(renderer, cam, (float)c + 0.8f, (float)r + 0.2f, 0.15f, 0.15f, 0.1f * cam.tileW, crateCol, 0.0f);
    }
};

class CropRenderer : public BaseRenderer {
public:
    void render(SDL_Renderer* renderer, const IsoCamera& cam, const json& board, int r, int c) override {
        const auto& tile = board[r][c];
        int status = tile["status"];
        if (status != 5 && status != 7) return; 

        SDL_Color soilCol = {101, 67, 33, 255}; 
        SDL_Color plantCol = {100, 200, 50, 255}; 

        Graphics::draw_iso_box(renderer, cam, (float)c + 0.1f, (float)r + 0.1f, 0.8f, 0.8f, 0.05f * cam.tileW, soilCol);

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
    void render(SDL_Renderer* renderer, const IsoCamera& cam, const json& board, int r, int c) override {
        const auto& tile = board[r][c];
        int status = tile["status"];
        if (status != 1 && status != 4) return; 

        Point2D p = cam.project(c + 0.5f, r + 0.5f);
        float z = cam.zoom;

        SDL_Color buildingCol = {139, 69, 19, 255}; 
        float cabinHeight = 0.6f * cam.tileW;
        Graphics::draw_iso_box(renderer, cam, (float)c + 0.2f, (float)r + 0.2f, 0.3f, 0.3f, cabinHeight, buildingCol);

        SDL_Color roofCol = {80, 80, 90, 255};
        Graphics::draw_iso_box(renderer, cam, (float)c + 0.15f, (float)r + 0.15f, 0.7f, 0.7f, 0.1f * cam.tileW, roofCol, cabinHeight); 

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
    void render(SDL_Renderer* renderer, const IsoCamera& cam, const json& board, int r, int c) override {
        const auto& tile = board[r][c];
        if (!tile.contains("wood") || tile["wood"].get<int>() <= 0) return;

        int status = tile["status"];
        if (status == 1 || status == 4) return;

        Point2D p = cam.project(c + 0.5f, r + 0.5f);
        float z = cam.zoom;

        SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 60); 
        Graphics::fill_circle(renderer, p.x, p.y, (int)(10 * z));

        SDL_SetRenderDrawColor(renderer, 101, 67, 33, 255);
        int trunkW = (int)(6 * z);
        int trunkH = (int)(25 * z);
        SDL_Rect trunk = { p.x - trunkW / 2, p.y - trunkH, trunkW, trunkH };
        SDL_RenderFillRect(renderer, &trunk);

        int crownY = p.y - (int)(35 * z);
        SDL_SetRenderDrawColor(renderer, 20, 100, 20, 255);
        Graphics::fill_circle(renderer, p.x, crownY, (int)(18 * z));
        SDL_SetRenderDrawColor(renderer, 34, 139, 34, 255);
        Graphics::fill_circle(renderer, p.x, crownY - (int)(4 * z), (int)(14 * z));
        SDL_SetRenderDrawColor(renderer, 60, 200, 60, 255);
        Graphics::fill_circle(renderer, p.x - (int)(3 * z), crownY - (int)(7 * z), (int)(7 * z));
    }
};

class WallRenderer : public BaseRenderer {
public:
    void render(SDL_Renderer* renderer, const IsoCamera& cam, const json& board, int r, int c) override {
        const auto& tile = board[r][c];
        if (!tile.contains("fortified")) return;
        int f_lvl = tile["fortified"];
        if (f_lvl <= 0) return;

        int rows = board.size();
        int cols = board[0].size();
        int owner = tile["owner"];

        // Find closest enemy tiles
        float min_dist = 1e9f;
        std::vector<std::pair<int, int>> enemies;

        for (int er = 0; er < rows; ++er) {
            for (int ec = 0; ec < cols; ++ec) {
                int e_owner = board[er][ec]["owner"];
                if (e_owner != 0 && e_owner != owner) {
                    float d = std::sqrt(std::pow(er - r, 2) + std::pow(ec - c, 2));
                    if (d < min_dist - 0.001f) {
                        min_dist = d;
                        enemies.clear();
                        enemies.push_back({er, ec});
                    } else if (std::abs(d - min_dist) < 0.001f) {
                        enemies.push_back({er, ec});
                    }
                }
            }
        }

        bool draw_n = false, draw_s = false, draw_w = false, draw_e = false;

        if (enemies.empty()) {
            draw_n = draw_s = draw_w = draw_e = true; // All borders if no enemy found
        } else {
            float min_border_dist = 1e9f;
            float dists[4]; // N, S, W, E
            float bx[4] = {(float)c + 0.5f, (float)c + 0.5f, (float)c, (float)c + 1.0f};
            float by[4] = {(float)r, (float)r + 1.0f, (float)r + 0.5f, (float)r + 0.5f};

            for (int i = 0; i < 4; ++i) {
                dists[i] = 1e9f;
                for (auto& e : enemies) {
                    float d = std::sqrt(std::pow(bx[i] - (e.second + 0.5f), 2) + std::pow(by[i] - (e.first + 0.5f), 2));
                    if (d < dists[i]) dists[i] = d;
                }
                if (dists[i] < min_border_dist - 0.001f) min_border_dist = dists[i];
            }

            if (std::abs(dists[0] - min_border_dist) < 0.001f) draw_n = true;
            if (std::abs(dists[1] - min_border_dist) < 0.001f) draw_s = true;
            if (std::abs(dists[2] - min_border_dist) < 0.001f) draw_w = true;
            if (std::abs(dists[3] - min_border_dist) < 0.001f) draw_e = true;
        }

        SDL_Color wallCol = (f_lvl == 1) ? SDL_Color{160, 120, 60, 255} : SDL_Color{140, 140, 150, 255};
        float wallH = (f_lvl == 1) ? 0.25f : 0.45f;
        float wallT = (f_lvl == 1) ? 0.12f : 0.18f;

        auto draw_pillar = [&](float px, float py) {
            Graphics::draw_iso_box(renderer, cam, px - wallT*0.8f, py - wallT*0.8f, wallT * 1.6f, wallT * 1.6f, (wallH + 0.1f) * cam.tileW, wallCol);
        };

        // Draw the wall segments
        if (draw_n) Graphics::draw_iso_box(renderer, cam, (float)c, (float)r, 1.0f, wallT, wallH * cam.tileW, wallCol);
        if (draw_w) Graphics::draw_iso_box(renderer, cam, (float)c, (float)r, wallT, 1.0f, wallH * cam.tileW, wallCol);
        if (draw_s) Graphics::draw_iso_box(renderer, cam, (float)c, (float)r + 1.0f - wallT, 1.0f, wallT, wallH * cam.tileW, wallCol);
        if (draw_e) Graphics::draw_iso_box(renderer, cam, (float)c + 1.0f - wallT, (float)r, wallT, 1.0f, wallH * cam.tileW, wallCol);

        // Draw Pillars to "join" segments
        if (draw_n || draw_w) draw_pillar((float)c, (float)r);
        if (draw_n || draw_e) draw_pillar((float)c + 1.0f, (float)r);
        if (draw_s || draw_w) draw_pillar((float)c, (float)r + 1.0f);
        if (draw_s || draw_e) draw_pillar((float)c + 1.0f, (float)r + 1.0f);
    }
};

class UnitRenderer : public BaseRenderer {
public:
    void render(SDL_Renderer* renderer, const IsoCamera& cam, const json& board, int r, int c) override {
        const auto& tile = board[r][c];
        if (tile.contains("soldiers") && tile["soldiers"].get<int>() > 0) {
            draw_unit(renderer, cam, r, c, tile["soldiers"].get<int>(), {220, 50, 50, 255}, -0.3f);
        }
        if (tile.contains("p2_soldiers") && tile["p2_soldiers"].get<int>() > 0) {
            draw_unit(renderer, cam, r, c, tile["p2_soldiers"].get<int>(), {50, 120, 220, 255}, 0.3f);
        }
    }

private:
    void draw_unit(SDL_Renderer* renderer, const IsoCamera& cam, int r, int c, int count, SDL_Color col, float offset) {
        float z = cam.zoom;

        // Animation phase: unique per tile position + offset side
        float phase = SDL_GetTicks() / 400.0f + (r * 7.3f + c * 3.7f + offset * 5.0f);
        float leg_swing  = sinf(phase) * 0.06f;   // leg bob in world units
        float arm_swing  = -sinf(phase) * 0.05f;  // opposite to legs
        float body_bob   = fabsf(sinf(phase)) * 2.0f * z; // vertical bob in pixels

        // Place 1 or 2 figures depending on count
        int num_figures = (count < 2) ? count : 2;
        float fig_offsets[2] = { offset - 0.08f, offset + 0.08f };

        for (int f = 0; f < num_figures; ++f) {
            float fx = c + 0.5f + fig_offsets[f];
            float fy = r + 0.5f + fig_offsets[f] * 0.5f;

            // ---- Skin & clothing colors ----
            SDL_Color skin    = {220, 170, 120, 255};
            SDL_Color armor   = col;                                         // torso = team color
            SDL_Color dark    = {(Uint8)(col.r*0.6f), (Uint8)(col.g*0.6f), (Uint8)(col.b*0.6f), 255}; // legs
            SDL_Color helmet  = {(Uint8)(col.r*0.8f), (Uint8)(col.g*0.8f), (Uint8)(col.b*0.8f), 255};

            float S = 0.09f;   // base unit size in world coords
            float body_bob_z = body_bob; // pixel offset upward

            // Heights in world-z units (tileW scale):
            // 1 unit z = cam.tileW pixels tall
            float boot_h   = 0.04f * cam.tileW;
            float leg_h    = 0.10f * cam.tileW;
            float torso_h  = 0.13f * cam.tileW;
            float head_h   = 0.09f * cam.tileW;
            float arm_h    = 0.10f * cam.tileW;

            float z0 = 0.0f;                         // ground
            float z1 = boot_h;                       // above boots
            float z2 = z1 + leg_h;                   // above legs (waist)
            float z3 = z2 + torso_h;                 // above torso (shoulder)
            float z4 = z3 + head_h;                  // top of head

            // --- Shadow ---
            Point2D ps = cam.project(fx, fy, 0.0f);
            SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
            SDL_SetRenderDrawColor(renderer, 0, 0, 0, 50);
            Graphics::fill_circle(renderer, ps.x, ps.y, (int)(8 * z));

            // Helper lambda: draw a box offset by body_bob_z pixels upward
            // We achieve the vertical bob by nudging the screen projection.
            // Since project() handles z in pixels via `z * zoom`, we add body_bob_z
            // to z_start in world units: body_bob_z / cam.zoom
            float bob_wu = body_bob_z / cam.zoom; // bob in world-z units

            // --- LEFT LEG (swings forward) ---
            SDL_Color boot_col = {40, 40, 40, 255};
            Graphics::draw_iso_box(renderer, cam,
                fx - S*1.1f + leg_swing, fy - S*0.5f,
                S, S*0.9f, boot_h, boot_col, bob_wu + z0);
            Graphics::draw_iso_box(renderer, cam,
                fx - S*1.1f + leg_swing, fy - S*0.5f,
                S*0.9f, S*0.9f, leg_h, dark, bob_wu + z1);

            // --- RIGHT LEG (swings backward) ---
            Graphics::draw_iso_box(renderer, cam,
                fx + S*0.1f - leg_swing, fy - S*0.5f,
                S, S*0.9f, boot_h, boot_col, bob_wu + z0);
            Graphics::draw_iso_box(renderer, cam,
                fx + S*0.1f - leg_swing, fy - S*0.5f,
                S*0.9f, S*0.9f, leg_h, dark, bob_wu + z1);

            // --- TORSO ---
            Graphics::draw_iso_box(renderer, cam,
                fx - S*1.1f, fy - S*0.5f,
                S*2.0f, S*0.9f, torso_h, armor, bob_wu + z2);

            // --- LEFT ARM (swings opposite to left leg) ---
            Graphics::draw_iso_box(renderer, cam,
                fx - S*1.5f + arm_swing, fy - S*0.5f,
                S*0.6f, S*0.6f, arm_h, armor, bob_wu + z2);

            // --- RIGHT ARM ---
            Graphics::draw_iso_box(renderer, cam,
                fx + S*0.85f - arm_swing, fy - S*0.5f,
                S*0.6f, S*0.6f, arm_h, armor, bob_wu + z2);

            // --- HEAD ---
            Graphics::draw_iso_box(renderer, cam,
                fx - S*0.7f, fy - S*0.45f,
                S*1.4f, S*0.9f, head_h, skin, bob_wu + z3);

            // --- HELMET (thin slab on top of head) ---
            Graphics::draw_iso_box(renderer, cam,
                fx - S*0.85f, fy - S*0.55f,
                S*1.6f, S*1.0f, head_h * 0.25f, helmet, bob_wu + z3 + head_h);

            // --- SOLDIER COUNT badge (if > 2) ---
            if (count > 2 && f == 0) {
                Point2D pb = cam.project(fx, fy, z4 + head_h * 0.3f + bob_wu);
                SDL_SetRenderDrawColor(renderer, 255, 255, 255, 200);
                Graphics::fill_circle(renderer, pb.x, pb.y - (int)(12*z), (int)(5*z));
            }
        }
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
