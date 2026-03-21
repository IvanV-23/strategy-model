#include "renderers/wall_renderer.hpp"
#include <vector>
#include <cmath>

void WallRenderer::render(SDL_Renderer* renderer, const IsoCamera& cam, const json& board, int r, int c) {
    const auto& tile = board[r][c];
    if (!tile.contains("fortified")) return;
    int f_lvl = tile["fortified"];
    if (f_lvl <= 0) return;

    int rows = (int)board.size();
    int cols = (int)board[0].size();
    int owner = tile["owner"];

    // Find closest enemy tiles
    float min_dist = 1e9f;
    std::vector<std::pair<int, int>> enemies;

    for (int er = 0; er < rows; ++er) {
        for (int ec = 0; ec < cols; ++ec) {
            int e_owner = board[er][ec]["owner"];
            if (e_owner != 0 && e_owner != owner) {
                float d = std::sqrt(std::pow((float)er - r, 2) + std::pow((float)ec - c, 2));
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
