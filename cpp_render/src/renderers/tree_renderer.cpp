#include "renderers/tree_renderer.hpp"

void TreeRenderer::render(SDL_Renderer* renderer, const IsoCamera& cam, const json& board, int r, int c) {
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
