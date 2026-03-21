#include "renderers/building_renderer.hpp"

void BuildingRenderer::render(SDL_Renderer* renderer, const IsoCamera& cam, const json& board, int r, int c) {
    const auto& tile = board[r][c];
    int status = tile["status"];

    if (status == 0 || status == 1 || status == 4 || status == 5 ||
        status == 7 || status == 2 || status == 6 || status == 3) return;

    SDL_Color col = {200, 200, 200, 255};
    float height = 0.5f;

    Graphics::draw_iso_box(renderer, cam, (float)c + 0.2f, (float)r + 0.2f, 0.6f, 0.6f, height * cam.tileW, col);
}
