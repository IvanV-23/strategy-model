#include "renderers/warehouse_renderer.hpp"

void WarehouseRenderer::render(SDL_Renderer* renderer, const IsoCamera& cam, const json& board, int r, int c) {
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
