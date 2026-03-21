#include "renderers/crop_renderer.hpp"

void CropRenderer::render(SDL_Renderer* renderer, const IsoCamera& cam, const json& board, int r, int c) {
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
