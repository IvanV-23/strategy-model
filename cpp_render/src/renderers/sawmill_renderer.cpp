#include "renderers/sawmill_renderer.hpp"

void SawmillRenderer::render(SDL_Renderer* renderer, const IsoCamera& cam, const json& board, int r, int c) {
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
