#include "renderers/tile_renderer.hpp"

void TileRenderer::render(SDL_Renderer* renderer, const IsoCamera& cam, const json& board, int r, int c) {
    const auto& tile = board[r][c];
    int owner = tile["owner"];
    int tile_type = tile.value("tile_type", 0);
    
    SDL_Color normal_color = {50, 160, 50, 255};
    SDL_Color water_color = {50, 100, 200, 255};
    SDL_Color tile_color = (tile_type == 1) ? water_color : normal_color;
    
    SDL_Color border_col = {255, 255, 255, 30}; // Default neutral (transparent white)
    
    if (owner == 1) border_col = {50, 100, 255, 255}; // P1 Blue
    else if (owner == 2) border_col = {255, 50, 50, 255}; // P2 Red
    
    Graphics::draw_iso_diamond(renderer, cam, (float)c, (float)r, 1.0f, tile_color, border_col);
}
