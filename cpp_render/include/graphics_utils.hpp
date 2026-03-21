#pragma once
#include <SDL.h>
#include "camera.hpp"

namespace Graphics {
    void draw_iso_diamond(SDL_Renderer* renderer, const IsoCamera& cam, float x, float y, float size, SDL_Color color, SDL_Color border_color = {255, 255, 255, 30});
    
    void draw_iso_box(SDL_Renderer* renderer, const IsoCamera& cam, float x, float y, float w, float d, float h, SDL_Color color, float z_start = 0.0f);
    
    void draw_iso_box_stacked(SDL_Renderer* renderer, const IsoCamera& cam, float x, float y, float w, float d, float h, SDL_Color color, float z_start);
    
    void draw_complete_building(SDL_Renderer* renderer, const IsoCamera& cam, float x, float y, float w, float d, float wallH, float roofH, SDL_Color wallCol, SDL_Color roofCol);
    
    void fill_circle(SDL_Renderer* renderer, int x, int y, int radius);
}
