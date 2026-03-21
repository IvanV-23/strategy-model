#pragma once
#include "base_renderer.hpp"

class TowerRenderer : public BaseRenderer {
public:
    void render(SDL_Renderer* renderer, const IsoCamera& cam, const json& board, int r, int c) override;

private:
    void draw_tower(SDL_Renderer* renderer, const IsoCamera& cam, float cx, float cy);
};
