#pragma once
#include "base_renderer.hpp"

class UnitRenderer : public BaseRenderer {
public:
    void render(SDL_Renderer* renderer, const IsoCamera& cam, const json& board, int r, int c) override;

private:
    void draw_unit(SDL_Renderer* renderer, const IsoCamera& cam, int r, int c, int count, SDL_Color col, float offset);
};
