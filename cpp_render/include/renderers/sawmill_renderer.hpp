#pragma once
#include "base_renderer.hpp"

class SawmillRenderer : public BaseRenderer {
public:
    void render(SDL_Renderer* renderer, const IsoCamera& cam, const json& board, int r, int c) override;
};
