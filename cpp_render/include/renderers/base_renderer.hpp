#pragma once
#include <SDL.h>
#include <nlohmann/json.hpp>
#include "camera.hpp"
#include "graphics_utils.hpp"

using json = nlohmann::json;

class BaseRenderer {
public:
    virtual ~BaseRenderer() = default;
    virtual void render(SDL_Renderer* renderer, const IsoCamera& cam, const json& board, int r, int c) = 0;
};
