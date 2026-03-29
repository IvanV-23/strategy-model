#pragma once

#include "game_state.hpp"
#include "resource_system.hpp"

class BuildSystem {
public:
    bool apply_economy_action(GameState& state, int action_idx, int player_id);
    bool apply_fortify_action(GameState& state, int tile_r, int tile_c, int player_id);
    std::vector<bool> get_build_mask(const GameState& state, int player_id);
};
