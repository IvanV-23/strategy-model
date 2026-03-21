#pragma once

#include "game_state.hpp"

class CombatSystem {
public:
    void spawn_soldier(GameState& state, int player_id);
    void move_soldier(GameState& state, int player_id, int target_r, int target_c);
    void resolve_combat(GameState& state);
    std::vector<ShotEvent> get_shot_events(const GameState& state);
};
