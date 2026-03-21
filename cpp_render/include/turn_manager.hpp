#pragma once

#include "game_state.hpp"
#include "resource_system.hpp"
#include "build_system.hpp"
#include "combat_system.hpp"

class TurnManager {
public:
    struct ActionBundle {
        int diplomacy = 0;
        std::vector<int> economy;
        int target_tile = -1;
        int soldier_target = -1;
        int fortify_tile = -1;
    };

    void step(GameState& state, const ActionBundle& p1_action, const ActionBundle& p2_action);
    bool is_terminal(const GameState& state) const;
    float compute_reward(const GameState& prev, const GameState& next, int player_id) const;

private:
    ResourceSystem resource_system;
    BuildSystem build_system;
    CombatSystem combat_system;

    void apply_diplomacy(GameState& state, const ActionBundle& action, int player_id);
    void apply_economy(GameState& state, const ActionBundle& action, int player_id);
    void apply_target_tile(GameState& state, const ActionBundle& action, int player_id);
    void apply_soldier_action(GameState& state, const ActionBundle& action, int player_id);
    void apply_fortify(GameState& state, const ActionBundle& action, int player_id);
};
