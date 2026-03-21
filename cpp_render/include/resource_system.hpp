#pragma once

#include "game_state.hpp"

enum class BuildAction {
    None = 0,
    Sawmill = 1,
    Warehouse = 2,
    Crops = 3,
    Fortify = 4
};

struct ActionCost {
    double gold = 0.0;
    double wood = 0.0;
};

class ResourceSystem {
public:
    void tick(GameState& state);
    void apply_build_cost(GameState& state, BuildAction action);
    bool can_afford(const GameState& state, BuildAction action) const;
    static ActionCost get_cost(BuildAction action);
};
