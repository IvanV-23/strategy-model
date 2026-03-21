#include "resource_system.hpp"

ActionCost ResourceSystem::get_cost(BuildAction action) {
    ActionCost cost;
    switch (action) {
        case BuildAction::Sawmill: cost.gold = 100; cost.wood = 50; break;
        case BuildAction::Warehouse: cost.gold = 60; cost.wood = 40; break;
        case BuildAction::Crops: cost.gold = 40; cost.wood = 20; break;
        case BuildAction::Fortify: cost.gold = 30; cost.wood = 15; break;
        default: break;
    }
    return cost;
}

void ResourceSystem::tick(GameState& state) {
    double gold_income = 0.0;
    double wood_income = 0.0;
    
    for (auto& row : state.board) {
        for (auto& tile : row) {
            if (tile.owner == 1) {
                switch (tile.status) {
                    case TileStatus::Sawmill:
                        wood_income += 10.0;
                        break;
                    case TileStatus::Warehouse:
                        gold_income += 5.0;
                        break;
                    case TileStatus::Crops:
                        gold_income += 3.0;
                        break;
                    case TileStatus::Base:
                        gold_income += 10.0;
                        break;
                    default:
                        if (tile.wood > 0) {
                            wood_income += static_cast<double>(tile.wood);
                        }
                        break;
                }
            }
        }
    }
    
    state.p_res.gold += gold_income;
    state.p_res.wood += wood_income;
}

void ResourceSystem::apply_build_cost(GameState& state, BuildAction action) {
    ActionCost cost = get_cost(action);
    state.p_res.gold -= cost.gold;
    state.p_res.wood -= cost.wood;
}

bool ResourceSystem::can_afford(const GameState& state, BuildAction action) const {
    ActionCost cost = get_cost(action);
    return state.p_res.gold >= cost.gold && state.p_res.wood >= cost.wood;
}
