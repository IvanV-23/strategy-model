#include "turn_manager.hpp"

void TurnManager::step(GameState& state, const ActionBundle& p1_action, const ActionBundle& p2_action) {
    state.turn += 1;
    
    resource_system.tick(state);
    resource_system.tick(state);
    
    apply_diplomacy(state, p1_action, 1);
    apply_diplomacy(state, p2_action, 2);
    
    apply_economy(state, p1_action, 1);
    apply_economy(state, p2_action, 2);
    
    apply_target_tile(state, p1_action, 1);
    apply_target_tile(state, p2_action, 2);
    
    apply_soldier_action(state, p1_action, 1);
    apply_soldier_action(state, p2_action, 2);
    
    apply_fortify(state, p1_action, 1);
    apply_fortify(state, p2_action, 2);
    
    combat_system.resolve_combat(state);
    
    state.shot_events.clear();
}

void TurnManager::apply_diplomacy(GameState& state, const ActionBundle& action, int player_id) {
    if (action.diplomacy == 1) {
        if (!action.economy.empty() && action.economy[0] >= 1) {
            build_system.apply_economy_action(state, action.economy[0], player_id);
        }
    }
}

void TurnManager::apply_economy(GameState& state, const ActionBundle& action, int player_id) {
    if (!action.economy.empty()) {
        int eco_action = action.economy[0];
        if (eco_action >= 1 && eco_action <= 3) {
            build_system.apply_economy_action(state, eco_action, player_id);
        }
    }
}

void TurnManager::apply_target_tile(GameState& state, const ActionBundle& action, int player_id) {
    if (action.target_tile < 0) return;
    
    int rows = (int)state.board.size();
    int cols = (int)state.board[0].size();
    int r = action.target_tile / cols;
    int c = action.target_tile % cols;
    
    if (r < 0 || r >= rows || c < 0 || c >= cols) return;
    
    int base_r = (player_id == 1) ? 0 : rows - 1;
    int base_c = (player_id == 1) ? 0 : cols - 1;
    
    Tile& target = state.board[r][c];
    
    if (player_id == 1) {
        if (target.soldiers > 0 && target.owner == 2) {
            int p1_power = state.board[base_r][base_c].soldiers * 10;
            int defense = target.soldiers * 10 + (target.fortified * 50);
            
            if (p1_power > defense) {
                target.owner = 1;
                target.soldiers = 0;
                target.p2_soldiers = 0;
            }
        }
    } else {
        if (target.p2_soldiers > 0 && target.owner == 1) {
            int p2_power = state.board[base_r][base_c].p2_soldiers * 10;
            int defense = target.p2_soldiers * 10 + (target.fortified * 50);
            
            if (p2_power > defense) {
                target.owner = 2;
                target.soldiers = 0;
                target.p2_soldiers = 0;
            }
        }
    }
}

void TurnManager::apply_soldier_action(GameState& state, const ActionBundle& action, int player_id) {
    if (!action.economy.empty() && action.economy.size() > 1) {
        int create_soldiers = action.economy[1];
        if (create_soldiers > 0) {
            int soldier_count = std::min(create_soldiers, 3);
            for (int i = 0; i < soldier_count; ++i) {
                combat_system.spawn_soldier(state, player_id);
            }
        }
    }
    
    if (action.soldier_target >= 0) {
        int rows = (int)state.board.size();
        int cols = (int)state.board[0].size();
        int r = action.soldier_target / cols;
        int c = action.soldier_target % cols;
        combat_system.move_soldier(state, player_id, r, c);
    }
}

void TurnManager::apply_fortify(GameState& state, const ActionBundle& action, int player_id) {
    if (action.fortify_tile >= 0) {
        int rows = (int)state.board.size();
        int cols = (int)state.board[0].size();
        int r = action.fortify_tile / cols;
        int c = action.fortify_tile % cols;
        build_system.apply_fortify_action(state, r, c, player_id);
    }
}

bool TurnManager::is_terminal(const GameState& state) const {
    for (const auto& row : state.board) {
        for (const auto& tile : row) {
            if (tile.status == TileStatus::Base && tile.owner == 1) {
                return false;
            }
        }
    }
    return true;
}

float TurnManager::compute_reward(const GameState& prev, const GameState& next, int player_id) const {
    float reward = 0.0f;
    
    int p1_tiles_prev = 0, p1_tiles_next = 0;
    int p2_tiles_prev = 0, p2_tiles_next = 0;
    
    for (const auto& row : prev.board) {
        for (const auto& tile : row) {
            if (tile.owner == 1) p1_tiles_prev++;
            if (tile.owner == 2) p2_tiles_prev++;
        }
    }
    
    for (const auto& row : next.board) {
        for (const auto& tile : row) {
            if (tile.owner == 1) p1_tiles_next++;
            if (tile.owner == 2) p2_tiles_next++;
        }
    }
    
    if (player_id == 1) {
        reward += (p1_tiles_next - p1_tiles_prev) * 1.0f;
    } else {
        reward += (p2_tiles_next - p2_tiles_prev) * 1.0f;
    }
    
    if (is_terminal(next)) {
        bool p1_base_exists = false;
        bool p2_base_exists = false;
        
        for (const auto& row : next.board) {
            for (const auto& tile : row) {
                if (tile.status == TileStatus::Base) {
                    if (tile.owner == 1) p1_base_exists = true;
                    if (tile.owner == 2) p2_base_exists = true;
                }
            }
        }
        
        if (player_id == 1 && !p1_base_exists) reward -= 100.0f;
        if (player_id == 2 && !p2_base_exists) reward -= 100.0f;
    }
    
    return reward;
}
