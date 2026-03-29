#include "combat_system.hpp"

void CombatSystem::spawn_soldier(GameState& state, int player_id) {
    int base_r = (player_id == 1) ? 0 : (int)state.board.size() - 1;
    int base_c = (player_id == 1) ? 0 : (int)state.board[0].size() - 1;
    
    if (player_id == 1) {
        state.board[base_r][base_c].soldiers += 1;
    } else {
        state.board[base_r][base_c].p2_soldiers += 1;
    }
}

void CombatSystem::move_soldier(GameState& state, int player_id, int target_r, int target_c) {
    if (target_r < 0 || target_r >= (int)state.board.size() || 
        target_c < 0 || target_c >= (int)state.board[0].size()) {
        return;
    }
    
    int base_r = (player_id == 1) ? 0 : (int)state.board.size() - 1;
    int base_c = (player_id == 1) ? 0 : (int)state.board[0].size() - 1;
    
    if (player_id == 1) {
        if (state.board[base_r][base_c].soldiers > 0) {
            state.board[base_r][base_c].soldiers -= 1;
            state.board[target_r][target_c].soldiers += 1;
            
            ShotEvent shot;
            shot.from = {base_r, base_c};
            shot.to = {target_r, target_c};
            state.shot_events.push_back(shot);
        }
    } else {
        if (state.board[base_r][base_c].p2_soldiers > 0) {
            state.board[base_r][base_c].p2_soldiers -= 1;
            state.board[target_r][target_c].p2_soldiers += 1;
            
            ShotEvent shot;
            shot.from = {base_r, base_c};
            shot.to = {target_r, target_c};
            state.shot_events.push_back(shot);
        }
    }
}

void CombatSystem::resolve_combat(GameState& state) {
    for (auto& row : state.board) {
        for (auto& tile : row) {
            int p1_soldiers = tile.soldiers;
            int p2_soldiers = tile.p2_soldiers;
            
            if (p1_soldiers > 0 && p2_soldiers > 0) {
                int p1_power = p1_soldiers;
                int p2_power = p2_soldiers;
                
                // Fortification gives +50 power per level
                p1_power += tile.fortified * 50;
                
                if (p1_power > p2_power) {
                    int remaining = p1_power - p2_power;
                    tile.soldiers = remaining;
                    tile.p2_soldiers = 0;
                } else if (p2_power > p1_power) {
                    int remaining = p2_power - p1_power;
                    tile.p2_soldiers = remaining;
                    tile.soldiers = 0;
                } else {
                    tile.soldiers = 0;
                    tile.p2_soldiers = 0;
                }
            }
        }
    }
}

std::vector<ShotEvent> CombatSystem::get_shot_events(const GameState& state) {
    return state.shot_events;
}
