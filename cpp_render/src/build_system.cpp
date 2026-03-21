#include "build_system.hpp"

bool BuildSystem::apply_economy_action(GameState& state, int action_idx, int player_id) {
    int rows = (int)state.board.size();
    int cols = (int)state.board[0].size();
    
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (state.board[r][c].owner == player_id && state.board[r][c].status == TileStatus::Empty) {
                TileStatus target_status = TileStatus::Empty;
                BuildAction action = BuildAction::None;
                
                switch (action_idx) {
                    case 1: // build_sawmill
                        target_status = (player_id == 1) ? TileStatus::Sawmill : TileStatus::Sawmill_P2;
                        action = BuildAction::Sawmill;
                        break;
                    case 2: // build_warehouse
                        target_status = (player_id == 1) ? TileStatus::Warehouse : TileStatus::Warehouse_P2;
                        action = BuildAction::Warehouse;
                        break;
                    case 3: // build_crops
                        target_status = (player_id == 1) ? TileStatus::Crops : TileStatus::Crops_P2;
                        action = BuildAction::Crops;
                        break;
                    default:
                        return false;
                }
                
                ResourceSystem rs;
                if (rs.can_afford(state, action)) {
                    rs.apply_build_cost(state, action);
                    state.board[r][c].status = target_status;
                    return true;
                }
                return false;
            }
        }
    }
    return false;
}

bool BuildSystem::apply_fortify_action(GameState& state, int tile_r, int tile_c, int player_id) {
    if (tile_r < 0 || tile_r >= (int)state.board.size() || tile_c < 0 || tile_c >= (int)state.board[0].size()) {
        return false;
    }
    
    Tile& tile = state.board[tile_r][tile_c];
    if (tile.owner != player_id || tile.fortified >= 2) {
        return false;
    }
    
    ResourceSystem rs;
    if (rs.can_afford(state, BuildAction::Fortify)) {
        rs.apply_build_cost(state, BuildAction::Fortify);
        tile.fortified += 1;
        return true;
    }
    return false;
}

std::vector<bool> BuildSystem::get_build_mask(const GameState& state, int player_id) {
    std::vector<bool> mask(8, false);
    ResourceSystem rs;
    
    // Buy soldiers (0) - always available if player has tiles
    bool has_owned_tiles = false;
    for (const auto& row : state.board) {
        for (const auto& tile : row) {
            if (tile.owner == player_id) {
                has_owned_tiles = true;
                break;
            }
        }
        if (has_owned_tiles) break;
    }
    mask[0] = has_owned_tiles;
    
    // Build actions
    if (state.p_res.gold >= 100 && state.p_res.wood >= 50) mask[1] = true;  // Sawmill
    if (state.p_res.gold >= 60 && state.p_res.wood >= 40) mask[2] = true;   // Warehouse
    if (state.p_res.gold >= 40 && state.p_res.wood >= 20) mask[3] = true;   // Crops
    if (state.p_res.gold >= 30 && state.p_res.wood >= 15) mask[4] = true;   // Fortify
    mask[5] = true; // Gather (always available)
    mask[6] = true; // Create units (always available)
    mask[7] = has_owned_tiles; // Wait
    
    return mask;
}
