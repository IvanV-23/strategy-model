#include "board_factory.hpp"

GameState BoardFactory::create_default(int rows, int cols,
                                   int p1_base_r, int p1_base_c,
                                   int p2_base_r, int p2_base_c) {
    GameState gs;
    gs.board.resize(rows, std::vector<Tile>(cols));
    
    gs.board[p1_base_r][p1_base_c].owner = 1;
    gs.board[p1_base_r][p1_base_c].status = TileStatus::Base;
    gs.board[p1_base_r][p1_base_c].fortified = 2;
    
    gs.board[p2_base_r][p2_base_c].owner = 2;
    gs.board[p2_base_r][p2_base_c].status = TileStatus::Base;
    gs.board[p2_base_r][p2_base_c].fortified = 2;
    
    return gs;
}
