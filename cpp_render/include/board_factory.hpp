#pragma once

#include "game_state.hpp"

class BoardFactory {
public:
    static GameState create_default(int rows, int cols,
                                   int p1_base_r, int p1_base_c,
                                   int p2_base_r, int p2_base_c);
};
