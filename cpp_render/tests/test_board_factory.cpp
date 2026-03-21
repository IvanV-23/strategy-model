#include <gtest/gtest.h>
#include "board_factory.hpp"

TEST(BoardFactory, CreateDefault_HasBothBases) {
    auto gs = BoardFactory::create_default(16, 16, 0, 0, 15, 15);
    EXPECT_EQ(gs.board[0][0].status, TileStatus::Base);
    EXPECT_EQ(gs.board[15][15].status, TileStatus::Base);
}

TEST(BoardFactory, CreateDefault_BaseOwnedByCorrectPlayer) {
    auto gs = BoardFactory::create_default(16, 16, 0, 0, 15, 15);
    EXPECT_EQ(gs.board[0][0].owner, 1);
    EXPECT_EQ(gs.board[15][15].owner, 2);
}

TEST(BoardFactory, CreateDefault_HasCorrectDimensions) {
    auto gs = BoardFactory::create_default(16, 16, 0, 0, 15, 15);
    EXPECT_EQ(gs.board.size(), 16);
    EXPECT_EQ(gs.board[0].size(), 16);
}

TEST(BoardFactory, CreateDefault_BasesAreFortified) {
    auto gs = BoardFactory::create_default(16, 16, 0, 0, 15, 15);
    EXPECT_EQ(gs.board[0][0].fortified, 2);
    EXPECT_EQ(gs.board[15][15].fortified, 2);
}

TEST(BoardFactory, CreateDefault_CustomPositions) {
    auto gs = BoardFactory::create_default(10, 10, 2, 3, 7, 8);
    EXPECT_EQ(gs.board[2][3].owner, 1);
    EXPECT_EQ(gs.board[2][3].status, TileStatus::Base);
    EXPECT_EQ(gs.board[7][8].owner, 2);
    EXPECT_EQ(gs.board[7][8].status, TileStatus::Base);
}
