#include <gtest/gtest.h>
#include "turn_manager.hpp"
#include "board_factory.hpp"

TEST(TurnManager, Step_IncreasesTurnCounter) {
    auto gs = BoardFactory::create_default(16, 16, 0, 0, 15, 15);
    gs.turn = 0;
    
    TurnManager tm;
    TurnManager::ActionBundle empty;
    tm.step(gs, empty, empty);
    
    EXPECT_EQ(gs.turn, 1);
}

TEST(TurnManager, Step_MultipleSteps) {
    auto gs = BoardFactory::create_default(16, 16, 0, 0, 15, 15);
    gs.turn = 0;
    
    TurnManager tm;
    TurnManager::ActionBundle empty;
    tm.step(gs, empty, empty);
    tm.step(gs, empty, empty);
    tm.step(gs, empty, empty);
    
    EXPECT_EQ(gs.turn, 3);
}

TEST(TurnManager, IsTerminal_WhenBaseDestroyed) {
    auto gs = BoardFactory::create_default(16, 16, 0, 0, 15, 15);
    
    TurnManager tm;
    EXPECT_FALSE(tm.is_terminal(gs));
    
    gs.board[0][0].status = TileStatus::Empty;
    EXPECT_TRUE(tm.is_terminal(gs));
}

TEST(TurnManager, IsTerminal_BothBasesExist) {
    auto gs = BoardFactory::create_default(16, 16, 0, 0, 15, 15);
    
    TurnManager tm;
    EXPECT_FALSE(tm.is_terminal(gs));
}

TEST(TurnManager, Reward_PositiveForTileCapture) {
    auto gs = BoardFactory::create_default(16, 16, 0, 0, 15, 15);
    
    GameState prev = gs;
    prev.board[1][1].owner = 0;
    gs.board[1][1].owner = 1;
    
    TurnManager tm;
    float reward = tm.compute_reward(prev, gs, 1);
    
    EXPECT_GT(reward, 0.0f);
}

TEST(TurnManager, Reward_NegativeForBaseLoss) {
    auto gs = BoardFactory::create_default(16, 16, 0, 0, 15, 15);
    
    GameState prev = gs;
    gs.board[0][0].status = TileStatus::Empty;
    
    TurnManager tm;
    float reward = tm.compute_reward(prev, gs, 1);
    
    EXPECT_LT(reward, 0.0f);
}

TEST(TurnManager, ApplyAction_SpawnSoldier) {
    auto gs = BoardFactory::create_default(16, 16, 0, 0, 15, 15);
    
    TurnManager tm;
    TurnManager::ActionBundle action;
    action.economy = {1, 1};
    tm.step(gs, action, TurnManager::ActionBundle());
    
    EXPECT_GT(gs.board[0][0].soldiers, 0);
}

TEST(TurnManager, ApplyAction_BuildStructure) {
    auto gs = BoardFactory::create_default(16, 16, 0, 0, 15, 15);
    gs.board[1][1].owner = 1;
    gs.board[1][1].status = TileStatus::Empty;
    gs.p_res.gold = 200;
    gs.p_res.wood = 100;
    
    TurnManager tm;
    TurnManager::ActionBundle action;
    action.diplomacy = 1;
    action.economy = {1};
    tm.step(gs, action, TurnManager::ActionBundle());
    
    EXPECT_EQ(gs.board[1][1].status, TileStatus::Sawmill);
}

TEST(TurnManager, Reward_ZeroForNoChange) {
    auto gs = BoardFactory::create_default(16, 16, 0, 0, 15, 15);
    
    GameState prev = gs;
    
    TurnManager tm;
    float reward = tm.compute_reward(prev, gs, 1);
    
    EXPECT_EQ(reward, 0.0f);
}

TEST(TurnManager, ActionBundle_DefaultValues) {
    TurnManager::ActionBundle action;
    
    EXPECT_EQ(action.diplomacy, 0);
    EXPECT_EQ(action.target_tile, -1);
    EXPECT_EQ(action.soldier_target, -1);
    EXPECT_EQ(action.fortify_tile, -1);
    EXPECT_TRUE(action.economy.empty());
}

TEST(TurnManager, Step_BothPlayersAct) {
    auto gs = BoardFactory::create_default(16, 16, 0, 0, 15, 15);
    
    TurnManager tm;
    TurnManager::ActionBundle p1, p2;
    p1.economy = {1, 1};
    p2.economy = {1, 1};
    tm.step(gs, p1, p2);
    
    EXPECT_GT(gs.board[0][0].soldiers, 0);
    EXPECT_GT(gs.board[15][15].p2_soldiers, 0);
}
