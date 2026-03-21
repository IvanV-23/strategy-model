#include <gtest/gtest.h>
#include "build_system.hpp"
#include "board_factory.hpp"

TEST(BuildSystem, BuildSawmill_UpdatesTileStatus) {
    auto gs = BoardFactory::create_default(16, 16, 0, 0, 15, 15);
    gs.board[1][1].owner = 1;
    gs.board[1][1].status = TileStatus::Empty;
    gs.p_res.gold = 100;
    gs.p_res.wood = 50;
    
    BuildSystem bs;
    bool result = bs.apply_economy_action(gs, 1, 1);
    
    EXPECT_TRUE(result);
    EXPECT_EQ(gs.board[1][1].status, TileStatus::Sawmill);
}

TEST(BuildSystem, BuildWarehouse_UpdatesTileStatus) {
    auto gs = BoardFactory::create_default(16, 16, 0, 0, 15, 15);
    gs.board[1][1].owner = 1;
    gs.board[1][1].status = TileStatus::Empty;
    gs.p_res.gold = 60;
    gs.p_res.wood = 40;
    
    BuildSystem bs;
    bool result = bs.apply_economy_action(gs, 2, 1);
    
    EXPECT_TRUE(result);
    EXPECT_EQ(gs.board[1][1].status, TileStatus::Warehouse);
}

TEST(BuildSystem, BuildCrops_UpdatesTileStatus) {
    auto gs = BoardFactory::create_default(16, 16, 0, 0, 15, 15);
    gs.board[1][1].owner = 1;
    gs.board[1][1].status = TileStatus::Empty;
    gs.p_res.gold = 40;
    gs.p_res.wood = 20;
    
    BuildSystem bs;
    bool result = bs.apply_economy_action(gs, 3, 1);
    
    EXPECT_TRUE(result);
    EXPECT_EQ(gs.board[1][1].status, TileStatus::Crops);
}

TEST(BuildSystem, BuildMask_InsufficientGold_BlocksBuild) {
    auto gs = BoardFactory::create_default(16, 16, 0, 0, 15, 15);
    gs.p_res.gold = 20;
    gs.p_res.wood = 10;
    
    BuildSystem bs;
    auto mask = bs.get_build_mask(gs, 1);
    
    EXPECT_FALSE(mask[1]); // Sawmill blocked (needs 100 gold, 50 wood)
    EXPECT_FALSE(mask[2]); // Warehouse blocked (needs 60 gold, 40 wood)
    EXPECT_FALSE(mask[3]); // Crops blocked (needs 40 gold, 20 wood)
    EXPECT_FALSE(mask[4]); // Fortify blocked (needs 30 gold, 15 wood)
}

TEST(BuildSystem, BuildMask_SufficientResources_EnablesBuild) {
    auto gs = BoardFactory::create_default(16, 16, 0, 0, 15, 15);
    gs.p_res.gold = 200;
    gs.p_res.wood = 200;
    
    BuildSystem bs;
    auto mask = bs.get_build_mask(gs, 1);
    
    EXPECT_TRUE(mask[1]);  // Sawmill enabled
    EXPECT_TRUE(mask[2]);  // Warehouse enabled
    EXPECT_TRUE(mask[3]);  // Crops enabled
    EXPECT_TRUE(mask[4]);  // Fortify enabled
}

TEST(BuildSystem, Fortify_CreatesWall) {
    auto gs = BoardFactory::create_default(16, 16, 0, 0, 15, 15);
    gs.board[1][1].owner = 1;
    gs.board[1][1].fortified = 0;
    gs.p_res.gold = 30;
    gs.p_res.wood = 15;
    
    BuildSystem bs;
    bool result = bs.apply_fortify_action(gs, 1, 1, 1);
    
    EXPECT_TRUE(result);
    EXPECT_EQ(gs.board[1][1].fortified, 1);
}

TEST(BuildSystem, Fortify_IncreasesFortificationLevel) {
    auto gs = BoardFactory::create_default(16, 16, 0, 0, 15, 15);
    gs.board[1][1].owner = 1;
    gs.board[1][1].fortified = 1;
    gs.p_res.gold = 60;
    gs.p_res.wood = 30;
    
    BuildSystem bs;
    bool result = bs.apply_fortify_action(gs, 1, 1, 1);
    
    EXPECT_TRUE(result);
    EXPECT_EQ(gs.board[1][1].fortified, 2);
}

TEST(BuildSystem, Fortify_CannotExceedMaxLevel) {
    auto gs = BoardFactory::create_default(16, 16, 0, 0, 15, 15);
    gs.board[1][1].owner = 1;
    gs.board[1][1].fortified = 2;
    
    BuildSystem bs;
    bool result = bs.apply_fortify_action(gs, 1, 1, 1);
    
    EXPECT_FALSE(result);
    EXPECT_EQ(gs.board[1][1].fortified, 2);
}

TEST(BuildSystem, Fortify_WrongPlayer_Fails) {
    auto gs = BoardFactory::create_default(16, 16, 0, 0, 15, 15);
    gs.board[1][1].owner = 2;
    gs.board[1][1].fortified = 0;
    gs.p_res.gold = 30;
    gs.p_res.wood = 15;
    
    BuildSystem bs;
    bool result = bs.apply_fortify_action(gs, 1, 1, 1);
    
    EXPECT_FALSE(result);
}

TEST(BuildSystem, Build_DeductsResources) {
    auto gs = BoardFactory::create_default(16, 16, 0, 0, 15, 15);
    gs.board[1][1].owner = 1;
    gs.board[1][1].status = TileStatus::Empty;
    gs.p_res.gold = 100;
    gs.p_res.wood = 50;
    
    BuildSystem bs;
    bs.apply_economy_action(gs, 1, 1);
    
    EXPECT_EQ(gs.p_res.gold, 0.0);
    EXPECT_EQ(gs.p_res.wood, 0.0);
}

TEST(BuildSystem, Build_CannotBuildOnNonEmptyTile) {
    auto gs = BoardFactory::create_default(16, 16, 0, 0, 15, 15);
    gs.board[1][1].owner = 1;
    gs.board[1][1].status = TileStatus::Sawmill;
    gs.p_res.gold = 200;
    gs.p_res.wood = 100;
    
    BuildSystem bs;
    bool result = bs.apply_economy_action(gs, 1, 1);
    
    EXPECT_FALSE(result);
}

TEST(BuildSystem, Build_P2BuildingGetsP2Status) {
    auto gs = BoardFactory::create_default(16, 16, 0, 0, 15, 15);
    gs.board[14][14].owner = 2;
    gs.board[14][14].status = TileStatus::Empty;
    gs.p_res.gold = 100;
    gs.p_res.wood = 50;
    
    BuildSystem bs;
    bool result = bs.apply_economy_action(gs, 1, 2);
    
    EXPECT_TRUE(result);
    EXPECT_EQ(gs.board[14][14].status, TileStatus::Sawmill_P2);
}

TEST(BuildSystem, Build_InvalidActionIdx_ReturnsFalse) {
    auto gs = BoardFactory::create_default(16, 16, 0, 0, 15, 15);
    gs.p_res.gold = 1000;
    gs.p_res.wood = 1000;
    
    BuildSystem bs;
    bool result = bs.apply_economy_action(gs, 99, 1);
    
    EXPECT_FALSE(result);
}
