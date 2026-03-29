#include <gtest/gtest.h>
#include "resource_system.hpp"
#include "board_factory.hpp"

TEST(ResourceSystem, Tick_SawmillGeneratesWood) {
    auto gs = BoardFactory::create_default(16, 16, 0, 0, 15, 15);
    gs.board[1][1].status = TileStatus::Sawmill;
    gs.board[1][1].owner = 1;
    double initial_wood = gs.p_res.wood;
    ResourceSystem rs;
    rs.tick(gs);
    EXPECT_GT(gs.p_res.wood, initial_wood);
}

TEST(ResourceSystem, Tick_WarehouseGeneratesGold) {
    auto gs = BoardFactory::create_default(16, 16, 0, 0, 15, 15);
    gs.board[1][1].status = TileStatus::Warehouse;
    gs.board[1][1].owner = 1;
    double initial_gold = gs.p_res.gold;
    ResourceSystem rs;
    rs.tick(gs);
    EXPECT_GT(gs.p_res.gold, initial_gold);
}

TEST(ResourceSystem, Tick_CropsGeneratesGold) {
    auto gs = BoardFactory::create_default(16, 16, 0, 0, 15, 15);
    gs.board[1][1].status = TileStatus::Crops;
    gs.board[1][1].owner = 1;
    double initial_gold = gs.p_res.gold;
    ResourceSystem rs;
    rs.tick(gs);
    EXPECT_GT(gs.p_res.gold, initial_gold);
}

TEST(ResourceSystem, Tick_BaseGeneratesGold) {
    auto gs = BoardFactory::create_default(16, 16, 0, 0, 15, 15);
    double initial_gold = gs.p_res.gold;
    ResourceSystem rs;
    rs.tick(gs);
    EXPECT_GT(gs.p_res.gold, initial_gold);
}

TEST(ResourceSystem, Tick_TileWoodGeneratesWood) {
    auto gs = BoardFactory::create_default(16, 16, 0, 0, 15, 15);
    gs.board[1][1].wood = 5;
    gs.board[1][1].owner = 1;
    gs.board[1][1].status = TileStatus::Empty;
    double initial_wood = gs.p_res.wood;
    ResourceSystem rs;
    rs.tick(gs);
    EXPECT_GT(gs.p_res.wood, initial_wood);
}

TEST(ResourceSystem, CanAfford_Sawmill) {
    auto gs = BoardFactory::create_default(16, 16, 0, 0, 15, 15);
    gs.p_res.gold = 100;
    gs.p_res.wood = 50;
    ResourceSystem rs;
    EXPECT_TRUE(rs.can_afford(gs, BuildAction::Sawmill));
}

TEST(ResourceSystem, CanAfford_InsufficientGold) {
    auto gs = BoardFactory::create_default(16, 16, 0, 0, 15, 15);
    gs.p_res.gold = 50;
    gs.p_res.wood = 50;
    ResourceSystem rs;
    EXPECT_FALSE(rs.can_afford(gs, BuildAction::Sawmill));
}

TEST(ResourceSystem, ApplyBuildCost_Sawmill) {
    auto gs = BoardFactory::create_default(16, 16, 0, 0, 15, 15);
    gs.p_res.gold = 100;
    gs.p_res.wood = 50;
    ResourceSystem rs;
    rs.apply_build_cost(gs, BuildAction::Sawmill);
    EXPECT_EQ(gs.p_res.gold, 0.0);
    EXPECT_EQ(gs.p_res.wood, 0.0);
}

TEST(ResourceSystem, GetCost_Sawmill) {
    ActionCost cost = ResourceSystem::get_cost(BuildAction::Sawmill);
    EXPECT_EQ(cost.gold, 100.0);
    EXPECT_EQ(cost.wood, 50.0);
}

TEST(ResourceSystem, GetCost_Warehouse) {
    ActionCost cost = ResourceSystem::get_cost(BuildAction::Warehouse);
    EXPECT_EQ(cost.gold, 60.0);
    EXPECT_EQ(cost.wood, 40.0);
}

TEST(ResourceSystem, GetCost_Crops) {
    ActionCost cost = ResourceSystem::get_cost(BuildAction::Crops);
    EXPECT_EQ(cost.gold, 40.0);
    EXPECT_EQ(cost.wood, 20.0);
}

TEST(ResourceSystem, GetCost_Fortify) {
    ActionCost cost = ResourceSystem::get_cost(BuildAction::Fortify);
    EXPECT_EQ(cost.gold, 30.0);
    EXPECT_EQ(cost.wood, 15.0);
}

TEST(ResourceSystem, Tick_DoesNotAffectP2Tiles) {
    auto gs = BoardFactory::create_default(16, 16, 0, 0, 15, 15);
    gs.board[1][1].status = TileStatus::Sawmill;
    gs.board[1][1].owner = 2;
    double initial_wood = gs.p_res.wood;
    ResourceSystem rs;
    rs.tick(gs);
    EXPECT_EQ(gs.p_res.wood, initial_wood);
}
