#include <gtest/gtest.h>
#include "combat_system.hpp"
#include "board_factory.hpp"

TEST(CombatSystem, Spawn_AddsSoldierOnBaseTile) {
    auto gs = BoardFactory::create_default(16, 16, 0, 0, 15, 15);
    
    CombatSystem cs;
    cs.spawn_soldier(gs, 1);
    
    EXPECT_EQ(gs.board[0][0].soldiers, 1);
}

TEST(CombatSystem, Spawn_P2SoldierOnP2Base) {
    auto gs = BoardFactory::create_default(16, 16, 0, 0, 15, 15);
    
    CombatSystem cs;
    cs.spawn_soldier(gs, 2);
    
    EXPECT_EQ(gs.board[15][15].p2_soldiers, 1);
}

TEST(CombatSystem, Move_RelocatesSoldiers) {
    auto gs = BoardFactory::create_default(16, 16, 0, 0, 15, 15);
    gs.board[0][0].soldiers = 5;
    
    CombatSystem cs;
    cs.move_soldier(gs, 1, 1, 1);
    
    EXPECT_EQ(gs.board[0][0].soldiers, 4);
    EXPECT_EQ(gs.board[1][1].soldiers, 1);
}

TEST(CombatSystem, Move_CreatesShotEvent) {
    auto gs = BoardFactory::create_default(16, 16, 0, 0, 15, 15);
    gs.board[0][0].soldiers = 5;
    
    CombatSystem cs;
    cs.move_soldier(gs, 1, 1, 1);
    
    EXPECT_FALSE(gs.shot_events.empty());
    EXPECT_EQ(gs.shot_events.back().from.first, 0);
    EXPECT_EQ(gs.shot_events.back().from.second, 0);
    EXPECT_EQ(gs.shot_events.back().to.first, 1);
    EXPECT_EQ(gs.shot_events.back().to.second, 1);
}

TEST(CombatSystem, ResolveCombat_SmallerArmyLoses) {
    auto gs = BoardFactory::create_default(16, 16, 0, 0, 15, 15);
    gs.board[5][5].soldiers = 3;
    gs.board[5][5].p2_soldiers = 1;
    
    CombatSystem cs;
    cs.resolve_combat(gs);
    
    EXPECT_EQ(gs.board[5][5].soldiers, 2);
    EXPECT_EQ(gs.board[5][5].p2_soldiers, 0);
}

TEST(CombatSystem, ResolveCombat_P2Wins) {
    auto gs = BoardFactory::create_default(16, 16, 0, 0, 15, 15);
    gs.board[5][5].soldiers = 2;
    gs.board[5][5].p2_soldiers = 5;
    
    CombatSystem cs;
    cs.resolve_combat(gs);
    
    EXPECT_EQ(gs.board[5][5].soldiers, 0);
    EXPECT_EQ(gs.board[5][5].p2_soldiers, 3);
}

TEST(CombatSystem, ResolveCombat_TieBothLose) {
    auto gs = BoardFactory::create_default(16, 16, 0, 0, 15, 15);
    gs.board[5][5].soldiers = 3;
    gs.board[5][5].p2_soldiers = 3;
    
    CombatSystem cs;
    cs.resolve_combat(gs);
    
    EXPECT_EQ(gs.board[5][5].soldiers, 0);
    EXPECT_EQ(gs.board[5][5].p2_soldiers, 0);
}

TEST(CombatSystem, ResolveCombat_FortifiedTileBonus) {
    auto gs = BoardFactory::create_default(16, 16, 0, 0, 15, 15);
    gs.board[5][5].soldiers = 1;
    gs.board[5][5].p2_soldiers = 1;
    gs.board[5][5].fortified = 2; // +100 power
    
    CombatSystem cs;
    cs.resolve_combat(gs);
    
    EXPECT_EQ(gs.board[5][5].soldiers, 100); // 101 - 1 = 100 remaining
    EXPECT_EQ(gs.board[5][5].p2_soldiers, 0);
}

TEST(CombatSystem, Move_NoSoldiersToMove) {
    auto gs = BoardFactory::create_default(16, 16, 0, 0, 15, 15);
    gs.board[0][0].soldiers = 0;
    
    CombatSystem cs;
    cs.move_soldier(gs, 1, 1, 1);
    
    EXPECT_EQ(gs.board[1][1].soldiers, 0);
}

TEST(CombatSystem, Move_InvalidTarget) {
    auto gs = BoardFactory::create_default(16, 16, 0, 0, 15, 15);
    gs.board[0][0].soldiers = 5;
    
    CombatSystem cs;
    cs.move_soldier(gs, 1, 100, 100);
    
    EXPECT_EQ(gs.board[0][0].soldiers, 5);
}

TEST(CombatSystem, GetShotEvents_ReturnsEvents) {
    auto gs = BoardFactory::create_default(16, 16, 0, 0, 15, 15);
    gs.board[0][0].soldiers = 5;
    
    CombatSystem cs;
    cs.move_soldier(gs, 1, 2, 2);
    cs.move_soldier(gs, 1, 3, 3);
    
    auto events = cs.get_shot_events(gs);
    EXPECT_EQ(events.size(), 2);
}

TEST(CombatSystem, Spawn_MultipleSoldiers) {
    auto gs = BoardFactory::create_default(16, 16, 0, 0, 15, 15);
    
    CombatSystem cs;
    cs.spawn_soldier(gs, 1);
    cs.spawn_soldier(gs, 1);
    cs.spawn_soldier(gs, 1);
    
    EXPECT_EQ(gs.board[0][0].soldiers, 3);
}

TEST(CombatSystem, ResolveCombat_NoCombatOnSeparateTiles) {
    auto gs = BoardFactory::create_default(16, 16, 0, 0, 15, 15);
    gs.board[1][1].soldiers = 3;
    gs.board[5][5].p2_soldiers = 3;
    
    CombatSystem cs;
    cs.resolve_combat(gs);
    
    EXPECT_EQ(gs.board[1][1].soldiers, 3);
    EXPECT_EQ(gs.board[5][5].p2_soldiers, 3);
}
