#include <gtest/gtest.h>
#include "game_state.hpp"

using json = nlohmann::json;

TEST(GameState, FromJson_ParsesOwner) {
    json j = R"({
        "board": [
            [{"owner": 1, "status": 0, "wood": 0, "fortified": 0, "soldiers": 0, "p2_soldiers": 0}]
        ],
        "p_res": [100.0, 50.0],
        "turn": 5
    })"_json;
    
    auto gs = GameState::from_json(j);
    EXPECT_EQ(gs.board[0][0].owner, 1);
}

TEST(GameState, FromJson_ParsesStatus) {
    json j = R"({
        "board": [
            [{"owner": 1, "status": 3, "wood": 0, "fortified": 0, "soldiers": 0, "p2_soldiers": 0}]
        ],
        "p_res": [100.0, 50.0],
        "turn": 1
    })"_json;
    
    auto gs = GameState::from_json(j);
    EXPECT_EQ(gs.board[0][0].status, TileStatus::Base);
}

TEST(GameState, FromJson_ParsesResources) {
    json j = R"({
        "board": [[{"owner": 0, "status": 0, "wood": 0, "fortified": 0, "soldiers": 0, "p2_soldiers": 0}]],
        "p_res": [150.5, 75.25],
        "turn": 10
    })"_json;
    
    auto gs = GameState::from_json(j);
    EXPECT_DOUBLE_EQ(gs.p_res.gold, 150.5);
    EXPECT_DOUBLE_EQ(gs.p_res.wood, 75.25);
}

TEST(GameState, FromJson_ParsesTurn) {
    json j = R"({
        "board": [[{"owner": 0, "status": 0, "wood": 0, "fortified": 0, "soldiers": 0, "p2_soldiers": 0}]],
        "p_res": [0, 0],
        "turn": 42
    })"_json;
    
    auto gs = GameState::from_json(j);
    EXPECT_EQ(gs.turn, 42);
}

TEST(GameState, FromJson_ParsesSoldiers) {
    json j = R"({
        "board": [
            [{"owner": 1, "status": 0, "wood": 0, "fortified": 0, "soldiers": 5, "p2_soldiers": 3}]
        ],
        "p_res": [0, 0],
        "turn": 0
    })"_json;
    
    auto gs = GameState::from_json(j);
    EXPECT_EQ(gs.board[0][0].soldiers, 5);
    EXPECT_EQ(gs.board[0][0].p2_soldiers, 3);
}

TEST(GameState, FromJson_ParsesTargetTile) {
    json j = R"({
        "board": [[{"owner": 0, "status": 0, "wood": 0, "fortified": 0, "soldiers": 0, "p2_soldiers": 0}]],
        "p_res": [0, 0],
        "turn": 0,
        "target_tile": [3, 7]
    })"_json;
    
    auto gs = GameState::from_json(j);
    ASSERT_TRUE(gs.target_tile.has_value());
    EXPECT_EQ(gs.target_tile->first, 3);
    EXPECT_EQ(gs.target_tile->second, 7);
}

TEST(GameState, FromJson_ParsesSoldierTargetTile) {
    json j = R"({
        "board": [[{"owner": 0, "status": 0, "wood": 0, "fortified": 0, "soldiers": 0, "p2_soldiers": 0}]],
        "p_res": [0, 0],
        "turn": 0,
        "soldier_target_tile": [1, 2]
    })"_json;
    
    auto gs = GameState::from_json(j);
    ASSERT_TRUE(gs.soldier_target_tile.has_value());
    EXPECT_EQ(gs.soldier_target_tile->first, 1);
    EXPECT_EQ(gs.soldier_target_tile->second, 2);
}

TEST(GameState, FromJson_ParsesShotEvents) {
    json j = R"({
        "board": [[{"owner": 0, "status": 0, "wood": 0, "fortified": 0, "soldiers": 0, "p2_soldiers": 0}]],
        "p_res": [0, 0],
        "turn": 0,
        "shot_events": [
            {"from": [0, 1], "to": [2, 3]},
            {"from": [5, 5], "to": [6, 6]}
        ]
    })"_json;
    
    auto gs = GameState::from_json(j);
    ASSERT_EQ(gs.shot_events.size(), 2);
    EXPECT_EQ(gs.shot_events[0].from.first, 0);
    EXPECT_EQ(gs.shot_events[0].from.second, 1);
    EXPECT_EQ(gs.shot_events[0].to.first, 2);
    EXPECT_EQ(gs.shot_events[0].to.second, 3);
}

TEST(GameState, FromJson_ParsesMultiRowBoard) {
    json j = R"({
        "board": [
            [{"owner": 1, "status": 0, "wood": 0, "fortified": 0, "soldiers": 0, "p2_soldiers": 0}],
            [{"owner": 2, "status": 0, "wood": 0, "fortified": 0, "soldiers": 0, "p2_soldiers": 0}]
        ],
        "p_res": [0, 0],
        "turn": 0
    })"_json;
    
    auto gs = GameState::from_json(j);
    EXPECT_EQ(gs.board.size(), 2);
    EXPECT_EQ(gs.board[0][0].owner, 1);
    EXPECT_EQ(gs.board[1][0].owner, 2);
}

TEST(GameState, FromJson_ParsesWood) {
    json j = R"({
        "board": [
            [{"owner": 0, "status": 0, "wood": 10, "fortified": 0, "soldiers": 0, "p2_soldiers": 0}]
        ],
        "p_res": [0, 0],
        "turn": 0
    })"_json;
    
    auto gs = GameState::from_json(j);
    EXPECT_EQ(gs.board[0][0].wood, 10);
}

TEST(GameState, FromJson_ParsesFortified) {
    json j = R"({
        "board": [
            [{"owner": 1, "status": 0, "wood": 0, "fortified": 2, "soldiers": 0, "p2_soldiers": 0}]
        ],
        "p_res": [0, 0],
        "turn": 0
    })"_json;
    
    auto gs = GameState::from_json(j);
    EXPECT_EQ(gs.board[0][0].fortified, 2);
}

TEST(GameState, FromJson_DefaultValues) {
    json j = R"({
        "board": [
            [{"owner": 0, "status": 0, "wood": 0, "fortified": 0, "soldiers": 0, "p2_soldiers": 0}]
        ],
        "p_res": [0, 0]
    })"_json;
    
    auto gs = GameState::from_json(j);
    EXPECT_EQ(gs.turn, 0);
    EXPECT_FALSE(gs.target_tile.has_value());
    EXPECT_FALSE(gs.soldier_target_tile.has_value());
    EXPECT_TRUE(gs.shot_events.empty());
}

TEST(GameState, FromJson_ParsesWaterTile) {
    json j = R"({
        "board": [
            [{"owner": 0, "status": 8, "wood": 0, "fortified": 0, "soldiers": 0, "p2_soldiers": 0}]
        ],
        "p_res": [0, 0],
        "turn": 0
    })"_json;
    
    auto gs = GameState::from_json(j);
    EXPECT_EQ(gs.board[0][0].status, TileStatus::Water);
}
