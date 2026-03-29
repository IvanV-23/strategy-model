#pragma once

#include <vector>
#include <optional>
#include <utility>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

enum class TileType {
    Normal = 0,
    Water = 1
};

enum class TileStatus {
    Empty = 0,
    Sawmill = 1,
    Warehouse = 2,
    Base = 3,
    Sawmill_P2 = 4,
    Crops = 5,
    Warehouse_P2 = 6,
    Crops_P2 = 7
};

struct Tile {
    int owner = 0;
    TileStatus status = TileStatus::Empty;
    TileType tile_type = TileType::Normal;
    int wood = 0;
    int fortified = 0;
    int soldiers = 0;
    int p2_soldiers = 0;
};

struct Resources {
    double gold = 0.0;
    double wood = 0.0;
};

struct ShotEvent {
    std::pair<int, int> from;
    std::pair<int, int> to;
};

struct GameState {
    std::vector<std::vector<Tile>> board;
    Resources p_res;
    int turn = 0;
    std::optional<std::pair<int, int>> target_tile;
    std::optional<std::pair<int, int>> soldier_target_tile;
    std::vector<ShotEvent> shot_events;

    static GameState from_json(const json& j) {
        GameState gs;
        
        if (j.contains("board") && j["board"].is_array()) {
            gs.board.reserve(j["board"].size());
            for (const auto& row : j["board"]) {
                std::vector<Tile> tile_row;
                tile_row.reserve(row.size());
                for (const auto& tile_json : row) {
                    Tile tile;
                    tile.owner = tile_json.value("owner", 0);
                    tile.status = static_cast<TileStatus>(tile_json.value("status", 0));
                    tile.tile_type = static_cast<TileType>(tile_json.value("tile_type", 0));
                    tile.wood = tile_json.value("wood", 0);
                    tile.fortified = tile_json.value("fortified", 0);
                    tile.soldiers = tile_json.value("soldiers", 0);
                    tile.p2_soldiers = tile_json.value("p2_soldiers", 0);
                    tile_row.push_back(tile);
                }
                gs.board.push_back(std::move(tile_row));
            }
        }

        if (j.contains("p_res") && j["p_res"].is_array()) {
            gs.p_res.gold = j["p_res"][0].get<double>();
            gs.p_res.wood = j["p_res"][1].get<double>();
        }

        gs.turn = j.value("turn", 0);

        if (j.contains("target_tile") && j["target_tile"].is_array() && j["target_tile"].size() == 2) {
            gs.target_tile = {j["target_tile"][0], j["target_tile"][1]};
        }

        if (j.contains("soldier_target_tile") && j["soldier_target_tile"].is_array() && j["soldier_target_tile"].size() == 2) {
            gs.soldier_target_tile = {j["soldier_target_tile"][0], j["soldier_target_tile"][1]};
        }

        if (j.contains("shot_events") && j["shot_events"].is_array()) {
            for (const auto& shot : j["shot_events"]) {
                if (shot.contains("from") && shot.contains("to") &&
                    shot["from"].is_array() && shot["to"].is_array() &&
                    shot["from"].size() == 2 && shot["to"].size() == 2) {
                    ShotEvent se;
                    se.from = {shot["from"][0], shot["from"][1]};
                    se.to = {shot["to"][0], shot["to"][1]};
                    gs.shot_events.push_back(se);
                }
            }
        }

        return gs;
    }
};
