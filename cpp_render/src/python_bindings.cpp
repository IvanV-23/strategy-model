#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "game_state.hpp"
#include "turn_manager.hpp"
#include "board_factory.hpp"
#include "resource_system.hpp"
#include "build_system.hpp"
#include "combat_system.hpp"

namespace py = pybind11;

PYBIND11_MODULE(strategy_engine, m) {
    m.doc() = "Strategy game engine Python bindings";

    py::enum_<TileStatus>(m, "TileStatus")
        .value("Empty", TileStatus::Empty)
        .value("Sawmill", TileStatus::Sawmill)
        .value("Warehouse", TileStatus::Warehouse)
        .value("Base", TileStatus::Base)
        .value("Sawmill_P2", TileStatus::Sawmill_P2)
        .value("Crops", TileStatus::Crops)
        .value("Warehouse_P2", TileStatus::Warehouse_P2)
        .value("Crops_P2", TileStatus::Crops_P2)
        .value("Water", TileStatus::Water)
        .export_values();

    py::class_<Tile>(m, "Tile")
        .def(py::init<>())
        .def_readwrite("owner", &Tile::owner)
        .def_readwrite("status", &Tile::status)
        .def_readwrite("wood", &Tile::wood)
        .def_readwrite("fortified", &Tile::fortified)
        .def_readwrite("soldiers", &Tile::soldiers)
        .def_readwrite("p2_soldiers", &Tile::p2_soldiers);

    py::class_<Resources>(m, "Resources")
        .def(py::init<>())
        .def_readwrite("gold", &Resources::gold)
        .def_readwrite("wood", &Resources::wood);

    py::class_<ShotEvent>(m, "ShotEvent")
        .def(py::init<>())
        .def_readwrite("from_coord", &ShotEvent::from)
        .def_readwrite("to_coord", &ShotEvent::to);

    py::class_<GameState>(m, "GameState")
        .def(py::init<>())
        .def_readwrite("board", &GameState::board)
        .def_readwrite("p_res", &GameState::p_res)
        .def_readwrite("turn", &GameState::turn)
        .def_readwrite("target_tile", &GameState::target_tile)
        .def_readwrite("soldier_target_tile", &GameState::soldier_target_tile)
        .def_readwrite("shot_events", &GameState::shot_events)
        .def_static("from_json", &GameState::from_json);

    py::enum_<BuildAction>(m, "BuildAction")
        .value("None", BuildAction::None)
        .value("Sawmill", BuildAction::Sawmill)
        .value("Warehouse", BuildAction::Warehouse)
        .value("Crops", BuildAction::Crops)
        .value("Fortify", BuildAction::Fortify)
        .export_values();

    py::class_<ActionCost>(m, "ActionCost")
        .def(py::init<>())
        .def_readwrite("gold", &ActionCost::gold)
        .def_readwrite("wood", &ActionCost::wood);

    py::class_<ResourceSystem>(m, "ResourceSystem")
        .def(py::init<>())
        .def("tick", &ResourceSystem::tick)
        .def("apply_build_cost", &ResourceSystem::apply_build_cost)
        .def("can_afford", &ResourceSystem::can_afford)
        .def_static("get_cost", &ResourceSystem::get_cost);

    py::class_<BuildSystem>(m, "BuildSystem")
        .def(py::init<>())
        .def("apply_economy_action", &BuildSystem::apply_economy_action)
        .def("apply_fortify_action", &BuildSystem::apply_fortify_action)
        .def("get_build_mask", &BuildSystem::get_build_mask);

    py::class_<CombatSystem>(m, "CombatSystem")
        .def(py::init<>())
        .def("spawn_soldier", &CombatSystem::spawn_soldier)
        .def("move_soldier", &CombatSystem::move_soldier)
        .def("resolve_combat", &CombatSystem::resolve_combat)
        .def("get_shot_events", &CombatSystem::get_shot_events);

    py::class_<BoardFactory>(m, "BoardFactory")
        .def_static("create_default", &BoardFactory::create_default,
                    "Create a default 16x16 game board",
                    py::arg("rows"), py::arg("cols"),
                    py::arg("p1_base_r"), py::arg("p1_base_c"),
                    py::arg("p2_base_r"), py::arg("p2_base_c"));

    py::class_<TurnManager::ActionBundle>(m, "ActionBundle")
        .def(py::init<>())
        .def_readwrite("diplomacy", &TurnManager::ActionBundle::diplomacy)
        .def_readwrite("economy", &TurnManager::ActionBundle::economy)
        .def_readwrite("target_tile", &TurnManager::ActionBundle::target_tile)
        .def_readwrite("soldier_target", &TurnManager::ActionBundle::soldier_target)
        .def_readwrite("fortify_tile", &TurnManager::ActionBundle::fortify_tile);

    py::class_<TurnManager>(m, "TurnManager")
        .def(py::init<>())
        .def("step", &TurnManager::step,
             "Execute one game step",
             py::arg("state"), py::arg("p1_action"), py::arg("p2_action"))
        .def("is_terminal", &TurnManager::is_terminal,
             "Check if game has ended",
             py::arg("state"))
        .def("compute_reward", &TurnManager::compute_reward,
             "Compute reward for a player",
             py::arg("prev"), py::arg("next"), py::arg("player_id"));
}
