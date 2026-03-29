#include "inference_engine.hpp"
#include <iostream>

void InferenceEngine::load(const std::string& manager_path, const std::string& soldier_path) {
    try {
        std::cout << "Loading manager model: " << manager_path << std::endl;
        manager = torch::jit::load(manager_path);
        manager.eval();
        loaded = true;
        std::cout << "Manager model loaded successfully" << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "Error loading manager model: " << e.what() << std::endl;
        loaded = false;
        return;
    }

    if (!soldier_path.empty()) {
        try {
            std::cout << "Loading soldier model: " << soldier_path << std::endl;
            soldier = torch::jit::load(soldier_path);
            soldier.eval();
            has_soldier_model = true;
            std::cout << "Soldier model loaded successfully" << std::endl;
        } catch (const c10::Error& e) {
            std::cerr << "Warning: Could not load soldier model: " << e.what() << std::endl;
            has_soldier_model = false;
        }
    }
}

CppActionBundle InferenceEngine::run_manager(const torch::Tensor& board,
                                           const torch::Tensor& stats,
                                           const torch::Tensor& action_mask,
                                           const torch::Tensor& build_mask,
                                           const torch::Tensor& fortify_mask) {
    CppActionBundle bundle;

    if (!loaded) {
        std::cerr << "Manager model not loaded!" << std::endl;
        return bundle;
    }

    try {
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(board);
        inputs.push_back(stats);
        inputs.push_back(action_mask);
        inputs.push_back(build_mask);
        inputs.push_back(fortify_mask);

        at::Dict<at::Tensor, at::Tensor> output = 
            manager.forward(inputs).toGenericDict();

        auto dip_out = output.at("dip").toTensor();
        auto mil_out = output.at("mil").toTensor();
        auto mil_soldier_out = output.at("mil_soldier").toTensor();
        auto mil_fortify_out = output.at("mil_fortify").toTensor();
        auto eco_out = output.at("eco").toList();

        bundle.diplomacy = dip_out.argmax(1).item<int>();
        bundle.target_tile = mil_out.argmax(1).item<int>();
        bundle.soldier_target = mil_soldier_out.argmax(1).item<int>();
        bundle.fortify_tile = mil_fortify_out.argmax(1).item<int>();

        bundle.economy.clear();
        for (size_t i = 0; i < eco_out.size(); ++i) {
            auto eco_tensor = eco_out.get(i).toTensor();
            bundle.economy.push_back(eco_tensor.argmax(1).item<int>());
        }

    } catch (const c10::Error& e) {
        std::cerr << "Error running manager inference: " << e.what() << std::endl;
    }

    return bundle;
}

int InferenceEngine::run_soldier(const torch::Tensor& board,
                                 const torch::Tensor& goal,
                                 const torch::Tensor& mask) {
    if (!has_soldier_model) {
        return torch::randint(0, 256, {1}).item<int>();
    }

    try {
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(board);
        inputs.push_back(goal);
        inputs.push_back(mask);

        auto output = soldier.forward(inputs).toTensor();
        return output.argmax(1).item<int>();
    } catch (const c10::Error& e) {
        std::cerr << "Error running soldier inference: " << e.what() << std::endl;
        return torch::randint(0, 256, {1}).item<int>();
    }
}

CppActionBundle InferenceEngine::select_action(const ObservationTensors& obs) {
    return run_manager(obs.board, obs.stats, 
                       obs.masks.action_mask, obs.masks.build_mask, obs.masks.fortify_mask);
}

ObservationTensors ObservationBuilder::build(const GameState& state) {
    ObservationTensors obs;
    obs.board = build_board_tensor(state);
    obs.stats = build_stats_tensor(state);
    obs.masks = build_masks(state);
    return obs;
}

torch::Tensor ObservationBuilder::build_board_tensor(const GameState& state) {
    auto options = torch::dtype(torch::kFloat32);
    auto tensor = torch::zeros({1, 8, 16, 16}, options);

    for (int r = 0; r < (int)state.board.size(); ++r) {
        for (int c = 0; c < (int)state.board[r].size(); ++c) {
            const auto& tile = state.board[r][c];
            tensor[0][0][r][c] = static_cast<float>(tile.owner);
            tensor[0][1][r][c] = static_cast<float>(tile.status);
            tensor[0][2][r][c] = static_cast<float>(tile.wood);
            tensor[0][3][r][c] = static_cast<float>(tile.fortified);
            tensor[0][4][r][c] = static_cast<float>(tile.soldiers);
            tensor[0][5][r][c] = static_cast<float>(tile.p2_soldiers);
        }
    }

    if (state.target_tile.has_value()) {
        auto [tr, tc] = state.target_tile.value();
        if (tr >= 0 && tr < 16 && tc >= 0 && tc < 16) {
            tensor[0][6][tr][tc] = 1.0f;
        }
    }

    if (state.soldier_target_tile.has_value()) {
        auto [sr, sc] = state.soldier_target_tile.value();
        if (sr >= 0 && sr < 16 && sc >= 0 && sc < 16) {
            tensor[0][7][sr][sc] = 1.0f;
        }
    }

    return tensor;
}

torch::Tensor ObservationBuilder::build_stats_tensor(const GameState& state) {
    auto options = torch::dtype(torch::kFloat32);
    auto tensor = torch::zeros({1, 28}, options);

    tensor[0][0] = static_cast<float>(state.p_res.gold);
    tensor[0][1] = static_cast<float>(state.p_res.wood);
    tensor[0][2] = static_cast<float>(state.turn);

    int p1_sawmills = count_tiles_by_status(state, 1, TileStatus::Sawmill);
    int p1_warehouses = count_tiles_by_status(state, 1, TileStatus::Warehouse);
    int p1_crops = count_tiles_by_status(state, 1, TileStatus::Crops);
    int p2_sawmills = count_tiles_by_status(state, 2, TileStatus::Sawmill_P2);
    int p2_warehouses = count_tiles_by_status(state, 2, TileStatus::Warehouse_P2);
    int p2_crops = count_tiles_by_status(state, 2, TileStatus::Crops_P2);

    tensor[0][3] = static_cast<float>(p1_sawmills);
    tensor[0][4] = static_cast<float>(p1_warehouses);
    tensor[0][5] = static_cast<float>(p1_crops);
    tensor[0][6] = static_cast<float>(p2_sawmills);
    tensor[0][7] = static_cast<float>(p2_warehouses);
    tensor[0][8] = static_cast<float>(p2_crops);

    int p1_soldiers = count_soldiers(state, 1);
    int p2_soldiers = count_soldiers(state, 2);
    tensor[0][9] = static_cast<float>(p1_soldiers);
    tensor[0][10] = static_cast<float>(p2_soldiers);

    return tensor;
}

ObservationMasks ObservationBuilder::build_masks(const GameState& state, int player_id) {
    ObservationMasks masks;
    
    auto bool_options = torch::dtype(torch::kBool);
    masks.action_mask = torch::ones({1, 256}, bool_options);
    masks.build_mask = torch::ones({1, 8}, bool_options);
    masks.fortify_mask = torch::ones({1, 256}, bool_options);

    for (int r = 0; r < (int)state.board.size(); ++r) {
        for (int c = 0; c < (int)state.board[r].size(); ++c) {
            const auto& tile = state.board[r][c];
            int idx = r * 16 + c;

            if (tile.owner == player_id) {
                masks.action_mask[0][idx] = false;
            }

            if (tile.owner != 0 && tile.owner != player_id) {
                masks.fortify_mask[0][idx] = false;
            }
        }
    }

    double gold = state.p_res.gold;
    double wood = state.p_res.wood;
    masks.build_mask[0][0] = (gold >= 100 && wood >= 50);
    masks.build_mask[0][1] = (gold >= 60 && wood >= 40);
    masks.build_mask[0][2] = (gold >= 40 && wood >= 20);
    masks.build_mask[0][3] = (gold >= 30 && wood >= 15);

    return masks;
}

int ObservationBuilder::count_tiles_by_status(const GameState& state, int player_id, TileStatus status) {
    int count = 0;
    for (const auto& row : state.board) {
        for (const auto& tile : row) {
            if (tile.owner == player_id && tile.status == status) {
                ++count;
            }
        }
    }
    return count;
}

int ObservationBuilder::count_soldiers(const GameState& state, int player_id) {
    int count = 0;
    for (const auto& row : state.board) {
        for (const auto& tile : row) {
            if (tile.owner == player_id) {
                count += tile.soldiers;
            }
        }
    }
    return count;
}
