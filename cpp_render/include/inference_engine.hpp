#pragma once

#include <torch/script.h>
#include <torch/torch.h>

#include "game_state.hpp"
#include "turn_manager.hpp"

struct CppActionBundle {
    int diplomacy = 1;
    std::vector<int> economy = {0, 0, 0, 0, 0, 0};
    int target_tile = 0;
    int soldier_target = 0;
    int fortify_tile = 0;
};

struct ObservationMasks {
    torch::Tensor action_mask;
    torch::Tensor build_mask;
    torch::Tensor fortify_mask;
};

struct ObservationTensors {
    torch::Tensor board;
    torch::Tensor stats;
    ObservationMasks masks;
};

class InferenceEngine {
public:
    bool loaded = false;
    bool has_soldier_model = false;

    void load(const std::string& manager_path, const std::string& soldier_path = "");
    bool is_loaded() const { return loaded; }
    bool has_soldier() const { return has_soldier_model; }

    CppActionBundle run_manager(const torch::Tensor& board,
                               const torch::Tensor& stats,
                               const torch::Tensor& action_mask,
                               const torch::Tensor& build_mask,
                               const torch::Tensor& fortify_mask);

    int run_soldier(const torch::Tensor& board,
                    const torch::Tensor& goal,
                    const torch::Tensor& mask);

    CppActionBundle select_action(const ObservationTensors& obs);

private:
    torch::jit::script::Module manager;
    torch::jit::script::Module soldier;
    torch::Device device = torch::kCPU;
};

class ObservationBuilder {
public:
    static ObservationTensors build(const GameState& state);
    static torch::Tensor build_board_tensor(const GameState& state);
    static torch::Tensor build_stats_tensor(const GameState& state);
    static ObservationMasks build_masks(const GameState& state, int player_id = 1);

private:
    static int count_tiles_by_status(const GameState& state, int player_id, TileStatus status);
    static int count_soldiers(const GameState& state, int player_id);
};
