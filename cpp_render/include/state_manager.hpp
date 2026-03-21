#pragma once
#include <mutex>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class GameStateManager {
public:
    json data;
    std::mutex mtx;
    bool updated = false;

    void update(const json& j) {
        std::lock_guard<std::mutex> lock(mtx);
        data = j;
        updated = true;
    }

    json get_data() {
        std::lock_guard<std::mutex> lock(mtx);
        return data;
    }
};

extern GameStateManager global_state;
