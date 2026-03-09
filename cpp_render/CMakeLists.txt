cmake_minimum_required(VERSION 3.10)
project(StrategyRenderer)

# --- Dependencies via vcpkg ---
find_package(SDL2 CONFIG REQUIRED)
find_package(SDL2_ttf CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

# --- Executable ---
add_executable(strategy_renderer main.cpp)

# --- Linking ---
# Note: target_link_libraries uses the names provided by the CONFIG files
target_link_libraries(strategy_renderer PRIVATE 
    SDL2::SDL2 
    SDL2::SDL2main
    SDL2_ttf::SDL2_ttf 
    nlohmann_json::nlohmann_json
)

if(WIN32)
    target_link_libraries(strategy_renderer PRIVATE ws2_32)
endif()
