cmake_minimum_required(VERSION 3.10)
project(StrategyRenderer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Dependencies via vcpkg ---
find_package(SDL2 CONFIG REQUIRED)
find_package(SDL2_ttf CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# --- Include Directories ---
include_directories(include)

# --- Source Files ---
set(SOURCES
    src/main.cpp
    src/camera.cpp
    src/graphics_utils.cpp
    src/network.cpp
    src/board_factory.cpp
    src/resource_system.cpp
    src/build_system.cpp
    src/combat_system.cpp
    src/turn_manager.cpp
    src/renderers/tile_renderer.cpp
    src/renderers/building_renderer.cpp
    src/renderers/tower_renderer.cpp
    src/renderers/warehouse_renderer.cpp
    src/renderers/crop_renderer.cpp
    src/renderers/sawmill_renderer.cpp
    src/renderers/tree_renderer.cpp
    src/renderers/wall_renderer.cpp
    src/renderers/unit_renderer.cpp
)

# --- Executable ---
add_executable(strategy_renderer ${SOURCES})

# --- Linking ---
target_link_libraries(strategy_renderer PRIVATE 
    SDL2::SDL2 
    SDL2::SDL2main
    SDL2_ttf::SDL2_ttf 
    nlohmann_json::nlohmann_json
)

if(WIN32)
    target_link_libraries(strategy_renderer PRIVATE ws2_32)
endif()

# --- Python Bindings (pybind11 module) ---
set(PYTHON_BINDINGS_SOURCES
    src/python_bindings.cpp
    src/board_factory.cpp
    src/resource_system.cpp
    src/build_system.cpp
    src/combat_system.cpp
    src/turn_manager.cpp
)

pybind11_add_module(strategy_engine ${PYTHON_BINDINGS_SOURCES})
target_link_libraries(strategy_engine PRIVATE nlohmann_json::nlohmann_json)

# --- Testing ---
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/main.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(test_camera tests/test_camera.cpp src/camera.cpp)
target_include_directories(test_camera PRIVATE include)
target_link_libraries(test_camera GTest::gtest GTest::gtest_main)

add_test(NAME CameraTest COMMAND test_camera)

add_executable(test_game_state tests/test_game_state.cpp)
target_include_directories(test_game_state PRIVATE include)
target_link_libraries(test_game_state GTest::gtest GTest::gtest_main nlohmann_json::nlohmann_json)

add_test(NAME GameStateTest COMMAND test_game_state)

add_executable(test_board_factory tests/test_board_factory.cpp src/board_factory.cpp)
target_include_directories(test_board_factory PRIVATE include)
target_link_libraries(test_board_factory GTest::gtest GTest::gtest_main nlohmann_json::nlohmann_json)

add_test(NAME BoardFactoryTest COMMAND test_board_factory)

add_executable(test_resource_system tests/test_resource_system.cpp src/resource_system.cpp src/board_factory.cpp)
target_include_directories(test_resource_system PRIVATE include)
target_link_libraries(test_resource_system GTest::gtest GTest::gtest_main nlohmann_json::nlohmann_json)

add_test(NAME ResourceSystemTest COMMAND test_resource_system)

add_executable(test_build_system tests/test_build_system.cpp src/build_system.cpp src/resource_system.cpp src/board_factory.cpp)
target_include_directories(test_build_system PRIVATE include)
target_link_libraries(test_build_system GTest::gtest GTest::gtest_main nlohmann_json::nlohmann_json)

add_test(NAME BuildSystemTest COMMAND test_build_system)

add_executable(test_combat_system tests/test_combat_system.cpp src/combat_system.cpp src/board_factory.cpp)
target_include_directories(test_combat_system PRIVATE include)
target_link_libraries(test_combat_system GTest::gtest GTest::gtest_main nlohmann_json::nlohmann_json)

add_test(NAME CombatSystemTest COMMAND test_combat_system)

add_executable(test_turn_manager tests/test_turn_manager.cpp src/turn_manager.cpp src/resource_system.cpp src/build_system.cpp src/combat_system.cpp src/board_factory.cpp)
target_include_directories(test_turn_manager PRIVATE include)
target_link_libraries(test_turn_manager GTest::gtest GTest::gtest_main nlohmann_json::nlohmann_json)

add_test(NAME TurnManagerTest COMMAND test_turn_manager)
