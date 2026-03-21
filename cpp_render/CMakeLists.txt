cmake_minimum_required(VERSION 3.10)
project(StrategyRenderer)

# --- Dependencies via vcpkg ---
find_package(SDL2 CONFIG REQUIRED)
find_package(SDL2_ttf CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

# --- Include Directories ---
include_directories(include)

# --- Source Files ---
set(SOURCES
    src/main.cpp
    src/camera.cpp
    src/graphics_utils.cpp
    src/network.cpp
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
