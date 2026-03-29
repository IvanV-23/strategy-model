#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build_python"

echo "Building strategy_engine Python module..."
echo "pybind11 must be installed: pip install pybind11"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake .. -DCMAKE_BUILD_TYPE=Release \
    -Dpybind11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")

cmake --build . --config Release

echo ""
echo "Build complete!"
echo "Module: ${BUILD_DIR}/strategy_engine*.so"
echo ""
echo "To test, copy to enviroment directory:"
echo "  cp strategy_engine*.so ../enviroment/"
