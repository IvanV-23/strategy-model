#include "camera.hpp"
#include <cmath>

IsoCamera::IsoCamera(int w, int h) : screenWidth(w), screenHeight(h) {
    offsetX = w / 2;
    offsetY = h / 4;
}

Point2D IsoCamera::project(float x, float y, float z) const {
    int sx = static_cast<int>((x - y) * (tileW / 2.0f) * zoom) + offsetX;
    int sy = static_cast<int>((x + y) * (tileH / 2.0f) * zoom) - static_cast<int>(z * zoom) + offsetY;
    return {sx, sy};
}
