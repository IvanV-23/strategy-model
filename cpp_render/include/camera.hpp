#pragma once

struct Point3D {
    float x, y, z;
};

struct Point2D {
    int x, y;
};

class IsoCamera {
public:
    int screenWidth, screenHeight;
    float zoom = 1.0f;
    int offsetX = 0, offsetY = 0;
    float tileW = 64.0f;
    float tileH = 32.0f;

    IsoCamera(int w, int h);

    Point2D project(float x, float y, float z = 0.0f) const;
};
