#include <gtest/gtest.h>
#include "camera.hpp"

TEST(IsoCamera, ProjectOrigin) {
    IsoCamera cam(1280, 720);
    auto p = cam.project(0.0f, 0.0f, 0.0f);
    // offsetX = w / 2 = 1280 / 2 = 640
    EXPECT_EQ(p.x, 640);  
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
