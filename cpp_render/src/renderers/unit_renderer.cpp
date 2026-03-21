#include "renderers/unit_renderer.hpp"
#include <cmath>

void UnitRenderer::render(SDL_Renderer* renderer, const IsoCamera& cam, const json& board, int r, int c) {
    const auto& tile = board[r][c];
    if (tile.contains("soldiers") && tile["soldiers"].get<int>() > 0) {
        draw_unit(renderer, cam, r, c, tile["soldiers"].get<int>(), {220, 50, 50, 255}, -0.3f);
    }
    if (tile.contains("p2_soldiers") && tile["p2_soldiers"].get<int>() > 0) {
        draw_unit(renderer, cam, r, c, tile["p2_soldiers"].get<int>(), {50, 120, 220, 255}, 0.3f);
    }
}

void UnitRenderer::draw_unit(SDL_Renderer* renderer, const IsoCamera& cam, int r, int c, int count, SDL_Color col, float offset) {
    float z = cam.zoom;

    // Animation phase: unique per tile position + offset side
    float phase = SDL_GetTicks() / 400.0f + (r * 7.3f + c * 3.7f + offset * 5.0f);
    float leg_swing  = sinf(phase) * 0.06f;   // leg bob in world units
    float arm_swing  = -sinf(phase) * 0.05f;  // opposite to legs
    float body_bob   = fabsf(sinf(phase)) * 2.0f * z; // vertical bob in pixels

    // Place 1 or 2 figures depending on count
    int num_figures = (count < 2) ? count : 2;
    float fig_offsets[2] = { offset - 0.08f, offset + 0.08f };

    for (int f = 0; f < num_figures; ++f) {
        float fx = (float)c + 0.5f + fig_offsets[f];
        float fy = (float)r + 0.5f + fig_offsets[f] * 0.5f;

        // ---- Skin & clothing colors ----
        SDL_Color skin    = {220, 170, 120, 255};
        SDL_Color armor   = col;                                         // torso = team color
        SDL_Color dark    = {(Uint8)(col.r*0.6f), (Uint8)(col.g*0.6f), (Uint8)(col.b*0.6f), 255}; // legs
        SDL_Color helmet  = {(Uint8)(col.r*0.8f), (Uint8)(col.g*0.8f), (Uint8)(col.b*0.8f), 255};

        float S = 0.09f;   // base unit size in world coords
        float body_bob_z = body_bob; // pixel offset upward

        // Heights in world-z units (tileW scale):
        float boot_h   = 0.04f * cam.tileW;
        float leg_h    = 0.10f * cam.tileW;
        float torso_h  = 0.13f * cam.tileW;
        float head_h   = 0.09f * cam.tileW;
        float arm_h    = 0.10f * cam.tileW;

        float z0 = 0.0f;                         // ground
        float z1 = boot_h;                       // above boots
        float z2 = z1 + leg_h;                   // above legs (waist)
        float z3 = z2 + torso_h;                 // above torso (shoulder)
        float z4 = z3 + head_h;                  // top of head

        // --- Shadow ---
        Point2D ps = cam.project(fx, fy, 0.0f);
        SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 50);
        Graphics::fill_circle(renderer, ps.x, ps.y, (int)(8 * z));

        float bob_wu = body_bob_z / (cam.zoom * cam.tileW / 2.0f); // Very rough conversion, but using z pixels directly in draw_iso_box is better

        // Actually body_bob is in pixels. draw_iso_box uses z_start in world units then * cam.zoom.
        // Let's pass body_bob_z / cam.zoom as z_start
        float bz = body_bob_z / cam.zoom;

        // --- LEFT LEG (swings forward) ---
        SDL_Color boot_col = {40, 40, 40, 255};
        Graphics::draw_iso_box(renderer, cam,
            fx - S*1.1f + leg_swing, fy - S*0.5f,
            S, S*0.9f, boot_h, boot_col, bz + z0);
        Graphics::draw_iso_box(renderer, cam,
            fx - S*1.1f + leg_swing, fy - S*0.5f,
            S*0.9f, S*0.9f, leg_h, dark, bz + z1);

        // --- RIGHT LEG (swings backward) ---
        Graphics::draw_iso_box(renderer, cam,
            fx + S*0.1f - leg_swing, fy - S*0.5f,
            S, S*0.9f, boot_h, boot_col, bz + z0);
        Graphics::draw_iso_box(renderer, cam,
            fx + S*0.1f - leg_swing, fy - S*0.5f,
            S*0.9f, S*0.9f, leg_h, dark, bz + z1);

        // --- TORSO ---
        Graphics::draw_iso_box(renderer, cam,
            fx - S*1.1f, fy - S*0.5f,
            S*2.0f, S*0.9f, torso_h, armor, bz + z2);

        // --- LEFT ARM (swings opposite to left leg) ---
        Graphics::draw_iso_box(renderer, cam,
            fx - S*1.5f + arm_swing, fy - S*0.5f,
            S*0.6f, S*0.6f, arm_h, armor, bz + z2);

        // --- RIGHT ARM ---
        Graphics::draw_iso_box(renderer, cam,
            fx + S*0.85f - arm_swing, fy - S*0.5f,
            S*0.6f, S*0.6f, arm_h, armor, bz + z2);

        // --- HEAD ---
        Graphics::draw_iso_box(renderer, cam,
            fx - S*0.7f, fy - S*0.45f,
            S*1.4f, S*0.9f, head_h, skin, bz + z3);

        // --- HELMET (thin slab on top of head) ---
        Graphics::draw_iso_box(renderer, cam,
            fx - S*0.85f, fy - S*0.55f,
            S*1.6f, S*1.0f, head_h * 0.25f, helmet, bz + z3 + head_h);

        // --- SOLDIER COUNT badge (if > 2) ---
        if (count > 2 && f == 0) {
            Point2D pb = cam.project(fx, fy, z4 + head_h * 0.3f + bz);
            SDL_SetRenderDrawColor(renderer, 255, 255, 255, 200);
            Graphics::fill_circle(renderer, pb.x, pb.y - (int)(12*z), (int)(5*z));
        }
    }
}
