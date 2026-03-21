#include "renderers/tower_renderer.hpp"

void TowerRenderer::render(SDL_Renderer* renderer, const IsoCamera& cam, const json& board, int r, int c) {
    const auto& tile = board[r][c];
    int status = tile["status"];
    if (status != 3) return; // Only render on "Base" tiles (status==3)

    draw_tower(renderer, cam, (float)c, (float)r);
}

void TowerRenderer::draw_tower(SDL_Renderer* renderer, const IsoCamera& cam, float cx, float cy) {
    float z = cam.zoom;

    // ---- Color Palette (stone-like) ----
    SDL_Color stone_light = {190, 185, 175, 255};  // top face
    SDL_Color stone_mid   = {155, 150, 140, 255};  // front face (auto-darkened by draw_iso_box)
    SDL_Color stone_dark  = {110, 105,  95, 255};  // dark side
    SDL_Color merlon_col  = {170, 165, 155, 255};  // battlements
    SDL_Color door_col    = { 45,  35,  25, 255};  // dark doorway
    SDL_Color shadow_col  = { 80,  75,  68, 255};  // buttress shadow

    float tW = cam.tileW;

    // =============================
    // 1. MAIN TOWER BODY
    // =============================
    float tower_x  = cx + 0.15f;
    float tower_y  = cy + 0.15f;
    float tower_w  = 0.70f;
    float tower_d  = 0.70f;
    float tower_h  = 1.8f * tW;   // tall tower

    Graphics::draw_iso_box(renderer, cam,
        tower_x, tower_y, tower_w, tower_d, tower_h, stone_mid, 0.0f);

    // =============================
    // 2. BUTTRESSES (side supports)
    // Two on the front-left face, two on front-right
    // =============================
    float butt_w = 0.10f, butt_d = 0.10f, butt_h = tower_h * 0.55f;

    // Left face buttresses
    Graphics::draw_iso_box(renderer, cam,
        tower_x - butt_w, tower_y + 0.15f,
        butt_w, butt_d, butt_h, shadow_col, 0.0f);
    Graphics::draw_iso_box(renderer, cam,
        tower_x - butt_w, tower_y + 0.45f,
        butt_w, butt_d, butt_h, shadow_col, 0.0f);

    // Right face buttresses
    Graphics::draw_iso_box(renderer, cam,
        tower_x + tower_w, tower_y + 0.15f,
        butt_w, butt_d, butt_h, shadow_col, 0.0f);
    Graphics::draw_iso_box(renderer, cam,
        tower_x + tower_w, tower_y + 0.45f,
        butt_w, butt_d, butt_h, shadow_col, 0.0f);

    // =============================
    // 3. DOOR RECESS
    // A dark thin box slightly inset on the front face
    // =============================
    float door_w = 0.14f, door_d = 0.04f, door_h = 0.22f * tW;
    Graphics::draw_iso_box(renderer, cam,
        tower_x + (tower_w - door_w) * 0.5f,
        tower_y - door_d,           // slightly in front
        door_w, door_d, door_h,
        door_col, 0.0f);

    // Door frame (slightly wider, lighter)
    SDL_Color frame_col = {140, 130, 115, 255};
    Graphics::draw_iso_box(renderer, cam,
        tower_x + (tower_w - door_w) * 0.5f - 0.02f,
        tower_y - door_d * 0.5f,
        door_w + 0.04f, door_d * 0.5f, door_h + 0.03f * tW,
        frame_col, 0.0f);

    // =============================
    // 4. PARAPET / WALL WALK
    // Slightly wider slab just below the battlements
    // =============================
    float parapet_h = 0.08f * tW;
    Graphics::draw_iso_box(renderer, cam,
        tower_x - 0.04f, tower_y - 0.04f,
        tower_w + 0.08f, tower_d + 0.08f,
        parapet_h, stone_light, tower_h);

    // =============================
    // 5. BATTLEMENTS (Merlons)
    // Row of square teeth around the top
    // =============================
    float merlon_w = 0.10f, merlon_d = 0.10f, merlon_h = 0.14f * tW;
    float merlon_gap = 0.16f;  // spacing between merlons
    float merlon_z = tower_h + parapet_h;

    // Front-left edge merlons (along X axis)
    for (float mx = tower_x - 0.04f; mx < tower_x + tower_w; mx += merlon_gap) {
        Graphics::draw_iso_box(renderer, cam,
            mx, tower_y - 0.04f,
            merlon_w, merlon_d, merlon_h,
            merlon_col, merlon_z);
    }
    // Front-right edge merlons (along Y axis)
    for (float my = tower_y - 0.04f; my < tower_y + tower_d; my += merlon_gap) {
        Graphics::draw_iso_box(renderer, cam,
            tower_x + tower_w - 0.04f, my,
            merlon_d, merlon_w, merlon_h,
            merlon_col, merlon_z);
    }

    // =============================
    // 6. STONE BLOCK TEXTURE LINES
    // Horizontal lines across both visible faces
    // =============================
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 35);
    SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
    int num_courses = 7;
    for (int i = 1; i <= num_courses; ++i) {
        float stone_z = tower_h * ((float)i / (num_courses + 1));

        // Left face stone line
        Point2D sl1 = cam.project(tower_x,           tower_y,           stone_z);
        Point2D sl2 = cam.project(tower_x,           tower_y + tower_d, stone_z);
        SDL_RenderDrawLine(renderer, sl1.x, sl1.y, sl2.x, sl2.y);

        // Right face stone line
        Point2D sr1 = cam.project(tower_x + tower_w, tower_y,           stone_z);
        Point2D sr2 = cam.project(tower_x + tower_w, tower_y + tower_d, stone_z);
        SDL_RenderDrawLine(renderer, sr1.x, sr1.y, sr2.x, sr2.y);
    }

    // =============================
    // 7. FLAG / BANNER on top
    // =============================
    float flag_z = merlon_z + merlon_h;
    Point2D flag_base = cam.project(tower_x + tower_w * 0.5f, tower_y + tower_d * 0.5f, flag_z);

    // Pole
    SDL_SetRenderDrawColor(renderer, 200, 200, 200, 255);
    SDL_RenderDrawLine(renderer,
        flag_base.x, flag_base.y,
        flag_base.x, flag_base.y - (int)(22 * z));

    // Banner (team color — gold for base)
    SDL_Color banner = {255, 200, 0, 255};
    SDL_Rect b1 = { flag_base.x, flag_base.y - (int)(22*z), (int)(10*z), (int)(7*z) };
    SDL_SetRenderDrawColor(renderer, banner.r, banner.g, banner.b, 255);
    SDL_RenderFillRect(renderer, &b1);
    SDL_SetRenderDrawColor(renderer, 200, 150, 0, 255);
    SDL_RenderDrawRect(renderer, &b1);
}
