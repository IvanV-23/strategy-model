#include "graphics_utils.hpp"
#include <cmath>

namespace Graphics {
    void draw_iso_diamond(SDL_Renderer* renderer, const IsoCamera& cam, float x, float y, float size, SDL_Color color, SDL_Color border_color) {
        Point2D p1 = cam.project(x, y);
        Point2D p2 = cam.project(x + size, y);
        Point2D p3 = cam.project(x + size, y + size);
        Point2D p4 = cam.project(x, y + size);

        SDL_Vertex vertices[4];
        for (int i = 0; i < 4; ++i) vertices[i].color = color;
        
        vertices[0].position = {(float)p1.x, (float)p1.y};
        vertices[1].position = {(float)p2.x, (float)p2.y};
        vertices[2].position = {(float)p3.x, (float)p3.y};
        vertices[3].position = {(float)p4.x, (float)p4.y};

        int indices[] = {0, 1, 2, 0, 2, 3};
        SDL_RenderGeometry(renderer, nullptr, vertices, 4, indices, 6);
        
        // Border
        SDL_SetRenderDrawColor(renderer, border_color.r, border_color.g, border_color.b, border_color.a);
        SDL_RenderDrawLine(renderer, p1.x, p1.y, p2.x, p2.y);
        SDL_RenderDrawLine(renderer, p2.x, p2.y, p3.x, p3.y);
        SDL_RenderDrawLine(renderer, p3.x, p3.y, p4.x, p4.y);
        SDL_RenderDrawLine(renderer, p4.x, p4.y, p1.x, p1.y);
        
        // If it's a player border, make it a bit thicker/more visible
        if (border_color.a > 100) {
            SDL_RenderDrawLine(renderer, p1.x, p1.y + 1, p2.x, p2.y + 1);
            SDL_RenderDrawLine(renderer, p2.x, p2.y + 1, p3.x, p3.y + 1);
            SDL_RenderDrawLine(renderer, p3.x, p3.y + 1, p4.x, p4.y + 1);
            SDL_RenderDrawLine(renderer, p4.x, p4.y + 1, p1.x, p1.y + 1);
        }
    }

    void draw_iso_box(SDL_Renderer* renderer, const IsoCamera& cam, float x, float y, float w, float d, float h, SDL_Color color, float z_start) {
        // Bottom Face
        Point2D b1 = cam.project(x, y, z_start);
        Point2D b2 = cam.project(x + w, y, z_start);
        Point2D b3 = cam.project(x + w, y + d, z_start);
        Point2D b4 = cam.project(x, y + d, z_start);

        // Top Face
        Point2D t1 = cam.project(x, y, z_start + h);
        Point2D t2 = cam.project(x + w, y, z_start + h);
        Point2D t3 = cam.project(x + w, y + d, z_start + h);
        Point2D t4 = cam.project(x, y + d, z_start + h);

        auto draw_quad = [&](Point2D p1, Point2D p2, Point2D p3, Point2D p4, SDL_Color col) {
            SDL_Vertex v[4];
            for (int i = 0; i < 4; ++i) v[i].color = col;
            v[0].position = {(float)p1.x, (float)p1.y};
            v[1].position = {(float)p2.x, (float)p2.y};
            v[2].position = {(float)p3.x, (float)p3.y};
            v[3].position = {(float)p4.x, (float)p4.y};
            int ind[] = {0, 1, 2, 0, 2, 3};
            SDL_RenderGeometry(renderer, nullptr, v, 4, ind, 6);
        };

        SDL_Color side1 = { (Uint8)(color.r * 0.8), (Uint8)(color.g * 0.8), (Uint8)(color.b * 0.8), color.a };
        SDL_Color side2 = { (Uint8)(color.r * 0.6), (Uint8)(color.g * 0.6), (Uint8)(color.b * 0.6), color.a };

        draw_quad(b1, b2, t2, t1, side1); // Side 1
        draw_quad(b2, b3, t3, t2, side2); // Side 2
        draw_quad(t1, t2, t3, t4, color); // Top

        // Outlines
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 50);
        SDL_RenderDrawLine(renderer, t1.x, t1.y, t2.x, t2.y);
        SDL_RenderDrawLine(renderer, t2.x, t2.y, t3.x, t3.y);
        SDL_RenderDrawLine(renderer, t3.x, t3.y, t4.x, t4.y);
        SDL_RenderDrawLine(renderer, t4.x, t4.y, t1.x, t1.y);
        SDL_RenderDrawLine(renderer, b1.x, b1.y, t1.x, t1.y);
        SDL_RenderDrawLine(renderer, b2.x, b2.y, t2.x, t2.y);
        SDL_RenderDrawLine(renderer, b3.x, b3.y, t3.x, t3.y);
    }

    void draw_iso_box_stacked(SDL_Renderer* renderer, const IsoCamera& cam, float x, float y, float w, float d, float h, SDL_Color color, float z_start) {
        draw_iso_box(renderer, cam, x, y, w, d, h, color, z_start); 
    }

    void draw_complete_building(SDL_Renderer* renderer, const IsoCamera& cam, float x, float y, float w, float d, float wallH, float roofH, SDL_Color wallCol, SDL_Color roofCol) {
        draw_iso_box(renderer, cam, x, y, w, d, wallH, wallCol, 0.0f);
        float midW = w / 2.0f;
        draw_iso_box(renderer, cam, x - 0.05f, y - 0.05f, midW + 0.05f, d + 0.1f, roofH, roofCol, wallH);
        draw_iso_box(renderer, cam, x + midW, y - 0.05f, midW + 0.05f, d + 0.1f, roofH, roofCol, wallH);
    }

    void fill_circle(SDL_Renderer* renderer, int x, int y, int radius) {
        for (int w = 0; w < radius * 2; w++) {
            for (int h = 0; h < radius * 2; h++) {
                int dx = radius - w;
                int dy = radius - h;
                if ((dx*dx + dy*dy) <= (radius * radius)) {
                    SDL_RenderDrawPoint(renderer, x + dx, y + dy);
                }
            }
        }
    }
}
