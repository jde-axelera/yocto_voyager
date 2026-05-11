// drawing.cpp — implementation of yvm/drawing.h
#include "drawing.h"
#include "coco_palette.h"

#include <algorithm>
#include <cstring>

namespace yvm {

void put_px(const Image& im, int x, int y, uint8_t b, uint8_t g, uint8_t r) {
    if ((unsigned)x >= (unsigned)im.w || (unsigned)y >= (unsigned)im.h) return;
    uint8_t* p = im.data + (y * im.w + x) * 3;
    p[0] = b; p[1] = g; p[2] = r;
}

void draw_hline(const Image& im, int x1, int x2, int y,
                uint8_t b, uint8_t g, uint8_t r) {
    if (y < 0 || y >= im.h) return;
    if (x1 > x2) std::swap(x1, x2);
    x1 = std::max(0, x1); x2 = std::min(im.w - 1, x2);
    uint8_t* p = im.data + (y * im.w + x1) * 3;
    for (int x = x1; x <= x2; ++x, p += 3) { p[0] = b; p[1] = g; p[2] = r; }
}

void draw_vline(const Image& im, int x, int y1, int y2,
                uint8_t b, uint8_t g, uint8_t r) {
    if (x < 0 || x >= im.w) return;
    if (y1 > y2) std::swap(y1, y2);
    y1 = std::max(0, y1); y2 = std::min(im.h - 1, y2);
    for (int y = y1; y <= y2; ++y) {
        uint8_t* p = im.data + (y * im.w + x) * 3;
        p[0] = b; p[1] = g; p[2] = r;
    }
}

void draw_rect(const Image& im, int x1, int y1, int x2, int y2,
               uint8_t b, uint8_t g, uint8_t r, int thick) {
    for (int t = 0; t < thick; ++t) {
        draw_hline(im, x1, x2, y1 + t, b, g, r);
        draw_hline(im, x1, x2, y2 - t, b, g, r);
        draw_vline(im, x1 + t, y1, y2, b, g, r);
        draw_vline(im, x2 - t, y1, y2, b, g, r);
    }
}

void fill_rect(const Image& im, int x1, int y1, int x2, int y2,
               uint8_t b, uint8_t g, uint8_t r) {
    if (x1 > x2) std::swap(x1, x2);
    if (y1 > y2) std::swap(y1, y2);
    x1 = std::max(0, x1); y1 = std::max(0, y1);
    x2 = std::min(im.w - 1, x2); y2 = std::min(im.h - 1, y2);
    for (int y = y1; y <= y2; ++y) {
        uint8_t* p = im.data + (y * im.w + x1) * 3;
        for (int x = x1; x <= x2; ++x, p += 3) { p[0] = b; p[1] = g; p[2] = r; }
    }
}

void fill_rect_alpha(uint8_t* dst, int dw, int dh,
                     int x1, int y1, int x2, int y2,
                     uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    if (x1 > x2) std::swap(x1, x2);
    if (y1 > y2) std::swap(y1, y2);
    x1 = std::max(0, x1); y1 = std::max(0, y1);
    x2 = std::min(dw - 1, x2); y2 = std::min(dh - 1, y2);
    for (int y = y1; y <= y2; ++y) {
        uint8_t* p = dst + (y * dw + x1) * 3;
        for (int x = x1; x <= x2; ++x, p += 3) {
            p[0] = (uint8_t)((p[0] * (255 - a) + b * a + 127) / 255);
            p[1] = (uint8_t)((p[1] * (255 - a) + g * a + 127) / 255);
            p[2] = (uint8_t)((p[2] * (255 - a) + r * a + 127) / 255);
        }
    }
}

void class_color(int cls, uint8_t& b, uint8_t& g, uint8_t& r) {
    if (cls < 0 || cls >= 80) cls = 0;
    r = COCO_PALETTE[cls][0];
    g = COCO_PALETTE[cls][1];
    b = COCO_PALETTE[cls][2];
}

}  // namespace yvm
