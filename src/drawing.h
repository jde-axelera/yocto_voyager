// drawing.h
//
// Tiny C++ drawing primitives that work directly on a BGR24 frame buffer.
// All coordinates are clipped; out-of-bounds calls are no-ops.
#pragma once

#include <cstdint>

namespace yvm {

/// Owning view of a packed BGR24 image. `data` is owned elsewhere; size is w*h*3.
struct Image {
    uint8_t* data;
    int      w;
    int      h;
};

/// Single-pixel BGR write with clipping.
void put_px(const Image& im, int x, int y, uint8_t b, uint8_t g, uint8_t r);

/// Solid horizontal/vertical 1-pixel line.
void draw_hline(const Image& im, int x1, int x2, int y,
                uint8_t b, uint8_t g, uint8_t r);
void draw_vline(const Image& im, int x, int y1, int y2,
                uint8_t b, uint8_t g, uint8_t r);

/// Open rectangle outline (no fill) of given `thick`ness in pixels.
void draw_rect(const Image& im, int x1, int y1, int x2, int y2,
               uint8_t b, uint8_t g, uint8_t r, int thick = 2);

/// Solid-filled rectangle.
void fill_rect(const Image& im, int x1, int y1, int x2, int y2,
               uint8_t b, uint8_t g, uint8_t r);

/// Alpha-blended filled rectangle. alpha is 0..255 (255 = fully opaque).
/// Color is given in RGB order for readability; stored in BGR.
void fill_rect_alpha(uint8_t* dst, int dw, int dh,
                     int x1, int y1, int x2, int y2,
                     uint8_t r, uint8_t g, uint8_t b, uint8_t alpha);

/// Look up a stable RGB triplet for a COCO class id (0..79).
/// Out-of-range ids map to class 0.
void class_color(int cls, uint8_t& b, uint8_t& g, uint8_t& r);

}  // namespace yvm
