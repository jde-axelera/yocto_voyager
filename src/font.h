// font.h
//
// TTF text rendering on a BGR24 frame.
//
// build_atlas() rasterizes ASCII 0x20..0x7E from an embedded Liberation Sans
// Bold (font_data.h, baked at build time) at a given pixel height, into a
// glyph cache. draw_text_ttf() then alpha-blends those glyphs onto a BGR
// image one character at a time.
//
// Atlas pointers are owned by the caller (heap-allocated); the canonical use
// is to build one atlas at startup per text size you need and never free it.
#pragma once

#include <cstdint>
#include <map>
#include <vector>

namespace yvm {

struct Glyph {
    int w = 0, h = 0;          ///< rasterized bitmap dims
    int x0 = 0, y0 = 0;         ///< glyph-relative origin (stbtt convention)
    int advance_px = 0;         ///< how far to step the pen after drawing
    std::vector<uint8_t> alpha; ///< w*h 0..255 alpha mask
};

struct FontAtlas {
    std::map<char, Glyph> glyphs;
    int line_height = 0;
    int baseline    = 0;
};

/// Rasterize an atlas at the given pixel height. Returns a heap-owned pointer.
/// Internally thread-safe (one stbtt_fontinfo is shared, guarded).
FontAtlas* build_atlas(float pixel_height);

/// Total advance width of a string in the given atlas.
int text_width(const FontAtlas& a, const char* s);

/// Alpha-blend a string into a BGR image at (x, y) (top-left of text bbox).
/// Color is given in (r, g, b) for readability.
void draw_text_ttf(uint8_t* dst, int dw, int dh, int x, int y, const char* s,
                   uint8_t r, uint8_t g, uint8_t b, const FontAtlas& atlas);

/// Same as draw_text_ttf but with a 1-pixel black drop shadow underneath.
void draw_text_shadow(uint8_t* dst, int dw, int dh, int x, int y, const char* s,
                      uint8_t r, uint8_t g, uint8_t b, const FontAtlas& atlas);

}  // namespace yvm
