// font.cpp — implementation of yvm/font.h (includes the stb_truetype impl).
#include "font.h"

#include <cmath>
#include <cstring>
#include <mutex>

#define STB_TRUETYPE_IMPLEMENTATION
#include "stb/stb_truetype.h"

#include "liberation_sans_bold.h"

namespace yvm {

namespace {
stbtt_fontinfo  g_font;
bool            g_font_inited = false;
std::mutex      g_font_mu;

/// One-time init of the shared stbtt_fontinfo for our embedded TTF.
bool init_font() {
    std::lock_guard<std::mutex> lk(g_font_mu);
    if (g_font_inited) return true;
    if (!stbtt_InitFont(&g_font, kLiberationSansBold,
                        stbtt_GetFontOffsetForIndex(kLiberationSansBold, 0))) {
        return false;
    }
    g_font_inited = true;
    return true;
}
}  // namespace

FontAtlas* build_atlas(float pixel_height) {
    if (!init_font()) return nullptr;
    auto* a = new FontAtlas();
    float scale = stbtt_ScaleForPixelHeight(&g_font, pixel_height);
    int ascent, descent, line_gap;
    stbtt_GetFontVMetrics(&g_font, &ascent, &descent, &line_gap);
    a->baseline    = (int)(ascent * scale);
    a->line_height = (int)((ascent - descent + line_gap) * scale);
    for (int c = 0x20; c < 0x7F; ++c) {
        Glyph g{};
        int adv, lsb;
        stbtt_GetCodepointHMetrics(&g_font, c, &adv, &lsb);
        g.advance_px = (int)std::round(adv * scale);
        int x0, y0, x1, y1;
        stbtt_GetCodepointBitmapBox(&g_font, c, scale, scale, &x0, &y0, &x1, &y1);
        g.w  = x1 - x0; g.h = y1 - y0;
        g.x0 = x0; g.y0 = y0;
        g.alpha.assign((size_t)g.w * g.h, 0);
        if (g.w > 0 && g.h > 0) {
            stbtt_MakeCodepointBitmap(&g_font, g.alpha.data(), g.w, g.h, g.w,
                                      scale, scale, c);
        }
        a->glyphs[(char)c] = std::move(g);
    }
    return a;
}

int text_width(const FontAtlas& a, const char* s) {
    int w = 0;
    for (; *s; ++s) {
        auto it = a.glyphs.find(*s);
        if (it != a.glyphs.end()) w += it->second.advance_px;
    }
    return w;
}

static inline void blit_glyph_bgr(uint8_t* dst, int dw, int dh, const Glyph& g,
                                  int x_pen, int y_pen,
                                  uint8_t r, uint8_t gr, uint8_t b) {
    int gy0 = y_pen + g.y0;
    int gx0 = x_pen + g.x0;
    for (int j = 0; j < g.h; ++j) {
        int yy = gy0 + j;
        if ((unsigned)yy >= (unsigned)dh) continue;
        for (int i = 0; i < g.w; ++i) {
            int xx = gx0 + i;
            if ((unsigned)xx >= (unsigned)dw) continue;
            uint8_t a = g.alpha[j * g.w + i];
            if (!a) continue;
            uint8_t* px = dst + (yy * dw + xx) * 3;
            px[0] = (uint8_t)((px[0] * (255 - a) + b  * a + 127) / 255);
            px[1] = (uint8_t)((px[1] * (255 - a) + gr * a + 127) / 255);
            px[2] = (uint8_t)((px[2] * (255 - a) + r  * a + 127) / 255);
        }
    }
}

void draw_text_ttf(uint8_t* dst, int dw, int dh, int x, int y, const char* s,
                   uint8_t r, uint8_t gr, uint8_t b, const FontAtlas& atlas) {
    int cur_x = x;
    int baseline_y = y + atlas.baseline;
    for (; *s; ++s) {
        auto it = atlas.glyphs.find(*s);
        if (it == atlas.glyphs.end()) continue;
        blit_glyph_bgr(dst, dw, dh, it->second, cur_x, baseline_y, r, gr, b);
        cur_x += it->second.advance_px;
    }
}

void draw_text_shadow(uint8_t* dst, int dw, int dh, int x, int y, const char* s,
                      uint8_t r, uint8_t gr, uint8_t b, const FontAtlas& atlas) {
    draw_text_ttf(dst, dw, dh, x + 1, y + 1, s, 0, 0, 0, atlas);
    draw_text_ttf(dst, dw, dh, x,     y,     s, r, gr, b, atlas);
}

}  // namespace yvm
