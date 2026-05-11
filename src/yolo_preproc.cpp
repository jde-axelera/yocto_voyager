// yolo_preproc.cpp
#include "yolo_preproc.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace yvm {

PreprocCtx make_preproc(const axrTensorInfo& info) {
    PreprocCtx c{};
    c.H        = (int)info.dims[1];
    c.W        = (int)info.dims[2];
    c.channels = (int)info.dims[3];
    c.yp_l = info.padding[1][0]; c.yp_r = info.padding[1][1];
    c.xp_l = info.padding[2][0]; c.xp_r = info.padding[2][1];
    c.cp_l = info.padding[3][0]; c.cp_r = info.padding[3][1];
    c.uh = c.H - c.yp_l - c.yp_r;
    c.uw = c.W - c.xp_l - c.xp_r;
    c.pad_val = (int8_t)std::clamp(info.zero_point, -128, 127);
    return c;
}

void preprocess(const uint8_t* src_bgr, int sw, int sh,
                int8_t* dst, const PreprocCtx& c,
                float& letter_scale, int& padx_640, int& pady_640)
{
    const int channels = c.channels;
    const int duw = c.uw, duh = c.uh;

    // Letterbox math: scale so the longer side fits inside (uw, uh).
    float sx = (float)duw / sw;
    float sy = (float)duh / sh;
    letter_scale = std::min(sx, sy);
    int new_w = (int)std::round(sw * letter_scale);
    int new_h = (int)std::round(sh * letter_scale);
    padx_640 = (duw - new_w) / 2;
    pady_640 = (duh - new_h) / 2;

    // Top model-padding rows (fully padded).
    std::fill_n(dst, c.yp_l * c.W * channels, c.pad_val);
    int8_t* row_out = dst + c.yp_l * c.W * channels;

    // 16.16 fixed-point step for nearest-neighbor resize.
    const uint32_t fp_x_step = (uint32_t)((double)sw * 65536.0 / new_w);
    const uint32_t fp_y_step = (uint32_t)((double)sh * 65536.0 / new_h);

    for (int y = 0; y < duh; ++y) {
        int8_t* row = row_out + y * c.W * channels;

        // Leading model x-pad.
        std::fill_n(row, c.xp_l * channels, c.pad_val);
        int8_t* p = row + c.xp_l * channels;

        if (y < pady_640 || y >= pady_640 + new_h) {
            // Whole row is letterbox padding (above/below the active region).
            std::fill_n(p, duw * channels, c.pad_val);
        } else {
            // Leading letterbox padding (left of active region).
            std::fill_n(p, padx_640 * channels, c.pad_val);
            p += padx_640 * channels;

            int src_y = (int)((uint64_t)(y - pady_640) * fp_y_step >> 16);
            if (src_y >= sh) src_y = sh - 1;
            const uint8_t* src_row = src_bgr + src_y * sw * 3;

            uint32_t fp_x = 0;
            for (int x = 0; x < new_w; ++x, fp_x += fp_x_step) {
                int src_x = fp_x >> 16;
                if (src_x >= sw) src_x = sw - 1;
                const uint8_t* px = src_row + src_x * 3;
                // Model expects RGB; source is BGR.
                // Channel layout: [cp_l pad][R][G][B][cp_r pad]
                for (int k = 0; k < c.cp_l; ++k) *p++ = c.pad_val;
                // Quantize via "pixel - 128" (model uses scale=1/255, zp=-128).
                *p++ = (int8_t)((int)px[2] - 128);
                *p++ = (int8_t)((int)px[1] - 128);
                *p++ = (int8_t)((int)px[0] - 128);
                for (int k = 0; k < c.cp_r; ++k) *p++ = c.pad_val;
            }

            // Trailing letterbox padding (right of active region).
            int rest = duw - padx_640 - new_w;
            if (rest > 0) std::fill_n(p, rest * channels, c.pad_val);
        }

        // Trailing model x-pad.
        std::fill_n(row + (c.xp_l + duw) * channels, c.xp_r * channels, c.pad_val);
    }

    // Bottom model-padding rows.
    std::fill_n(row_out + duh * c.W * channels, c.yp_r * c.W * channels, c.pad_val);
}

}  // namespace yvm
