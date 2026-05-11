// yolo_preproc.h
//
// CPU preprocess from BGR24 source → int8 RGBA-padded layout the AIPU expects.
//
// The model's quantization (scale = 1/255, zero_point = -128) makes the per-
// pixel quantize step equivalent to a single XOR with 0x80 — we don't actually
// need to compute (px/255 - 0)/(1/255) - 128.
//
// Layout:
//   - source is sw × sh × 3 packed BGR (uint8).
//   - dst   is c.H × c.W × c.channels int8, where the model's NHWC padding
//     contracts say `dst[1:H-1, 1:W-1, 0:3]` is the "active" region inside which
//     we place a letterboxed image at offset (padx, pady) of size new_w × new_h.
//     All padding bytes are pre-set to int8 -128 (= uint8 0x80) by the caller.
#pragma once

#include <cstdint>
#include "axruntime/axruntime.h"

namespace yvm {

/// Cached, model-derived layout/quantization parameters.
struct PreprocCtx {
    int H, W, channels;          ///< padded tensor dims (NHWC)
    int yp_l, yp_r, xp_l, xp_r;  ///< model padding on H and W (per side)
    int cp_l, cp_r;              ///< model padding on channel dim
    int uw, uh;                  ///< "unpadded" inner size = H-yp_l-yp_r, etc.
    int8_t pad_val;              ///< zero_point clamped to int8 (typically -128)
};

/// Derive a PreprocCtx from the model's input tensor info.
PreprocCtx make_preproc(const axrTensorInfo& info);

/// Letterbox the source BGR into the model's int8 RGBA-padded input slot.
/// out-params `letter_scale`, `padx_640`, `pady_640` describe the letterbox
/// transform so the postprocess can map detection boxes back to original coords.
void preprocess(const uint8_t* src_bgr, int sw, int sh,
                int8_t* dst, const PreprocCtx& c,
                float& letter_scale, int& padx_640, int& pady_640);

}  // namespace yvm
