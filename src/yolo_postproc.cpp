// yolo_postproc.cpp
#include "yolo_postproc.h"

#include <algorithm>
#include <cmath>

namespace yvm {

namespace {
inline float iou(const Detection& a, const Detection& b) {
    float xx1 = std::max(a.x1, b.x1), yy1 = std::max(a.y1, b.y1);
    float xx2 = std::min(a.x2, b.x2), yy2 = std::min(a.y2, b.y2);
    float iw = std::max(0.0f, xx2 - xx1), ih = std::max(0.0f, yy2 - yy1);
    float inter = iw * ih;
    float ua = (a.x2 - a.x1) * (a.y2 - a.y1);
    float ub = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (ua + ub - inter + 1e-9f);
}
}  // namespace

std::array<std::array<int, 2>, 3> classify_outputs(const std::vector<axrTensorInfo>& outs) {
    std::array<std::array<int, 2>, 3> table{};
    for (auto& row : table) row = {-1, -1};
    // For each output, infer (stride_index, kind) from shape:
    //   gh == 80 → stride 8;  gh == 40 → stride 16; gh == 20 → stride 32
    //   channel == 64 → bbox;        channel == 80 → class
    for (size_t i = 0; i < outs.size(); ++i) {
        const auto& o = outs[i];
        int gh   = (int)o.dims[1];
        int kind = (int)o.dims[3] == 64 ? 0 : 1;
        int s    = (gh == 80) ? 0 : (gh == 40) ? 1 : 2;
        table[s][kind] = (int)i;
    }
    return table;
}

void decode_dfl_sigmoid_filter(
    const std::vector<const int8_t*>& bufs,
    const std::vector<axrTensorInfo>& infos,
    const std::array<std::array<int, 2>, 3>& tbl,
    float conf_thresh,
    float scale_letterbox, int padx_640, int pady_640,
    int orig_w, int orig_h,
    std::vector<Detection>& out)
{
    out.clear();
    static constexpr int strides[3] = {8, 16, 32};
    float bins[16];

    for (int s = 0; s < 3; ++s) {
        int bbox_idx = tbl[s][0], cls_idx = tbl[s][1];
        if (bbox_idx < 0 || cls_idx < 0) continue;

        const axrTensorInfo& bi = infos[bbox_idx];
        const axrTensorInfo& ci = infos[cls_idx];

        const int gh     = (int)bi.dims[1], gw = (int)bi.dims[2];
        const int stride = strides[s];

        // Class output is 80 unpadded channels but is sometimes padded on the
        // channel axis (e.g. 80→128). Handle both transparently.
        const int cls_ch_total = (int)ci.dims[3];
        const int cls_unpad_l  = ci.padding[3][0];
        const int cls_unpad_n  = cls_ch_total - cls_unpad_l - ci.padding[3][1];

        const int bbox_ch_total = (int)bi.dims[3];  // = 64 = 4 sides × 16 DFL bins

        const float bs = bi.scale, bz = (float)bi.zero_point;
        const float cs = ci.scale, cz = (float)ci.zero_point;
        const int8_t* bbox_data = bufs[bbox_idx];
        const int8_t* cls_data  = bufs[cls_idx];

        for (int gy = 0; gy < gh; ++gy) {
            for (int gx = 0; gx < gw; ++gx) {
                const int8_t* bbox_q = bbox_data + ((gy * gw + gx) * bbox_ch_total);
                const int8_t* cls_q  = cls_data  + ((gy * gw + gx) * cls_ch_total) + cls_unpad_l;

                // class: take the argmax of the dequantized logits, then sigmoid.
                int   best_cls   = -1;
                float best_logit = -1e30f;
                for (int k = 0; k < cls_unpad_n; ++k) {
                    float f = ((int)cls_q[k] - cz) * cs;
                    if (f > best_logit) { best_logit = f; best_cls = k; }
                }
                float best_score = 1.0f / (1.0f + std::exp(-best_logit));
                if (best_score < conf_thresh) continue;

                // bbox: 4 distances (l,t,r,b) each as a 16-bin DFL (softmax → expectation).
                float dist[4];
                for (int side = 0; side < 4; ++side) {
                    float maxv = -1e30f;
                    for (int b = 0; b < 16; ++b) {
                        float f = ((int)bbox_q[side * 16 + b] - bz) * bs;
                        bins[b] = f;
                        if (f > maxv) maxv = f;
                    }
                    float sum = 0;
                    for (int b = 0; b < 16; ++b) { bins[b] = std::exp(bins[b] - maxv); sum += bins[b]; }
                    float inv = 1.0f / sum;
                    float exp_val = 0;
                    for (int b = 0; b < 16; ++b) exp_val += bins[b] * inv * (float)b;
                    dist[side] = exp_val;
                }

                // Anchor center in 640×640 letterbox-space.
                float cx = (gx + 0.5f) * stride;
                float cy = (gy + 0.5f) * stride;
                float x1 = cx - dist[0] * stride, y1 = cy - dist[1] * stride;
                float x2 = cx + dist[2] * stride, y2 = cy + dist[3] * stride;

                // Letterbox-undo: subtract the letterbox padding, divide by scale.
                x1 = (x1 - padx_640) / scale_letterbox;
                y1 = (y1 - pady_640) / scale_letterbox;
                x2 = (x2 - padx_640) / scale_letterbox;
                y2 = (y2 - pady_640) / scale_letterbox;

                // Clip to image bounds.
                x1 = std::clamp(x1, 0.0f, (float)orig_w - 1);
                y1 = std::clamp(y1, 0.0f, (float)orig_h - 1);
                x2 = std::clamp(x2, 0.0f, (float)orig_w - 1);
                y2 = std::clamp(y2, 0.0f, (float)orig_h - 1);
                if (x2 <= x1 || y2 <= y1) continue;

                out.push_back({x1, y1, x2, y2, best_score, best_cls});
            }
        }
    }
}

std::vector<Detection> nms(std::vector<Detection> dets, float iou_thresh, int max_out) {
    std::sort(dets.begin(), dets.end(),
              [](const Detection& a, const Detection& b) { return a.score > b.score; });
    std::vector<Detection> out;
    out.reserve(std::min<size_t>(dets.size(), max_out));
    std::vector<char> killed(dets.size(), 0);
    for (size_t i = 0; i < dets.size(); ++i) {
        if (killed[i]) continue;
        out.push_back(dets[i]);
        if ((int)out.size() >= max_out) break;
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (killed[j] || dets[j].cls != dets[i].cls) continue;
            if (iou(dets[i], dets[j]) > iou_thresh) killed[j] = 1;
        }
    }
    return out;
}

}  // namespace yvm
