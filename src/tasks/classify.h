// tasks/classify.h
//
// ImageNet-style single-label classification. One output tensor of shape
// (batch, N) where N is typically 1000 (or padded to 1024). Postproc is just
// argmax over the per-frame slice; draw renders the top-1 class index + raw
// int8 score in the top-left of the source frame.
//
// We deliberately do *not* dequantize: argmax is invariant to the model's
// affine quantization, so we save a per-frame floating-point pass.
#pragma once

#include <cstdint>

#include "task.h"
#include "yolo_preproc.h"

namespace yvm {

struct ClassifyResult : public TaskResult {
    int     top_cls   = -1;
    int     top_score = INT8_MIN;     // raw int8 logit, just for HUD context
    int     num_classes = 0;          // for log/debug
};

class ClassifyTask : public TaskHandler {
public:
    const char* name() const override { return "classify"; }

    void init(axrModel* model,
              const std::vector<axrTensorInfo>& in_infos,
              const std::vector<axrTensorInfo>& out_infos) override;

    void preproc(const uint8_t* bgr, int sw, int sh,
                 int8_t* dst_slot,
                 float& lscale, int& padx, int& pady) const override;

    std::unique_ptr<TaskResult> postproc(
        const std::vector<const int8_t*>& out_ptrs,
        const std::vector<axrTensorInfo>& out_infos,
        float conf_thresh, float iou_thresh,
        float lscale, int padx, int pady,
        int sw, int sh) const override;

    void draw(Image& canvas, const TaskResult& result,
              const FontAtlas* label_font) const override;

private:
    PreprocCtx ctx_{};
    size_t     slot_bytes_ = 0;
};

}  // namespace yvm
