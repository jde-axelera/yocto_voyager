// tasks/embed.h
//
// Re-identification / embedding models (osnet, facenet). The model output is
// a single int8 feature vector of length D (typically 128, 256, 512). For a
// real demo we'd compare incoming vectors against a gallery; for this
// benchmark harness we just compute the L2 norm of the vector and stash it.
#pragma once

#include "task.h"
#include "yolo_preproc.h"

namespace yvm {

struct EmbedResult : public TaskResult {
    int    dim       = 0;       // vector length (e.g. 512 for osnet)
    double l2_norm   = 0.0;     // sqrt(sum(score_i^2))  — sanity-check metric
    int8_t first_few[8] = {};   // first 8 components, for visual confirmation
};

class EmbedTask : public TaskHandler {
public:
    const char* name() const override { return "embed"; }

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
    size_t     vec_len_ = 0;
};

}  // namespace yvm
