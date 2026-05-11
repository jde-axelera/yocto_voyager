// tasks/classify.cpp
#include "tasks/classify.h"

#include <cstdio>
#include <cstring>

namespace yvm {

void ClassifyTask::init(axrModel* /*model*/,
                        const std::vector<axrTensorInfo>& in_infos,
                        const std::vector<axrTensorInfo>& out_infos) {
    ctx_ = make_preproc(in_infos[0]);
    // out_sizes[0] is the *full-batch* tensor size. Per-frame slot = total / batch.
    // We don't know the batch here, but the per-frame buf the orchestrator
    // hands us in postproc is already sized correctly; we only need to know
    // how many logits there are. Walk the dims.
    const auto& info = out_infos[0];
    size_t n = 1;
    for (size_t d = 0; d < info.ndims; ++d) n *= info.dims[d];
    // Strip the batch dim — first dim is batch for AIPU-compiled models.
    if (info.ndims >= 1 && info.dims[0] > 0) n /= info.dims[0];
    slot_bytes_ = n;   // 1 byte per int8 logit
}

void ClassifyTask::preproc(const uint8_t* bgr, int sw, int sh,
                           int8_t* dst,
                           float& lscale, int& padx, int& pady) const {
    // ImageNet-style classifiers are happy with letterbox+quantize too —
    // accuracy drops slightly vs. center-crop+normalize but the AIPU sees the
    // same int8 layout. For now we share preproc with detection; classification-
    // specific center-crop is a future refinement.
    yvm::preprocess(bgr, sw, sh, dst, ctx_, lscale, padx, pady);
}

std::unique_ptr<TaskResult> ClassifyTask::postproc(
        const std::vector<const int8_t*>& out_ptrs,
        const std::vector<axrTensorInfo>& /*out_infos*/,
        float /*conf_thresh*/, float /*iou_thresh*/,
        float /*lscale*/, int /*padx*/, int /*pady*/,
        int /*sw*/, int /*sh*/) const {
    auto r = std::make_unique<ClassifyResult>();
    if (out_ptrs.empty() || !out_ptrs[0]) return r;
    const int8_t* logits = out_ptrs[0];
    int    top_i = 0;
    int8_t top_v = logits[0];
    for (size_t i = 1; i < slot_bytes_; ++i) {
        if (logits[i] > top_v) { top_v = logits[i]; top_i = (int)i; }
    }
    r->top_cls     = top_i;
    r->top_score   = top_v;
    r->num_classes = (int)slot_bytes_;
    return r;
}

void ClassifyTask::draw(Image& canvas, const TaskResult& result,
                        const FontAtlas* label_font) const {
    const auto& cr = static_cast<const ClassifyResult&>(result);
    char buf[64];
    std::snprintf(buf, sizeof buf, "cls #%d  (%d)", cr.top_cls, cr.top_score);

    int x = 12, y = 12;
    int tw = label_font ? text_width(*label_font, buf) : (int)std::strlen(buf) * 8;
    int th = label_font ? label_font->line_height : 14;
    fill_rect_alpha(canvas.data, canvas.w, canvas.h,
                    x, y, x + tw + 16, y + th + 8, 0, 0, 0, 200);
    if (label_font)
        draw_text_shadow(canvas.data, canvas.w, canvas.h, x + 8, y + 4,
                         buf, 255, 255, 255, *label_font);
}

}  // namespace yvm
