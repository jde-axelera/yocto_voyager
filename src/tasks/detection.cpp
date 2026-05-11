// tasks/detection.cpp
#include "tasks/detection.h"

namespace yvm {

void DetectionTask::init(axrModel* /*model*/,
                         const std::vector<axrTensorInfo>& in_infos,
                         const std::vector<axrTensorInfo>& out_infos) {
    ctx_       = make_preproc(in_infos[0]);
    tbl_       = classify_outputs(out_infos);
    out_infos_ = out_infos;
}

void DetectionTask::preproc(const uint8_t* bgr, int sw, int sh,
                            int8_t* dst,
                            float& lscale, int& padx, int& pady) const {
    yvm::preprocess(bgr, sw, sh, dst, ctx_, lscale, padx, pady);
}

std::unique_ptr<TaskResult> DetectionTask::postproc(
        const std::vector<const int8_t*>& out_ptrs,
        const std::vector<axrTensorInfo>& /*out_infos*/,
        float conf_thresh, float iou_thresh,
        float lscale, int padx, int pady,
        int sw, int sh) const {
    auto r = std::make_unique<DetectionResult>();
    std::vector<Detection> raw;
    decode_dfl_sigmoid_filter(out_ptrs, out_infos_, tbl_,
                              conf_thresh, lscale, padx, pady,
                              sw, sh, raw);
    r->dets = nms(std::move(raw), iou_thresh);
    return r;
}

void DetectionTask::draw(Image& canvas, const TaskResult& result,
                         const FontAtlas* /*label_font*/) const {
    const auto& dr = static_cast<const DetectionResult&>(result);
    for (const auto& d : dr.dets) {
        uint8_t b, g, r;
        class_color(d.cls, b, g, r);
        draw_rect(canvas, (int)d.x1, (int)d.y1, (int)d.x2, (int)d.y2, b, g, r, 2);
    }
}

}  // namespace yvm
