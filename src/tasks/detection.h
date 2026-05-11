// tasks/detection.h
//
// YOLO-style object detection task: DFL bbox decode + sigmoid class scores +
// class-aware NMS. Wraps the existing yolo_preproc + yolo_postproc modules
// behind the generic TaskHandler interface.
#pragma once

#include <array>
#include <vector>

#include "task.h"
#include "yolo_postproc.h"
#include "yolo_preproc.h"

namespace yvm {

struct DetectionResult : public TaskResult {
    std::vector<Detection> dets;
};

class DetectionTask : public TaskHandler {
public:
    const char* name() const override { return "detection"; }

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
    PreprocCtx                          ctx_{};
    std::array<std::array<int, 2>, 3>   tbl_{};
    std::vector<axrTensorInfo>          out_infos_;
};

}  // namespace yvm
