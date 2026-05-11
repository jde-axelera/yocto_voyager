// tasks/stubs.h
//
// Light task wrappers for non-detection model classes. Each task here owns its
// own PreprocCtx (so the model loads cleanly and the AIPU runs at full speed
// inside --bench 2 / --bench 1 benchmarks), and ships an *intentional* no-op
// postproc + draw. Filling each task's real decoder is a follow-up: the goal
// of these stubs is to get the full model zoo through the deploy + benchmark
// harness so we know which deploys succeed before sinking effort into the
// per-task decoders.
#pragma once

#include "task.h"
#include "yolo_preproc.h"

namespace yvm {

struct EmptyResult : public TaskResult {};

class StubTask : public TaskHandler {
public:
    explicit StubTask(const char* name) : name_(name) {}

    const char* name() const override { return name_; }

    void init(axrModel* /*model*/,
              const std::vector<axrTensorInfo>& in_infos,
              const std::vector<axrTensorInfo>& /*out_infos*/) override {
        ctx_ = make_preproc(in_infos[0]);
    }

    void preproc(const uint8_t* bgr, int sw, int sh,
                 int8_t* dst,
                 float& lscale, int& padx, int& pady) const override {
        yvm::preprocess(bgr, sw, sh, dst, ctx_, lscale, padx, pady);
    }

    std::unique_ptr<TaskResult> postproc(
        const std::vector<const int8_t*>& /*out_ptrs*/,
        const std::vector<axrTensorInfo>& /*out_infos*/,
        float, float, float, int, int, int, int) const override {
        return std::make_unique<EmptyResult>();
    }

    void draw(Image& /*canvas*/, const TaskResult& /*result*/,
              const FontAtlas* /*label_font*/) const override {}

private:
    const char* name_;
    PreprocCtx  ctx_{};
};

}  // namespace yvm
