// task.h
//
// One interface every model-zoo task class implements. The orchestrator owns a
// single `TaskHandler*` chosen via `--task <name>` and dispatches to it inside
// the drawer thread for each completed inference.
//
// Why an interface and not a switch: each task's Result is genuinely different
// (boxes vs masks vs keypoints vs vectors) and so are the preproc / draw paths;
// dispatching by virtual call keeps the orchestrator dead-simple.
#pragma once

#include <memory>
#include <string>
#include <vector>
#include <cstdint>

#include "axruntime/axruntime.h"
#include "drawing.h"
#include "font.h"

namespace yvm {

/// One inference result for one frame. Each concrete task subclasses this and
/// returns its own derivative from postproc().
struct TaskResult {
    virtual ~TaskResult() = default;
};

class TaskHandler {
public:
    virtual ~TaskHandler() = default;

    /// Human-readable name (matched against `--task <name>`).
    virtual const char* name() const = 0;

    /// Called once after the model is loaded. Concrete tasks cache any model-
    /// derived state here (input PreprocCtx, output-tensor classification,
    /// label table, etc.).
    virtual void init(axrModel* model,
                      const std::vector<axrTensorInfo>& in_infos,
                      const std::vector<axrTensorInfo>& out_infos) = 0;

    /// CPU preprocess: source BGR → int8 model input. The task owns its
    /// own PreprocCtx — orchestrator does not.
    /// `lscale/padx/pady` are out-params describing the geometric transform
    /// applied (used by postproc to map outputs back to source coordinates).
    /// For tasks that don't letterbox (e.g. classification center-crop), the
    /// implementation should still write something sane to those fields
    /// (typically lscale=1, padx=pady=0).
    virtual void preproc(const uint8_t* bgr, int sw, int sh,
                         int8_t* dst_slot,
                         float& lscale, int& padx, int& pady) const = 0;

    /// Decode raw output tensors into a task-specific result.
    virtual std::unique_ptr<TaskResult> postproc(
        const std::vector<const int8_t*>& out_ptrs,
        const std::vector<axrTensorInfo>& out_infos,
        float conf_thresh, float iou_thresh,
        float lscale, int padx, int pady,
        int sw, int sh) const = 0;

    /// Draw the result on top of the source BGR frame (in-place).
    /// `label_font` may be null — implementations should degrade gracefully.
    virtual void draw(Image& canvas, const TaskResult& result,
                      const FontAtlas* label_font) const = 0;
};

/// Construct a handler by name. Returns nullptr for unknown names.
std::unique_ptr<TaskHandler> make_task(const std::string& name);

/// Comma-separated list of known task names (for --help text).
const char* known_task_names();

}  // namespace yvm
