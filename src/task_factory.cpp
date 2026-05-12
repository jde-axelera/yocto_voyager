// task_factory.cpp
//
// Registry mapping `--task <name>` strings to TaskHandler concrete
// implementations. Adding a new task is one line here plus the
// `src/tasks/<name>.{h,cpp}` files.
//
// Current state:
//   detection   — full decode + box draw (production)
//   classify    — argmax over per-frame slot + top-1 label draw (production)
//   pose, seg, obb, face, embed — preproc + inference only via StubTask; their
//   real decoders are follow-ups. Useful as benchmark targets right away.
#include "task.h"

#include "tasks/classify.h"
#include "tasks/detection.h"
#include "tasks/embed.h"
#include "tasks/stubs.h"

namespace yvm {

std::unique_ptr<TaskHandler> make_task(const std::string& name) {
    if (name == "detection") return std::make_unique<DetectionTask>();
    if (name == "classify")  return std::make_unique<ClassifyTask>();
    if (name == "pose")      return std::make_unique<StubTask>("pose");
    if (name == "seg")       return std::make_unique<StubTask>("seg");
    if (name == "obb")       return std::make_unique<StubTask>("obb");
    if (name == "face")      return std::make_unique<StubTask>("face");
    if (name == "embed")     return std::make_unique<EmbedTask>();
    return nullptr;
}

const char* known_task_names() {
    return "detection, classify, pose, seg, obb, face, embed";
}

}  // namespace yvm
