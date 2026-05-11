// task_factory.cpp
//
// One central registry mapping `--task <name>` strings to TaskHandler
// concrete implementations. Adding a new task is one line here plus the
// `src/tasks/<name>.{h,cpp}` files.
#include "task.h"

#include "tasks/detection.h"

namespace yvm {

std::unique_ptr<TaskHandler> make_task(const std::string& name) {
    if (name == "detection") return std::make_unique<DetectionTask>();
    return nullptr;
}

const char* known_task_names() {
    return "detection";
}

}  // namespace yvm
