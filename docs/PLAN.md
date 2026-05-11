# Model-zoo expansion plan

Branch: **`feat/model-zoo`** — all work isolated here. `main` is not touched.

## Goal

One unified C++ binary (`yvm_demo`) that can run any Voyager-SDK zoo model on the
Antelao SoM, with task-class–specific decode + draw modules sharing the existing
pipeline (preproc → AIPU → postproc → draw → ffmpeg). Cover one representative
model per task class, benchmark each end-to-end, and report.

## Task classes (one model each)

| Task | Selected | YAML path |
|---|---|---|
| Object detection | `yolo11n-coco-onnx` | `ax_models/zoo/yolo/object_detection/yolo11n-coco-onnx.yaml` |
| Instance segmentation | `yolov8nseg-coco-onnx` | `ax_models/zoo/yolo/instance_segmentation/yolov8nseg-coco-onnx.yaml` |
| Keypoint / pose | `yolov8npose-coco-onnx` | `ax_models/zoo/yolo/keypoint_detection/yolov8npose-coco-onnx.yaml` |
| OBB detection | `yolo11nobb-coco-onnx` | `ax_models/zoo/yolo/obb_detection/yolo11nobb-coco-onnx.yaml` |
| Classification | `mobilenetv2-imagenet-onnx` | `ax_models/zoo/torchvision/classification/mobilenetv2-imagenet-onnx.yaml` |
| Semantic segmentation | `unet_fcn_512-cityscapes` | `ax_models/zoo/mmlab/mmseg/unet_fcn_512-cityscapes.yaml` |
| Face detection | `retinaface-mobilenet0.25-widerface-onnx` | `ax_models/zoo/torch/retinaface-mobilenet0.25-widerface-onnx.yaml` |
| Embeddings (re-id) | `osnet-x1-0-market1501-onnx` | `ax_models/zoo/torch/osnet-x1-0-market1501-onnx.yaml` |

LLMs are excluded — they use a different runtime path (Python only, KV-cache
heavy, no real-time video use case).

## Code layout

```
src/
├── task.h                   # TaskHandler interface
├── tasks/
│   ├── detection.h / .cpp   # YOLO box decode + NMS + draw  (existing logic moved here)
│   ├── classify.h / .cpp    # softmax + top-K label overlay
│   ├── seg.h / .cpp         # mask decode + overlay (instance + semantic share most code)
│   ├── pose.h / .cpp        # keypoint regression + skeleton draw
│   ├── obb.h / .cpp         # oriented box decode + rotated-rect draw
│   ├── face.h / .cpp        # face boxes + 5-point landmarks
│   └── embed.h / .cpp       # feature vector (no drawing; renders the L2-normalized vector summary)
├── yvm_demo.cpp             # renamed orchestrator (was yolo_demo_multi.cpp)
└── ... (existing modules: dma_heap, drawing, font, subprocess, frame, py_aipu_client)
```

`yvm_demo` selects the active task via `--task <name>`. The default is
`detection`, so the existing `--py-dispatch` + `--boxes-only` demo behaviour is
preserved. Each `TaskHandler`:

```cpp
struct TaskResult { /* per-task struct */ };

class TaskHandler {
public:
    virtual ~TaskHandler() = default;

    // Optional. Default = letterbox + quantize matching the model's input.
    virtual void preproc(const Frame& src, int8_t* dst,
                         const PreprocCtx&) const;

    virtual void postproc(const std::vector<const int8_t*>& out_ptrs,
                          const std::vector<axrTensorInfo>& out_infos,
                          float conf, float iou,
                          float lscale, int padx, int pady,
                          int sw, int sh,
                          TaskResult& out) const = 0;

    virtual void draw(Image& canvas, const TaskResult&) const = 0;
};
```

The orchestrator owns one `TaskHandler*` for the run and dispatches to it in
the drawer thread.

## Deploy workflow (build host, space-aware)

`scripts/zoo_deploy.sh <task> <yaml-stem>` is a tightened version of the
existing `03_deploy_model.sh`:

1. Patch the YAML in-place for 4 AIPU cores (or 1 core for tiny classifiers
   where 4-core compile fails).
2. Run `deploy.py --mode QUANTCOMPILE --aipu-cores=4 <yaml-stem>`.
3. Tar **only** `build/<stem>/<stem>/<n>/{model.json,*.elf,*.json}` →
   `deploys/<task>/<stem>.tar.gz`.
4. `rm -rf build/<stem>/` immediately after tarring — frees the multi-GB
   intermediate before the next deploy.

## Benchmark workflow (SBC)

`scripts/zoo_bench.sh <task> <stem>`:

1. SCP the tarball from the build host to `~/zoo/`.
2. Extract → run `yvm_demo --task <task> --model ...` against
   `traffic4_480p_mt.mp4` for 30 s (no `--display`, output to `/dev/null`
   via `--out /tmp/null_`).
3. Parse the final `=== done in X s ===` block for fps + latency.
4. Append a row to `report.csv`.
5. `rm -rf ~/zoo/<stem>/` to keep disk under control.

## Output

`docs/MODEL_ZOO_REPORT.md` — one row per model:
- task class
- input size, batch
- device fps, system fps, single-call latency
- status (OK / DEPLOY_FAIL / RUN_FAIL / POSTPROC_FAIL)
- notes

## Branch hygiene

- Never `git checkout main`.
- Never `git push origin main`.
- All commits go to `feat/model-zoo`; rebase/squash on this branch only.
- The accidentally-pushed `hud_usb.png` on `main` history stays as-is.
