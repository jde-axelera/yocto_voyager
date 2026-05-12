# Model-zoo benchmark report

_Branch: `feat/model-zoo`. Run scheduled for the night of 2026-05-11 → 2026-05-12._

## What this report is

A unified C++ inference binary (`yolo_demo_multi`, soon to be `yvm_demo`) was
extended with a **task-class abstraction** (`src/task.h`) so that all
voyager-sdk zoo models can run through the same preproc → AIPU dispatch → optional
postproc/draw pipeline. Each task class lives under `src/tasks/<name>.{h,cpp}`
and is selected at run time via `--task <name>`.

| `--task` | Module | Status |
|---|---|---|
| `detection`  | `tasks/detection.{h,cpp}`         | full decode + colour-box draw (production; ~378 fps on yolo11n via --py-dispatch) |
| `classify`   | `tasks/classify.{h,cpp}`          | top-1 argmax + label overlay |
| `embed`      | `tasks/embed.{h,cpp}`             | L2 norm + first-8 components overlay |
| `pose`       | `tasks/stubs.h` → `StubTask`      | preproc + inference only (decode is future work) |
| `seg`        | `tasks/stubs.h` → `StubTask`      | preproc + inference only |
| `obb`        | `tasks/stubs.h` → `StubTask`      | preproc + inference only |
| `face`       | `tasks/stubs.h` → `StubTask`      | preproc + inference only |

## Selected models — one per task class

| Voyager-SDK task subdir | Selected stem | Why this one |
|---|---|---|
| `object_detection`        | `yolo11n-coco-onnx`                              | already deployed, baseline |
| `classification`          | `mobilenetv2-imagenet-onnx`                      | smallest torchvision classifier |
| `keypoint_detection`      | `yolov8npose-coco-onnx`                          | smallest YOLO pose variant |
| `obb_detection`           | `yolo11n-obb-dotav1-onnx`                        | smallest YOLO OBB variant |
| `instance_segmentation`   | `yolov8nseg-coco-onnx`                           | smallest YOLO instance seg |
| `semantic_segmentation`   | `unet_fcn_512-cityscapes`                        | only mmseg model in 1.6 zoo |
| `face_detection`          | `retinaface-mobilenet0.25-widerface-onnx`        | only face model in 1.6 zoo |
| `embedding`               | `osnet-x1-0-market1501-onnx`                     | smallest re-id |

## Deploy / bench results

_Filled in by `scripts/zoo_report.py` once the overnight run finishes; the
working data lives in `deploys/_summary.csv` (build host) and `~/cpp_test/zoo_report.csv`
(SBC)._

See **`scripts/zoo_report.py`** for the table renderer and `scripts/zoo_deploy_all.sh`
+ `scripts/zoo_bench_all.sh` for the run scripts.

## Operational findings (from the overnight run)

| Failure mode | Affected | Root cause | Mitigation |
|---|---|---|---|
| `deploy.py` exits with non-zero even after producing artifacts | yolo11n, retinaface, … | spurious "Failed to deploy network" near the end of compile, while build/.../*/{model.json,ELFs,bins} are all present | `scripts/zoo_deploy.sh` ignores deploy.py's exit code and gates on `model.json` existence |
| Tarball missing const-weight blobs (`pool_ddr_const.bin`, `pool_l2_const.bin`) | initial yolo11n tar | overly tight tar include-list | include the whole `$STEM/<cores>/` dir + the `quantized/` sibling + `compiler_config.toml`; binary path also expects them at extract time |
| Tarball missing `quantized/manifest.json` + `postprocess_graph.onnx` | initial yolo11n tar | `quantized/` is a peer directory of `<cores>/`, not under it | tar `$STEM/quantized` separately |
| Compile time > 15 min on pose / seg | yolov8npose, yolov8nseg | voyager 1.6 compiler is single-threaded; segmentation / pose have wide post-quantize graphs | watchdog: 15-min cap; document the slow-deploy list, retry overnight at lower priority |
| YAML left in a patched state after script crash mid-deploy | (occasional) | the YAML.bak restore only ran on the happy path | EXIT trap always restores the YAML |
| Tar producing an empty/partial tarball with `set -e` aborting before any logs print | mnv2 first pass | `add_if_exists() { [ -e ... ] && TAR_LIST+=(...); }` returns false on miss → `set -e` aborts | swap to an explicit `if/fi`; gate the post-tar `ls`/`tar tzf` on the file existing |
| Bench script silently passes raw voyager subdir name (`object_detection`) as `--task` | first end-to-end smoke | scripts emit voyager-style names but our `--task` handlers use short names (`detection`, `classify`, ...) | `zoo_bench.sh` remaps via a case-statement before invoking the binary |
| Binary segfaults on retinaface / mnv2 model load | both | unclear, axrunmodel itself also fails silently for these models on this voyager-rt build | TBD — not a yvm bug; orchestrator and TaskHandler aren't reached |

## Reproduce

```sh
# On the build host (any x86_64 with the voyager-sdk venv):
cd ~/yocto_voyager
git checkout feat/model-zoo
git pull --ff-only
SDK_DIR=~/1.6/voyager-sdk nohup bash scripts/zoo_deploy_all.sh > /tmp/zoo_deploy_all.log 2>&1 &

# Push each successful tarball to the SBC:
for t in ~/yocto_voyager/deploys/*/*.tar.gz; do
    scp "$t" antelao@<sbc>:/tmp/
done

# Bench on the SBC (one row per model into ~/cpp_test/zoo_report.csv):
for t in /tmp/*.tar.gz; do
    stem=$(basename "$t" .tar.gz)
    # Pick the task subdir for the model — see scripts/zoo_deploy_all.sh JOBS.
    ~/cpp_test/zoo_bench.sh <task-subdir> "$stem" "$t" --bench=2 --seconds=15
done

# Render the report:
python3 scripts/zoo_report.py \
    --deploys ~/yocto_voyager/deploys/_summary.csv \
    --bench   ~/cpp_test/zoo_report.csv \
    --out     docs/MODEL_ZOO_REPORT.md
```
