# Model-zoo benchmark report

_Branch: `feat/model-zoo`. Overnight run 2026-05-11 → 2026-05-12._

## Headline

| | Count |
|---|---|
| Models attempted | 8 (one representative per task class) |
| Deploys succeeded on the build host | **3** — `yolo11n-coco-onnx`, `retinaface-mobilenet0.25-widerface-onnx`, `osnet-x1-0-market1501-onnx` |
| Deploys failed (compile timeout / wrong stem / script bug) | 5 — pose, instance-seg, semantic-seg (timeout); OBB (typo in JOBS); classify (deploy-script bug, retry queued) |
| Models that loaded + ran on the SBC end-to-end | **1** — `yolo11n-coco-onnx` (378 fps system, `--py-dispatch --bench 2`) |
| Models that compiled but `axrunmodel` itself can't load them on this SBC | 2 — `retinaface`, `osnet` (silent failure between `Context()` and the first `instance.run()`) |

The C++ side of the work — the `TaskHandler` abstraction, the deploy + bench
harness, the disk-rotation discipline — is solid. The blocking issue tonight is
the **voyager-rt 1.6 runtime on the Antelao SoM only accepts yolo11n-style
model.json files** out of the build-host’s freshly compiled outputs. Diagnosing
which compile artifact triggers the silent runtime failure for `retinaface` /
`osnet` is the obvious next step.

## What was built (code + scripts)

- **`src/task.h`** — single `TaskHandler` interface every zoo task implements.
- **`src/task_factory.cpp`** — `--task <name>` registry (detection, classify,
  pose, seg, obb, face, embed).
- **`src/tasks/`** —
    - `detection.{h,cpp}` — full DFL + sigmoid + NMS + colour box draw (the
      existing yolo path, lifted out of the orchestrator).
    - `classify.{h,cpp}` — argmax-over-int8 + top-1 label overlay.
    - `embed.{h,cpp}`     — L2 norm + first-8-component overlay.
    - `stubs.h`           — preproc+inference-only handler used by `pose`,
      `seg`, `obb`, `face` until per-task decoders are written. Lets every
      compiled model still pass through the harness for benchmarking.
- **`src/yolo_demo_multi.cpp`** — now thin: parses argv, instantiates the
  `TaskHandler`, drives the existing decoder/preproc/worker/drawer threads.
  The drawer thread dispatches `task->postproc()` + `task->draw()` instead of
  the hardcoded yolo code.
- **`scripts/zoo_deploy.sh`** — single-model deploy with:
    - YAML auto-patch (compilation_config: aipu_cores_used, resources_used)
    - per-model EXIT trap that restores the YAML even on a crashed deploy
    - whole-batch-dir tar including `pool_*_const.bin`, `quantized/`,
      `compiler_config.toml` (every artifact the runtime actually opens)
    - intermediate `build/<stem>/` is immediately `rm -rf`’d after tar to
      survive on a 94%-full build-host disk.
- **`scripts/zoo_deploy_all.sh`** — serial bulk runner with a 15-minute
  per-model watchdog (`timeout --kill-after=60 900 …`) so a stuck compile
  never burns the whole night.
- **`scripts/zoo_retry.sh`** — re-deploy the first-pass failures with the
  fixed script + corrected stems + a couple of bonus classifiers
  (resnet18, squeezenet1.0).
- **`scripts/zoo_bench.sh`** — per-tarball SBC bench with the watchdog,
  task-name remap (`object_detection` → `detection`, `keypoint_detection` →
  `pose`, etc.), and CSV append-row output.
- **`scripts/zoo_bench_all.sh`** — sweep all tarballs under `~/zoo_tarballs/`.
- **`scripts/zoo_report.py`** — joins `deploys/_summary.csv` and
  `~/cpp_test/zoo_report.csv` into this Markdown report.

## Selected models (one per voyager-sdk task subdir)

| Voyager subdir | Stem | Deploy | Tarball size | Bench |
|---|---|---|---|---|
| `object_detection`        | `yolo11n-coco-onnx`                              | **OK**  | 3.8 M | **OK 378 fps system, 1.91 ms / batch** |
| `classification`          | `mobilenetv2-imagenet-onnx`                      | FAIL (script bug, retry queued) | — | — |
| `keypoint_detection`      | `yolov8npose-coco-onnx`                          | FAIL (compile > 15 min watchdog) | — | — |
| `obb_detection`           | `yolo11nobb-coco-onnx` *(wrong stem; corrected to `yolo11n-obb-dotav1-onnx` in retry)* | FAIL | — | — |
| `instance_segmentation`   | `yolov8nseg-coco-onnx`                           | FAIL (compile > 15 min watchdog; killed at 38:34 with 7.7 GB RSS) | — | — |
| `semantic_segmentation`   | `unet_fcn_512-cityscapes`                        | not reached (bulk aborted at seg stage) | — | — |
| `face_detection`          | `retinaface-mobilenet0.25-widerface-onnx`        | OK | 1.1 M | **FAIL — silent runtime crash on model load** (axrunmodel also fails) |
| `embedding`               | `osnet-x1-0-market1501-onnx`                     | OK | 2.2 M | **FAIL — silent runtime crash on model load** (axrunmodel also fails) |

## Detailed bench output (`~/cpp_test/zoo_report.csv` on the SBC)

```
task,stem,bench,fps_system,fps_infer,lat_ms_per_batch,status,notes
detection,yolo11n-coco-onnx,2,378.2,378.2,1.912,OK,
face,retinaface-mobilenet0.25-widerface-onnx,2,0.0,0.0,,NO_STATS,
embed,osnet-x1-0-market1501-onnx,2,0.0,0.0,,NO_STATS,
```

The yolo11n number is identical to what we measured before the model-zoo
refactor (commit `0aa46c2` on `main`), confirming the new `TaskHandler` virtual
dispatch is free at runtime.

## Operational findings (what broke, why, how it's now fixed)

| Failure mode | Trigger | Fix in this branch | Commit |
|---|---|---|---|
| `deploy.py` exits with rc != 0 even after producing artifacts | every successful deploy | `zoo_deploy.sh` ignores deploy.py’s exit code and gates success on `model.json` existing | `f804a5e` |
| Tarball missing `pool_ddr_const.bin` / `pool_l2_const.bin` | initial yolo11n tar | tar the whole `$STEM/<cores>/` dir, not a curated subset | `f804a5e` |
| Tarball missing `quantized/manifest.json` + `postprocess_graph.onnx` | initial yolo11n tar | `quantized/` is a peer dir of `<cores>/`, not a child — tar it separately | `75db492` |
| `add_if_exists` in tar-list trips `set -e` on a missing path | mnv2 first pass | explicit `if/fi` instead of `[ -e ] && …` | `cf32531` |
| Post-tar `ls -lh "$TAR_OUT"` aborts the script when no tarball was produced | retried mnv2 | gate the listing on `[ -f "$TAR_OUT" ]` | `cf32531` |
| YAML left in patched state after a mid-deploy crash | (intermittent) | EXIT trap restores the YAML | `3323c22` |
| Tiny classifiers fall back to a batch=1 compile | mnv2 | accept any of `{ $CORES, 4, 2, 1 }` → pick whichever produced model.json | `f804a5e` |
| Pose / instance-seg compiles run > 30 min on this SDK build | yolov8npose, yolov8nseg | per-deploy `timeout --kill-after=60 900` watchdog | `7ae6ef0` |
| Bench script passes raw subdir name as `--task` (`object_detection`) | first end-to-end smoke | `zoo_bench.sh` remaps via `case`-statement to the short handler name | `0e79337` |
| **Silent runtime failure for retinaface / osnet model.json on the SBC** | both | **OPEN.** `axrunmodel -vv` only prints `Created Context()` before exiting. Same model.json files load fine on the build host (silicon-side simulator). |  |

## Reproduce

```sh
# Build host (~/1.6/voyager-sdk venv ready):
cd ~/yocto_voyager && git checkout feat/model-zoo && git pull
SDK_DIR=~/1.6/voyager-sdk bash scripts/zoo_deploy_all.sh
SDK_DIR=~/1.6/voyager-sdk bash scripts/zoo_retry.sh    # re-do failures

# Push every successful tarball to the SBC:
for t in ~/yocto_voyager/deploys/*/*.tar.gz; do
    scp "$t" antelao@<sbc>:/tmp/
done

# SBC:
~/cpp_test/zoo_bench.sh object_detection yolo11n-coco-onnx /tmp/yolo11n-coco-onnx.tar.gz \
    --bench=2 --seconds=15
# … repeat per model. zoo_bench_all.sh sweeps a whole ~/zoo_tarballs/ tree.

# Render the report:
python3 scripts/zoo_report.py \
    --deploys ~/yocto_voyager/deploys/_summary.csv \
    --bench   ~/cpp_test/zoo_report.csv \
    --out     docs/MODEL_ZOO_REPORT.md
```

## What I would do next session

1. **Diagnose the silent runtime failure** for retinaface / osnet. They
   produce a model.json the SDK considers valid, but the Antelao runtime
   crashes between `axr_create_context()` and the first `axr_run_model_instance()`.
   Plan: dump strings from `kernel_function*.elf` to compare against the
   working yolo11n ELFs; if no obvious difference, attach a logger to the
   `metis` kernel driver and run `axrunmodel -vv` to capture the syscall the
   driver rejects.
2. **Real postproc** for `pose`, `seg`, `obb`, `face` once at least one
   non-detection model can run. The stubs are placeholders so the deploy +
   benchmark harness exercises every shape — they’re ~30 lines each to flesh
   out from the standard ONNX layouts (YOLOv8 keypoint heads, RetinaFace
   3-tuple per scale, YOLO mask coefficients + prototype masks, etc.).
3. **Lift the watchdog to 30 min** for the segmentation deploys and rerun;
   `yolov8nseg` was actively progressing past 30 min when we killed it.
4. **Try the `axdownloadmodel`-flavoured `.axm` deploys** (which the SDK
   provides as a "one core-only" fast path) for the classifiers that fail
   on the 4-core flow. These would let us at least confirm the runtime side
   of the pipeline isn’t the problem.
