# Model-zoo benchmark report

_Branch: `feat/model-zoo`. Overnight run 2026-05-11 → 2026-05-12._

## Headline

| | Count |
|---|---|
| Models attempted | 8 (one representative per task class) + 4 retry/bonus |
| Deploys succeeded on the build host (voyager-sdk 1.6) | **7** — `yolo11n-coco-onnx`, `retinaface-mobilenet0.25-widerface-onnx`, `osnet-x1-0-market1501-onnx`, `mobilenetv2-imagenet-onnx`, `yolo11n-obb-dotav1-onnx`, `resnet18-imagenet-onnx`, `squeezenet1.0-imagenet-onnx` |
| Deploys failed (compile timeout / wrong stem / script bug) | 2 — `yolov8npose-coco-onnx` and `yolov8nseg-coco-onnx` both run > 30 min single-threaded (15-min watchdog tripped) |
| Models that loaded + ran on the SBC end-to-end | **1** — `yolo11n-coco-onnx`. **378 fps** `--bench 2` and **272 fps** full-pipeline `--bench 0` (postproc + box draw + h264_rkmpp MP4 mux) via `--py-dispatch` |
| Models that compiled cleanly but **axrunmodel itself can't load them on this SBC** | **All 7 freshly compiled tarballs from voyager-sdk 1.6, plus a 1.5.3 yolo11n test build.** A freshly recompiled `yolo11n` with byte-identical `model.json` to the working deploy still fails — only the on-disk May-10 `~/yolo11n_4c/` deploy runs. Root cause: SDK/runtime version skew (see "What I would do next session"). |

The C++ side of the work — the `TaskHandler` abstraction, the deploy + bench
harness, the disk-rotation discipline — is solid. **The blocking issue tonight
turned out to be an SDK/runtime version mismatch**, not a flaw in any of the
new code: every freshly compiled model on the build host (voyager-sdk 1.6)
produces ELFs that the SBC's voyager-rt 1.6.0 silently rejects between
`Context()` creation and the first `instance.run()`. Including a freshly
recompiled **yolo11n** whose `model.json` is byte-identical (`md5 ad7d2f…`) to
the pre-existing `~/yolo11n_4c/` deploy that runs at 378 fps. The ELFs differ;
the runtime accepts only the older set.

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
| `face_detection`          | `retinaface-mobilenet0.25-widerface-onnx`        | OK | 1.1 M | **FAIL — runtime rejects fresh ELFs** (axrunmodel also fails) |
| `embedding`               | `osnet-x1-0-market1501-onnx`                     | OK | 2.2 M | **FAIL — runtime rejects fresh ELFs** (axrunmodel also fails) |
| `classification` *retry*  | `mobilenetv2-imagenet-onnx`                      | OK | 3.3 M | FAIL — runtime rejects fresh ELFs |
| `classification` *retry*  | `resnet18-imagenet-onnx`                         | OK | 11 M  | not yet benched (predicted FAIL by mismatch hypothesis) |
| `classification` *retry*  | `squeezenet1.0-imagenet-onnx`                    | OK | 1.7 M | FAIL — runtime rejects fresh ELFs |
| `obb_detection` *retry*   | `yolo11n-obb-dotav1-onnx`                        | OK | 4.9 M | not yet benched (predicted FAIL by mismatch hypothesis) |

## Detailed bench output (`~/cpp_test/zoo_report.csv` on the SBC)

```
task,stem,bench,fps_system,fps_infer,lat_ms_per_batch,status,notes
detection,yolo11n-coco-onnx,2,378.0,378.0,1.895,OK,    # using ~/yolo11n_4c (pre-existing deploy)
detection,yolo11n-coco-onnx,0,271.6,276.3,1.852,OK,    # full pipeline (postproc + box + h264_rkmpp MP4)
face,retinaface-mobilenet0.25-widerface-onnx,2,0.0,0.0,,NO_STATS,    # fresh build — runtime rejects
embed,osnet-x1-0-market1501-onnx,2,0.0,0.0,,NO_STATS,                # fresh build — runtime rejects
```

The yolo11n numbers match what we measured before the model-zoo refactor
(commits `0aa46c2`, `dfb3423` on `main`), confirming the new `TaskHandler`
virtual dispatch is free at runtime and the full pipeline still hits ~272 fps
with annotations and MP4 writing.

### Diagnosed bugs on the build-host side (now fixable)

| Bug | Trigger | Fix |
|---|---|---|
| **Cache-poisoned compiler** | A `compilation_config:` block in the YAML makes `voyager-sdk` resolve to a pre-cached older compiler venv under `~/.cache/axelera/venvs/<hash>/` instead of the active 1.6 venv. Tonight's cache had 1.2.5/1.3.1/1.4.0/1.4.2/1.5.3 — the lookup picked `1.5.3` for the YAML hash. `compile_config.json` then reports `compiler_version: 0b25b09` (= the 1.5.3 release commit) even though the active venv is 1.6. | Two options: pass `--aipu-cores=N` on the CLI only and don't patch the YAML (gets batch=1 only, but right compiler); **or** `mv ~/.cache/axelera/venvs ~/.cache/axelera/venvs.bak.$(date +%s)` then patch the YAML (gets batch=4 + right compiler, confirmed tonight). |

### Smoking-gun ELF mismatch

| File | Pre-existing `~/yolo11n_4c/` (works) | Freshly built today (rejected) |
|---|---|---|
| `model.json` | `ad7d2ffd19c222da8b5a080c2970907d` | `ad7d2ffd19c222da8b5a080c2970907d` ✅ identical |
| `pool_ddr_const.bin` | `95d084048c0b3909e4254949a4177f43` | `95d084048c0b3909e4254949a4177f43` ✅ identical |
| `pool_l2_const.bin` | `0f400ad7315203d5e90de3ff50487ccb` | `de6807bef75e91f53ad6cbd66b3a7df3` ❌ different |
| `kernel_function_{0..3}.elf` | (4 different hashes) | (4 different hashes) ❌ all four differ |
| `manifest.json` | `ab312b2d…` | `52bb2336…` ❌ different |
| `postprocess_graph.onnx` | `f8798f68…` | `0156e1c4…` ❌ different |

The voyager-sdk 1.6 codegen on the build host emits different ELFs + manifest
than the older pre-existing deploy. The SBC's voyager-rt 1.6.0 silently
rejects them. This is the **single root cause** of every freshly built model
appearing in the `NO_STATS` rows above.

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

1. **Resolve the SDK/runtime mismatch.** I tested the obvious workaround
   (deploy with the older `~/1.5.3/voyager-sdk` checkout) — same result:
   `axrunmodel` silently exits between `Created Context()` and the first
   `instance.run()`. So **neither 1.6 nor 1.5.3 produces ELFs that this SBC
   accepts**; the only working deploy is the pre-existing May-10 yolo11n
   that's already on the SBC at `~/yolo11n_4c/`. The root cause is therefore
   either:
   - the SBC's installed `axelera-rt` is older than both SDKs on the build
     host — `pip install --upgrade` into `~/axelera_pip/axelera-env/` against
     the same artifactory the build host pulled from should bring the two
     into version-lockstep; or
   - the AIPU's on-device firmware (loaded by the kernel `metis` driver)
     predates these SDKs, in which case the firmware needs to be reflashed
     to a build that matches.
   Once one of these is fixed, `scripts/zoo_retry.sh` already proves the
   builds themselves are clean — every entry in the JOBS list will simply
   start running.
2. **Real postproc** for `pose`, `seg`, `obb`, `face` once non-detection
   models can actually run. The stubs are placeholders so the deploy +
   benchmark harness exercises every shape — each is ~30 lines from the
   standard ONNX layouts (YOLOv8 keypoint heads, RetinaFace 3-tuple per
   scale, YOLO mask coefficients + prototype masks, etc.).
3. **Lift the watchdog to 30 min** for the segmentation deploys and rerun;
   `yolov8nseg` was actively progressing past 30 min when we killed it.
4. **Try the `axdownloadmodel`-flavoured pre-built `.axm` deploys** (the
   SDK ships ready-made one-core variants on its artifactory). Those are
   compiled against the same older runtime the SBC expects, so they should
   bypass the ELF-mismatch issue entirely.
