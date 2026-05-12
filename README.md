# yocto_voyager

End-to-end **YOLOv11n inference pipeline** for the **Axelera Metis AIPU on an
Antelao SoM (RK3588 + Voyager Linux)**, in C++. Reads 1–10 H.264 streams (or
USB cameras), runs detection on the AIPU, writes one annotated MP4 per stream,
and optionally serves a composite display window (local Wayland or TCP MPEG-TS
H.264 for remote viewing).

| 1 stream (1×1) | 4 streams (2×2) | 10 streams (4×3) |
|---|---|---|
| ![](docs/images/hud_grid_n1.png) | ![](docs/images/hud_grid_n4.png) | ![](docs/images/hud_grid_n10.png) |

Composite grid auto-sizes from the stream count (`cols = ⌈√N⌉, rows = ⌈N/cols⌉`).
A single overall HUD — **E2E fps · Infer fps · CPU %** — is drawn once at the
top-left of the composite, sampled once a second from `/proc/stat`.

---

## What's in the repo

- `src/` — ~2 500 lines of C++ split into focused modules (preproc, postproc,
  dma-heap pool, drawing, TTF, subprocess wiring, Python AIPU side-car client).
- `scripts/01..05_*.sh` — five idempotent scripts from "fresh SBC" to "running demo".
- `tools/aipu_worker.py` — Python side-car invoked by `--py-dispatch` (recovers
  the runtime's prefill/async pipeline that the public C API doesn't expose).
- `src/CMakeLists.txt` + `toolchain-aarch64.cmake` — cross-build from any
  x86_64 Linux build host.

---

## Prerequisites

1. **Target board** — Antelao SoM running Voyager Linux 1.3.1
   (`BOARD_TYPE=antelao-3588`, kernel 6.1.148-rockchip-standard, aarch64).
2. **Build host** — x86_64 Ubuntu 22.04 + Voyager SDK 1.6 checkout (for `deploy.py`).

The target SBC needs no C/C++ compiler, kernel headers, GStreamer dev, OpenCV,
or sudo. Everything that talks to the AIPU is delivered via pip wheels.

---

## Quickstart (five steps)

```sh
# 1. Update Metis kernel driver to ≥ 1.4.10  (on the SBC, as root)
sh 01_update_driver.sh                  # reboots; check /sys/class/metis/version

# 2. Install axelera-rt venv             (on the SBC, as the default user)
sh 02_install_runtime.sh                # creates ~/axelera_pip/axelera-env/

# 3. Compile yolo11n for 4 AIPU cores    (on the build host)
SDK_DIR=~/voyager-sdk-1.6 sh 03_deploy_model.sh
# → produces yolo11n_4core.tar.gz (~20 MB). scp to SBC, extract to ~/yolo11n_4c/.

# 4. Cross-compile the binary            (on the build host)
sh 04_build.sh                          # → build/yolo_demo_multi (aarch64 ELF)

# 5. Run                                 (on the SBC)
sh 05_run.sh ./yolo_demo_multi \
    --model   ~/yolo11n_4c/yolo11n-coco-onnx/yolo11n-coco-onnx/4/model.json \
    --inputs  some_video.mp4 \
    --out     ~/cpp_test/multi_out/single \
    --fps     60 --display 2
```

**USB cameras** — any `--inputs` entry starting with `usb:<N>` opens
`/dev/video<N>` over V4L2 (MJPEG → BGR24). `--usb-size WxH` sets resolution
(default `640x480`). Mix files and cameras freely as long as they share resolution.

**Remote viewing** — `--display 2` exposes an MPEG-TS H.264 listener on TCP/5000.
From your laptop:

```sh
ssh -fNL 5050:localhost:5000 <user>@<sbc-ip>     # macOS: avoid local :5000 (AirPlay)
ffplay -probesize 32M -analyzeduration 5M -framedrop tcp://localhost:5050
```

---

## CLI reference

```
./yolo_demo_multi --model PATH --inputs CSV --out PREFIX [options]
```

| Flag | Default | Meaning |
|---|---|---|
| `-m, --model PATH` | *required* | path to `model.json` or `.axm` |
| `-i, --inputs CSV` | *required* | 1–10 inputs; entry is a file path or `usb:<N>`. All streams must share resolution. |
| `-o, --out PREFIX` | *required* | output mp4 prefix → `<PREFIX>_0.mp4 … <PREFIX>_N-1.mp4` |
| `--conf FLOAT` | `0.25` | detection confidence threshold |
| `--iou FLOAT` | `0.45` | NMS IoU threshold |
| `--fps N` | `25` | per-stream target FPS. For files: `ffmpeg -re -r N` pacing. For `usb:<N>`: device capture rate. |
| `--usb-size WxH` | `640x480` | capture resolution for any `usb:<N>` entry |
| `--unpaced` | off | drop ffmpeg `-re` (decode at host speed, benchmark mode) |
| `--py-dispatch` | off | route the AIPU call through `tools/aipu_worker.py` (Python side-car). Requires `--bench 2` and `--workers 1`. |
| `--py-worker PATH` | `tools/aipu_worker.py` | path to the side-car script |
| `-d, --display MODE` | `0` | `0`=file only, `1`=local Wayland, `2`=TCP MPEG-TS on :5000 |
| `--fullscreen` | off | `--display 1` only: ask waylandsink for a fullscreen window |
| `--boxes-only` | off | draw colour-coded detection boxes on each stream (otherwise streams pass through clean) |
| `-b, --bench MODE` | `0` | `0`=full pipeline, `1`=skip draw+write, `2`=preproc+infer only |
| `-w, --workers N` | `1` | inference instances (batch=4 deploy: keep at 1) |

---

## Performance (measured on this SBC, yolo11n 4-core deploy, batch=4)

| Configuration | fps | Notes |
|---|---|---|
| AIPU silicon ceiling (`axrunmodel --explore-latency`) | **870 dev / 543 system** | best config: `dblbuf, odmabuf=1` |
| **This repo, single stream, full pipeline (`--display 2` no viewer)** | **246** | baseline through the public C API |
| **This repo, 10 streams × 25 fps (paced)** | **218 agg.** | all 10 mp4s finalised cleanly |
| **This repo, 4 streams × 80 fps (paced)** | **~305 agg.** | input-paced; src clip is 60 fps native, ffmpeg duplicates |
| **This repo, `--py-dispatch --bench 2 --unpaced`, 10 streams** | **380 agg.** | side-car recovers the runtime's prefill/async pipeline; 70 % of axrunmodel ceiling |
| `axelera.runtime.op.seq()` (Python sync API) | 13 | not for throughput |

Single-call latency through `axr_run_model_instance`: **~3.2 ms / batch-frame**.

### Why the public C API caps around 246 fps

1. `libruntime2_core.so` exports zero dynamic symbols — only the bundled Python
   `_core` extension can reach the prefill / async dispatcher. From C++ via
   `libaxruntime.so` you can only do one in-flight call at a time.
2. With batch=4 a single instance already claims all 4 AIPU sub-devices, so
   fanning out workers doesn't help either.

`--py-dispatch` works around this by spawning `tools/aipu_worker.py`, handing
it the input + output dma-buf fds over `SCM_RIGHTS`, and signalling one byte
per batch. The Python side calls `axelera.runtime.ModelInstance.run`, which
goes through `_core` and exercises the same prefill path `axrunmodel` does.

---

## Architecture (one-paragraph version)

```
decoders → raw_q → preproc threads → inst_q
                                       │
                                       ▼
                worker (batch-4 pack → axr_run_model_instance) → done_q
                                       │
                                       ▼
                drawer (per-stream reorder + postproc + box draw)
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
        per-stream write_q          snapshot[i] mutex
        (h264_rkmpp mp4)                   │
                                           ▼
                              display producer @ 30 Hz → LeakyOne (1-deep)
                                           ▼
                              display consumer → waylandsink / ffmpeg TCP MPEG-TS
```

Inference never blocks on the display path (1-deep leaky slot drops frames if
the consumer is slow). Per-stream MP4 writers run in parallel. The drawer
keeps a per-stream `pending` map so writers see frames in increasing order
even if the AIPU batch reorders them.

---

## Limitations

- All input streams must share resolution.
- `--py-dispatch` requires `--bench 2` (no postproc / draw / mp4 write while on).
- `--fps N` is capped by the source file's native fps; use `--unpaced` to decode at host speed.
- batch=4 only. Going to batch=1 trades ~15 % throughput for ~70 % lower latency.
- Fresh `deploy.py` outputs may be silently rejected by the SBC runtime (SDK/runtime version skew). The build host's `~/.cache/axelera/venvs/<hash>/` can bind a deploy to an older cached compiler when `compilation_config:` is in the YAML — pass `--aipu-cores` via the CLI and wipe the cache. Even with the right compiler, the SBC's `axelera-runtime 1.6.0 + axelera-runtime2 0.1.8` doesn't accept current SDK 1.6 ELFs. See [`docs/MODEL_ZOO_REPORT.md` on `feat/model-zoo`](https://github.com/jde-axelera/yocto_voyager/blob/feat/model-zoo/docs/MODEL_ZOO_REPORT.md).
- librga 2.1.0 (Voyager 1.3.1) has a singleton-destroyed bug — RGA resize path unusable.
- NEON pack variant regresses to ~235 fps (memory-bandwidth bound). Scalar single-pass is already at the bw limit.
- No process-level auto-restart. Wrap in systemd for long-running.

---

## TODO

In rough priority order:

1. **Resolve the SDK / runtime ABI gap** so freshly compiled models actually
   run on the SBC. Two paths: `pip install --upgrade axelera-runtime
   axelera-runtime2` into the SBC's `~/axelera_pip/axelera-env/` (closest to
   what fresh SDK builds expect), or roll the build host's compiler back to
   match what produced `~/yolo11n_4c/` (commit `e98f11f` of voyager-sdk
   1.6, but with the exact pip wheelset from that day).
2. **Wipe `~/.cache/axelera/venvs/` before each deploy** (or stop patching
   the YAML and pass `--aipu-cores` only on the CLI). Avoids the cache-poisoned
   1.5.3 compiler trap.
3. **Lift the `--py-dispatch` → `--bench 2` restriction.** Map the C++-owned
   output dma-bufs into the postproc path so full-pipeline runs can also use
   the side-car. Already prototyped on the
   [feat/model-zoo branch](https://github.com/jde-axelera/yocto_voyager/tree/feat/model-zoo).
4. **Generalise to other task classes.** A `TaskHandler` interface + per-task
   modules (`tasks/detection.cpp` already in place, `classify`/`pose`/`seg`/
   `obb`/`face`/`embed` stubs ready) is on `feat/model-zoo`. Merge once #1 is
   resolved so the model zoo actually runs.
5. **Ping-pong input dmabufs.** Currently the worker memcpys input N+1 into
   the dmabuf *after* AIPU run N completes. Two input dmabufs alternating
   would overlap input prep with AIPU execute — the missing piece between
   tonight's 380 fps and `axrunmodel`'s 543 fps system.
6. **Postproc decoders** for the non-detection task modules (currently
   stubs that exercise preproc + inference only). ~50 lines each from the
   standard ONNX layouts; gated on #1.

---

## License

MIT, see `LICENSE`.
