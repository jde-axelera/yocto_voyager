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

# 3. Compile yolo11n                      (on the build host)
SDK_DIR=~/voyager-sdk-1.6 sh 03_deploy_model.sh                  # 4-core (throughput)
SDK_DIR=~/voyager-sdk-1.6 AIPU_CORES=1 sh 03_deploy_model.sh     # 1-core (low latency)
# → produces yolo11n_<N>core.tar.gz (~20 MB). scp to SBC, extract to ~/yolo11n_<N>c/.

# 4. Cross-compile the binary            (on the build host)
sh 04_build.sh                          # → build/yolo_demo_multi (aarch64 ELF)

# 5. Run                                 (on the SBC)
# --- 4-core deploy: throughput ---
sh 05_run.sh ./yolo_demo_multi \
    --model   ~/yolo11n_4c/yolo11n-coco-onnx/yolo11n-coco-onnx/4/model.json \
    --inputs  some_video.mp4 \
    --out     ~/cpp_test/multi_out/single \
    --fps     60 --display 2

# --- 1-core deploy: low latency (single stream / live UI) ---
sh 05_run.sh ./yolo_demo_multi \
    --model   ~/yolo11n_1c/yolo11n-coco-onnx/yolo11n-coco-onnx/1/model.json \
    --inputs  some_video.mp4 \
    --out     ~/cpp_test/multi_out/single \
    --fps     30 --display 1 --fullscreen --boxes-only
```

Only the `--model` path changes between the two deploys; everything else
in the binary is identical. The model.json's input batch dim drives
internal sub-device allocation, so the same binary handles both.

### Throughput vs. latency: pick your deploy

| Deploy | `AIPU_CORES` | Device throughput | In-process `v4l2 → drawn` latency | When |
|---|---|---|---|---|
| **4-core (batch=4)** | `4` (default) | ~870 fps device / ~246 fps via C API | ~100 ms mean | many streams, throughput-bound (e.g. 10× 25 fps) |
| **1-core (batch=1)** | `1` | ~135 fps device / ~90 fps via C API | **~25 ms mean** | single stream, latency-sensitive (camera, live UI) |

The 1-core deploy gives up ~63 % of the 4-core's aggregate throughput but cuts the batch-4 gather-wait floor (~75 ms at 25 fps) entirely — roughly **4× lower** end-to-end latency in display mode. The on-screen HUD shows the live mean as `Lat`.

#### Recipe: low-latency live camera on the SBC

End-to-end run, USB cam → AIPU → fullscreen Wayland window with detection
boxes and an on-screen `Lat` HUD. Verified on Antelao SoM / Voyager
Linux 1.3.1 with a stock USB UVC camera at 640×480.

```sh
# One-time: deploy a 1-core model on the build host, ship to SBC
SDK_DIR=~/voyager-sdk-1.6 AIPU_CORES=1 sh scripts/03_deploy_model.sh
scp ~/voyager-sdk-1.6/yolo11n_1core.tar.gz <user>@<sbc>:~/
ssh <user>@<sbc> 'mkdir -p ~/yolo11n_1c && tar xzf ~/yolo11n_1core.tar.gz -C ~/yolo11n_1c'

# On the SBC (assumes a Wayland session is running and $WAYLAND_DISPLAY is set;
# over SSH you may need: export XDG_RUNTIME_DIR=/run/user/$(id -u); export WAYLAND_DISPLAY=wayland-0)
sh scripts/05_run.sh ./yolo_demo_multi \
    --model    ~/yolo11n_1c/yolo11n-coco-onnx/yolo11n-coco-onnx/1/model.json \
    --inputs   usb:0 \
    --usb-size 640x480 \
    --fps      30 \
    --out      ~/cpp_test/multi_out/cam \
    --display  1 --fullscreen --boxes-only \
    --workers  1
```

Expected on the HUD after warmup: `E2E 30 fps · Infer 30 fps · Lat ~43 ms ·
CPU ~##% · MEM ~##%`. Total glass-to-glass with USB capture + gstreamer +
DRM commit + panel scan-out adds ~30-80 ms on top of `Lat`.

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
| `--py-dispatch` | off | route the AIPU call through `tools/aipu_worker.py` (Python side-car). Works at any `--bench` level; requires `--workers 1`. |
| `--py-worker PATH` | `tools/aipu_worker.py` | path to the side-car script |
| `-d, --display MODE` | `0` | `0`=file only, `1`=local Wayland, `2`=TCP MPEG-TS on :5000 |
| `--fullscreen` | off | `--display 1` only: ask waylandsink for a fullscreen window |
| `--boxes-only` | off | draw colour-coded detection boxes on each stream (otherwise streams pass through clean) |
| `-b, --bench MODE` | `0` | `0`=full pipeline, `1`=skip draw+write, `2`=preproc+infer only |
| `-w, --workers N` | `1` | inference instances (batch=4 deploy: keep at 1) |
| `--connect-subdevs N` | auto | force `axr_device_connect` to request N sub-devices. Default `batch*N`. Diagnostic knob for SBC kernel-driver / firmware quirks; leave unset unless `device_connect` is failing. |

---

## Performance (measured on this SBC, yolo11n 4-core deploy, batch=4)

| Configuration | fps | Notes |
|---|---|---|
| AIPU silicon ceiling (`axrunmodel --explore-latency`) | **870 dev / 543 system** | best config: `dblbuf, odmabuf=1` |
| **This repo, single stream, full pipeline (`--display 2` no viewer)** | **246** | baseline through the public C API |
| **This repo, 10 streams × 25 fps (paced)** | **218 agg.** | all 10 mp4s finalised cleanly |
| **This repo, 4 streams × 80 fps (paced)** | **~305 agg.** | input-paced; src clip is 60 fps native, ffmpeg duplicates |
| **This repo, `--py-dispatch --bench 2 --unpaced`, 10 streams** | **380 agg.** | side-car recovers the runtime's prefill/async pipeline; 70 % of axrunmodel ceiling |
| **This repo, single stream, 1-core deploy, `--bench 2 --unpaced`** | **~90** | batch=1 ELF; trades throughput for the latency win below |
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

### Latency (the other axis)

The `[lat]` stats line emitted every 2 s breaks down two in-process segments:

- **v4l2 → drawn** — V4L2/ffmpeg pipe-read to drawer thread finishing
  postproc + box draw. Covers preproc, batch-gather wait, AIPU dispatch,
  postproc, box draw.
- **v4l2 → gst-stdin** — same start point through to bytes being written
  into the `gst-launch` stdin pipe (display path inside our process).

The on-screen HUD also shows the live mean v4l2 → gst-stdin as **`Lat`**
(refreshed every second), so you can see the latency change in real time
when toggling deploys.

Measured on `usb:0` single-stream, 30 fps, `--display 1 --fullscreen --boxes-only`:

| Deploy / path | v4l2 → drawn | v4l2 → gst-stdin |
|---|---|---|
| **4-core**, C API (default) | ~100 ms mean / ~180 ms max | ~110 ms / ~190 ms |
| **4-core**, `--py-dispatch` | ~98 ms / ~150 ms | ~99 ms / ~161 ms |
| **1-core**, C API | **~29 ms / ~50 ms** | **~43 ms / ~67 ms** |

The 4-core floor (~95 ms) is the **batch=4 gather wait** at the worker: at
30 fps a batch waits ~3 × 33 ms for siblings 2-4 before it dispatches.
`--py-dispatch` shaves ~10 ms via the async prefill path, but most of the
visible latency is the gather wait, not the AIPU. The **1-core deploy
removes the gather wait entirely** (batch=1, the dispatch fires per
frame), trading the device-side throughput ceiling (~870 → ~135 fps) for
~4× lower latency.

Not included in those numbers: USB MJPEG capture + ffmpeg decode ahead of
the pipe (~15-30 ms), GStreamer plugin chain after our write + DRM commit
+ panel scan-out (~30-50 ms). Expect total glass-to-glass of **~75-130 ms
on the 1-core deploy** vs. ~150-200 ms on the 4-core deploy at 30 fps
single-stream paced input.

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
- `--py-dispatch` is **broken on the SBC** in Voyager Linux 1.3.1 — `libruntime2_core.so` / `axelera.runtime` is unable to bring up a device context (`zeContextCreateEx` → `NULL_POINTER`) for any sub-device count. Works fine on the x86_64 build host. The C-API path (`libaxruntime.so`) is unaffected; use it by default. Until this is fixed the Python side-car can't be used on the SBC.
- `--py-dispatch` requires `--workers 1`.
- `--fps N` is capped by the source file's native fps; use `--unpaced` to decode at host speed.
- Pick batch at deploy time, not at runtime — see *Throughput vs. latency* above. Mixing batch sizes in one process is not supported.
- librga 2.1.0 (Voyager 1.3.1) has a singleton-destroyed bug — RGA resize path unusable.
- NEON pack variant regresses to ~235 fps (memory-bandwidth bound). Scalar single-pass is already at the bw limit.
- No process-level auto-restart. Wrap in systemd for long-running.

---

## TODO

In rough priority order:

1. **Fix the SBC's Python `axelera.runtime` device-connect path.** On the
   SBC `ctx.device_connect()` (and therefore `axrunmodel` and
   `--py-dispatch`) fails for any sub-device count with a
   `zeContextCreateEx` NULL_POINTER. The C-API path through `libaxruntime.so`
   works for both 4-core and 1-core deploys, so this is specifically a
   `libruntime2_core.so` / Python-binding regression — not a kernel-ELF
   ABI mismatch. Likely needs a `pip install --upgrade axelera-rt
   axelera-runtime2` into the SBC's `~/axelera_pip/axelera-env/` once a
   newer wheel ships.
2. **Generalise to other task classes.** A `TaskHandler` interface + per-task
   modules (`tasks/detection.cpp` already in place, `classify`/`pose`/`seg`/
   `obb`/`face`/`embed` stubs ready) is on `feat/model-zoo`. Merge so the
   model zoo actually runs.
3. **Ping-pong input dmabufs.** The worker memcpys input N+1 *after* AIPU run
   N completes. Two input dmabufs alternating would overlap input prep with
   AIPU execute — the missing piece between 380 fps and `axrunmodel`'s 543 fps.
   (Blocked on #1: needs `--py-dispatch` for the prefill/async path.)
4. **Postproc decoders** for the non-detection task modules (currently stubs
   exercising preproc + inference only). ~50 lines each.

### Resolved

- ~~SDK/runtime ABI gap blocks fresh batch=1 deploys.~~ Actually a bug in
  `03_deploy_model.sh`: an earlier version patched the YAML to inject
  `compilation_config: { aipu_cores_used: 4 }` under `extra_kwargs`, which
  conflicts with `--aipu-cores=1` on the CLI and produces a kernel ELF
  that segfaults even on the build host's own `axrunmodel`. The script
  now leaves the YAML alone (and refuses to deploy if a stray
  `compilation_config:` block is present) — pass `AIPU_CORES=1
  sh 03_deploy_model.sh` for a clean 1-core deploy.
- ~~Wipe `~/.cache/axelera/venvs/` before each deploy.~~ Was a workaround
  for the YAML-patch bug above; no longer needed now that the YAML is
  never modified.

---

## License

MIT, see `LICENSE`.
