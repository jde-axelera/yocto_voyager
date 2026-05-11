# yocto_voyager

End-to-end **YOLOv11n inference pipeline** for the **Axelera Metis AIPU on an Antelao SoM (RK3588 + Voyager Linux)**, written in C++. Reads 1–10 H.264 input streams, paces each to a configurable FPS, runs object detection on the AIPU, draws bounding boxes + labels on every frame, and emits one annotated MP4 per stream **plus** an optional live auto-sized composite grid display (X11 locally or TCP MPEG-TS H.264 for remote viewing).

| 1 stream (1×1) | 4 streams (2×2) | 10 streams (4×3) |
|---|---|---|
| ![](docs/images/hud_grid_n1.png) | ![](docs/images/hud_grid_n4.png) | ![](docs/images/hud_grid_n10.png) |

The composite display grid is auto-sized from the stream count: `cols = ⌈√N⌉`, `rows = ⌈N/cols⌉` (same shape `voyager-sdk`'s `display.App` window uses). With one stream the composite is the full source frame; with more streams each cell shrinks proportionally and unused cells stay black. A single live HUD — **E2E fps · Infer fps · CPU %** — is overlaid once at the top-left of the composite (refreshed every second from `/proc/stat`), instead of one HUD per stream.

---

## What you get

- `src/yolo_demo_multi.cpp` — the only C++ source, ~1300 lines. Everything from earlier experiments has been removed and the surviving optimizations consolidated here.
- `scripts/01_update_driver.sh` ... `05_run.sh` — five short shell scripts that take you from a fresh SBC to a running demo.
- `src/CMakeLists.txt` + `toolchain-aarch64.cmake` — cross-build setup for any x86_64 Linux build host.
- This README, plus reproducible per-step notes and the full FPS/latency analysis below.

---

## Prerequisites

You need **two machines**:

1. **The target board** — an Antelao SoM (RK3588 + Metis AIPU on-module) running **Voyager Linux 1.3.1**.
   - `BOARD_TYPE=antelao-3588`, kernel `6.1.148-rockchip-standard`, aarch64.
   - Default non-root user (typically `antelao`) with `su` to root using the documented default root password.
   - Internet access (for fetching the kernel `.deb` and pip packages).
2. **An x86_64 Linux build host** — used to (a) cross-compile the C++ binary and (b) compile YOLOv11n for 4 AIPU cores using the full Voyager SDK.
   - Ubuntu 22.04 LTS works out-of-the-box. WSL2 also works.
   - The full **Voyager SDK 1.6** checkout from Axelera (the source you use for `deploy.py`). The pip `axelera-rt` package alone is not enough on the build host.

You don't need any of the following on the target SBC: a C/C++ compiler, kernel headers, GStreamer development libraries, OpenCV, or sudo. Everything that talks to the AIPU is delivered via pip wheels, and the C++ binary is cross-built on the build host.

---

## End-to-end quickstart

The repo is organised as five sequential steps. Each one is idempotent — you can re-run it. Most of them are tiny.

### Step 1 — Update the Metis kernel driver (on the SBC, as root)

A fresh image ships `metis.ko 1.4.4`. `axelera-rt 1.6.0` refuses to open the device unless the driver is `≥ 1.4.10`. The fix is a pre-built `.deb` published by Amarula that targets this exact kernel build, so no compilation is needed.

```sh
# copy this script onto the SBC (any way you like — scp, USB, paste through SSH, etc.)
# then on the SBC:
su        # default root password: AxeRoot2025
sh 01_update_driver.sh
```

The script downloads the kernel-module-metis `.deb`, remounts the rootfs read-write, `dpkg -i`s it, adds `metis` to `/etc/modules-load.d`, and reboots. After the reboot, check:

```sh
cat /sys/class/metis/version    # → 1.4.16
```

### Step 2 — Install the Axelera runtime (on the SBC, as the default user)

```sh
sh 02_install_runtime.sh
```

This creates `~/axelera_pip/axelera-env/` (a Python 3.10 venv) and pip-installs `axelera-rt` + its dependencies from the Axelera artifactory. The script also runs `axdevice` as a smoke test — you should see:

```
Device 0: metis-0:1:0 4GiB metis-compute-board flver=1.5.0 bcver=7.1 clock=800MHz(0-3:800MHz) mvm=0-3:100%
```

`axelera-devkit` (the compiler-side pip package) is **not** installed; its transitive dependency `onnxoptimizer` has no aarch64 wheel and there's no C++ host compiler on Voyager Linux to build it from source. Compilation is done off-board on the build host.

### Step 3 — Compile YOLOv11n for 4 AIPU cores (on the build host)

`axdownloadmodel yolo11n-coco-onnx --axm` returns a 1-AIPU-core deploy, which caps device throughput at ~135 fps. For the full ~870 fps device ceiling we need a 4-core build, which means setting `compilation_config.aipu_cores_used: 4` and `resources_used: 1.0` in the model YAML before running `deploy.py`.

On the build host (the script does this for you):

```sh
# Point SDK_DIR at your voyager-sdk-1.6 checkout
export SDK_DIR=~/voyager-sdk-1.6
sh 03_deploy_model.sh
```

This patches the YAML, runs `deploy.py --mode QUANTCOMPILE --aipu-cores=4`, and tars the resulting build directory to `yolo11n_4core.tar.gz` (~20 MB). Compile takes ~4 minutes.

(The pipeline's last step will print `Model(s): yolo11n-coco-onnx not deployed` — that's a downstream metadata check; the model.json + 4 kernel ELFs are fully usable.)

Copy the tarball to the SBC and extract:

```sh
# from the build host
scp yolo11n_4core.tar.gz <user>@<sbc-ip>:~/
# on the SBC
mkdir -p ~/yolo11n_4c && tar xzf ~/yolo11n_4core.tar.gz -C ~/yolo11n_4c
```

The model path on the SBC is now:

```
~/yolo11n_4c/yolo11n-coco-onnx/yolo11n-coco-onnx/4/model.json
```

### Step 4 — Cross-compile the demo (on the build host)

```sh
sh 04_build.sh
```

This script will, in order:

1. `apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu cmake` if you don't already have them.
2. Download the `axelera-runtime` + `axelera-runtime2` aarch64 wheels and extract them into `sysroot/axelera/` (so we have `libaxruntime.so`, headers, CMake configs, etc. for the cross-link).
3. Run CMake + make. Output is `build/yolo_demo_multi` (~1.8 MB, aarch64 ELF, statically linked libstdc++/libgcc).

Copy the binary to the SBC:

```sh
scp build/yolo_demo_multi <user>@<sbc-ip>:~/cpp_test/
```

### Step 5 — Run on the SBC

```sh
mkdir -p ~/cpp_test/multi_out
cd ~/cpp_test

# Single-stream example: 60 fps, file output only.
sh 05_run.sh ./yolo_demo_multi \
    --model   ~/yolo11n_4c/yolo11n-coco-onnx/yolo11n-coco-onnx/4/model.json \
    --inputs  ~/some_video.mp4 \
    --out     ~/cpp_test/multi_out/single \
    --fps     60
```

For 10 streams at 25 fps each, with composite TCP H.264 display on port 5000:

```sh
VIDEO=~/some_video.mp4
INPUTS=$(python3 -c "print(','.join(['$VIDEO']*10))")

sh 05_run.sh ./yolo_demo_multi \
    --model   ~/yolo11n_4c/yolo11n-coco-onnx/yolo11n-coco-onnx/4/model.json \
    --inputs  "$INPUTS" \
    --out     ~/cpp_test/multi_out/s \
    --fps     25 \
    --display 2
```

(Run `./yolo_demo_multi --help` for the full flag reference.)

The run prints a `[stats]` line every 2 s with per-stream and aggregate FPS. Stop with **Ctrl-C** — every output MP4 closes cleanly with a valid `moov` atom.

To view the live composite from a remote machine (e.g., a laptop), tunnel the SBC's TCP port 5000 to a local port and open it in `ffplay`:

```sh
# on the laptop (one-time per session)
ssh -fNL 5050:<sbc-ip>:5000 <user>@<sbc-ip>
# or, if the SBC isn't directly reachable, through a jump host:
ssh -fNL 5050:<sbc-ip>:5000 -J <user>@<jump-host> <user>@<sbc-ip>

# play (reconnect anytime — the streamer auto-respawns)
ffplay -fflags nobuffer -flags low_delay -probesize 100k tcp://localhost:5050
```

> **Note on macOS:** macOS's AirPlay Receiver listens on `tcp/5000`, so you must use a different local port (e.g. `5050`) for the SSH tunnel.

---

## CLI reference

```
./yolo_demo_multi --model PATH --inputs CSV --out PREFIX [options]
```

| Flag | Default | Meaning |
|---|---|---|
| `-m, --model PATH` | *required* | path to `model.json` or `.axm` (from `03_deploy_model.sh`) |
| `-i, --inputs CSV` | *required* | 1–10 comma-separated input mp4 paths (must share resolution) |
| `-o, --out PREFIX` | *required* | output mp4 path prefix → writes `<PREFIX>_0.mp4 ... <PREFIX>_N-1.mp4` |
| `--conf FLOAT` | `0.25` | detection confidence threshold |
| `--iou FLOAT` | `0.45` | NMS IoU threshold |
| `--fps N` | `25` | per-stream target FPS for `ffmpeg -re -r N` pacing |
| `--preproc N` | `4` | preprocess thread count |
| `-d, --display MODE` | `0` | `0`=file only, `1`=local X11 composite, `2`=TCP MPEG-TS composite on port 5000 |
| `-b, --bench MODE` | `0` | `0`=full pipeline, `1`=skip draw+write, `2`=preproc+infer only |
| `-w, --workers N` | `1` | inference instances; only meaningful for batch=1 deploys (warning is printed if `>1`) |
| `-h, --help` |  | print the full flag reference and exit |

The CLI is fully named — order of flags does not matter, and flags can be specified in either long (`--model PATH`) or short (`-m PATH`) form.

---

## How it works

Pipeline (single AIPU instance; the model is compiled for batch=4 and uses all 4 sub-devices in one call):

```
[stream 0..N-1]  →  decoders  →  raw_q  →  preproc threads  →  inst_q
                                                                   │
                                                                   ↓
                                          worker (batch-4 packing) → axr_run_model_instance
                                                                   ↓
                                                                done_q
                                                                   ↓
                                                          drawer (per-stream
                                                                   reorder map)
                                                                   ↓
                                                          per-stream write_q
                                                                   ↓
                                                          per-stream writer  →  h264_rkmpp mp4

Optional (live display, decoupled):
       drawer  →  per-stream snapshot mutex  →  display producer (30 Hz, 4×4 grid)
                                                       ↓
                                                LeakyOne<vector<uint8_t>>   (overwrite-newest, 1-deep)
                                                       ↓
                                                display consumer (blocking write_full → ffmpeg/gst-launch)
                                                       ↓
                                                ffmpeg h264_rkmpp → TCP   /   gst-launch ! autovideosink
```

Key design choices:

- **Batch-4 packing across streams.** The model is compiled for batch=4. Frames from any of the 10 streams are accumulated by the worker until it has 4, then packed into the model's NHWC input slot (4 × 1.68 MB) via memcpy into a single dma-heap dmabuf. Each output tensor is then sliced 4 ways and dispatched back to each frame.
- **`input_dmabuf=1` via `/dev/dma_heap/system`.** The worker holds a 4-slot input buffer mmap'd as both a dmabuf fd (passed to `axr_run_model_instance`) and a CPU pointer (written by preproc). `DMA_BUF_IOCTL_SYNC` flushes are issued around CPU writes.
- **`double_buffer=1`** in axruntime properties.
- **`output_dmabuf=0`** — the `axrunmodel --explore-latency` sweep shows host-allocated output buffers are fastest for our pipeline.
- **Per-stream reorder buffer in the drawer.** Each `Stream` keeps its own `pending` map keyed by per-stream frame index, so the writer for stream N receives frames strictly in order.
- **Display is fully decoupled from inference.** The drawer mutex-writes a snapshot of each stream's most recent annotated frame. A separate producer thread, paced at 30 Hz, decimates each snapshot by integer nearest-neighbour (factors `cols`/`rows`) and tiles it into an **auto-sized grid** chosen as `cols = ⌈√N⌉, rows = ⌈N/cols⌉` (same shape `voyager-sdk`'s `display.App` uses): N=1 → 1×1 full window, N=4 → 2×2, N=10 → 4×3, etc. Unused cells stay black. The composite is pushed through a **1-deep leaky slot** to a separate consumer thread that does atomic `write_full()` to the display subprocess and respawns it on viewer disconnect. Inference never blocks on the display path.
- **Overall HUD overlaid once on the composite.** The producer thread also draws a single status line — *E2E fps · Infer fps · CPU %* — at the top-left of the composite (not per stream). Counters are sampled once per second by diffing per-stream `drawn`, the global `frames_inferred`, and `/proc/stat`. This keeps small cells uncluttered when N is large.
- **TTF text rendering**: `stb_truetype.h` (single-header, public domain) + `LiberationSans-Bold.ttf` baked as a C array. Pre-rasterised glyph atlases at 14 px (labels) and 18 px (HUD). Per-glyph blits use 8-bit alpha against the BGR frame buffer.
- **Curated 80-class COCO palette** so the same class consistently gets the same colour.
- **Graceful shutdown**: `SIGINT`/`SIGTERM` triggers an orderly drain — decoders return, queues close, all writer stdin fds are closed in parallel so every ffmpeg child can finalise its `moov` atom simultaneously, then we wait up to 15 s before SIGTERM/SIGKILL.
- **`SIGPIPE` ignored** so the app survives the display subprocess dying when a viewer disconnects.

---

## FPS and latency

All measured on the same Antelao SoM, with the same 4-core `.axm` deploy.

| Path | Visible FPS | Latency / call | Comment |
|---|---|---|---|
| Pure AIPU device (`axrunmodel` `--explore-latency`, best config) | **~870** | 1.15 ms / frame | silicon ceiling |
| GStreamer + `axinferencenet` plugin (x86_64 host, marketing figure) | ~750 | ~1.3 ms | VAAPI dmabuf in, SDK plugin out |
| `axrunmodel` Python harness (best config: `dblbuf, odmabuf=0`) | **446 host / 390 system** | 4.7 / 9.0 ms | the reference C++ ceiling, via the runtime |
| **This repo — single stream, file output** | **~246** | 3.2 ms / batch-frame | what `yolo_demo_multi` gets |
| **This repo — 10 streams × 25 fps target** | **~218 aggregate (~22 per stream)** | 3.6 ms / batch-frame | 89 % of single-stream, all 10 mp4s clean |
| `axelera.runtime.op.seq()` (Python "Pythonic API") | **~13** | 78 ms | synchronous frame loop in Python; not for throughput |

### Why `axrunmodel` is faster than this C++ binary (~446 vs ~246)

Two reasons, both about *concurrency on a single model instance*:

1. **`libruntime2_core.so` exports zero dynamic symbols.** It is consumed only by a bundled Python extension (`_core.cpython-310-aarch64-linux-gnu.so`). The optimized C++ operator pipeline **and** the prefill / async dispatcher live behind that private interface and aren't reachable from a C++ program that links against the public `libaxruntime.so` alone.
2. **The public C `axr_run_model_instance` is strictly synchronous.** With a batch=4 model, only **one** instance can exist (the model claims all 4 AIPU sub-devices per call), so we can't fan out across multiple workers either. One inference at a time per worker thread ⇒ at most ~310 fps single-thread, and ~246 fps once we add postproc + draw + h264 mux.

`axrunmodel` calls into `libruntime2_core` through its private interface, keeps multiple inferences "prefilled" (its log even says `prefill=2, drop=2`), and amortises per-call setup time across them. That trick is not available through the public C API.

### Why GStreamer + `axinferencenet` is faster again (~750 vs ~446)

`axinferencenet` is a GStreamer plugin from the full SDK. It does:

- VAAPI/iGPU H.264 decode → letterbox → colour convert → quantise, all as **dmabuf** with **no host memcpy** anywhere on the input path.
- Same private prefill through `libruntime2_core`, plus its own buffer pool that keeps the AIPU's I/O queues saturated.
- Postproc (DFL + NMS) on the host using its own worker pool.

The result is a near-zero host CPU role on the critical path. The remaining 870 → 750 gap is postproc + sink overhead.

### Why `axinferencenet` is not in our build (and won't be on this image)

The plugin is shipped only as part of the full Voyager SDK install (the `install.sh` Ubuntu path) and the `axelera-voyager-sdk-base` apt package. On a Voyager Linux 1.3.1 SBC:

- `apt`/`dpkg` exist, but `/etc/apt/sources.list.d/` is empty (no Axelera apt source configured) and there's no `sudo` to add one.
- The pip `axelera-rt` package contains only the runtime libraries, not the GStreamer plugin.
- The pip `axelera-devkit` package (which would pull in everything) **fails to install** on this image: its transitive dependency `onnxoptimizer` has no aarch64 wheel and Voyager Linux ships no C++ host compiler to build it from source.
- The Voyager Linux image was built for runtime, not development (no `dev-pkgs`, no `tools-sdk`).

So on this image, **`~246 fps` is the practical ceiling reachable through the public C API**. Closing the gap to `axrunmodel`'s ~446 fps or GStreamer's ~750 fps would require either getting the dev-kit working (which a development-flavour Yocto image would allow) or moving the workload to an x86 host with the full SDK installed.

### Why `display=2` costs FPS only when a viewer is connected

| Display | E2E FPS (single stream) | What's happening |
|---|---|---|
| 0 (file only) | **246** | inference saturates; ffmpeg `h264_rkmpp` runs on the VPU asynchronously |
| 1 (local X11) | ~187 | gst-launch + autovideosink back-pressures the writer |
| 2 (TCP H.264, **viewer connected**) | ~187 | the streamer ffmpeg + TCP write on the consumer thread |
| 2 (TCP H.264, **no viewer**) | ~246 | the producer pushes into the leaky 1-slot; the consumer blocks harmlessly; the inference path is unaffected |

The 4×4 composite producer adds ~3 ms of CPU per 33 ms tick (< 1 % CPU), so the composite itself is essentially free.

---

## Outputs

Each annotated MP4 is encoded by `ffmpeg -c:v h264_rkmpp` (RK3588 VPU, near-zero CPU cost). At 848×480 / 60 fps the bitrate is ~1.7 Mbps. The composite (when display=2 is active) is the same size, also h264_rkmpp.

After a clean Ctrl-C the files have valid `moov` atoms and play immediately in `ffplay`, QuickTime, or VLC.

---

## Limitations and notes

- **All input streams must share the same resolution.** The drawer + composite assume a uniform per-stream cell size and a single `pre_ctx`.
- **The 4-core deploy is batch=4 only.** A `--no-double-buffer` config would give lower latency (~14 ms vs ~50 ms) at the cost of ~15 % throughput. We chose throughput.
- **Auto-respawning the display child** is a runtime feature only; the SBC app itself does not auto-restart if it crashes for some other reason. Wrap it in a systemd unit if you want a long-running service.
- **librga 2.1.0** on this image has a known singleton-destroyed bug; we tried using RGA as the resize+letterbox engine but it fails the first `improcess` call. The fallback is CPU-side single-pass scalar resize+pack, which is already memory-bandwidth bound and is fine.
- **NEON** doesn't help in this pipeline. We tried a two-pass scalar-resize-then-NEON-pack variant; it regresses to ~235 fps because the temp buffer doubles memory bandwidth, which is the real bottleneck. The single-pass scalar code in this binary is at the bw limit.

---

## Repo layout

```
.
├── README.md                        # this file
├── LICENSE
├── .gitignore
├── toolchain-aarch64.cmake          # cross-toolchain file for CMake
├── src/
│   ├── CMakeLists.txt
│   ├── yolo_demo_multi.cpp          # ~570-line orchestrator (argv parsing + thread launch)
│   │
│   │   ── pipeline modules ──
│   ├── concurrency.h                # BoundedQueue + LeakyOne (header-only templates)
│   ├── subprocess.h / .cpp          # Subprocess type + ffmpeg/gst-launch wiring
│   ├── dma_heap.h / .cpp            # InputBufferPool + dma-heap ABI + cache sync
│   ├── drawing.h / .cpp             # BGR draw primitives + class_color
│   ├── font.h / .cpp                # TTF rasterizer (stb_truetype, Liberation Sans Bold)
│   ├── yolo_preproc.h / .cpp        # letterbox + quantize into model input layout
│   ├── yolo_postproc.h / .cpp       # DFL + sigmoid + class-aware NMS
│   ├── frame.h                      # Frame and Stream structs
│   │
│   │   ── data ──
│   ├── coco_names.h                 # 80 COCO class names
│   ├── coco_palette.h               # curated 80-class colour palette
│   ├── liberation_sans_bold.h       # TTF baked as a C uint8 array
│   └── stb/stb_truetype.h           # public-domain TTF rasteriser (single header)
├── scripts/
│   ├── 01_update_driver.sh          # on SBC, as root: metis driver 1.4.4 -> 1.4.16
│   ├── 02_install_runtime.sh        # on SBC: pip-install axelera-rt into a venv
│   ├── 03_deploy_model.sh           # on build host: compile yolo11n with 4 AIPU cores
│   ├── 04_build.sh                  # on build host: cross-compile yolo_demo_multi
│   └── 05_run.sh                    # on SBC: convenience wrapper around the binary
└── docs/
    └── images/
        ├── hud_grid_n1.png               # composite, N=1 (1x1) with overall HUD
        ├── hud_grid_n4.png               # composite, N=4 (2x2) with overall HUD
        └── hud_grid_n10.png              # composite, N=10 (4x3) with overall HUD
```

### Code organization

All non-trivial logic lives in the dedicated modules listed above. The C++
binary is a single static library (`yolo_voyager_core`) plus a thin executable
(`yolo_demo_multi`) that ties them together. Each module:

- has a short header with one-line doc comments for every public symbol;
- exposes its API in the `yvm::` namespace;
- compiles in isolation against just its declared dependencies.

If you want to reuse any piece of the pipeline elsewhere (the dma-heap pool, the
TTF rasterizer, the YOLO decode/NMS, the subprocess wrappers), it's a drop-in
include + link against the static library.

---

## License

MIT, see `LICENSE`.
