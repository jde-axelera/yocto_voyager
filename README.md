# yocto_voyager

YOLOv11n detection on the **Axelera Metis AIPU** (Antelao SoM · RK3588 · Voyager Linux 1.3.1) in C++.
Reads 1–10 H.264 streams or USB cameras, runs AIPU inference, and outputs annotated video to file, a local monitor, or a TCP stream to your laptop.

| 1 stream | 4 streams | 10 streams |
|---|---|---|
| ![](docs/images/hud_grid_n1.png) | ![](docs/images/hud_grid_n4.png) | ![](docs/images/hud_grid_n10.png) |

---

## What's in the repo

- `src/` — ~2 500 lines of C++ (preproc, postproc, DMA-heap pool, drawing, TTF, subprocess wiring)
- `scripts/01–05_*.sh` — five idempotent scripts that take a fresh SBC to a running demo
- `tools/aipu_worker.py` — Python AIPU side-car for `--py-dispatch` mode
- `toolchain-aarch64.cmake` + `src/CMakeLists.txt` — cross-build from any x86_64 Linux host

---

## Prerequisites

| Role | Requirement |
|---|---|
| **SBC (target)** | Antelao SoM · Voyager Linux 1.3.1 · kernel `6.1.148-rockchip-standard` · aarch64 |
| **Build host** | x86_64 Ubuntu 22.04 · Voyager SDK 1.6 (for `deploy.py` and cross-compiler) |

> The SBC has no `git` or `rsync` (BusyBox only). Transfer files with `scp` + `tar` — see step 0 below.

---

## Setup (fresh SBC → running inference)

### Step 0 — Get the repo onto the SBC

On your laptop / build host:

```sh
git clone git@github.com:jde-axelera/yocto_voyager.git
cd yocto_voyager
tar czf ../yocto_voyager.tar.gz --exclude='.git' .
scp ../yocto_voyager.tar.gz antelao@<sbc-ip>:~/
ssh antelao@<sbc-ip> 'mkdir -p ~/yocto_voyager && tar xzf ~/yocto_voyager.tar.gz -C ~/yocto_voyager'
```

---

### Step 1 — Update the Metis kernel driver (`01_update_driver.sh`)

> **Run on the SBC as root.** A fresh Voyager Linux 1.3.1 image ships driver v1.4.4; `axelera-rt ≥ 1.6.0` requires `≥ 1.4.10`. This script downloads the pre-built `.deb`, installs it, and reboots.

```sh
# On the SBC:
su                                        # default root password: AxeRoot2025
sh ~/yocto_voyager/scripts/01_update_driver.sh    # downloads, installs, reboots
```

Verify after reboot:
```sh
cat /sys/class/metis/version              # expect: 1.4.16
```

---

### Step 2 — Install the axelera-rt runtime venv (`02_install_runtime.sh`)

> **Run on the SBC as `antelao`.** Creates a Python venv at `~/axelera_pip/axelera-env/` and installs `axelera-rt` from the Axelera pip index. Runs `axdevice` as a smoke test.

```sh
# On the SBC:
sh ~/yocto_voyager/scripts/02_install_runtime.sh
```

Verify:
```sh
/home/antelao/axelera_pip/axelera-env/bin/axdevice
# expect: Device 0: metis-0:1:0  4GiB  metis-compute-board  clock=800MHz
```

---

### Step 3 — Compile the model (`03_deploy_model.sh`)

> **Run on the x86_64 build host** (needs the full Voyager SDK 1.6 with `deploy.py`). Runs `QUANTCOMPILE` on YOLOv11n-COCO and packages the result as a tar.gz. Only the **4-core deploy** currently works on the SBC — see [Known issues](#known-issues).

```sh
# On the build host:
SDK_DIR=~/voyager-sdk-1.6 sh scripts/03_deploy_model.sh
# → ~/voyager-sdk-1.6/yolo11n_4core.tar.gz  (~20 MB)
```

Ship to the SBC and extract:
```sh
scp ~/voyager-sdk-1.6/yolo11n_4core.tar.gz antelao@<sbc-ip>:~/
ssh antelao@<sbc-ip> 'mkdir -p ~/yolo11n_4c && tar xzf ~/yolo11n_4core.tar.gz -C ~/yolo11n_4c'
```

Model path on SBC: `~/yolo11n_4c/yolo11n-coco-onnx/yolo11n-coco-onnx/4/model.json`

---

### Step 4 — Cross-compile the binary (`04_build.sh`)

> **Run on the x86_64 build host.** Installs `aarch64-linux-gnu-g++` if needed, downloads the Axelera aarch64 runtime wheels to build a sysroot, then CMake-builds `yolo_demo_multi`. The result is a statically-linked aarch64 ELF — no compiler needed on the SBC.

```sh
# On the build host:
sh scripts/04_build.sh
# → build/yolo_demo_multi  (aarch64 ELF)
```

Ship to the SBC:
```sh
scp build/yolo_demo_multi antelao@<sbc-ip>:~/yocto_voyager/build/
```

---

### Step 5 — Run (`05_run.sh`)

> **Run on the SBC.** A thin wrapper that sets `LD_LIBRARY_PATH` to the axelera-rt venv's `.so` files, then `exec`s whatever you pass. The binary itself handles video decode, AIPU dispatch, drawing, and output.

Before every run, confirm the AIPU is free — it is **single-tenant** (one process at a time):

```sh
ps | grep yolo_demo_multi               # must be empty
# Kill a stale process if needed:
kill $(ps | grep yolo_demo_multi | grep -v grep | awk '{print $1}')
```

**Modes 1 and 2 (`--display 1/2`) require Wayland env vars** — set these in your shell first:

```sh
export XDG_RUNTIME_DIR="/run/user/2001"
export WAYLAND_DISPLAY="wayland-0"
export GST_VIDEO_SINK="waylandsink"
unset DISPLAY
```

> **`~` does not expand inside comma-separated `--inputs`.** Always use absolute paths there.

#### File output (no screen needed)

```sh
cd ~/yocto_voyager
sh scripts/05_run.sh ./build/yolo_demo_multi \
    --model   ~/yolo11n_4c/yolo11n-coco-onnx/yolo11n-coco-onnx/4/model.json \
    --inputs  /home/antelao/cpp_test/traffic3_640x480.mp4 \
    --out     ~/cpp_test/multi_out/test \
    --fps     30 --boxes-only
# output → ~/cpp_test/multi_out/test_0.mp4
```

#### Local monitor (HDMI attached)

```sh
sh scripts/05_run.sh ./build/yolo_demo_multi \
    --model   ~/yolo11n_4c/yolo11n-coco-onnx/yolo11n-coco-onnx/4/model.json \
    --inputs  /home/antelao/cpp_test/traffic3_640x480.mp4 \
    --out     /tmp/discard \
    --fps     30 --display 1 --fullscreen --boxes-only
```

#### Remote viewing on your laptop (TCP stream)

**SBC:**
```sh
sh scripts/05_run.sh ./build/yolo_demo_multi \
    --model   ~/yolo11n_4c/yolo11n-coco-onnx/yolo11n-coco-onnx/4/model.json \
    --inputs  /home/antelao/cpp_test/traffic3_640x480.mp4 \
    --out     /tmp/discard \
    --fps     30 --display 2 --boxes-only
```

**Laptop:**
```sh
ssh -fNL 5050:localhost:5000 antelao@<sbc-ip>   # 5050 avoids macOS AirPlay conflict
ffplay -probesize 32M -analyzeduration 5M -framedrap tcp://localhost:5050
```

#### Multiple streams

```sh
sh scripts/05_run.sh ./build/yolo_demo_multi \
    --model   ~/yolo11n_4c/yolo11n-coco-onnx/yolo11n-coco-onnx/4/model.json \
    --inputs  /home/antelao/cpp_test/traffic3_640x480.mp4,/home/antelao/cpp_test/traffic3_640x480.mp4,/home/antelao/cpp_test/traffic3_640x480.mp4,/home/antelao/cpp_test/traffic3_640x480.mp4 \
    --out     /tmp/discard \
    --fps     30 --display 1 --fullscreen --boxes-only
```

Grid auto-sizes: 4 streams → 2×2, 9 → 3×3, 10 → 4×3.

#### USB camera (live, low-latency)

The 1-core (batch=1) model removes the batch-gather wait and cuts latency from ~100 ms to ~29 ms — use it for live camera. See [Known issues](#known-issues) if the board doesn't support single sub-device allocation yet.

```sh
sh scripts/05_run.sh ./build/yolo_demo_multi \
    --model    ~/yolo11n_1c/yolo11n-coco-onnx/yolo11n-coco-onnx/1/model.json \
    --inputs   usb:0 \
    --usb-size 640x480 \
    --fps      30 \
    --out      /tmp/discard \
    --display  1 --fullscreen --boxes-only \
    --workers  1
```

Expected HUD after warmup: `E2E 30 fps · Infer 30 fps · Lat ~43 ms`. Total glass-to-glass (USB capture + GStreamer + DRM) adds ~30–80 ms on top.

---

## CLI reference

```
sh scripts/05_run.sh ./build/yolo_demo_multi --model PATH --inputs CSV --out PREFIX [options]
```

| Flag | Default | Meaning |
|---|---|---|
| `--model PATH` | required | path to `model.json` |
| `--inputs CSV` | required | comma-separated file paths or `usb:<N>`. **Use absolute paths** — `~` doesn't expand in CSV. All streams must share resolution. |
| `--out PREFIX` | required | output MP4 prefix → `<PREFIX>_0.mp4 … <PREFIX>_N.mp4`. Use `/tmp/discard` to throw away. |
| `--fps N` | `25` | per-stream target FPS (capped by source; use `--unpaced` to exceed) |
| `--usb-size WxH` | `640x480` | resolution for `usb:<N>` inputs |
| `--display MODE` | `0` | `0`=file only · `1`=local Wayland · `2`=TCP MPEG-TS on :5000 |
| `--fullscreen` | off | `--display 1` only: fullscreen Wayland window |
| `--boxes-only` | off | draw detection boxes (otherwise clean pass-through with HUD only) |
| `--conf FLOAT` | `0.25` | confidence threshold |
| `--iou FLOAT` | `0.45` | NMS IoU threshold |
| `--bench MODE` | `0` | `0`=full pipeline · `1`=skip draw+write · `2`=preproc+infer only |
| `--workers N` | `1` | inference threads (keep at 1 for batch=4 model) |
| `--unpaced` | off | decode at host speed (benchmark mode, ignores `--fps`) |
| `--py-dispatch` | off | route AIPU call through `tools/aipu_worker.py` *(broken on SBC — see Known issues)* |
| `--connect-subdevs N` | auto | diagnostic: force sub-device count passed to `axr_device_connect` |

---

## Performance (4-core deploy, this SBC, yolo11n)

| Config | fps |
|---|---|
| AIPU silicon ceiling (`axrunmodel --explore-latency`) | **870 device / 543 system** |
| Single stream, full pipeline (`--display 2`, no viewer) | **246** |
| 10 streams × 25 fps (paced) | **218 agg.** |
| 4 streams × 80 fps (paced) | **~305 agg.** |
| `--py-dispatch --bench 2 --unpaced`, 10 streams | **380 agg.** *(broken on SBC — see Known issues)* |

**Latency** (single stream, 30 fps, `--display 1 --boxes-only`):

| Deploy | v4l2 → drawn | v4l2 → gst-stdin |
|---|---|---|
| 4-core C API | ~100 ms mean / ~180 ms max | ~110 ms / ~190 ms |
| 1-core C API | ~29 ms / ~50 ms | ~43 ms / ~67 ms — *broken on this board* |

The 4-core ~100 ms floor is the batch-4 gather wait: at 30 fps each frame waits ~3 × 33 ms for its 3 siblings before dispatch.

---

## Architecture

```
decoders → raw_q → preproc threads → inst_q
                                       │
                    worker (batch-4 → axr_run_model_instance) → done_q
                                       │
                    drawer (reorder + postproc + box draw)
                              │
              ┌───────────────┴────────────────┐
              ▼                                ▼
      per-stream write_q               snapshot[i] mutex
      (h264_rkmpp mp4)                         │
                                display producer → LeakyOne → display consumer
                                                      (waylandsink / TCP MPEG-TS)
```

The display path is non-blocking (1-deep leaky slot drops frames if the consumer is slow). MP4 writers run in parallel. Frames are reordered per-stream so writers always see monotonic frame order even if the AIPU batch reorders them.

---

## Known issues

| Issue | Detail | Status |
|---|---|---|
| **1-core (batch=1) broken** | Driver `libaxldev_linux.c` returns `zeContextCreateEx → ZE_RESULT_ERROR_INVALID_NULL_POINTER` when allocating a single AIPU sub-device. Both C-API and Python paths are affected. Use 4-core only. | Needs driver/firmware update from Amarula/Axelera |
| **`--py-dispatch` broken on SBC** | `libruntime2_core.so` / `axelera.runtime` hits the same `NULL_POINTER` for any sub-device count. Works on x86_64 build host. | Same fix as above |
| **AIPU is single-tenant** | Only one process can hold the device at a time. A new connection fails with `Fail to alloc ctx associate to N device` if a previous run is still alive. Kill stale processes before starting. | Expected behaviour |
| **`~` doesn't expand in `--inputs` CSV** | Shell glob expansion only works on word boundaries. `--inputs ~/a.mp4,~/b.mp4` silently passes the literal `~` to ffmpeg, which fails. Use full absolute paths. | Known shell limitation |
| librga 2.1.0 singleton-destroyed bug | RGA resize path unusable on Voyager 1.3.1. | Upstream limitation |

---

## TODO

1. **Fix single sub-device allocation** — kernel driver returns `ZE_RESULT_ERROR_INVALID_NULL_POINTER` for `N=1` sub-devices but not `N=4`. Likely an Amarula/Axelera firmware patch for the Antelao SoM.
2. **Generalise to other task classes** — `TaskHandler` interface + per-task modules (`seg`/`classify`/`pose`/`obb`) are stubbed on `feat/model-zoo`. ~50 lines each for the postproc decoders.
3. **Ping-pong input dmabufs** — overlap input prep with AIPU execute; missing piece between 380 fps and the 543 fps `axrunmodel` ceiling. Blocked on fixing `--py-dispatch`.

### Resolved

- ~~`03_deploy_model.sh` tar wrong (`-C build/yolo11n-coco-onnx` → single-nesting, mismatched paths)~~ Fixed to `-C build` so tarball produces the double-nested `yolo11n-coco-onnx/yolo11n-coco-onnx/<N>/model.json` that matches the README paths.
- ~~SDK/runtime ABI gap blocks batch=1 deploys~~ Was a bug in the old deploy script injecting `compilation_config: { aipu_cores_used: 4 }` into the YAML, conflicting with `--aipu-cores=1`. Script now leaves the YAML untouched and refuses to deploy if a stray block is present.
- ~~Wipe `~/.cache/axelera/venvs/` before each deploy~~ Was a workaround for the YAML bug; no longer needed.

---

## License

MIT — see `LICENSE`.
