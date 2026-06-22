# yocto_voyager

YOLOv11n detection on the **Axelera Metis AIPU** (Antelao SoM · RK3588 · Voyager Linux 1.3.1) in C++.
Reads 1–10 H.264 streams or USB cameras, runs AIPU inference, and streams live to a monitor or laptop.

| 1 stream | 4 streams | 10 streams |
|---|---|---|
| ![](docs/images/hud_grid_n1.png) | ![](docs/images/hud_grid_n4.png) | ![](docs/images/hud_grid_n10.png) |

---

## This SBC — current state

| Item | Path / value |
|---|---|
| Binary | `~/yocto_voyager/build/yolo_demo_multi` |
| 4-core model | `~/yolo11n_4c/yolo11n-coco-onnx/yolo11n-coco-onnx/4/model.json` |
| Runtime venv | `~/axelera_pip/axelera-env/` |
| Metis driver | `1.4.16` (verified OK) |
| Test video | `/home/antelao/cpp_test/traffic3_640x480.mp4` |

> **Only the 4-core model works on this board.** The driver cannot allocate a single sub-device — see [Known issues](#known-issues).

Check the AIPU is free before running (it is single-tenant):
```sh
ps | grep yolo_demo_multi          # must be empty
/home/antelao/axelera_pip/axelera-env/bin/axdevice   # should show metis-0:1:0
```

Kill a stale process if needed:
```sh
kill $(ps | grep yolo_demo_multi | grep -v grep | awk '{print $1}')
```

---

## Run inference

All commands run from `~/yocto_voyager`. The `--out` flag is always required (even when you don't want the file — point it at `/tmp/discard`). **Use absolute paths in `--inputs`** — `~` does not expand inside comma-separated values.

### File output (no screen needed)

```sh
cd ~/yocto_voyager
sh scripts/05_run.sh ./build/yolo_demo_multi \
    --model   ~/yolo11n_4c/yolo11n-coco-onnx/yolo11n-coco-onnx/4/model.json \
    --inputs  /home/antelao/cpp_test/traffic3_640x480.mp4 \
    --out     ~/cpp_test/multi_out/test \
    --fps     30 --boxes-only
```

Output MP4 → `~/cpp_test/multi_out/test_0.mp4`. scp it to your laptop to watch.

### Local monitor (HDMI attached)

Set the Wayland env vars first — required even over SSH:

```sh
export XDG_RUNTIME_DIR="/run/user/2001"
export WAYLAND_DISPLAY="wayland-0"
export GST_VIDEO_SINK="waylandsink"
unset DISPLAY
```

Then run:

```sh
cd ~/yocto_voyager
sh scripts/05_run.sh ./build/yolo_demo_multi \
    --model   ~/yolo11n_4c/yolo11n-coco-onnx/yolo11n-coco-onnx/4/model.json \
    --inputs  /home/antelao/cpp_test/traffic3_640x480.mp4 \
    --out     /tmp/discard \
    --fps     30 --display 1 --fullscreen --boxes-only
```

### Remote viewing on your laptop (TCP stream)

Same Wayland env vars as above, then:

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
ssh -fNL 5050:localhost:5000 antelao@192.168.16.132   # 5050 avoids macOS AirPlay
ffplay -probesize 32M -analyzeduration 5M -framedrop tcp://localhost:5050
```

### Multiple streams

Use absolute paths separated by commas (`~` doesn't expand in CSV):

```sh
sh scripts/05_run.sh ./build/yolo_demo_multi \
    --model   ~/yolo11n_4c/yolo11n-coco-onnx/yolo11n-coco-onnx/4/model.json \
    --inputs  /home/antelao/cpp_test/traffic3_640x480.mp4,/home/antelao/cpp_test/traffic3_640x480.mp4,/home/antelao/cpp_test/traffic3_640x480.mp4,/home/antelao/cpp_test/traffic3_640x480.mp4 \
    --out     /tmp/discard \
    --fps     30 --display 1 --fullscreen --boxes-only
```

Grid auto-sizes: 4 streams → 2×2, 9 streams → 3×3, 10 streams → 4×3.

### USB camera (live)

```sh
sh scripts/05_run.sh ./build/yolo_demo_multi \
    --model    ~/yolo11n_4c/yolo11n-coco-onnx/yolo11n-coco-onnx/4/model.json \
    --inputs   usb:0 \
    --usb-size 640x480 \
    --fps      30 \
    --out      /tmp/discard \
    --display  1 --fullscreen --boxes-only
```

---

## Setup from scratch

Steps 1–4 run on a **build host** (x86_64 Ubuntu 22.04 + Voyager SDK 1.6). Step 5 deploys to the SBC.

> **Note:** The SBC has no `git` or `rsync` (BusyBox only). Transfer files with `scp` + `tar`.

```sh
# 1. Update Metis kernel driver  (SBC, as root — default root pw: AxeRoot2025)
su -c "sh scripts/01_update_driver.sh"   # reboots; target: /sys/class/metis/version ≥ 1.4.10

# 2. Install axelera-rt venv  (SBC, as antelao)
sh scripts/02_install_runtime.sh         # creates ~/axelera_pip/axelera-env/

# 3. Compile the model  (build host)
SDK_DIR=~/voyager-sdk-1.6 sh scripts/03_deploy_model.sh
# → ~/voyager-sdk-1.6/yolo11n_4core.tar.gz

# Ship to SBC:
scp ~/voyager-sdk-1.6/yolo11n_4core.tar.gz antelao@<sbc-ip>:~/
ssh antelao@<sbc-ip> 'mkdir -p ~/yolo11n_4c && tar xzf ~/yolo11n_4core.tar.gz -C ~/yolo11n_4c'
# model.json path: ~/yolo11n_4c/yolo11n-coco-onnx/yolo11n-coco-onnx/4/model.json

# 4. Cross-compile the binary  (build host)
sh scripts/04_build.sh                   # → build/yolo_demo_multi  (aarch64 ELF)

# Ship to SBC:
scp build/yolo_demo_multi antelao@<sbc-ip>:~/yocto_voyager/build/

# 5. Run  (SBC)
# → See "Run inference" section above
```

### Transfer the repo itself (no git on SBC)

```sh
# On build host / laptop:
git clone git@github.com:jde-axelera/yocto_voyager.git
tar czf yocto_voyager.tar.gz --exclude='.git' -C yocto_voyager .
scp yocto_voyager.tar.gz antelao@<sbc-ip>:~/
ssh antelao@<sbc-ip> 'mkdir -p ~/yocto_voyager && tar xzf ~/yocto_voyager.tar.gz -C ~/yocto_voyager'
```

---

## CLI reference

```
sh scripts/05_run.sh ./build/yolo_demo_multi --model PATH --inputs CSV --out PREFIX [options]
```

| Flag | Default | Meaning |
|---|---|---|
| `--model PATH` | required | path to `model.json` |
| `--inputs CSV` | required | comma-separated file paths or `usb:<N>`. **Use absolute paths** — `~` doesn't expand in CSV. All streams must share resolution. |
| `--out PREFIX` | required | output MP4 prefix → `<PREFIX>_0.mp4 … <PREFIX>_N.mp4`. Use `/tmp/discard` to skip keeping the file. |
| `--fps N` | `25` | per-stream FPS (capped by source native fps; use `--unpaced` to go faster) |
| `--usb-size WxH` | `640x480` | resolution for `usb:<N>` inputs |
| `--display MODE` | `0` | `0`=file only · `1`=local Wayland · `2`=TCP MPEG-TS on :5000 |
| `--fullscreen` | off | `--display 1` only: fullscreen Wayland window |
| `--boxes-only` | off | draw detection boxes (otherwise clean pass-through) |
| `--conf FLOAT` | `0.25` | confidence threshold |
| `--iou FLOAT` | `0.45` | NMS IoU threshold |
| `--bench MODE` | `0` | `0`=full · `1`=skip draw+write · `2`=preproc+infer only |
| `--workers N` | `1` | inference threads (keep at 1 for batch=4 model) |
| `--unpaced` | off | decode at host speed (benchmark mode) |
| `--connect-subdevs N` | auto | diagnostic: force sub-device count passed to `axr_device_connect` |

**Display modes 1 and 2 require Wayland env vars** (even over SSH):
```sh
export XDG_RUNTIME_DIR="/run/user/2001"
export WAYLAND_DISPLAY="wayland-0"
export GST_VIDEO_SINK="waylandsink"
unset DISPLAY
```

---

## Performance (4-core deploy, this SBC)

| Config | fps |
|---|---|
| AIPU silicon ceiling (`axrunmodel`) | **870 device / 543 system** |
| Single stream, full pipeline | **246** |
| 10 streams × 25 fps | **218 agg.** |
| 4 streams × 80 fps | **~305 agg.** |
| `--py-dispatch --bench 2 --unpaced`, 10 streams | **380 agg.** *(broken on SBC — see below)* |

**Latency** (single stream, 30 fps, `--display 1 --boxes-only`):

| Deploy | v4l2 → drawn | v4l2 → gst-stdin |
|---|---|---|
| 4-core (C API) | ~100 ms / ~180 ms max | ~110 ms / ~190 ms |
| 1-core (C API) | ~29 ms / ~50 ms | ~43 ms / ~67 ms — *broken on this board* |

The 4-core ~100 ms floor is the **batch-4 gather wait**: at 30 fps each frame waits ~3 × 33 ms for the 3 siblings before dispatch.

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

The display path is non-blocking (1-deep leaky slot). MP4 writers run in parallel. Frames are reordered per-stream so writers always see monotonic frame order even if the AIPU batch reorders them.

---

## Known issues

| Issue | Status |
|---|---|
| **1-core (batch=1) broken** — driver `zeContextCreateEx → NULL_POINTER` for single sub-device allocation. C-API and Python paths both affected. Use 4-core only. | Needs driver/firmware update from Amarula/Axelera |
| **`--py-dispatch` broken on SBC** — `libruntime2_core.so` / `axelera.runtime` also hits the same `NULL_POINTER` for any sub-device count. Works on x86_64 build host. | Same fix as above |
| **AIPU is single-tenant** — only one process at a time. New connection fails with `Fail to alloc ctx associate to N device` if a previous run is still alive. Kill it first. | Expected behaviour |
| librga 2.1.0 singleton-destroyed bug — RGA resize path unusable | Voyager 1.3.1 limitation |

---

## TODO

1. **Fix single sub-device allocation** — kernel driver returns `ZE_RESULT_ERROR_INVALID_NULL_POINTER` for `N=1` but not `N=4`. Likely an Amarula/Axelera firmware patch.
2. **Generalise to other task classes** — `TaskHandler` interface + per-task modules (`seg`/`classify`/`pose`/`obb`) on `feat/model-zoo`. ~50 lines each for the postproc decoders.
3. **Ping-pong input dmabufs** — overlap input prep with AIPU execute; the missing piece between 380 fps and the 543 fps `axrunmodel` ceiling. Blocked on fixing `--py-dispatch`.

### Resolved

- ~~`03_deploy_model.sh` tar command wrong (`-C build/yolo11n-coco-onnx` → single nesting, wrong paths)~~ Fixed to `-C build` → produces `yolo11n-coco-onnx/yolo11n-coco-onnx/<N>/model.json` matching README paths.
- ~~SDK/runtime ABI gap blocks batch=1 deploys~~ Was a bug in the old deploy script injecting `compilation_config: { aipu_cores_used: 4 }` into the YAML, conflicting with `--aipu-cores=1`. Script now leaves the YAML untouched.
- ~~Wipe `~/.cache/axelera/venvs/` before each deploy~~ Was a workaround for the YAML bug; no longer needed.

---

## License

MIT — see `LICENSE`.
