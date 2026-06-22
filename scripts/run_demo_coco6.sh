#!/bin/sh
# run_demo_coco6.sh — 12-stream video-only demo on the attached HDMI monitor.
#
# NO live camera (the USB capture path added too much latency). All 12 streams
# are looping video files at 960x544:
#   5 military clips : tanks, soldiers_drones, drones, heli, ships
#   7 traffic clips  : traffic_1..4 (15s segments of traffic3),
#                      traffic_5..7 (20s segments of traffic4) — all distinct.
# Auto-arranged into a 4x3 grid, fullscreen, colour-coded boxes (no labels).
#
# Driven by the stock COCO yolo11n 4-core model (the military deploy SIGSEGVs
# this board's runtime — see ~/run_mil_demo.sh / the staged notes).
#
# Aggregate fps is capped by the AIPU (~224 fps); with 12 streams each tile runs
# ~18 fps. Stop from a 2nd SSH session with:  ~/stop_cam.sh
#
# Usage:
#   ~/run_demo_coco6.sh             # fullscreen
#   ~/run_demo_coco6.sh --windowed  # resizable window
#   ~/run_demo_coco6.sh <extra>     # extra flags pass through to the binary

set -e

CPP="$HOME/cpp_test"
MODEL="$HOME/yolo11n_4c/yolo11n-coco-onnx/yolo11n-coco-onnx/4/model.json"
VID="$CPP/mil_videos"
OUT="$CPP/multi_out/coco6"

export XDG_RUNTIME_DIR="/run/user/2001"
export WAYLAND_DISPLAY="wayland-0"
export GST_VIDEO_SINK="waylandsink"
unset DISPLAY

SCREEN="--fullscreen"
case "$1" in
    --windowed) SCREEN=""; shift ;;
esac

# 12 video streams (no usb:0). All are 960x544 so they match stream 0.
INPUTS="$VID/tanks.mp4,$VID/soldiers_drones.mp4,$VID/drones.mp4,$VID/heli.mp4,$VID/ships.mp4,$VID/traffic_1.mp4,$VID/traffic_2.mp4,$VID/traffic_3.mp4,$VID/traffic_4.mp4,$VID/traffic_5.mp4,$VID/traffic_6.mp4,$VID/traffic_7.mp4"

mkdir -p "$CPP/multi_out"
. "$HOME/axelera_pip/axelera-env/bin/activate"
cd "$CPP"

echo "launching 12 streams (no camera): 5 military + 7 traffic @ 960x544  — ~/stop_cam.sh to quit"
exec ./run.sh ./yolo_demo_multi \
    --model    "$MODEL" \
    --inputs   "$INPUTS" \
    --fps      25 \
    --out      "$OUT" \
    --display  1 $SCREEN --boxes-only \
    --workers  1 --bench 0 \
    "$@"
