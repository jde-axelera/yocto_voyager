#!/bin/sh
# run_mil_demo.sh — military YOLOv11n (9 classes) demo on the attached HDMI monitor.
#   stream 0     = live USB camera (/dev/video0)
#   streams 1..5 = the 5 Kontron military test clips (transcoded to 960x544 h264)
# 6 streams total, auto-arranged into a 3x2 grid, fullscreen, with colour-coded
# detection boxes (boxes only — no class-name text). Model = yolo11n military-only
# v2, the 1-core (single-kernel) deploy.
#
# All inputs share 960x544 (the binary requires every stream to match stream 0,
# and the camera is opened at 960x544). The military clips were transcoded from
# 1920x1080 .mov (one of them ProRes, which the gst pipeline cannot decode) to
# 960x544 h264 yuv420p @25fps.
#
# Press 'q' on the attached keyboard to quit gracefully. Ctrl-C / ~/stop_cam.sh
# also work.  Add --record to also write one MP4 per stream to multi_out/.
#
# Usage:
#   ~/run_mil_demo.sh              # fullscreen, 6 streams, no recording
#   ~/run_mil_demo.sh --windowed   # resizable window instead of fullscreen
#   ~/run_mil_demo.sh --record     # also write multi_out/mil_<i>.mp4
#   ~/run_mil_demo.sh <extra>      # extra flags pass through to the binary

set -e

CPP="$HOME/cpp_test"
MODEL="$HOME/mil_model/1/model.json"
VID="$CPP/mil_videos"
OUT="$CPP/multi_out/mil"
SIZE="960x544"

# Wayland session owned by the tty7 antelao login (Voyager uses Weston).
export XDG_RUNTIME_DIR="/run/user/2001"
export WAYLAND_DISPLAY="wayland-0"
export GST_VIDEO_SINK="waylandsink"
unset DISPLAY

# Fullscreen by default; --windowed drops it.
SCREEN="--fullscreen"
case "$1" in
    --windowed) SCREEN=""; shift ;;
esac

# stream 0 = camera, streams 1..5 = the five military clips.
INPUTS="usb:0,$VID/tanks.mp4,$VID/soldiers_drones.mp4,$VID/drones.mp4,$VID/heli.mp4,$VID/ships.mp4"

for f in tanks.mp4 soldiers_drones.mp4 drones.mp4 heli.mp4 ships.mp4; do
    [ -f "$VID/$f" ] || { echo "ERROR: $VID/$f not found" >&2; exit 1; }
done
[ -f "$MODEL" ] || { echo "ERROR: model $MODEL not found" >&2; exit 1; }

mkdir -p "$CPP/multi_out"
. "$HOME/axelera_pip/axelera-env/bin/activate"
cd "$CPP"

echo "launching 6 streams: 1 camera + 5 military clips @ $SIZE  — press 'q' to quit"
exec ./run.sh ./yolo_demo_multi \
    --model    "$MODEL" \
    --inputs   "$INPUTS" \
    --usb-size "$SIZE" \
    --fps      25 \
    --out      "$OUT" \
    --display  1 $SCREEN --boxes-only \
    --workers  1 --bench 0 \
    "$@"
