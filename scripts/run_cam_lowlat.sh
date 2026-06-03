#!/bin/sh
# run_cam_lowlat.sh — low-latency single USB-camera YOLOv11n demo on the
# attached HDMI monitor (fullscreen). Batch=1 deploy = lowest latency.
#
# Recording is OFF by default. Add --record to write the annotated MP4.
#
# Usage:
#   ~/run_cam_lowlat.sh                 # fullscreen, boxes drawn, no recording
#   ~/run_cam_lowlat.sh --windowed      # resizable window instead of fullscreen
#   ~/run_cam_lowlat.sh --record        # also write multi_out/cam_lowlat_0.mp4
#   ~/run_cam_lowlat.sh <extra flags>   # any extra flags pass through to the binary
#
# Quit gracefully (closes the window; finalizes the MP4 if --record):
#   press 'q' on the attached keyboard, or kill -INT $(pgrep yolo_demo_multi),
#   or Ctrl-C if you launched this in the foreground.

set -e

CPP="$HOME/cpp_test"
MODEL="$HOME/yolo11n_b1/yolo11n-coco-onnx/1/model.json"
OUT="$CPP/multi_out/cam_lowlat"

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

# Disk guard only matters when recording.
RECORD=0
for a in "$@"; do [ "$a" = "--record" ] && RECORD=1; done
if [ "$RECORD" -eq 1 ]; then
    AVAIL_MB=$(df -m "$HOME" | awk 'NR==2{print $4}')
    if [ "${AVAIL_MB:-0}" -lt 1024 ]; then
        echo "WARNING: only ${AVAIL_MB} MB free on \$HOME — recording may fill the disk." >&2
    fi
fi

mkdir -p "$CPP/multi_out"
. "$HOME/axelera_pip/axelera-env/bin/activate"
cd "$CPP"

# --out is harmless without --record (ignored); kept so '--record' works as a
# pass-through flag without the caller also having to supply --out.
exec ./run.sh ./yolo_demo_multi \
    --model    "$MODEL" \
    --inputs   usb:0 \
    --usb-size 640x480 \
    --fps      30 \
    --out      "$OUT" \
    --display  1 $SCREEN --boxes-only \
    --workers  1 --bench 0 \
    "$@"
