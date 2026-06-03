#!/bin/sh
# run_multi_demo.sh — N-stream demo on the attached HDMI monitor:
#   stream 0       = live USB camera (/dev/video0)
#   streams 1..N-1 = the traffic3_640x480 clip (same file, opened N-1x, looped)
# Auto-arranged into a grid (cols=ceil(sqrt N), rows=ceil(N/cols)), fullscreen,
# with detection boxes. Uses the batch=4 (4-core) deploy for throughput.
#
# Recording is OFF by default (the h264_rkmpp encoders are a big CPU/disk cost).
# Add --record to write one MP4 per stream into multi_out/multi_<i>.mp4.
#
# N can be 1..40. The AIPU throughput ceiling is fixed (~3-500 fps aggregate),
# so beyond ~10 streams each stream just runs at a lower fps — a scaling demo.
#
# Press 'q' on the attached keyboard to quit gracefully. Ctrl-C / ~/stop_cam.sh
# also work. A per-component host-CPU breakdown prints when the run ends.
#
# Usage:
#   ~/run_multi_demo.sh                 # default N=10, fullscreen, no recording
#   ~/run_multi_demo.sh 20              # N=20 (1 cam + 19 videos)
#   ~/run_multi_demo.sh 40              # N=40 — stress test
#   ~/run_multi_demo.sh 6 --windowed    # N=6 in a resizable window
#   ~/run_multi_demo.sh 10 --record     # also write 10 MP4s
#   ~/run_multi_demo.sh 8 --fps 30      # extra flags pass through to the binary

set -e

CPP="$HOME/cpp_test"
MODEL="$HOME/yolo11n_4c/yolo11n-coco-onnx/yolo11n-coco-onnx/4/model.json"
VIDEO="traffic3_640x480.mp4"          # relative to $CPP
OUT="$CPP/multi_out/multi"

# Wayland session owned by the tty7 antelao login (Voyager uses Weston).
export XDG_RUNTIME_DIR="/run/user/2001"
export WAYLAND_DISPLAY="wayland-0"
export GST_VIDEO_SINK="waylandsink"
unset DISPLAY

# First positional arg, if a plain number, is the total stream count N.
N=10
case "$1" in
    ''|*[!0-9]*) ;;                   # absent or non-numeric -> keep default
    *) N="$1"; shift ;;
esac
if [ "$N" -lt 1 ] || [ "$N" -gt 40 ]; then
    echo "ERROR: N must be between 1 and 40 (got $N)" >&2; exit 1
fi

SCREEN="--fullscreen"
case "$1" in
    --windowed) SCREEN=""; shift ;;
esac

# Detect --record among the pass-through flags (for the disk guard + message).
RECORD=0
for a in "$@"; do [ "$a" = "--record" ] && RECORD=1; done

# 1 camera + (N-1) traffic streams, all 640x480 so they share resolution.
INPUTS="usb:0"
i=1
while [ "$i" -le $((N - 1)) ]; do INPUTS="$INPUTS,$VIDEO"; i=$((i + 1)); done

# Disk guard only matters when recording (N MP4 writers fill the overlay fast).
if [ "$RECORD" -eq 1 ]; then
    AVAIL_MB=$(df -m "$HOME" | awk 'NR==2{print $4}')
    if [ "${AVAIL_MB:-0}" -lt 2048 ]; then
        echo "WARNING: only ${AVAIL_MB} MB free on \$HOME — $N recorders may fill the disk." >&2
    fi
fi

mkdir -p "$CPP/multi_out"
if [ "$N" -gt 1 ] && [ ! -f "$CPP/$VIDEO" ]; then
    echo "ERROR: $CPP/$VIDEO not found (needed for N>1)" >&2; exit 1
fi
. "$HOME/axelera_pip/axelera-env/bin/activate"
cd "$CPP"

[ "$RECORD" -eq 1 ] && REC_MSG="recording ON" || REC_MSG="no recording"
echo "launching $N stream(s): 1 camera + $((N - 1))x $VIDEO  ($REC_MSG)  — press 'q' to quit"
# --out is harmless without --record (ignored); kept so '--record' works as a
# pass-through flag without the caller also having to supply --out.
exec ./run.sh ./yolo_demo_multi \
    --model     "$MODEL" \
    --inputs    "$INPUTS" \
    --usb-size  640x480 \
    --fps       25 \
    --out       "$OUT" \
    --display   1 $SCREEN --boxes-only \
    --workers   1 --bench 0 \
    "$@"
