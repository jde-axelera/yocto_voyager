#!/usr/bin/env bash
# zoo_deploy_all.sh — serial deploy of one representative model per task class.
#
# Designed to run unattended on the build host. Each deploy is followed by an
# immediate cleanup of intermediate build/ so the next one has room. Logs each
# deploy result to deploys/_summary.csv.
#
# Usage (on build host):
#   SDK_DIR=~/1.6/voyager-sdk nohup bash scripts/zoo_deploy_all.sh > /tmp/zoo_deploy_all.log 2>&1 &
set -u

SDK_DIR="${SDK_DIR:-$HOME/1.6/voyager-sdk}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SUMMARY="$REPO_ROOT/deploys/_summary.csv"
mkdir -p "$REPO_ROOT/deploys"
if [ ! -f "$SUMMARY" ]; then
    echo "task,stem,status,tarball,size,duration_s,notes" > "$SUMMARY"
fi

# One representative model per task class. Pick the smallest / fastest variant
# to maximize the chance of a successful compile + benchmark inside our disk
# budget and the timeline.
JOBS=(
    "object_detection|yolo11n-coco-onnx|4"
    "classification|mobilenetv2-imagenet-onnx|1"
    "instance_segmentation|yolov8nseg-coco-onnx|4"
    "keypoint_detection|yolov8npose-coco-onnx|4"
    "obb_detection|yolo11nobb-coco-onnx|4"
    "semantic_segmentation|unet_fcn_512-cityscapes|4"
    "face_detection|retinaface-mobilenet0.25-widerface-onnx|4"
    "embedding|osnet-x1-0-market1501-onnx|4"
)

for j in "${JOBS[@]}"; do
    IFS='|' read -r TASK STEM CORES <<< "$j"
    if [ -f "$REPO_ROOT/deploys/$TASK/$STEM.tar.gz" ]; then
        echo "[skip] $TASK/$STEM already in deploys/"
        continue
    fi
    echo "=== [$(date +%H:%M:%S)] deploying $TASK / $STEM (cores=$CORES) ==="
    t0=$(date +%s)
    SDK_DIR="$SDK_DIR" bash "$REPO_ROOT/scripts/zoo_deploy.sh" "$TASK" "$STEM" --cores="$CORES"
    rc=$?
    t1=$(date +%s); dur=$((t1 - t0))
    tar="$REPO_ROOT/deploys/$TASK/$STEM.tar.gz"
    if [ "$rc" = 0 ] && [ -f "$tar" ]; then
        size=$(du -h "$tar" | awk '{print $1}')
        echo "$TASK,$STEM,OK,$tar,$size,$dur," >> "$SUMMARY"
        echo "=== [$(date +%H:%M:%S)] OK $STEM ($size, ${dur}s) ==="
    else
        notes=$(tail -3 /tmp/zoo_deploy_all.log 2>/dev/null | tr '\n' ' ' | tr ',' ';' | cut -c1-200)
        echo "$TASK,$STEM,FAIL,,,$dur,$notes" >> "$SUMMARY"
        echo "=== [$(date +%H:%M:%S)] FAIL $STEM (${dur}s, rc=$rc) ==="
    fi
    # Belt-and-braces clean of any leftover intermediate.
    rm -rf "$SDK_DIR/build/$STEM" 2>/dev/null || true
done

echo "=== ALL DONE [$(date +%H:%M:%S)] ==="
echo "Summary: $SUMMARY"
cat "$SUMMARY"
df -h ~ | tail -1
