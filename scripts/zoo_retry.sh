#!/usr/bin/env bash
# zoo_retry.sh — re-deploy the models that failed in the first bulk pass,
# either because the deploy script had a bug or the JOB list had a wrong stem.
#
# Idempotent: deletes the stale tarball before re-running deploy.sh.
# Skip pose / instance-seg by default since they time out on this SDK build.
set -u

SDK_DIR="${SDK_DIR:-$HOME/1.6/voyager-sdk}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

RETRY=(
    "classification|mobilenetv2-imagenet-onnx|1"   # was: rc=1, script bug
    "obb_detection|yolo11n-obb-dotav1-onnx|4"      # was: wrong stem (typo)
    "classification|resnet18-imagenet-onnx|4"      # bonus: try a 4-core classifier
    "classification|squeezenet1.0-imagenet-onnx|4" # bonus: smallest classifier
)

for j in "${RETRY[@]}"; do
    IFS='|' read -r TASK STEM CORES <<< "$j"
    rm -f "$REPO_ROOT/deploys/$TASK/$STEM.tar.gz" 2>/dev/null
    echo "=== retry $TASK / $STEM (cores=$CORES) ==="
    SDK_DIR="$SDK_DIR" timeout --kill-after=60 900 \
        bash "$REPO_ROOT/scripts/zoo_deploy.sh" "$TASK" "$STEM" --cores="$CORES"
    rc=$?
    if [ "$rc" = 0 ] && [ -f "$REPO_ROOT/deploys/$TASK/$STEM.tar.gz" ]; then
        size=$(du -h "$REPO_ROOT/deploys/$TASK/$STEM.tar.gz" | awk '{print $1}')
        echo "[retry] OK $STEM ($size)"
    else
        echo "[retry] FAIL $STEM (rc=$rc)"
    fi
done

echo
echo "=== DONE ==="
ls -lh "$REPO_ROOT"/deploys/*/*.tar.gz 2>/dev/null
df -h ~ | tail -1
