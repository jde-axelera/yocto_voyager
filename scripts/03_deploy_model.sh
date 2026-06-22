#!/usr/bin/env bash
# 03_deploy_model.sh
#
# Run on an x86_64 Ubuntu 22.04 BUILD HOST that has the full Voyager SDK 1.6 installed.
# Compiles YOLOv11n with N AIPU cores and packages it for the SBC.
#
# Default is AIPU_CORES=4 → ~870 fps device-side throughput ceiling, ~100 ms
# in-process v4l2->drawn latency. For low-latency (~25 ms), pass AIPU_CORES=1.
#
#   sh 03_deploy_model.sh                 # 4-core (throughput)
#   AIPU_CORES=1 sh 03_deploy_model.sh    # 1-core (low latency)
#
# Output: yolo11n_${AIPU_CORES}core.tar.gz  (~20 MB; model.json + kernel ELFs)
#
# Note: this script intentionally does NOT patch the YAML. Earlier versions
# injected a `compilation_config: { aipu_cores_used: 4 }` block under
# extra_kwargs; that block can conflict with the --aipu-cores CLI flag and
# produce a kernel ELF that segfaults at load. Pass --aipu-cores on the CLI
# only — leave the YAML alone.

set -euo pipefail

: "${SDK_DIR:?ERROR: set SDK_DIR to the voyager-sdk-1.6 checkout (e.g. ~/voyager-sdk)}"
: "${AIPU_CORES:=4}"

YAML="${SDK_DIR}/ax_models/zoo/yolo/object_detection/yolo11n-coco-onnx.yaml"
test -f "$YAML" || { echo "ERROR: YAML not found at $YAML" >&2; exit 1; }

if grep -q "compilation_config:" "$YAML"; then
    echo "WARNING: $YAML contains a compilation_config block (leftover from a" >&2
    echo "         previous version of this script). Remove it before deploying;" >&2
    echo "         the CLI --aipu-cores flag is sufficient." >&2
    exit 1
fi

cd "$SDK_DIR"
rm -rf build/yolo11n-coco-onnx
# shellcheck source=/dev/null
. venv/bin/activate

./deploy.py --mode QUANTCOMPILE --aipu-cores="$AIPU_CORES" yolo11n-coco-onnx -v 2>&1 | tail -n 50 || true

# Verify the requested-core compile (deploy.py final pipeline step may complain
# about the postproc graph but the kernel ELF + model.json are still built).
BUILD="${SDK_DIR}/build/yolo11n-coco-onnx/yolo11n-coco-onnx"
test -f "$BUILD/${AIPU_CORES}/model.json" || {
    echo "ERROR: ${AIPU_CORES}-core compile did not produce model.json" >&2
    exit 1
}
grep -E '"aipu_cores_used"' "$BUILD/compile_config.json"

# Tar and announce
OUT="${SDK_DIR}/yolo11n_${AIPU_CORES}core.tar.gz"
tar czf "$OUT" -C "${SDK_DIR}/build" yolo11n-coco-onnx/
echo
echo "Done.  Output: $OUT"
echo "Copy to the SBC and extract:"
echo "  scp $OUT <user>@<sbc>:~/"
echo "  ssh <user>@<sbc> 'mkdir -p ~/yolo11n_${AIPU_CORES}c && tar xzf ~/yolo11n_${AIPU_CORES}core.tar.gz -C ~/yolo11n_${AIPU_CORES}c'"
echo "Then the model.json path on the SBC is:"
echo "  ~/yolo11n_${AIPU_CORES}c/yolo11n-coco-onnx/yolo11n-coco-onnx/${AIPU_CORES}/model.json"
