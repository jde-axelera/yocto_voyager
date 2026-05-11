#!/usr/bin/env bash
# zoo_deploy.sh — space-aware deploy of a single voyager-sdk zoo model.
#
# Runs on an x86_64 Ubuntu build host that has voyager-sdk-1.6 checked out at
# $SDK_DIR. Patches the YAML for `aipu_cores_used=4`, runs deploy.py, tars only
# the runtime artifacts (model.json + ELFs + manifest), and immediately wipes
# the multi-GB intermediate build/ subtree.
#
# Output: $REPO_ROOT/deploys/<task>/<stem>.tar.gz   (typically 10–30 MB).
#
# Usage:
#   SDK_DIR=~/1.6/voyager-sdk \
#     scripts/zoo_deploy.sh <task> <yaml-stem> [--cores=4]
#
# Examples:
#   scripts/zoo_deploy.sh classification mobilenetv2-imagenet-onnx
#   scripts/zoo_deploy.sh instance_segmentation yolov8nseg-coco-onnx
#
# Exit codes:
#   0 success; tarball at deploys/<task>/<stem>.tar.gz
#   1 deploy.py failed; build/ already wiped, tarball NOT produced
#   2 usage / setup error

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "usage: $0 <task> <yaml-stem> [--cores=N]" >&2
    exit 2
fi
TASK="$1"; STEM="$2"; CORES=4
if [ "${3:-}" = --cores=* ] || [[ "${3:-}" == --cores=* ]]; then
    CORES="${3#--cores=}"
fi

SDK_DIR="${SDK_DIR:-$HOME/1.6/voyager-sdk}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$REPO_ROOT/deploys/$TASK"
mkdir -p "$OUT_DIR"

# Locate the YAML (zoo trees vary by task subdir layout).
YAML=""
for c in \
    "$SDK_DIR/ax_models/zoo/yolo/object_detection/$STEM.yaml" \
    "$SDK_DIR/ax_models/zoo/yolo/instance_segmentation/$STEM.yaml" \
    "$SDK_DIR/ax_models/zoo/yolo/keypoint_detection/$STEM.yaml" \
    "$SDK_DIR/ax_models/zoo/yolo/obb_detection/$STEM.yaml" \
    "$SDK_DIR/ax_models/zoo/tensorflow/object_detection/$STEM.yaml" \
    "$SDK_DIR/ax_models/zoo/torchvision/classification/$STEM.yaml" \
    "$SDK_DIR/ax_models/zoo/timm/$STEM.yaml" \
    "$SDK_DIR/ax_models/zoo/mmlab/mmseg/$STEM.yaml" \
    "$SDK_DIR/ax_models/zoo/torch/$STEM.yaml"
do
    if [ -f "$c" ]; then YAML="$c"; break; fi
done
if [ -z "$YAML" ]; then
    echo "ERROR: no YAML found for stem '$STEM' under $SDK_DIR/ax_models/zoo/" >&2
    exit 2
fi
echo "[deploy] yaml=$YAML"
echo "[deploy] task=$TASK stem=$STEM cores=$CORES"

# 1) Patch YAML with compilation_config (idempotent).
if ! grep -q "compilation_config:" "$YAML"; then
    cp "$YAML" "$YAML.bak"
    python3 - "$YAML" "$CORES" <<'PY'
import pathlib, re, sys
yaml_path = pathlib.Path(sys.argv[1])
cores = int(sys.argv[2])
s = yaml_path.read_text()
# Try the v1.6 anchor first.
old = "    extra_kwargs:\n      cal_seed: 129"
new = f"""    extra_kwargs:
      cal_seed: 129
      compilation_config:
        aipu_cores_used: {cores}
        resources_used: 1.0"""
if old in s:
    yaml_path.write_text(s.replace(old, new))
    print(f"[deploy] patched {yaml_path}")
else:
    # Generic anchor: insert under `extra_kwargs:` if present.
    if "extra_kwargs:" in s:
        s2 = re.sub(
            r"(extra_kwargs:\n)",
            f"\\1      compilation_config:\n        aipu_cores_used: {cores}\n        resources_used: 1.0\n",
            s, count=1)
        yaml_path.write_text(s2)
        print(f"[deploy] patched (generic) {yaml_path}")
    else:
        # Append a new extra_kwargs block at end-of-pipeline.
        yaml_path.write_text(s.rstrip() + f"""
    extra_kwargs:
      compilation_config:
        aipu_cores_used: {cores}
        resources_used: 1.0
""")
        print(f"[deploy] patched (appended) {yaml_path}")
PY
fi

# 2) Run deploy.py.
# We accept non-zero exit because deploy.py sometimes complains about a final
# pipeline step (post-quantize sanity) even after producing a usable build/.
cd "$SDK_DIR"
. venv/bin/activate
rm -rf "build/$STEM"
set +e
./deploy.py --mode QUANTCOMPILE --aipu-cores="$CORES" "$STEM" -v 2>&1 | tail -60
deploy_exit=$?
set -e

BUILD="$SDK_DIR/build/$STEM/$STEM"
if [ ! -f "$BUILD/$CORES/model.json" ]; then
    echo "[deploy] FAIL: $BUILD/$CORES/model.json missing (deploy_exit=$deploy_exit)"
    rm -rf "$SDK_DIR/build/$STEM"
    # Restore YAML.
    if [ -f "$YAML.bak" ]; then mv "$YAML.bak" "$YAML"; fi
    exit 1
fi

# 3) Tar only the model.json + ELFs + manifest. Drop quantized debug/tmp.
TAR_DIR="$SDK_DIR/build/$STEM/$STEM"
TAR_OUT="$OUT_DIR/$STEM.tar.gz"
echo "[deploy] tarring $TAR_DIR/$CORES → $TAR_OUT"
tar czf "$TAR_OUT" \
    -C "$SDK_DIR/build/$STEM" \
    "$STEM/$CORES/model.json" \
    "$STEM/$CORES/manifest.json" \
    "$STEM/compile_config.json" \
    "$STEM/model_info.json" \
    $(cd "$SDK_DIR/build/$STEM" && find "$STEM/$CORES" -maxdepth 1 -name "*.elf" -print 2>/dev/null) \
    2>/dev/null || true

# 4) Wipe the intermediate.
rm -rf "$SDK_DIR/build/$STEM"
if [ -f "$YAML.bak" ]; then mv "$YAML.bak" "$YAML"; fi

# 5) Report.
SIZE=$(du -h "$TAR_OUT" | awk '{print $1}')
echo "[deploy] OK  $TAR_OUT  ($SIZE)"
