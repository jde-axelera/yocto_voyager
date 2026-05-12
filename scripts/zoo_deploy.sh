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

# Restore YAML on any exit so a script crash mid-deploy doesn't leave the
# voyager-sdk YAML permanently patched (which causes the next manual run to
# misbehave because compilation_config is already present).
YAML_BAK_PATH=""
restore_yaml() {
    if [ -n "$YAML_BAK_PATH" ] && [ -f "$YAML_BAK_PATH" ]; then
        local orig="${YAML_BAK_PATH%.bak}"
        mv "$YAML_BAK_PATH" "$orig" 2>/dev/null || true
    fi
}
trap restore_yaml EXIT

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
    YAML_BAK_PATH="$YAML.bak"
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
# Pick whichever batch directory the compiler actually produced. Tiny models
# (mobilenet, squeezenet) often only compile at batch=1 even when the YAML
# requested aipu_cores=4 — falling back is fine, we just bench what we get.
ACTUAL_CORES=""
for c in "$CORES" 4 2 1; do
    if [ -f "$BUILD/$c/model.json" ]; then ACTUAL_CORES="$c"; break; fi
done
if [ -z "$ACTUAL_CORES" ]; then
    echo "[deploy] FAIL: no model.json found under $BUILD/{4,2,1}/ (deploy_exit=$deploy_exit)"
    rm -rf "$SDK_DIR/build/$STEM"
    if [ -f "$YAML.bak" ]; then mv "$YAML.bak" "$YAML"; fi
    exit 1
fi
echo "[deploy] picked batch=$ACTUAL_CORES (requested $CORES)"

# 3) Tar everything the AIPU runtime needs at load time.
#
# Required:
#   model.json, manifest.json, kernel_function*.elf, pool_ddr_const.bin,
#   pool_l2_const.bin, postprocess_graph.onnx (if present).
#
# Excluded as bulk:
#   kernel_function*.c            (source, runtime doesn't read them)
#   quantized_model.pt            (PyTorch checkpoint, ~20MB, host-only)
#   pool_ddr_input.bin            (sample inputs)
#   any .png / .npy debug dumps
TAR_OUT="$OUT_DIR/$STEM.tar.gz"
echo "[deploy] tarring $BUILD/$ACTUAL_CORES → $TAR_OUT"
# Build a tar list dynamically — only include files that actually exist.
# `set -e` is on, so we explicitly swallow the test exit code.
TAR_LIST=()
add_if_exists() {
    if [ -e "$SDK_DIR/build/$STEM/$1" ]; then
        TAR_LIST+=("$1")
    fi
}
add_if_exists "$STEM/$ACTUAL_CORES"          # batch dir (ELFs, model.json, const bins, etc.)
add_if_exists "$STEM/compile_config.json"
add_if_exists "$STEM/compiler_config.toml"
add_if_exists "$STEM/model_info.json"
add_if_exists "$STEM/quantized"              # contains sibling manifest.json + postprocess_graph for many models
( cd "$SDK_DIR/build/$STEM" && \
  tar czf "$TAR_OUT" \
    --exclude="*.c" \
    --exclude="quantized_model.pt" \
    --exclude="pool_ddr_input.bin" \
    --exclude="*.png" \
    --exclude="*.npy" \
    "${TAR_LIST[@]}" \
    2>/dev/null ) || true
if [ -f "$TAR_OUT" ]; then
    ls -lh "$TAR_OUT" 2>/dev/null || true
    echo "[deploy] tarball contents:"
    tar tzf "$TAR_OUT" 2>/dev/null | head -20 || true
else
    echo "[deploy] WARN: tar did not produce $TAR_OUT"
fi

# 4) Wipe the intermediate. YAML restore happens via the EXIT trap.
rm -rf "$SDK_DIR/build/$STEM"

# 5) Report.
SIZE=$(du -h "$TAR_OUT" | awk '{print $1}')
echo "[deploy] OK  $TAR_OUT  ($SIZE)"
