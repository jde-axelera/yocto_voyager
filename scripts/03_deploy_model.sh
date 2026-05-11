#!/usr/bin/env bash
# 03_deploy_model.sh
#
# Run on an x86_64 Ubuntu 22.04 BUILD HOST that has the full Voyager SDK 1.6 installed.
# Compiles YOLOv11n with 4 AIPU cores and packages it for the SBC.
#
# Produces:  yolo11n_4core.tar.gz    (~20 MB; contains model.json + 4x kernel_function*.elf)
#
# Why: `axdownloadmodel yolo11n-coco-onnx --axm` returns a 1-core deploy that caps
# device-side throughput at ~135 fps. A proper 4-core deploy gives ~870 fps device.

set -euo pipefail

: "${SDK_DIR:?ERROR: set SDK_DIR to the voyager-sdk-1.6 checkout (e.g. ~/voyager-sdk)}"

YAML="${SDK_DIR}/ax_models/zoo/yolo/object_detection/yolo11n-coco-onnx.yaml"
test -f "$YAML" || { echo "ERROR: YAML not found at $YAML" >&2; exit 1; }

# Inject compilation_config under extra_kwargs if it isn't there yet.
if ! grep -q "compilation_config:" "$YAML"; then
    cp "$YAML" "$YAML.bak"
    python3 - <<EOF
import pathlib
p = pathlib.Path("$YAML")
s = p.read_text()
old = "    extra_kwargs:\n      cal_seed: 129"
new = """    extra_kwargs:
      cal_seed: 129
      compilation_config:
        aipu_cores_used: 4
        resources_used: 1.0"""
assert old in s, "anchor missing — YAML schema differs from expected v1.6"
p.write_text(s.replace(old, new))
EOF
    echo "patched $YAML  (backup at $YAML.bak)"
fi

cd "$SDK_DIR"
rm -rf build/yolo11n-coco-onnx
# shellcheck source=/dev/null
. venv/bin/activate

./deploy.py --mode QUANTCOMPILE --aipu-cores=4 yolo11n-coco-onnx -v 2>&1 | tail -n 50 || true

# Verify 4-core compile (deploy.py final pipeline step may complain but the model is built)
BUILD="${SDK_DIR}/build/yolo11n-coco-onnx/yolo11n-coco-onnx"
test -f "$BUILD/4/model.json" || { echo "ERROR: 4-core compile did not produce model.json" >&2; exit 1; }
grep -E '"aipu_cores_used"' "$BUILD/compile_config.json"

# Restore the YAML to its original
if [ -f "$YAML.bak" ]; then mv "$YAML.bak" "$YAML"; fi

# Tar and announce
OUT="${SDK_DIR}/yolo11n_4core.tar.gz"
tar czf "$OUT" -C "${SDK_DIR}/build/yolo11n-coco-onnx" yolo11n-coco-onnx/
echo
echo "Done.  Output: $OUT"
echo "Copy to the SBC and extract:"
echo "  scp $OUT <user>@<sbc>:~/"
echo "  ssh <user>@<sbc> 'mkdir -p ~/yolo11n_4c && tar xzf ~/yolo11n_4core.tar.gz -C ~/yolo11n_4c'"
echo "Then the model.json path on the SBC is:"
echo "  ~/yolo11n_4c/yolo11n-coco-onnx/yolo11n-coco-onnx/4/model.json"
