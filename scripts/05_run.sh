#!/bin/sh
# 05_run.sh
#
# Convenience wrapper for running yolo_demo_multi on the SBC.
# Sets LD_LIBRARY_PATH to the Axelera userspace .so files inside the pip venv.
#
# Examples:
#   # single stream, file only:
#   sh 05_run.sh ./yolo_demo_multi MODEL VIDEO OUT 0.25 0.45 1 0 0 4 0 60
#
#   # 10 streams at 25 fps, composite TCP MPEG-TS on port 5000:
#   sh 05_run.sh ./yolo_demo_multi MODEL "V1.mp4,V2.mp4,...,V10.mp4" OUT_PREFIX \
#                                       0.25 0.45 1 0 0 4 2 25
#
# Where:
#   MODEL          path to .axm or model.json (e.g. ~/yolo11n_4c/.../4/model.json)
#   VIDEO          single mp4 OR comma-separated list (up to 10)
#   OUT[_PREFIX]   output file (single) or prefix (multi: produces <prefix>_0.mp4, _1.mp4, ...)
#   conf iou       detection thresholds (defaults 0.25, 0.45)
#   workers        keep at 1 (the 4-core .axm uses all 4 AIPU sub-devices in one call)
#   bench          0=full, 1=skip draw+write, 2=skip postproc/draw/write (preproc+infer only)
#   dmabuf         unused (binary always uses an internal dma-heap dmabuf for input)
#   preproc        number of preproc threads (default 4)
#   display        0=file, 1=local X11 composite, 2=TCP MPEG-TS composite on :5000
#   fps            target fps per stream (default 25)

set -e

if [ $# -lt 3 ]; then
    echo "usage: $0 ./yolo_demo_multi <model> <vids> <out_prefix> [args...]" >&2
    exit 1
fi

SP="${HOME}/axelera_pip/axelera-env/lib/python3.10/site-packages"
if [ ! -d "$SP/axelera/lib" ]; then
    echo "ERROR: axelera-rt venv not found at $SP" >&2
    echo "       Run scripts/02_install_runtime.sh first." >&2
    exit 1
fi
export LD_LIBRARY_PATH="$SP/axelera/lib:$SP/axelera/runtime2:$SP/axelera_runtime.libs:$SP/axelera_runtime2.libs:${LD_LIBRARY_PATH:-}"

exec "$@"
