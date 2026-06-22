#!/bin/sh
# 05_run.sh
#
# Convenience wrapper for running yolo_demo_multi on the SBC. Just sets
# LD_LIBRARY_PATH so the binary can find the Axelera userspace .so files
# inside the pip venv, then execs whatever you pass.
#
# Examples:
#
#   # 1) file-only single stream at 60 fps
#   sh 05_run.sh ./yolo_demo_multi \
#       --model  MODEL \
#       --inputs VIDEO \
#       --out    OUT_PREFIX \
#       --fps    60
#
#   # 2) 10 streams at 25 fps, composite TCP MPEG-TS on port 5000
#   sh 05_run.sh ./yolo_demo_multi \
#       --model   MODEL \
#       --inputs  V1.mp4,V2.mp4,...,V10.mp4 \
#       --out     OUT_PREFIX \
#       --fps     25 \
#       --display 2
#
# Run `./yolo_demo_multi --help` for the full flag reference.

set -e

if [ $# -lt 1 ]; then
    echo "usage: $0 ./yolo_demo_multi [flags...]" >&2
    exit 1
fi

SP="${HOME}/axelera_pip/axelera-env/lib/python3.10/site-packages"
if [ ! -d "$SP/axelera/lib" ]; then
    echo "ERROR: axelera-rt venv not found at $SP" >&2
    echo "       Run scripts/02_install_runtime.sh first." >&2
    exit 1
fi
export LD_LIBRARY_PATH="$SP/axelera/lib:$SP/axelera/runtime2:$SP/axelera_runtime.libs:$SP/axelera_runtime2.libs:${LD_LIBRARY_PATH:-}"

# Activate the venv so that --py-dispatch can import axelera Python modules.
# shellcheck source=/dev/null
. "${HOME}/axelera_pip/axelera-env/bin/activate"

exec "$@"
