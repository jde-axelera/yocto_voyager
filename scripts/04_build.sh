#!/usr/bin/env bash
# 04_build.sh
#
# Cross-compile yolo_demo_multi for aarch64 on an x86_64 Linux build host.
#
# Steps:
#   1. Install gcc-aarch64-linux-gnu / g++-aarch64-linux-gnu (apt).
#   2. Extract the Axelera aarch64 runtime wheels into sysroot/ so we have
#      libaxruntime.so + headers + level_zero/* available at link time.
#   3. CMake + make.
#
# Output: build/yolo_demo_multi  (aarch64 ELF, statically linked libstdc++/libgcc).
#         Copy it to the SBC's ~/cpp_test/.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# 1. Verify cross-toolchain
if ! command -v aarch64-linux-gnu-g++ >/dev/null 2>&1; then
    echo "Installing cross-toolchain..."
    sudo apt-get update
    sudo apt-get install -y gcc-aarch64-linux-gnu g++-aarch64-linux-gnu cmake
fi

# 2. Populate sysroot from the published aarch64 wheels (one-time)
if [ ! -f "$ROOT/sysroot/axelera/include/axruntime/axruntime.h" ]; then
    echo "Fetching Axelera aarch64 runtime wheels..."
    mkdir -p "$ROOT/sysroot" "$ROOT/wheels"
    pip download --no-deps --no-cache-dir \
        --platform manylinux_2_28_aarch64 \
        --python-version 310 --implementation cp --abi cp310 \
        --only-binary=:all: \
        --extra-index-url https://software.axelera.ai/artifactory/api/pypi/axelera-pypi/simple \
        -d "$ROOT/wheels" \
        axelera-runtime axelera-runtime2
    for w in "$ROOT/wheels"/axelera_runtime-*.whl "$ROOT/wheels"/axelera_runtime2-*.whl; do
        unzip -q -o "$w" -d "$ROOT/sysroot"
    done
fi

# 3. Build
mkdir -p "$ROOT/build"
cd "$ROOT/build"
cmake -DCMAKE_TOOLCHAIN_FILE="$ROOT/toolchain-aarch64.cmake" \
      -DCMAKE_BUILD_TYPE=Release \
      -DAX_SYSROOT="$ROOT/sysroot/axelera" \
      "$ROOT/src"
cmake --build . -- -j

# Sanity
file ./yolo_demo_multi
echo
echo "Binary: $(pwd)/yolo_demo_multi"
echo "Copy it to the SBC and run via scripts/05_run.sh"
