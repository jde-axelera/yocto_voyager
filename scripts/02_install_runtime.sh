#!/bin/sh
# 02_install_runtime.sh
#
# Run on the Antelao SBC as the default non-root user (e.g. `antelao`).
# Installs the Voyager pip runtime (axelera-rt) into a venv at ~/axelera_pip/axelera-env.
#
# Prerequisites: metis kernel driver >= 1.4.10 (run 01_update_driver.sh first).
# Verify after: $ axdevice  -> "Device 0: metis-0:1:0 ..."

set -e

cd "$HOME"
mkdir -p axelera_pip
cd axelera_pip

if [ ! -d axelera-env ]; then
    python3 -m venv axelera-env
fi
. axelera-env/bin/activate

pip install --upgrade pip wheel
pip install --extra-index-url \
    https://software.axelera.ai/artifactory/api/pypi/axelera-pypi/simple \
    axelera-rt

# Smoke test
axdevice

# Optional: also install requests so axdownloadmedia/axdownloadmodel work
pip install requests tqdm

echo
echo "Runtime installed in ${PWD}/axelera-env"
echo "Activate with:  . ${PWD}/axelera-env/bin/activate"
