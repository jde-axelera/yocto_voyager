#!/bin/sh
# 01_update_driver.sh
#
# Run on the Antelao SBC AS ROOT (`su` first; default root password on a fresh
# Voyager Linux 1.3.1 image is `AxeRoot2025`).
#
# A fresh image ships metis kernel module v1.4.4 but axelera-rt >= 1.6.0 needs
# >= 1.4.10. Amarula publishes a pre-built .deb that targets this exact kernel.
#
# Usage:   su -c "sh 01_update_driver.sh"
# Verify:  cat /sys/class/metis/version    -> 1.4.16

set -e

DEB_URL='https://amarula-share.s3.eu-central-1.amazonaws.com/yocto_build/v1.3.1-21-gca48dd0_248/deb/kernel-module-metis-6.1.148-rockchip-standard_v1.6.0-rc1-r0_arm64.deb'
DEB_NAME="$(basename "$DEB_URL")"

# 1) confirm we are on the expected kernel
KERN="$(uname -r)"
if [ "$KERN" != "6.1.148-rockchip-standard" ]; then
    echo "ERROR: this driver .deb targets 6.1.148-rockchip-standard, but kernel is $KERN" >&2
    exit 1
fi

# 2) confirm we are root
if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: must be root (run via: su)" >&2
    exit 1
fi

# 3) current version
echo "current metis driver: $(cat /sys/class/metis/version 2>/dev/null || echo none)"

# 4) fetch
cd /tmp
if [ ! -f "$DEB_NAME" ]; then
    curl -fsSL -O "$DEB_URL"
fi

# 5) install (rootfs is read-only by default on Voyager Linux)
mount -o remount,rw /
dpkg -i "$DEB_NAME"
echo metis > /etc/modules-load.d/metis.conf
sync
mount -o remount,ro / 2>/dev/null || true

# 6) reboot
echo
echo "Driver installed. Rebooting in 5 s..."
sleep 5
reboot
