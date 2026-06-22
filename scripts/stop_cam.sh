#!/bin/sh
# stop_cam.sh — gracefully stop the camera demo from a second SSH session.
# Sends SIGINT so the MP4 trailer is written and waylandsink tears down.
# Run this from your laptop:  ssh antelao@<ip> '~/stop_cam.sh'

PIDS=$(pgrep -f yolo_demo_multi)
if [ -z "$PIDS" ]; then
    echo "no yolo_demo_multi running"
    exit 0
fi
echo "stopping: $PIDS"
for p in $PIDS; do kill -INT "$p" 2>/dev/null; done

# Wait up to 8s for a clean exit, then escalate to TERM.
i=0
while [ $i -lt 8 ]; do
    pgrep -f yolo_demo_multi >/dev/null || { echo "stopped cleanly"; exit 0; }
    sleep 1; i=$((i+1))
done
echo "still up after 8s — sending TERM"
for p in $(pgrep -f yolo_demo_multi); do kill -TERM "$p" 2>/dev/null; done
sleep 2
pgrep -f yolo_demo_multi >/dev/null && echo "WARNING: still running" || echo "stopped"
