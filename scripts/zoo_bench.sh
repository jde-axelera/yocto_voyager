#!/usr/bin/env bash
# zoo_bench.sh — pull a model tarball, run `yolo_demo_multi`, parse stats,
# append a row to docs/MODEL_ZOO_REPORT.csv.
#
# Runs on the SBC. Assumes ~/cpp_test/yolo_demo_multi + aipu_worker.py are
# already deployed.
#
# Usage:
#   zoo_bench.sh <task> <stem> <tarball-path-on-sbc> [--bench=2] [--seconds=20]
#
# Output columns:
#   task,stem,batch,input_wh,fps_system,fps_infer,latency_ms_per_batch,
#   status,notes
set -eu

TASK_RAW="${1:?task required}"
STEM="${2:?stem required}"
TARBALL="${3:?tarball path required}"

# Remap voyager-sdk task subdir names to the unified --task handlers.
case "$TASK_RAW" in
    object_detection|obb_detection)                  TASK=detection ;;
    instance_segmentation|semantic_segmentation)     TASK=seg ;;
    keypoint_detection)                              TASK=pose ;;
    face_detection)                                  TASK=face ;;
    embedding)                                       TASK=embed ;;
    classification)                                  TASK=classify ;;
    *)                                               TASK="$TASK_RAW" ;;
esac
echo "[bench] task=$TASK_RAW (→ --task $TASK)  stem=$STEM"
BENCH="${4:---bench=2}"; BENCH="${BENCH#--bench=}"
SECONDS_ARG="${5:---seconds=20}"; SECONDS_ARG="${SECONDS_ARG#--seconds=}"

CPP_TEST="$HOME/cpp_test"
ZOO_ROOT="$HOME/zoo"
REPORT="$CPP_TEST/zoo_report.csv"
mkdir -p "$ZOO_ROOT"

# 1. Extract.
EXTRACT_DIR="$ZOO_ROOT/$STEM"
rm -rf "$EXTRACT_DIR"
mkdir -p "$EXTRACT_DIR"
tar xzf "$TARBALL" -C "$EXTRACT_DIR" || { echo "$TASK,$STEM,,,,,,EXTRACT_FAIL,tarball corrupt" >> "$REPORT"; exit 1; }

# Hunt for the model.json (varies by deploy: <stem>/<cores>/model.json or <stem>/<stem>/<cores>/model.json).
MODEL_JSON=""
for c in 4 1; do
    for cand in "$EXTRACT_DIR/$STEM/$c/model.json" "$EXTRACT_DIR/$STEM/$STEM/$c/model.json"; do
        [ -f "$cand" ] && MODEL_JSON="$cand" && break 2
    done
done
if [ -z "$MODEL_JSON" ]; then
    echo "$TASK,$STEM,,,,,,NO_MODEL_JSON,$(find "$EXTRACT_DIR" -name '*.json' | head -3 | tr '\n' '|')" >> "$REPORT"
    rm -rf "$EXTRACT_DIR"
    exit 1
fi
echo "[bench] model.json=$MODEL_JSON"

# 2. Run.
OUT_PREFIX="$CPP_TEST/multi_out/zoo_$STEM"
LOG="$CPP_TEST/multi_out/zoo_$STEM.log"
mkdir -p "$CPP_TEST/multi_out"

source "$HOME/axelera_pip/axelera-env/bin/activate" 2>/dev/null || true

# Watchdog: kill after seconds+15 in case the binary hangs.
(
    sleep $((SECONDS_ARG + 15))
    for p in $(pgrep -f "$STEM" 2>/dev/null); do kill -KILL $p 2>/dev/null || true; done
    for p in $(pgrep -f "aipu_worker" 2>/dev/null); do kill -KILL $p 2>/dev/null || true; done
) &
WATCHDOG=$!

set +e
"$CPP_TEST/run.sh" "$CPP_TEST/yolo_demo_multi" \
    --task    "$TASK" \
    --model   "$MODEL_JSON" \
    --inputs  "$CPP_TEST/traffic4_480p_mt.mp4" \
    --out     "$OUT_PREFIX" \
    --unpaced --display 0 --bench "$BENCH" \
    --py-dispatch \
    --py-worker "$CPP_TEST/aipu_worker.py" \
    > "$LOG" 2>&1 &
BIN_PID=$!

# Let it run for the configured window.
sleep "$SECONDS_ARG"
kill -INT "$BIN_PID" 2>/dev/null || true
# Give graceful shutdown up to 10 s, then force.
for _ in 1 2 3 4 5 6 7 8 9 10; do
    if ! kill -0 "$BIN_PID" 2>/dev/null; then break; fi
    sleep 1
done
for p in $(pgrep -f "yolo_demo_multi" 2>/dev/null); do kill -KILL $p 2>/dev/null || true; done
for p in $(pgrep -f "aipu_worker" 2>/dev/null); do kill -KILL $p 2>/dev/null || true; done
kill -KILL "$WATCHDOG" 2>/dev/null || true
set -e

# 3. Parse.
# Grab the stats lines and average over the last few (steady state).
STAT_LINES=$(grep '\[stats\]' "$LOG" | tail -8 || true)
INFER_FPS=$(echo "$STAT_LINES" | awk -F'infer=' '{print $2}' | awk -F' ' '{print $1}' | awk '{s+=$1;n++} END {if(n>0) printf "%.1f", s/n; else print ""}')
AGG_FPS=$(echo "$STAT_LINES"  | awk -F'agg-drawn=' '{print $2}' | awk -F' ' '{print $1}' | awk '{s+=$1;n++} END {if(n>0) printf "%.1f", s/n; else print ""}')
LAT_LINE=$(grep 'single-call lat' "$LOG" | tail -1 || true)
LAT_MS=$(echo "$LAT_LINE" | sed -nE 's/.*single-call lat ([0-9.]+) ms.*/\1/p')

# Detect setup-time crashes.
STATUS="OK"; NOTES=""
if grep -q 'run failed' "$LOG"; then STATUS="RUN_FAIL"; NOTES="$(grep 'run failed' "$LOG" | head -1)"; fi
if ! echo "$STAT_LINES" | grep -q infer=; then
    STATUS="NO_STATS"
    NOTES="$(tail -2 "$LOG" | tr '\n' ' ')"
fi
# Strip commas from notes for CSV-safety.
NOTES=$(echo "$NOTES" | tr ',' ';' | tr '\n' ' ' | cut -c1-200)

# 4. Append CSV row.
if [ ! -f "$REPORT" ]; then
    echo "task,stem,bench,fps_system,fps_infer,lat_ms_per_batch,status,notes" > "$REPORT"
fi
echo "$TASK,$STEM,$BENCH,$AGG_FPS,$INFER_FPS,$LAT_MS,$STATUS,$NOTES" >> "$REPORT"

# 5. Clean up the extracted model + the output mp4 (regeneratable).
rm -rf "$EXTRACT_DIR"
rm -f "$OUT_PREFIX"_*.mp4 || true

# Wait for device to be reclaimable.
for _ in 1 2 3 4 5; do
    if ! fuser /dev/metis* >/dev/null 2>&1; then break; fi
    sleep 1
done

echo "[bench] $TASK/$STEM → fps=$AGG_FPS lat=$LAT_MS status=$STATUS"
