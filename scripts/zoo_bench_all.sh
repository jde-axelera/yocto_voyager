#!/usr/bin/env bash
# zoo_bench_all.sh — bench every tarball under ~/zoo_tarballs/ on the SBC.
# Driven from the SBC. Pulls the tarball list and dispatches to zoo_bench.sh.
#
# Layout expected:
#   ~/zoo_tarballs/<task>/<stem>.tar.gz
set -eu

BENCH_SCRIPT="$HOME/cpp_test/zoo_bench.sh"
TARBALL_ROOT="$HOME/zoo_tarballs"
REPORT="$HOME/cpp_test/zoo_report.csv"

rm -f "$REPORT"

for tar in "$TARBALL_ROOT"/*/*.tar.gz; do
    [ -e "$tar" ] || continue
    rel="${tar#$TARBALL_ROOT/}"
    task="${rel%%/*}"
    file="${rel##*/}"
    stem="${file%.tar.gz}"
    # Pick task name → handler name. The voyager-sdk path categories differ
    # from our handler names; remap.
    case "$task" in
        object_detection|obb_detection) HANDLER="detection" ;;
        classification)                 HANDLER="classify" ;;
        instance_segmentation|semantic_segmentation) HANDLER="seg" ;;
        keypoint_detection)             HANDLER="pose" ;;
        face_detection)                 HANDLER="face" ;;
        embedding)                      HANDLER="embed" ;;
        *)                              HANDLER="detection" ;;
    esac
    echo
    echo "=== [$(date +%H:%M:%S)] $task / $stem  →  --task $HANDLER ==="
    # bench=2 measures pure dispatch throughput (no postproc/draw/write).
    "$BENCH_SCRIPT" "$HANDLER" "$stem" "$tar" --bench=2 --seconds=15 || true
done

echo
echo "=== DONE [$(date +%H:%M:%S)] ==="
echo "Report: $REPORT"
echo
cat "$REPORT"
