#!/usr/bin/env python3
"""
zoo_report.py — render zoo_report.csv + deploys/_summary.csv into a single
Markdown table for docs/MODEL_ZOO_REPORT.md.

Usage:
    python3 scripts/zoo_report.py \
        --deploys deploys/_summary.csv \
        --bench   zoo_report.csv \
        --out     docs/MODEL_ZOO_REPORT.md
"""
from __future__ import annotations

import argparse
import csv
import datetime
import pathlib
from typing import Dict, List


def load_csv(path: pathlib.Path) -> List[Dict[str, str]]:
    if not path or not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def render(deploys: List[Dict[str, str]], bench: List[Dict[str, str]]) -> str:
    # Index bench rows by stem for join.
    by_stem: Dict[str, Dict[str, str]] = {}
    for row in bench:
        by_stem[row.get("stem", "")] = row

    lines: List[str] = []
    lines.append("# Model-zoo benchmark report")
    lines.append("")
    lines.append(
        f"_Generated {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}_"
    )
    lines.append("")
    lines.append("Branch: `feat/model-zoo`. Source data:")
    lines.append("- `deploys/_summary.csv` — one row per `scripts/zoo_deploy.sh` run on the build host.")
    lines.append("- `zoo_report.csv` (on SBC) — one row per `scripts/zoo_bench.sh` run.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    ok_dep    = sum(1 for r in deploys if r.get("status") == "OK")
    fail_dep  = sum(1 for r in deploys if r.get("status") and r.get("status") != "OK")
    ok_bench  = sum(1 for r in bench   if r.get("status") == "OK")
    fail_bench= sum(1 for r in bench   if r.get("status") and r.get("status") != "OK")
    lines.append(f"- Deploys attempted: {len(deploys)}  (OK: **{ok_dep}**, FAIL: **{fail_dep}**)")
    lines.append(f"- Benchmarks attempted: {len(bench)}  (OK: **{ok_bench}**, FAIL: **{fail_bench}**)")
    lines.append("")
    lines.append("## Per-model table")
    lines.append("")
    lines.append("| Task | Model | Deploy | Bench | fps (system) | fps (infer) | Latency / batch | Notes |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for d in deploys:
        stem = d.get("stem", "")
        b = by_stem.get(stem, {})
        notes = (d.get("notes", "") or "") + (" / " + b.get("notes", "") if b.get("notes") else "")
        notes = (notes or "").strip(" /").replace("|", "/")[:120]
        lines.append("| {task} | `{stem}` | {ds} | {bs} | {fps_s} | {fps_i} | {lat} | {notes} |".format(
            task=d.get("task", "")[:24],
            stem=stem,
            ds=d.get("status", ""),
            bs=b.get("status", "—"),
            fps_s=b.get("fps_system", "—"),
            fps_i=b.get("fps_infer", "—"),
            lat=b.get("lat_ms_per_batch", "—"),
            notes=notes or "—",
        ))
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--deploys", type=pathlib.Path, required=True)
    p.add_argument("--bench",   type=pathlib.Path, required=False)
    p.add_argument("--out",     type=pathlib.Path, required=True)
    args = p.parse_args()

    deploys = load_csv(args.deploys)
    bench   = load_csv(args.bench) if args.bench else []
    text = render(deploys, bench)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"wrote {args.out} ({len(text)} bytes; {len(deploys)} deploys, {len(bench)} benches)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
