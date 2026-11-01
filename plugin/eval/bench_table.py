"""Summarize bench_engine.py runs: per kit, size and call, the median seconds over repetitions and the
first repetition's memory increase, for each labelled run directory.

usage: bench_table.py LABEL=DIR [LABEL=DIR ...] [--kit 001]
Prints one Markdown table per kit and variant: rows are size x call, columns the labels (seconds;
GiB added). Throttled samples during a call, if any, are counted in the last column.
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load(d):
    rows = defaultdict(list)
    for line in (Path(d) / "timings.jsonl").read_text().splitlines():
        r = json.loads(line)
        rows[(r["kit"], r["variant"], r["size"], r["call"])].append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--kit")
    a = ap.parse_args()
    runs = [(lab, load(d)) for lab, d in (x.split("=", 1) for x in a.runs)]
    keys = sorted({k for _, rows in runs for k in rows if not a.kit or k[0] == a.kit})
    for kit, variant in sorted({(k[0], k[1]) for k in keys}):
        print(f"\n### {kit}, {variant} variant\n")
        print("| size | call | " + " | ".join(f"{lab} s | {lab} GiB" for lab, _ in runs) + " | throttled |")
        print("|---|---|" + "---|---|" * len(runs) + "---|")
        for k in [k for k in keys if k[:2] == (kit, variant)]:
            cells, thr = [], 0
            for _, rows in runs:
                rs = rows.get(k)
                if not rs:
                    cells += ["", ""]
                    continue
                first = min(rs, key=lambda r: r["rep"])
                cells += [f"{statistics.median(r['seconds'] for r in rs):.3g}",
                          f"{first['call_peak_increase'] / 2**30:.2f}"]
                thr += max(r.get("thermal", {}).get("throttled", 0) for r in rs)
            print(f"| {k[2]} | {k[3]} | " + " | ".join(cells) + f" | {thr} |")


if __name__ == "__main__":
    main()
