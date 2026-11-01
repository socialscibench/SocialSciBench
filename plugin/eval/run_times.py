"""Seconds per estimate from the timing files each run's code wrote (task/TASK.md's formats), from the
rerun's copies (META/rerun_results/) or the agent's own (META/agent_results/).

usage: run_times.py [agent|rerun] [RUN_NAME_PREFIX ...]
For each run: the median over repetitions of the cells the reports compare, and the sum of all cells'
medians: 001, columns 8, 1 and 7 at 100% (all: 15 cells); 002, call 7 at 100 copies (all: 35 cells);
003, 100% and 1% (all: 5 sizes). Cells missing from a run's files are counted in "cells".
"""
import csv
import statistics
import sys
from collections import defaultdict

from summarize import metas

KEYS = {"001": (("column", "size_pct"), [("8", "100"), ("1", "100"), ("7", "100")], 15),
        "002": (("call", "scale"), [("7", "100")], 35),
        "003": (("size_pct",), [("100",), ("1",)], 5)}
NUM = lambda s: str(int(float(s))) if s.strip() else s           # "100.0" -> "100"


def cell_times(kit, files):
    """Median seconds per cell over the rows of every timing file with the kit's fields."""
    fields, _, _ = KEYS[kit]
    rows = defaultdict(list)
    for f in files:
        with open(f, newline="") as fh:
            r = csv.DictReader(fh)
            if not r.fieldnames or "seconds" not in r.fieldnames or not all(k in r.fieldnames for k in fields):
                continue
            for row in r:
                try:
                    rows[tuple(NUM(row[k]) for k in fields)].append(float(row["seconds"]))
                except (TypeError, ValueError):
                    continue
    return {k: statistics.median(v) for k, v in rows.items()}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ("agent", "rerun") else "rerun"
    prefixes = [a for a in sys.argv[1:] if a not in ("agent", "rerun")]
    print(f"| run | headline cells ({which}), median s | all cells, sum of medians | cells |")
    print("|---|---|---|---|")
    for m in metas():
        name = m.name[:-5]
        if prefixes and not any(name.startswith(p) for p in prefixes):
            continue
        kit = name[:3]
        t = cell_times(kit, sorted((m / f"{which}_results").rglob("*.csv")))
        _, heads, n = KEYS[kit]
        head = ", ".join(f"{'/'.join(h)}: {t[h]:.3g}" if h in t else f"{'/'.join(h)}: -" for h in heads)
        print(f"| {name} | {head} | {sum(t.values()):.4g} | {len(t)}/{n} |")


if __name__ == "__main__":
    main()
