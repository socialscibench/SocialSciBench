"""Per-case ratios between each task's memory and speed candidates at one thread, from the evaluator runs of a
report config (the ranges quoted in optimizations.md).

usage: ratios.py CONFIG.json      (repository root; e.g. runs/study-20261025/report-config.json)

For each task and suite (real, synthetic), per case:
- peak RSS of the speed candidate / the memory candidate's;
- median time of the memory candidate / the speed candidate's;
- peak RSS of the reference / each candidate's, the reference measured in that candidate's own run.
The two candidates' runs may be days apart, so the time ratios include differences in host throttling.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from report_tables import from_run, order  # noqa: E402


def span(xs):
    return f"{min(xs):.2f}-{max(xs):.2f}"


def main():
    cfg = json.loads(Path(sys.argv[1]).read_text())
    for task, spec in cfg["tasks"].items():
        rows = defaultdict(dict)                       # (solution, threads) -> case -> row
        for run in spec["runs"]:
            for r in from_run(Path(run)):
                case = "104k20" if task == "001" and r["case"] == "104" else r["case"]
                rows[(r["solution"], r["threads"])][case] = r
        sp, mem = rows[(spec["speed"], 1)], rows[(spec["memory"], 1)]
        print(f"\nTask {task}: {spec['speed']} (speed) and {spec['memory']} (memory), one thread")
        for suite in ("real", "synthetic"):
            cases = sorted([c for c in sp if c in mem and c.startswith("s") == (suite == "synthetic")], key=order)
            if not cases:
                continue
            peak, time, ref_mem, ref_sp = [], [], [], []
            for c in cases:
                a, b = sp[c], mem[c]
                peak.append(a["cand_peak_rss_gib"] / b["cand_peak_rss_gib"])
                time.append(b["cand_median_s"] / a["cand_median_s"])
                ref_mem.append(b["ref_peak_rss_gib"] / b["cand_peak_rss_gib"])
                ref_sp.append(a["ref_peak_rss_gib"] / a["cand_peak_rss_gib"])
                print(f"  {c:7s} peak {a['cand_peak_rss_gib']:7.3f} / {b['cand_peak_rss_gib']:7.3f} GiB = {peak[-1]:5.2f}x;  "
                      f"time {b['cand_median_s']:.4g} / {a['cand_median_s']:.4g} s = {time[-1]:5.2f}x;  "
                      f"reference peak / memory candidate's {ref_mem[-1]:5.2f}x, / speed candidate's {ref_sp[-1]:5.2f}x")
            print(f"  {suite}: speed/memory peak {span(peak)}x; memory/speed time {span(time)}x; reference/memory peak "
                  f"{span(ref_mem)}x; reference/speed peak {span(ref_sp)}x")


if __name__ == "__main__":
    main()
