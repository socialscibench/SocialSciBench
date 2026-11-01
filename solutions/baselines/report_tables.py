"""Tables of the 2026-10-25 study report (study tooling).

usage: report_tables.py CONFIG.json

CONFIG.json: {"tasks": {"001": {"speed": NAME, "memory": NAME, "runs": [RUN_DIR, ...]}, ...},
              "baselines": {"001": [RESULTS_DIR, ...], ...}}

Evaluator runs are matched to a task and candidate by their run.json. For each task: real and
synthetic workloads, with the reference and the speed candidate at every thread count measured,
the memory candidate, and the workers' peak RSS; then the baselines (compare.py rows; the last
row per case and thread count that thermal_queue.py did not exclude) against the speed candidate
at the same thread count.
"""
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "solutions/baselines")
from summarize import from_run  # noqa: E402


def secs(v):
    if v is None:
        return "-"
    return f"{v:.3f} s" if v < 10 else (f"{v:.2f} s" if v < 100 else f"{v:,.0f} s")


def gib(v):
    return "-" if not v else (f"{v * 1024:.0f} MB" if v < 1 else f"{v:.2f} GiB")


def order(case):
    m = re.match(r"s?(\d+)(k\d+)?", case)
    return (case.startswith("s"), int(m.group(1)), m.group(2) or "")


def gmean(xs):
    xs = [x for x in xs if x]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def summary(xs):
    xs = [x for x in xs if x]
    return f"{gmean(xs):.1f}x ({min(xs):.1f}-{max(xs):.1f}x, {len(xs)} cases)" if xs else "-"


SUPERSEDED = defaultdict(list)       # task -> [(solution, threads, case, row)]: earlier rows of a case


def baseline_rows(dirs):
    """task -> solution -> (threads, case) -> row (the last one; earlier rows go to SUPERSEDED)."""
    out = defaultdict(lambda: defaultdict(dict))
    for d in map(Path, dirs):
        excluded = set()
        if (d / "exclusions.json").exists():      # the earlier format: a list of {file, case, created}
            excluded |= {(e["file"], e["case"], e["created"]) for e in json.loads((d / "exclusions.json").read_text())}
        if (d / "exclusions.jsonl").exists():
            for e in map(json.loads, (d / "exclusions.jsonl").read_text().splitlines()):
                if "file" in e and not e.get("final"):
                    excluded.add((e["file"], e["case"], e.get("created")))
        for f in sorted(d.glob("*.jsonl")):
            m = re.match(r"(\d{3})-(.+)-t(\d+)\.jsonl$", f.name)
            if not m:
                continue
            for line in f.read_text().splitlines():
                r = json.loads(line)
                if (f.name, r["case"], r.get("created")) in excluded:
                    continue
                case = "104k20" if m.group(1) == "001" and r["case"] == "104" else r["case"]
                old = out[m.group(1)][m.group(2)].get((r["threads"], case))
                if old is not None:
                    SUPERSEDED[m.group(1)].append((m.group(2), r["threads"], case, old))
                out[m.group(1)][m.group(2)][(r["threads"], case)] = r
    return out


def status(r):
    if r.get("status") != "COMPLETED":
        return {"TIMEOUT": "timeout", "RUNTIME_ERROR": "error"}.get(r.get("status"), str(r.get("status")).lower())
    return "passed" if r.get("checks_passed") else "fails checks"


def main():
    cfg = json.loads(Path(sys.argv[1]).read_text())
    base = baseline_rows(list(dict.fromkeys(d for ds in cfg.get("baselines", {}).values() for d in ds)))
    for task, spec in cfg["tasks"].items():
        rows = defaultdict(dict)                       # (solution, threads) -> case -> row
        for run in spec["runs"]:
            for r in from_run(Path(run)):
                case = "104k20" if task == "001" and r["case"] == "104" else r["case"]
                rows[(r["solution"], r["threads"])][case] = r
        speed, memory = spec["speed"], spec.get("memory")
        threads = sorted({t for (s, t) in rows if s == speed})
        print(f"\n### Task {task}: {speed}" + (f" (speed) and {memory} (memory)" if memory else "") + "\n")
        for suite in ("real", "synthetic"):
            cases = sorted({c for d in rows.values() for c in d if (c.startswith("s") == (suite == "synthetic"))},
                           key=order)
            if not cases:
                continue
            head = ["Case", "Calls", "Reference, 1 thread"]
            for t in threads:
                head.append(f"{speed}, {t} thread{'s' if t > 1 else ''}")
            if memory:
                head.append(f"{memory}, 1 thread")
            head += ["Peak RSS, 1 thread: reference / speed / memory", "Max. rel. error"]
            print(f"\n{suite.capitalize()} workloads (evaluator medians; speedup against the reference of the same run):\n")
            print("| " + " | ".join(head) + " |")
            print("|---|---|" + "---:|" * (len(head) - 2))
            sp = defaultdict(list)
            for c in cases:
                r1 = rows[(speed, 1)].get(c, {})
                cells = [c, f"1+{r1.get('warmups', '-')}+{r1.get('repeats', '-')}", secs(r1.get("ref_median_s"))]
                err = 0.0
                for t in threads:
                    r = rows[(speed, t)].get(c)
                    if r:
                        ok = r["status"] == "PASSED"
                        cells.append(f"{secs(r['cand_median_s'])} ({r['speedup']:.1f}x)" if ok and r.get("speedup")
                                     else r["status"].lower())
                        if ok:
                            sp[(speed, t)].append(r["speedup"])
                        err = max(err, r.get("cand_max_rel_error") or 0)
                    else:
                        cells.append("-")
                rm = rows[(memory, 1)].get(c) if memory else None
                if memory:
                    if rm:
                        ok = rm["status"] == "PASSED"
                        cells.append(f"{secs(rm['cand_median_s'])} ({rm['speedup']:.1f}x)" if ok and rm.get("speedup")
                                     else rm["status"].lower())
                        if ok:
                            sp[(memory, 1)].append(rm["speedup"])
                        err = max(err, rm.get("cand_max_rel_error") or 0)
                    else:
                        cells.append("-")
                cells.append(f"{gib(r1.get('ref_peak_rss_gib'))} / {gib(r1.get('cand_peak_rss_gib'))} / "
                             f"{gib((rm or {}).get('cand_peak_rss_gib'))}")
                cells.append(f"{err:.1e}")
                print("| " + " | ".join(cells) + " |")
            print()
            for (s, t), xs in sorted(sp.items()):
                print(f"- {s}, {t} thread{'s' if t > 1 else ''}: against the reference {summary(xs)}")
            for s in [speed] + ([memory] if memory else []):    # peak RSS against the reference, one thread
                ratios = [r["cand_peak_rss_gib"] / r["ref_peak_rss_gib"] for c in cases
                          for r in [rows[(s, 1)].get(c)] if r and r.get("ref_peak_rss_gib") and r.get("cand_peak_rss_gib")]
                if ratios:
                    print(f"- {s}, 1 thread: peak RSS / the reference's {gmean(ratios):.2f} "
                          f"({min(ratios):.2f}-{max(ratios):.2f}, {len(ratios)} cases)")
        # baselines against the speed candidate at the same thread count
        sols = sorted(base.get(task, {}))
        for t in sorted({t for s in sols for (t, _) in base[task][s]}):
            have = [s for s in sols if any(k[0] == t for k in base[task][s])]
            cases = sorted({c for s in have for (tt, c) in base[task][s] if tt == t}, key=order)
            print(f"\nBaselines, {t} thread{'s' if t > 1 else ''} (compare.py: first call + one measured call; "
                  f"median, and baseline / {speed} median):\n")
            print(f"| Case | {speed} | " + " | ".join(have) + " |")
            print("|---|---:|" + "---:|" * len(have))
            fails = []
            for c in cases:
                cm = (rows[(speed, t)].get(c) or {}).get("cand_median_s")
                cells = [c, secs(cm)]
                for s in have:
                    r = base[task][s].get((t, c))
                    if not r:
                        cells.append("")
                        continue
                    st = status(r)
                    if st == "passed":
                        cells.append(f"{secs(r['median_s'])} ({r['median_s'] / cm:.0f}x)" if cm else secs(r["median_s"]))
                    else:
                        med = r.get("median_s")
                        cells.append(f"{st}" + (f", {secs(med)}" if med else ""))
                        fails.append((s, c, r))
                    if r.get("not_compared"):
                        cells[-1] += " *"
                print("| " + " | ".join(cells) + " |")
            if any(r.get("not_compared") for s in have for r in base[task][s].values()):
                print("\n\\* bootstrap outputs not compared (the tool's own multipliers).")
            for s, c, r in fails:
                print(f"- {s} {c}: {failure(r)}")
            earlier = [(s, c, r) for s, tt, c, r in SUPERSEDED.get(task, []) if tt == t]
            if earlier:
                print("\nEarlier rows of the same cases, superseded by the rows above (kept in the results files):")
                for s, c, r in earlier:
                    st = status(r)
                    med = r.get("median_s")
                    print(f"- {s} {c} ({r.get('created')}): {st}" + (f", {secs(med)}" if med else "")
                          + ("" if st == "passed" else f"; {failure(r)}"))


def failure(r):
    """One line on why a baseline row did not pass."""
    if r.get("status") == "TIMEOUT":
        return f"timed out (worker wall time {secs(r['wall_s'])})"
    if r.get("error"):
        return " ".join(str(r["error"]).split())[:200]
    fc = r.get("failed_checks") or []
    if not fc:
        return str(r.get("status")).lower()
    parts = []
    for name, v in fc[0]["failed"].items():
        if "reason" in v:
            parts.append(f"{name} ({v['reason']}: {v.get('actual_shape')} against {v.get('expected_shape')})")
        elif "max_rel_error" in v:
            parts.append(f"{name} (max. rel. error {v['max_rel_error']:.2g}, tolerance {v.get('rtol')})")
        else:
            parts.append(name)
    return f"{fc[0]['phase']} output: " + "; ".join(parts)


if __name__ == "__main__":
    main()
