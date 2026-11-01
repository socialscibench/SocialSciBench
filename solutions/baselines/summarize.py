"""Tabulate evaluator run directories and compare.py result files (study tooling).

usage: summarize.py [--json OUT.json] RUN_DIR_OR_RESULTS.jsonl ...

For each scored case of an evaluator run: reference and candidate median/min/max of the measured
calls, speedup (reference median / candidate median), candidate first call, both workers' peak
RSS, the largest relative error of the candidate's outputs over all phases, and the status.
compare.py rows (baselines) are listed with their median, first call, peak RSS and validation.
"""
import json
import sys
from pathlib import Path


def max_rel(correctness):
    worst = 0.0
    for phase in (correctness or {}).get("candidate", []) or []:
        for check in phase.get("checks", {}).values():
            if isinstance(check, dict) and "max_rel_error" in check:
                worst = max(worst, float(check["max_rel_error"]))
    return worst


def case_name(case):
    p = case.get("params", {})
    wid = p.get("id")
    opts = p.get("loader_options") or {}
    tag = f"k{opts['n_regressors']}" if "n_regressors" in opts else ""
    if case.get("kind") == "real":
        return f"{wid}{tag}"
    return f"s{wid}" if case.get("scored") else case.get("id")


def from_run(run):
    meta = json.loads((run / "run.json").read_text())
    rows = []
    for line in (run / "traces.jsonl").read_text().splitlines():
        t = json.loads(line)
        if not t["case"].get("scored"):
            continue
        w = t.get("workers") or {}
        ref, cand = w.get("reference") or {}, w.get("candidate") or {}
        perf = t.get("performance") or {}
        rows.append(dict(
            task=meta["task"], solution=meta["solution"]["name"], suite=meta.get("suite"),
            threads=meta["config"]["threads"], warmups=meta["config"]["warmups"], repeats=meta["config"]["repeats"],
            case=case_name(t["case"]), status=t["status"], run=run.name,
            ref_median_s=(perf.get("reference") or {}).get("median_s"),
            cand_median_s=(perf.get("candidate") or {}).get("median_s"),
            cand_min_s=(perf.get("candidate") or {}).get("min_s"), cand_max_s=(perf.get("candidate") or {}).get("max_s"),
            speedup=perf.get("speedup"), cand_first_call_s=cand.get("first_call_s"),
            ref_first_call_s=ref.get("first_call_s"),
            ref_peak_rss_gib=(ref.get("peak_rss_bytes") or 0) / 2 ** 30 or None,
            cand_peak_rss_gib=(cand.get("peak_rss_bytes") or 0) / 2 ** 30 or None,
            cand_max_rel_error=max_rel(t.get("correctness"))))
    return rows


def from_results(path):
    rows = []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        rows.append(dict(task=r["task"], solution=r["solution"], suite="real-development" if not r["case"].startswith("s") else "synthetic",
                         threads=r["threads"], warmups=r["warmups"], repeats=r["repeats"], case=r["case"],
                         status=r.get("status") if r.get("status") != "COMPLETED" else ("PASSED" if r.get("checks_passed") else "FAILED"),
                         run=path.name, cand_median_s=r.get("median_s"), cand_first_call_s=r.get("first_call_s"),
                         cand_peak_rss_gib=(r.get("peak_rss_bytes") or 0) / 2 ** 30 or None,
                         failed=r.get("failed_checks"), wall_s=r.get("wall_s")))
    return rows


def main():
    args = sys.argv[1:]
    out = None
    if args[:1] == ["--json"]:
        out, args = Path(args[1]), args[2:]
    rows = []
    for a in args:
        p = Path(a)
        rows += from_run(p) if p.is_dir() else from_results(p)
    f = lambda v, fmt: "-" if v is None else format(v, fmt)
    print("| task | solution | threads | case | status | reference median | candidate median (min-max) | speedup | first call | peak RSS ref / cand | max rel. error |")
    print("|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        rng = f"{f(r.get('cand_median_s'), '.3f')} s" + (f" ({f(r.get('cand_min_s'), '.3f')}-{f(r.get('cand_max_s'), '.3f')})" if r.get("cand_min_s") is not None else "")
        print(f"| {r['task']} | {r['solution']} | {r['threads']} | {r['case']} | {r['status']} | {f(r.get('ref_median_s'), '.2f')} s | {rng} | "
              f"{f(r.get('speedup'), '.1f')} | {f(r.get('cand_first_call_s'), '.2f')} s | {f(r.get('ref_peak_rss_gib'), '.2f')} / {f(r.get('cand_peak_rss_gib'), '.2f')} GiB | "
              f"{f(r.get('cand_max_rel_error'), '.1e')} |")
    if out:
        out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
