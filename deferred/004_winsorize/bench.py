"""Generate workloads; time Stata (winsor2), R (data.table + quantile type 2) and the Python reference;
check agreement.

usage: python bench.py --stata /path/to/stata-mp --out results/ [--ids 1 2 ...] [--limit 1800]
                       [--threads 12] [--skip stata r reference]

Stata and the reference are stopped after --limit seconds. Each tool reports, per variable,
the count, sum, sum of squares, min and max of the winsorized values and how many values
were changed; these are compared with the reference.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import reference
from workloads import WORKLOADS, axes, generate

HERE = Path(__file__).resolve().parent

STATA_DO = """
set more off
set max_memory {max_memory}
use "{dta}", clear
foreach v of varlist v* {{
    qui clonevar o_`v' = `v'
}}
file open m using "{loaded}", write replace
file close m
timer clear
timer on 1
winsor2 {vars}, cuts({p_lo} {p_hi}) replace {by}
timer off 1
qui timer list 1
file open f using "{out}", write replace
file write f "time " %15.5f (r(t1)) _n
foreach v of varlist {vars} {{
    qui summarize `v', meanonly
    local n = r(N)
    local s = r(sum)
    local mn = r(min)
    local mx = r(max)
    tempvar sq
    qui gen double `sq' = `v'^2
    qui summarize `sq', meanonly
    local ss = r(sum)
    qui count if `v' != o_`v' & !missing(`v')
    file write f "check `n' " %21.0g (`s') " " %21.0g (`ss') " " %21.0g (`mn') " " %21.0g (`mx') " " (r(N)) _n
    drop `sq'
}}
file close f
"""


def parse(path):
    res = {"check": []}
    for line in path.read_text().splitlines():
        key, *vals = line.split()
        if key == "check":
            n, s, ss, mn, mx, ch = vals
            res["check"].append(dict(n=int(float(n)), sum=float(s), sumsq=float(ss), min=float(mn), max=float(mx),
                                     changed=int(float(ch))))
        else:
            res[key] = float(vals[0])
    return res


def run_stata(stata, dta, k, has_group, cuts, workdir, limit, max_memory):
    out, do, loaded = workdir / "stata_out.txt", workdir / "stata.do", workdir / "stata_loaded.txt"
    for f in (out, loaded):
        f.unlink(missing_ok=True)
    vars_ = " ".join(f"v{j+1}" for j in range(k))
    do.write_text(STATA_DO.format(dta=dta, out=out, loaded=loaded, vars=vars_, p_lo=cuts[0], p_hi=cuts[1],
                                  by="by(g)" if has_group else "", max_memory=max_memory))
    proc = subprocess.Popen([stata, "-b", "do", str(do)], cwd=workdir)
    t_loaded = None
    while proc.poll() is None:
        time.sleep(0.5)
        if t_loaded is None and loaded.exists():
            t_loaded = time.monotonic()
        if t_loaded is not None and time.monotonic() - t_loaded > limit:
            proc.kill()
            proc.wait()
            subprocess.run(["pkill", "-f", str(do)], check=False)
            return {"status": "stopped", "elapsed_s": round(time.monotonic() - t_loaded, 1)}
    if not out.exists():
        log = (workdir / "stata.log").read_text(errors="ignore")
        return {"status": "error", "log_tail": log[-500:]}
    return {"status": "ok", **parse(out)}


def run_r(csv, k, has_group, cuts, workdir, threads):
    out = workdir / "r_out.txt"
    out.unlink(missing_ok=True)
    p = subprocess.run(["Rscript", str(HERE / "bench_r.R"), str(csv), str(k), "1" if has_group else "0",
                        str(cuts[0]), str(cuts[1]), str(out), str(threads)], cwd=workdir, capture_output=True,
                       text=True)
    if p.returncode != 0 or not out.exists():
        return {"status": "error", "error": (p.stderr or p.stdout).strip().splitlines()[-1:]}
    return {"status": "ok", **parse(out)}


def compare(a, b):
    """Max relative difference over sums/min/max; whether non-missing counts match; and the largest
    per-variable difference in the number of values changed by winsorizing."""
    rel = max(abs(x[key] - y[key]) / max(abs(y[key]), 1e-300) for x, y in zip(a, b) for key in ("sum", "sumsq", "min", "max"))
    n_match = all(x["n"] == y["n"] for x, y in zip(a, b))
    changed_diff = max(abs(x["changed"] - y["changed"]) for x, y in zip(a, b))
    return rel, n_match, changed_diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stata", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--ids", type=int, nargs="*")
    ap.add_argument("--limit", type=float, default=1800)
    ap.add_argument("--max-memory", default="12g")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--skip", nargs="*", default=[], choices=["stata", "r", "reference"])
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    for w in WORKLOADS:
        if args.ids and w["id"] not in args.ids:
            continue
        d = generate(w)
        k, has_group = d["X"].shape[1], d["group"] is not None
        row = dict(id=w["id"], design=w["design"], **axes(w, d))
        ref_check = None
        if "reference" not in args.skip:
            t0 = time.perf_counter()
            res = reference.run(**d)
            row["reference_s"] = round(time.perf_counter() - t0, 5)
            ref_check = reference.checksums(d["X"], res["X_w"])
            row["changed_share"] = round(sum(c["changed"] for c in ref_check) / sum(c["n"] for c in ref_check), 4)

        df = pd.DataFrame({f"v{j+1}": d["X"][:, j] for j in range(k)})
        if has_group:
            df.insert(0, "g", d["group"] + 1)

        if "r" not in args.skip:
            csv = out / f"w{w['id']}.csv"
            df.to_csv(csv, index=False)
            r = run_r(csv, k, has_group, d["cuts"], out, args.threads)
            csv.unlink(missing_ok=True)
            row["r_status"] = r["status"]
            if r["status"] == "ok":
                row["r_s"] = r["time"]
                if ref_check:
                    row["r_max_rel_diff"], row["r_n_match"], row["r_changed_diff"] = compare(r["check"], ref_check)
            else:
                row["r_error"] = r["error"]

        if "stata" not in args.skip:
            dta = out / f"w{w['id']}.dta"
            df.to_stata(dta, write_index=False)
            st = run_stata(args.stata, dta, k, has_group, d["cuts"], out, args.limit, args.max_memory)
            dta.unlink(missing_ok=True)
            row["stata_status"] = st["status"]
            if st["status"] == "ok":
                row["stata_s"] = st["time"]
                if ref_check:
                    row["stata_max_rel_diff"], row["stata_n_match"], row["stata_changed_diff"] = compare(st["check"], ref_check)
            else:
                row.update({f"stata_{key}": v for key, v in st.items() if key != "status"})

        print(json.dumps(row), flush=True)
        with open(out / "results.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
