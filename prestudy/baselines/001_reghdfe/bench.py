"""Generate workloads, time Stata reghdfe (baseline) and the Python reference, check agreement.

usage: python bench.py --stata /path/to/stata-mp --out results/ [--ids 1 2 ...] [--skip-reference]
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

import sys
from datetime import datetime, timezone
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "001_reghdfe"))

import reference
from workloads import WORKLOADS, axes, generate

STATA_DO = """
set more off
sysuse auto, clear
qui reghdfe price weight, absorb(foreign rep78) vce(cluster rep78)
use "{dta}", clear
timer clear 1
timer on 1
reghdfe y {xs}, absorb({fes}) vce(cluster cl)
timer off 1
qui timer list 1
file open f using "{out}", write replace
file write f "time " (r(t1)) _n "N " (e(N)) _n "df_a " (e(df_a)) _n "N_clust " (e(N_clust)) _n
matrix b = e(b)
matrix V = e(V)
forvalues i = 1/{k} {{
    file write f "b " %21.15e (b[1,`i']) _n
    forvalues j = 1/{k} {{
        file write f "V " %21.15e (V[`i',`j']) _n
    }}
}}
file close f
"""


def run_stata(stata, d, workdir):
    k, nfe = d["X"].shape[1], d["fe_ids"].shape[1]
    df = pd.DataFrame({"y": d["y"], "cl": d["cluster_ids"]})
    for j in range(k):
        df[f"x{j+1}"] = d["X"][:, j]
    for g in range(nfe):
        df[f"fe{g+1}"] = d["fe_ids"][:, g]
    dta, out, do = workdir / "w.dta", workdir / "w_out.txt", workdir / "w.do"
    df.to_stata(dta, write_index=False)
    do.write_text(STATA_DO.format(dta=dta, out=out, k=k,
                                  xs=" ".join(f"x{j+1}" for j in range(k)),
                                  fes=" ".join(f"fe{g+1}" for g in range(nfe))))
    subprocess.run([stata, "-b", "do", str(do)], cwd=workdir, check=True)
    res = {"b": [], "V": []}
    for line in out.read_text().splitlines():
        key, val = line.split()
        if key in ("b", "V"):
            res[key].append(float(val))
        else:
            res[key] = float(val)
    res["b"] = np.array(res["b"])
    res["V"] = np.array(res["V"]).reshape(k, k)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stata", required=True)
    ap.add_argument("--out", default=str(ROOT / "runs" / ("baseline-001_reghdfe-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])))
    ap.add_argument("--ids", type=int, nargs="*")
    ap.add_argument("--skip-reference", action="store_true")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    for w in WORKLOADS:
        if args.ids and w["id"] not in args.ids:
            continue
        d = generate(w)
        row = dict(id=w["id"], design=w["design"], **axes(w, d))
        st = run_stata(args.stata, d, out)
        row.update(stata_s=st["time"], stata_nobs=int(st["N"]), stata_df_a=int(st["df_a"]))
        if not args.skip_reference:
            t0 = time.perf_counter()
            ref = reference.run(**d)
            row["reference_s"] = time.perf_counter() - t0
            se_ref, se_st = np.sqrt(np.diag(ref["vcov"])), np.sqrt(np.diag(st["V"]))
            row.update(ref_nobs=int(ref["nobs"]), ref_df_a=int(ref["df_a"]),
                       max_rel_err_beta=float(np.max(np.abs(ref["beta"] - st["b"]) / np.abs(st["b"]))),
                       max_rel_err_se=float(np.max(np.abs(se_ref - se_st) / se_st)))
        print(json.dumps(row), flush=True)
        with open(out / "results.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
