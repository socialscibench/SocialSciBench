"""Time R fixest::feols on the task 001 workloads and compare it with the Python reference.

usage: python bench_fixest.py --out results/ [--ids 1 3 ...] [--threads 12]

Writes one JSON line per workload to <out>/results_fixest.jsonl.
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import sys
from datetime import datetime, timezone
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "001_reghdfe"))

import reference
from workloads import WORKLOADS, generate

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "runs" / ("baseline-001_reghdfe-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])))
    ap.add_argument("--ids", type=int, nargs="*")
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    for w in WORKLOADS:
        if args.ids and w["id"] not in args.ids:
            continue
        d = generate(w)
        k, G = d["X"].shape[1], d["fe_ids"].shape[1]
        ref = reference.run(**d)
        df = pd.DataFrame({"y": d["y"], "cl": d["cluster_ids"] + 1,
                           **{f"x{j+1}": d["X"][:, j] for j in range(k)},
                           **{f"fe{g+1}": d["fe_ids"][:, g] + 1 for g in range(G)}})
        csv, rout = out / f"w{w['id']}.csv", out / "fixest_out.txt"
        df.to_csv(csv, index=False)
        rout.unlink(missing_ok=True)
        p = subprocess.run(["Rscript", str(HERE / "bench_fixest.R"), str(csv), str(k), str(rout), str(args.threads)],
                           capture_output=True, text=True)
        csv.unlink(missing_ok=True)
        row = dict(id=w["id"])
        if p.returncode == 0 and rout.exists():
            vals = {"coef": []}
            for line in rout.read_text().splitlines():
                key, *v = line.split()
                if key == "coef":
                    vals["coef"].append([float(x) for x in v])
                else:
                    vals[key] = float(v[0])
            c = np.array(vals["coef"])
            se_ref = np.sqrt(np.diag(ref["vcov"]))
            row.update(fixest_status="ok", fixest_s=vals["time"], fixest_nobs=int(vals["N"]), ref_nobs=int(ref["nobs"]),
                       fixest_max_rel_err_beta=float(np.max(np.abs(c[:, 0] - ref["beta"]) / np.abs(ref["beta"]))),
                       fixest_max_rel_err_se=float(np.max(np.abs(c[:, 1] - se_ref) / se_ref)))
        else:
            row.update(fixest_status="error", fixest_error=(p.stderr or p.stdout).strip().splitlines()[-1:])
        print(json.dumps(row), flush=True)
        with open(out / "results_fixest.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
