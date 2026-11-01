"""Generate workloads; time Stata melogit, R lme4::glmer (nAGQ = 7) and the Python reference; check agreement.

usage: python bench.py --stata /path/to/stata-mp --out results/ [--ids 1 2 ...] [--limit 1800]
                       [--skip stata r reference]

Stata and R are stopped after --limit seconds. A stopped run is then timed on a 10% sample (10% of
the groups, or 10% of the observations in every group when there are fewer than 1,000 groups) and
its full time is estimated as (sample time) x (n_obs / sample n_obs).
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

import reference
from workloads import WORKLOADS, axes, generate

HERE = Path(__file__).resolve().parent

STATA_DO = """
set more off
set max_memory 12g
use "{dta}", clear
file open m using "{loaded}", write replace
file close m
timer clear
timer on 1
melogit y {xs} || g:
timer off 1
qui timer list 1
file open f using "{out}", write replace
file write f "time " %15.5f (r(t1)) _n "N " (e(N)) _n "ll " %21.0g (e(ll)) _n "converged " (e(converged)) _n
file write f "var_u " %21.0g (_b[/var(_cons[g])]) " " %21.0g (_se[/var(_cons[g])]) _n
foreach x in {xs} _cons {{
    file write f "coef " %21.0g (_b[`x']) " " %21.0g (_se[`x']) _n
}}
file close f
"""


def parse(path):
    vals = {"coef": []}
    for line in path.read_text().splitlines():
        key, *v = line.split()
        if key == "coef":
            vals["coef"].append([float(x) for x in v])
        else:
            vals[key] = [float(x) for x in v] if len(v) > 1 else float(v[0])
    vals["coef"] = np.array(vals["coef"])
    return vals


def run_stata(stata, dta, k, workdir, limit):
    out, do, log, loaded = workdir / "stata_out.txt", workdir / "stata.do", workdir / "stata.log", workdir / "loaded.txt"
    for f in (out, log, loaded):
        f.unlink(missing_ok=True)
    xs = " ".join(f"x{j + 1}" for j in range(k))
    do.write_text(STATA_DO.format(dta=dta, loaded=loaded, out=out, xs=xs))
    proc = subprocess.Popen([stata, "-b", "do", do.name], cwd=workdir)   # relative: the path may contain spaces
    t_loaded = None
    while proc.poll() is None:
        time.sleep(1)
        now = time.monotonic()
        if t_loaded is None:
            if loaded.exists():
                t_loaded = now
            continue
        if now - t_loaded > limit:
            proc.kill()
            proc.wait()
            subprocess.run(["pkill", "-f", "stata.do"], check=False)
            return {"status": "stopped"}
    if not out.exists():
        tail = log.read_text(errors="ignore")[-600:] if log.exists() else ""
        return {"status": "error", "log_tail": tail}
    return {"status": "ok", **parse(out)}


def run_r(csv, k, workdir, limit):
    rout = workdir / "r_out.txt"
    rout.unlink(missing_ok=True)
    try:
        p = subprocess.run(["Rscript", str(HERE / "bench_r.R"), str(csv), str(k), str(rout)], cwd=workdir,
                           capture_output=True, text=True, timeout=limit)
    except subprocess.TimeoutExpired:
        subprocess.run(["pkill", "-f", "bench_r.R"], check=False)
        time.sleep(2)
        return {"status": "stopped"}
    if p.returncode == 0 and rout.exists():
        return {"status": "ok", **parse(rout)}
    return {"status": "error", "error": (p.stderr or p.stdout).strip().splitlines()[-3:]}


def agreement(res, ref, prefix):
    """Coefficient differences in units of the reference standard error, and relative differences."""
    c = res["coef"]
    out = {f"{prefix}_max_beta_diff_in_se": float(np.max(np.abs(c[:, 0] - ref["beta"]) / ref["se"])),
           f"{prefix}_max_rel_err_se": float(np.max(np.abs(c[:, 1] - ref["se"]) / ref["se"])),
           f"{prefix}_rel_err_var_u": float(abs(np.atleast_1d(res["var_u"])[0] - ref["var_u"]) / ref["var_u"]),
           f"{prefix}_ll_diff": float(res["ll"] - ref["ll"])}
    if isinstance(res["var_u"], list):
        out[f"{prefix}_rel_err_se_var_u"] = float(abs(res["var_u"][1] - ref["se_var_u"]) / ref["se_var_u"])
    return out


def frame(d, rows=None):
    k = d["X"].shape[1]
    df = pd.DataFrame({"y": d["y"], **{f"x{j + 1}": d["X"][:, j] for j in range(k)}, "g": d["group"] + 1})
    return df if rows is None else df.iloc[rows]


def sample_rows(d, seed):
    """10% of the groups, or 10% of the observations in each group if there are fewer than 1,000 groups."""
    rng = np.random.default_rng(seed)
    g = d["group"]
    G = g.max() + 1
    if G >= 1000:
        keep = rng.random(G) < 0.1
        return np.flatnonzero(keep[g])
    return np.flatnonzero(rng.random(len(g)) < 0.1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stata", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--ids", type=int, nargs="*")
    ap.add_argument("--limit", type=float, default=1800)
    ap.add_argument("--skip", nargs="*", default=[], choices=["stata", "r", "reference"])
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    for w in WORKLOADS:
        if args.ids and w["id"] not in args.ids:
            continue
        d = generate(w)
        k = d["X"].shape[1]
        row = dict(id=w["id"], design=w["design"], **axes(w, d))
        ref = None
        if "reference" not in args.skip:
            t0 = time.perf_counter()
            ref = reference.run(d["y"], d["X"], d["group"])
            row.update(reference_s=round(time.perf_counter() - t0, 5), ref_iterations=ref["iterations"],
                       ref_ll=ref["ll"], ref_var_u=ref["var_u"], ref_se_var_u=ref["se_var_u"],
                       ref_beta=ref["beta"].tolist(), ref_se=ref["se"].tolist())
        sample = sample_rows(d, w["seed"])
        row["calibration_n_obs"] = int(len(sample))

        if "r" not in args.skip:
            csv, csv_s = out / "data.csv", out / "data_sample.csv"
            frame(d).to_csv(csv, index=False)
            r = run_r(csv, k, out, args.limit)
            row["r_status"] = r["status"]
            if r["status"] == "ok":
                row.update(r_s=r["time"], r_warnings=int(r["warnings"]), r_ll=r["ll"], r_var_u=r["var_u"],
                           r_coef=r["coef"].tolist())
                if ref:
                    row.update(agreement(r, ref, "r"))
            elif r["status"] == "stopped":
                frame(d, sample).to_csv(csv_s, index=False)
                rc = run_r(csv_s, k, out, args.limit)
                if rc["status"] == "ok":
                    row.update(r_cal_s=rc["time"], r_est_s=round(rc["time"] * len(d["y"]) / len(sample), 1))
                csv_s.unlink(missing_ok=True)
            else:
                row["r_error"] = r["error"]
            csv.unlink(missing_ok=True)

        if "stata" not in args.skip:
            dta, dta_s = out / "data.dta", out / "data_sample.dta"
            frame(d).to_stata(dta, write_index=False)
            st = run_stata(args.stata, dta, k, out, args.limit)
            row["stata_status"] = st["status"]
            if st["status"] == "ok":
                row.update(stata_s=st["time"], stata_converged=int(st["converged"]), stata_ll=st["ll"],
                           stata_var_u=st["var_u"], stata_coef=st["coef"].tolist())
                if ref:
                    row.update(agreement(st, ref, "stata"))
            elif st["status"] == "stopped":
                frame(d, sample).to_stata(dta_s, write_index=False)
                sc = run_stata(args.stata, dta_s, k, out, args.limit)
                if sc["status"] == "ok":
                    row.update(stata_cal_s=sc["time"],
                               stata_est_s=round(sc["time"] * len(d["y"]) / len(sample), 1))
                dta_s.unlink(missing_ok=True)
            else:
                row["stata_log_tail"] = st["log_tail"]
            dta.unlink(missing_ok=True)

        print(json.dumps(row), flush=True)
        with open(out / "results.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
