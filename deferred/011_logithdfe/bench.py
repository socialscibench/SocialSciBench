"""Generate workloads; time Stata (logit with fixed-effect dummies + margins), R (fixest feglm +
marginaleffects) and the Python reference; check agreement.

usage: python bench.py --stata /path/to/stata-mp --out results/ [--ids 1 2 ...] [--limit 1800]
                       [--max-memory 12g] [--threads 12] [--skip stata r reference]

Stata (and the Python reference) are stopped once they have run for --limit seconds; its progress (logit
iterations, and whether margins had started) is logged with timestamps so the total time can
be estimated. Out-of-memory and other Stata errors are recorded and the run moves on. R and
the reference always run to completion.
"""
import argparse
import json
import re
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
set maxvar 120000
set max_memory {max_memory}
use "{dta}", clear
file open m using "{loaded}", write replace
file close m
timer clear
timer on 1
logit y i.d {xs} {ife}, vce(cluster cl)
timer off 1
local N = e(N)
local G = e(N_clust)
local ic = e(ic)
local b0 = _b[1.d]
local s0 = _se[1.d]
forvalues j = 1/{k} {{
    local b`j' = _b[x`j']
    local s`j' = _se[x`j']
}}
file open m using "{logit_done}", write replace
file close m
timer on 2
margins, dydx(d {xs}) post
timer off 2
local m0 = _b[1.d]
local ms0 = _se[1.d]
forvalues j = 1/{k} {{
    local m`j' = _b[x`j']
    local ms`j' = _se[x`j']
}}
qui timer list
file open f using "{out}", write replace
file write f "time_logit " %15.5f (r(t1)) _n "time_margins " %15.5f (r(t2)) _n
file write f "N `N'" _n "G `G'" _n "iterations `ic'" _n
forvalues j = 0/{k} {{
    file write f "coef " %21.15e (`b`j'') " " %21.15e (`s`j'') _n
}}
forvalues j = 0/{k} {{
    file write f "ame " %21.15e (`m`j'') " " %21.15e (`ms`j'') _n
}}
file close f
"""

OOM_CODES = {"909", "910", "950"}   # Stata's out-of-memory error codes


def parse_out(path):
    res = {"coef": [], "ame": []}
    for line in path.read_text().splitlines():
        key, *vals = line.split()
        if key in ("coef", "ame"):
            res[key].append([float(v) for v in vals])
        else:
            res[key] = float(vals[0])
    return {k: np.array(v) if k in ("coef", "ame") else v for k, v in res.items()}


def logit_iterations(log_text):
    """Logit iterations finished so far (Stata prints one 'Iteration n:' line per iteration)."""
    text = log_text[log_text.find(". logit y"):]
    return len(re.findall(r"^Iteration \d+:", text, flags=re.M))


def run_stata(stata, dta, k, n_fe, workdir, limit, max_memory):
    out, do, log = workdir / "stata_out.txt", workdir / "stata.do", workdir / "stata.log"
    loaded, logit_done = workdir / "stata_loaded.txt", workdir / "stata_logit_done.txt"
    for f in (out, log, loaded, logit_done):
        f.unlink(missing_ok=True)
    xs = " ".join(f"x{j+1}" for j in range(k))
    ife = " ".join(f"i.fe{g+1}" for g in range(n_fe))
    do.write_text(STATA_DO.format(dta=dta, out=out, loaded=loaded, logit_done=logit_done, xs=xs, ife=ife, k=k,
                                  max_memory=max_memory))
    proc = subprocess.Popen([stata, "-b", "do", str(do)], cwd=workdir)
    t_loaded = t_logit = None
    progress, last = [], 0
    while proc.poll() is None:
        time.sleep(1)
        now = time.monotonic()
        if t_loaded is None:
            if loaded.exists():
                t_loaded = now
            continue
        if t_logit is None and logit_done.exists():
            t_logit = now
        if log.exists():
            it = logit_iterations(log.read_text(errors="ignore"))
            if it > last:
                progress.append((round(now - t_loaded, 1), it))
                last = it
        if limit is not None and now - t_loaded > limit:
            proc.kill()
            proc.wait()
            subprocess.run(["pkill", "-f", str(do)], check=False)
            return {"status": "stopped", "elapsed_s": round(now - t_loaded, 1),
                    "phase": "margins" if t_logit else "logit",
                    "logit_s": round(t_logit - t_loaded, 1) if t_logit else None,
                    "iterations_done": last, "progress": progress}
    log_text = log.read_text(errors="ignore") if log.exists() else ""
    if not out.exists():
        err = re.findall(r"^r\((\d+)\);", log_text, flags=re.M)
        msg = ""
        if err:
            # Error message: first line after the command that failed
            block = log_text[:log_text.find(f"r({err[0]});")].splitlines()
            cmd = max(i for i, line in enumerate(block) if line.startswith(". "))
            rest = [line.strip() for line in block[cmd + 1:] if line.strip() and not line.startswith(">")]
            msg = rest[0] if rest else ""
        oom = bool(err) and err[0] in OOM_CODES
        elapsed = round(time.monotonic() - t_loaded, 1) if t_loaded else None
        return {"status": "out_of_memory" if oom else "error", "error_code": err[0] if err else None,
                "error_message": msg, "elapsed_s": elapsed, "phase": "margins" if t_logit else "logit",
                "iterations_done": logit_iterations(log_text), "progress": progress}
    return {"status": "ok", **parse_out(out)}


def run_r(csv, k, workdir, threads):
    out = workdir / "r_out.txt"
    out.unlink(missing_ok=True)
    p = subprocess.run(["Rscript", str(HERE / "bench_fixest.R"), str(csv), str(k), str(out), str(threads)],
                       cwd=workdir, capture_output=True, text=True)
    if p.returncode != 0 or not out.exists():
        return {"status": "error", "error_message": (p.stderr or p.stdout).strip().splitlines()[-1:]}
    res = parse_out(out)
    return {"status": "ok", **res}


def run_reference(w_id, workdir, limit):
    """Run the reference in a separate process (run_reference.py); stop it after `limit` seconds."""
    import sys
    out = workdir / "ref_out.json"
    out.unlink(missing_ok=True)
    try:
        subprocess.run([sys.executable, str(HERE / "run_reference.py"), str(w_id), str(out)], check=True,
                       timeout=limit, cwd=HERE)
    except subprocess.TimeoutExpired:
        return None
    r = json.loads(out.read_text())
    return {k: np.array(v) if k in ("beta", "se", "ame", "ame_se") else v for k, v in r.items()}


def rel_err(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stata", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--ids", type=int, nargs="*")
    ap.add_argument("--limit", type=float, default=1800, help="stop Stata after this many seconds")
    ap.add_argument("--max-memory", default="12g", help="Stata max_memory, so memory errors are reported cleanly")
    ap.add_argument("--threads", type=int, default=12, help="threads for fixest")
    ap.add_argument("--skip", nargs="*", default=[], choices=["stata", "r", "reference"])
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    for w in WORKLOADS:
        if args.ids and w["id"] not in args.ids:
            continue
        d = generate(w)
        k = d["X"].shape[1]                                  # continuous covariates (plus d)
        n_fe = d["fe_ids"].shape[1]
        row = dict(id=w["id"], design=w["design"], **axes(w, d))
        ref = None
        if "reference" not in args.skip:
            ref = run_reference(w["id"], out, args.limit)
            if ref is None:
                row.update(reference_status="stopped", reference_elapsed_s=args.limit)
        if ref is not None:
            row.update(reference_status="ok", reference_s=round(sum(ref["timings"].values()), 5),
                       reference_estimation_s=round(ref["timings"]["estimation"], 5),
                       reference_ame_s=round(ref["timings"]["ame"], 5),
                       reference_iterations=ref["iterations"], ref_nobs=int(ref["nobs"]),
                       ref_n_clusters=int(ref["n_clusters"]), fe_levels_used=ref["fe_levels"])

        df = pd.DataFrame({"y": d["y"], "d": d["d"], "cl": d["cluster_ids"] + 1})
        for j in range(k):
            df[f"x{j+1}"] = d["X"][:, j]
        for g in range(n_fe):
            df[f"fe{g+1}"] = d["fe_ids"][:, g] + 1

        if "r" not in args.skip:
            csv = out / f"w{w['id']}.csv"
            df.to_csv(csv, index=False)
            r = run_r(csv, k, out, args.threads)
            csv.unlink(missing_ok=True)
            row["r_status"] = r["status"]
            if r["status"] == "ok":
                row.update(r_feglm_s=r["time_feglm"], r_ame_s=r["time_ame"],
                           r_total_s=round(r["time_feglm"] + r["time_ame"], 5), r_nobs=int(r["N"]),
                           r_iterations=int(r["iterations"]))
                if ref is not None:
                    row.update(r_max_rel_err_beta=rel_err(r["coef"][:, 0], ref["beta"]),
                               r_max_rel_err_se=rel_err(r["coef"][:, 1], ref["se"]),
                               r_max_rel_err_ame=rel_err(r["ame"][:, 0], ref["ame"]))
            else:
                row["r_error"] = r["error_message"]

        if "stata" not in args.skip:
            dta = out / f"w{w['id']}.dta"
            df.to_stata(dta, write_index=False)
            st = run_stata(args.stata, dta, k, n_fe, out, args.limit, args.max_memory)
            dta.unlink(missing_ok=True)
            row["stata_status"] = st["status"]
            if st["status"] == "ok":
                row.update(stata_logit_s=st["time_logit"], stata_margins_s=st["time_margins"],
                           stata_total_s=round(st["time_logit"] + st["time_margins"], 5),
                           stata_nobs=int(st["N"]), stata_n_clusters=int(st["G"]),
                           stata_iterations=int(st["iterations"]))
                if ref is not None:
                    row.update(stata_max_rel_err_beta=rel_err(st["coef"][:, 0], ref["beta"]),
                               stata_max_rel_err_se=rel_err(st["coef"][:, 1], ref["se"]),
                               stata_max_rel_err_ame=rel_err(st["ame"][:, 0], ref["ame"]),
                               stata_max_rel_err_ame_se=rel_err(st["ame"][:, 1], ref["ame_se"]))
            else:
                row.update({f"stata_{key}": v for key, v in st.items() if key != "status"})
                print(json.dumps({"id": w["id"], "stata": st}), flush=True)

        print(json.dumps(row), flush=True)
        with open(out / "results.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
