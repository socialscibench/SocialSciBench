"""Generate workloads; time Stata (teffects psmatch, psmatch2, cem), R (MatchIt) and the Python
reference; check agreement.

usage: python bench.py --stata /path/to/stata-mp --out results/ [--ids 1 2 ...] [--limit 1800]
                       [--skip stata r reference]

Each Stata command runs in its own Stata process and is stopped after --limit seconds.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import pandas as pd

import reference
from workloads import CEM_COVARIATES, WORKLOADS, axes, generate

HERE = Path(__file__).resolve().parent

STATA_CMD = {
    "teffects": """
teffects psmatch (y) (d {xs}, logit), atet nneighbor(1)
timer off 1
matrix b = e(b)
file write f "att " %21.0g (b[1,1]) _n
""",
    "psmatch2": """
psmatch2 d {xs}, outcome(y) logit neighbor(1)
timer off 1
file write f "att " %21.0g (r(att)) _n
""",
    "cem": """
cem {cxs}, treatment(d)
timer off 1
file write f "matched_strata " (r(n_mstrata)) _n
qui count if cem_matched == 1 & d == 1
file write f "matched_treated " (r(N)) _n
qui count if cem_matched == 1 & d == 0
file write f "matched_controls " (r(N)) _n
qui regress y d [iweight = cem_weights]
file write f "att " %21.0g (_b[d]) _n
""",
}

STATA_DO = """
set more off
set max_memory 12g
use "{dta}", clear
file open m using "{loaded}", write replace
file close m
file open f using "{out}", write replace
timer clear
timer on 1
{cmd}
qui timer list 1
file write f "time " %15.5f (r(t1)) _n
file close f
"""


def parse(path):
    res = {}
    for line in path.read_text().splitlines():
        key, val = line.split()
        res[key] = float(val)
    return res


def run_stata(stata, method, dta, k, workdir, limit):
    out, do, loaded = workdir / f"{method}_out.txt", workdir / f"{method}.do", workdir / f"{method}_loaded.txt"
    for f in (out, loaded):
        f.unlink(missing_ok=True)
    xs = " ".join(f"x{j+1}" for j in range(k))
    cxs = " ".join(f"x{j+1}" for j in range(min(k, CEM_COVARIATES)))
    do.write_text(STATA_DO.format(dta=dta, out=out, loaded=loaded, cmd=STATA_CMD[method].format(xs=xs, cxs=cxs)))
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
            return {"status": "stopped"}
    res = parse(out) if out.exists() else {}
    if "time" not in res:
        return {"status": "error", "log_tail": (workdir / f"{method}.log").read_text(errors="ignore")[-400:]}
    return {"status": "ok", **res}


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
        if "reference" not in args.skip:
            t0 = time.perf_counter()
            p = reference.psm(d["y"], d["d"], d["X"])
            t1 = time.perf_counter()
            c = reference.cem(d["y"], d["d"], d["X"][:, :CEM_COVARIATES])
            t2 = time.perf_counter()
            row.update(ref_psm_s=round(t1 - t0, 5), ref_cem_s=round(t2 - t1, 5), psm_att=p["att"], psm_ties=p["ties"],
                       cem_att=c["att"], cem_strata=c["n_strata"], cem_matched_strata=c["n_matched_strata"],
                       cem_matched_treated=c["n_matched_treated"], cem_matched_controls=c["n_matched_controls"])

        df = pd.DataFrame({"y": d["y"], "d": d["d"], **{f"x{j+1}": d["X"][:, j] for j in range(k)}})
        if "r" not in args.skip:
            csv = out / f"w{w['id']}.csv"
            df.to_csv(csv, index=False)
            rout = out / "r_out.txt"
            rout.unlink(missing_ok=True)
            pr = subprocess.run(["Rscript", str(HERE / "bench_r.R"), str(csv), str(min(k, CEM_COVARIATES)), str(rout)],
                                cwd=out, capture_output=True, text=True)
            csv.unlink(missing_ok=True)
            if pr.returncode == 0 and rout.exists():
                r = parse(rout)
                row.update(r_status="ok", r_psm_s=r["psm_time"], r_cem_s=r["cem_time"], r_psm_att=r["psm_att"],
                           r_cem_att=r["cem_att"], r_cem_matched_strata=int(r["cem_matched_strata"]),
                           r_cem_matched_treated=int(r["cem_matched_treated"]),
                           r_cem_matched_controls=int(r["cem_matched_controls"]))
            else:
                row.update(r_status="error", r_error=(pr.stderr or pr.stdout).strip().splitlines()[-1:])

        if "stata" not in args.skip:
            dta = out / f"w{w['id']}.dta"
            df.to_stata(dta, write_index=False)
            for method in ("teffects", "psmatch2", "cem"):
                st = run_stata(args.stata, method, dta, k, out, args.limit)
                row[f"stata_{method}_status"] = st["status"]
                if st["status"] == "ok":
                    row.update({f"stata_{method}_{key}": v for key, v in st.items() if key != "status"})
                elif st["status"] == "error":
                    row[f"stata_{method}_log"] = st["log_tail"]
            dta.unlink(missing_ok=True)

        print(json.dumps(row), flush=True)
        with open(out / "results.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
