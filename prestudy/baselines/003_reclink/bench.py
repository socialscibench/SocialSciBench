"""Generate workloads; time Stata (reclink2, reclink), R (port) and the Python reference; check agreement.

usage: python bench.py --stata /path/to/stata-mp --out results/ [--ids 1 2 ...] [--limit 1800]
                       [--skip reclink2 reclink r reference] [--n-cal 200]

Each Stata command runs in its own Stata process and is stopped after --limit seconds. A stopped
command is then timed on a random sample of --n-cal master records (same using file), and its full
time is estimated as (sample time) x (n_master / n_cal). reclink is not run in full on workloads
where reclink2 (the same algorithm) was stopped; it is only timed on the sample.
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
sys.path.insert(0, str(ROOT / "003_reclink"))

import reference
from workloads import MINSCORE, REQUIRED, VARS, WMATCH, WNOMATCH, WORKLOADS, axes, generate

HERE = Path(__file__).resolve().parent

STATA_DO = """
set more off
set max_memory 12g
use "{master}", clear
file open m using "{loaded}", write replace
file close m
timer clear
timer on 1
{cmd} {vars} using "{using}", idmaster(idm) idusing(idu) gen(score) wmatch({wm}) wnomatch({wnm}) required({req})
timer off 1
qui timer list 1
local t = r(t1)
qui keep if !missing(idu)
local n = _N
qui summarize score, meanonly
local s = r(sum)
qui summarize idm, meanonly
local im = r(sum)
qui summarize idu, meanonly
local iu = r(sum)
qui bysort idm: gen byte first = _n == 1
qui count if first
local nm = r(N)
qui count if score == 1
local np = r(N)
file open f using "{out}", write replace
file write f "time " %15.5f (`t') _n "pairs `n'" _n "matched_master `nm'" _n "score1 `np'" _n
file write f "score_sum " %21.0g (`s') _n "id_master_sum " %21.0g (`im') _n "id_using_sum " %21.0g (`iu') _n
file close f
"""


def parse(path):
    return {k: float(v) for k, v in (line.split() for line in path.read_text().splitlines())}


def run_stata(stata, cmd, master, using, workdir, limit):
    out, do, log, loaded = (workdir / f"{cmd}_out.txt", workdir / f"{cmd}.do", workdir / f"{cmd}.log",
                            workdir / f"{cmd}_loaded.txt")
    for f in (out, log, loaded):
        f.unlink(missing_ok=True)
    do.write_text(STATA_DO.format(cmd=cmd, master=master, using=using, loaded=loaded, out=out, vars=" ".join(VARS),
                                  wm=" ".join(map(str, WMATCH)), wnm=" ".join(map(str, WNOMATCH)),
                                  req=" ".join(REQUIRED)))
    proc = subprocess.Popen([stata, "-b", "do", do.name], cwd=workdir)       # relative: the path may contain spaces
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
            subprocess.run(["pkill", "-f", str(do)], check=False)
            return {"status": "stopped"}
    if not out.exists():
        return {"status": "error", "log_tail": log.read_text(errors="ignore")[-400:] if log.exists() else ""}
    return {"status": "ok", **parse(out)}


def compare(a, ref):
    exact = all(int(a[k]) == ref[k] for k in ("pairs", "matched_master", "id_master_sum", "id_using_sum"))
    return exact, abs(a["score_sum"] - ref["score_sum"]) / ref["score_sum"]


def write_dta(d, n_master, out, sample=None):
    m = pd.DataFrame({"idm": np.arange(1, n_master + 1, dtype=np.int32), **d["master"]})
    if sample is not None:
        m = m.iloc[sample]
    m.to_stata(out, write_index=False, version=118)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stata", required=True)
    ap.add_argument("--out", default=str(ROOT / "runs" / ("baseline-003_reclink-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])))
    ap.add_argument("--ids", type=int, nargs="*")
    ap.add_argument("--limit", type=float, default=1800)
    ap.add_argument("--n-cal", type=int, default=200)
    ap.add_argument("--skip", nargs="*", default=[], choices=["reclink2", "reclink", "r", "reference"])
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    for w in WORKLOADS:
        if args.ids and w["id"] not in args.ids:
            continue
        d = generate(w)
        nm = w["n_master"]
        row = dict(id=w["id"], design=w["design"], **axes(w, d))
        ref = None
        if "reference" not in args.skip:
            t0 = time.perf_counter()
            i, j, s, n_perf = reference.run(d["master"], d["using"], VARS, WMATCH, WNOMATCH, REQUIRED)
            row["reference_s"] = round(time.perf_counter() - t0, 5)
            ref = reference.checksums(i, j, s, n_perf, d["truth"])
            row.update({f"ref_{k}": v for k, v in ref.items()})

        if "r" not in args.skip:
            mc, uc, rout = out / "master.csv", out / "using.csv", out / "r_out.txt"
            pd.DataFrame(d["master"]).to_csv(mc, index=False)
            pd.DataFrame(d["using"]).to_csv(uc, index=False)
            rout.unlink(missing_ok=True)
            try:
                p = subprocess.run(["Rscript", str(HERE / "bench_r.R"), str(mc), str(uc), str(rout)],
                                   cwd=out, capture_output=True, text=True, timeout=args.limit)
            except subprocess.TimeoutExpired:
                p = None
                row["r_status"] = "stopped"
                subprocess.run(["pkill", "-f", "bench_r.R"], check=False)
                time.sleep(2)
            if p is None:
                pass
            elif p.returncode == 0 and rout.exists():
                r = parse(rout)
                row.update(r_status="ok", r_s=r["time"], r_pairs=int(r["pairs"]), r_perfect=int(r["perfect"]))
                if ref:
                    row["r_exact"], row["r_score_rel_diff"] = compare(r, ref)
            else:
                row.update(r_status="error", r_error=(p.stderr or p.stdout).strip().splitlines()[-3:])
            mc.unlink(missing_ok=True)
            uc.unlink(missing_ok=True)

        cmds = [c for c in ("reclink2", "reclink") if c not in args.skip]
        if cmds:
            mdta, udta, cdta = out / "master.dta", out / "using.dta", out / "master_cal.dta"
            write_dta(d, nm, mdta)
            pd.DataFrame({"idu": np.arange(1, w["n_using"] + 1, dtype=np.int32), **d["using"]}).to_stata(
                udta, write_index=False, version=118)
            n_cal = min(args.n_cal, nm)
            write_dta(d, nm, cdta, np.sort(np.random.default_rng(w["seed"]).choice(nm, n_cal, replace=False)))
            for cmd in cmds:
                if cmd == "reclink" and row.get("reclink2_status") == "stopped":
                    st = {"status": "not run"}
                else:
                    st = run_stata(args.stata, cmd, mdta, udta, out, args.limit)
                row[f"{cmd}_status"] = st["status"]
                if st["status"] == "ok":
                    row[f"{cmd}_s"] = st["time"]
                    if ref:
                        row[f"{cmd}_exact"], row[f"{cmd}_score_rel_diff"] = compare(st, ref)
                        row[f"{cmd}_pairs"] = int(st["pairs"])
                elif st["status"] in ("stopped", "not run"):
                    cal = run_stata(args.stata, cmd, cdta, udta, out, args.limit)
                    if cal["status"] == "ok":
                        row.update({f"{cmd}_cal_n": n_cal, f"{cmd}_cal_s": cal["time"],
                                    f"{cmd}_est_s": round(cal["time"] * nm / n_cal, 1)})
                else:
                    row[f"{cmd}_log_tail"] = st.get("log_tail", "")
            for f in (mdta, udta, cdta):
                f.unlink(missing_ok=True)

        print(json.dumps(row), flush=True)
        with open(out / "results.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
