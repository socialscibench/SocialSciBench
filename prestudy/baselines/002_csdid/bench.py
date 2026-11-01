"""Generate workloads, time Stata csdid and csdid2 (baselines) and the Python reference, check agreement.

usage: python bench.py --stata /path/to/stata-mp --out results/ [--ids 1 2 ...]
                       [--engines csdid csdid2] [--limit 1800] [--limit-engines csdid]
                       [--max-memory 12g] [--skip-reference]

Commands listed in --limit-engines (default: csdid only) are stopped once they have run for
--limit seconds; their progress (cells finished, from the progress dots Stata prints) is
logged with timestamps so the total time can be estimated. Other commands (csdid2) and the
Python reference always run to completion. Out-of-memory errors are recorded and the run
moves on.
"""
import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

import sys
from datetime import datetime, timezone
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "002_csdid"))

import reference
from workloads import WORKLOADS, axes, generate

# Estimation step for each Stata command. Each leaves the ATT(g,t) results in b_gt/V_gt;
# csdid2 is run with `short` so its base period matches csdid's default.
STATA_CMD = {
    "csdid": """
csdid y {xs}, ivar(id) time(t) gvar(g) method(dripw)
timer off 1
matrix b_gt = e(b)
matrix V_gt = e(V)
timer on 2
qui estat simple
matrix b_s = r(b)
matrix V_s = r(V)
qui estat event, post
matrix b_e = e(b)
matrix V_e = e(V)
""",
    "csdid2": """
csdid2 y {xs}, ivar(id) time(t) gvar(g) method(dripw) short
timer off 1
timer on 2
qui estat attgt
matrix b_gt = r(b)
matrix V_gt = r(V)
qui estat simple
matrix b_s = r(b)
matrix V_s = r(V)
qui estat event
matrix b_e = r(b)
matrix V_e = r(V)
""",
}

STATA_DO = """
set more off
set max_memory {max_memory}
use "{dta}", clear
file open m using "{loaded}", write replace
file close m
timer clear
timer on 1
{cmd}
timer off 2
qui timer list
local names : colnames b_e
file open f using "{out}", write replace
file write f "time_cmd " %15.5f (r(t1)) _n "time_estat " %15.5f (r(t2)) _n
forvalues i = 1/{n_gt} {{
    file write f "gt " %21.15e (b_gt[1,`i']) " " %21.15e (sqrt(V_gt[`i',`i'])) _n
}}
file write f "simple " %21.15e (b_s[1,1]) " " %21.15e (sqrt(V_s[1,1])) _n
forvalues i = 1/`=colsof(b_e)' {{
    file write f "event_`:word `i' of `names'' " %21.15e (b_e[1,`i']) " " %21.15e (sqrt(V_e[`i',`i'])) _n
}}
file close f
"""

OOM_PATTERNS = ("r(909)", "r(910)", "r(950)", "op. sys. refuses to provide memory", "no room to add more",
                "insufficient memory", "max_memory")


def event_name(e):
    return f"tm{-e}" if e < 0 else f"tp{e}"


def write_dta(d, path):
    N, T = d["y"].shape
    df = pd.DataFrame({"id": np.repeat(np.arange(1, N + 1), T), "t": np.tile(np.arange(1, T + 1), N),
                       "y": d["y"].ravel(), "g": np.repeat(d["gvar"], T)})
    for j in range(d["X"].shape[1]):
        df[f"x{j+1}"] = np.repeat(d["X"][:, j], T)
    df.to_stata(path, write_index=False)


def progress_cells(log_text, cmd):
    """Cells finished so far: progress dots printed after the estimation command (complete lines only)."""
    text = log_text[log_text.find(f". {cmd} y"):]
    return sum(line.count(".") + line.count("x") for line in text.splitlines()[1:]
               if line.strip() and set(line.strip()) <= {".", "x"})


def run_stata(stata, engine, dta, d, n_gt, workdir, limit, max_memory):
    """Run one Stata command. Returns parsed results, or a status dict if stopped or failed."""
    k = d["X"].shape[1]
    out, do, log, loaded = (workdir / f"{engine}_out.txt", workdir / f"{engine}.do",
                            workdir / f"{engine}.log", workdir / f"{engine}_loaded.txt")
    for f in (out, log, loaded):
        f.unlink(missing_ok=True)
    xs = " ".join(f"x{j+1}" for j in range(k))
    do.write_text(STATA_DO.format(dta=dta, out=out, loaded=loaded, n_gt=n_gt, max_memory=max_memory,
                                  cmd=STATA_CMD[engine].format(xs=xs)))
    proc = subprocess.Popen([stata, "-b", "do", str(do)], cwd=workdir)
    t_loaded, progress, last_cells = None, [], 0
    while proc.poll() is None:
        time.sleep(1)
        now = time.monotonic()
        if t_loaded is None:
            if loaded.exists():
                t_loaded = now
            continue
        if log.exists():
            cells = progress_cells(log.read_text(errors="ignore"), engine)
            if cells > last_cells:
                progress.append((round(now - t_loaded, 1), cells))
                last_cells = cells
        if limit is not None and now - t_loaded > limit:
            proc.kill()
            proc.wait()
            subprocess.run(["pkill", "-f", str(do)], check=False)
            return {"status": "stopped", "elapsed_s": round(now - t_loaded, 1), "cells_done": last_cells,
                    "progress": progress}
    elapsed = round(time.monotonic() - t_loaded, 1) if t_loaded else None
    log_text = log.read_text(errors="ignore") if log.exists() else ""
    if not out.exists():
        oom = any(p in log_text for p in OOM_PATTERNS)
        err = re.findall(r"^r\((\d+)\);", log_text, flags=re.M)
        return {"status": "out_of_memory" if oom else "error", "error_code": err[-1] if err else None,
                "elapsed_s": elapsed, "cells_done": progress_cells(log_text, engine), "progress": progress}
    res = {"status": "ok", "gt": [], "event": {}}
    for line in out.read_text().splitlines():
        key, *vals = line.split()
        if key == "gt":
            res["gt"].append([float(v) for v in vals])
        elif key.startswith("event_"):
            res["event"][key[6:].lower()] = [float(v) for v in vals]
        elif key == "simple":
            res["simple"] = [float(v) for v in vals]
        else:
            res[key] = float(vals[0])
    res["gt"] = np.array(res["gt"])
    return res


def rel_err(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stata", required=True)
    ap.add_argument("--out", default=str(ROOT / "runs" / ("baseline-002_csdid-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])))
    ap.add_argument("--ids", type=int, nargs="*")
    ap.add_argument("--engines", nargs="*", default=["csdid", "csdid2"], choices=list(STATA_CMD))
    ap.add_argument("--limit", type=float, default=1800, help="stop a capped Stata command after this many seconds")
    ap.add_argument("--limit-engines", nargs="*", default=["csdid"], help="commands the time limit applies to")
    ap.add_argument("--max-memory", default="12g", help="Stata max_memory, so memory errors are reported cleanly")
    ap.add_argument("--skip-reference", action="store_true")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    for w in WORKLOADS:
        if args.ids and w["id"] not in args.ids:
            continue
        d = generate(w)
        row = dict(id=w["id"], design=w["design"], **axes(w, d))
        ref = None
        if not args.skip_reference:
            t0 = time.perf_counter()
            ref = reference.run(**d)
            row["reference_s"] = round(time.perf_counter() - t0, 5)
        dta = out / f"w{w['id']}.dta"
        write_dta(d, dta)
        for engine in args.engines:
            limit = args.limit if engine in args.limit_engines else None
            st = run_stata(args.stata, engine, dta, d, row["n_gt"], out, limit, args.max_memory)
            row[f"{engine}_status"] = st["status"]
            if st["status"] != "ok":
                row.update({f"{engine}_{k}": v for k, v in st.items() if k != "status"})
                print(json.dumps({"id": w["id"], "engine": engine, **st}), flush=True)
                continue
            row.update({f"{engine}_cmd_s": st["time_cmd"], f"{engine}_estat_s": st["time_estat"],
                        f"{engine}_total_s": round(st["time_cmd"] + st["time_estat"], 5)})
            if ref is None:
                continue
            names = ["pre_avg", "post_avg"] + [event_name(e) for e in ref["event_times"]]
            ref_att = np.r_[ref["att_gt"], ref["att_simple"], ref["att_pre_avg"], ref["att_post_avg"], ref["att_event"]]
            ref_se = np.r_[ref["se_gt"], ref["se_simple"], ref["se_pre_avg"], ref["se_post_avg"], ref["se_event"]]
            st_ev = np.array([st["event"][n] for n in names])
            st_att = np.r_[st["gt"][:, 0], st["simple"][0], st_ev[:, 0]]
            st_se = np.r_[st["gt"][:, 1], st["simple"][1], st_ev[:, 1]]
            row.update({f"{engine}_max_rel_err_att": rel_err(ref_att, st_att),
                        f"{engine}_max_rel_err_se": rel_err(ref_se, st_se)})
        dta.unlink(missing_ok=True)
        print(json.dumps(row), flush=True)
        with open(out / "results.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
