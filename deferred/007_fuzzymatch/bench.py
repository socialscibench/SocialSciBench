"""Generate workloads; time Stata (matchit), R (stringdist) and the Python reference; check agreement.

usage: python bench.py --stata /path/to/stata-mp --out results/ [--ids 1 2 ...] [--limit 1800]
                       [--skip stata r reference]

Stata and R are stopped after --limit seconds; matchit prints its progress (flag(1): every 1% of the
master list), which is logged with timestamps so the total time can be estimated.
"""
import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import pandas as pd

import sys
from datetime import datetime, timezone
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import reference
from workloads import WORKLOADS, axes, generate

HERE = Path(__file__).resolve().parent

STATA_DO = """
set more off
set max_memory 12g
use "{master}", clear
file open m using "{loaded}", write replace
file close m
timer clear
timer on 1
matchit idm namem using "{using}", idusing(idu) txtusing(nameu) threshold({thr}) flag(1)
timer off 1
qui timer list 1
local t = r(t1)
qui summarize similscore, meanonly
local n = r(N)
local s = r(sum)
qui summarize idm, meanonly
local im = r(sum)
qui summarize idu, meanonly
local iu = r(sum)
file open f using "{out}", write replace
file write f "time " %15.5f (`t') _n "pairs `n'" _n "score_sum " %21.0g (`s') _n
file write f "id_master_sum " %21.0g (`im') _n "id_using_sum " %21.0g (`iu') _n
file close f
"""


def parse(path):
    return {k: float(v) for k, v in (line.split() for line in path.read_text().splitlines())}


def progress_pct(log_text):
    """Last 'Percent completed' value printed by matchit's main loop."""
    text = log_text[log_text.find("Percent completed"):]
    vals = re.findall(r"^\s*(\d+)%\s", text, flags=re.M)
    return int(vals[-1]) if vals else 0


def run_stata(stata, master, using, thr, workdir, limit):
    out, do, log, loaded = (workdir / "stata_out.txt", workdir / "stata.do", workdir / "stata.log",
                            workdir / "stata_loaded.txt")
    for f in (out, log, loaded):
        f.unlink(missing_ok=True)
    do.write_text(STATA_DO.format(master=master, using=using, loaded=loaded, out=out, thr=thr))
    proc = subprocess.Popen([stata, "-b", "do", str(do)], cwd=workdir)
    t_loaded, progress, last = None, [], 0
    while proc.poll() is None:
        time.sleep(1)
        now = time.monotonic()
        if t_loaded is None:
            if loaded.exists():
                t_loaded = now
            continue
        if log.exists():
            pct = progress_pct(log.read_text(errors="ignore"))
            if pct > last:
                progress.append((round(now - t_loaded, 1), pct))
                last = pct
        if now - t_loaded > limit:
            proc.kill()
            proc.wait()
            subprocess.run(["pkill", "-f", str(do)], check=False)
            return {"status": "stopped", "progress": progress}
    if not out.exists():
        return {"status": "error", "log_tail": log.read_text(errors="ignore")[-400:] if log.exists() else ""}
    return {"status": "ok", **parse(out)}


def compare(a, ref):
    exact = int(a["pairs"]) == ref["pairs"] and int(a["id_master_sum"]) == ref["id_master_sum"] and \
        int(a["id_using_sum"]) == ref["id_using_sum"]
    return exact, abs(a["score_sum"] - ref["score_sum"]) / ref["score_sum"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stata", required=True)
    ap.add_argument("--out", default=str(ROOT / "runs" / ("baseline-007_fuzzymatch-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])))
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
        row = dict(id=w["id"], design=w["design"], **axes(w, d))
        ref = None
        if "reference" not in args.skip:
            t0 = time.perf_counter()
            res = reference.run(d["master"], d["using"], d["threshold"])
            row["reference_s"] = round(time.perf_counter() - t0, 5)
            ref = reference.checksums(res, d["truth"])
            row.update(pairs_found=ref["pairs"], true_links_found=ref["true_links_found"])

        if "r" not in args.skip:
            mt, ut, rout = out / "master.txt", out / "using.txt", out / "r_out.txt"
            mt.write_text("\n".join(d["master"]) + "\n")
            ut.write_text("\n".join(d["using"]) + "\n")
            rout.unlink(missing_ok=True)
            try:
                p = subprocess.run(["Rscript", str(HERE / "bench_r.R"), str(mt), str(ut), str(d["threshold"]), str(rout)],
                                   cwd=out, capture_output=True, text=True, timeout=args.limit)
            except subprocess.TimeoutExpired:
                p = None
                row.update(r_status="stopped")
                subprocess.run(["pkill", "-f", "bench_r.R"], check=False)     # make sure R itself has stopped
                time.sleep(2)
            if p is None:
                pass
            elif p.returncode == 0 and rout.exists():
                r = parse(rout)
                row.update(r_status="ok", r_s=r["time"])
                if ref:
                    row["r_exact"], row["r_score_rel_diff"] = compare(r, ref)
                    row["r_pairs"] = int(r["pairs"])
            else:
                row.update(r_status="error", r_error=(p.stderr or p.stdout).strip().splitlines()[-1:])

        if "stata" not in args.skip:
            mdta, udta = out / "master.dta", out / "using.dta"
            pd.DataFrame({"idm": range(1, len(d["master"]) + 1), "namem": d["master"]}).to_stata(mdta, write_index=False)
            pd.DataFrame({"idu": range(1, len(d["using"]) + 1), "nameu": d["using"]}).to_stata(udta, write_index=False)
            st = run_stata(args.stata, mdta, udta, d["threshold"], out, args.limit)
            row["stata_status"] = st["status"]
            if st["status"] == "ok":
                row["stata_s"] = st["time"]
                if ref:
                    row["stata_exact"], row["stata_score_rel_diff"] = compare(st, ref)
                    row["stata_pairs"] = int(st["pairs"])
            else:
                row.update({f"stata_{k}": v for k, v in st.items() if k != "status"})

        print(json.dumps(row), flush=True)
        with open(out / "results.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
