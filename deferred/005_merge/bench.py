"""Generate workloads; time Stata (merge), R (data.table merge) and Python (pandas merge); check agreement.

usage: python bench.py --stata /path/to/stata-mp --out results/ [--ids 1 2 ...] [--limit 1800]
                       [--threads 12] [--skip stata r reference]

Stata's merge reads the using table from disk, so the Stata run also times `use using.dta` on
its own; R and pandas merge two tables already in memory. Every tool reports the number of
rows, the _merge counts, and the count and sum of every numeric column.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import reference
from workloads import WORKLOADS, axes, generate

HERE = Path(__file__).resolve().parent

STATA_DO = """
set more off
set max_memory {max_memory}
timer clear
timer on 1
use "{using}", clear
timer off 1
use "{master}", clear
file open m using "{loaded}", write replace
file close m
timer on 2
merge {kind} {keys} using "{using}"
timer off 2
qui timer list
file open f using "{out}", write replace
file write f "time_read_using " %15.5f (r(t1)) _n "time " %15.5f (r(t2)) _n "rows " (_N) _n
forvalues k = 1/3 {{
    qui count if _merge == `k'
    file write f "merge `k' " (r(N)) _n
}}
foreach v of varlist m* u* {{
    qui summarize `v', meanonly
    file write f "col `v' " (r(N)) " " %21.0g (r(sum)) _n
}}
file close f
"""


def parse(path):
    res = {"merge": {}, "cols": {}}
    for line in path.read_text().splitlines():
        key, *vals = line.split()
        if key == "merge":
            res["merge"][int(vals[0])] = int(float(vals[1]))
        elif key == "col":
            res["cols"][vals[0]] = (int(float(vals[1])), float(vals[2]))
        elif key == "rows":
            res["rows"] = int(float(vals[0]))
        else:
            res[key] = float(vals[0])
    return res


def run_stata(stata, master, using, d, workdir, limit, max_memory):
    out, do, loaded = workdir / "stata_out.txt", workdir / "stata.do", workdir / "stata_loaded.txt"
    for f in (out, loaded):
        f.unlink(missing_ok=True)
    do.write_text(STATA_DO.format(master=master, using=using, loaded=loaded, out=out, kind=d["kind"],
                                  keys=" ".join(d["keys"]), max_memory=max_memory))
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
    if not out.exists():
        return {"status": "error", "log_tail": (workdir / "stata.log").read_text(errors="ignore")[-400:]}
    return {"status": "ok", **parse(out)}


def run_r(master_csv, using_csv, d, key_type, workdir, threads):
    out = workdir / "r_out.txt"
    out.unlink(missing_ok=True)
    p = subprocess.run(["Rscript", str(HERE / "bench_r.R"), str(master_csv), str(using_csv), ",".join(d["keys"]),
                        key_type, str(out), str(threads)], cwd=workdir, capture_output=True, text=True)
    if p.returncode != 0 or not out.exists():
        return {"status": "error", "error": (p.stderr or p.stdout).strip().splitlines()[-1:]}
    return {"status": "ok", **parse(out)}


def compare(a, ref):
    """Exact match of rows, _merge counts and column counts; max relative difference of column sums."""
    exact = a["rows"] == ref["rows"] and a["merge"] == ref["merge"] and \
        all(a["cols"][c][0] == ref["cols"][c][0] for c in ref["cols"])
    rel = max(abs(a["cols"][c][1] - ref["cols"][c][1]) / max(abs(ref["cols"][c][1]), 1e-300) for c in ref["cols"])
    return exact, rel


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
        row = dict(id=w["id"], design=w["design"], **axes(w, d))
        ref = None
        if "reference" not in args.skip:
            t0 = time.perf_counter()
            merged = reference.run(d["master"], d["using"], d["keys"], d["kind"])
            row["pandas_s"] = round(time.perf_counter() - t0, 5)
            ref = reference.checksums(merged, d["keys"])
            row.update(rows=ref["rows"], master_only=ref["merge"][1], using_only=ref["merge"][2],
                       matched=ref["merge"][3])
            del merged

        if "r" not in args.skip:
            mcsv, ucsv = out / "master.csv", out / "using.csv"
            d["master"].to_csv(mcsv, index=False)
            d["using"].to_csv(ucsv, index=False)
            r = run_r(mcsv, ucsv, d, w["key_type"], out, args.threads)
            mcsv.unlink(missing_ok=True)
            ucsv.unlink(missing_ok=True)
            row["r_status"] = r["status"]
            if r["status"] == "ok":
                row["r_s"] = r["time"]
                if ref:
                    row["r_exact"], row["r_max_rel_diff"] = compare(r, ref)
            else:
                row["r_error"] = r["error"]

        if "stata" not in args.skip:
            mdta, udta = out / "master.dta", out / "using.dta"
            d["master"].to_stata(mdta, write_index=False)
            d["using"].to_stata(udta, write_index=False)
            st = run_stata(args.stata, mdta, udta, d, out, args.limit, args.max_memory)
            mdta.unlink(missing_ok=True)
            udta.unlink(missing_ok=True)
            row["stata_status"] = st["status"]
            if st["status"] == "ok":
                row.update(stata_s=st["time"], stata_read_using_s=st["time_read_using"])
                if ref:
                    row["stata_exact"], row["stata_max_rel_diff"] = compare(st, ref)
            else:
                row.update({f"stata_{k}": v for k, v in st.items() if k != "status"})

        print(json.dumps(row), flush=True)
        with open(out / "results.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
