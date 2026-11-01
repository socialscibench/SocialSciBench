"""Run task 010_larger_than_memory: every tool x recipe on every workload, under a peak-memory budget.

usage: python bench.py --stata /path/to/stata-mp --work DIR --out results/ [--ids 1 2 ...]
                       [--tools reference parquet duckdb pandas r stata] [--recipes extract aggregate append]
                       [--limit 1800] [--min-free-gb 15]

Each run is a fresh process started under /usr/bin/time -l, which reports its peak memory footprint.
The footprint is also sampled every second; a run is stopped when it exceeds the workload's memory
budget (none for the real-data workload), runs longer than --limit seconds, or free disk space falls
below --min-free-gb (swap and outputs share the disk). Before each run the input files are read once,
so every tool starts with the same file cache. Outputs are checked against the reference's checksums
and deleted. Generated data live in --work (outside Dropbox) and are deleted after their workloads.
"""
import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import checksums as C
import generate

HERE = Path(__file__).resolve().parent
PY = sys.executable
# Workload 7 reads licensed position files, which are not distributed: set SSB_TASK010_REAL_DIR to their folder.
REAL_DIR = Path(os.environ.get("SSB_TASK010_REAL_DIR", "licensed-data-not-included"))
REAL = [str(REAL_DIR / f"Positions_{y}_3216rcid.dta") for y in range(2020, 2027)]

# Workloads: data size and memory budget. The ratio data / budget is what makes a workload hard.
WORKLOADS = [
    dict(id=1, tier="development", data_gb=0.5, budget_gb=1),
    dict(id=2, tier="development", data_gb=2, budget_gb=1),
    dict(id=3, tier="development", data_gb=8, budget_gb=1),
    dict(id=4, tier="standard", data_gb=2, budget_gb=4),
    dict(id=5, tier="standard", data_gb=8, budget_gb=4),
    dict(id=6, tier="standard", data_gb=32, budget_gb=4),
    dict(id=7, tier="real data", data_gb="real", budget_gb=None),
]
TOOLS = ["reference", "parquet", "duckdb", "pandas", "r", "stata"]
RECIPES = ["extract", "aggregate", "append"]


def footprint(pid):
    """Current memory footprint of a process in bytes (macOS `footprint`), or 0."""
    try:
        s = subprocess.run(["footprint", "-p", str(pid)], capture_output=True, text=True, timeout=20).stdout
    except subprocess.TimeoutExpired:
        return 0
    m = re.search(r"Footprint:\s*([\d.]+)\s*(B|KB|MB|GB)", s)
    return float(m.group(1)) * {"B": 1, "KB": 2 ** 10, "MB": 2 ** 20, "GB": 2 ** 30}[m.group(2)] if m else 0


def free_gb(path):
    return shutil.disk_usage(path).free / 1e9


def settle(max_wait=90):
    """Wait until macOS reports at least 30% of memory free (after a large run)."""
    t0 = time.time()
    while time.time() - t0 < max_wait:
        out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
        m = re.search(r"free percentage:\s*(\d+)%", out)
        if m and int(m.group(1)) >= 30:
            return
        time.sleep(5)


def read_all(paths, block=64 << 20):
    t0 = time.perf_counter()
    for p in paths:
        with open(p, "rb") as f:
            while f.read(block):
                pass
    return time.perf_counter() - t0


def write_bytes(path, n, block=64 << 20):
    buf = b"\0" * block
    t0 = time.perf_counter()
    with open(path, "wb") as f:
        for _ in range(n // block):
            f.write(buf)
        f.write(b"\0" * (n % block))
    t = time.perf_counter() - t0
    os.unlink(path)
    return t


def run(cmd, cwd, budget, limit, min_free, log):
    """Run cmd under /usr/bin/time -l; enforce budget (bytes or None), time limit and free disk."""
    with open(log, "w") as err:
        proc = subprocess.Popen(["/usr/bin/time", "-l", *cmd], cwd=cwd, stdout=err, stderr=err,
                                start_new_session=True)
    t0, status, sampled = time.time(), "ok", 0.0
    while proc.poll() is None:
        time.sleep(1)
        kids = subprocess.run(["pgrep", "-P", str(proc.pid)], capture_output=True, text=True).stdout.split()
        for k in kids:
            sampled = max(sampled, footprint(k))
        reason = None
        if budget and sampled > budget:
            reason = "over budget"
        elif time.time() - t0 > limit:
            reason = "stopped (time)"
        elif free_gb(cwd) < min_free:
            reason = "stopped (disk)"
        if reason:
            status = reason
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            break
    wall = time.time() - t0
    text = Path(log).read_text(errors="ignore")
    m = re.search(r"(\d+)\s+peak memory footprint", text)
    peak = max(float(m.group(1)) if m else 0.0, sampled)
    if status == "ok" and budget and peak > budget:
        status = "over budget"
    return dict(status=status, wall_s=round(wall, 3), peak_mb=round(peak / 2 ** 20, 1),
                log_tail=text[-600:] if status == "ok" and proc.returncode else None)


def command(tool, recipe, out, timefile, work, budget_mb, files, stata):
    if tool in ("reference", "pandas", "duckdb", "parquet"):
        return [PY, str(HERE / "bench_python.py"), tool, recipe, str(out), str(timefile), str(work),
                str(budget_mb or 0), *map(str, files)]
    if tool == "r":
        return ["R", "--no-echo", "--no-restore", f"--file={HERE / 'bench_r.R'}", "--args", recipe, str(out),
                str(timefile), *map(str, files)]
    filelist = Path(work) / "stata_files.txt"                      # Stata crashes on long command lines
    filelist.write_text("\n".join(map(str, files)) + "\n")
    return [stata, "-b", "do", "bench_stata.do", recipe, str(out), str(timefile), str(work), str(filelist)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stata", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--ids", type=int, nargs="*")
    ap.add_argument("--tools", nargs="*", default=TOOLS)
    ap.add_argument("--recipes", nargs="*", default=RECIPES)
    ap.add_argument("--limit", type=float, default=1800)
    ap.add_argument("--min-free-gb", type=float, default=15)
    args = ap.parse_args()
    work, out = Path(args.work).resolve(), Path(args.out).resolve()
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    (work / "duckdb_tmp").mkdir(exist_ok=True)
    shutil.copy(HERE / "bench_stata.do", work / "bench_stata.do")
    todo = [w for w in WORKLOADS if not args.ids or w["id"] in args.ids]

    for size in dict.fromkeys(w["data_gb"] for w in todo):            # each data size generated once
        if size == "real":
            files = REAL
        else:
            files = generate.generate(size, work / f"data_{size}")
        in_bytes = sum(os.path.getsize(f) for f in files)
        for w in [w for w in todo if w["data_gb"] == size]:
            budget = w["budget_gb"] * 2 ** 30 if w["budget_gb"] else None
            budget_mb = w["budget_gb"] * 1024 if w["budget_gb"] else 0
            read_all(files)
            sol_read = read_all(files)                                 # same cache state as each tool's run
            truth, parquet = {}, None
            for recipe in args.recipes:
                for tool in [t for t in args.tools if t != "parquet"] if recipe else []:
                    if tool == "duckdb" and parquet is None:
                        pq_dir = work / f"parquet_{w['id']}"
                        pq_dir.mkdir(exist_ok=True)
                        read_all(files)
                        tf = work / "time_parquet.txt"
                        tf.unlink(missing_ok=True)
                        r = run(command("parquet", "-", pq_dir, tf, work, budget_mb, files, args.stata), work,
                                budget, args.limit, args.min_free_gb, work / "log_parquet.txt")
                        r["time_s"] = float(tf.read_text().split()[1]) if tf.exists() and r["status"] == "ok" else None
                        parquet = sorted(pq_dir.glob("*.parquet")) if r["status"] == "ok" else []
                        r.update(id=w["id"], tool="parquet", recipe="convert", output_mb=round(
                            sum(os.path.getsize(p) for p in pq_dir.glob("*.parquet")) / 2 ** 20, 1))
                        print(json.dumps(r), flush=True)
                        with open(out / "results.jsonl", "a") as f:
                            f.write(json.dumps(r) + "\n")
                    if tool == "duckdb" and not parquet:
                        continue
                    inputs = parquet if tool == "duckdb" else files
                    o = work / f"out_{tool}_{recipe}.dta"
                    tf = work / f"time_{tool}.txt"
                    for p in (o, tf):
                        p.unlink(missing_ok=True)
                    settle()
                    read_all(inputs)
                    r = run(command(tool, recipe, o, tf, work, budget_mb, inputs, args.stata), work, budget,
                            args.limit, args.min_free_gb, work / f"log_{tool}.txt")
                    r["time_s"] = float(tf.read_text().split()[1]) if tf.exists() and r["status"] == "ok" else None
                    if r["status"] == "ok" and not (o.exists() and r["time_s"] is not None):
                        r["status"] = "error"
                    if r["status"] == "ok":
                        r["output_mb"] = round(o.stat().st_size / 2 ** 20, 1)
                        cs = C.checksums(o, chunk_rows=200_000)
                        if tool == "reference":
                            truth[recipe] = cs
                            r["sol_s"] = round(sol_read + write_bytes(work / "sol.bin", o.stat().st_size), 3)
                            truth[recipe + "_sol"] = r["sol_s"]
                            truth[recipe + "_rows"] = cs["N"]
                        elif recipe in truth:
                            r["bad_keys"] = C.compare(cs, truth[recipe])
                            r["exact"] = not r["bad_keys"]
                    if truth.get(recipe + "_sol") and r.get("time_s"):
                        r["efficiency"] = round(truth[recipe + "_sol"] / r["time_s"], 4)
                    r.update(id=w["id"], tier=w["tier"], data_gb=w["data_gb"], budget_gb=w["budget_gb"],
                             input_gb=round(in_bytes / 1e9, 2), recipe=recipe, tool=tool,
                             output_rows=truth.get(recipe + "_rows"))
                    o.unlink(missing_ok=True)
                    print(json.dumps(r), flush=True)
                    with open(out / "results.jsonl", "a") as f:
                        f.write(json.dumps(r) + "\n")
            if parquet is not None:
                shutil.rmtree(work / f"parquet_{w['id']}", ignore_errors=True)
        if size != "real":
            shutil.rmtree(work / f"data_{size}", ignore_errors=True)


if __name__ == "__main__":
    main()
