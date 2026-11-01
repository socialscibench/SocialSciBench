"""Time the engine's calls directly on a kit's data, without an agent: one fresh process per call.

usage: bench_engine.py KIT --python PY [--pythonpath DIR] [--variant speed|memory|auto] [--threads 4]
                       [--sizes ...] [--calls ...] [--reps 3] [--strings object|str] [--tables] [--gate 30]
                       --out DIR

Each child process loads the kit's data the way the pilot's agents did (eval/pilot-0.1.md), warms
the kernels up, then runs one call `reps` times. It records each repetition's wall time and the resident
memory before, during (sampled every 5 ms) and after, and saves the results to DIR for comparisons
between versions (compare_engine.py). DIR holds estimates from licensed data: keep it outside the repo.
KIT is 001, 002 or 003; sizes are the kit's (001 and 003: percent of rows; 002: copies of the panel).
"""
import argparse
import ctypes
import ctypes.util
import json
import os
import pickle
import subprocess
import sys
import threading
import time
from pathlib import Path

KITS = Path(os.environ.get("SSB_KITS", "~/SocialSciBench-kits")).expanduser()
SIZES = {"001": [1, 3, 10, 30, 100], "002": [1, 3, 10, 30, 100], "003": [1, 3, 10, 30, 100]}
CALLS = {"001": [1, 7, 8], "002": [1, 2, 3, 4, 5, 6, 7], "003": [1]}


# ---- resident memory of this process (mach_task_basic_info) ----------------------------------------
class _TimeValue(ctypes.Structure):
    _fields_ = [("seconds", ctypes.c_int), ("microseconds", ctypes.c_int)]


class _TaskInfo(ctypes.Structure):
    _fields_ = [("virtual_size", ctypes.c_uint64), ("resident_size", ctypes.c_uint64),
                ("resident_size_max", ctypes.c_uint64), ("user_time", _TimeValue),
                ("system_time", _TimeValue), ("policy", ctypes.c_int), ("suspend_count", ctypes.c_int)]


_libc = ctypes.CDLL(ctypes.util.find_library("c"))
_task = ctypes.c_uint.in_dll(_libc, "mach_task_self_")


def rss():
    info = _TaskInfo()
    count = ctypes.c_uint(ctypes.sizeof(_TaskInfo) // 4)
    _libc.task_info(_task, 20, ctypes.byref(info), ctypes.byref(count))      # MACH_TASK_BASIC_INFO
    return int(info.resident_size)


class Sampler:
    def __init__(self, period=0.005):
        self.period, self.peak, self.stop = period, rss(), threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while not self.stop.is_set():
            self.peak = max(self.peak, rss())
            time.sleep(self.period)

    def close(self):
        self.stop.set()
        self.thread.join()
        self.peak = max(self.peak, rss())
        return self.peak


# ---- the kits, as the pilot's agents called the engine ---------------------------------------------
CLASSES = [f"myclass{i}" for i in range(1, 130)]
SPECS_001 = {1: dict(y="lfcit01W", x=["lrrfsim01W"] + CLASSES, absorb=["issue_year"]),
             7: dict(y="lfcit2AW", x=["lfcit01W", "lrrfsim01W"] + CLASSES, absorb=["issue_year"]),
             8: dict(y="lfcit2AW", x=["lfcit01W", "lrrfsim01W"] + CLASSES, absorb=["firm_year"])}
X_002 = ["perc_female", "perc_white", "perc_hispanic", "unemp_rate_pc", "poverty_rate", "median_income_k"]
CALLS_002 = {1: ("2xT", None, "dripw", "never", 1.0, "event"), 2: ("2xT", X_002, "reg", "never", 0.995, "event"),
             3: ("2xT", X_002, "stdipw", "never", 0.995, "event"), 4: ("2xT", X_002, "dripw", "never", 0.995, "event"),
             5: ("full", None, "dripw", "notyet", 1.0, "gt"), 6: ("full", None, "dripw", "notyet", 1.0, "event"),
             7: ("full", X_002, "dripw", "notyet", 0.995, "event")}
COLS_002 = ["county_code", "year", "treat_year", "crude_rate_20_64", "set_wt"] + X_002
VARS_003 = ["name", "addr", "city", "zip", "state"]


def prepare(kit, size, call, strings, tables=False):
    """The call's inputs, built before the clock as the agents built them; returns (fn, kwargs).
    tables (002): bootstrap only the table the call reports (boot_tables=, turbo-socialsci 0.2)."""
    import pandas as pd
    import turbo_socialsci as ts
    if kit == "001":
        df = pd.read_parquet(KITS / "001/data/frozen.parquet")
        df["firm_year"] = df["firm"].astype("Int64") * 100 + (df["issue_year"].astype("Int64") - 1940)
        sub = df if size == 100 else df[df[f"sub{size:02d}"] == 1].reset_index(drop=True)
        del df
        return ts.feols, dict(data=sub, cluster="issue_year", **SPECS_001[call])
    if kit == "002":
        import numpy as np
        df = pd.read_parquet(KITS / "002/data/frozen.parquet")
        sample, cov, method, control, trim, _ = CALLS_002[call]
        base = df[COLS_002] if sample == "full" else df.loc[df["in_2xT"] == 1, COLS_002]
        parts = []
        for j in range(1, size + 1):
            d = base.copy()
            if size > 1:
                d["county_code"] = j * 100000 + d["county_code"].astype(np.int64)
            parts.append(d)
        data = pd.concat(parts, ignore_index=True)
        kw = dict(data=data, y="crude_rate_20_64", unit="county_code", time="year", cohort="treat_year",
                  covariates=cov or [], weights="set_wt", control=control, method=method,
                  base_period="universal", trim=trim, rules="csdid181", n_boot=25000,
                  seed=1000 * size + 10 * call + 1)
        if tables:
            kw["boot_tables"] = [CALLS_002[call][5]]
        return ts.att_gt, kw
    if kit == "003":
        master = pd.read_parquet(KITS / "003/data/master.parquet").rename(columns={"id": "idm"})
        using = pd.read_parquet(KITS / "003/data/using.parquet").rename(columns={"id": "idu"})[["idu"] + VARS_003]
        for df in (master, using):
            for v in VARS_003:
                df[v] = df[v].fillna("").astype(object) if strings == "object" else df[v].fillna("")
        sub = master if size == 100 else master[master[f"sub{size:02d}"] == 1]
        sub = sub[["idm"] + VARS_003].reset_index(drop=True)
        return ts.reclink, dict(master=sub, using=using, idmaster="idm", idusing="idu", vars=VARS_003,
                                wmatch=[10, 8, 5, 5, 5], wnomatch=[2, 3, 4, 4, 4], required=["state"],
                                minscore=0.6, minbigram=0.6)
    raise ValueError(kit)


def digest(kit, res):
    """What compare_engine.py compares between versions."""
    import numpy as np
    if kit == "001":
        return dict(coef=res.coef[["coef", "se", "dropped"]].copy(), nobs=res.nobs, r2=res.r2,
                    r2_within=res.r2_within, df_a=res.df_a, n_clusters=res.n_clusters, variant=res.variant)
    if kit == "002":
        return dict(gt=res.gt.copy(), event=res.event.copy(), averages=res.averages.copy(), crit=dict(res.crit),
                    variant=res.variant)
    m = res.sort_values(list(res.columns[:2])).reset_index(drop=True)
    return dict(i=m.iloc[:, 0].to_numpy(), j=m.iloc[:, 1].to_numpy(), score=m["score"].to_numpy(np.float64),
                n_perfect=res.attrs.get("n_perfect"), variant=res.attrs.get("variant"))


def child(a):
    import resource
    import turbo_socialsci as ts
    fn, kw = prepare(a.kit, a.size, a.call, a.strings, a.tables)
    which = {"001": "hdfe", "002": "did", "003": "reclink"}[a.kit]
    ts.warmup(which, variant="memory" if a.variant == "memory" else "speed")
    rows = []
    for rep in range(1, a.reps + 1):
        before = rss()
        s = Sampler()
        t0 = time.perf_counter()
        res = fn(**kw, variant=a.variant, threads=a.threads)
        sec = time.perf_counter() - t0
        peak = s.close()
        after = rss()
        if rep == 1:
            with open(Path(a.out) / f"{a.kit}-{a.size}-{a.call}-{a.variant}.pkl", "wb") as fh:
                pickle.dump(digest(a.kit, res), fh)
        del res
        rows.append(dict(kit=a.kit, version=ts.__version__, variant=a.variant, threads=a.threads, size=a.size,
                         call=a.call, rep=rep, seconds=round(sec, 4), rss_before=before, rss_peak=peak,
                         rss_after=after, call_peak_increase=peak - before))
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for r in rows:
        r["process_maxrss"] = maxrss
        print(json.dumps(r), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kit", choices=["001", "002", "003"])
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--pythonpath")
    ap.add_argument("--variant", default="speed", choices=["speed", "memory", "auto"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--sizes", type=int, nargs="+")
    ap.add_argument("--calls", type=int, nargs="+")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--strings", default="object", choices=["object", "str"])
    ap.add_argument("--tables", action="store_true", help="002: boot_tables= the call's reported table (0.2)")
    ap.add_argument("--gate", type=float, default=0, help="before each call, wait for this many seconds of "
                    "unthrottled, cooled CPU samples (eval/thermal.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--size", type=int)
    ap.add_argument("--call", type=int)
    a = ap.parse_args()
    if a.child:
        return child(a)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import thermal
    mon = thermal.Monitor()
    env = dict(os.environ)
    if a.pythonpath:
        env["PYTHONPATH"] = a.pythonpath
    for size in a.sizes or SIZES[a.kit]:
        for call in a.calls or CALLS[a.kit]:
            cmd = [a.python, __file__, a.kit, "--child", "--size", str(size), "--call", str(call), "--variant",
                   a.variant, "--threads", str(a.threads), "--reps", str(a.reps), "--strings", a.strings,
                   "--out", a.out] + (["--tables"] if a.tables else [])
            waited = mon.gate(window=a.gate) if a.gate else 0.0
            t0 = time.time()
            p = subprocess.run(cmd, env=env, capture_output=True, text=True)
            therm = dict(mon.summary(t0, time.time()), gate_wait_s=round(waited, 1))
            lines = [json.dumps(dict(json.loads(ln), thermal=therm)) for ln in p.stdout.splitlines()
                     if ln.startswith("{")]
            with open(Path(a.out) / "timings.jsonl", "a") as fh:
                for ln in lines:
                    fh.write(ln + "\n")
            if p.returncode:
                print(f"{a.kit} size {size} call {call}: exit {p.returncode}\n{p.stderr[-3000:]}", file=sys.stderr)
            for ln in lines:
                r = json.loads(ln)
                print(f"{r['kit']} v{r['version']} {r['variant']} size {r['size']} call {r['call']} rep {r['rep']}: "
                      f"{r['seconds']:.3f} s, +{r['call_peak_increase'] / 2**30:.2f} GiB "
                      f"(peak {r['rss_peak'] / 2**30:.2f} GiB; throttled {therm['throttled']}/{therm['samples']})",
                      flush=True)


if __name__ == "__main__":
    main()
