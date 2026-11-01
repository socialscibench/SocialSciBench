"""Baseline measurement driver (study tooling, not part of the benchmark).

Runs a solution through the evaluator's own worker (harness.worker: fresh process per case, same
input loaders, read-only inputs, first call + warmups + measured calls, perf_counter around run,
threadpool limits and thread environment variables as harness.runner sets them), then validates
every output with the task's validator against a saved reference output computed by
reference.run in the same environment. The only difference from `harness evaluate` is that the
reference is not recomputed in each invocation.

Cases: real-development ids (101, 104k5, 104k50; 104 means its default 20 regressors) and
synthetic ids prefixed with s (s1, s9). Saved reference outputs come from make_refs.py.

usage: compare.py --task 001 --solution path/solution.json --cases 101 104k50 s9 --threads 1
                  --refdir DIR --out results.jsonl [--warmups 1 --repeats 5] [--timeout 1800]
                  [--python path/to/python] [--not-comparable PREFIX ...]

--not-comparable drops the outputs whose names start with the given prefixes from the validation
(they are still computed inside the timed call): for task 002, a baseline's bootstrap standard
errors and critical values (se_boot_, crit_) come from its own random multipliers, not the given
ones, and differ from the reference's by the bootstrap's sampling noise. The dropped names are
listed in each result row.
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from harness.common import read_json, write_json  # noqa: E402
from harness.validation import normalize, validate  # noqa: E402

TASK_DIR = {"001": "001_reghdfe", "002": "002_csdid", "003": "003_reclink"}


def case_spec(task, name):
    if name.startswith("s"):
        from harness.common import load_module
        wl = load_module(ROOT / TASK_DIR[task] / "workloads.py", f"wl{task}")
        w = [w for w in wl.WORKLOADS if w["id"] == int(name[1:])][0]
        return dict(id=f"workload-{w['id']}", kind="generated", params=w, scored=True), "synthetic"
    wid, _, kk = name.partition("k")
    options = {"n_regressors": int(kk)} if kk else ({"n_regressors": 20} if (task == "001" and wid == "104") else {})
    variant = f"-k{options['n_regressors']}" if options else ""
    return dict(id=f"real-development-{wid}{variant}", kind="real", suite="real-development",
                params=dict(id=int(wid), loader_options=options),
                scored=True), "real-development"


def load_ref(task, refdir, name):
    z = dict(np.load(Path(refdir) / f"{name}.npz", allow_pickle=False))
    if task == "003":
        return normalize("003", (z["i"], z["j"], z["score"], int(z["n_perfect"])))
    if task == "001":
        return normalize("001", {k: z[k] for k in ("beta", "vcov", "nobs", "df_a", "n_clusters")})
    return {k: v for k, v in z.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--solution", required=True, type=Path)
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--warmups", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--refdir", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--not-comparable", nargs="*", default=[])
    a = ap.parse_args()
    a.python = os.path.abspath(a.python) if os.sep in a.python else a.python  # the worker runs in its own directory
    tolerances = read_json(ROOT / TASK_DIR[a.task] / "task.json")["tolerances"]
    sol = read_json(a.solution)
    work = Path(tempfile.mkdtemp(prefix="cmp-"))
    snap = work / "solution"
    snap.mkdir()
    for f in sol["sources"]:
        shutil.copy2(a.solution.parent / f, snap / f)
    for name in a.cases:
        case, suite = case_spec(a.task, name)
        warmups = a.warmups if a.warmups is not None else 1
        repeats = a.repeats if a.repeats is not None else (3 if suite == "synthetic" else 5)
        outdir = work / name
        outdir.mkdir()
        config = dict(task=a.task, task_dir=str(ROOT / TASK_DIR[a.task]), case=case, output_dir=str(outdir),
                      threads=a.threads, warmups=warmups, repeats=repeats, source_dir=str(snap),
                      entry_point=sol["entry_point"], synchronize=None,
                      data_root=str(ROOT / "datasets" / a.task / "data"), stream_outputs=False)
        write_json(outdir / "config.json", config)
        env = os.environ.copy()
        env.update(PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0",
                   SOCIALSCIBENCH_BASELINE_VENV=str(ROOT / "solutions" / "baselines" / ".venv"))
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "RAYON_NUM_THREADS", "NUMBA_NUM_THREADS"):
            env[key] = str(a.threads)
        t0 = time.monotonic()
        with (outdir / "stdout.log").open("w") as so, (outdir / "stderr.log").open("w") as se:
            p = subprocess.Popen([a.python, "-m", "harness.worker", str(outdir / "config.json")], cwd=outdir, env=env,
                                 stdout=so, stderr=se, start_new_session=True)
            try:
                p.wait(timeout=a.timeout)
                status = None
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
                status = "TIMEOUT"
        wall = time.monotonic() - t0
        row = dict(task=a.task, solution=sol["name"], case=name, threads=a.threads, warmups=warmups, repeats=repeats,
                   wall_s=wall, created=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        if status is None and (outdir / "result.json").exists():
            res = read_json(outdir / "result.json")
            row.update(status=res["status"], samples_s=res.get("samples_s"), first_call_s=res.get("first_call_s"),
                       import_setup_s=res.get("import_setup_s"), peak_rss_bytes=res.get("peak_rss_bytes"),
                       error=res.get("error"))
            if res["status"] == "COMPLETED":
                exp = load_ref(a.task, a.refdir, name)
                drop = lambda d: {k: v for k, v in d.items() if not any(k.startswith(p_) for p_ in a.not_comparable)}
                row["not_compared"] = sorted(k for k in exp if k not in drop(exp))
                exp = drop(exp)
                checks = []
                for item in res["outputs"]:
                    with np.load(outdir / item["file"], allow_pickle=False) as fh:
                        actual = drop(dict(fh))
                    rep = validate(a.task, actual, exp, tolerances, res.get("dimensions") or {})
                    checks.append(dict(phase=item["phase"], passed=bool(rep["passed"]),
                                       failed={k: v for k, v in rep["checks"].items() if not v.get("passed", True)}))
                row["checks_passed"] = all(c["passed"] for c in checks)
                row["failed_checks"] = [c for c in checks if not c["passed"]][:2]
                if row["samples_s"]:
                    row["median_s"] = float(np.median(row["samples_s"]))
        else:
            row["status"] = status or "RUNTIME_ERROR"
            row["stderr_tail"] = (outdir / "stderr.log").read_text()[-2000:]
        with a.out.open("a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
        print(json.dumps({k: row.get(k) for k in ("case", "status", "checks_passed", "median_s", "first_call_s")}), flush=True)
        for f in outdir.glob("output-*.npz"):
            f.unlink()
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
