"""Save one reference output per case for compare.py (study tooling, not part of the benchmark).

Runs the task's reference.run once per case in this process with one BLAS thread and saves its
output (task 001: as returned; tasks 002 and 003: normalized by harness.validation.normalize)
with the time and peak RSS. Run each case in a fresh process for meaningful memory figures.

usage: make_refs.py TASK CASE OUT_DIR      (CASE: 101, 104k50, s9, ...; from the repository root)
"""
import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from harness.common import load_module  # noqa: E402
from harness.validation import normalize  # noqa: E402

TASK_DIR = {"001": "001_reghdfe", "002": "002_csdid", "002b": "002b_csdid_options", "003": "003_reclink"}
task, case, out_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
tdir = ROOT / TASK_DIR[task]
ref = load_module(tdir / "reference.py", f"ref{task}")
wl = load_module(tdir / "workloads.py", f"wl{task}")
if case.startswith("s"):
    d = wl.generate([w for w in wl.WORKLOADS if w["id"] == int(case[1:])][0])
    if task == "003":
        d.pop("truth", None)
        d.update(vars=wl.VARS, wmatch=wl.WMATCH, wnomatch=wl.WNOMATCH, required=wl.REQUIRED,
                 minscore=wl.MINSCORE, minbigram=wl.MINBIGRAM)
else:
    rd = load_module(tdir / "realdata.py", f"rd{task}")
    wid, _, k = case.partition("k")
    d = rd.load(wid if task == "002b" else int(wid), data_root=str(ROOT / "datasets" / task / "data"),
                **({"n_regressors": int(k)} if k else {}))
with threadpool_limits(1):
    t0 = time.perf_counter()
    out = ref.run(**d)
    t = time.perf_counter() - t0
out_dir.mkdir(parents=True, exist_ok=True)
if task == "001":
    np.savez(out_dir / f"{case}.npz", **{k: np.asarray(v) for k, v in out.items()})
elif task in ("002", "002b"):
    np.savez(out_dir / f"{case}.npz", **normalize(task, out))
else:
    o = normalize("003", out)
    np.savez(out_dir / f"{case}.npz", i=o["i"], j=o["j"], score=o["score"], n_perfect=o["n_perfect"])
print(json.dumps(dict(task=task, case=case, seconds=t, maxrss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30)))
