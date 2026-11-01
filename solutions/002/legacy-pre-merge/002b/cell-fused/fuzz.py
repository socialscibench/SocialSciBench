"""Random small task-002b problems compared with the reference (harness validator).

usage (repository root): python solutions/002b/cell-fused/fuzz.py CANDIDATE.py N_CASES SEED
Every case draws a panel (balanced or with gaps and exits), covariates (continuous and
discrete), cohorts (including tiny ones), optional weights, the comparison group, the method
and an event window; cases where the reference itself fails are counted separately.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from harness.common import load_module  # noqa: E402
from harness.validation import normalize, validate  # noqa: E402

ref = load_module(Path("002b_csdid_options/reference.py"), "ref002b")
cand = load_module(Path(sys.argv[1]), "cand002b")
tol = json.load(open("002b_csdid_options/task.json"))["tolerances"]
n_cases, seed0 = int(sys.argv[2]), int(sys.argv[3])
stats = dict(passed=0, failed=0, ref_invalid=0, cand_error=0)
for case in range(n_cases):
    rng = np.random.default_rng(seed0 + case)
    N = int(rng.integers(30, 500)); T = int(rng.integers(3, 13)); k = int(rng.integers(0, 4))
    X = rng.standard_normal((N, k)) * rng.choice([1, 3], size=k) + rng.choice([0, 5], size=k)
    if k and rng.random() < 0.3:
        X[:, 0] = rng.integers(0, 3, N)
    ncoh = int(rng.integers(1, T))
    cohorts = np.sort(rng.choice(np.arange(2, T + 1), size=min(ncoh, T - 1), replace=False))
    g = np.where(rng.random(N) < rng.uniform(0.3, 0.8), 0, rng.choice(cohorts, N))
    if rng.random() < 0.2:
        g[rng.choice(N, size=3, replace=False)] = 1          # always treated: dropped by the sample rule
    y = rng.standard_normal((N, T)) + (X[:, :1] if k else 0) * np.arange(T) / T
    if rng.random() < 0.2:
        y = (y > 0.5).astype(float)
    if rng.random() < 0.5:                                    # unbalanced: gaps and exits
        obs = rng.random((N, T)) > rng.uniform(0.05, 0.3)
        obs &= np.arange(T)[None, :] <= (rng.integers(1, T, N) + (rng.random(N) < 0.5) * T)[:, None]   # exits
        y = np.where(obs, y, np.nan)
    w = np.exp(rng.standard_normal(N) * 0.5) if rng.random() < 0.5 else None
    d = dict(y=y, X=X, gvar=g.astype(np.int64), w=w, control=str(rng.choice(["never", "notyet"])),
             method=str(rng.choice(["dripw", "reg", "ipw", "stdipw", "drimp"])),
             event_window=(int(rng.integers(-6, 0)), int(rng.integers(0, 6))))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exp = normalize("002b", ref.run(**d))
        if not all(np.all(np.isfinite(v)) for v in exp.values()):
            stats["ref_invalid"] += 1
            continue
    except Exception:
        stats["ref_invalid"] += 1
        continue
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = normalize("002b", cand.run(**d))
    except Exception as e:
        stats["cand_error"] += 1
        print("cand error", seed0 + case, repr(e))
        continue
    rep = validate("002b", out, exp, tol, {})
    if rep["passed"]:
        stats["passed"] += 1
    else:
        stats["failed"] += 1
        bad = {kk: vv for kk, vv in rep["checks"].items() if not vv.get("passed", True)}
        print("FAIL case", seed0 + case, f"N={N} T={T} k={k} {d['control']} {d['method']} w={w is not None} "
              f"nan={int(np.isnan(y).sum())}", json.dumps(bad, default=str)[:500])
print(stats)
