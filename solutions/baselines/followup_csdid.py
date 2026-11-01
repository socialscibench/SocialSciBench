"""Follow-up check of the csdid 1.82 baseline on task 002 (study tooling, development inputs only).

usage: followup_csdid.py CASE REFDIR OUTDIR [--numpy-only]

Runs the csdid adapter once on a real-development case and lists, per cohort, the largest
relative difference of its ATT(g,t) from the saved reference output REFDIR/CASE.npz. For each
cohort it then fits the propensity score that drdid fits inside csdid (Stata's `logit` of the
cohort indicator on the covariates, cohort and never-treated units; the covariates do not vary
over time, so every cell of a cohort uses the same fit), logged to OUTDIR/CASE-logit.log so that
Stata's notes on perfect prediction or omitted covariates are visible, and reports the reference's
Newton iteration on the same sample (iterations, last step, Hessian condition number, extreme
fitted probabilities, comparison units above the 0.995 trimming threshold). --numpy-only skips
Stata and reports only the latter. Run with the baseline interpreter from the repository root,
with nothing else running.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "solutions" / "baselines"))
os.environ.setdefault("SOCIALSCIBENCH_BASELINE_VENV", str(ROOT / "solutions" / "baselines" / ".venv"))
os.environ.setdefault("OMP_NUM_THREADS", "1")

from compare import case_spec  # noqa: E402
from harness.common import read_json  # noqa: E402
from harness.datasets import load_inputs  # noqa: E402

case_name, refdir, outdir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
use_stata = "--numpy-only" not in sys.argv[4:]
outdir.mkdir(parents=True, exist_ok=True)
task_dir = ROOT / "002_csdid"
case, _ = case_spec("002", case_name)
data = load_inputs(task_dir, read_json(task_dir / "task.json"), case, str(ROOT / "datasets" / "002" / "data"))
y, X, gvar = (np.asarray(data[k]) for k in ("y", "X", "gvar"))
N, T = y.shape
print(f"case {case_name}: {N} units, {T} periods, {X.shape[1]} covariates, cohorts {np.unique(gvar[gvar > 0]).tolist()}")

if use_stata:
    sys.path.insert(0, str(ROOT / "solutions" / "002" / "stata-csdid-fra"))
    import solution as adapter  # noqa: E402  (starts Stata MP with one processor)
    from pystata import stata  # noqa: E402

    t0 = time.perf_counter()
    out = adapter.run(**data)
    print(f"csdid adapter: {time.perf_counter() - t0:.1f} s")
    ref = np.load(refdir / f"{case_name}.npz")
    got = {tuple(c): a for c, a in zip(out["gt"].tolist(), out["att_gt"])}
    rgt = ref["gt"][[tuple(c) in got for c in ref["gt"].tolist()]]
    print(f"cells: csdid {len(got)}, reference {len(ref['gt'])}, common {len(rgt)}")
    ratt = np.array([a for c, a in zip(ref["gt"].tolist(), ref["att_gt"]) if tuple(c) in got])
    catt = np.array([got[tuple(c)] for c in rgt.tolist()])
    rel = np.abs(catt - ratt) / np.maximum(np.abs(ratt), 1e-300)
    for g in np.unique(rgt[:, 0]):
        sel = rgt[:, 0] == g
        print(f"cohort {g}: {int((gvar == g).sum())} units; ATT(g,t) max rel. difference {rel[sel].max():.2e} "
              f"over {int(sel.sum())} cells ({int((rel[sel] > 1e-6).sum())} above 1e-6)")
    log = outdir / f"{case_name}-logit.log"
    stata.run(f'log using "{log}", text replace', quietly=True)

Xc = np.column_stack([X, np.ones(N)])
for g in np.unique(gvar[gvar > 0]):
    s = (gvar == 0) | (gvar == g)
    D = (gvar[s] == g).astype(float)
    if use_stata:
        df = pd.DataFrame({"D": D, **{f"x{j + 1}": X[s, j] for j in range(X.shape[1])}})
        path = os.path.join(adapter.TMP, "cohort.dta")
        df.to_stata(path, write_index=False, version=118)
        stata.run(f'display "=== cohort {g}"\nuse "{path}", clear\n'
                  f'capture noisily logit D {" ".join(df.columns[1:])}', quietly=False)
    # the reference's Newton iteration (002_csdid/reference.py: logit, tol 1e-12, maxiter 100)
    b = np.zeros(Xc.shape[1])
    try:
        for it in range(100):
            p = 1.0 / (1.0 + np.exp(-Xc[s] @ b))
            H = (Xc[s] * (p * (1 - p))[:, None]).T @ Xc[s]
            step = np.linalg.solve(H, Xc[s].T @ (D - p))
            b += step
            if np.max(np.abs(step)) < 1e-12:
                break
    except np.linalg.LinAlgError as error:
        print(f"cohort {g}: reference Newton failed at iteration {it + 1}: {error}")
        continue
    p = 1.0 / (1.0 + np.exp(-Xc[s] @ b))
    print(f"cohort {g}: reference Newton {it + 1} iterations, last max step {np.max(np.abs(step)):.1e}, "
          f"Hessian condition {np.linalg.cond(H):.1e}, max |coef| {np.max(np.abs(b)):.1f}, "
          f"fitted p of comparison units: min {p[D == 0].min():.2e}, max {p[D == 0].max():.6f}, "
          f"above 0.995: {int((p[D == 0] > 0.995).sum())}; treated min p {p[D == 1].min():.2e}")
if use_stata:
    stata.run("log close", quietly=True)
    print(f"Stata logit output: {log}")
