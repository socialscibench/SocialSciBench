"""Why csdid 1.82 differs from the reference on 002/108 (ipw, not-yet-treated controls, universal base).

usage (repository root, nothing else running):
  OMP_NUM_THREADS=1 solutions/baselines/.venv/py/bin/python solutions/002/stata-csdid181/diagnosis-108/diag108.py [N_WORST [CELLS_NPZ]]

1. The reference on 108 without the bootstrap, recording each cell's propensity logit (one per cell
   for ipw, in the order of the reference's gt): sizes, the largest control score and odds, the
   Hessian's condition number.
2. The stata-csdid181 adapter's analytic part on the same inputs; per-cell errors as the validator
   scales them (|csdid - reference| / max(|reference|, reference SE)).
3. For the N_WORST cells: Stata's own `logit D x...` on the cell's data (drdid's call), its
   iterations and coefficients, and the reference's log likelihood and gradient at both solutions.
4. With CELLS_NPZ, the per-cell estimates and SEs of both are saved there. They include reference
   outputs, so keep the file outside the repository.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from harness.common import load_module  # noqa: E402

TASK = Path("002_csdid")
ref = load_module(TASK / "reference.py", "ref002_diag")
rd = load_module(TASK / "realdata.py", "rd002_diag")
d = rd.load(108, "datasets/002/data")
d.pop("boot_weights", None)
n_worst = int(sys.argv[1]) if len(sys.argv) > 1 else 6
cells_npz = sys.argv[2] if len(sys.argv) > 2 else None

log = []
_logit = ref.logit


def logit(D, X, w, tol=1e-10, maxiter=100):
    index, V = _logit(D, X, w, tol, maxiter)
    ps = 1 / (1 + np.exp(-index))
    c = D == 0
    H = np.linalg.inv(V)
    log.append(dict(D=D.copy(), X=X.copy(), w=w.copy(), n=D.size, n_treated=int(D.sum()), n_controls=int(c.sum()),
                    max_ps=float(ps[c].max()), max_odds=float((np.minimum(ps[c], ref.CAP) / (1 - np.minimum(ps[c], ref.CAP))).max()),
                    cond=float(np.linalg.cond(H)), b=np.linalg.lstsq(X, index, rcond=None)[0]))
    return index, V


ref.logit = logit
t0 = time.perf_counter()
R = ref.run(**d)
print(f"reference: {time.perf_counter() - t0:.1f} s, {len(R['gt'])} cells, {len(log)} logits")
assert len(log) == len(R["gt"]), "expected one logit per cell"

csdid = load_module(Path("solutions/002/stata-csdid181/solution.py"), "csdid181_diag")
t0 = time.perf_counter()
C = csdid.run(**d)
print(f"csdid 1.82 (analytic part): {time.perf_counter() - t0:.1f} s, {len(C['gt'])} cells")
assert np.array_equal(C["gt"], R["gt"]), "different cells"
scale = np.maximum(np.abs(R["att_gt"]), R["se_gt"])
err = np.abs(C["att_gt"] - R["att_gt"]) / scale
order = np.argsort(-err)
print(f"cells failing 1e-6: {(err > 1e-6).sum()} of {err.size}; above 1e-3: {(err > 1e-3).sum()}; above 0.1: {(err > 0.1).sum()}")
fail_g, fail_n = np.unique(R["gt"][err > 1e-6, 0], return_counts=True)
print("failing cells by cohort:", {int(g): int(n) for g, n in zip(fail_g, fail_n)},
      "; cells per cohort among them:", {int(g): int((R["gt"][:, 0] == g).sum()) for g in fail_g})
if cells_npz:
    np.savez(cells_npz, gt=R["gt"], att_ref=R["att_gt"], se_ref=R["se_gt"], att_csdid=C["att_gt"], se_csdid=C["se_gt"],
             err=err)
odds = np.array([e["max_odds"] for e in log])
cond = np.array([e["cond"] for e in log])
for lab, m in (("failing", err > 1e-6), ("passing", err <= 1e-6)):
    if m.any():
        print(f"  {lab}: max control odds median {np.median(odds[m]):.3g} (max {odds[m].max():.3g}); "
              f"Hessian condition median {np.median(cond[m]):.3g}; controls median {np.median([log[i]['n_controls'] for i in np.nonzero(m)[0]]):.0f}")
print("\nworst cells (g, t): error, reference ATT and SE, csdid ATT, controls, treated, max control score, odds, condition")
for i in order[:15]:
    e = log[i]
    print(f"  {tuple(R['gt'][i])}: {err[i]:.3g}, {R['att_gt'][i]:.4g} ({R['se_gt'][i]:.3g}), {C['att_gt'][i]:.4g}, "
          f"{e['n_controls']}, {e['n_treated']}, {e['max_ps']:.6f}, {e['max_odds']:.3g}, {e['cond']:.3g}")

# Stata's logit on the worst cells, as drdid calls it
from pystata import stata  # noqa: E402  (initialized by the adapter)
from sfi import Matrix, Scalar  # noqa: E402
import pandas as pd  # noqa: E402


def loglik(b, D, X, w):
    xb = X @ b
    return float(np.sum(w * (D * xb - np.logaddexp(0, xb))))


def grad(b, D, X, w):
    p = 1 / (1 + np.exp(-(X @ b)))
    return X.T @ (w * (D - p))


print("\nStata logit on the worst cells (drdid: logit trt x [iw = w]); reference ll at its MLE and at Stata's")
for i in order[:n_worst]:
    e = log[i]
    k = e["X"].shape[1] - 1                       # the last column is the constant
    df = pd.DataFrame({"D": e["D"], "w": e["w"], **{f"x{j + 1}": e["X"][:, j] for j in range(k)}})
    path = os.path.join(csdid.TMP, "cell.dta")
    df.to_stata(path, write_index=False, version=118)
    stata.run(f'use "{path}", clear', quietly=True)
    stata.run("logit D " + " ".join(f"x{j + 1}" for j in range(k)) + " [iw = w]", quietly=True)
    stata.run("matrix SSB_lb = e(b)\nscalar SSB_it = e(ic)\nscalar SSB_cv = e(converged)\nscalar SSB_ll = e(ll)", quietly=True)
    bs = np.asarray(Matrix.get("SSB_lb"), dtype=np.float64)[0]          # x1..xk, _cons: the reference's order
    br = e["b"]
    llr, lls = loglik(br, e["D"], e["X"], e["w"]), loglik(bs, e["D"], e["X"], e["w"])
    ps_r = 1 / (1 + np.exp(-(e["X"] @ br)))
    ps_s = 1 / (1 + np.exp(-(e["X"] @ bs)))
    c = e["D"] == 0
    odds = lambda p: np.minimum(p, ref.CAP) / (1 - np.minimum(p, ref.CAP))
    print(f"  {tuple(R['gt'][i])}: Stata {int(Scalar.getValue('SSB_it'))} iterations, converged "
          f"{int(Scalar.getValue('SSB_cv'))}, ll {Scalar.getValue('SSB_ll'):.10g}; reference ll {llr:.10g} at its MLE, "
          f"{lls:.10g} at Stata's; |gradient| {np.abs(grad(br, e['D'], e['X'], e['w'])).max():.2g} / "
          f"{np.abs(grad(bs, e['D'], e['X'], e['w'])).max():.2g}; max |b diff| {np.abs(bs - br).max():.3g}; "
          f"control odds sum {odds(ps_r[c]).sum():.6g} / {odds(ps_s[c]).sum():.6g}")
stata.run("clear", quietly=True)
