"""Random small problems against the reference (development check, not part of the candidate).

Run from the repository root:  python solutions/002/shared-nuisance/fuzz.py [n_cases]
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "002_csdid"))
import reference  # noqa: E402

spec = importlib.util.spec_from_file_location("cand", Path(__file__).with_name("solution.py"))
cand = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cand)


def case(rng):
    N = int(rng.integers(60, 600))
    T = int(rng.integers(3, 12))
    k = int(rng.integers(1, 5))
    X = rng.standard_normal((N, k)) * rng.uniform(0.3, 2.0, k) + rng.normal(0, 1, k)
    first = int(rng.integers(2, T + 1))
    cohorts = np.unique(rng.integers(first, T + 1, int(rng.integers(1, 5))))
    strength = rng.uniform(0, 2.5)                # large values push propensities past 0.995
    p = 1 / (1 + np.exp(-(rng.normal(0, 1) + strength * X[:, 0])))
    gvar = np.where(rng.random(N) < p, rng.choice(cohorts, N), 0)
    gvar[: 5] = 0                                  # keep some controls
    for g in cohorts:                              # and at least three units per cohort
        idx = rng.choice(np.flatnonzero(gvar == 0)[5:], 3, replace=False)
        gvar[idx] = g
    t = np.arange(1, T + 1)
    y = (rng.standard_normal(N)[:, None] + 0.1 * t + X @ rng.standard_normal((k, T)) * 0.2
         + ((gvar[:, None] > 0) & (t >= gvar[:, None])) * 1.0 + rng.standard_normal((N, T)))
    order = rng.permutation(N)
    return dict(y=y[order], X=X[order], gvar=gvar[order].astype(np.int64))


def compare(out, ref):
    worst = 0.0
    for key, v in ref.items():
        a, b = np.asarray(out[key]), np.asarray(v)
        if key in ("gt", "event_times"):
            if not np.array_equal(a, b):
                return np.inf
            continue
        if not np.all(np.isfinite(b)):
            continue                               # the reference itself is not finite
        tol = 1e-4 if key.startswith("se") else 1e-6
        worst = max(worst, float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12))) / tol)
    return worst


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    rng = np.random.default_rng(20261018)
    failures, trimmed, worst = 0, 0, 0.0
    for i in range(n):
        d = case(rng)
        with np.errstate(all="ignore"):
            ref = reference.run(**d)
            Xc = np.column_stack([d["X"], np.ones(len(d["gvar"]))])
            for g in np.unique(d["gvar"][d["gvar"] > 0]):
                s = (d["gvar"] == 0) | (d["gvar"] == g)
                b, _ = reference.logit((d["gvar"][s] == g).astype(float), Xc[s])
                if np.any(1 / (1 + np.exp(-Xc[d["gvar"] == 0] @ b)) > cand.TRIM):
                    trimmed += 1
                    break
        out = cand.run(**d)
        score = compare(out, ref)
        worst = max(worst, score)
        if score > 1:
            failures += 1
            print(f"case {i}: worst error / tolerance = {score:.3g}")
    print(f"{n} cases ({trimmed} with trimmed controls), {failures} failures, "
          f"worst error / tolerance {worst:.2e}")
