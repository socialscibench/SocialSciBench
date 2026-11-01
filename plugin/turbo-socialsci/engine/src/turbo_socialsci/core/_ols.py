"""Collinear regressors and sums of squares for the fixed-effects regressions (hdfe_speed, hdfe_memory).

reghdfe drops a regressor whose partialled values are all close to zero (collinear with the fixed
effects) and, scanning left to right, a regressor that is a linear combination of earlier kept ones.
Both are decided here from the partialled cross-product matrix, which the solvers form anyway.
"""
import numpy as np
import numba as nb

COLLINEAR_TOL = 1e-9     # reghdfe's tolerance for "collinear with the fixed effects"


@nb.njit(parallel=True, cache=True)
def centered_ss(X, idx):
    """Sum of squared deviations from the column mean of each column of X over the rows idx (two
    passes, per-part partial sums; no copy of X)."""
    n = idx.size
    k = X.shape[1]
    nparts = max(1, min(256, n // 4096))
    part = np.zeros((nparts, k))
    for c in nb.prange(nparts):
        for r in range(c * n // nparts, (c + 1) * n // nparts):
            i = idx[r]
            for j in range(k):
                part[c, j] += X[i, j]
    mean = np.zeros(k)
    for c in range(nparts):
        for j in range(k):
            mean[j] += part[c, j]
    for j in range(k):
        mean[j] /= max(n, 1)
    part[:, :] = 0.0
    for c in nb.prange(nparts):
        for r in range(c * n // nparts, (c + 1) * n // nparts):
            i = idx[r]
            for j in range(k):
                d = X[i, j] - mean[j]
                part[c, j] += d * d
    ss = np.zeros(k)
    for c in range(nparts):
        for j in range(k):
            ss[j] += part[c, j]
    return ss


def independent(XX, ss, tol=COLLINEAR_TOL):
    """Mask of the regressors kept, from XX = X~'X~ (partialled regressors) and ss = their sums of
    squares before partialling. Column j is dropped when, after projection on the kept columns to its
    left, its squared norm is at most tol * ss[j] (a constant column, ss[j] = 0, is always dropped)."""
    k = XX.shape[0]
    keep = np.zeros(k, dtype=bool)
    L = np.zeros((k, k))                      # Cholesky factor of the kept columns' block
    m = 0
    kept = []
    for j in range(k):
        if not ss[j] > 0:
            continue
        if m:
            z = np.linalg.solve(L[:m, :m], XX[kept, j])
            d = XX[j, j] - z @ z
        else:
            z = None
            d = XX[j, j]
        if not d > tol * ss[j]:
            continue
        if m:
            L[m, :m] = z
        L[m, m] = np.sqrt(d)
        m += 1
        kept.append(j)
        keep[j] = True
    return keep


def fit_stats(rss, y, idx, yy_within):
    """R² of the full model (fixed effects included) and within R², on the estimation sample."""
    yk = y[idx]
    tss = float(((yk - yk.mean()) ** 2).sum()) if yk.size else 0.0
    r2 = 1.0 - rss / tss if tss > 0 else float("nan")
    r2_within = 1.0 - rss / yy_within if yy_within > 0 else float("nan")
    return dict(rss=float(rss), tss=tss, r2=float(r2), r2_within=float(r2_within))
