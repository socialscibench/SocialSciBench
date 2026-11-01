"""Reference implementation for task 004_winsorize.

Winsorize each variable at the p_lo-th and p_hi-th percentiles, pooled or within groups,
following Stata's

    winsor2 v1 ... vk, cuts(p_lo p_hi) [by(group)] replace

Percentiles use Stata's default definition (as in _pctile and egen pctile()): with n
non-missing values sorted x(1) <= ... <= x(n), if n*p/100 is a whole number the p-th
percentile is (x(n*p/100) + x(n*p/100 + 1)) / 2, otherwise x(floor(n*p/100) + 1). This is
R's quantile(type = 2) and NumPy's "averaged_inverted_cdf", NOT the default of R's
quantile() (type 7), pandas or NumPy, which interpolate. Missing values stay missing.

Deliberately simple: one sort per variable. Correct, readable, not optimized.
"""
import numpy as np


def stata_percentile_index(n, p):
    """For group sizes n (array) and percentile p, return (j, average): the percentile is
    x(j+1) if not average, else (x(j) + x(j+1)) / 2, with 1-based order statistics x.
    Uses the same floating-point test as Stata's egen pctile()."""
    np_ = n * p / 100
    average = 100 * np.round(np_) == n * p
    j = np.where(average, np.round(np_), np.floor(np_)).astype(np.int64)
    return j, average


def run(X, group, cuts):
    """
    X      [n_obs, n_vars]  float64  NaN = missing
    group  [n_obs] int64 or None     by() groups (None = pooled)
    cuts   (p_lo, p_hi)              percentiles, e.g. (1, 99)
    returns dict(X_w [n_obs, n_vars], lo, hi [n_groups, n_vars])
    """
    n_obs, k = X.shape
    g = np.zeros(n_obs, dtype=np.int64) if group is None else np.unique(group, return_inverse=True)[1]
    G = g.max() + 1
    Xw = X.copy()
    lo, hi = np.full((G, k), np.nan), np.full((G, k), np.nan)
    for v in range(k):
        x = X[:, v]
        order = np.lexsort((x, g))                      # by group, then value; NaN last within group
        xs, gs = x[order], g[order]
        start = np.searchsorted(gs, np.arange(G))       # first row of each group in sorted order
        n = np.bincount(g, weights=~np.isnan(x), minlength=G).astype(np.int64)
        ok = n > 0
        for p, out in ((cuts[0], lo), (cuts[1], hi)):
            j, avg = stata_percentile_index(n, p)
            upper = xs[np.minimum(start + j, n_obs - 1)]                 # x(j+1), 0-based start + j
            lower = xs[np.maximum(start + j - 1, 0)]                     # x(j)
            out[:, v] = np.where(ok, np.where(avg, (lower + upper) / 2, upper), np.nan)
        keep = ~np.isnan(x) & ok[g]
        Xw[keep, v] = np.minimum(np.maximum(x[keep], lo[g[keep], v]), hi[g[keep], v])
    return dict(X_w=Xw, lo=lo, hi=hi)


def checksums(X, X_w):
    """Per-variable summaries used to compare tools: count, sum, sum of squares, min, max,
    and how many values winsorizing changed."""
    out = []
    for v in range(X.shape[1]):
        x, w = X[:, v], X_w[:, v]
        m = ~np.isnan(w)
        out.append(dict(n=int(m.sum()), sum=float(w[m].sum()), sumsq=float((w[m] ** 2).sum()),
                        min=float(w[m].min()), max=float(w[m].max()), changed=int((w[m] != x[m]).sum())))
    return out
