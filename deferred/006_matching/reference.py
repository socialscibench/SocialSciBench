"""Reference implementation for task 006_matching.

Two matching estimators of the average treatment effect on the treated (ATT):

1. Propensity-score matching (PSM): logit propensity score on all covariates, then each
   treated unit is matched to the control unit with the nearest propensity score, with
   replacement (one control may serve several treated units). Exactly tied controls are all
   used and averaged. ATT = mean over treated of (y_treated - y_matched controls). As in
   Stata's  teffects psmatch (y) (d x1 ... xk, logit), atet nneighbor(1)
   and      psmatch2 d x1 ... xk, outcome(y) logit neighbor(1).

2. Coarsened exact matching (CEM), following Stata's  cem x1 ... x4, treatment(d):
   each covariate is cut at ceil(log2(n) + 1) equally spaced cutpoints from its minimum to
   its maximum (Sturges' rule, as cem defines it), a value equal to a cutpoint goes to the
   lower bin, and observations with the same bins on all covariates form a stratum. Strata
   with both treated and control units are matched. Weights: 1 for matched treated units;
   (treated_s / controls_s) * (matched controls / matched treated) for matched controls in
   stratum s; 0 otherwise. ATT = weighted mean of y for treated minus that for controls.

Deliberately simple. Correct, readable, not optimized.
"""
import numpy as np


def logit(d, X, tol=1e-12, maxiter=100):
    """Logit MLE (with a constant) by Newton-Raphson; returns coefficients and fitted probabilities."""
    Z = np.column_stack([X, np.ones(len(d))])
    b = np.zeros(Z.shape[1])
    for _ in range(maxiter):
        p = 1 / (1 + np.exp(-Z @ b))
        step = np.linalg.solve((Z * (p * (1 - p))[:, None]).T @ Z, Z.T @ (d - p))
        b += step
        if np.max(np.abs(step)) < tol:
            break
    return b, 1 / (1 + np.exp(-Z @ b))


def psm(y, d, X):
    _, ps = logit(d, X)
    t, c = np.flatnonzero(d == 1), np.flatnonzero(d == 0)
    order = np.argsort(ps[c], kind="stable")
    cps, cy = ps[c][order], y[c][order]
    # Nearest control for each treated unit: compare the neighbours on each side in sorted order
    pos = np.searchsorted(cps, ps[t])
    left, right = np.clip(pos - 1, 0, len(c) - 1), np.clip(pos, 0, len(c) - 1)
    dist = np.minimum(np.abs(ps[t] - cps[left]), np.abs(ps[t] - cps[right]))
    # All controls at exactly the nearest distance (ties) are averaged
    lo = np.searchsorted(cps, ps[t] - dist, side="left")
    hi = np.searchsorted(cps, ps[t] + dist, side="right")
    csum = np.r_[0, np.cumsum(cy)]
    matched_mean = (csum[hi] - csum[lo]) / (hi - lo)
    att = float(np.mean(y[t] - matched_mean))
    return dict(att=att, n_treated=len(t), n_matched_controls=int(np.unique(order[lo]).size),
                ties=int(np.sum(hi - lo > 1)))


def cem_cutpoints(x):
    """Stata cem's Sturges rule: ceil(log2(n) + 1) equally spaced cutpoints from min to max."""
    k = int(np.ceil(np.log10(len(x)) / np.log10(2) + 1))
    return np.linspace(x.min(), x.max(), k)


def cem(y, d, X):
    bins = np.column_stack([(X[:, j][:, None] >= (b := cem_cutpoints(X[:, j]))[:1]).sum(1)
                            + (X[:, j][:, None] > b[1:]).sum(1) for j in range(X.shape[1])])
    strata = np.unique(bins, axis=0, return_inverse=True)[1].ravel()
    S = strata.max() + 1
    nt = np.bincount(strata, weights=d, minlength=S)
    nc = np.bincount(strata, weights=1 - d, minlength=S)
    matched = ((nt > 0) & (nc > 0))[strata]
    mt, mc = (matched & (d == 1)).sum(), (matched & (d == 0)).sum()
    w = np.where(matched, np.where(d == 1, 1.0, (nt / np.maximum(nc, 1))[strata] * mc / mt), 0.0)
    att = float(np.sum(w * d * y) / np.sum(w * d) - np.sum(w * (1 - d) * y) / np.sum(w * (1 - d)))
    return dict(att=att, n_strata=int(S), n_matched_strata=int(((nt > 0) & (nc > 0)).sum()),
                n_matched_treated=int(mt), n_matched_controls=int(mc), weight_sum=float(w.sum()))


def run(y, d, X, cem_covariates=4):
    """
    y  [n] float64 outcome;  d [n] float64 treatment (0/1);  X [n, k] float64 covariates
    returns dict(psm=..., cem=...)
    """
    return dict(psm=psm(y, d, X), cem=cem(y, d, X[:, :cem_covariates]))
