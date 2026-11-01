"""Reference implementation for task 011_logithdfe_ame.

Logit with any number of high-dimensional fixed effects (the uncorrected "unconditional"
fixed-effects logit), cluster-robust standard errors, and average marginal effects (AMEs)
with delta-method standard errors. Follows the conventions of Stata's

    logit y i.d x1 ... xk i.fe1 i.fe2 ..., vce(cluster c)
    margins, dydx(d x1 ... xk)

where `d` is a binary treatment (its AME is the average discrete change P(d=1) - P(d=0))
and x1..xk are continuous (their AMEs are average derivatives).

Deliberately simple: iteratively reweighted least squares (IRLS). In each iteration the fixed
effects are partialled out of the working response and the covariates by weighted least
squares on the sparse matrix of fixed-effect dummies (SciPy's LSMR), as in task 001.
Correct, readable, not optimized.
"""
import numpy as np
from scipy.sparse import csr_matrix, diags, hstack
from scipy.sparse.linalg import lsmr


def drop_separated(y, fe_ids):
    """Iteratively drop observations in any fixed-effect level whose outcome is all 0 or all 1."""
    keep = np.ones(y.size, dtype=bool)
    while True:
        changed = False
        for g in range(fe_ids.shape[1]):
            f = fe_ids[:, g]
            n = np.bincount(f[keep], minlength=f.max() + 1)
            pos = np.bincount(f[keep], weights=y[keep], minlength=f.max() + 1)
            bad = keep & ((pos[f] == 0) | (pos[f] == n[f]))
            if bad.any():
                keep &= ~bad
                changed = True
        if not changed:
            return keep


def recode(ids):
    return np.unique(ids, return_inverse=True)[1]


def dummies(fe_ids):
    """Sparse matrix of fixed-effect dummies, one block per fixed effect."""
    n = fe_ids.shape[0]
    return hstack([csr_matrix((np.ones(n), (np.arange(n), fe_ids[:, g])), shape=(n, fe_ids[:, g].max() + 1))
                   for g in range(fe_ids.shape[1])]).tocsr()


def demean(V, D, w, tol=1e-11):
    """Weighted residuals of each column of V after regressing it on the fixed-effect dummies D."""
    sw = np.sqrt(w)
    A = diags(sw) @ D @ diags(1 / np.sqrt(D.T @ w))      # weighted, columns scaled to unit norm
    out = np.empty_like(V)
    for j in range(V.shape[1]):
        a = lsmr(A, sw * V[:, j], atol=tol, btol=tol, maxiter=100_000)[0]
        out[:, j] = V[:, j] - (A @ a) / sw
    return out


def run(y, d, X, fe_ids, cluster_ids, tol=1e-10, maxiter=100):
    """
    y            [n_obs]                 float64  binary outcome
    d            [n_obs]                 float64  binary treatment (0/1)
    X            [n_obs, n_covariates]   float64  continuous covariates
    fe_ids       [n_obs, n_fe]           int64    fixed-effect codes (interactions pre-encoded)
    cluster_ids  [n_obs]                 int64    cluster codes
    returns dict(beta, se, ame, ame_se [1 + n_covariates], with d first; nobs, n_clusters,
                 fe_levels, iterations, timings)
    """
    import time
    t0 = time.perf_counter()
    keep = drop_separated(y, fe_ids)
    y, cl = y[keep], recode(cluster_ids[keep])
    W = np.column_stack([d[keep], X[keep]])                  # coefficients: d first, then x1..xk
    fe = np.column_stack([recode(fe_ids[keep, g]) for g in range(fe_ids.shape[1])])
    D = dummies(fe)
    n, k = W.shape

    # 1. IRLS: weighted least squares of the working response on W and the fixed effects
    mu = (y + 0.5) / 2
    eta = np.log(mu / (1 - mu))
    beta = np.zeros(k)
    for it in range(1, maxiter + 1):
        p = 1 / (1 + np.exp(-eta))
        lam = p * (1 - p)
        z = eta + (y - p) / lam
        R = demean(np.column_stack([z, W]), D, lam)
        zt, Wt = R[:, 0], R[:, 1:]
        beta_new = np.linalg.solve((Wt * lam[:, None]).T @ Wt, (Wt * lam[:, None]).T @ zt)
        eta = z - (zt - Wt @ beta_new)                       # fitted index, fixed effects included
        step = np.max(np.abs(beta_new - beta))
        beta = beta_new
        if step < tol:
            break
    else:
        raise RuntimeError(f"did not converge in {maxiter} iterations")
    p = 1 / (1 + np.exp(-eta))
    lam = p * (1 - p)
    Wt = demean(W, D, lam)
    S_inv = np.linalg.inv((Wt * lam[:, None]).T @ Wt)
    G = cl.max() + 1
    q = G / (G - 1)
    e = y - p

    # 2. Cluster-robust variance of the coefficients (the coefficient block of the full sandwich)
    psi = e[:, None] * (Wt @ S_inv)
    Vc = np.stack([np.bincount(cl, weights=psi[:, j], minlength=G) for j in range(k)], axis=1)
    se = np.sqrt(np.diag(q * Vc.T @ Vc))
    t1 = time.perf_counter()

    # 3. AMEs. Each is g = mean_i f_i; its gradient has a coefficient part J and a fixed-effect
    #    part sum_i a_i * (dummy row i). The delta-method variance over ALL parameters reduces to
    #    q * sum_c (sum_{i in c} e_i * (Wt_i' S_inv J_eff + r_i - rt_i))^2, with r = a / lam,
    #    rt = r with the fixed effects partialled out, and J_eff = J - (W - Wt)' diag(lam) r.
    def ame_and_se(f, J, a):
        r = a / lam
        rt = demean(r[:, None], D, lam)[:, 0]
        J_eff = J - (W - Wt).T @ (lam * r)
        infl = e * (Wt @ (S_inv @ J_eff) + r - rt)
        return f.mean(), np.sqrt(q * (np.bincount(cl, weights=infl, minlength=G) ** 2).sum())

    ame, ame_se = np.empty(k), np.empty(k)
    # Binary treatment: average of P(y=1 | d=1) - P(y=1 | d=0)
    eta1, eta0 = eta + beta[0] * (1 - W[:, 0]), eta - beta[0] * W[:, 0]
    p1, p0 = 1 / (1 + np.exp(-eta1)), 1 / (1 + np.exp(-eta0))
    l1, l0 = p1 * (1 - p1), p0 * (1 - p0)
    J = np.r_[l1.mean(), ((l1 - l0)[:, None] * W[:, 1:]).mean(0)]
    ame[0], ame_se[0] = ame_and_se(p1 - p0, J, (l1 - l0) / n)
    # Continuous covariates: average derivative beta_j * p (1 - p)
    dlam = lam * (1 - 2 * p)
    for j in range(1, k):
        J = beta[j] * (dlam[:, None] * W).mean(0)
        J[j] += lam.mean()
        ame[j], ame_se[j] = ame_and_se(beta[j] * lam, J, beta[j] * dlam / n)
    t2 = time.perf_counter()

    return dict(beta=beta, se=se, ame=ame, ame_se=ame_se, nobs=n, n_clusters=G,
                fe_levels=[int(fe[:, g].max() + 1) for g in range(fe.shape[1])], iterations=it,
                timings=dict(estimation=t1 - t0, ame=t2 - t1))
