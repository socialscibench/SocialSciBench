"""Reference implementation for task 001_reghdfe_ols_cluster.

Linear regression with multiple high-dimensional fixed effects and one-way
cluster-robust standard errors, following the conventions of Stata's
`reghdfe y X, absorb(fe1 fe2 ...) vce(cluster c)` (Correia 2017, v6.12).

Deliberately simple: fixed effects are partialled out with an off-the-shelf
sparse least-squares solver (SciPy's LSMR), one column at a time. Correct,
readable, not optimized.
"""
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, hstack
from scipy.sparse.linalg import lsmr
from scipy.sparse.csgraph import connected_components


def drop_singletons(fe_ids):
    """Iteratively flag observations that are the only member of some FE level."""
    keep = np.ones(fe_ids.shape[0], dtype=bool)
    while True:
        changed = False
        for g in range(fe_ids.shape[1]):
            counts = np.bincount(fe_ids[keep, g], minlength=fe_ids[:, g].max() + 1)
            single = keep & (counts[fe_ids[:, g]] == 1)
            if single.any():
                keep &= ~single
                changed = True
        if not changed:
            return keep


def recode(ids):
    """Map arbitrary integer codes to 0..L-1."""
    _, codes = np.unique(ids, return_inverse=True)
    return codes.astype(np.int64)


def demean(M, fe_ids, tol, maxiter):
    """Partial out all fixed effects from every column of M, in place.

    Each column v is regressed on the sparse matrix D of FE dummies, and the
    residual v - D a is kept. The least-squares problem is solved with LSMR
    (Fong & Saunders 2011), with D's columns scaled to unit norm, so it converges
    even when the FE graph is poorly connected (e.g. low worker mobility).
    """
    n, n_fe = fe_ids.shape
    blocks = []
    for g in range(n_fe):
        L = fe_ids[:, g].max() + 1
        scale = 1.0 / np.sqrt(np.bincount(fe_ids[:, g]))
        blocks.append(csr_matrix((scale[fe_ids[:, g]], (np.arange(n), fe_ids[:, g])), shape=(n, L)))
    D = hstack(blocks).tocsr()
    for j in range(M.shape[1]):
        a, istop, itn = lsmr(D, M[:, j], atol=tol, btol=tol, maxiter=maxiter)[:3]
        if istop == 7:
            raise RuntimeError(f"column {j} did not converge in {maxiter} iterations")
        M[:, j] -= D @ a
    return M


def nested_in(fe, cluster):
    """True if every level of `fe` lies inside exactly one cluster."""
    pairs = np.unique(np.stack([fe, cluster], axis=1), axis=0)
    return pairs.shape[0] == fe.max() + 1


def absorbed_df(fe_ids, cluster_ids):
    """Degrees of freedom absorbed by the fixed effects (reghdfe conventions).

    - FEs nested within the cluster variable contribute 0.
    - The first non-nested FE contributes all its levels.
    - The second contributes levels minus the number of connected components of
      the bipartite graph it forms with the first (exact redundancy).
    - Any further FE contributes levels minus 1, or minus the levels of an earlier
      FE it is nested in (e.g. region#year within year). Conservative, like reghdfe.
    One level is then handed to the intercept, which reghdfe reports as _cons
    (so this matches e(df_a), and K = k + 1 + df_a).
    """
    df_a = 0
    seen = []
    for g in range(fe_ids.shape[1]):
        fe = fe_ids[:, g]
        L = fe.max() + 1
        if nested_in(fe, cluster_ids):
            continue
        if not seen:
            df_a += L
        elif len(seen) == 1:
            a, b = seen[0], fe
            La = a.max() + 1
            graph = coo_matrix((np.ones(a.size), (a, b + La)), shape=(La + L, La + L))
            n_comp, _ = connected_components(graph, directed=False)
            df_a += L - n_comp
        else:
            nested_levels = [h.max() + 1 for h in seen if nested_in(fe, h)]
            df_a += L - max([1] + nested_levels)
        seen.append(fe)
    return df_a - 1 if seen else 0


def run(y, X, fe_ids, cluster_ids, tol=1e-8, maxiter=100_000):
    """
    y           [n_obs]                float64
    X           [n_obs, n_regressors]  float64
    fe_ids      [n_obs, n_fe]          int64   (FE level codes, any integers)
    cluster_ids [n_obs]                int64
    returns dict(beta [k], vcov [k, k], nobs, df_a, n_clusters)
    """
    keep = drop_singletons(np.column_stack([recode(fe_ids[:, g]) for g in range(fe_ids.shape[1])]))
    y, X = y[keep], X[keep]
    fe_ids = np.column_stack([recode(fe_ids[keep, g]) for g in range(fe_ids.shape[1])])
    cluster_ids = recode(cluster_ids[keep])
    n, k = X.shape

    # 1. Partial out fixed effects (Frisch-Waugh-Lovell)
    Z = demean(np.column_stack([y, X]), fe_ids, tol, maxiter)
    y_t, X_t = Z[:, 0], Z[:, 1:]

    # 2. OLS on demeaned data
    XtX_inv = np.linalg.inv(X_t.T @ X_t)
    beta = XtX_inv @ (X_t.T @ y_t)
    resid = y_t - X_t @ beta

    # 3. One-way cluster-robust sandwich
    G = cluster_ids.max() + 1
    scores = np.zeros((G, k))
    np.add.at(scores, cluster_ids, X_t * resid[:, None])
    meat = scores.T @ scores

    # 4. Small-sample correction: (N-1)/(N-K) * G/(G-1), K = k + _cons + df_a
    df_a = absorbed_df(fe_ids, cluster_ids)
    K = k + 1 + df_a
    q = (n - 1) / (n - K) * G / (G - 1)
    vcov = q * XtX_inv @ meat @ XtX_inv

    return dict(beta=beta, vcov=vcov, nobs=n, df_a=df_a, n_clusters=G)
