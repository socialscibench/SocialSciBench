"""Task 001 competitor: standalone within-py 0.4.0 (Rust), adapted as in the 14 October 2026 study.

Follows research/2026-10-14/native-extension/task001/adapters.py::within_run in the companion
repository: label encoding and recursive singleton pruning, within's batched demeaning with its
default LSMR options, OLS, one-way clustered covariance and the task's absorbed-df rule
(the study's linear-time fast_ssc.absorbed_df). All of it is inside the timed call.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import within


def _recode(ids):
    return np.unique(ids, return_inverse=True)[1].astype(np.int64)


def _drop_singletons(fe):
    keep = np.ones(fe.shape[0], dtype=bool)
    while True:
        changed = False
        for g in range(fe.shape[1]):
            counts = np.bincount(fe[keep, g], minlength=fe[:, g].max() + 1)
            single = keep & (counts[fe[:, g]] == 1)
            if single.any():
                keep &= ~single
                changed = True
        if not changed:
            return keep


def _nested_in(fe, cluster):
    representative = np.empty(int(fe.max()) + 1, dtype=cluster.dtype)
    representative[fe] = cluster
    return bool(np.all(representative[fe] == cluster))


def _absorbed_df(fe_ids, cluster_ids):
    df_a, seen = 0, []
    for g in range(fe_ids.shape[1]):
        fe = fe_ids[:, g]
        L = int(fe.max()) + 1
        if _nested_in(fe, cluster_ids):
            continue
        if not seen:
            df_a += L
        elif len(seen) == 1:
            a = seen[0]
            La = int(a.max()) + 1
            graph = coo_matrix((np.ones(a.size), (a, fe + La)), shape=(La + L, La + L))
            n_comp, _ = connected_components(graph, directed=False)
            df_a += L - n_comp
        else:
            levels = [int(h.max()) + 1 for h in seen if _nested_in(fe, h)]
            df_a += L - max([1] + levels)
        seen.append(fe)
    return df_a - 1 if seen else 0


def run(y, X, fe_ids, cluster_ids, tol=1e-8, maxiter=100_000):
    fe0 = np.column_stack([_recode(fe_ids[:, j]) for j in range(fe_ids.shape[1])])
    keep = _drop_singletons(fe0)
    fe = fe0 if keep.all() else np.column_stack([_recode(fe0[keep, j]) for j in range(fe0.shape[1])])
    cl = _recode(cluster_ids[keep])
    design = np.asfortranarray(fe, dtype=np.uint32)
    matrix = np.asfortranarray(np.column_stack([y[keep], X[keep]]))
    solved = within.solve_batch(design, matrix, options=None)
    if not np.asarray(solved.converged).all():
        raise RuntimeError(f"within did not converge: {solved.converged}, iterations {solved.iterations}")
    Z = np.asarray(solved.demeaned)
    yt, Xt = Z[:, 0], Z[:, 1:]
    n, k = Xt.shape
    G = int(cl.max()) + 1
    bread = np.linalg.inv(Xt.T @ Xt)
    beta = bread @ (Xt.T @ yt)
    residual = yt - Xt @ beta
    scores = np.zeros((G, k))
    np.add.at(scores, cl, Xt * residual[:, None])
    df_a = int(_absorbed_df(fe, cl))
    q = (n - 1) / (n - k - 1 - df_a) * G / (G - 1)
    return dict(beta=beta, vcov=q * bread @ (scores.T @ scores) @ bread, nobs=int(n), df_a=df_a, n_clusters=G)
