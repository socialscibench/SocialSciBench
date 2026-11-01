"""Task 001 baseline: pyfixest.feols (py-econometrics) with its default demeaner, the method of alternating projections
(Rust backend).

    feols("y ~ x1 + ... + xk | fe1 + ... + feG", data, vcov={"CRV1": "cl"}, fixef_rm="singleton",
          ssc=ssc(k_adj=False, G_adj=False), demeaner=DEMEANER, lean=True, store_data=False)

pyfixest drops singleton fixed-effect levels recursively, partials out the fixed effects and
returns the coefficients and the unadjusted clustered sandwich. reghdfe's df_a and the reference's
small-sample factor are computed as in the fixest baseline (NumPy/SciPy/pandas). Building the
data frame, this computation and output conversion are inside the timed call.

Threads: the Rust backends use Rayon (RAYON_NUM_THREADS = the evaluator's thread limit).
"""
import numpy as np
import pandas as pd
import pyfixest as pf
from pyfixest.demeaners import LsmrDemeaner, MapDemeaner
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

DEMEANER = MapDemeaner(backend="rust", fixef_tol=1e-8, fixef_maxiter=100_000)


def _codes(a):
    return pd.factorize(a, sort=False)[0]


def _nested(a, b):
    return bool((pd.Series(b).groupby(a).nunique() == 1).all())


def _absorbed_df(fe, cl):
    """reghdfe's df_a, as in reference.absorbed_df (codes of the kept rows)."""
    df_a = 0
    seen = []
    for g in range(fe.shape[1]):
        a = fe[:, g]
        La = int(a.max()) + 1
        if _nested(a, cl):
            continue
        if not seen:
            df_a += La
        elif len(seen) == 1:
            h = seen[0]
            Lh = int(h.max()) + 1
            graph = coo_matrix((np.ones(a.size), (h, a + Lh)), shape=(Lh + La, Lh + La))
            df_a += La - connected_components(graph, directed=False)[0]
        else:
            nested_levels = [int(h.max()) + 1 for h in seen if _nested(a, h)]
            df_a += La - max([1] + nested_levels)
        seen.append(a)
    return df_a - 1 if seen else 0


def run(y, X, fe_ids, cluster_ids, tol=1e-8, maxiter=100_000):
    n0 = len(y)
    X = np.asarray(X, dtype=np.float64).reshape(n0, -1)
    fe = np.asarray(fe_ids).reshape(n0, -1)
    k, G = X.shape[1], fe.shape[1]
    cols = {"y": np.asarray(y, dtype=np.float64)}
    cols.update({f"x{j}": X[:, j] for j in range(k)})
    cols.update({f"fe{g}": fe[:, g] for g in range(G)})
    cols["cl"] = np.asarray(cluster_ids)
    data = pd.DataFrame(cols)
    fml = "y ~ " + " + ".join(f"x{j}" for j in range(k)) + " | " + " + ".join(f"fe{g}" for g in range(G))
    m = pf.feols(fml, data, vcov={"CRV1": "cl"}, fixef_rm="singleton", ssc=pf.ssc(k_adj=False, G_adj=False),
                 demeaner=DEMEANER, lean=True, store_data=False, copy_data=False)
    beta = np.asarray(m.coef().values, dtype=np.float64)
    vcov = np.asarray(m._vcov, dtype=np.float64)
    keep = np.ones(n0, bool)
    na = np.fromiter(sorted(m._na_index or ()), dtype=np.int64)
    keep[na] = False
    n = int(keep.sum())
    codes = np.column_stack([_codes(fe[keep, g]) for g in range(G)])
    cl = _codes(np.asarray(cluster_ids)[keep])
    Gc = int(cl.max()) + 1
    df_a = _absorbed_df(codes, cl)
    q = (n - 1) / (n - (k + 1 + df_a)) * Gc / (Gc - 1)
    return dict(beta=beta, vcov=q * vcov, nobs=n, df_a=int(df_a), n_clusters=Gc)
