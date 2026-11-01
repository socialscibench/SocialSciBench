"""Task 001 competitor: xhdfe 2.28.1.20260929 (C++), adapted as in the 14 October 2026 study.

Follows research/2026-10-14/native-extension/task001/adapters.py::xhdfe_run in the companion
repository: labels are re-encoded (the array API rejects negative IDs), xhdfe fits with
reghdfe-comparable defaults, and its native outputs are returned. xhdfe's reported df_a
includes the absorbed intercept, so it is converted to this task's K = k + 1 + df_a
convention from the model's own fields. The study's redundant small-sample audit is not run.
"""
import os

import numpy as np

# Fresh fits only: disable xhdfe's opt-in disk reuse and mobility profiles, CPU backend.
os.environ.update(XHDFE_ABSORPTION_CACHE_MODE="off", XHDFE_FE_STRUCTURE_MODE="off",
                  XHDFE_MOBILITY_MODE="off", XHDFE_GPU_BACKEND="cpu", XHDFE_CERTIFY="0")
import xhdfe  # noqa: E402


def _recode(ids):
    return np.unique(ids, return_inverse=True)[1].astype(np.int64)


def run(y, X, fe_ids, cluster_ids, tol=1e-8, maxiter=100_000):
    threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
    fe = [_recode(fe_ids[:, j]) for j in range(fe_ids.shape[1])]
    cl = _recode(cluster_ids)
    model = xhdfe.HdfeRegressor(se_type="cluster", fit_intercept=False, num_threads=threads)
    model.fit(np.asarray(y), np.asarray(X), fes=fe, clusters=cl)
    k = X.shape[1]
    K_native = int(model.df_m_) + int(model.df_a_) + int(model.df_a_nested_ > 0)
    return dict(beta=np.asarray(model.coef_, dtype=float).copy(),
                vcov=np.asarray(model.covariance_, dtype=float).copy(),
                nobs=int(model.nobs_), df_a=K_native - k - 1, n_clusters=int(model.num_clusters_))
