"""Task 001 baseline: Stata reghdfe 6.14.1 (Correia 2017; ftools 2.50.0, require 1.4.0), in-process
via pystata (Stata 18 MP):

    reghdfe v1 v2-v<k+1>, absorb(<fixed effects>) vce(cluster <cluster>) tolerance(<tol>) iterate(<maxiter>)

with reghdfe's defaults otherwise (alternating projections accelerated by conjugate gradients).
The arrays are written to a temporary .dta file (pandas `to_stata`, format 118) and loaded with `use`
inside the timed call (ids as doubles, exact below 2^53; pystata's `nparray_to_data` was 20x slower);
beta and the covariance are e(b) and e(V) without _cons, nobs = e(N), df_a = e(df_a),
n_clusters = e(N_clust).
Stata MP is limited to the evaluator's thread count (`set processors`). The ado path is Stata's
own plus the downloaded packages (SOCIALSCIBENCH_REGHDFE_ADO); compiled Mata libraries go to a
private temporary PLUS directory. Mata is compiled and warmed up on a small generated problem at
import (data independent).
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd

STATA_DIR = os.environ.get("SOCIALSCIBENCH_STATA", "/Applications/Stata 18")
def _venv():
    """The baseline environments: SOCIALSCIBENCH_BASELINE_VENV (set by compare.py), else
    solutions/baselines/.venv of the repository above this file (also from an evaluator run's snapshot)."""
    if os.environ.get("SOCIALSCIBENCH_BASELINE_VENV"):
        return os.environ["SOCIALSCIBENCH_BASELINE_VENV"]
    d = os.path.dirname(os.path.abspath(__file__))
    while not os.path.isdir(os.path.join(d, "solutions", "baselines", ".venv")):
        if os.path.dirname(d) == d:
            raise RuntimeError("solutions/baselines/.venv not found above " + os.path.abspath(__file__))
        d = os.path.dirname(d)
    return os.path.join(d, "solutions", "baselines", ".venv")


ADO = os.environ.get("SOCIALSCIBENCH_REGHDFE_ADO", os.path.join(_venv(), "stata-ado", "hdfe"))
TMP = tempfile.mkdtemp(prefix="ssb001-stata-")

sys.path.append(os.path.join(STATA_DIR, "utilities"))
from pystata import config  # noqa: E402

config.init("mp", splash=False)
from pystata import stata  # noqa: E402
from sfi import Matrix, Scalar  # noqa: E402

THREADS = int(os.environ.get("OMP_NUM_THREADS", "1"))
for line in ("set more off", f"set processors {THREADS}", "set max_memory 28g",
             f'sysdir set PERSONAL "{TMP}"', f'sysdir set PLUS "{TMP}"', f'sysdir set OLDPLACE "{TMP}"',
             f'adopath ++ "{os.path.abspath(ADO)}"', "which reghdfe", "which ftools", "ftools, compile"):
    stata.run(line, quietly=True)


def _load(cols):
    A = np.column_stack(cols)
    path = os.path.join(TMP, "data.dta")
    pd.DataFrame(A, columns=[f"v{i + 1}" for i in range(A.shape[1])]).to_stata(path, write_index=False, version=118)
    stata.run(f'use "{path}", clear', quietly=True)


def _fit(n, k, G, tol, maxiter):
    xs = f"v2-v{k + 1}" if k > 1 else "v2"
    fes = " ".join(f"v{k + 2 + g}" for g in range(G))
    stata.run(f"reghdfe v1 {xs}, absorb({fes}) vce(cluster v{k + 2 + G}) tolerance({tol!r}) "
              f"iterate({int(maxiter)})", quietly=True)


# warm-up on a small generated problem (compiles the Mata code paths)
_rng = np.random.default_rng(0)
_n = 2000
_fe = np.column_stack([_rng.integers(0, 50, _n), _rng.integers(0, 20, _n)]).astype(np.float64)
_load([_rng.standard_normal(_n), _rng.standard_normal((_n, 2)), _fe, _fe[:, 0]])
_fit(_n, 2, 2, 1e-8, 1000)


def run(y, X, fe_ids, cluster_ids, tol=1e-8, maxiter=100_000):
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    X = np.asarray(X, dtype=np.float64).reshape(n, -1)
    F = np.asarray(fe_ids).reshape(n, -1)
    k, G = X.shape[1], F.shape[1]
    if max(np.abs(F).max(), np.abs(np.asarray(cluster_ids)).max()) >= 2 ** 53:
        raise ValueError("ids must be exact as doubles")
    _load([y, X, F.astype(np.float64), np.asarray(cluster_ids, dtype=np.float64)])
    _fit(n, k, G, tol, maxiter)
    names = Matrix.getColNames("e(b)")
    keep = [i for i, nm in enumerate(names) if nm != "_cons"]
    b = np.asarray(Matrix.get("e(b)"), dtype=np.float64)[0, keep]
    V = np.asarray(Matrix.get("e(V)"), dtype=np.float64)[np.ix_(keep, keep)]
    out = dict(beta=b, vcov=V, nobs=int(Scalar.getValue("e(N)")), df_a=int(Scalar.getValue("e(df_a)")),
               n_clusters=int(Scalar.getValue("e(N_clust)")))
    stata.run("clear", quietly=True)
    return out
