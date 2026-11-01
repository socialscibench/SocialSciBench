"""Deterministic synthetic workloads for task 001_reghdfe_ols_cluster.

Each workload mimics a common organization/strategy/economics design. Data are
generated from a fixed seed, so every implementation sees identical inputs.
"""
import numpy as np

# Workload set v2. Ids are stable and never reused: id 2 retired in v1 (duplicate
# of 1 and 3), id 5 retired in v2 (duplicate of 8 except for size).
# The seed for each workload is 1000 + id.
WORKLOADS = [
    # Fast check.
    dict(id=1, design="firm_year", n_units=10_000, n_periods=10, n_regressors=3),
    dict(id=3, design="firm_year", n_units=500_000, n_periods=10, n_regressors=20),
    # Mobility sweep at fixed size: only mobility changes across 4, 8 and 9.
    dict(id=4, design="worker_firm", n_units=100_000, n_periods=10, n_firms=10_000, mobility=0.10, n_regressors=20),
    dict(id=8, design="worker_firm", n_units=100_000, n_periods=10, n_firms=10_000, mobility=0.02, n_regressors=20),
    dict(id=9, design="worker_firm", n_units=100_000, n_periods=10, n_firms=10_000, mobility=0.005, n_regressors=20),
    # Structural gravity: few time-varying regressors, as in practice.
    dict(id=6, design="gravity", n_countries=150, n_periods=20, n_regressors=3),
    dict(id=7, design="four_way", n_units=200_000, n_periods=10, n_industries=300, n_regions=50, n_regressors=20),
    # Many regressors, e.g. an event study with many leads and lags.
    dict(id=10, design="firm_year", n_units=100_000, n_periods=10, n_regressors=50),
    # Long-running cases.
    dict(id=11, design="worker_firm", n_units=1_000_000, n_periods=10, n_firms=100_000, mobility=0.005, n_regressors=20),
    dict(id=12, design="firm_year", n_units=1_000_000, n_periods=10, n_regressors=40),
]


def _outcome(rng, X, effects):
    beta = np.arange(1, X.shape[1] + 1) / 10.0
    return X @ beta + sum(effects) + rng.standard_normal(X.shape[0])


def _regressors(rng, n, k, effects):
    # Regressors correlated with the fixed effects, so omitting them biases OLS.
    base = sum(effects)
    return np.column_stack([0.5 * base + rng.standard_normal(n) for _ in range(k)])


def generate(w):
    rng = np.random.default_rng(w.get("seed", 1000 + w["id"]))
    k = w["n_regressors"]

    if w["design"] == "firm_year":
        # absorb(firm year), cluster(firm)
        N, T = w["n_units"], w["n_periods"]
        firm = np.repeat(np.arange(N), T)
        year = np.tile(np.arange(T), N)
        a, d = rng.standard_normal(N), rng.standard_normal(T)
        eff = [a[firm], d[year]]
        fe, cl = np.column_stack([firm, year]), firm

    elif w["design"] == "worker_firm":
        # AKM-style: absorb(worker firm year), cluster(firm). Low mobility = slow convergence.
        N, T, F = w["n_units"], w["n_periods"], w["n_firms"]
        worker = np.repeat(np.arange(N), T)
        year = np.tile(np.arange(T), N)
        cur = rng.integers(0, F, N)
        firm = np.empty(N * T, dtype=np.int64)
        for t in range(T):
            move = rng.random(N) < w["mobility"]
            cur = np.where(move, rng.integers(0, F, N), cur)
            firm[t::T] = cur
        a, p, d = rng.standard_normal(N), rng.standard_normal(F), rng.standard_normal(T)
        eff = [a[worker], p[firm], d[year]]
        fe, cl = np.column_stack([worker, firm, year]), firm

    elif w["design"] == "gravity":
        # Structural gravity: absorb(exporter#year importer#year exporter#importer), cluster(pair)
        C, T = w["n_countries"], w["n_periods"]
        ex, im = np.meshgrid(np.arange(C), np.arange(C), indexing="ij")
        off = ex != im
        ex, im = ex[off], im[off]
        P = ex.size
        pair = np.repeat(np.arange(P), T)
        exp_ = np.repeat(ex, T)
        imp_ = np.repeat(im, T)
        year = np.tile(np.arange(T), P)
        ext, imt = exp_ * T + year, imp_ * T + year
        eff = [rng.standard_normal(C * T)[ext], rng.standard_normal(C * T)[imt], rng.standard_normal(P)[pair]]
        fe, cl = np.column_stack([ext, imt, pair]), pair

    elif w["design"] == "four_way":
        # absorb(firm year industry#year region#year), cluster(firm)
        N, T = w["n_units"], w["n_periods"]
        firm = np.repeat(np.arange(N), T)
        year = np.tile(np.arange(T), N)
        ind = np.repeat(rng.integers(0, w["n_industries"], N), T)
        reg = np.repeat(rng.integers(0, w["n_regions"], N), T)
        indyr, regyr = ind * T + year, reg * T + year
        eff = [rng.standard_normal(N)[firm], rng.standard_normal(T)[year],
               rng.standard_normal(w["n_industries"] * T)[indyr],
               rng.standard_normal(w["n_regions"] * T)[regyr]]
        fe, cl = np.column_stack([firm, year, indyr, regyr]), firm

    X = _regressors(rng, fe.shape[0], k, eff)
    y = _outcome(rng, X, eff)
    return dict(y=y, X=X, fe_ids=fe.astype(np.int64), cluster_ids=cl.astype(np.int64))


def axes(w, d):
    """Workload axes as reported in the spec table."""
    fe = d["fe_ids"]
    levels = [int(np.unique(fe[:, g]).size) for g in range(fe.shape[1])]
    return dict(n_obs=fe.shape[0], n_regressors=d["X"].shape[1], n_fe=fe.shape[1],
                fe_levels=levels, n_clusters=int(np.unique(d["cluster_ids"]).size))
