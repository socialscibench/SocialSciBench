"""Deterministic synthetic workloads for task 006_matching.

Cross-sections with a binary treatment that depends on continuous covariates (selection on
observables), an outcome that depends on the same covariates, and a constant treatment
effect of 1.0. Propensity-score matching uses all covariates; coarsened exact matching uses
the first four, as applied work typically coarsens a few key covariates.
"""
import numpy as np

# Numbered 1-5 from small to large. Each workload has a fixed seed.
WORKLOADS = [
    dict(id=1, seed=6001, design="sanity check", n=1_000, treated_share=0.20, n_covariates=3),
    dict(id=2, seed=6002, design="firms (e.g. VC-backed vs not)", n=20_000, treated_share=0.10, n_covariates=6),
    dict(id=3, seed=6003, design="startups (Crunchbase-scale)", n=100_000, treated_share=0.10, n_covariates=8),
    dict(id=4, seed=6004, design="workers or inventors", n=500_000, treated_share=0.05, n_covariates=6),
    dict(id=5, seed=6005, design="administrative records", n=2_000_000, treated_share=0.05, n_covariates=5),
]
CEM_COVARIATES = 4


def generate(w):
    rng = np.random.default_rng(w["seed"])
    n, k = w["n"], w["n_covariates"]
    X = rng.standard_normal((n, k))
    X[:, 1::2] = np.exp(0.5 * X[:, 1::2])            # some skewed covariates (size-like)
    gamma = np.resize([0.6, -0.4, 0.3, 0.5, -0.3], k)
    index = (X - X.mean(0)) @ gamma
    lo, hi = -20.0, 20.0                               # intercept so the treated share matches (bisection)
    for _ in range(100):
        c = (lo + hi) / 2
        lo, hi = (c, hi) if (1 / (1 + np.exp(-(c + index)))).mean() < w["treated_share"] else (lo, c)
    d = (rng.random(n) < 1 / (1 + np.exp(-(c + index)))).astype(np.float64)
    y = 1.0 * d + X @ np.resize([1.0, 0.5, -0.5, 0.8, 0.3], k) + rng.standard_normal(n)
    return dict(y=y, d=d, X=X)


def axes(w, d):
    return dict(n=w["n"], n_treated=int(d["d"].sum()), n_covariates=w["n_covariates"], cem_covariates=CEM_COVARIATES)
