"""Deterministic synthetic workloads for task 004_winsorize.

Panels of heavy-tailed variables (Student t with 3 degrees of freedom, like accounting
ratios and stock returns) with about 5% missing values, winsorized at the 1st and 99th
percentiles, either pooled or within groups (by year, industry-year, month or day).
"""
import numpy as np

# Numbered 1-6 from small to large. Each workload has a fixed seed.
# group: None = pooled; otherwise the by() variable and its number of levels.
WORKLOADS = [
    dict(id=1, seed=4001, design="sanity check", n_obs=1_000, n_vars=3, group=None, n_groups=1),
    dict(id=2, seed=4002, design="firm-year, pooled (Compustat)", n_obs=150_000, n_vars=20, group=None, n_groups=1),
    dict(id=3, seed=4003, design="firm-year, by year (Compustat)", n_obs=200_000, n_vars=20, group="year",
         n_groups=30),
    dict(id=4, seed=4004, design="firm-year, by industry-year", n_obs=1_000_000, n_vars=10, group="industry#year",
         n_groups=48 * 30),
    dict(id=5, seed=4005, design="stock-month, by month (CRSP monthly)", n_obs=3_000_000, n_vars=5, group="month",
         n_groups=360),
    dict(id=6, seed=4006, design="stock-day, by day (CRSP daily)", n_obs=20_000_000, n_vars=3, group="day",
         n_groups=2_500),
]
CUTS = (1, 99)          # percentiles, as in winsor2's cuts(1 99)
MISSING_SHARE = 0.05


def generate(w):
    rng = np.random.default_rng(w["seed"])
    n, k = w["n_obs"], w["n_vars"]
    X = rng.standard_t(3, size=(n, k)) * rng.uniform(0.5, 5.0, size=k) + rng.normal(0, 1, size=k)
    X[rng.random((n, k)) < MISSING_SHARE] = np.nan
    group = None if w["group"] is None else np.sort(rng.integers(0, w["n_groups"], n)).astype(np.int64)
    return dict(X=X, group=group, cuts=CUTS)


def axes(w, d):
    g = d["group"]
    return dict(n_obs=w["n_obs"], n_vars=w["n_vars"], by=w["group"] or "–",
                n_groups=1 if g is None else int(np.unique(g).size),
                missing_share=round(float(np.isnan(d["X"]).mean()), 3), cuts=list(CUTS))
