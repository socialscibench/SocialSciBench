"""Deterministic synthetic workloads for task 009_melogit.

Two-level random-intercept logit: observations i in groups j (employees in firms, firm-years in
firms, individuals in countries, patents in firms, job postings in firms),

    P(y_ij = 1 | u_j) = logistic(x_ij'b + u_j),   u_j ~ N(0, sigma_u^2).

Some regressors vary within groups and some only between groups (e.g. firm size, country GDP).
Group sizes are unbalanced.
"""
import numpy as np

# Numbered 1-6 from small to large. Each workload has a fixed seed.
WORKLOADS = [
    dict(id=1, seed=9001, design="sanity check", n_groups=100, n_obs=5_000, k=5, k_between=1),
    dict(id=2, seed=9002, design="employees in firms (survey)", n_groups=1_000, n_obs=50_000, k=10, k_between=3),
    dict(id=3, seed=9003, design="firm-years in firms (panel)", n_groups=20_000, n_obs=200_000, k=15,
         k_between=3),
    dict(id=4, seed=9004, design="individuals in countries (cross-national survey)", n_groups=60, n_obs=500_000,
         k=10, k_between=3),
    dict(id=5, seed=9005, design="patents in firms", n_groups=10_000, n_obs=1_000_000, k=15, k_between=3),
    dict(id=6, seed=9006, design="job postings in firms", n_groups=100_000, n_obs=2_000_000, k=20, k_between=4),
]
SIGMA_U = 1.0


def generate(w):
    rng = np.random.default_rng(w["seed"])
    G, N, k, kb = w["n_groups"], w["n_obs"], w["k"], w["k_between"]
    size = rng.lognormal(0, 0.7, G)
    g = np.sort(rng.choice(G, N, p=size / size.sum()))
    g = np.unique(g, return_inverse=True)[1]                    # drop empty groups, ids 0..G'-1
    G = g.max() + 1
    Xw = rng.standard_normal((N, k - kb))
    Xb = rng.standard_normal((G, kb))[g]
    X = np.column_stack([Xw, Xb])
    X[:, 0] = (X[:, 0] > 0.5).astype(float)                     # one binary regressor (e.g. treatment)
    beta = np.linspace(0.5, -0.5, k) * 0.6
    u = rng.normal(0, SIGMA_U, G)
    eta = -1.2 + X @ beta + u[g]
    y = (rng.random(N) < 1 / (1 + np.exp(-eta))).astype(np.int8)
    return dict(y=y, X=X, group=g)


def axes(w, d):
    sizes = np.bincount(d["group"])
    return dict(n_obs=len(d["y"]), n_groups=len(sizes), k=d["X"].shape[1], k_between=w["k_between"],
                mean_group_size=round(float(sizes.mean()), 1), max_group_size=int(sizes.max()),
                share_y1=round(float(d["y"].mean()), 3))
