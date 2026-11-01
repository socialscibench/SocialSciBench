"""Deterministic synthetic workloads for task 011_logithdfe_ame.

Panels with a binary outcome (e.g. "the firm adopts AI this year"), a binary treatment `d`
and continuous covariates, generated from a logit with several high-dimensional fixed
effects. Covariates and treatment are correlated with the unit effects, so ignoring the
fixed effects biases the estimates. Low event rates mean many units never experience the
event and are dropped, as in real data.
"""
import numpy as np

# Numbered 1-7 from small to large. Each workload has a fixed seed.
# absorb: fixed effects, in order (names refer to the panel's id columns; a#b = interaction).
WORKLOADS = [
    dict(id=1, seed=3101, design="sanity check", panel="firm", n_units=300, n_periods=10,
         absorb=["firm", "year"], cluster="firm", n_covariates=3, event_rate=0.30),
    dict(id=2, seed=3102, design="small multi-way check", panel="firm", n_units=600, n_periods=10,
         n_industries=10, n_states=10, absorb=["firm", "industry#year", "state#year"], cluster="firm",
         n_covariates=3, event_rate=0.30),
    dict(id=3, seed=3103, design="large public firms (S&P 1500)", panel="firm", n_units=1_500, n_periods=15,
         absorb=["firm", "year"], cluster="firm", n_covariates=5, event_rate=0.20),
    dict(id=4, seed=3104, design="public firms + industry shocks (Compustat)", panel="firm", n_units=5_000,
         n_periods=15, n_industries=48, absorb=["firm", "industry#year"], cluster="firm", n_covariates=10,
         event_rate=0.15),
    dict(id=5, seed=3105, design="firms + industry and local shocks", panel="firm", n_units=20_000, n_periods=10,
         n_industries=48, n_states=50, absorb=["firm", "industry#year", "state#year"], cluster="state",
         n_covariates=10, event_rate=0.10),
    dict(id=6, seed=3106, design="workers + employers (résumé data)", panel="worker", n_units=50_000,
         n_periods=10, n_employers=1_000, mobility=0.05, absorb=["worker", "employer", "year"], cluster="employer",
         n_covariates=5, event_rate=0.10),
    dict(id=7, seed=3107, design="large private firms / establishments", panel="firm", n_units=100_000,
         n_periods=10, n_industries=48, absorb=["firm", "industry#year"], cluster="firm", n_covariates=10,
         event_rate=0.10),
]

SD = {"firm": 1.0, "worker": 1.0, "employer": 0.5, "year": 0.3, "industry#year": 0.3, "state#year": 0.3}


def _ids(w, rng):
    """Integer id columns of the panel: unit, year, and industry/state/employer where used."""
    N, T = w["n_units"], w["n_periods"]
    ids = {w["panel"]: np.repeat(np.arange(N), T), "year": np.tile(np.arange(T), N)}
    if "n_industries" in w:
        ids["industry"] = np.repeat(rng.integers(0, w["n_industries"], N), T)
    if "n_states" in w:
        ids["state"] = np.repeat(rng.integers(0, w["n_states"], N), T)
    if "n_employers" in w:
        cur = rng.integers(0, w["n_employers"], N)
        emp = np.empty(N * T, dtype=np.int64)
        for t in range(T):
            move = rng.random(N) < w["mobility"]
            cur = np.where(move, rng.integers(0, w["n_employers"], N), cur)
            emp[t::T] = cur
        ids["employer"] = emp
    return ids


def fe_column(ids, name):
    """Codes for one absorbed fixed effect; `a#b` interactions are encoded as one code."""
    parts = name.split("#")
    code = np.zeros(ids["year"].size, dtype=np.int64)
    for p in parts:
        code = code * (ids[p].max() + 1) + ids[p]
    return np.unique(code, return_inverse=True)[1].astype(np.int64)


def generate(w):
    rng = np.random.default_rng(w["seed"])
    ids = _ids(w, rng)
    n, k = ids["year"].size, w["n_covariates"]
    fe_ids = np.column_stack([fe_column(ids, name) for name in w["absorb"]])
    effects = [SD[name] * rng.standard_normal(fe_ids[:, g].max() + 1)[fe_ids[:, g]]
               for g, name in enumerate(w["absorb"])]
    unit = effects[0]
    d = (rng.random(n) < 1 / (1 + np.exp(-(-0.8 + 0.5 * unit)))).astype(np.float64)   # binary treatment
    X = 0.5 * unit[:, None] + rng.standard_normal((n, k))
    index = 0.4 * d + X @ np.resize([0.5, -0.3, 0.2, 0.4, -0.2], k) + sum(effects)
    lo, hi = -20.0, 20.0                       # intercept so that the event rate matches (bisection)
    for _ in range(100):
        c = (lo + hi) / 2
        lo, hi = (c, hi) if (1 / (1 + np.exp(-(c + index)))).mean() < w["event_rate"] else (lo, c)
    y = (rng.random(n) < 1 / (1 + np.exp(-(c + index)))).astype(np.float64)
    cluster = fe_column(ids, w["cluster"])
    return dict(y=y, d=d, X=X, fe_ids=fe_ids, cluster_ids=cluster)


def axes(w, d):
    """Workload axes as reported in the spec table. fe_levels are counted before dropping."""
    fe = d["fe_ids"]
    return dict(absorb=" ".join(w["absorb"]), cluster=w["cluster"], n_obs=int(d["y"].size),
                n_covariates=1 + d["X"].shape[1], n_fe=fe.shape[1],
                fe_levels=[int(fe[:, g].max() + 1) for g in range(fe.shape[1])],
                n_clusters=int(d["cluster_ids"].max() + 1), mobility=w.get("mobility"),
                event_rate=round(float(d["y"].mean()), 3))
