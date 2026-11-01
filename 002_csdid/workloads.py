"""Deterministic synthetic workloads for task 002 (staggered DiD, csdid).

Balanced panels with staggered adoption. Treatment timing depends on the covariates (so the
propensity score matters) and the covariates have time-varying effects on the outcome (so the
outcome regression matters). The six workloads are the former task 002's synthetic set, with the
task's default options (never-treated controls, dripw, varying base period, no trimming, analytic
standard errors, csdid 1.81 rules), so they generate exactly the former inputs.

The evaluator's development fixtures also draw the task's options here (only when a workload
names them, from a separate random stream, so the six workloads are unchanged):
  control, method, base_period, trim, rules, event_sets, alpha   passed to run()
  weighted     positive sampling weights, log-normal and correlated with X
  missing      about this share of unit-periods unobserved (NaN in y): half of the units exit after
               a random period and observed periods have random gaps; every unit is observed in
               periods 1 and 2
  varying      covariates that vary over time (X [n_units, n_periods, k]): the fixed covariates
               plus a unit-specific trend and noise
  boot         number of bootstrap draws (Rademacher multipliers, int8 +1/-1)
"""
import numpy as np

# Workload set v2: recognizable research settings, numbered 1-6 from small to large. Each workload
# has a fixed seed (kept from its original v2 numbering, so the data match the published timings).
# Optional keys: share_treated (share of units ever treated; default ~70%) and first_cohort
# (earliest adoption period; default 3).
WORKLOADS = [
    dict(id=1, seed=2001, design="sanity check", n_units=1_000, n_periods=10, n_cohorts=4, n_covariates=2),
    dict(id=2, seed=2010, design="state/county policy adoption", n_units=3_000, n_periods=20, n_cohorts=10,
         n_covariates=3, share_treated=0.50),
    dict(id=3, seed=2007, design="firm-year study", n_units=5_000, n_periods=30, n_cohorts=20, n_covariates=5,
         share_treated=0.25),
    dict(id=4, seed=2006, design="private firms / establishments, annual", n_units=50_000, n_periods=20, n_cohorts=10,
         n_covariates=10),
    dict(id=5, seed=2008, design="monthly platform rollout", n_units=100_000, n_periods=60, n_cohorts=40, n_covariates=5),
    dict(id=6, seed=2009, design="GenAI adoption, firm-quarter", n_units=285_000, n_periods=42, n_cohorts=10,
         n_covariates=5, share_treated=0.10, first_cohort=33),
]
RUN_OPTIONS = ("control", "method", "base_period", "trim", "rules", "event_sets", "alpha")
DATA_OPTIONS = ("weighted", "missing", "varying", "boot")


def _intercept(index, share):
    """Logit intercept a such that mean(expit(a + index)) equals share (bisection)."""
    lo, hi = -20.0, 20.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if (1.0 / (1.0 + np.exp(-(mid + index)))).mean() < share:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def generate(w):
    rng = np.random.default_rng(w["seed"])
    N, T, C, k = w["n_units"], w["n_periods"], w["n_cohorts"], w["n_covariates"]
    X = rng.standard_normal((N, k))

    # Adoption: P(ever treated) rises with X (about 70% treated unless share_treated is given);
    # cohorts spread evenly over periods first_cohort..T
    index = 0.5 * X[:, 0] - 0.3 * X[:, 1 % k]
    a = _intercept(index, w["share_treated"]) if "share_treated" in w else 0.85
    p_treat = 1.0 / (1.0 + np.exp(-(a + index)))
    cohorts = np.unique(np.linspace(w.get("first_cohort", 3), T, C).round().astype(int))
    gvar = np.where(rng.random(N) < p_treat, rng.choice(cohorts, N), 0)

    # Outcome: unit FE + period FE + time-varying covariate effects + dynamic treatment effect
    t = np.arange(1, T + 1)
    gamma = rng.standard_normal((k, T)) * 0.1 * t / T
    y = (rng.standard_normal(N)[:, None] + 0.1 * t[None, :] + X @ gamma
         + np.where((gvar[:, None] > 0) & (t[None, :] >= gvar[:, None]), 1.0 + 0.2 * (t[None, :] - gvar[:, None]), 0.0)
         + rng.standard_normal((N, T)))
    out = dict(y=y, X=X, gvar=gvar.astype(np.int64))
    if not any(key in w for key in RUN_OPTIONS + DATA_OPTIONS):
        return out                                   # the six workloads: the former task's inputs

    opt = np.random.default_rng(w["seed"] + 104729)  # options draw from their own stream
    if w.get("missing"):
        observed = opt.random((N, T)) >= w["missing"] / 2                    # gaps
        last = np.where(opt.random(N) < 0.5, opt.integers(2, T + 1, N), T)
        observed &= t[None, :] <= last[:, None]                               # exit after the last period
        observed[:, :2] = True
        out["y"] = np.where(observed, y, np.nan)
    if w.get("varying"):
        trend = opt.standard_normal((N, k)) * 0.05
        out["X"] = X[:, None, :] + trend[:, None, :] * t[None, :, None] + 0.1 * opt.standard_normal((N, T, k))
    out["w"] = np.exp(0.3 * X[:, 0] + 0.5 * opt.standard_normal(N)) if w.get("weighted") else None
    out["boot_weights"] = ((opt.integers(0, 2, size=(N, w["boot"]), dtype=np.int8) * 2 - 1).astype(np.int8)
                           if w.get("boot") else None)
    defaults = dict(control="never", method="dripw", base_period="varying", trim=1.0, rules="csdid181",
                    event_sets=None, alpha=0.05)
    out.update({key: w.get(key, value) for key, value in defaults.items()}, event_window=(-5, 5))
    return out


def axes(w, d):
    """Workload axes as reported in the spec table."""
    gvar = d["gvar"]
    cohorts = np.unique(gvar[gvar > 0])
    T = d["y"].shape[1]
    return dict(n_units=d["y"].shape[0], n_periods=T, n_cohorts=int(cohorts.size),
                n_covariates=d["X"].shape[-1], n_gt=int(cohorts.size * (T - 1)),
                n_obs=int(d["y"].size), share_treated=round(float((gvar > 0).mean()), 3))
