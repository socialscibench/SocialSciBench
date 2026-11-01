"""Synthetic workload 109 proposed in patentsview_workload_design.md.

109 SEC-shaped smoke test: directors and officers x firm x year, with board interlocks.
    Contains no SEC data. Its parameters were fixed before any SEC file was read, from
    general published patterns (boards of roughly 9 directors plus a few officers, about
    10% annual turnover, a minority of directors on several boards, insider purchases in a
    minority of insider-years). It is for checking that a candidate runs on governance-
    shaped inputs (binary outcome, small panels, interlock dyads); do not tune on it.

Returns task-001 inputs: dict(y, X, fe_ids, cluster_ids).
"""
import numpy as np

# Workload 109 parameters (fixed in advance; see module docstring).
SEC_SHAPE = dict(n_firms=5_000, n_years=10, seats_mean=11, turnover=0.10,
                 reuse_existing=0.20, purchase_rate=0.12, n_regressors=12)


def _effects_xy(rng, fe_ids, k, binary=False, rate=0.5):
    """Same construction as workloads.py: X loads on the fixed effects; y = X beta + effects + noise."""
    n = fe_ids.shape[0]
    eff = sum(rng.standard_normal(fe_ids[:, g].max() + 1)[fe_ids[:, g]] for g in range(fe_ids.shape[1]))
    X = np.column_stack([0.5 * eff + rng.standard_normal(n) for _ in range(k)])
    beta = np.arange(1, k + 1) / 10.0
    latent = X @ beta + eff + rng.standard_normal(n)
    if not binary:
        return latent, X
    # Linear probability model outcome: 1 for the top `rate` share of the latent index.
    return (latent > np.quantile(latent, 1 - rate)).astype(np.float64), X


def _simulate_boards(rng, p):
    """Seats per firm; each year a seat is kept, or refilled by an existing insider (interlock) or a new one."""
    F, T = p["n_firms"], p["n_years"]
    seats = np.maximum(3, rng.poisson(p["seats_mean"], F))
    holders = [list(range(s)) for s in seats]  # placeholder ids, replaced below
    next_id = 0
    for f in range(F):
        holders[f] = list(range(next_id, next_id + seats[f]))
        next_id += seats[f]
    rows = []
    for t in range(T):
        pool = np.concatenate([np.array(h) for h in holders])
        for f in range(F):
            for i in range(len(holders[f])):
                if t > 0 and rng.random() < p["turnover"]:
                    if rng.random() < p["reuse_existing"]:
                        holders[f][i] = int(pool[rng.integers(0, pool.size)])
                    else:
                        holders[f][i] = next_id
                        next_id += 1
            for person in set(holders[f]):
                rows.append((person, f, t))
    return np.array(rows, dtype=np.int64)


def workload_109(design="network"):
    """design='network': insider x firm x year, absorb(insider firm year), cluster firm.
    design='dyad': firm pair x year for pairs that ever share an insider,
    absorb(firm_i#year firm_j#year pair), cluster pair; y = 1{pair shares an insider that year}."""
    p = SEC_SHAPE
    rng = np.random.default_rng(1109)
    rows = _simulate_boards(rng, p)
    if design == "network":
        fe = rows.copy()
        y, X = _effects_xy(rng, fe, p["n_regressors"], binary=True, rate=p["purchase_rate"])
        return dict(y=y, X=X, fe_ids=fe, cluster_ids=fe[:, 1].copy())
    # Dyads: pairs of firms sharing a person in a year.
    T = p["n_years"]
    order = np.lexsort((rows[:, 1], rows[:, 2], rows[:, 0]))
    r = rows[order]
    ties = set()
    start = 0
    for end in range(1, len(r) + 1):
        if end == len(r) or r[end, 0] != r[start, 0] or r[end, 2] != r[start, 2]:
            firms = np.unique(r[start:end, 1])
            for a in range(len(firms)):
                for b in range(a + 1, len(firms)):
                    ties.add((int(firms[a]), int(firms[b]), int(r[start, 2])))
            start = end
    pairs = sorted({(i, j) for i, j, _ in ties})
    pair_idx = {pr: k for k, pr in enumerate(pairs)}
    n = len(pairs) * T
    i_ = np.repeat([pr[0] for pr in pairs], T)
    j_ = np.repeat([pr[1] for pr in pairs], T)
    t_ = np.tile(np.arange(T), len(pairs))
    pid = np.repeat(np.arange(len(pairs)), T)
    y = np.zeros(n)
    for i, j, t in ties:
        y[pair_idx[(i, j)] * T + t] = 1.0
    F = p["n_firms"]
    fe = np.column_stack([i_ * T + t_, F * T + j_ * T + t_, pid])
    _, X = _effects_xy(rng, fe, 10)
    return dict(y=y, X=X, fe_ids=fe, cluster_ids=pid.copy())
