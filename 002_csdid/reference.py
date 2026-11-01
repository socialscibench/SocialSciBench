"""Reference implementation for task 002 (combined: the former tasks 002, 002b and 002c).

Callaway & Sant'Anna (2021) staggered difference-in-differences as Stata's `csdid` computes it,
under either csdid 1.81's rules (the version on SSC) or csdid 2.0.1's (October 2026). With the
options at their defaults it reproduces the original task-002 reference (never-treated controls,
dripw, balanced panel, short gaps, analytic SEs) and the task-002b reference exactly. Options:

  control, method, w, unbalanced y, aggregations: as in task 002b (see task.md)

  base_period   "varying" (short gaps, the default) or "universal" (long gaps: every cell
                compares period t with g - 1; pre-treatment cells run from period 1 and the
                reference cell t = g - 1 is not estimated; estimates are y_t - y_(g-1), the
                sign of csdid 1.81's `long2`, csdid2's `long` and csdid 2.0.1's
                base_period(universal))
  trim          propensity-score trimming: comparison units scoring above `trim` get weight 0
                (1.0 = none, csdid 1.81's and csdid2's behaviour; csdid 2.0.1's default is .995)
  boot_weights  [n_units, B] multipliers (+1/-1, Rademacher) for the multiplier bootstrap.
                Returns bootstrap standard errors (interquartile range / 1.349) for every
                estimate and the critical values of the simultaneous (uniform) bands. The
                weights are an input, so the result is deterministic. Order statistics follow
                the chosen rules: 2.0.1 (R did 2.5.1) uses index ceil(p B) and floors the
                aggregate tables' critical value at the pointwise one; 1.81 uses ceil(p (B+1))
                and no floor.
  event_sets    event-time subsets, each averaged with equal weights (csdid2's
                `estat event, revent()`); the event-study covariance matrix is always returned
  rules         "csdid181" (default) or "csdid201": the four places where csdid 2.0.1, run
                with Version 1.82's settings, differs from csdid 1.81:
                  - not-yet-treated comparison units: 1.81 uses units first treated after
                    max(g, t); 2.0.1 uses units untreated in both periods of the cell
                    (they differ only in short-gap pre-treatment cells)
                  - cohort weights on an unbalanced panel: 1.81 counts the cohort's units
                    observed in the cell; 2.0.1 counts all of the cohort's units
                  - the cohort average's standard error: 1.81 holds the cohort shares fixed;
                    2.0.1 (and csdid2) include the estimation of the shares (R did)
                  - overlap: for the propensity-score methods, 2.0.1 (like R did) leaves a cell
                    unestimated when an unweighted logit of the cohort indicator gives some unit
                    a score of 0.999 or more; aggregates then use the estimated cells (2.0.1's
                    `estat ..., dropmissing`). 1.81 estimates every cell.

Covariates may be fixed ([n_units, k]) or vary over time ([n_units, n_periods, k]). When they
vary, each 2x2 cell uses the values in the cell's earlier period, as csdid 1.81 and drdid do
(their propensity-score and outcome regressions run on the earlier period's rows), and a
unit-period with a missing covariate counts as not observed.

Everything else is common to both: control "never"/"notyet"; methods dripw, reg,
ipw (csdid 1.81's unnormalized), stdipw, drimp; one weight per unit; NaN for unobserved
unit-periods with pair balancing; simple, cohort, calendar and windowed event-study
aggregates; analytic standard errors from the influence functions.

Deliberately simple: one small estimation problem per cell, plain NumPy.
"""
import numpy as np
from statistics import NormalDist

CAP = 1 - 1e-6        # drdid caps the propensity score below 1
METHODS = ("dripw", "reg", "ipw", "stdipw", "drimp")
RULES = ("csdid181", "csdid201")
IQR_NORM = NormalDist().inv_cdf(0.75) - NormalDist().inv_cdf(0.25)


def expit(v):
    return 0.5 * (1.0 + np.tanh(0.5 * v))


def _solve(A, b):
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:          # collinear covariates: Stata's invsym gives a generalized inverse
        return np.linalg.pinv(A) @ b


def _inv(A):
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A)


def logit(D, X, w, tol=1e-10, maxiter=100):
    """Weighted logit MLE (Stata `logit ... [iw = w]`). Returns the index X b and e(V)."""
    b = np.zeros(X.shape[1])
    for _ in range(maxiter):
        p = expit(X @ b)
        H = (X * (w * p * (1 - p))[:, None]).T @ X
        step = _solve(H, X.T @ (w * (D - p)))
        b += step
        if np.max(np.abs(step)) < tol:
            break
    else:
        raise RuntimeError("logit did not converge")
    p = expit(X @ b)
    return X @ b, _inv((X * (w * p * (1 - p))[:, None]).T @ X)


def ipt(D, X, w, tol=1e-10, maxiter=200):
    """Inverse probability tilting (drdid's `mlexp` for drimp): maximize
    sum w [D x'b - (1 - D) exp(x'b)]. Returns the index X b."""
    b = np.zeros(X.shape[1])
    b[-1] = np.log(np.sum(w * D) / np.sum(w * (1 - D)))     # constant-only solution
    obj = lambda v: np.sum(w * (D * (X @ v) - (1 - D) * np.exp(X @ v)))
    for _ in range(maxiter):
        e = np.exp(X @ b)
        g = X.T @ (w * (D - (1 - D) * e))
        H = (X * (w * (1 - D) * e)[:, None]).T @ X
        step = _solve(H, g)
        f0, s = obj(b), 1.0
        with np.errstate(over="ignore", invalid="ignore"):
            while not obj(b + s * step) >= f0 - 1e-12 * abs(f0) and s > 1e-8:   # step halving; overflow counts as worse
                s /= 2
        b += s * step
        if np.max(np.abs(s * step)) < tol:
            break
    else:
        raise RuntimeError("inverse probability tilting did not converge")
    return X @ b


def wls(dy, X, sw):
    """Weighted least squares coefficients and (X' diag(sw) X)^-1."""
    XtWX = (X * sw[:, None]).T @ X
    inv = _inv(XtWX)
    return inv @ (X.T @ (sw * dy)), inv


def cell_rif(dy, X, D, w, method, trim=1.0):
    """One 2x2 cell (panel data). Returns the per-unit recentered influence function; its mean is the ATT.

    dy: outcome change between the cell's two periods; X: covariates with a constant (last column);
    D: 1 for the cohort, 0 for comparison units; w: sampling weights normalized to mean 1 in the cell.
    The formulas are those of drdid v1.91's Mata functions drdid_panel, reg_panel, ipw_abadie_panel,
    std_ipw_panel and drdid_imp_panel. Comparison units with a score above `trim` get weight 0.
    """
    n = dy.size
    w1 = w * D
    if method == "reg":
        b, inv = wls(dy[D == 0], X[D == 0], w[D == 0])
        xb = X @ b
        eta_t, eta_c = np.mean(w1 * dy) / w1.mean(), np.mean(w1 * xb) / w1.mean()
        lin_ols = ((w * (1 - D) * (dy - xb))[:, None] * X) @ (inv * n)
        inf_t = (w1 * dy - w1 * eta_t) / w1.mean()
        inf_c = (w1 * xb - w1 * eta_c + lin_ols @ (w1[:, None] * X).mean(0)) / w1.mean()
        return (eta_t - eta_c) + inf_t - inf_c

    if method == "drimp":
        index = ipt(D, X, w)
    else:
        index, V = logit(D, X, w)
    ps_raw = expit(index)
    ps = np.minimum(ps_raw, CAP)
    keep = ~((D == 0) & (ps_raw > trim))                    # trimmed comparison units (none at trim = 1)

    if method == "drimp":
        w0r = (1 - D) * ps / (1 - ps) * w                   # WLS weights (not trimmed)
        b, _ = wls(dy[D == 0], X[D == 0], w0r[D == 0])
        resid = dy - X @ b
        summand = keep * (1 - (1 - D) / (1 - ps)) * resid
        att = np.mean(w * summand) / w1.mean()
        return att + w * keep * (summand - D * att) / w1.mean()

    w0 = w * ps * (1 - D) / (1 - ps) * keep
    lin_ps = ((w * (D - ps))[:, None] * X) @ (V * n)

    if method == "ipw":                                     # Abadie (2005), weights not normalized
        att = np.mean((w1 - w0) * dy) / w1.mean()
        lin2 = lin_ps @ ((w0 * dy)[:, None] * X).mean(0)
        return att + (w1 * dy - w0 * dy - lin2 - w1 * att) / w1.mean()

    if method == "stdipw":                                  # normalized (Hajek) IPW
        eta_t, eta_c = np.mean(w1 * dy) / w1.mean(), np.mean(w0 * dy) / w0.mean()
        inf_t = (w1 * dy - w1 * eta_t) / w1.mean()
        inf_c = (w0 * dy - w0 * eta_c + lin_ps @ ((w0 * (dy - eta_c))[:, None] * X).mean(0)) / w0.mean()
        return (eta_t - eta_c) + inf_t - inf_c

    if method == "dripw":                                   # Sant'Anna & Zhao (2020) doubly robust
        wo = w * (1 - D)
        b, inv = wls(dy[D == 0], X[D == 0], w[D == 0])
        r = dy - X @ b
        eta_t, eta_c = np.mean(w1 * r) / w1.mean(), np.mean(w0 * r) / w0.mean()
        lin_ols = ((wo * r)[:, None] * X) @ (inv * n)
        inf_t = (w1 * r - w1 * eta_t - lin_ols @ (w1[:, None] * X).mean(0)) / w1.mean()
        inf_c = (w0 * r - w0 * eta_c + lin_ps @ ((w0 * (r - eta_c))[:, None] * X).mean(0)
                 - lin_ols @ (w0[:, None] * X).mean(0)) / w0.mean()
        return (eta_t - eta_c) + inf_t - inf_c

    raise ValueError(f"method must be one of {METHODS}")


def aggte(rif, wgt):
    """Weighted average of ATT columns, with the influence function of the estimated weights (csdid's aggte)."""
    m_att, m_w = rif.mean(0), wgt.mean(0)
    att = (m_att * m_w).sum() / m_w.sum()
    out = ((m_w / m_w.sum()) * (rif - m_att)).sum(1)
    out += ((m_att / m_w.sum()) * (wgt - m_w)).sum(1)
    out -= ((wgt - m_w) * (att / m_w.sum())).sum(1)
    return out + att


def se(rif):
    return np.sqrt(((rif - rif.mean(0)) ** 2).sum(0)) / rif.shape[0]


def sample(y, gvar, control):
    """csdid's estimation sample. Returns (units kept, observed mask [kept units, T], T).

    Units treated in or before the first period are dropped (always treated). Periods after
    the last one in which a never-treated unit is observed are not used; with not-yet-treated
    controls, periods after both that and the last cohort are not used. Every unit still
    observed in some period is kept and counts in N, the denominator of the standard errors.
    """
    obs = np.isfinite(y) & ((gvar == 0) | (gvar > 1))[:, None]
    seen = np.flatnonzero(obs[gvar == 0].any(0))
    cut = seen.max() + 1 if seen.size else 0
    if control == "notyet":
        cut = max(cut, int(gvar[obs.any(1)].max()))
    obs[:, cut:] = False
    units = obs.any(1)
    return units, obs[units], cut


def cells(obs, gvar, control, base_period="varying", rules="csdid181"):
    """The 2x2 cells in csdid's order: yields (g, t, base period, unit mask). Periods are 1-based.
    Cells without both cohort and comparison units are skipped, as Stata does."""
    T = obs.shape[1]
    for g in np.unique(gvar[gvar > 0]):
        if base_period == "universal":
            base = g - 1
            periods = [t for t in range(1, T + 1) if t != base]
        else:
            base = 1
            periods = range(2, T + 1)
        for t in periods:
            if control == "never":
                group = (gvar == 0) | (gvar == g)
            elif rules == "csdid201" and base_period == "varying":
                group = (gvar == 0) | (gvar == g) | (gvar > t)          # untreated in both periods
            else:
                group = (gvar == 0) | (gvar == g) | (gvar > max(g, t))
            s = group & obs[:, base - 1] & obs[:, t - 1]               # pair balanced
            n1 = np.count_nonzero(s & (gvar == g))
            if 0 < n1 < np.count_nonzero(s):
                yield g, t, base, s
            if base_period == "varying" and t < g:
                base = t                                                # pre-periods: compare t with t-1


def overlap_fails(D, X, cut=0.999):
    """csdid 2.0.1's (and R did's) overlap check: an UNWEIGHTED logit of the cohort indicator on
    the covariates; the cell is not estimated if any fitted score reaches `cut`, or if the fit fails."""
    if X.shape[1] == 1:
        return D.mean() >= cut
    try:
        index, _ = logit(D, X, np.ones(D.size))
    except RuntimeError:
        return True
    return expit(index).max() >= cut


def bootstrap(rif, v, block=64):
    """Multiplier-bootstrap draws: column k of draw r is mean_i (psi_ik - mean psi_k) * v_ir.
    Returns [B, K]. Processed in blocks of draws to bound memory."""
    C = rif - rif.mean(0)
    N, B = v.shape
    out = np.empty((B, C.shape[1]))
    for s in range(0, B, block):
        out[s:s + block] = (v[:, s:s + block].T.astype(np.float64) @ C) / N
    return out


def order_stat(x_sorted, p, rules):
    """The order statistic each version uses for quantile p: ceil(p B) (2.0.1, R type 1) or
    ceil(p (B + 1)) (1.81's iqrse and qtp), clamped to 1..B."""
    B = x_sorted.shape[0]
    k = np.ceil(p * B) if rules == "csdid201" else np.ceil(p * (B + 1))
    return x_sorted[min(max(int(k), 1), B) - 1]


def boot_stats(draws, alpha, floor, rules="csdid201"):
    """Bootstrap SEs (interquartile range / 1.349) and the simultaneous critical value (quantile
    1 - alpha of max |draw / SE| over the table). Under 2.0.1's rules the aggregation tables floor
    the critical value at the pointwise one (R compute.aggte) and the ATT(g,t) table does not
    (R att_gt); 1.81 never floors."""
    srt = np.sort(draws, 0)
    se_b = (order_stat(srt, 0.75, rules) - order_stat(srt, 0.25, rules)) / IQR_NORM
    point = NormalDist().inv_cdf(1 - alpha / 2)
    tmax = np.max(np.abs(draws / se_b), 1)
    crit = float(order_stat(np.sort(tmax[np.isfinite(tmax)]), 1 - alpha, rules))
    if rules == "csdid201" and floor and not crit >= point:
        crit = point
    return se_b, crit


def run(y, X, gvar, w=None, control="never", method="dripw", event_window=(-5, 5),
        base_period="varying", trim=1.0, boot_weights=None, alpha=0.05, event_sets=None, rules="csdid181"):
    """
    y      [n_units, n_periods]     float64  outcome, periods 1..n_periods; NaN = unit-period not observed
    X      [n_units, n_covariates] or [n_units, n_periods, n_covariates]  float64  covariates,
           fixed or varying over time (a cell uses its earlier period's values)
    gvar   [n_units]                int64    first treated period (0 = never treated)
    w      [n_units] or None        float64  sampling weight per unit (None = unweighted)
    boot_weights [n_units, B] or None  int8  bootstrap multipliers (+1/-1)
    event_sets   list of sequences of event times, or None
    other options: see the module docstring
    returns dict with ATT(g,t) and the simple, group, calendar and event-study aggregates, with
    analytic SEs; the event-study covariance; event-set averages; and, with boot_weights,
    bootstrap SEs and simultaneous critical values
    """
    if control not in ("never", "notyet"):
        raise ValueError("control must be 'never' or 'notyet'")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if base_period not in ("varying", "universal"):
        raise ValueError("base_period must be 'varying' or 'universal'")
    if rules not in RULES:
        raise ValueError(f"rules must be one of {RULES}")
    y = np.asarray(y, dtype=np.float64)
    gvar = np.asarray(gvar, dtype=np.int64)
    w = np.ones(y.shape[0]) if w is None else np.asarray(w, dtype=np.float64)

    # 1. Sample (a unit-period with a missing time-varying covariate counts as not observed)
    X = np.asarray(X, dtype=np.float64)
    varying = X.ndim == 3
    if varying:
        y = np.where(np.isfinite(X).all(2), y, np.nan)
    units, obs, T = sample(y, gvar, control)
    y, gvar, w = y[units], gvar[units], w[units]
    X = X[units]
    N = units.sum()
    if varying:   # covariates of period p, with a constant
        Xp = lambda p: np.column_stack([X[:, p - 1, :], np.ones(N)])
    else:
        Xc = np.column_stack([X, np.ones(N)])
        Xp = lambda p: Xc
    w = w / w.mean()
    cohorts = np.unique(gvar[gvar > 0])

    # 2. ATT(g,t) for every cell; dy = y_t - y_base (for a long-gap pre-treatment cell, t < base)
    gt, rif_cols, wgt_cols = [], [], []
    for g, t, base, s in cells(obs, gvar, control, base_period, rules):
        D = (gvar[s] == g).astype(np.float64)
        Xs = Xp(min(base, t))[s]                            # the cell's earlier period
        if rules == "csdid201" and method != "reg" and overlap_fails(D, Xs):
            continue                                       # 2.0.1 leaves the cell unestimated
        ws = w[s] / w[s].mean()
        r = cell_rif(y[s, t - 1] - y[s, base - 1], Xs, D, ws, method, trim)
        full = np.zeros(N)
        full[s] = (r - r.mean()) * N / s.sum()            # scale to the full sample; others contribute 0
        rif_cols.append(full + r.mean())
        members = (gvar == g) if rules == "csdid201" else (s & (gvar == g))
        wgt_cols.append(w * members)                       # cohort weight of this cell
        gt.append((g, t))
    rif = np.column_stack(rif_cols)
    wgt = np.column_stack(wgt_cols)
    wgt /= wgt.mean(0).sum()
    gt = np.array(gt, dtype=np.int64)
    e = gt[:, 1] - gt[:, 0]
    post = e >= 0

    # 3. Simple ATT
    rif_simple = aggte(rif[:, post], wgt[:, post])

    # 4. By cohort, and their average
    groups = np.array([g for g in cohorts if (post & (gt[:, 0] == g)).any()], dtype=np.int64)
    rif_group = np.column_stack([aggte(rif[:, post & (gt[:, 0] == g)], wgt[:, post & (gt[:, 0] == g)])
                                 for g in groups])
    if rules == "csdid201":   # cohort shares estimated from unit-level membership (R did, csdid2)
        rif_group_avg = aggte(rif_group, np.column_stack([w * (gvar == g) for g in groups]))
    else:                     # cohort shares held fixed at the mean cohort weight (csdid 1.81)
        gw = np.array([wgt[:, post & (gt[:, 0] == g)].mean(0).mean() for g in groups])
        rif_group_avg = rif_group @ (gw / gw.sum())

    # 5. By calendar period, and their simple average
    periods = np.array(sorted({t for t in gt[post, 1]}), dtype=np.int64)
    rif_cal = np.column_stack([aggte(rif[:, post & (gt[:, 1] == t)], wgt[:, post & (gt[:, 1] == t)])
                               for t in periods])
    rif_cal_avg = rif_cal.mean(1)

    # 6. Event study within the window, the pre/post averages, and the event-time subsets
    lo, hi = event_window
    events = np.array([k for k in np.unique(e) if lo <= k <= hi], dtype=np.int64)
    rif_event = np.column_stack([aggte(rif[:, e == k], wgt[:, e == k]) for k in events])
    pre, pst = events < 0, events >= 0
    rif_pre = rif_event[:, pre].mean(1) if pre.any() else np.full(N, np.nan)
    rif_post = rif_event[:, pst].mean(1) if pst.any() else np.full(N, np.nan)
    sets = [np.asarray(sorted(set(int(k) for k in es)), dtype=np.int64) for es in (event_sets or [])]
    all_events = np.unique(e)
    rif_sets = [np.column_stack([aggte(rif[:, e == k], wgt[:, e == k]) for k in es if k in all_events]).mean(1)
                for es in sets]

    one = lambda r: (float(r.mean()), float(se(r[:, None])[0]))
    out = dict(gt=gt, att_gt=rif.mean(0), se_gt=se(rif),
               group_ids=groups, att_group=rif_group.mean(0), se_group=se(rif_group),
               calendar_periods=periods, att_calendar=rif_cal.mean(0), se_calendar=se(rif_cal),
               event_times=events, att_event=rif_event.mean(0), se_event=se(rif_event),
               cov_event=((rif_event - rif_event.mean(0)).T @ (rif_event - rif_event.mean(0))) / N ** 2,
               event_sets=[s.tolist() for s in sets],
               att_event_sets=np.array([r.mean() for r in rif_sets]),
               se_event_sets=np.array([se(r[:, None])[0] for r in rif_sets]))
    for name, r in (("simple", rif_simple), ("group_avg", rif_group_avg), ("calendar_avg", rif_cal_avg),
                    ("pre_avg", rif_pre), ("post_avg", rif_post)):
        out[f"att_{name}"], out[f"se_{name}"] = one(r)

    # 7. Multiplier bootstrap (one set of multipliers for every table)
    if boot_weights is not None:
        v = np.asarray(boot_weights)[units]
        tables = dict(gt=rif, group=rif_group, calendar=rif_cal, event=rif_event)
        singles = dict(simple=rif_simple, group_avg=rif_group_avg, calendar_avg=rif_cal_avg,
                       pre_avg=rif_pre, post_avg=rif_post)
        blocks = [tables[k] for k in tables] + [singles[k][:, None] for k in singles if np.isfinite(singles[k]).all()]
        draws = bootstrap(np.column_stack(blocks), v)
        col = 0
        for k, m in tables.items():
            d = draws[:, col:col + m.shape[1]]
            col += m.shape[1]
            out[f"se_boot_{k}"], out[f"crit_{k}"] = boot_stats(d, alpha, floor=(k != "gt"), rules=rules)
        for k in singles:
            if np.isfinite(singles[k]).all():
                out[f"se_boot_{k}"] = float(boot_stats(draws[:, col:col + 1], alpha, floor=True, rules=rules)[0][0])
                col += 1
            else:
                out[f"se_boot_{k}"] = np.nan
    return out
