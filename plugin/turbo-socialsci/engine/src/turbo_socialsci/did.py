"""Callaway & Sant'Anna (2021) staggered difference-in-differences with panel data, as Stata's csdid
1.81 or 2.0.1 computes it, on the SocialSciBench task-002 solvers."""
from dataclasses import dataclass, field

import numpy as np

from ._common import check_variant, choose, not_numeric, read_table, require, thread_limit

RULES = {"csdid181": "csdid181", "1.81": "csdid181", "181": "csdid181",
         "csdid201": "csdid201", "2.0.1": "csdid201", "201": "csdid201"}
METHODS = ("dripw", "drimp", "reg", "ipw", "stdipw")


@dataclass
class DiDResult:
    """gt: one row per ATT(g,t) (cohort, time: the data's period values); event, group, calendar: the
    aggregations; averages: simple, group_avg, calendar_avg, pre_avg, post_avg; crit: uniform critical
    values (bootstrap only); cov_event: the event-study covariance. se is the analytic (influence-
    function) SE; se_boot the multiplier-bootstrap SE."""
    gt: object
    event: object
    group: object
    calendar: object
    averages: object
    crit: dict
    cov_event: object
    n_units: int
    variant: str
    raw: dict = field(repr=False, default_factory=dict)


def att_gt(data, y, unit, time, cohort, *, covariates=(), weights=None, control="never", method="dripw",
           base_period="varying", trim=1.0, rules="csdid181", n_boot=None, seed=1, alpha=0.05,
           event_window=None, event_sets=None, boot_tables=None, variant="auto", threads=None, check_only=False):
    """Group-time ATTs and their aggregations from a long panel (one row per unit and period).

    y, unit, time, cohort: column names. cohort = the period of first treatment (0 = never treated
        within the panel); constant within unit; time must be consecutive integers (e.g. years).
    covariates: columns, fixed or varying over time (a cell uses its earlier period's values).
    weights:    a column constant within unit (sampling weights), or None.
    control:    "never" or "notyet" (csdid's notyet).
    method:     "dripw" (csdid's default: doubly robust with a logit and least squares), "drimp"
                (inverse probability tilting), "reg", "stdipw" (normalized IPW), "ipw".
    base_period: "varying" or "universal" (csdid 1.81's long2: every cell against g - 1).
    trim:       propensity-score trimming level (csdid's pscoretrim(); 1.0 = none).
    rules:      "csdid181" (Stata csdid 1.81 + drdid 1.91) or "csdid201" (csdid 2.0.1).
    n_boot, seed: multiplier bootstrap with n_boot Rademacher draws per unit (shared by every
                parameter), drawn inside the call from seed; None = analytic SEs only.
    event_window: (lo, hi) event times reported; None = every event time.
    boot_tables: the tables that get bootstrap SEs and critical values, from "gt", "event", "group",
                "calendar" (None: all). The averages follow their tables (pre_avg and post_avg:
                "event"; group_avg: "group"; calendar_avg: "calendar"; simple: always). Without
                "gt", the speed variant draws only the aggregates' columns, which is faster; results
                of tables not asked for are NaN.
    variant:    "speed", "memory" (bounded working memory, slower; NumPy only except the bootstrap's
                kernel) or "auto".

    Unbalanced panels are allowed (csdid's pair-balanced cells). Raises NotApplicable for what this
    does not implement: repeated cross-sections, weights varying over time, anticipation, clustered
    SEs or bootstrap, non-consecutive periods, cohorts outside the panel's periods.
    """
    check_variant(variant)
    require(method in METHODS, f"method {method!r} is not implemented (one of {METHODS})")
    require(control in ("never", "notyet"), f"control {control!r} is not implemented ('never' or 'notyet')")
    require(base_period in ("varying", "universal"), "base_period must be 'varying' or 'universal'")
    require(str(rules) in RULES, f"rules {rules!r}: use 'csdid181' (csdid 1.81) or 'csdid201' (csdid 2.0.1)")
    rules = RULES[str(rules)]
    if boot_tables is not None:
        boot_tables = [boot_tables] if isinstance(boot_tables, str) else list(boot_tables)
        unknown = [t for t in boot_tables if t not in ("gt", "event", "group", "calendar")]
        if unknown:
            raise ValueError(f"boot_tables: unknown tables {unknown} (use 'gt', 'event', 'group', 'calendar')")
    covariates = [covariates] if isinstance(covariates, str) else list(covariates)
    cols = [y, unit, time, cohort] + covariates + ([weights] if weights else [])
    df = read_table(data, cols)
    bad = not_numeric(df, [y, time, cohort] + covariates + ([weights] if weights else []))
    require(not bad, f"non-numeric columns: {bad}")
    tv = df[time].to_numpy()
    require(np.all(np.isfinite(tv)) and np.all(tv == np.round(tv)), "time must be integer-valued")
    ti = tv.astype(np.int64)
    del tv
    t0 = int(ti.min()) if ti.size else 0
    tcode = ti - t0
    del ti
    T = int(tcode.max()) + 1 if tcode.size else 0
    require(T >= 2 and bool((np.bincount(tcode, minlength=T) > 0).all()),
            "periods must be consecutive integers (gaps change csdid's base periods)")
    ucode, N = _unit_codes(df[unit], numpy_only=(variant == "memory"))
    ucode = ucode.astype(np.int64)
    require(ucode.min(initial=0) >= 0, "missing unit or time values")
    require(np.bincount(ucode * T + tcode, minlength=N * T).max(initial=0) <= 1,
            "several rows for one unit and period: repeated cross-sections are not implemented")
    g_row = df[cohort].to_numpy(dtype=np.float64, na_value=np.nan)
    require(np.isfinite(g_row).all(), "missing cohort values (use 0 for never treated)")
    g_unit = np.full(N, np.nan)
    g_unit[ucode] = g_row
    require(np.array_equal(g_unit[ucode], g_row), "cohort varies within a unit")
    gv = g_unit.astype(np.int64)
    treated = gv != 0
    require(np.all((gv[treated] >= t0) & (gv[treated] <= t0 + T - 1)),
            "cohort values must be 0 (never treated) or one of the panel's periods")
    gvar = np.where(treated, gv - t0 + 1, 0).astype(np.int64)
    w = None
    if weights:
        w_row = df[weights].to_numpy(dtype=np.float64, na_value=np.nan)
        require(np.isfinite(w_row).all() and (w_row > 0).all(), "weights must be positive and not missing")
        w = np.full(N, np.nan)
        w[ucode] = w_row
        require(np.array_equal(w[ucode], w_row), "weights that vary over time are not implemented")
    Y = np.full((N, T), np.nan)
    Y[ucode, tcode] = df[y].to_numpy(dtype=np.float64, na_value=np.nan)
    k = len(covariates)
    varying = False
    if k:
        # covariates constant over each unit's rows are passed as fixed [N, k]; otherwise (or with a
        # missing value, which makes that unit-period unobserved, as csdid drops it) as varying [N, T, k]
        C = np.full((N, T, k), np.nan)
        for j, c in enumerate(covariates):
            v = df[c].to_numpy(dtype=np.float64, na_value=np.nan)
            C[ucode, tcode, j] = v
            if not varying:
                if np.isnan(v).any():
                    varying = True
                else:
                    Cj = C[:, :, j]
                    varying = not np.array_equal(np.nanmin(Cj, axis=1), np.nanmax(Cj, axis=1))
            del v
        X = C if varying else np.nanmax(C, axis=1)
        del C
    else:
        X = np.zeros((N, 0))
    ew = (-T, T) if event_window is None else tuple(int(v) for v in event_window)
    speed_bytes = 8 * N * (T * (k + 2) + 64) + 400 * 2**20
    use = choose(variant, speed_bytes)
    if check_only:
        return dict(applies=True, units=int(N), periods=T, varying_covariates=varying, variant=use,
                    speed_bytes_estimate=speed_bytes)
    if use == "speed":
        from .core import did_speed as core
    else:
        from .core import did_memory as core
    with thread_limit(threads, numba=(use == "speed" or n_boot is not None)):
        r = core.run(Y, X, gvar, w=w, control=control, method=method, event_window=ew,
                     base_period=base_period, trim=float(trim), alpha=alpha, event_sets=event_sets,
                     rules=rules, n_boot=n_boot, seed=seed, boot_tables=boot_tables)
    return _result(r, t0, use, n_boot is not None, int(N))


def _result(r, t0, variant, boot, n_units):
    import pandas as pd
    year = lambda p: np.asarray(p, dtype=np.int64) + t0 - 1
    gt = pd.DataFrame(dict(cohort=year(r["gt"][:, 0]), time=year(r["gt"][:, 1]), att=r["att_gt"], se=r["se_gt"]))
    event = pd.DataFrame(dict(e=np.asarray(r["event_times"], dtype=np.int64), att=r["att_event"], se=r["se_event"]))
    group = pd.DataFrame(dict(cohort=year(r["group_ids"]), att=r["att_group"], se=r["se_group"]))
    calendar = pd.DataFrame(dict(time=year(r["calendar_periods"]), att=r["att_calendar"], se=r["se_calendar"]))
    names = ["simple", "group_avg", "calendar_avg", "pre_avg", "post_avg"]
    averages = pd.DataFrame(dict(att=[r[f"att_{n}"] for n in names], se=[r[f"se_{n}"] for n in names]),
                            index=pd.Index(names, name="aggregate"))
    crit = {}
    if boot:
        gt["se_boot"] = r["se_boot_gt"]
        event["se_boot"] = r["se_boot_event"]
        group["se_boot"] = r["se_boot_group"]
        calendar["se_boot"] = r["se_boot_calendar"]
        averages["se_boot"] = [r[f"se_boot_{n}"] for n in names]
        crit = dict(gt=r["crit_gt"], event=r["crit_event"], event_with_avgs=r["crit_event_with_avgs"],
                    group=r["crit_group"], calendar=r["crit_calendar"])
    e = event["e"].to_numpy()
    cov = pd.DataFrame(r["cov_event"], index=e, columns=e)
    return DiDResult(gt=gt, event=event, group=group, calendar=calendar, averages=averages, crit=crit,
                     cov_event=cov, n_units=n_units, variant=variant, raw=r)


def _unit_codes(series, numpy_only=False):
    """Codes of the units in increasing order of their ids (np.unique's order, which keys the
    bootstrap multipliers) and their number; -1 for a missing id. numpy_only keeps the JIT
    compiler out of the process (the "memory" variant)."""
    dt = series.dtype
    if not numpy_only and isinstance(dt, np.dtype) and dt.kind in "iuf":
        from ._codes import codes
        return codes(series)                       # integer ids: ranks of the values
    a = series.to_numpy()
    miss = series.isna().to_numpy()
    uid, inv = np.unique(a[~miss], return_inverse=True)
    out = np.full(a.size, -1, np.int64)
    out[~miss] = inv.reshape(-1)
    return out, int(uid.size)
