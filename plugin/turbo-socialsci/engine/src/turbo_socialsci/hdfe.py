"""OLS with absorbed fixed effects and one-way cluster-robust standard errors (Stata reghdfe's
conventions): one fixed effect by the exact within transformation (core/hdfe_onefe.py), several on the
SocialSciBench task-001 solvers."""
from dataclasses import dataclass, field

import numpy as np

from ._common import check_variant, choose, codes_of, not_numeric, read_table, require, thread_limit


@dataclass
class HDFEResult:
    """coef: DataFrame indexed by regressor (coef, se, t, p, ci_low, ci_high, dropped); vcov: DataFrame
    of the kept regressors. p-values and 95% intervals use the t distribution with n_clusters - 1
    degrees of freedom, as reghdfe does with clustered errors."""
    coef: object
    vcov: object
    nobs: int
    n_clusters: int
    df_a: int
    r2: float
    r2_within: float
    rss: float
    tss: float
    dropped: list
    n_missing: int
    n_singletons: int
    variant: str
    extra: dict = field(default_factory=dict)

    def summary(self):
        lines = [f"N = {self.nobs:,}  clusters = {self.n_clusters:,}  df_a = {self.df_a:,}  "
                 f"R2 = {self.r2:.6f}  within R2 = {self.r2_within:.6f}  ({self.variant} core)"]
        if self.dropped:
            lines.append(f"dropped as collinear: {', '.join(self.dropped)}")
        lines.append(self.coef.to_string())
        return "\n".join(lines)


def feols(data, y, x, absorb, cluster, *, variant="auto", threads=None, collinear_tol=1e-9, check_only=False):
    """Fit y on x with the fixed effects in `absorb` partialled out; one-way clustered SEs.

    data:     DataFrame or path (.parquet/.csv/.dta/.feather)
    y:        outcome column; x: list of regressor columns (at least one)
    absorb:   list of fixed effects, each a column or an interaction "a#b" (one effect per combination)
    cluster:  cluster column (or "a#b")
    variant:  "speed", "memory" or "auto". With one fixed effect both compute the same numbers
              (exactly, by group means); "speed" also copies the sample into the groups' order (about
              8 bytes per row and column). With several, "memory" forms no n x k demeaned matrix and
              works through batches of columns.
    threads:  BLAS and Numba threads for this call (None: library defaults / environment variables)

    As reghdfe: rows with a missing outcome, regressor, fixed effect or cluster are dropped; then
    singleton groups are dropped repeatedly; regressors collinear with the fixed effects or with
    regressors to their left are dropped; an intercept is included; the small-sample factor is
    (N-1)/(N-K) * G/(G-1) with K = kept regressors + 1 + df_a, where fixed effects nested in the
    cluster variable add nothing to df_a. R2 is the full model's (fixed effects included).
    Raises NotApplicable for what this does not implement (no cluster variable, no fixed effect,
    weights, other variance estimators, IV, nonlinear models).
    """
    check_variant(variant)
    x = [x] if isinstance(x, str) else list(x)
    absorb = [absorb] if isinstance(absorb, str) else list(absorb)
    require(cluster is not None and str(cluster).strip() != "",
            "only one-way cluster-robust standard errors are implemented (give cluster=)")
    require(len(absorb) >= 1, "at least one absorbed fixed effect is required")
    require(len(x) >= 1, "at least one regressor is required")
    require(len(set(x)) == len(x), "a regressor is listed twice")
    need = [y] + x + [p.strip() for spec in absorb + [cluster] for p in str(spec).split("#")]
    df = read_table(data, need)
    bad = not_numeric(df, [y] + x)
    require(not bad, f"non-numeric outcome or regressors: {bad} (encode them first)")
    if len(absorb) == 1:
        return _one_fe(df, y, x, absorb[0], cluster, variant, threads, collinear_tol, check_only)
    return _several_fe(df, y, x, absorb, cluster, variant, threads, collinear_tol, check_only)


# ---- one fixed effect ------------------------------------------------------------------------------

def _readonly(a):
    v = np.ascontiguousarray(a).view()
    v.flags.writeable = False
    return v


def _split_columns(df, y, x, demote=()):
    """The outcome and the regressors as arrays read in place where the table allows: regressors
    holding only 0 and 1 as uint8 dummy columns (views of one-byte integer and boolean columns, copies
    of others), the rest (y first) in one float type. Returns (others, dummies, ftype, xmap) with
    xmap[j] the position of x[j] in others[1:] + dummies."""
    from .core.hdfe_onefe import _zero_one
    others, dummies, where = [], [], []
    for name in [y] + x:
        s = df[name]
        dt = s.dtype
        arr = None
        if name != y and name not in demote and isinstance(dt, np.dtype):
            if dt.kind == "b" or dt.itemsize == 1 and dt.kind in "iu":
                arr = s.to_numpy().view(np.uint8)           # checked for 0/1 by the core
            elif dt.kind in "iu":
                a = s.to_numpy()
                if a.size and a.min() >= 0 and a.max() <= 1:
                    arr = a.astype(np.uint8)
            elif dt.kind == "f":
                a = s.to_numpy()
                if a.size and _zero_one(_readonly(a)):
                    arr = a.astype(np.uint8)
        if arr is not None and len(dummies) < 65535:     # the core numbers dummy columns in 16 bits
            where.append(("d", len(dummies)))
            dummies.append(arr)
            continue
        if arr is not None:
            a = arr.astype(np.float64)
        elif isinstance(dt, np.dtype) and dt.kind == "f":
            a = s.to_numpy()
        elif isinstance(dt, np.dtype) and dt.kind in "iub":
            a = s.to_numpy().astype(np.float64)
        else:                                               # nullable and other extension dtypes
            a = s.to_numpy(dtype=np.float64, na_value=np.nan)
        where.append(("z", len(others)))
        others.append(a)
    if all(a.dtype == np.float32 for a in others):
        ftype = np.float32
    else:
        ftype = np.float64
        others = [a if a.dtype == np.float64 else a.astype(np.float64) for a in others]
    kd = len(others) - 1
    xmap = [i - 1 if kind == "z" else kd + i for kind, i in where[1:]]
    return others, dummies, ftype, np.asarray(xmap, dtype=np.int64)


def _typed(arrays, dtype):
    import numba as nb
    lst = nb.typed.List.empty_list(nb.types.Array(nb.from_dtype(np.dtype(dtype)), 1, "C", readonly=True))
    for a in arrays:
        lst.append(_readonly(a))
    return lst


def _one_fe(df, y, x, fe_spec, cluster, variant, threads, collinear_tol, check_only):
    from . import _codes
    fe, G0 = _codes.codes_of(df, fe_spec)
    cl, Lc0 = _codes.codes_of(df, str(cluster))
    others, dummies, ftype, xmap = _split_columns(df, y, x)
    n0 = len(df)
    speed_bytes = n0 * (16 * len(others) + 64)
    use = choose(variant, speed_bytes)
    from .core import hdfe_onefe
    if check_only:
        import numba as nb
        ok = hdfe_onefe._complete(_typed(others, ftype), fe, cl, max(1, min(nb.get_num_threads(), n0 // 65536 + 1)))
        n = int(ok.sum())
        require(n > 1, "fewer than two complete rows")
        return dict(applies=True, rows=n, rows_missing=n0 - n, variant=use, speed_bytes_estimate=speed_bytes)
    with thread_limit(threads):
        r = hdfe_onefe.run(_typed(others, ftype), _typed(dummies, np.uint8), fe, G0, cl, Lc0, xmap,
                           collinear_tol=collinear_tol, gather=(use == "speed"))
        if "not_dummies" in r:                              # 1-byte integer columns that are not 0/1
            kd = len(others) - 1
            demote = [x[j] for j in range(len(x)) if xmap[j] - kd in set(r["not_dummies"].tolist())]
            others, dummies, ftype, xmap = _split_columns(df, y, x, demote=demote)
            r = hdfe_onefe.run(_typed(others, ftype), _typed(dummies, np.uint8), fe, G0, cl, Lc0, xmap,
                               collinear_tol=collinear_tol, gather=(use == "speed"))
    require(r["n_complete"] > 1, "fewer than two complete rows")
    require(not r.get("empty"), "fewer than two rows left after dropping singleton groups")
    return _result(r, x, n0 - r["n_complete"], r["n_complete"], use)


# ---- several fixed effects (the 0.1 path) -----------------------------------------------------------

def _several_fe(df, y, x, absorb, cluster, variant, threads, collinear_tol, check_only):
    fe = np.column_stack([codes_of(df, spec) for spec in absorb])
    cl = codes_of(df, str(cluster))
    Y = df[y].to_numpy(dtype=np.float64, na_value=np.nan)
    ok = np.isfinite(Y) & (fe >= 0).all(1) & (cl >= 0)
    Xdf = df[x]
    for c in x:
        ok &= np.isfinite(Xdf[c].to_numpy(dtype=np.float64, na_value=np.nan))
    n_missing = int((~ok).sum())
    n = int(ok.sum())
    require(n > 1, "fewer than two complete rows")
    k = len(x)
    speed_bytes = 8 * n * (2 * k + 1 + 4 * len(absorb) + 4)
    use = choose(variant, speed_bytes)
    if check_only:
        return dict(applies=True, rows=n, rows_missing=n_missing, variant=use, speed_bytes_estimate=speed_bytes)
    idx = np.flatnonzero(ok)
    Xa = np.empty((n, k))
    for j, c in enumerate(x):
        Xa[:, j] = Xdf[c].to_numpy(dtype=np.float64, na_value=np.nan)[idx]
    del Xdf
    Ya = np.ascontiguousarray(Y[idx])
    fe_a = np.ascontiguousarray(fe[idx])
    cl_a = np.ascontiguousarray(cl[idx])
    del fe, cl, Y
    if use == "speed":
        from .core import hdfe_speed as core
    else:
        from .core import hdfe_memory as core
    with thread_limit(threads):
        r = core.run(Ya, Xa, fe_a, cl_a, collinear_tol=collinear_tol)
    return _result(r, x, n_missing, n, use)


def _result(r, x, n_missing, n, use):
    import pandas as pd
    from scipy import special
    G = int(r["n_clusters"])
    require(G > 1, "only one cluster in the estimation sample")
    beta, V = r["beta"], r["vcov"]
    se = np.sqrt(np.diag(V))
    with np.errstate(invalid="ignore", divide="ignore"):
        t = beta / se
    dfr = G - 1
    p = 2 * special.stdtr(dfr, -np.abs(t))
    q = special.stdtrit(dfr, 0.975)
    kept = r["kept"]
    coef = pd.DataFrame(dict(coef=beta, se=se, t=t, p=p, ci_low=beta - q * se, ci_high=beta + q * se,
                             dropped=~kept), index=pd.Index(x, name="variable"))
    kx = [c for c, kk in zip(x, kept) if kk]
    vcov = pd.DataFrame(V[np.ix_(kept, kept)], index=kx, columns=kx)
    return HDFEResult(coef=coef, vcov=vcov, nobs=int(r["nobs"]), n_clusters=G, df_a=int(r["df_a"]),
                      r2=r["r2"], r2_within=r["r2_within"], rss=r["rss"], tss=r["tss"],
                      dropped=[c for c, kk in zip(x, kept) if not kk], n_missing=n_missing,
                      n_singletons=n - int(r["nobs"]), variant=use)
