"""Task 002 baseline: moderndid (Python port of R did/DRDID), att_gt + aggte.

    att_gt(data=<long panel>, yname="y", tname="t", idname="id", gname="g", xformla="~ x1 + ... + xk",
           est_method=partial(drdid_panel, trim_level=1.0), control_group="nevertreated",
           base_period="varying", boot=False,
           cband=False, n_jobs=N)
    aggte(<att_gt>, type="simple" / "dynamic", boot=False, cband=False)

est_method is moderndid's own drdid_panel (what est_method="dr" calls) with trim_level=1.0: no
comparison unit is trimmed, as in Stata's csdid and the task (moderndid's default trims scores at or
above 0.995); att_gt passes y1, y0, d, covariates, i_weights and influence_func to a callable.
The long panel (unit x period rows) is built from the wide inputs inside the timed call. The
pre/post averages of the event study are formed from the dynamic aggregation's influence
functions (mean effect; standard error sqrt(mean(IF^2) / n)).

Threads: n_jobs = the evaluator's thread limit (OMP_NUM_THREADS).
"""
import os
from functools import partial

import numpy as np
import polars as pl
import moderndid as md
from moderndid.drdid.estimators.drdid_panel import drdid_panel

DR_NOTRIM = partial(drdid_panel, trim_level=1.0)


def _get(obj, *names):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    raise AttributeError(names)


def run(y, X, gvar):
    threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
    y = np.asarray(y, dtype=np.float64)
    N, T = y.shape
    X = np.asarray(X, dtype=np.float64).reshape(N, -1)
    k = X.shape[1]
    cols = {"id": np.tile(np.arange(1, N + 1), T), "t": np.repeat(np.arange(1, T + 1), N),
            "y": y.T.ravel(), "g": np.tile(np.asarray(gvar, dtype=np.int64), T)}
    for j in range(k):
        cols[f"x{j + 1}"] = np.tile(X[:, j], T)
    data = pl.DataFrame(cols)
    xf = "~ " + " + ".join(f"x{j + 1}" for j in range(k)) if k else None
    a = md.att_gt(data=data, yname="y", tname="t", idname="id", gname="g", xformla=xf, est_method=DR_NOTRIM,
                  control_group="nevertreated", base_period="varying", boot=False, cband=False, n_jobs=threads)
    s = md.aggte(a, type="simple", boot=False, cband=False)
    e = md.aggte(a, type="dynamic", boot=False, cband=False)
    egt = np.asarray(_get(e, "event_times", "egt"), dtype=np.int64)
    att_e = np.asarray(_get(e, "att_by_event", "att_egt"), dtype=np.float64)
    se_e = np.asarray(_get(e, "se_by_event", "se_egt"), dtype=np.float64)
    IFe = np.asarray(_get(e, "influence_func", "inf_func_mat"), dtype=np.float64)
    if IFe.ndim == 2 and IFe.shape[0] != N:
        IFe = IFe.T

    def avg(sel):
        f = IFe[:, sel].mean(axis=1)
        return float(att_e[sel].mean()), float(np.sqrt(np.mean(f ** 2) / N))
    pre = egt < 0
    gt = np.column_stack([np.asarray(_get(a, "groups", "group")), np.asarray(_get(a, "times", "t"))]).astype(np.int64)
    att_pre, se_pre = avg(pre)
    att_post, se_post = avg(~pre)
    return dict(gt=gt, att_gt=np.asarray(_get(a, "att_gt", "att"), dtype=np.float64),
                se_gt=np.asarray(_get(a, "se_gt", "se"), dtype=np.float64),
                att_simple=float(_get(s, "overall_att")), se_simple=float(_get(s, "overall_se")),
                event_times=egt, att_event=att_e, se_event=se_e,
                att_pre_avg=att_pre, se_pre_avg=se_pre, att_post_avg=att_post, se_post_avg=se_post)
