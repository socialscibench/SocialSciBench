"""Task 002 competitor: ModernDiD 0.2.0 att_gt and aggte.

Follows the 14 October 2026 study's `moderndid` engine (research/2026-10-14/task002/
benchmark.py in the companion repository): never-treated controls, varying base period,
doubly robust estimator, analytic standard errors, no bootstrap, one job. The equal-weight
pre-period average and its standard error come from the event-time influence functions, as in
the study. The long data frame is built from the task's wide inputs inside the timed call (the
study built it outside its timer).
"""
import os

os.environ.setdefault("POLARS_MAX_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import moderndid as md  # noqa: E402


def run(y, X, gvar):
    n, T = y.shape
    k = X.shape[1]
    cols = dict(id=np.repeat(np.arange(1, n + 1), T), t=np.tile(np.arange(1, T + 1), n),
                y=np.asarray(y, dtype=float).ravel(), g=np.repeat(gvar, T))
    cols.update({f"x{j + 1}": np.repeat(X[:, j], T) for j in range(k)})
    df = pd.DataFrame(cols)
    formula = "~" + " + ".join(f"x{j + 1}" for j in range(k))
    fit = md.att_gt(data=df, yname="y", tname="t", idname="id", gname="g", xformla=formula,
                    panel=True, control_group="nevertreated", base_period="varying",
                    est_method="dr", boot=False, cband=False, n_jobs=1)
    simple = md.aggte(fit, type="simple", boot=False, cband=False)
    event = md.aggte(fit, type="dynamic", boot=False, cband=False)
    times = np.asarray(event.event_times)
    pre = times < 0
    ipre = np.asarray(event.influence_func)[:, pre].mean(axis=1)
    att_event = np.asarray(event.att_by_event, dtype=float)
    return dict(
        gt=np.column_stack([fit.groups, fit.times]).astype(np.int64),
        att_gt=np.asarray(fit.att_gt, dtype=float), se_gt=np.asarray(fit.se_gt, dtype=float),
        att_simple=float(simple.overall_att), se_simple=float(simple.overall_se),
        event_times=times.astype(np.int64), att_event=att_event,
        se_event=np.asarray(event.se_by_event, dtype=float),
        att_pre_avg=float(att_event[pre].mean()),
        se_pre_avg=float(np.sqrt(np.sum((ipre - ipre.mean()) ** 2)) / len(ipre)),
        att_post_avg=float(event.overall_att), se_post_avg=float(event.overall_se),
    )
