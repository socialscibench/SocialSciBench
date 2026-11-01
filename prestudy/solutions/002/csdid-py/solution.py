"""Task 002 competitor: Python csdid 0.4.2 ATTgt (faster mode) and aggte.

Follows the 14 October 2026 study's `csdid_fast` engine (research/2026-10-14/task002/
benchmark.py in the companion repository): never-treated controls, varying base period,
doubly robust estimator with csdid's own 0.995 trimming, analytic standard errors, no
bootstrap, one core. The equal-weight pre-period average and its standard error come from the
event-time influence functions, as in the study. The long data frame is built from the task's
wide inputs inside the timed call (the study built it outside its timer).
"""
import numpy as np
import pandas as pd
from csdid.att_gt import ATTgt


def _scalar(x):
    return float(np.asarray(x).reshape(-1)[0])


def run(y, X, gvar):
    n, T = y.shape
    k = X.shape[1]
    cols = dict(id=np.repeat(np.arange(1, n + 1), T), t=np.tile(np.arange(1, T + 1), n),
                y=np.asarray(y, dtype=float).ravel(), g=np.repeat(gvar, T))
    cols.update({f"x{j + 1}": np.repeat(X[:, j], T) for j in range(k)})
    df = pd.DataFrame(cols)
    formula = "y ~ " + " + ".join(f"x{j + 1}" for j in range(k))
    fit = ATTgt(data=df, yname="y", tname="t", idname="id", gname="g", xformla=formula,
                panel=True, control_group="nevertreated", cband=False, faster_mode=True,
                cores=1, pl=False).fit(est_method="dr", base_period="varying", bstrap=False)
    cell = fit.results
    fit.aggte("simple", bstrap=False, cband=False)
    simple = fit.atte
    fit.aggte("dynamic", bstrap=False, cband=False)
    event = fit.atte
    times = np.asarray(event["egt"])
    pre = times < 0
    ipre = np.asarray(event["inf_function"]["dynamic_inf_func_e"])[:, pre].mean(axis=1)
    att_event = np.asarray(event["att_egt"], dtype=float).reshape(-1)
    return dict(
        gt=np.column_stack([cell["group"], cell["year"]]).astype(np.int64),
        att_gt=np.asarray(cell["att"], dtype=float), se_gt=np.asarray(cell["se"], dtype=float),
        att_simple=_scalar(simple["overall_att"]), se_simple=_scalar(simple["overall_se"]),
        event_times=times.astype(np.int64), att_event=att_event,
        se_event=np.asarray(event["se_egt"], dtype=float).reshape(-1),
        att_pre_avg=float(att_event[pre].mean()),
        se_pre_avg=float(np.sqrt(np.sum((ipre - ipre.mean()) ** 2)) / len(ipre)),
        att_post_avg=_scalar(event["overall_att"]), se_post_avg=_scalar(event["overall_se"]),
    )
