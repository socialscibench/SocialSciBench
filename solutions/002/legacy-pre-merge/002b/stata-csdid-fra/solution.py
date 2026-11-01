"""Task 002b baseline: Stata csdid 1.82 with drdid 1.91 (Rios-Avila, Sant'Anna & Callaway), in-process
via pystata (Stata 18 MP). The commands the task targets (the package's stata/run_002b.do):

    csdid y x1 ... xk [iw = w], ivar(id) time(t) gvar(gvar) method(m) [notyet]
    estat simple
    estat group                                      // GAverage, G#
    estat calendar                                   // CAverage, T#
    estat event, window(lo hi)                       // Pre_avg, Post_avg, Tm#, Tp#

The varying base period and no propensity-score trimming are csdid's defaults (it passes
pscoretrim(1) to drdid). The inputs are written as a long panel with one row per observed
unit-period (NaN outcomes are absent rows, so an unbalanced panel reaches csdid as a dataset with
gaps), to a temporary .dta file inside the timed call. ATT(g,t) and their standard errors come from
e(b)/e(V) (equation g<cohort>, column t_<base>_<t>; csdid's cohort-weight equation `wgt` is
skipped), the aggregates from r(b)/r(V) of estat. Stata MP is limited to the evaluator's thread
count. The ado path is Stata's own plus the downloaded packages (SOCIALSCIBENCH_CSDID_ADO).
"""
import os
import re
import sys
import tempfile

import numpy as np
import pandas as pd

TOOL = "csdid"
STATA_DIR = os.environ.get("SOCIALSCIBENCH_STATA", "/Applications/Stata 18")
def _venv():
    """The baseline environments: SOCIALSCIBENCH_BASELINE_VENV (set by compare.py), else
    solutions/baselines/.venv of the repository above this file (also from an evaluator run's snapshot)."""
    if os.environ.get("SOCIALSCIBENCH_BASELINE_VENV"):
        return os.environ["SOCIALSCIBENCH_BASELINE_VENV"]
    d = os.path.dirname(os.path.abspath(__file__))
    while not os.path.isdir(os.path.join(d, "solutions", "baselines", ".venv")):
        if os.path.dirname(d) == d:
            raise RuntimeError("solutions/baselines/.venv not found above " + os.path.abspath(__file__))
        d = os.path.dirname(d)
    return os.path.join(d, "solutions", "baselines", ".venv")


PKG = os.environ.get("SOCIALSCIBENCH_CSDID_ADO",
                     os.path.join(_venv(), "stata-src", "csdid-dl", "stpackages-main"))
TMP = tempfile.mkdtemp(prefix="ssb002b-stata-")

sys.path.append(os.path.join(STATA_DIR, "utilities"))
from pystata import config  # noqa: E402

config.init("mp", splash=False)
from pystata import stata  # noqa: E402
from sfi import Macro, Matrix  # noqa: E402

THREADS = int(os.environ.get("OMP_NUM_THREADS", "1"))
SETUP = ["set more off", f"set processors {THREADS}", "set max_memory 28g",
         f'sysdir set PERSONAL "{TMP}"', f'sysdir set PLUS "{TMP}"', f'sysdir set OLDPLACE "{TMP}"']
if TOOL == "csdid":
    SETUP += [f'adopath ++ "{os.path.abspath(os.path.join(PKG, "drdid"))}"',
              f'adopath ++ "{os.path.abspath(os.path.join(PKG, "csdid"))}"', "which csdid", "which drdid"]
else:
    SETUP += [f'adopath ++ "{os.path.abspath(os.path.join(PKG, "csdid2"))}"', "mata: mata mlib index", "which csdid2"]
for line in SETUP:
    stata.run(line, quietly=True)

NUM = r"(\d+)(?:\.0*)?"


def _matrix(kind):
    """b, V, column equations and column names of e(b)/e(V) or r(b)/r(V)."""
    stata.run(f"matrix SSB_b = {kind}(b)\nmatrix SSB_V = {kind}(V)\n"
              "global SSB_eq : coleq SSB_b\nglobal SSB_nm : colnames SSB_b", quietly=True)
    b = np.asarray(Matrix.get("SSB_b"), dtype=np.float64)[0]
    V = np.asarray(Matrix.get("SSB_V"), dtype=np.float64)
    return b, np.sqrt(np.diag(V)), Macro.getGlobal("SSB_eq").split(), Macro.getGlobal("SSB_nm").split()


def _labelled(command):
    """estat results by label: 'avg' for the GAverage/CAverage/ATT rows, 'pre'/'post' for the event
    averages, integers for G#/T# (cohort, period) and signed integers for Tm#/Tp# (event time)."""
    stata.run(command, quietly=True)
    b, se, _, names = _matrix("r")
    out = {}
    for nm, bi, si in zip(names, b, se):
        low = nm.lower()
        m = re.fullmatch(r"t([mp])" + NUM, low)
        if "average" in low or low in ("att", "simpleatt"):
            out["avg"] = (bi, si)
        elif low.startswith("pre"):
            out["pre"] = (bi, si)
        elif low.startswith("post"):
            out["post"] = (bi, si)
        elif m:
            out[(-1 if m.group(1) == "m" else 1) * int(m.group(2))] = (bi, si)
        else:
            out[int(re.findall(r"\d+", low)[-1])] = (bi, si)
    if "avg" not in out and len(names) == 1:        # csdid 1.82's estat simple: one unlabelled column (c1)
        out = {"avg": (b[0], se[0])}
    return out


def _arrays(d):
    keys = np.array(sorted(k for k in d if not isinstance(k, str)), dtype=np.int64)
    return keys, np.array([d[k][0] for k in keys]), np.array([d[k][1] for k in keys])


def run(y, X, gvar, w=None, control="never", method="dripw", event_window=(-5, 5)):
    if TOOL == "csdid2" and method == "ipw":
        raise ValueError("csdid2 has no ipw method")
    y = np.asarray(y, dtype=np.float64)
    N, T = y.shape
    X = np.asarray(X, dtype=np.float64).reshape(N, -1)
    k = X.shape[1]
    g = np.asarray(gvar, dtype=np.int64)
    ii, tt = np.nonzero(np.isfinite(y))
    cols = {"id": ii + 1, "t": tt + 1, "gvar": g[ii], "y": y[ii, tt]}
    for j in range(k):
        cols[f"x{j + 1}"] = X[ii, j]
    if w is not None:
        cols["w"] = np.asarray(w, dtype=np.float64)[ii]
    path = os.path.join(TMP, "panel.dta")
    pd.DataFrame(cols).to_stata(path, write_index=False, version=118)
    stata.run(f'use "{path}", clear', quietly=True)
    xs = " ".join(f"x{j + 1}" for j in range(k))
    wopt = " [iw = w]" if w is not None else ""
    options = f"ivar(id) time(t) gvar(gvar) method({method})" + (" notyet" if control == "notyet" else "")
    stata.run(f"{TOOL} y {xs}{wopt}, {options}" + (" short" if TOOL == "csdid2" else ""), quietly=True)
    if TOOL == "csdid":
        b, se, eqs, names = _matrix("e")
    else:
        stata.run("estat attgt", quietly=True)
        b, se, eqs, names = _matrix("r")
    cells = []
    for eq, nm, bi, si in zip(eqs, names, b, se):
        if eq.lower() == "wgt" or not re.fullmatch(r"g" + NUM, eq.lower()):
            continue
        nums = [int(x) for x in re.findall(r"\d+", f"{eq} {nm}")]
        cells.append((nums[0], nums[-1], bi, si))      # cohort first, the cell's later period last
    cells.sort()
    simple = _labelled("estat simple")
    group = _labelled("estat group")
    calendar = _labelled("estat calendar")
    event = _labelled(f"estat event, window({event_window[0]} {event_window[1]})")
    stata.run("clear", quietly=True)
    gid, ga, gs = _arrays(group)
    cid, ca, cs = _arrays(calendar)
    eid, ea, es = _arrays(event)
    return dict(gt=np.array([(c[0], c[1]) for c in cells], dtype=np.int64).reshape(-1, 2),
                att_gt=np.array([c[2] for c in cells]), se_gt=np.array([c[3] for c in cells]),
                att_simple=float(simple["avg"][0]), se_simple=float(simple["avg"][1]),
                group_ids=gid, att_group=ga, se_group=gs,
                att_group_avg=float(group["avg"][0]), se_group_avg=float(group["avg"][1]),
                calendar_periods=cid, att_calendar=ca, se_calendar=cs,
                att_calendar_avg=float(calendar["avg"][0]), se_calendar_avg=float(calendar["avg"][1]),
                event_times=eid, att_event=ea, se_event=es,
                att_pre_avg=float(event["pre"][0]) if "pre" in event else float("nan"),
                se_pre_avg=float(event["pre"][1]) if "pre" in event else float("nan"),
                att_post_avg=float(event["post"][0]), se_post_avg=float(event["post"][1]))
