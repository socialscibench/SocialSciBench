"""Task 002 baseline: Stata csdid 1.82 with drdid 1.91 (Rios-Avila, Sant'Anna & Callaway; the SSC
versions), in-process via pystata (Stata 18 MP), for the workloads that follow csdid 1.81's rules
(rules="csdid181"; 1.82 is 1.81 with a bug fix, the version on SSC):

    csdid y x1 ... xk [iw = w], ivar(id) time(t) gvar(gvar) method(m) [notyet] [long2]
          pscoretrim(#) level(#) [saverif(ssbrif) replace]
    estat simple / group / calendar
    estat event, window(lo hi)                         // Pre_avg, Post_avg, Tm#, Tp#

and, with bootstrap multipliers, on the saved influence functions (RIFs):

    use ssbrif, clear
    csdid_stats attgt|simple|group|calendar|event, wboot(reps(B) wbtype(rademacher) rseed(#))

- **Options:** the universal base period is csdid's `long2` (y_t - y_(g-1), the task's sign).
  Trimming is `pscoretrim()`, which csdid passes to drdid. Covariates that vary over time are
  given per unit-period row, and drdid uses the cell's earlier period, as the task does.
- **Estimates and analytic standard errors:** ATT(g,t) comes from e(b)/e(V) (equation
  g<cohort>, column t_<base>_<t>; the cohort-weight equation `wgt` is skipped), the aggregations
  from r(b)/r(V) of estat. The event-study covariance and the event-time subsets come from estat
  event's r(V); 1.82 has no event-subset command.
- **Event window under long2:** csdid 1.82 keeps a cohort's long2 cells in period order without
  g - 1, but its event aggregation (estat event and csdid_stats event) pairs them with e(tlev),
  the periods from the second on. A pre-treatment cell of period t is then taken as period
  t + 1: its Tm# label is right, but the window selects it as e + 1, so window(-5 5) returns
  e = -6..-2 before treatment. The window is moved by one period on the pre-treatment side, so
  the event times returned (and Pre_avg) are the task's.
- **Bootstrap:** csdid_stats' wild bootstrap of the saved RIFs gives the bootstrap standard
  errors (its interquartile-range estimate) and the uniform bands' critical values for ATT(g,t)
  and every aggregation. (estat after an estimation with wboot() reports asymptotic standard
  errors for the aggregations, so it is not used.) Stata draws its own Rademacher multipliers,
  so these differ from the reference's (computed from the given multipliers) by the bootstrap's
  sampling noise; they are timed but not comparable (compare.py --not-comparable se_boot_ crit_).

The long panel (one row per observed unit-period) is written to a temporary .dta file inside the
timed call. Stata MP is limited to the evaluator's thread count.
"""
import os
import re
import sys
import tempfile

import numpy as np
import pandas as pd

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
TMP = tempfile.mkdtemp(prefix="ssb002-csdid181-")
SEED = 20261025

sys.path.append(os.path.join(STATA_DIR, "utilities"))
from pystata import config  # noqa: E402

config.init("mp", splash=False)
from pystata import stata  # noqa: E402
from sfi import Macro, Matrix, Scalar  # noqa: E402

THREADS = int(os.environ.get("OMP_NUM_THREADS", "1"))
for line in ["set more off", f"set processors {THREADS}", "set max_memory 28g",
             f'sysdir set PERSONAL "{TMP}"', f'sysdir set PLUS "{TMP}"', f'sysdir set OLDPLACE "{TMP}"',
             f'adopath ++ "{os.path.abspath(os.path.join(PKG, "drdid"))}"',
             f'adopath ++ "{os.path.abspath(os.path.join(PKG, "csdid"))}"', "which csdid", "which drdid"]:
    stata.run(line, quietly=True)

NUM = r"(\d+)(?:\.0*)?"
RIF = "ssbrif"          # saverif() takes a name: the file is saved in Stata's working directory


def _matrix(kind):
    """b, V, column equations and column names of e(b)/e(V) or r(b)/r(V)."""
    stata.run(f"matrix SSB_b = {kind}(b)\nmatrix SSB_V = {kind}(V)\n"
              "global SSB_eq : coleq SSB_b\nglobal SSB_nm : colnames SSB_b", quietly=True)
    b = np.asarray(Matrix.get("SSB_b"), dtype=np.float64)[0]
    V = np.asarray(Matrix.get("SSB_V"), dtype=np.float64)
    return b, V, Macro.getGlobal("SSB_eq").split(), Macro.getGlobal("SSB_nm").split()


def _band():
    """r(cband) of csdid_stats with wboot (columns b se t ll ul, one row per estimate) and its row
    equations."""
    stata.run("matrix SSB_c = r(cband)\nglobal SSB_req : roweq SSB_c", quietly=True)
    return np.asarray(Matrix.get("SSB_c"), dtype=np.float64), Macro.getGlobal("SSB_req").split()


def _crit(cb):
    """The uniform band's critical value, (upper limit - estimate) / se (the same for every row)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        c = (cb[:, 4] - cb[:, 0]) / cb[:, 1]
    c = c[np.isfinite(c)]
    return float(c[0]) if c.size else float("nan")


def _window(lo, hi, universal):
    """estat event's window option for the event times lo..hi (see the notes on long2 above)."""
    if universal:
        lo, hi = (lo + 1 if lo < 0 else lo), (hi + 1 if hi < -1 else hi)
    return f"window({lo} {hi})"


def _labelled(command, band=False):
    """estat (or csdid_stats) results by label: 'avg' for the GAverage/CAverage/ATT rows, 'pre'/'post'
    for the event averages, integers for G#/T# (cohort, period) and signed integers for Tm#/Tp# (event
    time); with b and V of the labelled rows, and with band=True the bootstrap band's critical value."""
    stata.run(command, quietly=True)
    crit = _crit(_band()[0]) if band else None
    b, V, _, names = _matrix("r")
    out = {}
    for j, nm in enumerate(names):
        low = nm.lower()
        m = re.fullmatch(r"t([mp])" + NUM, low)
        if "average" in low or low in ("att", "simpleatt"):
            key = "avg"
        elif low.startswith("pre"):
            key = "pre"
        elif low.startswith("post"):
            key = "post"
        elif m:
            key = (-1 if m.group(1) == "m" else 1) * int(m.group(2))
        else:
            key = int(re.findall(r"\d+", low)[-1])
        out[key] = j
    if "avg" not in out and len(names) == 1:        # csdid 1.82's estat simple: one unlabelled column (c1)
        out = {"avg": 0}
    return (out, b, V, crit) if band else (out, b, V)


def _arrays(idx, b, V):
    keys = np.array(sorted(k for k in idx if not isinstance(k, str)), dtype=np.int64)
    rows = np.array([idx[k] for k in keys], dtype=np.int64)
    return keys, b[rows], np.sqrt(np.diag(V)[rows])


def _one(idx, b, V, key):
    if key not in idx:
        return float("nan"), float("nan")
    j = idx[key]
    return float(b[j]), float(np.sqrt(V[j, j]))


def run(y, X, gvar, w=None, control="never", method="dripw", event_window=(-5, 5), base_period="varying",
        trim=1.0, boot_weights=None, alpha=0.05, event_sets=None, rules="csdid181"):
    if rules != "csdid181":
        raise ValueError("this baseline follows csdid 1.81's rules (rules='csdid181')")
    y = np.asarray(y, dtype=np.float64)
    N, T = y.shape
    X = np.asarray(X, dtype=np.float64)
    varying = X.ndim == 3
    k = X.shape[-1]
    g = np.asarray(gvar, dtype=np.int64)
    ok = np.isfinite(y) & (np.isfinite(X).all(2) if varying else True)
    ii, tt = np.nonzero(ok)
    cols = {"id": ii + 1, "t": tt + 1, "gvar": g[ii], "y": y[ii, tt]}
    for j in range(k):
        cols[f"x{j + 1}"] = X[ii, tt, j] if varying else X.reshape(N, -1)[ii, j]
    if w is not None:
        cols["w"] = np.asarray(w, dtype=np.float64)[ii]
    path = os.path.join(TMP, "panel.dta")
    pd.DataFrame(cols).to_stata(path, write_index=False, version=118)
    stata.run(f'use "{path}", clear', quietly=True)
    xs = " ".join(f"x{j + 1}" for j in range(k))
    wopt = " [iw = w]" if w is not None else ""
    cmd = (f"csdid y {xs}{wopt}, ivar(id) time(t) gvar(gvar) method({method})"
           + (" notyet" if control == "notyet" else "") + (" long2" if base_period == "universal" else "")
           + f" pscoretrim({trim!r}) level({round(100 * (1 - alpha))})")
    lo, hi = event_window
    universal = base_period == "universal"
    boot = boot_weights is not None

    # estimation; with the bootstrap, the RIFs are saved too (in TMP: saverif() writes to Stata's
    # working directory, which is restored before anything else runs)
    if boot:
        here = os.getcwd()
        stata.run(f'cd "{TMP}"', quietly=True)
        try:
            stata.run(cmd + f" saverif({RIF}) replace", quietly=True)
        finally:
            stata.run(f'cd "{here}"', quietly=True)
            os.chdir(here)
    else:
        stata.run(cmd, quietly=True)
    b, V, eqs, names = _matrix("e")
    cells = []
    for j, (eq, nm) in enumerate(zip(eqs, names)):
        if eq.lower() == "wgt" or not re.fullmatch(r"g" + NUM, eq.lower()):
            continue
        nums = [int(x) for x in re.findall(r"\d+", f"{eq} {nm}")]
        if np.isfinite(b[j]):
            cells.append((nums[0], nums[-1], b[j], np.sqrt(V[j, j])))   # cohort first, the later period last
    if universal:          # long2 columns name the cell's period and the base g - 1; the period is the
        cells = []         # one that is not g - 1
        for j, (eq, nm) in enumerate(zip(eqs, names)):
            if eq.lower() == "wgt" or not re.fullmatch(r"g" + NUM, eq.lower()) or not np.isfinite(b[j]):
                continue
            gg = int(re.findall(r"\d+", eq)[0])
            ts = [int(x) for x in re.findall(r"\d+", nm)]
            t_ = [x for x in ts if x != gg - 1]
            cells.append((gg, t_[0] if t_ else ts[-1], b[j], np.sqrt(V[j, j])))
    cells.sort()
    out = dict(gt=np.array([(c[0], c[1]) for c in cells], dtype=np.int64).reshape(-1, 2),
               att_gt=np.array([c[2] for c in cells]), se_gt=np.array([c[3] for c in cells]))
    s_idx, sb, sV = _labelled("estat simple")
    out["att_simple"], out["se_simple"] = _one(s_idx, sb, sV, "avg")
    g_idx, gb, gV = _labelled("estat group")
    out["group_ids"], out["att_group"], out["se_group"] = _arrays(g_idx, gb, gV)
    out["att_group_avg"], out["se_group_avg"] = _one(g_idx, gb, gV, "avg")
    c_idx, cb_, cV = _labelled("estat calendar")
    out["calendar_periods"], out["att_calendar"], out["se_calendar"] = _arrays(c_idx, cb_, cV)
    out["att_calendar_avg"], out["se_calendar_avg"] = _one(c_idx, cb_, cV, "avg")

    def event(lo_, hi_):
        e_idx, eb, eV = _labelled(f"estat event, {_window(lo_, hi_, universal)}")
        keys = np.array(sorted(k for k in e_idx if not isinstance(k, str) and not (universal and k == -1)),
                        dtype=np.int64)
        rows = np.array([e_idx[k] for k in keys], dtype=np.int64)
        return e_idx, eb, eV, keys, rows

    e_idx, eb, eV, et, rows = event(lo, hi)
    out["event_times"], out["att_event"] = et, eb[rows]
    out["se_event"] = np.sqrt(np.diag(eV)[rows])
    out["cov_event"] = eV[np.ix_(rows, rows)]
    out["att_pre_avg"], out["se_pre_avg"] = _one(e_idx, eb, eV, "pre")
    out["att_post_avg"], out["se_post_avg"] = _one(e_idx, eb, eV, "post")
    sets = [sorted(set(int(v) for v in s)) for s in (event_sets or [])]
    out["event_sets"] = sets
    atts, ses = [], []
    if sets:
        alls = [v for s in sets for v in s]
        _, sb2, sV2, st, srows = event(min(alls + [lo]), max(alls + [hi]))
        for s in sets:
            sel = srows[np.isin(st, s)]
            c = np.full(sel.size, 1.0 / max(sel.size, 1))
            atts.append(float(c @ sb2[sel]))
            ses.append(float(np.sqrt(c @ sV2[np.ix_(sel, sel)] @ c)))
    out["att_event_sets"], out["se_event_sets"] = np.array(atts, dtype=np.float64), np.array(ses, dtype=np.float64)

    # bootstrap: csdid_stats on the saved RIFs (Stata's own Rademacher multipliers, as many draws as
    # given; the same seed, so the same draws, for every aggregation)
    if boot:
        B = int(np.asarray(boot_weights).shape[1])
        wb = f"wboot(reps({B}) wbtype(rademacher) rseed({SEED}))"
        stata.run(f'use "{os.path.join(TMP, RIF)}.dta", clear', quietly=True)
        stata.run(f"csdid_stats attgt, {wb}", quietly=True)
        cb, req = _band()               # rows: the cells (equations g#), then the weights (wgt)
        sel = [j for j, eq in enumerate(req) if eq.lower() != "wgt" and re.fullmatch(r"g" + NUM, eq.lower())
               and np.isfinite(cb[j, 0])]
        out["se_boot_gt"] = cb[sel, 1]
        out["crit_gt"] = _crit(cb[sel])
        for name, command in (("simple", "simple,"), ("group", "group,"), ("calendar", "calendar,"),
                              ("event", f"event, {_window(lo, hi, universal)}")):
            idx, bb, VV, crit = _labelled(f"csdid_stats {command} {wb}", band=True)
            if name == "simple":
                out["se_boot_simple"] = _one(idx, bb, VV, "avg")[1]
                continue
            keys, _, ses_ = _arrays(idx, bb, VV)
            if name == "event" and universal:
                ses_ = ses_[keys != -1]
            out[f"se_boot_{name}"] = ses_
            out[f"crit_{name}"] = crit
            if name in ("group", "calendar"):
                out[f"se_boot_{name}_avg"] = _one(idx, bb, VV, "avg")[1]
            else:
                out["se_boot_pre_avg"] = _one(idx, bb, VV, "pre")[1]
                out["se_boot_post_avg"] = _one(idx, bb, VV, "post")[1]
    stata.run("clear", quietly=True)
    return out
