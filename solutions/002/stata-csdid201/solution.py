"""Task 002 baseline: Stata csdid 2.0.1 (Callaway, Rios-Avila & Sant'Anna; commit f98bcea of
https://github.com/pedrohcgs/csdid-stata, installed in its own folder), in-process via pystata
(Stata 18 MP), for the workloads that follow csdid 2.0.1's rules (rules="csdid201"):

    csdid y x1 ... xk [iw = w], ivar(id) time(t) gvar(gvar) method(dr|reg|ipw) notyet|never
          base_period(varying|universal) pscoretrim(#) bal(pair) level(#) analytical pointwise
    estat simple|group|calendar, post dropmissing
    estat event, window(lo hi) post dropmissing

and, with bootstrap multipliers, the same estimation again with `reps(B) rseed(#)` for the
multiplier bootstrap. The task's dripw, reg and stdipw are 2.0.1's dr, reg and ipw (its ipw is
the normalized estimator); 1.81's unnormalized ipw and drimp are not available in 2.0.1.

- **Analytic run:** gives ATT(g,t) and their standard errors (e(b)/e(V); cells left
  unestimated by 2.0.1's overlap check are missing and dropped) and the aggregations.
  - `pointwise` keeps the analytic run from drawing the bootstrap it otherwise uses for
    simultaneous bands.
  - The pre-period average, the event-time subsets and the event-study covariance come from
    estat event's posted b and V. Under a universal base period the reference period (e = -1)
    row is dropped; the task does not estimate it.
- **Bootstrap run:** gives the bootstrap standard errors and simultaneous critical values. Stata
  draws its own Rademacher multipliers, so these differ from the reference's (computed from the
  given multipliers) by the bootstrap's sampling noise. They are timed but not comparable
  (compare.py --not-comparable se_boot_ crit_).

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


PKG = os.environ.get("SOCIALSCIBENCH_CSDID201_ADO", os.path.join(_venv(), "stata-src", "csdid-2.0.1"))
TMP = tempfile.mkdtemp(prefix="ssb002-csdid201-")
SEED = 20261025

sys.path.append(os.path.join(STATA_DIR, "utilities"))
from pystata import config  # noqa: E402

config.init("mp", splash=False)
from pystata import stata  # noqa: E402
from sfi import Matrix, Macro, Scalar  # noqa: E402


def _escalar(name):
    """An e() scalar (missing as NaN)."""
    stata.run(f"scalar SSB_s = e({name})", quietly=True)
    v = Scalar.getValue("SSB_s")
    return float(v) if v is not None and v < 8.9e307 else np.nan

THREADS = int(os.environ.get("OMP_NUM_THREADS", "1"))
for line in ["set more off", f"set processors {THREADS}", "set max_memory 28g",
             f'sysdir set PERSONAL "{TMP}"', f'sysdir set OLDPLACE "{TMP}"',
             f'sysdir set PLUS "{os.path.abspath(PKG)}"', "which csdid"]:
    stata.run(line, quietly=True)

METHOD = {"dripw": "dr", "reg": "reg", "stdipw": "ipw"}


def _post(kind):
    """b, V and column names of e(b)/e(V)."""
    stata.run(f"matrix SSB_b = {kind}(b)\nmatrix SSB_V = {kind}(V)\nglobal SSB_nm : colnames SSB_b", quietly=True)
    b = np.asarray(Matrix.get("SSB_b"), dtype=np.float64)[0]
    V = np.asarray(Matrix.get("SSB_V"), dtype=np.float64)
    return b, V, Macro.getGlobal("SSB_nm").split()


def _aggregation(command):
    """estat aggregation posted: {label: (b, se)}, b, V, names."""
    stata.run("estimates restore SSB_CS", quietly=True)
    stata.run(command + (" post dropmissing" if "," in command else ", post dropmissing"), quietly=True)
    b, V, names = _post("e")
    return b, V, names


def _keyed(names, b, V):
    """G#/T# rows by number, Overall/ATT as 'avg'."""
    out = {}
    for j, nm in enumerate(names):
        low = nm.lower()
        if low in ("overall", "att"):
            out["avg"] = (b[j], np.sqrt(V[j, j]))
        else:
            out[int(re.findall(r"\d+", low)[-1])] = (b[j], np.sqrt(V[j, j]))
    return out


def _arrays(d):
    keys = np.array(sorted(k for k in d if not isinstance(k, str)), dtype=np.int64)
    return keys, np.array([d[k][0] for k in keys]), np.array([d[k][1] for k in keys])


def _event(lo, hi, universal):
    """Event-time rows (event time, b) and their covariance, Post_avg dropped; under a universal
    base the reference period e = -1 is dropped."""
    b, V, names = _aggregation(f"estat event, window({lo} {hi})")
    keep, times = [], []
    for j, nm in enumerate(names):
        m = re.fullmatch(r"t([mp])(\d+)", nm.lower())
        if not m:
            continue
        e = (-1 if m.group(1) == "m" else 1) * int(m.group(2))
        if universal and e == -1:
            continue
        keep.append(j)
        times.append(e)
    keep = np.array(keep, dtype=np.int64)
    return np.array(times, dtype=np.int64), b[keep], V[np.ix_(keep, keep)]


def run(y, X, gvar, w=None, control="never", method="dripw", event_window=(-5, 5), base_period="varying",
        trim=1.0, boot_weights=None, alpha=0.05, event_sets=None, rules="csdid181"):
    if rules != "csdid201":
        raise ValueError("this baseline follows csdid 2.0.1's rules (rules='csdid201')")
    if method not in METHOD:
        raise ValueError(f"csdid 2.0.1 has no {method} estimator")
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
    cmd = (f"csdid y {xs}{wopt}, ivar(id) time(t) gvar(gvar) method({METHOD[method]}) "
           f"{'notyet' if control == 'notyet' else 'never'} base_period({base_period}) pscoretrim({trim!r}) "
           f"bal(pair) level({100 * (1 - alpha)!r})")
    universal = base_period == "universal"
    lo, hi = event_window

    # analytic run: ATT(g,t), the aggregations and their influence-function standard errors
    stata.run(cmd + " analytical pointwise", quietly=True)
    b, V, names = _post("e")
    cells = []
    for j, nm in enumerate(names):
        m = re.fullmatch(r"g(\d+)___(\d+)_(\d+)", nm.lower())
        if m and np.isfinite(b[j]):
            cells.append((int(m.group(1)), int(m.group(2)), b[j], np.sqrt(V[j, j])))
    cells.sort()
    stata.run("estimates store SSB_CS", quietly=True)
    sb, sV, _ = _aggregation("estat simple")
    gb, gV, gn = _aggregation("estat group")
    group = _keyed(gn, gb, gV)
    cb_, cV, cn = _aggregation("estat calendar")
    calendar = _keyed(cn, cb_, cV)
    et, eb, eV = _event(lo, hi, universal)
    pre, post = et < 0, et >= 0

    def avg(sel_b, sel_V):
        c = np.full(sel_b.size, 1.0 / max(sel_b.size, 1))
        return float(c @ sel_b), float(np.sqrt(c @ sel_V @ c))

    out = dict(gt=np.array([(c[0], c[1]) for c in cells], dtype=np.int64).reshape(-1, 2),
               att_gt=np.array([c[2] for c in cells]), se_gt=np.array([c[3] for c in cells]),
               att_simple=float(sb[0]), se_simple=float(np.sqrt(sV[0, 0])))
    out["group_ids"], out["att_group"], out["se_group"] = _arrays(group)
    out["att_group_avg"], out["se_group_avg"] = float(group["avg"][0]), float(group["avg"][1])
    out["calendar_periods"], out["att_calendar"], out["se_calendar"] = _arrays(calendar)
    out["att_calendar_avg"], out["se_calendar_avg"] = float(calendar["avg"][0]), float(calendar["avg"][1])
    out["event_times"], out["att_event"], out["se_event"] = et, eb, np.sqrt(np.diag(eV))
    out["cov_event"] = eV
    out["att_pre_avg"], out["se_pre_avg"] = avg(eb[pre], eV[np.ix_(pre, pre)]) if pre.any() else (np.nan, np.nan)
    out["att_post_avg"], out["se_post_avg"] = avg(eb[post], eV[np.ix_(post, post)]) if post.any() else (np.nan, np.nan)
    sets = [sorted(set(int(v) for v in s)) for s in (event_sets or [])]
    out["event_sets"] = sets
    if sets:
        alls = [v for s in sets for v in s]
        st, sb2, sV2 = _event(min(alls + [lo]), max(alls + [hi]), universal)
        atts, ses = [], []
        for s in sets:
            sel = np.isin(st, s)
            a_, s_ = avg(sb2[sel], sV2[np.ix_(sel, sel)])
            atts.append(a_)
            ses.append(s_)
        out["att_event_sets"], out["se_event_sets"] = np.array(atts), np.array(ses)
    else:
        out["att_event_sets"], out["se_event_sets"] = np.zeros(0), np.zeros(0)

    # bootstrap run (Stata's own multipliers, as many draws as given)
    if boot_weights is not None:
        B = int(np.asarray(boot_weights).shape[1])
        stata.run(cmd + f" reps({B}) rseed({SEED})", quietly=True)
        b, V, names = _post("e")
        sel = [j for j, nm in enumerate(names) if re.fullmatch(r"g\d+___\d+_\d+", nm.lower()) and np.isfinite(b[j])]
        out["se_boot_gt"] = np.sqrt(np.diag(V)[sel])
        out["crit_gt"] = _escalar("crit_val")
        stata.run("estimates store SSB_CS", quietly=True)
        for name, command in (("simple", "estat simple"), ("group", "estat group"), ("calendar", "estat calendar")):
            bb, VV, nn = _aggregation(command)
            ses = np.sqrt(np.diag(VV))
            crit = _escalar("crit_val")
            if name == "simple":
                out["se_boot_simple"] = float(ses[0])
            else:
                rows = [j for j, n_ in enumerate(nn) if n_.lower() not in ("overall",)]
                out[f"se_boot_{name}"] = ses[rows]
                out[f"crit_{name}"] = crit
                avg_row = [j for j, n_ in enumerate(nn) if n_.lower() == "overall"]
                out[f"se_boot_{name}_avg"] = float(ses[avg_row[0]]) if avg_row else np.nan
        bt, bb, bV = _event(lo, hi, universal)
        out["se_boot_event"] = np.sqrt(np.diag(bV))
        out["crit_event"] = _escalar("crit_val")
        bpre, bpost = bt < 0, bt >= 0
        out["se_boot_pre_avg"] = avg(bb[bpre], bV[np.ix_(bpre, bpre)])[1] if bpre.any() else np.nan
        out["se_boot_post_avg"] = avg(bb[bpost], bV[np.ix_(bpost, bpost)])[1] if bpost.any() else np.nan
    stata.run("clear", quietly=True)
    return out
