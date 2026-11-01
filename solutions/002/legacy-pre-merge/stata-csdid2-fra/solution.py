"""Task 002 baseline: Stata csdid2 1.3 (F. Rios-Avila; the Mata rewrite of csdid), in-process via pystata
(Stata 18 MP). The command the task gives as csdid's equivalent:

    csdid2 y x1 ... xk, ivar(id) time(t) gvar(g) method(dripw) short    // never-treated controls
    estat attgt
    estat simple
    estat event                                                         // Pre_avg, Post_avg, tm#, tp#

`short` selects csdid's varying base period (csdid2's default is the long one). csdid2 1.3 has no
propensity-score trimming option (csdid's pscoretrim), so comparison units with a propensity score
above .995 keep their weight; results differ from the task's where that rule binds.
The wide inputs are reshaped to a long panel and written to a temporary .dta file inside the timed
call; every result comes from r(b)/r(V) of estat (ATT(g,t): equation g<cohort>, column
t<base>_<t>). Stata MP is limited to the evaluator's thread count. The ado path is Stata's own
plus the downloaded package (SOCIALSCIBENCH_CSDID2_ADO).
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


PKG = os.environ.get("SOCIALSCIBENCH_CSDID2_ADO",
                     os.path.join(_venv(), "stata-src", "csdid-dl", "stpackages-main", "csdid2"))
TMP = tempfile.mkdtemp(prefix="ssb002-stata-")

sys.path.append(os.path.join(STATA_DIR, "utilities"))
from pystata import config  # noqa: E402

config.init("mp", splash=False)
from pystata import stata  # noqa: E402
from sfi import Macro, Matrix  # noqa: E402

THREADS = int(os.environ.get("OMP_NUM_THREADS", "1"))
for line in ("set more off", f"set processors {THREADS}", "set max_memory 28g",
             f'sysdir set PERSONAL "{TMP}"', f'sysdir set PLUS "{TMP}"', f'sysdir set OLDPLACE "{TMP}"',
             f'adopath ++ "{os.path.abspath(PKG)}"', "mata: mata mlib index", "which csdid2"):
    stata.run(line, quietly=True)

NUM = r"(\d+(?:\.0*)?)"


def _matrix(kind):
    """b, V, column equations and column names of e(b)/e(V) or r(b)/r(V)."""
    stata.run(f"matrix SSB_b = {kind}(b)\nmatrix SSB_V = {kind}(V)\n"
              "global SSB_eq : coleq SSB_b\nglobal SSB_nm : colnames SSB_b", quietly=True)
    b = np.asarray(Matrix.get("SSB_b"), dtype=np.float64)[0]
    V = np.asarray(Matrix.get("SSB_V"), dtype=np.float64)
    return b, V, Macro.getGlobal("SSB_eq").split(), Macro.getGlobal("SSB_nm").split()


def run(y, X, gvar):
    y = np.asarray(y, dtype=np.float64)
    N, T = y.shape
    X = np.asarray(X, dtype=np.float64).reshape(N, -1)
    k = X.shape[1]
    g = np.asarray(gvar, dtype=np.int64)
    cols = {"id": np.tile(np.arange(1, N + 1), T), "t": np.repeat(np.arange(1, T + 1), N),
            "g": np.tile(g, T), "y": y.T.ravel()}
    for j in range(k):
        cols[f"x{j + 1}"] = np.tile(X[:, j], T)
    path = os.path.join(TMP, "panel.dta")
    pd.DataFrame(cols).to_stata(path, write_index=False, version=118)
    stata.run(f'use "{path}", clear', quietly=True)
    xs = " ".join(f"x{j + 1}" for j in range(k))
    stata.run(f"csdid2 y {xs}, ivar(id) time(t) gvar(g) method(dripw) short", quietly=True)
    stata.run("estat attgt", quietly=True)
    b, V, eqs, names = _matrix("r")
    cells = []
    for e, nm, bi, vi in zip(eqs, names, b, np.diag(V)):
        m1, m2 = re.fullmatch(r"g" + NUM, e), re.fullmatch(r"t" + NUM + "_" + NUM, nm)
        if m1 and m2:
            cells.append((int(float(m1.group(1))), int(float(m2.group(2))), bi, np.sqrt(vi)))
    cells.sort()
    stata.run("estat simple", quietly=True)
    bs, Vs, _, _ = _matrix("r")
    stata.run("estat event", quietly=True)
    be, Ve, _, ne = _matrix("r")
    se_e = np.sqrt(np.diag(Ve))
    ev = {}
    for nm, bi, si in zip(ne, be, se_e):
        m = re.fullmatch(r"[Tt]([mp])" + NUM, nm)
        if m:
            ev[(-1 if m.group(1) == "m" else 1) * int(float(m.group(2)))] = (bi, si)
        else:
            ev[nm] = (bi, si)
    et = np.array(sorted(k for k in ev if isinstance(k, int)), dtype=np.int64)
    stata.run("clear", quietly=True)
    return dict(gt=np.array([(c[0], c[1]) for c in cells], dtype=np.int64).reshape(-1, 2),
                att_gt=np.array([c[2] for c in cells]), se_gt=np.array([c[3] for c in cells]),
                att_simple=float(bs[0]), se_simple=float(np.sqrt(Vs[0, 0])),
                event_times=et, att_event=np.array([ev[e][0] for e in et]), se_event=np.array([ev[e][1] for e in et]),
                att_pre_avg=float(ev["Pre_avg"][0]), se_pre_avg=float(ev["Pre_avg"][1]),
                att_post_avg=float(ev["Post_avg"][0]), se_post_avg=float(ev["Post_avg"][1]))
