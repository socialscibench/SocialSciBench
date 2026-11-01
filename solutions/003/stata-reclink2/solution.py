"""Task 003 baseline: Stata reclink2 (Wasi & Flaaen 2015, Stata Journal dm0082), in-process via pystata.

    use <master>, clear
    reclink2 name addr city zip state using <using.dta>, idmaster(idm) idusing(idu) gen(score) ///
        wmatch(10 8 5 5 5) wnomatch(2 3 4 4 4) required(state)

with the other options at their defaults (task.md). Both record sets are written to temporary .dta
files (pandas `to_stata`, format 118) inside the timed call; reclink2 reads the using file from
disk. Kept pairs are the records with a using id; the perfect-pair count is the number of returned
pairs equal on all five variables (reclink2 scores exactly those 1). Stata MP is limited to the
evaluator's thread count (`set processors`). The ado path is Stata's own plus the downloaded
reclink2 (SOCIALSCIBENCH_RECLINK2).
"""
import os
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


ADO = os.environ.get("SOCIALSCIBENCH_RECLINK2", os.path.join(_venv(), "stata-ado"))
VARS = ["name", "addr", "city", "zip", "state"]
WMATCH = [10, 8, 5, 5, 5]
WNOMATCH = [2, 3, 4, 4, 4]
TMP = tempfile.mkdtemp(prefix="ssb003-stata-")

sys.path.append(os.path.join(STATA_DIR, "utilities"))
from pystata import config  # noqa: E402

config.init("mp", splash=False)
from pystata import stata  # noqa: E402
from sfi import Data  # noqa: E402

THREADS = int(os.environ.get("OMP_NUM_THREADS", "1"))
for line in ("set more off", f"set processors {THREADS}", "set max_memory 28g",
             f'sysdir set PERSONAL "{TMP}"', f'sysdir set PLUS "{TMP}"', f'sysdir set OLDPLACE "{TMP}"',
             f'adopath ++ "{os.path.abspath(ADO)}"', "which reclink2"):
    stata.run(line, quietly=True)


def _write(path, idname, rec):
    n = len(rec[VARS[0]])
    df = pd.DataFrame({idname: np.arange(1, n + 1, dtype=np.int32), **{v: rec[v].astype(object) for v in VARS}})
    df.to_stata(path, write_index=False, version=118)


def run(master, using, vars, wmatch, wnomatch, required, minscore=0.6, minbigram=0.6):
    if (list(vars), list(wmatch), list(wnomatch), list(required), minscore, minbigram) != (
            VARS, WMATCH, WNOMATCH, ["state"], 0.6, 0.6):
        raise ValueError("this adapter runs the task's reclink2 command with the default thresholds")
    m = {v: np.asarray(master[v], dtype=str) for v in VARS}
    u = {v: np.asarray(using[v], dtype=str) for v in VARS}
    mpath, upath = os.path.join(TMP, "master.dta"), os.path.join(TMP, "using.dta")
    _write(mpath, "idm", m)
    _write(upath, "idu", u)
    stata.run(f'use "{mpath}", clear', quietly=True)
    stata.run(f'reclink2 {" ".join(VARS)} using "{upath}", idmaster(idm) idusing(idu) gen(score) '
              f'wmatch({" ".join(map(str, WMATCH))}) wnomatch({" ".join(map(str, WNOMATCH))}) required(state)',
              quietly=True)
    stata.run("keep if !missing(idu)", quietly=True)
    if Data.getObsTotal() == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0), 0
    out = stata.pdataframe_from_data(var=["idm", "idu", "score"])
    i = out["idm"].to_numpy(np.int64) - 1
    j = out["idu"].to_numpy(np.int64) - 1
    score = out["score"].to_numpy(np.float32).astype(np.float64)
    perfect = np.ones(len(i), bool)
    for v in VARS:
        perfect &= m[v][i] == u[v][j]
    stata.run("clear", quietly=True)
    return i, j, score, int(perfect.sum())
