"""Task 008 competitor: Stata reclink2 (Wasi & Flaaen 2015, Stata Journal dm0082), in-process via
pystata (Stata 18 MP, one processor).

    reclink2 name addr city zip state using using.dta, idmaster(idm) idusing(idu) gen(score) ///
        wmatch(10 8 5 5 5) wnomatch(2 3 4 4 4) required(state)

with the other options at their defaults, as in task.md and baselines/008_recordlinkage/bench.py.
reclink2 reads its using file from disk, so the using records are copied into Stata and saved to a
temporary .dta inside the timed call (the baseline wrote the file before starting its timer); the
master records are copied into Stata. The kept pairs are the records with a using id. The perfect-
pair count is the number of returned pairs whose records are equal on all five fields (reclink2
gives exactly these score 1 and removes them before fuzzy matching).
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd

STATA_DIR = os.environ.get("SOCIALSCIBENCH_STATA", "/Applications/Stata 18")
RECLINK2 = os.environ.get("SOCIALSCIBENCH_RECLINK2", os.path.join(sys.prefix, "stata-ado"))
VARS = ["name", "addr", "city", "zip", "state"]
WMATCH = [10, 8, 5, 5, 5]
WNOMATCH = [2, 3, 4, 4, 4]
REQUIRED = ["state"]
TMP = tempfile.mkdtemp(prefix="ssb008-stata-")
EMPTY = os.path.join(TMP, "empty")
os.makedirs(EMPTY)

sys.path.append(os.path.join(STATA_DIR, "utilities"))
from pystata import config  # noqa: E402

config.init("mp", splash=False)
from pystata import stata  # noqa: E402
from sfi import Data  # noqa: E402

for line in ("set more off", "set processors 1", "set max_memory 24g",
             f'sysdir set PERSONAL "{EMPTY}"', f'sysdir set PLUS "{EMPTY}"', f'sysdir set OLDPLACE "{EMPTY}"',
             f'adopath ++ "{RECLINK2}"', "which reclink2"):
    stata.run(line, quietly=True)


def run(master, using, vars, wmatch, wnomatch, required, minscore=0.6, minbigram=0.6):
    if (list(vars), list(wmatch), list(wnomatch), list(required), minscore, minbigram) != (
            VARS, WMATCH, WNOMATCH, REQUIRED, 0.6, 0.6):
        raise ValueError("this adapter runs the task's reclink2 command with the default thresholds")
    m = {v: np.asarray(master[v], dtype=str) for v in VARS}
    u = {v: np.asarray(using[v], dtype=str) for v in VARS}
    nm, nu = len(m[VARS[0]]), len(u[VARS[0]])
    path = os.path.join(TMP, "using.dta")
    stata.run("clear", quietly=True)
    stata.pdataframe_to_data(pd.DataFrame({"idu": np.arange(1, nu + 1, dtype=np.int32),
                                           **{v: u[v].astype(object) for v in VARS}}), force=True)
    stata.run(f'save "{path}", replace', quietly=True)
    stata.run("clear", quietly=True)
    stata.pdataframe_to_data(pd.DataFrame({"idm": np.arange(1, nm + 1, dtype=np.int32),
                                           **{v: m[v].astype(object) for v in VARS}}), force=True)
    stata.run(f'reclink2 {" ".join(VARS)} using "{path}", idmaster(idm) idusing(idu) gen(score) '
              f'wmatch({" ".join(map(str, WMATCH))}) wnomatch({" ".join(map(str, WNOMATCH))}) '
              f'required({" ".join(REQUIRED)})', quietly=True)
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
    return i, j, score, int(perfect.sum())
