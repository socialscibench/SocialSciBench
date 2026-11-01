"""Task 002 competitor: Stata csdid 2.0.0 (Mata rewrite), in-process via pystata (Stata 18 MP).

Follows the 14 October 2026 study's Stata extension block (research/2026-10-14/
native-extension/task002/stata_benchmark.do.in in the companion repository): the pinned
official release (pedrohcgs/csdid-stata commit 31a5773) loaded from the study's vendored
source, one processor, and

    csdid y x1 ..., ivar(id) time(t) gvar(g) method(dr) nevertreated base_period(varying) ///
        analytical pointwise
    estat simple
    estat event, post

The pre-period average and its standard error come from the native event covariance, as in the
study. The study timed the Stata commands on data already loaded from a .dta file; here the
long data are built and copied into Stata inside the timed call.
"""
import os
import sys

import numpy as np
import pandas as pd

STATA_DIR = os.environ.get("SOCIALSCIBENCH_STATA", "/Applications/Stata 18")
CSDID_PKG = os.environ.get(
    "SOCIALSCIBENCH_CSDID_PKG",
    "/path/to/repro-socialscibench/research/2026-10-14/native-extension/task002/"
    "vendor/pedrohcgs-csdid-stata-31a5773/pkg")
EMPTY = os.environ.get(
    "SOCIALSCIBENCH_STATA_EMPTY",
    "/path/to/repro-socialscibench/research/2026-10-14/native-extension/task002/cache/empty")

sys.path.append(os.path.join(STATA_DIR, "utilities"))
from pystata import config  # noqa: E402

config.init("mp", splash=False)
from pystata import stata  # noqa: E402
from sfi import Matrix  # noqa: E402

for line in ("set more off", "set processors 1", "set max_memory 24g",
             f'sysdir set PERSONAL "{EMPTY}"', f'sysdir set PLUS "{EMPTY}"',
             f'sysdir set OLDPLACE "{EMPTY}"', f'adopath ++ "{CSDID_PKG}"'):
    stata.run(line, quietly=True)


def run(y, X, gvar):
    n, T = y.shape
    k = X.shape[1]
    cols = dict(id=np.repeat(np.arange(1, n + 1), T), t=np.tile(np.arange(1, T + 1), n),
                y=np.asarray(y, dtype=float).ravel(), g=np.repeat(gvar, T))
    cols.update({f"x{j + 1}": np.repeat(X[:, j], T) for j in range(k)})
    stata.run("clear", quietly=True)
    stata.pdataframe_to_data(pd.DataFrame(cols), force=True)
    xs = " ".join(f"x{j + 1}" for j in range(k))
    stata.run(f"csdid y {xs}, ivar(id) time(t) gvar(g) method(dr) nevertreated "
              "base_period(varying) analytical pointwise", quietly=True)
    cells = np.array(Matrix.get("e(attgt)"), dtype=float)
    stata.run("estat simple", quietly=True)
    simple = np.array(Matrix.get("e(aggte)"), dtype=float).reshape(-1)
    stata.run("estat event, post", quietly=True)
    events = np.array(Matrix.get("e(aggte)"), dtype=float)
    V = np.array(Matrix.get("e(V)"), dtype=float)
    cells = cells[np.lexsort((cells[:, 1], cells[:, 0]))]
    pre = events[:, 0] < 0
    return dict(
        gt=cells[:, :2].astype(np.int64), att_gt=cells[:, 3], se_gt=cells[:, 4],
        att_simple=float(simple[1]), se_simple=float(simple[2]),
        event_times=events[:, 0].astype(np.int64), att_event=events[:, 1], se_event=events[:, 2],
        att_pre_avg=float(events[pre, 1].mean()),
        se_pre_avg=float(np.sqrt(V[np.ix_(pre, pre)].sum()) / pre.sum()),
        att_post_avg=float(events[0, 3]), se_post_avg=float(events[0, 4]),
    )
