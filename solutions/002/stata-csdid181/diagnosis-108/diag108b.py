"""Follow-up to diag108.py: one cell of cohort 25 on 002/108 (ipw), the reference's logit coefficients
and Stata's logit as drdid runs it (estimation sample, omitted covariates, notes).

usage (repository root, nothing else running):
  OMP_NUM_THREADS=1 solutions/baselines/.venv/py/bin/python solutions/002/stata-csdid181/diagnosis-108/diag108b.py
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from harness.common import load_module  # noqa: E402

TASK = Path("002_csdid")
ref = load_module(TASK / "reference.py", "ref002_diag")
rd = load_module(TASK / "realdata.py", "rd002_diag")
d = rd.load(108, "datasets/002/data")
d.pop("boot_weights", None)
cells = []
_logit = ref.logit


def logit(D, X, w, tol=1e-10, maxiter=100):
    index, V = _logit(D, X, w, tol, maxiter)
    cells.append((D.copy(), X.copy(), w.copy(), index.copy()))
    return index, V


ref.logit = logit
R = ref.run(**d)
i = next(j for j, (g, t) in enumerate(R["gt"]) if g == 25 and t == 6)
D, X, w, index = cells[i]
b = np.linalg.lstsq(X, index, rcond=None)[0]
k = X.shape[1] - 1
print(f"cell (25, 6): {D.size} units, {int(D.sum())} treated; reference coefficients (x1..x{k}, constant): {np.round(b, 3)}")
for j in range(k):
    v = X[:, j]
    vals = np.unique(v)
    print(f"  x{j + 1}: {vals.size} distinct values; among the treated {np.unique(v[D == 1])[:6]}; "
          f"controls with a value no treated unit has: {int(np.sum((D == 0) & ~np.isin(v, v[D == 1])))}")
ps = 1 / (1 + np.exp(-index))
print(f"  reference scores of the controls: {np.sum(ps[D == 0] < 1e-8)} below 1e-8, max {ps[D == 0].max():.4f}")

sys.path.append("/Applications/Stata 18/utilities")
from pystata import config  # noqa: E402
config.init("mp", splash=False)
from pystata import stata  # noqa: E402
from sfi import Macro, Matrix, Scalar  # noqa: E402

tmp = tempfile.mkdtemp(prefix="ssb-diag108-")
df = pd.DataFrame({"D": D, "w": w, **{f"x{j + 1}": X[:, j] for j in range(k)}})
df.to_stata(os.path.join(tmp, "cell.dta"), write_index=False, version=118)
stata.run(f'use "{os.path.join(tmp, "cell.dta")}", clear')
stata.run("set processors 1")
stata.run("logit D " + " ".join(f"x{j + 1}" for j in range(k)) + " [iw = w]")
stata.run("matrix SSB_b = e(b)\nglobal SSB_nm : colnames SSB_b\nscalar SSB_N = e(N)", quietly=True)
print(f"Stata: e(N) = {int(Scalar.getValue('SSB_N'))} of {D.size}; e(b) columns {Macro.getGlobal('SSB_nm')}: "
      f"{np.round(np.asarray(Matrix.get('SSB_b'))[0], 3)}")
stata.run("predict double SSB_ps, pr", quietly=True)
stata.run("quietly summarize SSB_ps if D == 0\nscalar SSB_mx = r(max)\nquietly count if D == 0 & SSB_ps < 1e-8\n"
          "scalar SSB_lo = r(N)", quietly=True)
print(f"  drdid's scores (predict after logit): controls below 1e-8: {int(Scalar.getValue('SSB_lo'))}, "
      f"max {Scalar.getValue('SSB_mx'):.4f}")
