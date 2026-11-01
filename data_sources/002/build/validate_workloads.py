"""Check every task-002 workload and solve it with the reference implementation.

For each workload: finite values, gvar codes valid (0 or 2..T), at least one never-treated
unit, cohort sizes, then reference.run() with its wall time. Reference outputs are saved as
DATA_ROOT/reference_outputs/<id>.npz; one summary line per workload goes to validation.jsonl.

usage: python validate_workloads.py DATA_ROOT CODE_DIR [--no-reference] [ids ...]
       --no-reference runs the data checks only (no model is estimated).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ORDER = [106, 101, 108, 102, 103, 105, 107, 104, 301, 401, 402, 403, 203, 202, 201]


def main(data_root, code_dir, ids, run_reference=True):
    sys.path.insert(0, str(Path(code_dir)))
    import realdata
    import reference
    out = Path(data_root)
    (out / "reference_outputs").mkdir(exist_ok=True)
    for wid in ids or ORDER:
        rec = dict(id=wid)
        try:
            d = realdata.load(wid, data_root)
            y, X, g = d["y"], d["X"], d["gvar"]
            n, T = y.shape
            assert np.isfinite(y).all() and np.isfinite(X).all(), "non-finite values"
            assert ((g == 0) | ((g >= 2) & (g <= T))).all(), "invalid gvar"
            assert (g == 0).any(), "no never-treated units"
            cohorts, sizes = np.unique(g[g > 0], return_counts=True)
            rec.update(n_units=n, n_periods=T, n_covariates=X.shape[1], n_cohorts=int(cohorts.size),
                       n_never_treated=int((g == 0).sum()), share_treated=round(float((g > 0).mean()), 4),
                       cohort_size_min=int(sizes.min()), cohort_size_max=int(sizes.max()),
                       n_gt=int(cohorts.size * (T - 1)))
            if not run_reference:
                rec["status"] = "data ok"
                print(json.dumps(rec), flush=True)
                continue
            t0 = time.perf_counter()
            r = reference.run(**d)
            rec["reference_s"] = round(time.perf_counter() - t0, 2)
            finite = all(np.isfinite(np.asarray(v, dtype=float)).all() for k, v in r.items() if k != "gt")
            rec.update(status="ok" if finite else "non-finite output", att_simple_finite=bool(np.isfinite(r["att_simple"])))
            np.savez(out / "reference_outputs" / f"{wid}.npz", **{k: np.asarray(v) for k, v in r.items()})
        except Exception as e:
            rec.update(status="error", error=f"{type(e).__name__}: {e}")
        print(json.dumps(rec), flush=True)
        with open(out / "validation.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    args = sys.argv[3:]
    main(sys.argv[1], sys.argv[2], [int(x) for x in args if x != "--no-reference"], "--no-reference" not in args)
