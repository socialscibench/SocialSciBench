"""Load the workloads of the combined task 002 as reference.run() inputs.

    from realdata import load
    inputs = load(101, data_root="datasets/002/data")
    # dict(y, X, gvar, w, control, method, event_window, base_period, trim, boot_weights,
    #      event_sets, rules)

Each data file has one row per unit: `unit`, `gvar` (first treated period, 0 = never treated),
covariates, an optional sampling weight `w`, and the outcome in y001 ... yT (NaN where the
unit is not observed). Covariates that vary over time (109, 404) are stored one column per
period, <name>_t001 ... <name>_tT, and loaded as X [n_units, n_periods, n_covariates]. A workload is a file plus the settings in WORKLOADS. Bootstrap
multipliers are generated here from a fixed seed (NumPy PCG64 seeded with the workload number,
Rademacher +1/-1), so every run gets the same draws; generating them is not part of the timed call.
Workload IDs are integers (101-109 development, 301, 302 and 401-404 held out). Use the shared
evaluator for split enforcement and file checksum verification; held-out workloads require
frozen final evaluation.
"""
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

WINDOW = (-5, 5)

# id: (file, control, method, weighted, base_period, trim, bootstrap draws (0 = analytic), event_sets, rules)
WORKLOADS = {
    # development: PatentsView, technology areas A-F
    101: ("101", "notyet", "dripw", False, "universal", 1.0, 999, None, "csdid201"),
    102: ("102", "never", "dripw", True, "varying", 1.0, 0, None, "csdid181"),
    103: ("103", "never", "reg", False, "universal", 1.0, 0, [[1, 2], [1, 2, 3, 4, 5], [-5, -4, -3, -2]], "csdid201"),
    104: ("104", "never", "dripw", False, "varying", 1.0, 0, None, "csdid201"),
    105: ("105", "notyet", "stdipw", False, "varying", 1.0, 0, None, "csdid201"),
    106: ("101", "never", "reg", False, "varying", 1.0, 0, None, "csdid181"),       # 101 without covariates
    107: ("107", "never", "dripw", False, "varying", 0.995, 999, None, "csdid201"),  # headline, full size
    108: ("108", "notyet", "ipw", False, "universal", 1.0, 25000, None, "csdid181"),
    109: ("109", "notyet", "dripw", False, "universal", 0.995, 0, None, "csdid181"),  # covariates vary over time
    # held out: SEC insider filings (3xx) and QCEW county employment with state minimum wages (4xx)
    301: ("301", "notyet", "dripw", False, "varying", 0.995, 0, None, "csdid201"),
    302: ("302", "never", "ipw", False, "universal", 1.0, 0, [[1, 2], [1, 2, 3, 4, 5], [-5, -4, -3, -2]], "csdid181"),
    401: ("401", "notyet", "dripw", True, "universal", 1.0, 25000, None, "csdid201"),
    402: ("402", "notyet", "stdipw", True, "varying", 1.0, 999, None, "csdid201"),   # headline, full size
    403: ("403", "never", "reg", True, "varying", 1.0, 999, None, "csdid181"),
    404: ("404", "notyet", "dripw", True, "universal", 0.995, 25000, None, "csdid181"),  # covariates vary over time
}
DEVELOPMENT = [k for k in WORKLOADS if k < 200]
HELDOUT = [k for k in WORKLOADS if k > 300]
NO_COVARIATES = {106}


def _path(wid, data_root):
    if not isinstance(wid, (int, np.integer)) or int(wid) not in WORKLOADS:
        raise ValueError(f"Unknown real workload ID for task 002: {wid!r}")
    folder = "development" if int(wid) in DEVELOPMENT else "heldout"
    return Path(data_root) / folder / f"{WORKLOADS[int(wid)][0]}.parquet"


def boot_weights(n_units, draws, seed):
    """Rademacher multipliers [n_units, draws] (int8, +1/-1) from NumPy's PCG64 with a fixed seed."""
    rng = np.random.Generator(np.random.PCG64(seed))
    return (rng.integers(0, 2, size=(n_units, draws), dtype=np.int8) * 2 - 1).astype(np.int8)


def load(wid, data_root, with_boot=True):
    path = _path(wid, data_root)
    wid = int(wid)
    _, control, method, weighted, base_period, trim, draws, sets, rules = WORKLOADS[wid]
    table = pq.read_table(path)
    cols = table.column_names
    ycols = sorted(c for c in cols if c.startswith("y") and c[1:].isdigit())
    xcols = [] if wid in NO_COVARIATES else [c for c in cols if c not in ("unit", "gvar", "w") and c not in ycols]
    y = np.column_stack([table[c].to_numpy(zero_copy_only=False).astype(np.float64) for c in ycols])
    varying = any(c.rsplit("_t", 1)[-1].isdigit() for c in xcols)
    if varying:   # <name>_t001 ... _tT -> [n_units, n_periods, n_covariates]
        names = list(dict.fromkeys(c.rsplit("_t", 1)[0] for c in xcols))
        X = np.stack([np.column_stack([table[f"{nm}_t{t + 1:03d}"].to_numpy().astype(np.float64)
                                       for t in range(len(ycols))]) for nm in names], axis=2)
    elif xcols:
        X = np.column_stack([table[c].to_numpy().astype(np.float64) for c in xcols])
    else:
        X = np.zeros((y.shape[0], 0))
    gvar = table["gvar"].to_numpy().astype(np.int64)
    w = table["w"].to_numpy().astype(np.float64) if weighted else None
    v = boot_weights(len(gvar), draws, wid) if (draws and with_boot) else None
    return dict(y=y, X=X, gvar=gvar, w=w, control=control, method=method, event_window=WINDOW,
                base_period=base_period, trim=trim, boot_weights=v, event_sets=sets, rules=rules)


def manifest(wid, data_root):
    p = _path(wid, data_root)
    return json.loads((p.parent / f"manifest_{p.stem}.json").read_text())
