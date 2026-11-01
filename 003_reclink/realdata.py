"""Load task-003 inputs described in DATA.md and real_workloads.json.

With this task directory on the Python import path, from the repository root:
    from realdata import load
    inputs = load(102, data_root="datasets/003/data")
    # inputs == dict(master={name, addr, city, zip, state}, using={...}, vars, wmatch, wnomatch,
    #                required, minscore, minbigram)   -> reference.run(**inputs)

Strings are uppercase ASCII with "" for missing values, as the task requires. The record
identifiers of the sources (PPP LoanNumber, IRS EIN, OSHA activity_nr, EPA REGISTRY_ID) are
in the files' `id` column; reference.run() returns zero-based row positions.
Use the shared evaluator for split enforcement and file checksum verification;
held-out workloads require frozen final evaluation.
"""
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from harness.common import load_module

_constants = load_module(Path(__file__).with_name("workloads.py"), "ssb_reclink_constants")
MINBIGRAM, MINSCORE, REQUIRED, VARS, WMATCH, WNOMATCH = (
    getattr(_constants, name) for name in ("MINBIGRAM", "MINSCORE", "REQUIRED", "VARS", "WMATCH", "WNOMATCH"))

DEVELOPMENT = [101, 102, 103, 104, 105, 106, 107, 108]
HELDOUT = [201, 202, 203, 204, 205]


def _folder(wid, data_root):
    if not isinstance(wid, (int, np.integer)) or wid not in DEVELOPMENT + HELDOUT:
        raise ValueError(f"Unknown real workload ID for task 003: {wid!r}")
    return Path(data_root) / ("development" if wid in DEVELOPMENT else "heldout")


def _read(path):
    t = pq.read_table(path)
    return {v: np.asarray(t[v].to_pylist(), dtype=object).astype(str) for v in VARS}


def load(wid, data_root):
    d = _folder(wid, data_root)
    return dict(master=_read(d / f"{wid}_master.parquet"), using=_read(d / f"{wid}_using.parquet"),
                vars=list(VARS), wmatch=list(WMATCH), wnomatch=list(WNOMATCH), required=list(REQUIRED),
                minscore=MINSCORE, minbigram=MINBIGRAM)


def ids(wid, data_root):
    """Source record identifiers of the master and using rows, in row order."""
    d = _folder(wid, data_root)
    return (pq.read_table(d / f"{wid}_master.parquet", columns=["id"])["id"].to_pylist(),
            pq.read_table(d / f"{wid}_using.parquet", columns=["id"])["id"].to_pylist())


def manifest(wid, data_root):
    return json.loads((_folder(wid, data_root) / f"manifest_{wid}.json").read_text())
