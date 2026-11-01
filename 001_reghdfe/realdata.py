"""Load task-001 inputs described in DATA.md and real_workloads.json.

With this task directory on the Python import path, from the repository root:
    from realdata import load, CATALOG
    inputs = load(101, data_root="datasets/001/data")
    # inputs == dict(y=..., X=..., fe_ids=..., cluster_ids=...)

Development workloads (101-108) store real ID structures; synthetic y and X are added here
with the same construction as the task's generator (X loads on the fixed effects, y = X beta
+ effects + noise), seeded with 1000 + id. Workload 109 is fully synthetic
(synthetic_extra.py) and is only for compatibility/crash checks, never tuning.
Held-out workloads (201-206, 301-304) carry real y and X and require frozen final evaluation.
Use the shared evaluator for split enforcement and file checksum verification.
"""
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from harness.common import load_module

synthetic_extra = load_module(Path(__file__).with_name("synthetic_extra.py"), "ssb_reghdfe_synthetic_extra")

# id -> role (development, smoke, heldout), absorbed columns and regressor count
CATALOG = {
    101: dict(role="development", absorb=["fe_inventor", "fe_firm", "fe_year"]),
    102: dict(role="development", absorb=["fe_inventor", "fe_firm", "fe_year"]),
    103: dict(role="development", absorb=["fe_inventor", "fe_firm", "fe_year"]),
    104: dict(role="development", absorb=["fe_inventor", "fe_firm", "fe_year"]),
    105: dict(role="development", absorb=["fe_inventor", "fe_firm", "fe_year", "fe_subclass_year"]),
    106: dict(role="development", absorb=["fe_firm", "fe_year"], n_regressors=15),
    107: dict(role="development", absorb=["fe_citing_year", "fe_cited_year", "fe_pair"], n_regressors=10),
    108: dict(role="development", absorb=["fe_inventor", "fe_year", "fe_firm"]),
    109: dict(role="smoke"),
    201: dict(role="heldout", absorb=["fe_inventor", "fe_firm", "fe_year"]),
    202: dict(role="heldout", absorb=["fe_inventor", "fe_firm", "fe_year", "fe_subclass_year"]),
    203: dict(role="heldout", absorb=["fe_firm", "fe_year", "fe_class_year", "fe_region_year"]),
    204: dict(role="heldout", absorb=["fe_citing_year", "fe_cited_year", "fe_pair"]),
    205: dict(role="heldout", absorb=["fe_inventor", "fe_year", "fe_firm"]),
    206: dict(role="heldout", absorb=["fe_inventor", "fe_firm", "fe_year"]),
    301: dict(role="heldout", absorb=["fe_insider", "fe_firm", "fe_year"]),
    302: dict(role="heldout", absorb=["fe_firm", "fe_year", "fe_state_year"]),
    303: dict(role="heldout", absorb=["fe_firm_i_year", "fe_firm_j_year", "fe_pair"]),
    304: dict(role="heldout", absorb=["fe_firm", "fe_year"]),
}


def _spec(wid):
    if not isinstance(wid, (int, np.integer)) or wid not in CATALOG:
        raise ValueError(f"Unknown real workload ID for task 001: {wid!r}")
    return CATALOG[wid]


def _synthetic_values(rng, fe_ids, k, extra_X=None):
    """X loads on the fixed effects; y = X beta + effects + noise (as in workloads.py)."""
    n = fe_ids.shape[0]
    eff = np.zeros(n)
    for g in range(fe_ids.shape[1]):
        eff += rng.standard_normal(fe_ids[:, g].max() + 1)[fe_ids[:, g]]
    X = np.column_stack([0.5 * eff + rng.standard_normal(n) for _ in range(k)])
    if extra_X is not None:
        X = np.column_stack([extra_X, X])
    beta = np.arange(1, X.shape[1] + 1) / 10.0
    y = X @ beta + eff + rng.standard_normal(n)
    return y, X


def load(wid, data_root, n_regressors=None, absorb=None):
    """Return task-001 inputs for workload `wid`.

    The only official override is n_regressors=5, 20 or 50 for workload 104.
    Broader n_regressors/absorb overrides are retained as package utilities;
    they do not define scored benchmark variants. See DATA.md and use the
    evaluator to select official cases. Workload 109 needs no data files."""
    spec = _spec(wid)
    if wid == 109:
        return synthetic_extra.workload_109("network")
    folder = Path(data_root) / ("development" if spec["role"] == "development" else "heldout")
    table = pq.read_table(folder / f"{wid}.parquet")
    cols = table.column_names
    absorb = absorb or spec["absorb"]
    fe_ids = np.column_stack([table[c].to_numpy().astype(np.int64) for c in absorb])
    cluster_ids = table["cluster"].to_numpy().astype(np.int64)
    if spec["role"] == "development":
        rng = np.random.default_rng(1000 + wid)
        k = n_regressors or spec.get("n_regressors", 20)
        dummies = [c for c in cols if c.startswith("d_")]
        extra = np.column_stack([table[c].to_numpy().astype(np.float64) for c in dummies]) if dummies else None
        y, X = _synthetic_values(rng, fe_ids, k, extra)
    else:
        xcols = [c for c in cols if c.startswith("d_")] + sorted(c for c in cols if c.startswith("x"))
        X = np.column_stack([table[c].to_numpy().astype(np.float64) for c in xcols])
        y = table["y"].to_numpy().astype(np.float64)
    return dict(y=y, X=X, fe_ids=fe_ids, cluster_ids=cluster_ids)


def manifest(wid, data_root):
    """Read a file-backed workload's manifest; generated workload 109 has none."""
    spec = _spec(wid)
    if spec["role"] == "smoke":
        raise ValueError("Workload 109 is generated compatibility data and has no file manifest")
    folder = Path(data_root) / ("development" if spec["role"] == "development" else "heldout")
    return json.loads((folder / f"manifest_{wid}.json").read_text())
