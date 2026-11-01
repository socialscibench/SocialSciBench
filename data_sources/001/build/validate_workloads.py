"""Load every real-data workload, check it, compute descriptors, and solve it with the reference.

For each workload: finite values, contiguous codes, rows before/after singleton removal,
fixed-effect levels, clusters, person-organization mobility and connectivity (for network
designs), then reference.run() with its wall time. Reference outputs are saved as
OUT_DIR/reference_outputs/<id>.npz and a summary line is appended to OUT_DIR/validation.jsonl.

usage: python validate_workloads.py DATA_ROOT TASK_DIR [ids ...]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ORDER = [109, 101, 302, 304, 106, 203, 102, 303, 301, 103, 107, 204, 104, 105, 201, 202, 108, 205, 206]
NETWORK = {101: (0, 1), 102: (0, 1), 103: (0, 1), 104: (0, 1), 105: (0, 1), 201: (0, 1), 202: (0, 1),
           206: (0, 1), 108: (0, 2), 205: (0, 2), 301: (0, 1), 109: (0, 1)}


def descriptors(wid, d, reference):
    fe = d["fe_ids"]
    keep = reference.drop_singletons(np.column_stack([reference.recode(fe[:, g]) for g in range(fe.shape[1])]))
    out = dict(rows=int(fe.shape[0]), rows_after_singletons=int(keep.sum()),
               fe_levels=[int(np.unique(fe[keep, g]).size) for g in range(fe.shape[1])],
               n_clusters=int(np.unique(d["cluster_ids"][keep]).size), n_regressors=int(d["X"].shape[1]))
    if wid in NETWORK:
        p, o = NETWORK[wid]
        pc, oc = reference.recode(fe[keep, p]), reference.recode(fe[keep, o])
        pairs = np.unique(np.stack([pc, oc], 1), axis=0)
        per_person = np.bincount(pairs[:, 0])
        out["share_persons_2plus_orgs"] = float((per_person >= 2).mean())
        P, O = pc.max() + 1, oc.max() + 1
        g = coo_matrix((np.ones(pairs.shape[0]), (pairs[:, 0], pairs[:, 1] + P)), shape=(P + O, P + O))
        ncomp, lab = connected_components(g, directed=False)
        sizes = np.bincount(lab[pc])
        out["components"] = int(ncomp)
        out["share_rows_largest_component"] = float(sizes.max() / pc.size)
    return out


def main(data_root, task_dir, ids):
    sys.path.insert(0, str(Path(task_dir)))
    import reference
    import realdata
    out_dir = Path(data_root)
    (out_dir / "reference_outputs").mkdir(exist_ok=True)
    for wid in ids or ORDER:
        rec = dict(id=wid)
        try:
            t0 = time.perf_counter()
            d = realdata.load(wid, data_root)
            rec["load_s"] = round(time.perf_counter() - t0, 2)
            assert np.isfinite(d["y"]).all() and np.isfinite(d["X"]).all(), "non-finite values"
            rec.update(descriptors(wid, d, reference))
            t0 = time.perf_counter()
            r = reference.run(**d)
            rec["reference_s"] = round(time.perf_counter() - t0, 2)
            se = np.sqrt(np.diag(r["vcov"]))
            rec.update(nobs=int(r["nobs"]), df_a=int(r["df_a"]), status="ok",
                       beta_finite=bool(np.isfinite(r["beta"]).all()), se_finite=bool(np.isfinite(se).all()),
                       min_se=float(se.min()), max_abs_beta=float(np.abs(r["beta"]).max()))
            np.savez(out_dir / "reference_outputs" / f"{wid}.npz", beta=r["beta"], vcov=r["vcov"],
                     nobs=r["nobs"], df_a=r["df_a"], n_clusters=r["n_clusters"])
        except Exception as e:  # report and continue
            rec.update(status="error", error=f"{type(e).__name__}: {e}")
        print(json.dumps(rec), flush=True)
        with open(out_dir / "validation.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3:]])
