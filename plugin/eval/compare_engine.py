"""Compare the results that bench_engine.py saved for two versions (or variants) of the engine.

usage: compare_engine.py DIR_A DIR_B [--variant-a speed] [--variant-b speed]
For every kit, size and call saved in both: 001, the largest relative difference of the coefficients
and standard errors, R2's absolute difference and whether N and the dropped regressors agree; 002, the
same for the estimates and analytic SEs, and for the bootstrap SEs and critical values; 003, whether the
pairs are identical and the largest score difference.
"""
import argparse
import pickle
import re
from pathlib import Path

import numpy as np


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    both = np.isnan(a) & np.isnan(b)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.abs(a - b) / np.maximum(np.abs(b), 1e-300)
    d = np.where(both, 0.0, d)
    return float(np.nanmax(np.where(np.isnan(a) != np.isnan(b), np.inf, d))) if d.size else 0.0


def compare(kit, a, b):
    if kit == "001":
        ca, cb = a["coef"], b["coef"]
        return dict(N=a["nobs"] == b["nobs"], dropped=bool((ca["dropped"] == cb["dropped"]).all()),
                    coef=rel(ca["coef"], cb["coef"]), se=rel(ca["se"], cb["se"]), r2=abs(a["r2"] - b["r2"]))
    if kit == "002":
        out = {}
        for t in ("gt", "event", "averages"):
            out[f"{t}_att"] = rel(a[t]["att"], b[t]["att"])
            out[f"{t}_se"] = rel(a[t]["se"], b[t]["se"])
            if "se_boot" in a[t] and "se_boot" in b[t]:
                out[f"{t}_se_boot"] = rel(a[t]["se_boot"], b[t]["se_boot"])
        for c in set(a["crit"]) & set(b["crit"]):
            out[f"crit_{c}"] = rel(a["crit"][c], b["crit"][c])
        return out
    same = a["i"].size == b["i"].size and np.array_equal(a["i"], b["i"]) and np.array_equal(a["j"], b["j"])
    return dict(pairs=int(a["i"].size), same_pairs=same, n_perfect=a["n_perfect"] == b["n_perfect"],
                score=float(np.max(np.abs(a["score"] - b["score"]))) if same and a["i"].size else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--variant-a", default=None)
    ap.add_argument("--variant-b", default=None)
    args = ap.parse_args()
    pat = re.compile(r"(\d{3})-(\d+)-(\d+)-(\w+)\.pkl")
    for pa in sorted(Path(args.a).glob("*.pkl")):
        m = pat.fullmatch(pa.name)
        if not m or (args.variant_a and m.group(4) != args.variant_a):
            continue
        kit, size, call, var = m.groups()
        pb = Path(args.b) / f"{kit}-{size}-{call}-{args.variant_b or var}.pkl"
        if not pb.exists():
            continue
        a, b = (pickle.loads(p.read_bytes()) for p in (pa, pb))
        res = compare(kit, a, b)
        print(f"{kit} size {size:>3} call {call} {var}->{args.variant_b or var}: " +
              ", ".join(f"{k}={v:.2e}" if isinstance(v, float) else f"{k}={v}" for k, v in res.items()))


if __name__ == "__main__":
    main()
