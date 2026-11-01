"""Random small problems against the reference (development check, not part of the candidate).

Run from the repository root:  python solutions/008/bigram-table-scan/fuzz.py [n_cases]

Values come from tiny alphabets and short words, so short strings (fewer than 3 characters),
empty values (in every field, including the required one), exact duplicates in both files,
shared bigrams and tied scores are common. Most cases use the task's settings; some use other
variable lists, required() fields and weights, which the reference also supports (except three
or fewer variables with no required() field, where its join has no keys).
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "008_recordlinkage"))
import reference  # noqa: E402
import workloads  # noqa: E402

spec = importlib.util.spec_from_file_location("cand", Path(__file__).with_name("solution.py"))
cand = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cand)


def value(rng, alphabet, p_empty):
    if rng.random() < p_empty:
        return ""
    n = int(rng.choice([1, 2, 2, 3, 3, 4, 5, 6, 8, 11]))
    return "".join(rng.choice(alphabet, n))


def case(rng):
    standard = rng.random() < 0.7
    if standard:
        vars_, req = list(workloads.VARS), list(workloads.REQUIRED)
        wm, wnm = list(workloads.WMATCH), list(workloads.WNOMATCH)
    else:
        F = int(rng.integers(2, 6))
        vars_ = [f"v{k}" for k in range(F)]
        req = [v for v in vars_ if rng.random() < 0.3]
        if F <= 3 and not req:                   # the reference cannot join on no variables
            req = [vars_[-1]]
        wm = [int(x) for x in rng.integers(1, 12, F)]
        wnm = [int(x) for x in rng.integers(1, 6, F)]
    alphabets = [np.array(list(rng.choice(list("ABCDE 12.,"), int(rng.integers(1, 6)), replace=False)))
                 for _ in vars_]
    nu = int(rng.integers(1, 80))
    nm = int(rng.integers(1, 40))
    p_empty = rng.uniform(0, 0.3)
    using = [[value(rng, alphabets[k], p_empty) for k in range(len(vars_))] for _ in range(nu)]
    for r in range(nu):                          # duplicates and near-duplicates of using records
        if r and rng.random() < 0.2:
            using[r] = list(using[int(rng.integers(0, r))])
    master = []
    for _ in range(nm):
        if rng.random() < 0.5:                   # a copy of a using record, maybe with changes
            rec = list(using[int(rng.integers(0, nu))])
            for k in range(len(vars_)):
                if rng.random() < 0.3:
                    rec[k] = value(rng, alphabets[k], p_empty)
        else:
            rec = [value(rng, alphabets[k], p_empty) for k in range(len(vars_))]
        master.append(rec)
    for r in range(nm):
        if r and rng.random() < 0.1:
            master[r] = list(master[int(rng.integers(0, r))])
    m = {v: np.array([rec[k] for rec in master], dtype=str) for k, v in enumerate(vars_)}
    u = {v: np.array([rec[k] for rec in using], dtype=str) for k, v in enumerate(vars_)}
    return dict(master=m, using=u, vars=vars_, wmatch=wm, wnomatch=wnm, required=req)


def same(out, ref):
    i, j, s, n = out
    ri, rj, rs, rn = ref
    return (int(n) == int(rn) and np.array_equal(np.asarray(i, np.int64), np.asarray(ri, np.int64))
            and np.array_equal(np.asarray(j, np.int64), np.asarray(rj, np.int64))
            and np.array_equal(np.asarray(s, np.float64), np.asarray(rs, np.float64)))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    rng = np.random.default_rng(20261019)
    failures = pairs = short = 0
    for k in range(n):
        d = case(rng)
        with np.errstate(all="ignore"):
            ref = reference.run(**d)
        out = cand.run(**d)
        pairs += len(ref[0])
        short += sum(int(np.sum((np.char.str_len(a) > 0) & (np.char.str_len(a) < 3))) for a in d["master"].values())
        if not same(out, ref):
            failures += 1
            print(f"case {k}: candidate {len(out[0])} pairs / perfect {out[3]}, reference {len(ref[0])} / {ref[3]}"
                  f" (vars {d['vars']}, required {d['required']})")
    print(f"{n} cases, {pairs} reference pairs, {short} short master values, {failures} failures "
          "(pairs, perfect count and scores compared bitwise)")
