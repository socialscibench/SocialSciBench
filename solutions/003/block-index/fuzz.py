"""Random small task-003 problems compared with the reference (development check, not imported at
runtime). Pairs, the perfect-pair count and scores are compared exactly (scores bitwise).

usage: python fuzz.py SOLUTION.py N_CASES SEED0   (from the repository root)
"""
import sys
import warnings

import numpy as np

sys.path.insert(0, ".")
from pathlib import Path  # noqa: E402

from harness.common import load_module  # noqa: E402

ref = load_module(Path("003_reclink/reference.py"), "ref003")
cand = load_module(Path(sys.argv[1]), "cand")
n_cases, seed0 = int(sys.argv[2]), int(sys.argv[3])
ALPH = np.array(list("ABCDE ,.-1"))
stats = dict(cases=0, failed=0, ref_error=0, ref_pairs=0, short_values=0)


def word(rng, alph, lo, hi):
    n = int(rng.integers(lo, hi))
    return "".join(rng.choice(alph, n))


for case in range(n_cases):
    rng = np.random.default_rng(seed0 + case)
    nv = int(rng.choice([5, 5, 5, 4, 6]))
    names = [f"v{t}" for t in range(nv)]
    if rng.random() < 0.7 and nv == 5:
        names = ["name", "addr", "city", "zip", "state"]
        required = ["state"]
        wm, wn = [10, 8, 5, 5, 5], [2, 3, 4, 4, 4]
    else:
        required = list(rng.choice(names, size=int(rng.integers(0, 3)), replace=False))
        wm = list(rng.integers(1, 12, nv).astype(float))
        wn = list(rng.integers(1, 6, nv).astype(float))
    k = int(rng.integers(1, 6))
    alph = ALPH[:k + 3]
    nm, nu = int(rng.integers(1, 40)), int(rng.integers(1, 80))
    U = {v: np.array([word(rng, alph, 0, 9) for _ in range(nu)]) for v in names}
    for v in required:
        U[v] = rng.choice(np.array(["AA", "BB", ""]), nu)
    # duplicate some using records
    if nu > 3 and rng.random() < 0.5:
        src = rng.integers(0, nu, size=nu // 4)
        dst = rng.integers(0, nu, size=nu // 4)
        for v in names:
            U[v][dst] = U[v][src]
    # master: copies of using records with edits, plus new ones
    M = {v: np.empty(nm, dtype=object) for v in names}
    for i in range(nm):
        j = int(rng.integers(0, nu))
        for v in names:
            s = U[v][j]
            r = rng.random()
            if r < 0.15:
                s = ""
            elif r < 0.45 and v not in required:
                s = word(rng, alph, 0, 9)
            elif r < 0.6 and len(s) > 1 and v not in required:
                p = int(rng.integers(0, len(s)))
                s = s[:p] + str(rng.choice(alph)) + s[p + 1:]
            M[v][i] = s
    M = {v: np.array(list(M[v]), dtype=str) for v in names}
    d = dict(master=M, using=U, vars=names, wmatch=wm, wnomatch=wn, required=required,
             minscore=float(rng.choice([0.6, 0.6, 0.5, 0.7])), minbigram=float(rng.choice([0.6, 0.6, 0.5])))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ri, rj, rs, rn = ref.run(**d)
    except Exception:
        stats["ref_error"] += 1
        continue
    stats["cases"] += 1
    stats["ref_pairs"] += len(ri)
    stats["short_values"] += sum(int((np.char.str_len(M[v]) < 3).sum()) for v in names)
    ci, cj, cs, cn = cand.run(**d)
    o1, o2 = np.lexsort((rj, ri)), np.lexsort((cj, ci))
    same = (rn == cn and len(ri) == len(ci) and np.array_equal(ri[o1], ci[o2]) and np.array_equal(rj[o1], cj[o2])
            and np.array_equal(np.asarray(rs, float)[o1], np.asarray(cs, float)[o2]))
    if not same:
        stats["failed"] += 1
        print("FAIL", seed0 + case, nv, required, len(ri), len(ci), rn, cn)
print(stats)
