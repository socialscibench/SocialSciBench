"""Random small task-002 problems compared with the reference (harness validator)."""
import sys, json, warnings, numpy as np
sys.path.insert(0, '.')
from pathlib import Path
from harness.common import load_module
from harness.validation import validate, normalize
ref = load_module(Path('002_csdid/reference.py'), 'ref002')
cand = load_module(Path(sys.argv[1]), 'cand')
tol = json.load(open('002_csdid/task.json'))['tolerances']
n_cases, seed0 = int(sys.argv[2]), int(sys.argv[3])
stats = dict(passed=0, failed=0, ref_invalid=0, cand_error=0)
for case in range(n_cases):
    rng = np.random.default_rng(seed0 + case)
    N = int(rng.integers(15, 400)); T = int(rng.integers(3, 13)); k = int(rng.integers(0, 5))
    X = rng.standard_normal((N, k)) * rng.choice([1, 3, 10], size=k) + rng.choice([0, 5], size=k)
    if k and rng.random() < 0.3:
        X[:, 0] = rng.integers(0, 3, N)                      # discrete covariate
    ncoh = int(rng.integers(1, T))
    cohorts = np.sort(rng.choice(np.arange(2, T + 1), size=min(ncoh, T - 1), replace=False))
    share_never = rng.uniform(0.2, 0.9)
    g = np.where(rng.random(N) < share_never, 0, rng.choice(cohorts, N))
    if rng.random() < 0.3:                                   # a tiny cohort
        c = int(rng.choice(np.arange(2, T + 1))); idx = rng.choice(N, size=int(rng.integers(1, 4)), replace=False); g[idx] = c
    if (g == 0).sum() < k + 3 or (g > 0).sum() == 0:
        continue
    y = rng.standard_normal((N, T)) + (X[:, :1] if k else 0) * np.arange(T) / T
    if rng.random() < 0.2:
        y = (y > 0.5).astype(float)                          # binary outcome
    d = dict(y=y, X=X, gvar=g.astype(np.int64))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            exp = normalize('002', ref.run(**d))
        if not all(np.all(np.isfinite(v)) for v in exp.values()):
            stats['ref_invalid'] += 1; continue
    except Exception:
        stats['ref_invalid'] += 1; continue
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = normalize('002', cand.run(**d))
    except Exception as e:
        stats['cand_error'] += 1; print('cand error', case, repr(e)); continue
    rep = validate('002', out, exp, tol, {})
    if rep['passed']:
        stats['passed'] += 1
    else:
        stats['failed'] += 1
        bad = {kk: vv for kk, vv in rep['checks'].items() if not vv.get('passed', True)}
        print('FAIL case', seed0 + case, f'N={N} T={T} k={k} cohorts={np.unique(g[g>0], return_counts=True)}', json.dumps(bad, default=str)[:600])
print(stats)
