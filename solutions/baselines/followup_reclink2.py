"""Follow-up checks of the reclink2 baseline on task 003 (study tooling, development inputs only).

usage: followup_reclink2.py CASE REFDIR OUTDIR [--trace]

Loads a real-development case as the evaluator does, counts values containing characters that
reclink2 handles specially in Stata macros (double quote, backquote, apostrophe, dollar,
backslash, braces, parentheses), runs the adapter's Stata session once with reclink2's output
visible (logged to OUTDIR/CASE.log; with --trace, `set trace on` at depth 1), saves the returned
pairs to OUTDIR/CASE.npz and compares them with the saved reference output REFDIR/CASE.npz:
missing and extra pairs (with the reference's tied pairs of the same master record) and score
differences of common pairs in float32 units in the last place. Run with the baseline interpreter
from the repository root, with nothing else running.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "solutions" / "baselines"))
os.environ.setdefault("SOCIALSCIBENCH_BASELINE_VENV", str(ROOT / "solutions" / "baselines" / ".venv"))
os.environ.setdefault("OMP_NUM_THREADS", "1")

from compare import case_spec  # noqa: E402
from harness.common import read_json  # noqa: E402
from harness.datasets import load_inputs  # noqa: E402

case_name, refdir, outdir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
trace = "--trace" in sys.argv[4:]
outdir.mkdir(parents=True, exist_ok=True)
task_dir = ROOT / "003_reclink"
case, _ = case_spec("003", case_name)
data = load_inputs(task_dir, read_json(task_dir / "task.json"), case, str(ROOT / "datasets" / "003" / "data"))
VARS = data["vars"]

print(f"case {case_name}: {len(data['master'][VARS[0]])} master, {len(data['using'][VARS[0]])} using records")
for side in ("master", "using"):
    for ch in ['"', "`", "'", "$", "\\", "{", "}", "(", ")"]:
        n = {v: int(np.char.find(data[side][v], ch).__ge__(0).sum()) for v in VARS}
        if any(n.values()):
            print(f"  {side} values containing {ch!r}: {n}")

sys.path.insert(0, str(ROOT / "solutions" / "003" / "stata-reclink2"))
import solution as adapter  # noqa: E402  (starts Stata MP with one processor)
from pystata import stata  # noqa: E402
from sfi import Data, Macro  # noqa: E402

m = {v: np.asarray(data["master"][v], dtype=str) for v in VARS}
u = {v: np.asarray(data["using"][v], dtype=str) for v in VARS}
mpath, upath = os.path.join(adapter.TMP, "master.dta"), os.path.join(adapter.TMP, "using.dta")
adapter._write(mpath, "idm", m)
adapter._write(upath, "idu", u)
log = outdir / f"{case_name}.log"
stata.run(f'log using "{log}", text replace', quietly=True)
stata.run(f'use "{mpath}", clear\ndescribe\ndescribe using "{upath}"', quietly=False)
if trace:
    stata.run("set trace on\nset tracedepth 1", quietly=True)
t0 = time.perf_counter()
cmd = (f'capture noisily reclink2 {" ".join(VARS)} using "{upath}", idmaster(idm) idusing(idu) gen(score) '
       f'wmatch({" ".join(map(str, adapter.WMATCH))}) wnomatch({" ".join(map(str, adapter.WNOMATCH))}) required(state)')
stata.run(cmd + "\nglobal SSB_rc = _rc\nset trace off\nlog close", quietly=False)
rc = int(Macro.getGlobal("SSB_rc"))
print(f"reclink2 returned code {rc} after {time.perf_counter() - t0:.1f} s; log {log}")
if rc:
    sys.exit(1)

stata.run("keep if !missing(idu)", quietly=True)
if Data.getObsTotal() == 0:
    sys.exit("no pairs returned")
out = stata.pdataframe_from_data(var=["idm", "idu", "score"])
i = out["idm"].to_numpy(np.int64) - 1
j = out["idu"].to_numpy(np.int64) - 1
score = out["score"].to_numpy(np.float32)
np.savez(outdir / f"{case_name}.npz", i=i, j=j, score=score)

ref = np.load(refdir / f"{case_name}.npz")
ri, rj, rs = ref["i"].astype(np.int64), ref["j"].astype(np.int64), ref["score"]
print(f"pairs: reclink2 {len(i)}, reference {len(ri)}")
got = {(a, b): s for a, b, s in zip(i, j, score)}
exp = {(a, b): s for a, b, s in zip(ri, rj, rs)}


def show(pairs, label):
    for a, b in sorted(pairs)[:20]:
        same_master = sorted((k[1], float(exp[k])) for k in exp if k[0] == a)
        print(f"  {label} master {a} using {b}: reference pairs of this master {same_master}; "
              f"reclink2 pairs {sorted((k[1], float(got[k])) for k in got if k[0] == a)}")
        print("    master:", {v: str(m[v][a]) for v in VARS})
        print("    using: ", {v: str(u[v][b]) for v in VARS})


show(set(exp) - set(got), "missing")
show(set(got) - set(exp), "extra")
common = sorted(set(got) & set(exp))
d = np.array([abs(float(np.float32(got[k])) - float(exp[k])) for k in common])
ulp = np.array([np.spacing(np.float32(exp[k])) for k in common], dtype=np.float64)
print(f"common pairs {len(common)}; scores differ on {(d > 0).sum()}; "
      f"max abs diff {d.max() if len(d) else 0:.3g}; max diff in float32 ulps {(d / ulp).max() if len(d) else 0:.1f}")
for k in [common[x] for x in np.argsort(-d)[:5] if d[x] > 0]:
    print(f"  pair {k}: reclink2 {float(got[k]):.9g}, reference {float(exp[k]):.9g}")
    print("    master:", {v: str(m[v][k[0]]) for v in VARS})
    print("    using: ", {v: str(u[v][k[1]]) for v in VARS})
