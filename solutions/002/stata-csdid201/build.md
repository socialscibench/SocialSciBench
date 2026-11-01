# Baseline environment

Stata 18 MP with pystata (`/Applications/Stata 18/utilities`), run from the baselines' Python
environment (`solutions/baselines/.venv/py`, see `solutions/baselines/README.md`). csdid 2.0.1 is
installed in its own folder, so that it does not replace the SSC csdid 1.82 used by
`stata-csdid181`:

```stata
sysdir set PLUS "solutions/baselines/.venv/stata-src/csdid-2.0.1"
net set ado "solutions/baselines/.venv/stata-src/csdid-2.0.1"
net install csdid, from("https://raw.githubusercontent.com/pedrohcgs/csdid-stata/f98bcea0691b1670c409bb7e3cc81d60705b4f84") replace
```

(`csdid version` then reports 2.0.1, 23oct2026.) Measure with compare.py, e.g.

```bash
solutions/baselines/.venv/py/bin/python solutions/baselines/compare.py --task 002 \
  --solution solutions/002/stata-csdid201/solution.json --cases 103 --threads 1 --warmups 0 --repeats 1 \
  --refdir REFS/002 --out results.jsonl --python solutions/baselines/.venv/py/bin/python \
  --not-comparable se_boot_ crit_
```
