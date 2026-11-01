# Environment and build

Baseline adapter (comparison only). Environment, package versions and hashes, and the
measurement driver: `solutions/baselines/README.md`. Run with the baseline interpreter:

```bash
solutions/baselines/.venv/py/bin/python solutions/baselines/compare.py --task 001 \
  --solution solutions/001/stata-reghdfe/solution.json --cases <ids> --threads 1 \
  --refdir <saved reference outputs> --out <results.jsonl> --python solutions/baselines/.venv/py/bin/python
```
