# Historical baselines

Each active task's original software baseline scripts and recorded `results/*.jsonl` are retained here. Raw historical measurement files are unchanged. New baseline invocations default to unique directories under the repository's ignored `runs/` directory. `--out` can explicitly select another destination.

Run scripts from the repository root, for example `python baselines/001_reghdfe/bench.py --stata /path/to/stata-mp --ids 1`. These scripts require the external software documented in the [reports](../reports/README.md). For submitted implementations use the [shared evaluator](../../harness/README.md), which needs neither Stata nor R.
