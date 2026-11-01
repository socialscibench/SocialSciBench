# 003_reclink

Prioritize real-development workloads while retaining synthetic correctness and performance coverage.
Read [task.md](task.md) for the computation and [DATA.md](DATA.md) for the real workloads and split rules.
The machine-readable contract is [task.json](task.json); [reference.py](reference.py) defines the answer
and [workloads.py](workloads.py) supplies synthetic inputs.

From the repository root, install the [root requirements](../requirements.txt) and obtain the
[development data](../datasets/README.md), then:

```bash
python -m harness init-solution --task 003 --out solutions/003/first
python -m harness evaluate --task 003 --solution solutions/003/first/solution.json --smoke
python -m harness evaluate --task 003 --solution solutions/003/first/solution.json --suite real-development --ids 102 --threads 1
```

Use the [shared evaluator](../harness/README.md) for baseline comparisons, CPU budgets,
validation, timing and final evaluation. Independent agents receive staged workspaces with
selected development inputs; held-out evaluation belongs to the maintainer.

**Informed research only:** consult the [historical study](../prestudy/reports/003_reclink.md) and
[baseline runners/measurements](../prestudy/baselines/003_reclink/) only when the experiment policy permits
prior results. Independent-agent development must not retrieve prior SocialSciBench candidates or results.
