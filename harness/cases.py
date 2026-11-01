"""Public development fixtures and adapters to the unchanged task interfaces."""
from pathlib import Path

import numpy as np

from .common import load_module, read_json, task_hook


def development_cases(task, seed=1729):
    cases = []
    if task == "001":
        designs = [
            dict(id=1, design="firm_year", n_units=40, n_periods=5, n_regressors=3),
            dict(id=4, design="worker_firm", n_units=200, n_periods=6, n_firms=20,
                 n_regressors=3, mobility=0.02),
            dict(id=6, design="gravity", n_countries=6, n_periods=4, n_regressors=2),
            dict(id=7, design="four_way", n_units=60, n_periods=5, n_regressors=3,
                 n_industries=4, n_regions=3),
        ]
        for i, w in enumerate(designs):
            cases.append(dict(id="dev-" + w["design"], kind="generated", params=dict(w, seed=seed+i)))
        cases.append(dict(id="dev-singletons-labels", kind="generated",
                          params=dict(designs[0], seed=seed+4), variant="singletons"))
    elif task == "002":
        for i, share in enumerate((0.3, 0.7)):
            cases.append(dict(id=f"dev-panel-{i}", kind="generated", params=dict(
                id=1, seed=seed+i, n_units=240, n_periods=7, n_cohorts=3,
                n_covariates=2, share_treated=share)))
        cases.append(dict(id="dev-late-adoption", kind="generated", params=dict(
            id=1, seed=seed+2, n_units=300, n_periods=10, n_cohorts=2,
            n_covariates=3, share_treated=0.25, first_cohort=8), variant="shuffle"))
        cases.append(dict(id="dev-propensity-trimming", kind="propensity-trimming", seed=seed+3))
        # The combined task's options (workloads.py): both comparison groups, every method, weights,
        # unbalanced panels, both base periods, trimming, both rule sets, covariates that vary over
        # time, event-time subsets and the multiplier bootstrap.
        base = dict(id=1, n_units=300, n_periods=8, n_cohorts=3, n_covariates=2)
        for i, (name, options) in enumerate((
                ("notyet-universal-bootstrap-201", dict(control="notyet", base_period="universal", boot=199, rules="csdid201")),
                ("weighted-unbalanced-stdipw-201", dict(method="stdipw", weighted=True, missing=0.25, rules="csdid201")),
                ("reg-universal-event-sets-201", dict(method="reg", base_period="universal", rules="csdid201",
                                                      event_sets=[[1, 2], [-3, -2], [0, 1, 2, 3]])),
                ("notyet-ipw-universal-bootstrap-181", dict(control="notyet", method="ipw", base_period="universal",
                                                            boot=199, rules="csdid181")),
                ("varying-covariates-trim-181", dict(control="notyet", base_period="universal", varying=True,
                                                     trim=0.995, rules="csdid181")),
                ("weighted-unbalanced-dripw-181", dict(weighted=True, missing=0.25, rules="csdid181")),
                ("notyet-drimp-weighted-unbalanced", dict(control="notyet", method="drimp", weighted=True, missing=0.25)),
                ("notyet-varying-bootstrap-201", dict(control="notyet", weighted=True, boot=99, rules="csdid201")))):
            cases.append(dict(id=f"dev-{name}", kind="generated", params=dict(base, seed=seed+10+i, **options)))
        cases.append(dict(id="dev-high-propensity-trim", kind="propensity-trimming", seed=seed+3, variant="trim"))
        cases.append(dict(id="dev-overlap-201", kind="propensity-trimming", seed=seed+3, variant="overlap-201"))
    elif task == "003":
        for i in range(2):
            cases.append(dict(id=f"dev-random-{i}", kind="generated", params=dict(
                id=1, seed=seed+i, n_master=60, n_using=80, overlap=0.6)))
        for name in ("ties-missing-perfect", "no-candidates"):
            cases.append(dict(id="dev-" + name, kind=name, seed=seed))
    else:
        raise ValueError(f"No development cases for {task}")
    return cases


def generate_inputs(task_dir, task, case, generate=None):
    if task == "003":
        # The built-in linkage adapter takes contract constants from this task file.
        workloads = load_module(Path(task_dir) / "workloads.py", "task_workloads")
    if case["kind"] == "generated":
        if generate is None:
            generate = task_hook(task_dir, read_json(Path(task_dir) / "task.json"), "workload_entry_point")
        d = generate(case["params"])
        d.pop("truth", None)  # Ground-truth linkage labels are never candidate inputs.
        if case.get("variant") == "singletons":
            rng = np.random.default_rng(case["params"]["seed"])
            d["y"] = np.concatenate([d["y"], rng.standard_normal(3)])
            d["X"] = np.vstack([d["X"], rng.standard_normal((3, d["X"].shape[1]))])
            d["fe_ids"] = np.vstack([d["fe_ids"], [[999, 0], [999, 999], [998, 999]]]) * 7 - 13
            d["cluster_ids"] = np.concatenate([d["cluster_ids"], [0, 0, 0]]) * 11 - 5
        if case.get("variant") == "shuffle":
            order = np.random.default_rng(case["params"]["seed"] + 19).permutation(len(d["y"]))
            d = {key: value[order] for key, value in d.items()}
    elif task == "002" and case["kind"] == "propensity-trimming":
        rng = np.random.default_rng(case["seed"])
        X = np.column_stack([np.linspace(-4, 4, 1000), rng.standard_normal(1000)])
        p = 1 / (1 + np.exp(-2 * X[:, 0]))
        gvar = np.where(rng.random(1000) < p, 4, 0)
        gvar[-1] = 0  # A control scoring above 0.995, which csdid does not trim (pscoretrim(1)).
        t = np.arange(1, 7)
        y = (rng.standard_normal((1000, 6)) + X[:, :1] * t / 6
             + ((gvar[:, None] > 0) & (t >= 4)) * (1 + 0.2 * (t-4)))
        d = dict(y=y, X=X, gvar=gvar)
        if case.get("variant") == "trim":           # the 0.995 rule: that control gets weight 0
            d.update(trim=0.995)
        elif case.get("variant") == "overlap-201":
            # csdid 2.0.1 leaves cohort 4's cells unestimated (an unweighted score >= 0.999); a second
            # cohort, assigned at random, keeps its cells
            g5 = (gvar == 0) & (np.random.default_rng(case["seed"] + 1).random(1000) < 0.2)
            g5[-1] = False
            d.update(gvar=np.where(g5, 5, gvar), rules="csdid201", control="notyet", base_period="universal")
    elif task == "003":
        using = [
            ["ALPHA INC", "1 MAIN ST", "CITY", "10000", "CA"],
            ["ALPHA INC", "1 MAIN ST", "CITY", "10000", "CA"],
            ["BETA CORP", "2 MAIN ST", "CITY", "20000", "CA"],
            ["BETA CORP", "2 MAIN ST", "CITY", "20000", "CA"],
            ["A", "", "CITY", "", "CA"],
            ["ABCDE", "3 MAIN ST", "CITY", "30000", "CA"],
            ["ZBCDEZ", "4 MAIN ST", "CITY", "40000", "CA"],
        ]
        master = [using[0], ["ALPHA LTD", "1 MAIN ST", "CITY", "10000", "CA"],
                  ["BETA LLC", "2 MAIN ST", "CITY", "20000", "CA"],
                  ["AB", "", "CITY", "", "CA"],
                  ["ABCXYZ", "3 MAIN ST", "CITY", "30000", "CA"],
                  ["ABCDEF", "4 MAIN ST", "CITY", "40000", "CA"]]
        if case["kind"] == "no-candidates":
            master = [["NO MATCH", "9 OTHER RD", "TOWN", "99999", "TX"]]
        d = {key: {v: np.array(rows)[:, j] for j, v in enumerate(workloads.VARS)}
             for key, rows in (("master", master), ("using", using))}
    else:
        raise ValueError(f"Unknown fixture {case}")
    if task == "003":
        d.update(vars=workloads.VARS, wmatch=workloads.WMATCH, wnomatch=workloads.WNOMATCH,
                 required=workloads.REQUIRED, minscore=workloads.MINSCORE, minbigram=workloads.MINBIGRAM)
    return d
