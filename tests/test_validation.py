import copy

import numpy as np
import pytest

from harness.cases import development_cases, generate_inputs
from harness.common import ROOT, load_module, task_info
from harness.validation import normalize, validate


def check(task, actual, expected, dimensions=None):
    _, manifest = task_info(task)
    return validate(task, actual, expected, manifest["tolerances"], dimensions or {})["passed"]


def pairs():
    return dict(i=np.array([0, 1]), j=np.array([0, 1]), score=np.array([0.7, 0.8]), n_perfect=np.array(0))


def test_pair_identity_cannot_be_replaced_by_equal_sums():
    ref = pairs()
    bad = dict(ref, j=np.array([1, 0]))
    assert bad["j"].sum() == ref["j"].sum()
    assert not check("003", bad, ref, dict(n_master=2, n_using=2))


def test_pair_scores_checked_individually_and_order_independent():
    ref = pairs()
    reordered = dict(ref, **{key: ref[key][::-1] for key in ("i", "j", "score")})
    assert check("003", reordered, ref, dict(n_master=2, n_using=2))
    bad = dict(ref, score=np.array([0.6, 0.9]))
    assert np.isclose(bad["score"].sum(), ref["score"].sum())
    assert not check("003", bad, ref, dict(n_master=2, n_using=2))


@pytest.mark.parametrize("key,value", [
    ("score", [np.nan, 0.8]), ("score", [np.inf, 0.8]),
    ("i", [-1, 1]), ("i", [0, 2]), ("i", [0.0, 1.0]),
    ("j", [[0], [1]]), ("score", [0.7]),
])
def test_invalid_pair_outputs(key, value):
    ref = pairs()
    assert not check("003", dict(ref, **{key: np.asarray(value)}), ref, dict(n_master=2, n_using=2))


def test_duplicate_pairs_and_perfect_count():
    ref = dict(i=np.array([0, 0]), j=np.array([1, 1]), score=np.array([1., 1.]), n_perfect=np.array(2))
    assert not check("003", ref, ref, dict(n_master=2, n_using=2))
    ref = dict(pairs(), n_perfect=np.array(0))
    assert not check("003", dict(ref, n_perfect=np.array(1)), ref, dict(n_master=2, n_using=2))


def test_ols_coefficients_standard_errors_and_counts():
    ref = dict(beta=np.array([1., 2.]), vcov=np.diag([0.1, 0.2]),
               nobs=np.array(100), df_a=np.array(2), n_clusters=np.array(10))
    assert check("001", ref, ref, dict(n_regressors=2))
    for key, value in [("beta", np.array([1.001, 2.])), ("vcov", ref["vcov"]*1.01),
                       ("nobs", np.array(99)), ("nobs", np.array(100.0)),
                       ("vcov", np.array([[0.1, 0.1], [0., 0.2]])), ("vcov", -ref["vcov"])]:
        assert not check("001", dict(ref, **{key: value}), ref, dict(n_regressors=2))


def test_near_zero_and_overflow_errors_are_finite_json():
    import json
    ref = dict(beta=np.array([0., -1e308]), vcov=np.eye(2), nobs=np.array(100),
               df_a=np.array(2), n_clusters=np.array(10))
    _, manifest = task_info("001")
    result = validate("001", dict(ref, beta=np.array([1e-10, 1e308])), ref,
                      manifest["tolerances"], dict(n_regressors=2))
    assert not result["passed"]
    json.dumps(result, allow_nan=False)
    ref["vcov"] = np.diag([1e308, 1e308])
    result = validate("001", dict(ref, vcov=-ref["vcov"]), ref,
                      manifest["tolerances"], dict(n_regressors=2))
    assert not result["passed"]
    json.dumps(result, allow_nan=False)


def test_did_checks_all_aggregates_and_time_labels():
    entry, _ = task_info("002")
    d = generate_inputs(ROOT / entry["directory"], "002", development_cases("002")[0])
    ref = normalize("002", load_module(ROOT / entry["directory"] / "reference.py", "did_test_ref").run(**d))
    assert check("002", ref, ref)
    for key in ref:
        if ref[key].size == 0:          # e.g. event-set outputs when no subsets are requested
            continue
        bad = copy.deepcopy(ref)
        bad[key] = bad[key] + (1 if bad[key].dtype.kind in "iu" else 0.1)
        assert not check("002", bad, ref), key


def test_did_fixtures_cover_every_option_and_check_every_output():
    entry, _ = task_info("002")
    reference = load_module(ROOT / entry["directory"] / "reference.py", "did_options_test_ref")
    seen = {k: set() for k in ("method", "control", "base_period", "rules")}
    flags = set()
    for case in development_cases("002"):
        d = generate_inputs(ROOT / entry["directory"], "002", case)
        for k in seen:
            seen[k].add(d.get(k, {"method": "dripw", "control": "never", "base_period": "varying", "rules": "csdid181"}[k]))
        flags |= {f for f, on in (("weights", d.get("w") is not None), ("bootstrap", d.get("boot_weights") is not None),
                                  ("varying covariates", np.asarray(d["X"]).ndim == 3), ("event sets", bool(d.get("event_sets"))),
                                  ("trim", d.get("trim", 1.0) < 1), ("unbalanced", not np.isfinite(d["y"]).all())) if on}
        ref = normalize("002", reference.run(**d))
        assert check("002", ref, ref), case["id"]
        if d.get("boot_weights") is not None:
            boot = ref
    assert seen["method"] == set(reference.METHODS) and seen["control"] == {"never", "notyet"}
    assert seen["base_period"] == {"varying", "universal"} and seen["rules"] == {"csdid181", "csdid201"}
    assert flags == {"weights", "bootstrap", "varying covariates", "event sets", "trim", "unbalanced"}
    for key in boot:                    # bootstrap SEs and critical values are checked too
        if boot[key].size == 0:
            continue
        bad = copy.deepcopy(boot)
        bad[key] = bad[key] + (1 if bad[key].dtype.kind in "iu" else 0.1)
        assert not check("002", bad, boot), key


def test_did_fixture_exercises_propensity_trimming():
    entry, _ = task_info("002")
    case = next(c for c in development_cases("002") if c["kind"] == "propensity-trimming")
    data = generate_inputs(ROOT / entry["directory"], "002", case)
    reference = load_module(ROOT / entry["directory"] / "reference.py", "did_trim_ref")
    X = np.column_stack([data["X"], np.ones(len(data["gvar"]))])
    index, _ = reference.logit((data["gvar"] > 0).astype(float), X, np.ones(len(X)))   # returns X b and e(V)
    p = 1 / (1 + np.exp(-index))
    assert np.any((data["gvar"] == 0) & (p > 0.995))
    output = normalize("002", reference.run(**data))
    assert check("002", output, output)


def test_record_linkage_fixture_exercises_ties_exclusion_and_float32_boundary():
    entry, _ = task_info("003")
    case = next(c for c in development_cases("003") if c["kind"] == "ties-missing-perfect")
    data = generate_inputs(ROOT / entry["directory"], "003", case)
    reference = load_module(ROOT / entry["directory"] / "reference.py", "linkage_edges_ref")
    i, j, score, perfect = reference.run(**data)
    assert perfect == 2
    found = set(zip(i, j))
    assert (2, 2) in found and (2, 3) in found
    assert (1, 0) not in found and (1, 1) not in found
    strings = reference.Strings(np.array(["ABCDEF", "ZBCDEZ"]))
    bi = reference.bigram1(strings, np.array([0]), np.array([1]))[0]
    assert bi == float(np.float32(0.6)) and bi > 0.6
