"""Complete-output validation, independent of candidate code execution."""
import numpy as np


def normalize(task, output):
    if task == "003":
        if not isinstance(output, (tuple, list)) or len(output) != 4:
            raise ValueError("Task 003 must return (i, j, score, n_perfect)")
        output = dict(zip(("i", "j", "score", "n_perfect"), output))
    if not isinstance(output, dict):
        raise ValueError("Expected a dictionary output")
    if task == "002":
        # the reference's event_sets output echoes the (ragged) event_sets argument; not compared
        output = {k: v for k, v in output.items() if k != "event_sets"}
    return {k: np.asarray(v) for k, v in output.items() if isinstance(k, str)}


def validate(task, actual, expected, tolerances, dimensions):
    checks = {}

    def check(name, passed, **details):
        checks[name] = dict(passed=bool(passed), **details)

    def array(name, kind, shape=None):
        if name not in actual or name not in expected:
            check(name, False, reason="missing output")
            return None
        a, b = np.asarray(actual[name]), np.asarray(expected[name])
        valid_kind = a.dtype.kind in ("iu" if kind == "int" else "f")
        if a.shape != b.shape or (shape is not None and a.shape != shape) or not valid_kind:
            check(name, False, reason="shape or dtype mismatch", actual_shape=list(a.shape),
                  expected_shape=list(b.shape), dtype=str(a.dtype))
            return None
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            check(name, False, reason="non-finite value")
            return None
        return a, b

    def exact(name, shape=None):
        values = array(name, "int", shape)
        if values is not None:
            check(name, np.array_equal(*values))

    def close(name, rtol, values=None, scale=None):
        values = array(name, "float") if values is None else values
        if values is None:
            return
        a, b = values
        with np.errstate(over="ignore", invalid="ignore"):
            delta = np.abs(a - b)
            relative = delta / (np.maximum(np.abs(b), tolerances["relative_floor"]) if scale is None else scale)
        maximum = lambda x: float(x.max(initial=0)) if np.isfinite(x).all() else None
        check(name, np.all(relative <= rtol), max_abs_error=maximum(delta),
              max_rel_error=maximum(relative), rtol=rtol)

    if task == "001":
        close("beta", tolerances["estimate_rtol"])
        for key in ("nobs", "df_a", "n_clusters"):
            exact(key, ())
        k = dimensions["n_regressors"]
        v = array("vcov", "float", (k, k))
        if v is not None:
            a, b = v
            check("vcov_structure", np.all(np.diag(a) >= 0) and np.allclose(a, a.T, rtol=1e-8, atol=1e-12))
            if checks["vcov_structure"]["passed"]:
                close("se", tolerances["se_rtol"], (np.sqrt(np.diag(a)), np.sqrt(np.diag(b))))
            # The original contract does not set an off-diagonal covariance tolerance.
            with np.errstate(over="ignore"):
                error = np.abs(a-b)
            check("vcov_diagnostic", True,
                  max_abs_error=float(error.max(initial=0)) if np.isfinite(error).all() else None)
    elif task == "002":
        # Combined task (2026-10-25): estimates relative to max(|reference|, its reference SE);
        # analytic and bootstrap SEs and critical values relative; the event-study covariance
        # relative to SE_i SE_j. Only the outputs the reference returns are expected.
        for key in ("gt", "group_ids", "calendar_periods", "event_times"):
            exact(key)
        for name in ("gt", "simple", "group", "group_avg", "calendar", "calendar_avg", "event", "pre_avg",
                     "post_avg", "event_sets"):
            att, se_key = "att_" + name, "se_" + name
            values = array(att, "float")
            if values is not None:
                a, b = values
                ref_se = np.asarray(expected.get(se_key, np.zeros_like(b)), dtype=np.float64)
                scale = np.maximum(np.maximum(np.abs(b), np.abs(ref_se) if ref_se.shape == b.shape else 0.0),
                                   tolerances["relative_floor"])
                close(att, tolerances["estimate_rtol"], (a, b), scale=scale)
            close(se_key, tolerances["se_rtol"])
            if se_key in actual and np.asarray(actual[se_key]).dtype.kind == "f":
                check(se_key + "_nonnegative", np.all(np.asarray(actual[se_key]) >= 0))
        for key in sorted(expected):
            if key.startswith("se_boot_"):
                close(key, tolerances.get("bootstrap_rtol", tolerances["se_rtol"]))
            elif key.startswith("crit_"):
                close(key, tolerances.get("crit_rtol", tolerances["se_rtol"]))
        if "cov_event" in expected:
            values = array("cov_event", "float")
            ses = np.asarray(expected.get("se_event", []), dtype=np.float64)
            if values is not None and ses.shape == (values[1].shape[0],):
                a, b = values
                scale = np.maximum(np.outer(ses, ses), tolerances["relative_floor"] ** 2)
                close("cov_event", tolerances.get("cov_rtol", tolerances["se_rtol"]), (a, b), scale=scale)
    else:
        ii, jj, ss = array("i", "int"), array("j", "int"), array("score", "float")
        if task == "003":
            exact("n_perfect", ())
        if ii is not None and jj is not None and ss is not None:
            i, ri = ii
            j, rj = jj
            s, rs = ss
            shape_ok = i.ndim == j.ndim == s.ndim == 1 and len(i) == len(j) == len(s)
            check("pair_shapes", shape_ok)
            if shape_ok:
                check("index_bounds", np.all((i >= 0) & (i < dimensions["n_master"])) and
                      np.all((j >= 0) & (j < dimensions["n_using"])))
                order, ref_order = np.lexsort((j, i)), np.lexsort((rj, ri))
                i, j, s = i[order], j[order], s[order]
                duplicate = (i[1:] == i[:-1]) & (j[1:] == j[:-1])
                check("unique_pairs", not duplicate.any())
                check("pair_identity", np.array_equal(i, ri[ref_order]) and np.array_equal(j, rj[ref_order]))
                close("score", tolerances["score_rtol"], (s, rs[ref_order]))
    return dict(passed=bool(checks) and all(c["passed"] for c in checks.values()), checks=checks)
