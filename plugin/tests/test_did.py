"""The DiD additions (multipliers drawn inside the call, the event-study critical value with Pre_avg and
Post_avg) against task 002's reference, and the long-format interface against the cores; on the
evaluator's public development fixtures only."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from harness.cases import development_cases, generate_inputs
from turbo_socialsci import NotApplicable, att_gt
from turbo_socialsci.core import _draws, did_memory, did_speed

ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "002_csdid"
BOOT = ["dev-notyet-universal-bootstrap-201", "dev-notyet-ipw-universal-bootstrap-181", "dev-notyet-varying-bootstrap-201"]


def reference():
    spec = importlib.util.spec_from_file_location("task002_reference", TASK / "reference.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fixture(name):
    case = next(c for c in development_cases("002") if c["id"] == name)
    return generate_inputs(TASK, "002", case)


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12))) if a.size else 0.0


def test_generators_agree():
    key = _draws.seed_key(12345)
    units = np.array([0, 1, 7, 99999, 260399], dtype=np.int64)
    for B, b0, b1 in ((1000, 0, 1000), (25000, 64, 191), (25000, 24990, 25000)):
        nw = _draws.n_words(B)
        a = _draws.rademacher(key, units, nw, b0, b1)
        out = np.empty((units.size, b1 - b0))
        did_speed._rademacher_nb(np.uint64(key), units, nw, b0, b1, out)
        assert np.array_equal(a, out) and set(np.unique(a)) == {-1.0, 1.0}
    v = _draws.rademacher(key, np.arange(4000), _draws.n_words(512), 0, 512)
    assert abs(v.mean()) < 0.01 and abs(np.corrcoef(v[:-1].ravel(), v[1:].ravel())[0, 1]) < 0.01


@pytest.mark.parametrize("name", BOOT)
@pytest.mark.parametrize("core", [did_speed, did_memory], ids=["speed", "memory"])
def test_self_drawn_bootstrap_matches_reference(core, name):
    kw = fixture(name)
    B = kw.pop("boot_weights").shape[1]
    ours = core.run(**kw, n_boot=B, seed=7)
    V = _draws.rademacher_int8(_draws.seed_key(7), kw["y"].shape[0], B)
    given = core.run(**kw, boot_weights=V)
    ref = reference().run(**kw, boot_weights=V)
    for k in ref:
        if k.startswith(("se_boot_", "crit_")):
            assert rel(ours[k], given[k]) < 1e-10, k             # drawn here == the same draws given
            assert rel(ours[k], ref[k]) <= 1e-4, k               # the task's tolerance


@pytest.mark.parametrize("name", BOOT)
def test_variants_draw_the_same_multipliers(name):
    kw = fixture(name)
    B = kw.pop("boot_weights").shape[1]
    a = did_speed.run(**kw, n_boot=B, seed=3)
    b = did_memory.run(**kw, n_boot=B, seed=3)
    for k in a:
        if k.startswith(("se_boot_", "crit_")):
            assert rel(a[k], b[k]) < 1e-9, k


@pytest.mark.parametrize("name", BOOT)
@pytest.mark.parametrize("core", [did_speed, did_memory], ids=["speed", "memory"])
def test_event_critical_value_with_averages(core, name):
    kw = fixture(name)
    ref = reference()
    captured = {}
    original = ref.bootstrap

    def spy(rif, v, block=64):
        captured["draws"] = original(rif, v, block)
        return captured["draws"]
    ref.bootstrap = spy
    out = ref.run(**kw)
    draws = captured["draws"]
    off = sum(len(out[k]) for k in ("att_gt", "att_group", "att_calendar"))
    ne = len(out["att_event"])
    singles = [n for n in ("simple", "group_avg", "calendar_avg", "pre_avg", "post_avg")
               if np.isfinite(out[f"att_{n}"])]
    base = off + ne
    cols = list(range(off, off + ne)) + [base + singles.index(n) for n in ("pre_avg", "post_avg") if n in singles]
    expected = ref.boot_stats(draws[:, cols], kw.get("alpha", 0.05), True, kw.get("rules", "csdid181"))[1]
    ours = core.run(**kw)
    assert rel(ours["crit_event_with_avgs"], expected) <= 1e-4
    assert rel(ours["crit_event"], out["crit_event"]) <= 1e-4


@pytest.mark.parametrize("name", BOOT)
@pytest.mark.parametrize("tables", [("event",), ("group", "calendar"), ("gt",)])
def test_boot_tables(name, tables):
    """Bootstrapping only some tables (0.2): without "gt" the speed core draws the aggregates' own
    influence functions; the results equal the full computation's, the other tables are NaN."""
    kw = fixture(name)
    B = kw.pop("boot_weights").shape[1]
    full = did_speed.run(**kw, n_boot=B, seed=5)
    for core in (did_speed, did_memory):
        part = core.run(**kw, n_boot=B, seed=5, boot_tables=tables)
        follow = dict(gt="gt", event="event", group="group", calendar="calendar", group_avg="group",
                      calendar_avg="calendar", pre_avg="event", post_avg="event", event_with_avgs="event")
        for k in full:
            if not k.startswith(("se_boot_", "crit_")) or k == "se_boot_simple":
                continue
            table = follow[k.split("_", 2)[2] if k.startswith("se_boot_") else k[5:]]
            if table in tables:
                assert rel(part[k], full[k]) < 1e-10, (k, core.__name__)
            else:
                assert np.all(np.isnan(part[k])), (k, core.__name__)


def long_frame(kw, t0=2001):
    y, X, g = kw["y"], kw["X"], kw["gvar"]
    N, T = y.shape
    u, t = np.nonzero(np.isfinite(y))
    df = pd.DataFrame(dict(unit=u + 1000, year=t + t0, y=y[u, t],
                           cohort=np.where(g[u] > 0, g[u] + t0 - 1, 0)))
    for j in range(X.shape[-1]):
        df[f"x{j}"] = X[u, t, j] if X.ndim == 3 else X[u, j]
    if kw.get("w") is not None:
        df["w"] = kw["w"][u]
    return df


@pytest.mark.parametrize("name", ["dev-panel-0", "dev-weighted-unbalanced-dripw-181", "dev-varying-covariates-trim-181",
                                  "dev-notyet-drimp-weighted-unbalanced"])
@pytest.mark.parametrize("variant", ["speed", "memory"])
def test_long_format_interface(name, variant):
    kw = fixture(name)
    kw.pop("boot_weights", None)
    df = long_frame(kw)
    covs = [c for c in df.columns if c.startswith("x")]
    opts = {k: kw[k] for k in ("control", "method", "base_period", "trim", "rules", "event_sets") if k in kw}
    res = att_gt(df, "y", "unit", "year", "cohort", covariates=covs, weights="w" if "w" in df else None,
                 event_window=kw.get("event_window", (-5, 5)), variant=variant, **opts)
    keep = np.isfinite(kw["y"]).any(1)                     # units without a row are absent from the long data
    core = did_speed if variant == "speed" else did_memory
    sub = dict(kw, y=kw["y"][keep], X=kw["X"][keep], gvar=kw["gvar"][keep])
    if kw.get("w") is not None:
        sub["w"] = kw["w"][keep]
    r = core.run(**sub)
    assert np.array_equal(res.gt[["cohort", "time"]].to_numpy(), r["gt"] + 2000)
    np.testing.assert_allclose(res.gt["att"], r["att_gt"], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(res.gt["se"], r["se_gt"], rtol=1e-12)
    np.testing.assert_allclose(res.event["att"], r["att_event"], rtol=1e-12, atol=1e-14)
    assert res.averages.loc["post_avg", "att"] == pytest.approx(r["att_post_avg"], rel=1e-12, abs=1e-14)


def test_not_applicable():
    kw = fixture("dev-panel-0")
    df = long_frame(kw)
    with pytest.raises(NotApplicable):                       # repeated cross-section rows
        att_gt(pd.concat([df, df.head(3)]), "y", "unit", "year", "cohort")
    gap = df[df["year"] != 2003]
    with pytest.raises(NotApplicable):                       # a gap in the periods
        att_gt(gap, "y", "unit", "year", "cohort")
    tv = df.assign(w=1.0 + (df["year"] % 2))
    with pytest.raises(NotApplicable):                       # weights varying over time
        att_gt(tv, "y", "unit", "year", "cohort", weights="w")
    out = df.assign(cohort=np.where(df["cohort"] > 0, df["cohort"] + 20, 0))
    with pytest.raises(NotApplicable):                       # cohorts outside the panel
        att_gt(out, "y", "unit", "year", "cohort")
    with pytest.raises(NotApplicable):
        att_gt(df, "y", "unit", "year", "cohort", method="sunab")
    info = att_gt(df, "y", "unit", "year", "cohort", check_only=True)
    assert info["applies"] and info["periods"] == kw["y"].shape[1]
