"""The one-fixed-effect path (core/hdfe_onefe.py, turbo-socialsci 0.2) against the general solver
(hdfe_speed) on the same rows: dummies read as 0/1 lists, other columns in float32 or float64, missing
values, singleton groups, collinear dummies, clusters that nest the groups or not."""
import numpy as np
import pandas as pd
import pytest

from turbo_socialsci import feols
from turbo_socialsci.core import hdfe_speed


def data(n=20000, groups=900, clusters=40, nested=True, seed=0, f32=False):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, groups, n)
    cl = (g % clusters) if nested else rng.integers(0, clusters, n)
    df = pd.DataFrame(dict(g=g, cl=cl))
    k = 25
    D = np.zeros((n, k), np.int8)
    for i in range(n):
        D[i, rng.choice(k, size=rng.integers(0, 4), replace=False)] = 1
    for j in range(k):
        df[f"d{j}"] = D[:, j]
    df["d_zero"] = np.zeros(n, np.int8)                             # all zero: dropped
    df["d_const"] = (g % 50 == 7).astype(np.int8)                   # constant within groups: dropped
    df["d_bool"] = rng.random(n) < 0.3
    df["x1"] = rng.normal(size=n)
    df["x2"] = 0.5 * df["x1"] + rng.normal(size=n)
    df["count"] = rng.integers(0, 5, n)                            # integer, not 0/1: a regular column
    beta = rng.normal(size=k)
    df["y"] = (D @ beta + 0.3 * df["x1"] - 0.2 * df["x2"] + 0.1 * df["count"] + rng.normal(size=groups)[g]
               + rng.normal(size=n))
    df.loc[rng.choice(n, 30, replace=False), "x1"] = np.nan
    df.loc[rng.choice(n, 20, replace=False), "y"] = np.nan
    if f32:
        for c in ("x1", "x2", "y"):
            df[c] = df[c].astype(np.float32)
    df = pd.concat([df, df.iloc[:3].assign(g=groups + np.arange(3))], ignore_index=True)   # singletons
    return df, [f"d{j}" for j in range(5)] + ["x1", "d_zero"] + [f"d{j}" for j in range(5, k)] + \
        ["d_const", "x2", "d_bool", "count"]


def general(df, x):
    """hdfe_speed on the complete rows (its own singleton and collinearity handling)."""
    ok = df[["y"] + x].notna().all(axis=1).to_numpy()
    d = df[ok]
    X = d[x].to_numpy(dtype=np.float64)
    return hdfe_speed.run(d["y"].to_numpy(np.float64), X, d[["g"]].to_numpy(), d["cl"].to_numpy())


@pytest.mark.parametrize("variant", ["speed", "memory"])
@pytest.mark.parametrize("nested", [True, False])
@pytest.mark.parametrize("f32", [False, True])
def test_one_fe_matches_general_solver(variant, nested, f32):
    df, x = data(nested=nested, f32=f32, seed=3 + nested + 2 * f32)
    res = feols(df, "y", x, ["g"], "cl", variant=variant)
    r = general(df, x)
    assert res.dropped == ["d_zero", "d_const"]
    assert res.nobs == r["nobs"] and res.n_clusters == r["n_clusters"] and res.df_a == r["df_a"]
    np.testing.assert_allclose(res.coef["coef"].to_numpy(), r["beta"], rtol=1e-10, atol=1e-13, equal_nan=True)
    np.testing.assert_allclose(res.coef["se"].to_numpy(), np.sqrt(np.diag(r["vcov"])), rtol=1e-9, equal_nan=True)
    assert res.r2 == pytest.approx(r["r2"], abs=1e-12)
    assert res.r2_within == pytest.approx(r["r2_within"], abs=1e-12)
    assert res.rss == pytest.approx(r["rss"], rel=1e-11)
    complete = int(df[["y"] + x].notna().all(axis=1).sum())
    assert res.n_missing == len(df) - complete and res.n_singletons == complete - r["nobs"] >= 3


def test_variants_identical():
    df, x = data(seed=11)
    a = feols(df, "y", x, ["g"], "cl", variant="speed")
    b = feols(df, "y", x, ["g"], "cl", variant="memory")
    np.testing.assert_array_equal(a.coef["coef"].to_numpy(), b.coef["coef"].to_numpy())
    np.testing.assert_array_equal(a.coef["se"].to_numpy(), b.coef["se"].to_numpy())
    assert a.r2 == b.r2 and a.variant == "speed" and b.variant == "memory"


def test_not_a_dummy_column():
    """A one-byte integer column with values other than 0 and 1 is used as a regular column."""
    df, x = data(seed=5)
    df["small"] = (df["count"] * 3).astype(np.int8)
    res = feols(df, "y", x + ["small"], ["g"], "cl")
    r = general(df, x + ["small"])
    np.testing.assert_allclose(res.coef["coef"].to_numpy(), r["beta"], rtol=1e-10, atol=1e-13, equal_nan=True)


def test_interaction_and_nullable_groups():
    """Fixed effect as an interaction a#b and as a nullable integer column (missing -> dropped)."""
    df, x = data(seed=7)
    df["a"] = df["g"] // 10
    df["b"] = df["g"] % 10
    r1 = feols(df, "y", x, ["a#b"], "cl")
    r0 = feols(df, "y", x, ["g"], "cl")
    np.testing.assert_allclose(r1.coef["coef"].to_numpy(), r0.coef["coef"].to_numpy(), rtol=1e-12, equal_nan=True)
    gn = df["g"].astype("Int64")
    gn[df.index % 97 == 0] = pd.NA
    df["gn"] = gn
    r2 = feols(df, "y", x, ["gn"], "cl")
    sub = df[df["gn"].notna()]
    r3 = feols(sub, "y", x, ["g"], "cl")
    np.testing.assert_allclose(r2.coef["coef"].to_numpy(), r3.coef["coef"].to_numpy(), rtol=1e-12, equal_nan=True)
    assert r2.nobs == r3.nobs


@pytest.mark.parametrize("nested", [True, False])
def test_score_layouts(monkeypatch, nested):
    """With many clusters the cluster scores are summed by parts that own their clusters (groups nested
    in clusters) or by fewer parts (otherwise); forced here with a tiny size limit."""
    from turbo_socialsci.core import hdfe_onefe
    df, x = data(nested=nested, clusters=300, seed=13)
    a = feols(df, "y", x, ["g"], "cl")
    monkeypatch.setattr(hdfe_onefe, "SCORE_PARTS_BYTES", 1)
    b = feols(df, "y", x, ["g"], "cl")
    np.testing.assert_allclose(b.coef["coef"].to_numpy(), a.coef["coef"].to_numpy(), rtol=1e-12, equal_nan=True)
    np.testing.assert_allclose(b.coef["se"].to_numpy(), a.coef["se"].to_numpy(), rtol=1e-10, equal_nan=True)
    r = general(df, x)
    np.testing.assert_allclose(b.coef["se"].to_numpy(), np.sqrt(np.diag(r["vcov"])), rtol=1e-9, equal_nan=True)
