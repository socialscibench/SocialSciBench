"""The fixed-effects regression's additions (collinear regressors, R2) against a direct dummy-variable
OLS, and the DataFrame interface against the cores."""
import numpy as np
import pandas as pd
import pytest

from turbo_socialsci import NotApplicable, feols
from turbo_socialsci.core import hdfe_memory, hdfe_speed

CORES = [hdfe_speed, hdfe_memory]


def panel(n=3000, firms=150, years=8, seed=0):
    rng = np.random.default_rng(seed)
    firm = rng.permutation(np.arange(n) % firms)          # 20 rows per firm: no singletons
    year = rng.integers(0, years, n)
    x1, x2, x3 = rng.normal(size=(3, n))
    y = 0.5 * x1 - 0.2 * x2 + 0.1 * x3 + rng.normal(size=firms)[firm] + rng.normal(size=years)[year] + rng.normal(size=n)
    return rng, firm, year, x1, x2, x3, y


def dummy_ols(y, X, firm, year):
    """beta, cluster-robust vcov (by firm; reghdfe's factor with df_a = years - 1), RSS, R2."""
    n = len(y)
    D = np.column_stack([np.ones(n), (firm[:, None] == np.unique(firm)[1:]).astype(float),
                         (year[:, None] == np.unique(year)[1:]).astype(float)])
    Z = np.column_stack([X, D])
    coef, *_ = np.linalg.lstsq(Z, y, rcond=None)
    e = y - Z @ coef
    k = X.shape[1]
    Xt = X - D @ np.linalg.lstsq(D, X, rcond=None)[0]        # partialled regressors (FWL)
    bread = np.linalg.inv(Xt.T @ Xt)
    G = np.unique(firm).size
    S = np.zeros((G, k))
    np.add.at(S, np.searchsorted(np.unique(firm), firm), Xt * e[:, None])
    df_a = np.unique(year).size - 1                          # firm is nested in the cluster: 0
    K = k + 1 + df_a
    q = (n - 1) / (n - K) * G / (G - 1)
    rss = e @ e
    return coef[:k], q * bread @ (S.T @ S) @ bread, rss, 1 - rss / ((y - y.mean()) ** 2).sum()


@pytest.mark.parametrize("core", CORES, ids=["speed", "memory"])
def test_collinear_regressors_and_r2(core):
    rng, firm, year, x1, x2, x3, y = panel()
    absorbed = rng.normal(size=150)[firm]                     # constant within firm
    X = np.column_stack([x1, np.zeros_like(x1), x2, x1.copy(), x3, absorbed, x1 + 2 * x2])
    r = core.run(y, X, np.column_stack([firm, year]), firm)
    assert r["dropped"].tolist() == [1, 3, 5, 6]
    b, V, rss, r2 = dummy_ols(y, np.column_stack([x1, x2, x3]), firm, year)
    kept = [0, 2, 4]
    np.testing.assert_allclose(r["beta"][kept], b, rtol=1e-9)
    np.testing.assert_allclose(r["vcov"][np.ix_(kept, kept)], V, rtol=1e-8)
    assert np.isnan(r["beta"][[1, 3, 5, 6]]).all()
    assert r["rss"] == pytest.approx(rss, rel=1e-10)
    assert r["r2"] == pytest.approx(r2, abs=1e-12)
    assert r["nobs"] == len(y) and r["df_a"] == 7 and r["n_clusters"] == 150


@pytest.mark.parametrize("core", CORES, ids=["speed", "memory"])
def test_full_rank_r2(core):
    _, firm, year, x1, x2, x3, y = panel(seed=1)
    X = np.column_stack([x1, x2, x3])
    r = core.run(y, X, np.column_stack([firm, year]), firm)
    b, V, rss, r2 = dummy_ols(y, X, firm, year)
    assert r["dropped"].size == 0
    np.testing.assert_allclose(r["beta"], b, rtol=1e-9)
    np.testing.assert_allclose(r["vcov"], V, rtol=1e-8)
    assert r["r2"] == pytest.approx(r2, abs=1e-12)
    yt = y - pd.Series(y).groupby(firm).transform("mean").to_numpy()
    assert 0.0 < r["r2_within"] < r["r2"] < 1.0 and yt.size == len(y)


def frame(seed=2):
    rng, firm, year, x1, x2, x3, y = panel(seed=seed)
    df = pd.DataFrame(dict(y=y, x1=x1, x2=x2, x3=x3, zero=0.0, firm=firm, year=year, state=firm % 7))
    df.loc[[5, 17, 40], "y"] = np.nan
    df.loc[[8], "x2"] = np.nan
    return df


@pytest.mark.parametrize("variant", ["speed", "memory"])
def test_dataframe_interface(variant):
    df = frame()
    res = feols(df, "y", ["x1", "zero", "x2", "x3"], ["firm", "state#year"], "firm", variant=variant)
    ok = df[["y", "x1", "x2", "x3"]].notna().all(axis=1).to_numpy()
    d = df[ok]
    sy = d.groupby(["state", "year"]).ngroup().to_numpy()
    core = hdfe_speed if variant == "speed" else hdfe_memory
    r = core.run(d["y"].to_numpy(), d[["x1", "zero", "x2", "x3"]].to_numpy(),
                 np.column_stack([d["firm"].to_numpy(), sy]), d["firm"].to_numpy())
    assert res.dropped == ["zero"] and res.n_missing == 4 and res.variant == variant
    np.testing.assert_allclose(res.coef["coef"].to_numpy(), r["beta"], rtol=1e-12, equal_nan=True)
    np.testing.assert_allclose(res.coef["se"].to_numpy(), np.sqrt(np.diag(r["vcov"])), rtol=1e-12, equal_nan=True)
    assert res.nobs == r["nobs"] and res.r2 == pytest.approx(r["r2"], abs=1e-14)
    assert list(res.vcov.index) == ["x1", "x2", "x3"]


def test_not_applicable():
    df = frame()
    for kw in (dict(cluster=None), dict(absorb=[]), dict(x=[])):
        args = dict(data=df, y="y", x=["x1"], absorb=["firm"], cluster="firm")
        args.update(kw)
        with pytest.raises(NotApplicable):
            feols(args["data"], args["y"], args["x"], args["absorb"], args["cluster"])
    df["s"] = "a"
    with pytest.raises(NotApplicable):
        feols(df, "y", ["s"], ["firm"], "firm")
    info = feols(df, "y", ["x1"], ["firm"], "firm", check_only=True)
    assert info["applies"] and info["rows"] == len(df) - 3
