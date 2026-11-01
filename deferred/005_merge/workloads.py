"""Deterministic synthetic workloads for task 005_merge.

Each workload is a pair of tables: a master table (in memory) and a using table, merged on
one or two keys as 1:1 or m:1, keeping all rows (Stata's default outer merge). Key coverage
overlaps partly, so all three _merge outcomes occur: master only, using only, and matched.
"""
import string

import numpy as np
import pandas as pd

# Numbered 1-6 from small to large. Each workload has a fixed seed.
WORKLOADS = [
    dict(id=1, seed=5001, design="sanity check", kind="1:1", keys=["id"], key_type="numeric",
         n_master=1_000, n_using=1_000, master_vars=3, using_vars=3),
    dict(id=2, seed=5002, design="firm-year 1:1 (Compustat + ExecuComp)", kind="1:1", keys=["firm", "year"],
         key_type="numeric", n_firms=10_000, n_years=20, using_firms=2_000, master_vars=20, using_vars=10),
    dict(id=3, seed=5003, design="firm-year m:1 firm data, string key (CUSIP)", kind="m:1", keys=["cusip"],
         key_type="string", n_firms=10_000, n_years=20, master_vars=20, using_vars=10),
    dict(id=4, seed=5004, design="stock-month m:1 firm-year (CRSP + Compustat)", kind="m:1", keys=["permno", "year"],
         key_type="numeric", n_firms=25_000, n_years=10, months_per_year=12, master_vars=5, using_vars=20),
    dict(id=5, seed=5005, design="job postings m:1 employers, string key", kind="m:1", keys=["employer"],
         key_type="string", n_master=10_000_000, n_employers=500_000, master_vars=5, using_vars=10),
    dict(id=6, seed=5006, design="stock-day m:1 stock-month (CRSP daily + monthly)", kind="m:1",
         keys=["permno", "month"], key_type="numeric", n_firms=8_000, n_months=120, days_per_month=21,
         master_vars=3, using_vars=5),
]


def _codes(rng, n, length):
    """n distinct uppercase alphanumeric codes of the given length (e.g. CUSIP-like ids)."""
    alphabet = np.array(list(string.ascii_uppercase + string.digits))
    out = set()
    while len(out) < n:
        out.update("".join(c) for c in rng.choice(alphabet, size=(n - len(out), length)))
    return np.array(sorted(out))


def _vars(rng, n, k, prefix):
    X = rng.standard_normal((n, k))
    X[rng.random((n, k)) < 0.03] = np.nan                     # some missing values
    return {f"{prefix}{j+1}": X[:, j] for j in range(k)}


def generate(w):
    rng = np.random.default_rng(w["seed"])
    if w["id"] == 1:
        mk = rng.permutation(1_200)[:1_000]
        uk = rng.permutation(1_200)[:1_000]
        master = pd.DataFrame({"id": mk, **_vars(rng, 1_000, 3, "m")})
        using = pd.DataFrame({"id": uk, **_vars(rng, 1_000, 3, "u")})
    elif w["id"] == 2:
        F, T = w["n_firms"], w["n_years"]
        master = pd.DataFrame({"firm": np.repeat(np.arange(F), T), "year": np.tile(np.arange(2000, 2000 + T), F)})
        # Using: 2,000 firms (some not in master) over 22 years (two years not in master)
        uf = rng.choice(np.arange(F + 200), w["using_firms"], replace=False)
        using = pd.DataFrame({"firm": np.repeat(uf, T + 2), "year": np.tile(np.arange(1999, 2001 + T), uf.size)})
        master = master.assign(**_vars(rng, len(master), w["master_vars"], "m"))
        using = using.assign(**_vars(rng, len(using), w["using_vars"], "u"))
    elif w["id"] == 3:
        F, T = w["n_firms"], w["n_years"]
        codes = _codes(rng, int(F * 1.05), 8)
        firms = rng.permutation(codes)
        master = pd.DataFrame({"cusip": np.repeat(firms[:F], T), **_vars(rng, F * T, w["master_vars"], "m")})
        ucodes = rng.permutation(np.r_[firms[:int(F * 0.95)], firms[F:]])          # 95% of master firms + extras
        using = pd.DataFrame({"cusip": ucodes, **_vars(rng, ucodes.size, w["using_vars"], "u")})
    elif w["id"] == 4:
        F, T, M = w["n_firms"], w["n_years"], w["months_per_year"]
        master = pd.DataFrame({"permno": np.repeat(np.arange(F), T * M),
                               "year": np.tile(np.repeat(np.arange(2010, 2010 + T), M), F)})
        uf = rng.choice(np.arange(int(F * 1.1)), int(F * 0.8), replace=False)        # 80% coverage + extra firms
        using = pd.DataFrame({"permno": np.repeat(uf, T), "year": np.tile(np.arange(2010, 2010 + T), uf.size)})
        master = master.assign(**_vars(rng, len(master), w["master_vars"], "m"))
        using = using.assign(**_vars(rng, len(using), w["using_vars"], "u"))
    elif w["id"] == 5:
        E = w["n_employers"]
        codes = _codes(rng, int(E * 1.05), 10)
        emp = rng.permutation(codes)
        # Postings: employer sizes are skewed (a few employers post a lot)
        weights = rng.pareto(1.2, E) + 1
        post_emp = emp[:E][rng.choice(E, w["n_master"], p=weights / weights.sum())]
        master = pd.DataFrame({"employer": post_emp, **_vars(rng, w["n_master"], w["master_vars"], "m")})
        ucodes = rng.permutation(np.r_[emp[:int(E * 0.9)], emp[E:]])
        using = pd.DataFrame({"employer": ucodes, **_vars(rng, ucodes.size, w["using_vars"], "u")})
    elif w["id"] == 6:
        F, Mo, D = w["n_firms"], w["n_months"], w["days_per_month"]
        master = pd.DataFrame({"permno": np.repeat(np.arange(F), Mo * D),
                               "month": np.tile(np.repeat(np.arange(Mo), D), F)})
        using = pd.DataFrame({"permno": np.repeat(np.arange(F + 500), Mo + 12),
                              "month": np.tile(np.arange(-12, Mo), F + 500)})
        keep = rng.random(len(using)) < 0.97                                        # a few stock-months missing
        using = using[keep].reset_index(drop=True)
        master = master.assign(**_vars(rng, len(master), w["master_vars"], "m"))
        using = using.assign(**_vars(rng, len(using), w["using_vars"], "u"))
    return dict(master=master, using=using, keys=w["keys"], kind=w["kind"])


def axes(w, d):
    m, u = d["master"], d["using"]
    return dict(kind=w["kind"], keys=" ".join(w["keys"]), key_type=w["key_type"], n_master=len(m), n_using=len(u),
                master_cols=m.shape[1], using_cols=u.shape[1])
