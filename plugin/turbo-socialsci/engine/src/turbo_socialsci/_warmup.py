"""Compile (first use) or load (later uses) the Numba kernels on tiny synthetic problems, so that the
first real call in a process is not slowed by them. The compiled code is cached next to the
package (Numba's cache=True), so compilation happens once per installation. The calls go through
the public functions, so the libraries they import are loaded here too."""
import numpy as np


def warmup(which=("hdfe", "did", "reclink"), variant="speed"):
    import pandas as pd
    which = [which] if isinstance(which, str) else list(which)
    rng = np.random.default_rng(0)
    if "hdfe" in which:
        from .hdfe import feols
        n = 400
        df = pd.DataFrame(dict(g=rng.integers(0, 40, n), h=rng.integers(0, 8, n).astype(np.int16),
                               c=rng.integers(0, 10, n).astype(np.float64), d=rng.integers(0, 2, n).astype(np.int8),
                               x=rng.normal(size=n), y=rng.normal(size=n)))
        for ftype in (np.float64, np.float32):          # one fixed effect: both float types of the columns
            d = df.astype({"x": ftype, "y": ftype})
            feols(d, "y", ["x", "d"], ["g"], "c", variant=variant)
            feols(d, "y", ["x", "d"], ["g#h"], "h", variant=variant)
        feols(df, "y", ["x", "d"], ["g", "h"], "c", variant=variant)       # several fixed effects
    if "did" in which:
        from .did import att_gt
        N, T = 60, 5
        panel = pd.DataFrame(dict(u=np.repeat(np.arange(N), T), t=np.tile(np.arange(2000, 2000 + T), N),
                                  g=np.repeat(np.repeat([0, 2002, 2003], N // 3), T)))
        panel["y"] = rng.normal(size=len(panel))
        panel["x"] = rng.normal(size=len(panel))
        att_gt(panel, "y", "u", "t", "g", covariates=["x"], n_boot=99, variant=variant)
    if "reclink" in which:
        from .reclink import reclink
        m = pd.DataFrame({"name": ["ACME CORP", "BETA LLC"], "addr": ["1 MAIN ST", "2 OAK AVE"],
                          "city": ["AUSTIN", "BOSTON"], "zip": ["73301", "02108"], "state": ["TX", "MA"]})
        u = pd.concat([m, m.iloc[:1].assign(name="ACME CORPX")], ignore_index=True)
        m["idm"], u["idu"] = np.arange(len(m)), np.arange(len(u))
        kw = dict(idmaster="idm", idusing="idu", vars=["name", "addr", "city", "zip", "state"],
                  wmatch=[10, 8, 5, 5, 5], wnomatch=[2, 3, 4, 4, 4], required=["state"], variant=variant)
        reclink(m, u, **kw)                              # str columns (Arrow-backed in pandas 3)
        reclink(m.astype(object), u.astype(object), **kw)      # Python str objects
