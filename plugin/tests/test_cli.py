"""The command line on small files: results, --check, and exit code 3 when not applicable."""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from harness.cases import development_cases, generate_inputs
from tests_common import ENGINE, ROOT
from test_did import long_frame
from test_hdfe import frame


def cli(*args):
    env = dict(os.environ, PYTHONPATH=str(ENGINE))
    return subprocess.run([sys.executable, "-m", "turbo_socialsci.cli", *map(str, args)], capture_output=True,
                          text=True, env=env, cwd=ROOT)


def test_hdfe(tmp_path):
    p = tmp_path / "d.parquet"
    frame().to_parquet(p)
    r = cli("hdfe", "--data", p, "--y", "y", "--x", "x1", "zero", "x2", "--absorb", "firm", "state#year",
            "--cluster", "firm", "--out", tmp_path / "coef.csv")
    assert r.returncode == 0, r.stderr
    coef = pd.read_csv(tmp_path / "coef.csv", index_col=0)
    assert list(coef.index) == ["x1", "zero", "x2"] and bool(coef.loc["zero", "dropped"])
    r = cli("hdfe", "--data", p, "--y", "y", "--x", "x1", "--absorb", "firm", "--cluster", "firm", "--check")
    assert r.returncode == 0 and json.loads(r.stdout)["applies"]


def test_did(tmp_path):
    case = next(c for c in development_cases("002") if c["id"] == "dev-panel-0")
    kw = generate_inputs(ROOT / "002_csdid", "002", case)
    p = tmp_path / "panel.csv"
    long_frame(kw).to_csv(p, index=False)
    r = cli("did", "--data", p, "--y", "y", "--unit", "unit", "--time", "year", "--cohort", "cohort",
            "--covariates", "x0", "x1", "--n-boot", 199, "--out", tmp_path / "out")
    assert r.returncode == 0, r.stderr
    gt = pd.read_csv(tmp_path / "out" / "gt.csv")
    assert {"cohort", "time", "att", "se", "se_boot"} <= set(gt.columns) and len(gt) > 0
    assert json.loads(r.stderr.strip().splitlines()[-1])["crit"]["event_with_avgs"] > 0
    bad = long_frame(kw)
    pd.concat([bad, bad.head(2)]).to_csv(tmp_path / "dup.csv", index=False)
    r = cli("did", "--data", tmp_path / "dup.csv", "--y", "y", "--unit", "unit", "--time", "year", "--cohort", "cohort")
    assert r.returncode == 3 and "not applicable" in r.stderr


def test_reclink(tmp_path):
    case = next(c for c in development_cases("003") if c["id"] == "dev-random-1")
    kw = generate_inputs(ROOT / "003_reclink", "003", case)
    M = pd.DataFrame(kw["master"]).assign(idm=lambda d: np.arange(len(d)))
    U = pd.DataFrame(kw["using"]).assign(idu=lambda d: np.arange(len(d)))
    M.to_parquet(tmp_path / "m.parquet")
    U.to_parquet(tmp_path / "u.parquet")
    base = ["reclink", "--master", tmp_path / "m.parquet", "--using", tmp_path / "u.parquet", "--idmaster", "idm",
            "--idusing", "idu", "--wmatch", *kw["wmatch"], "--wnomatch", *kw["wnomatch"], "--required", "state"]
    r = cli(*base, "--vars", *kw["vars"], "--out", tmp_path / "m.csv")
    assert r.returncode == 0, r.stderr
    assert list(pd.read_csv(tmp_path / "m.csv").columns) == ["idm", "idu", "score"]
    r = cli(*base[:9], "--wmatch", 1, 1, 1, "--wnomatch", 1, 1, 1, "--vars", "name", "city", "state")
    assert r.returncode == 3
