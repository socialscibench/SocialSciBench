"""The record-linkage interface (DataFrames with id columns) against the cores, on the evaluator's
public development fixtures."""
import numpy as np
import pandas as pd
import pytest

from harness.cases import development_cases, generate_inputs
from turbo_socialsci import NotApplicable, reclink
from turbo_socialsci.core import reclink_memory, reclink_speed
from tests_common import ROOT

TASK = ROOT / "003_reclink"


def fixture(name):
    case = next(c for c in development_cases("003") if c["id"] == name)
    return generate_inputs(TASK, "003", case)


def frames(kw):
    M = pd.DataFrame({v: kw["master"][v] for v in kw["vars"]})
    U = pd.DataFrame({v: kw["using"][v] for v in kw["vars"]})
    M.insert(0, "idm", np.arange(len(M)) * 3 + 100)
    U.insert(0, "idu", np.arange(len(U)) * 7 + 5000)
    return M, U


@pytest.mark.parametrize("name", ["dev-random-0", "dev-random-1", "dev-ties-missing-perfect", "dev-no-candidates"])
@pytest.mark.parametrize("variant", ["speed", "memory"])
def test_interface_matches_core(name, variant):
    kw = fixture(name)
    M, U = frames(kw)
    res = reclink(M, U, idmaster="idm", idusing="idu", vars=kw["vars"], wmatch=kw["wmatch"], wnomatch=kw["wnomatch"],
                  required=kw["required"], minscore=kw.get("minscore", 0.6), minbigram=kw.get("minbigram", 0.6),
                  variant=variant)
    core = reclink_speed if variant == "speed" else reclink_memory
    i, j, s, n_perfect = core.run(**kw)
    want = sorted(zip(np.asarray(i) * 3 + 100, np.asarray(j) * 7 + 5000, np.asarray(s, dtype=float)))
    got = sorted(zip(res["idm"], res["idu"], res["score"]))
    assert len(got) == len(want) and res.attrs["n_perfect"] == n_perfect
    for (a, b, x), (c, d, z) in zip(got, want):
        assert (a, b) == (c, d) and x == pytest.approx(z, rel=1e-12)


def test_missing_values_are_empty_strings():
    kw = fixture("dev-ties-missing-perfect")
    M, U = frames(kw)
    M2 = M.replace("", None)
    a = reclink(M, U, idmaster="idm", idusing="idu", vars=kw["vars"], wmatch=kw["wmatch"], wnomatch=kw["wnomatch"],
                required=kw["required"])
    b = reclink(M2, U, idmaster="idm", idusing="idu", vars=kw["vars"], wmatch=kw["wmatch"], wnomatch=kw["wnomatch"],
                required=kw["required"])
    assert a.equals(b)


def test_not_applicable():
    kw = fixture("dev-random-0")
    M, U = frames(kw)
    args = dict(idmaster="idm", idusing="idu", wmatch=kw["wmatch"][:3], wnomatch=kw["wnomatch"][:3], required=["state"])
    with pytest.raises(NotApplicable):                       # three variables: reclink2's other blocking rule
        reclink(M, U, vars=["name", "city", "state"], **args)
    M.loc[0, "name"] = "CAFÉ ÉLAN"
    with pytest.raises(NotApplicable):
        reclink(M, U, idmaster="idm", idusing="idu", vars=kw["vars"], wmatch=kw["wmatch"], wnomatch=kw["wnomatch"],
                required=kw["required"])
