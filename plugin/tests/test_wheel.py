"""The wheel the skills install holds exactly the engine's current sources, and the cores' provenance
record is current."""
import hashlib
import importlib.util
import json
from pathlib import Path

from tests_common import ROOT


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "plugin/tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_wheel_matches_sources():
    wheels = sorted((ROOT / "plugin/turbo-socialsci/wheels").glob("turbo_socialsci-*.whl"))
    assert len(wheels) == 1
    assert load("build_wheel").check(wheels[0]) > 10


def test_provenance_is_current():
    core = ROOT / "plugin/turbo-socialsci/engine/src/turbo_socialsci/core"
    prov = json.loads((core / "PROVENANCE.json").read_text())
    assert set(prov) == {"hdfe_speed", "hdfe_memory", "did_speed", "did_memory", "reclink_speed", "reclink_memory"}
    for name, rec in prov.items():
        assert hashlib.sha256((core / f"{name}.py").read_bytes()).hexdigest() == rec["sha256"], name
        assert hashlib.sha256((ROOT / rec["source"]).read_bytes()).hexdigest() == rec["frozen_sha256"], name
