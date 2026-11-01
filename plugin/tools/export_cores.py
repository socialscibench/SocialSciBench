"""Copy the frozen candidates' computational code into the engine package, and report how each core
differs from its frozen source.

usage: export_cores.py FREEZE_DIR [--check]     (repository root; FREEZE_DIR holds the freezes of 2026-10-26)

Without --check, every core that does not exist yet is copied from its candidate, after its SHA-256 is
checked against the freeze record (so the copy is the frozen code). With --check, nothing is copied:
each core's provenance and current hash are written to core/PROVENANCE.json, and its unified diff
against the frozen source to plugin/tools/patches/<core>.diff (empty for an unchanged core).
"""
import difflib
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "plugin/turbo-socialsci/engine/src/turbo_socialsci/core"
PATCHES = ROOT / "plugin/tools/patches"

# core module -> (freeze name, file in the frozen snapshot, candidate file in the repository)
CORES = {
    "hdfe_speed": ("001-schur-cg-t1", "solution/solution.py", "solutions/001/schur-cg/solution.py"),
    "hdfe_memory": ("001-schur-lean-t1", "solution/schur_lean.py", "solutions/001/schur-lean/schur_lean.py"),
    "did_speed": ("002-cell-fused-t1", "solution/solution.py", "solutions/002/cell-fused/solution.py"),
    "did_memory": ("002-cell-lean-t1", "solution/solution.py", "solutions/002/cell-lean/solution.py"),
    "reclink_speed": ("003-block-index-t1", "solution/solution.py", "solutions/003/block-index/solution.py"),
    "reclink_memory": ("003-block-lean-t1", "solution/block_lean.py", "solutions/003/block-lean/block_lean.py"),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    freezes = Path(sys.argv[1]).expanduser()
    check = "--check" in sys.argv[2:]
    prov = {}
    for core, (fz, snap, repo) in CORES.items():
        record = json.loads((freezes / fz / "freeze.json").read_text())
        frozen = record["source_sha256"][snap]
        target = CORE / f"{core}.py"
        if not check:
            if sha256(ROOT / repo) != frozen:
                sys.exit(f"{repo} differs from freeze {fz}; check out the frozen commit {record['git_commit']}")
            if not target.exists():
                target.write_bytes((ROOT / repo).read_bytes())
                print(f"copied {repo} -> {target.relative_to(ROOT)}")
            continue
        current = sha256(target)
        prov[core] = dict(freeze=fz, freeze_id=record["freeze_id"], git_commit=record["git_commit"],
                          source=repo, frozen_sha256=frozen, sha256=current, modified=current != frozen)
        PATCHES.mkdir(parents=True, exist_ok=True)
        old = (ROOT / repo).read_text().splitlines(keepends=True)
        new = target.read_text().splitlines(keepends=True)
        diff = "".join(difflib.unified_diff(old, new, f"frozen/{repo}", f"engine/core/{core}.py"))
        (PATCHES / f"{core}.diff").write_text(diff)
        plus = sum(1 for ln in diff.splitlines() if ln.startswith("+") and not ln.startswith("+++"))
        minus = sum(1 for ln in diff.splitlines() if ln.startswith("-") and not ln.startswith("---"))
        print(f"{core:15s} {'modified' if current != frozen else 'unchanged':9s} +{plus} -{minus}  ({repo})")
    if check:
        (CORE / "PROVENANCE.json").write_text(json.dumps(prov, indent=2) + "\n")


if __name__ == "__main__":
    main()
