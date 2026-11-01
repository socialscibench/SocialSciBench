"""Grade a run's result files with the kit's own grader (reference/grade.py), outside the workspace.

usage: grade_run.py KIT RUN_NAME [agent|rerun]     (default: both)
The kit's folder in the kits' storage must hold reference/ (fetched after the plugin was frozen). The
grader runs from the kit's folder with the candidates' interpreter, on copies of the result files saved
in the run's record (META/agent_results/ or META/rerun_results/). Its output goes to META/grade_<which>.txt.
Invocations follow the kits' READMEs: 001 and 002 take one results file; 003 takes a matches file and its
size.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CFG = json.loads((HERE / "kits.json").read_text())
PY = str(ROOT / "solutions/.venv/bin/python")


def find_meta(name):
    for base in (Path(CFG["runs_root"]), Path(CFG["storage"]).expanduser() / "agent-runs"):
        if (base / f"{name}.meta").exists():
            return base / f"{name}.meta"
    sys.exit(f"no record for {name}")


def targets(kit, files):
    """(file, extra args) for every gradable result file."""
    out = []
    for f in files:
        n = f.name
        if kit == "003":
            m = re.search(r"_matches_(\d+)\.csv$", n)
            if m:
                out.append((f, [m.group(1)]))
        elif kit == "002":
            if n.endswith("_estimates.csv"):
                out.append((f, []))
        elif kit == "001":
            if n == "results.csv" or n.endswith("_timings.csv"):
                out.append((f, []))
    return out


def main():
    kit, name = sys.argv[1], sys.argv[2]
    which = sys.argv[3:] or ["agent", "rerun"]
    meta = find_meta(name)
    kit_dir = Path(CFG["storage"]).expanduser() / kit
    if not (kit_dir / "reference" / "grade.py").exists():
        sys.exit(f"{kit_dir}/reference/grade.py missing (fetch_kit.py {kit} reference, after the freeze)")
    for w in which:
        res = meta / f"{w}_results"
        if not res.exists():
            print(f"{w}: no saved results")
            continue
        lines = []
        for f, extra in targets(kit, sorted(res.rglob("*.csv"))):
            r = subprocess.run([PY, "-I", "reference/grade.py", str(f), *extra], cwd=kit_dir,
                               capture_output=True, text=True, timeout=7200)
            lines.append(f"### {f.relative_to(res)} {' '.join(extra)} (exit {r.returncode})\n{r.stdout}{r.stderr}")
        (meta / f"grade_{w}.txt").write_text("\n".join(lines))
        print(f"{w}: graded {len(lines)} files -> {meta / f'grade_{w}.txt'}")


if __name__ == "__main__":
    main()
