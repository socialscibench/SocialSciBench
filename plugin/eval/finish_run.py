"""Move a finished run (workspace and record) from the shared runs root into the kits' storage, so that
the next agent run cannot see it.

usage: finish_run.py RUN_NAME
"""
import json
import shutil
import sys
from pathlib import Path

CFG = json.loads((Path(__file__).with_name("kits.json")).read_text())


def main():
    name = sys.argv[1]
    root = Path(CFG["runs_root"])
    dest = Path(CFG["storage"]).expanduser() / "agent-runs"
    dest.mkdir(parents=True, exist_ok=True)
    for part in (name, f"{name}.meta"):
        src = root / part
        if src.exists():
            shutil.move(str(src), str(dest / part))
            print(f"moved {src} -> {dest / part}")
    left = list(root.iterdir())
    print("runs root now empty" if not left else f"runs root still holds {[p.name for p in left]}")


if __name__ == "__main__":
    main()
