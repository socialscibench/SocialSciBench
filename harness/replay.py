"""Internal replay entry point, launched with the saved snapshot as the import root."""
import sys
from pathlib import Path
from types import SimpleNamespace

from .bundles import verify_run
from .common import ROOT, task_info
from .runner import evaluate


def main():
    directory, metadata = verify_run(sys.argv[1])
    if ROOT != directory / "snapshot":
        raise ValueError("Replay must execute the saved evaluator snapshot")
    if metadata.get("final_evaluation") or any(c.get("suite") == "real-heldout" for c in metadata["cases"]):
        raise ValueError("Held-out replay requires another frozen final round")
    config = metadata["config"]
    options = SimpleNamespace(out=Path(sys.argv[2]), threads=config["threads"],
                              warmups=config["warmups"], repeats=config["repeats"],
                              timeout=config["timeout_s"], validation_seed=config["validation_seed"],
                              keep_outputs=config["keep_outputs"],
                              data_root=sys.argv[3] if len(sys.argv) > 3 else (metadata.get("dataset") or {}).get("root"),
                              suite=metadata.get("suite", "synthetic"),
                              replay_of=dict(run_id=metadata["run_id"], git_commit=metadata["git_commit"]))
    entry, manifest = task_info(metadata["task"])
    return evaluate(entry, manifest, ROOT / "solution" / "solution.json", metadata["cases"], options)


if __name__ == "__main__":
    raise SystemExit(main())
