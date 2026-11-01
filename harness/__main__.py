"""Run with python -m harness from the repository root."""
import argparse
import math
import shutil
from pathlib import Path

from .common import ROOT, read_json, safe_relative, task_hook, task_info, tasks, write_json
from .datasets import SUITES, catalog, select_cases, stage_data


def workload_id(text):
    """Workload IDs are integers (101) or strings (a catalog may use IDs such as 503a)."""
    return int(text) if text.isdigit() else text


def measurement_arguments(command, repeats=None):
    command.add_argument("--threads", type=int, default=1)
    command.add_argument("--warmups", type=int, default=1)
    command.add_argument("--repeats", type=int, default=repeats)
    command.add_argument("--timeout", type=float, default=300, help="Wall-time limit per worker, including setup")
    command.add_argument("--validation-seed", type=int, default=1729)
    command.add_argument("--keep-outputs", action="store_true")


def validate_measurement(args):
    if (args.threads < 1 or args.repeats < 1 or args.warmups < 0 or args.validation_seed < 0 or
            not math.isfinite(args.timeout) or args.timeout <= 0):
        raise ValueError("Threads, repeats and timeout must be positive; warmups and seed must be nonnegative")


def parser():
    ap = argparse.ArgumentParser(description="SocialSciBench candidate evaluator (001, 002, 003)")
    commands = ap.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="List active tasks and workload IDs")
    listing.add_argument("--task")
    listing.add_argument("--suite", choices=SUITES, default="synthetic")
    stage = commands.add_parser("stage", help="Export a workspace without past results or solutions")
    stage.add_argument("--task", required=True)
    stage.add_argument("--out", required=True, type=Path)
    stage.add_argument("--suite", choices=SUITES[:-1], default="synthetic")
    stage.add_argument("--ids", nargs="+", type=workload_id)
    stage.add_argument("--data-root", type=Path)
    starter = commands.add_parser("init-solution", help="Create a runnable reference-based candidate")
    starter.add_argument("--task", required=True)
    starter.add_argument("--out", required=True, type=Path)
    starter.add_argument("--name", default="reference-starter")
    export = commands.add_parser("export", help="Export a completed run as a portable publication bundle")
    export.add_argument("--run", required=True, type=Path)
    export.add_argument("--out", required=True, type=Path, help="New bundle directory; outputs are excluded")
    replay = commands.add_parser("replay", help="Repeat a saved run using its snapshot and settings")
    replay.add_argument("--run", required=True, type=Path)
    replay.add_argument("--out", type=Path, default=ROOT / "runs", help="Parent of a new unique run")
    replay.add_argument("--data-root", type=Path, help="Relocated package data directory")
    freeze = commands.add_parser("freeze", help="Freeze code, evaluator, held-out cases and settings without running them")
    freeze.add_argument("--task", required=True)
    freeze.add_argument("--solution", required=True, type=Path)
    freeze.add_argument("--out", required=True, type=Path, help="New freeze directory outside the repository")
    freeze.add_argument("--ids", nargs="+", type=workload_id, help="Held-out IDs; default: every held-out workload")
    freeze.add_argument("--rounds", type=int, choices=(1, 2, 3), default=3)
    measurement_arguments(freeze, repeats=5)
    final = commands.add_parser("final", help="Run one recorded evaluation round from an existing freeze")
    final.add_argument("--freeze", required=True, type=Path)
    final.add_argument("--data-root", required=True, type=Path, help="External data directory containing heldout/")
    final.add_argument("--out", type=Path, help="External run parent; default: a runs/ sibling of --data-root")
    final.add_argument("--note", default="")
    evaluate = commands.add_parser("evaluate", help="Validate and time a submitted solution")
    evaluate.add_argument("--task", required=True)
    evaluate.add_argument("--solution", required=True, type=Path, help="Path to solution.json")
    evaluate.add_argument("--suite", choices=SUITES, default="synthetic")
    evaluate.add_argument("--data-root", type=Path, help="Data directory containing the selected split")
    evaluate.add_argument("--n-regressors", type=int, help="Task 001 workload 104: 5, 20 or 50")
    selection = evaluate.add_mutually_exclusive_group()
    selection.add_argument("--smoke", action="store_true", help="Development fixtures only; no score")
    selection.add_argument("--ids", nargs="+", type=workload_id, help="Official workload IDs; default: 1")
    selection.add_argument("--all-workloads", action="store_true")
    evaluate.add_argument("--out", type=Path, default=ROOT / "runs", help="Parent of a new unique run directory")
    measurement_arguments(evaluate)
    return ap


def main(argv=None):
    ap = parser()
    args = ap.parse_args(argv)
    try:
        if args.command == "list":
            entries = [task_info(args.task)[0]] if args.task else tasks()
            for entry in entries:
                entry, manifest = task_info(entry["id"])
                if args.suite != "synthetic" and "real_workload_catalog" not in manifest and not args.task:
                    continue
                if args.suite != "synthetic" and not args.task:
                    available = read_json(ROOT / entry["directory"] / manifest["real_workload_catalog"])
                    if not any(w["suite"] == args.suite for w in available["workloads"]):
                        continue
                workloads = catalog(ROOT / entry["directory"], manifest, args.suite)
                print(f"{entry['id']}  {entry['name']}  workloads: {', '.join(str(w['id']) for w in workloads)}")
            return 0
        if args.command in ("export", "replay"):
            from .bundles import export_run, replay_run
            if args.command == "export":
                print(export_run(args.run, args.out))
                return 0
            return replay_run(args.run, args.out, args.data_root)
        if args.command == "final":
            from .frozen import run_final
            return run_final(args.freeze, args.data_root, args.out, args.note)
        entry, manifest = task_info(args.task)
        task_dir = ROOT / entry["directory"]
        if args.command == "freeze":
            from .frozen import create_freeze
            validate_measurement(args)
            print(create_freeze(entry, manifest, args))
        elif args.command == "stage":
            out = args.out.resolve()
            if out.is_relative_to(ROOT / "harness"):
                raise ValueError("Stage output must be outside the evaluator source directory")
            out.mkdir(parents=True, exist_ok=False)
            target = out / entry["directory"]
            target.mkdir()
            for name in manifest["prompt_files"]:
                dest = target / safe_relative(name)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(task_dir / name, dest)
            shutil.copytree(ROOT / "harness", out / "harness", ignore=shutil.ignore_patterns("__pycache__"))
            for name in ("requirements.txt", "requirements.lock", "AGENTS.md"):
                shutil.copy2(ROOT / name, out / name)
            # Keep the exported workflow scoped to its single registered task.
            agents = out / "AGENTS.md"
            agents.write_text(agents.read_text().replace("001, 002, and 003", entry["id"]))
            guide = out / "harness" / "README.md"
            # Scope the generic protocol examples, not task-001-only real-data variants.
            generic = guide.read_text().split("The evaluator supports tasks", 1)[1]
            generic = "The evaluator supports tasks" + generic
            generic = generic.replace("--task 001", f"--task {entry['id']}").replace("solutions/001/", f"solutions/{entry['id']}/")
            generic = generic.replace("socialscibench-001", f"socialscibench-{entry['id']}")
            guide.write_text("# Staged evaluation workspace\n\n" + generic)
            write_json(out / "tasks.json", dict(schema_version=1, tasks=[entry]))
            staged_data = stage_data(out, entry, manifest, args.suite, args.ids, args.data_root)
            if staged_data:
                command = (f"python -m harness evaluate --task {entry['id']} --solution "
                           f"solutions/{entry['id']}/first/solution.json --suite {args.suite} --ids "
                           + " ".join(map(str, staged_data["workload_ids"])))
                guide.write_text(guide.read_text() + "\n## Selected real inputs\n\n" +
                    f"This workspace includes only {args.suite} IDs {staged_data['workload_ids']}. "
                    "Run the public --smoke check first, then select the inputs explicitly:\n\n```bash\n" +
                    command + "\n```\n\nReal development uses five measured calls by default; "
                    "real-smoke is an unscored compatibility check. Read the task's DATA.md. "
                    "Input files are checksum-verified outside timing. Held-out data and catalog entries "
                    "are absent, so final evaluation must be performed by the study maintainer. "
                    "Only give agents this directory; filesystem/network isolation requires an external runner.\n")
            write_json(out / "stage.json", dict(task=entry["id"], contract_version=manifest["contract_version"],
                                               dataset=staged_data,
                                               visibility="task, public evaluator and selected development inputs only"))
            (out / "README.md").write_text(
                f"# SocialSciBench task {entry['id']}\n\nRead [{entry['directory']}/task.md]({entry['directory']}/task.md), "
                "`AGENTS.md`, and [harness/README.md](harness/README.md). "
                "This workspace contains no historical results or prior candidates.\n")
            print(out)
        elif args.command == "init-solution":
            args.out.mkdir(parents=True, exist_ok=False)
            reference_file, reference_function = manifest["reference_entry_point"].split("::")
            shutil.copy2(task_dir / reference_file, args.out / "solution.py")
            shutil.copy2(ROOT / "requirements.lock", args.out / "requirements.lock")
            for name in ("research.md", "build.md"):
                shutil.copy2(ROOT / "harness" / "templates" / name, args.out / name)
            write_json(args.out / "solution.json", dict(schema_version=1, name=args.name, task=entry["id"],
                       entry_point=f"solution.py::{reference_function}", sources=["solution.py"],
                       artifacts=["research.md", "requirements.lock", "build.md"], target_hardware="cpu",
                       dependencies=["numpy", "scipy", "pandas"], description="Unoptimized reference-based starter"))
            print(args.out / "solution.json")
        else:
            if args.suite == "real-heldout":
                raise ValueError("Held-out evaluation requires 'harness freeze' followed by 'harness final'")
            if args.smoke and (args.suite != "synthetic" or args.n_regressors is not None):
                raise ValueError("--smoke selects public fixtures; use --suite real-smoke for workload 109")
            args.repeats = args.repeats if args.repeats is not None else (3 if args.suite == "synthetic" else 5)
            validate_measurement(args)
            from .runner import evaluate
            development_cases = task_hook(task_dir, manifest, "development_entry_point")
            cases = development_cases(entry["id"], args.validation_seed)
            if not args.smoke:
                ids = args.ids or (manifest["default_workload_ids"] if args.suite == "synthetic" else None)
                cases += select_cases(entry, manifest, args.suite, ids, args.all_workloads, args.n_regressors)
            return evaluate(entry, manifest, args.solution, cases, args)
    except (ValueError, OSError) as error:
        ap.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
