"""Rerun the code an agent wrote, the way the kits time tools: on this host, alone, after the thermal gate,
with 4 threads, recording the process tree's peak memory.

usage: rerun.py RUN_NAME --cmd "COMMAND" [--limit-hours 2] [--threads 4]
COMMAND is the one command the agent's note gives, run from its workspace. The agent's own result files
(*.csv outside data/) are saved to META/agent_results/ first; the rerun's are saved to META/rerun_results/.
The run is stopped when no result file changes for --limit-hours (one estimate over the limit). Memory is
the summed RSS of the command's process tree, sampled every 0.25 s; result-file changes are timestamped,
so each estimate's peak can be read from the timeline (META/rerun_memory.jsonl, META/rerun_files.jsonl).
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from thermal import Monitor  # noqa: E402

CFG = json.loads((HERE / "kits.json").read_text())


def results(ws):
    return {str(p.relative_to(ws)): p for p in ws.rglob("*.csv")
            if not p.relative_to(ws).parts[0] in ("data", ".venv", "venv") and ".venv" not in p.parts}


def tree_rss(root):
    out = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,rss="], capture_output=True, text=True).stdout.split("\n")
    kids, rss = {}, {}
    for ln in out:
        f = ln.split()
        if len(f) == 3:
            pid, ppid, r = map(int, f)
            kids.setdefault(ppid, []).append(pid)
            rss[pid] = r
    total, stack = 0, [root]
    while stack:
        p = stack.pop()
        total += rss.get(p, 0)
        stack.extend(kids.get(p, []))
    return total * 1024


def main():
    name = sys.argv[1]
    cmd = sys.argv[sys.argv.index("--cmd") + 1]
    limit = float(sys.argv[sys.argv.index("--limit-hours") + 1]) * 3600 if "--limit-hours" in sys.argv else 7200.0
    threads = sys.argv[sys.argv.index("--threads") + 1] if "--threads" in sys.argv else "4"
    root = Path(CFG["runs_root"])
    ws, meta = root / name, root / f"{name}.meta"
    saved = meta / "agent_results"
    if not saved.exists():
        for rel, p in results(ws).items():
            (saved / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, saved / rel)
    env = dict(os.environ)
    env = {k: v for k, v in env.items() if not k.startswith("CLAUDE")}
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
              "NUMBA_NUM_THREADS", "RAYON_NUM_THREADS", "POLARS_MAX_THREADS"):
        env[v] = threads
    mon = Monitor(meta / "rerun_thermal.log")
    time.sleep(1)
    waited = mon.gate()
    seen = {rel: p.stat().st_mtime for rel, p in results(ws).items()}
    t0 = time.time()
    p = subprocess.Popen(["/bin/zsh", "-c", cmd], cwd=ws, env=env, stdout=open(meta / "rerun_stdout.txt", "w"),
                         stderr=subprocess.STDOUT, start_new_session=True)
    peak, last_change, stopped, out_size = 0, t0, False, 0
    with open(meta / "rerun_memory.jsonl", "w") as mem, open(meta / "rerun_files.jsonl", "w") as files:
        while p.poll() is None:
            r = tree_rss(p.pid)
            peak = max(peak, r)
            mem.write(json.dumps(dict(t=round(time.time() - t0, 2), rss=r)) + "\n")
            size = (meta / "rerun_stdout.txt").stat().st_size     # printed progress also counts as activity
            if size != out_size:
                out_size, last_change = size, time.time()
                files.write(json.dumps(dict(t=round(time.time() - t0, 2), file="<stdout>", bytes=size)) + "\n")
            for rel, f in results(ws).items():
                m = f.stat().st_mtime
                if seen.get(rel) != m:
                    seen[rel] = m
                    last_change = time.time()
                    files.write(json.dumps(dict(t=round(time.time() - t0, 2), file=rel,
                                                lines=sum(1 for _ in open(f)))) + "\n")
            if time.time() - last_change > limit:
                stopped = True
                os.killpg(p.pid, signal.SIGINT)
                time.sleep(30)
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGTERM)
                break
            time.sleep(0.25)
    code = p.wait()
    t1 = time.time()
    out = meta / "rerun_results"
    if out.exists():
        shutil.rmtree(out)
    for rel, f in results(ws).items():
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, out / rel)
    info = dict(cmd=cmd, threads=int(threads), exit_code=code, wall_s=round(t1 - t0, 1), peak_rss_bytes=peak,
                stopped_at_limit=stopped, gate_wait_s=round(waited, 1), thermal=mon.summary(t0, t1))
    (meta / "rerun.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
