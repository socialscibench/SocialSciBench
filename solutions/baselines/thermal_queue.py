"""Sequential measurement queue with a thermal gate (study tooling, macOS).

usage: thermal_queue.py PLAN.json OUT_DIR

PLAN.json is a list of steps, run in order, one at a time:
  {"kind": "compare", "task": "001", "solution": "fixest-r", "threads": 6, "timeout": 3600,
   "cases": ["104", "105"], "out": "001-fixest-r-t6.jsonl"}       compare.py, one case per invocation
                     ("not_comparable": ["se_boot_", "crit_"] passes compare.py --not-comparable)
  {"kind": "evaluate", "args": ["--task", "002", ...]}            python -m harness evaluate ARGS
                                       ("python": "baseline" runs it with the baselines' interpreter)
  {"kind": "cmd", "args": [...], "stdout": PATH}                    any other command (diagnostics)
"refs" (key of a step, or REFS in the environment) is the directory with one saved-reference
directory per task; "out" may be an absolute path (another results directory). A step with
"stop_on_sleep": true ends the whole queue at the first sleep of the host instead of measuring
again (for loads that the host cannot sustain).

A monitor thread samples `pmset -g therm` (CPU_Speed_Limit), the xcpm CPU thermal level and the
last sleep and wake times every 5 s into OUT_DIR/thermal.log. Before each case the queue waits until
the speed limit has been 100 and the thermal level at most 90 for 60 s (the level was 33 on the idle
host, and 119 with the speed limited to 63 under six-thread load). After each compare or evaluate
case it summarizes the samples of that case's window in OUT_DIR/thermal-cases.jsonl (share of
samples with a speed limit below 100, the lowest limit, sleeps, gaps in the monitor). An attempt
that overlapped a sleep or a monitor gap is listed in OUT_DIR/exclusions.jsonl and measured again
after cooling, at most twice more (the last attempt is kept and flagged when all are affected).
Throttling is recorded, not retried (it is marked "severe" below a limit of 70 or when more than
half of the samples are throttled; THERMAL_RETRY=1 also repeats those attempts, as the queue did
until 2026-10-24 22:40 PDT): on this host it occurs under one-thread load too (limits of 71-80% in
most samples of long cases). Everything is logged to
OUT_DIR/queue.log. Keep the host otherwise idle; run under `caffeinate -i -s`.
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BPY = str(ROOT / "solutions" / "baselines" / ".venv" / "py" / "bin" / "python")
PY = str(ROOT / "solutions" / ".venv" / "bin" / "python")
plan = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2]).resolve()
out.mkdir(parents=True, exist_ok=True)
LOG, THERM, EXCL = out / "queue.log", out / "thermal.log", out / "exclusions.jsonl"
samples = []  # (epoch, speed_limit, thermal_level, sleeptime, waketime)
# Repeating severely throttled attempts did not help on this host (2026-10-24: the speed limit stayed
# at 71-76% through all three attempts of long cases), so by default only sleeps are repeated;
# throttling is recorded per case (thermal-cases.jsonl, "severe"). THERMAL_RETRY=1 restores retries.
RETRY_THROTTLED = os.environ.get("THERMAL_RETRY") == "1"
lock = threading.Lock()


def log(msg):
    with LOG.open("a") as fh:
        fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {msg}\n")


def sample():
    therm = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True).stdout
    m = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", therm)
    speed = int(m.group(1)) if m else 100  # no notification recorded: unthrottled
    level = subprocess.run(["sysctl", "-n", "machdep.xcpm.cpu_thermal_level"], capture_output=True, text=True).stdout
    times = subprocess.run(["sysctl", "-n", "kern.sleeptime", "kern.waketime"], capture_output=True, text=True).stdout
    secs = [int(x) for x in re.findall(r"(?<![a-z])sec = (\d+)", times)]
    return time.time(), speed, int(level.strip() or 0), *(secs + [0, 0])[:2]


def monitor():
    while True:
        s = sample()
        with lock:
            samples.append(s)
        with THERM.open("a") as fh:
            fh.write(json.dumps(dict(t=round(s[0], 1), speed_limit=s[1], thermal_level=s[2],
                                     sleeptime=s[3], waketime=s[4])) + "\n")
        time.sleep(5)


def gate():
    """Wait until the last 60 s of samples all show an unthrottled, cooled CPU."""
    t0 = time.time()
    while True:
        with lock:
            recent = [s for s in samples if s[0] >= time.time() - 60]
        if len(recent) >= 10 and all(s[1] == 100 and s[2] <= 90 for s in recent):
            if time.time() - t0 > 70:
                log(f"gate: waited {time.time() - t0:.0f} s for an unthrottled CPU")
            return
        time.sleep(5)


def thermal(t_start, t_end):
    """Summary of the monitor samples in [t_start, t_end] and whether the attempt must be repeated."""
    with lock:
        win = [s for s in samples if t_start - 5 <= s[0] <= t_end + 5]
    slow = [s for s in win if s[1] < 100]
    gaps = [b[0] - a[0] for a, b in zip(win, win[1:])]
    stats = dict(samples=len(win), throttled=len(slow), min_speed_limit=min((s[1] for s in win), default=None),
                 max_thermal_level=max((s[2] for s in win), default=None),
                 slept=bool(win) and (len({s[3] for s in win}) > 1 or any(t_start <= s[3] <= t_end for s in win)),
                 max_gap_s=round(max(gaps), 1) if gaps else None)
    reasons = []
    if stats["slept"]:
        reasons.append("host slept")
    if gaps and max(gaps) > 30:
        reasons.append(f"monitor gap of {max(gaps):.0f} s (possible sleep)")
    stats["severe"] = bool(slow) and (stats["min_speed_limit"] < 70 or len(slow) > len(win) / 2)
    if stats["severe"] and RETRY_THROTTLED:
        reasons.append(f"severe throttling: speed limit down to {stats['min_speed_limit']} in {len(slow)} of {len(win)} samples")
    return stats, "; ".join(reasons)


def record(entry):
    with (out / "thermal-cases.jsonl").open("a") as fh:
        fh.write(json.dumps(entry) + "\n")


def run(cmd, stdout=None):
    log("run: " + " ".join(map(str, cmd)) + (f" > {stdout}" if stdout else ""))
    with open(stdout or LOG, "a") as fh:
        r = subprocess.run(list(map(str, cmd)), cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    log(f"exit {r.returncode}")
    return r.returncode


def last_row(path):
    lines = path.read_text().splitlines() if path.exists() else []
    return json.loads(lines[-1]) if lines else {}


def latest_run():
    runs = sorted((ROOT / "runs").glob("2026*T*Z-*"), key=lambda p: p.stat().st_mtime)
    return str(runs[-1].relative_to(ROOT)) if runs else None


threading.Thread(target=monitor, daemon=True).start()
time.sleep(1)
log(f"queue start: {len(plan)} steps; host {os.uname().nodename}")
for step in plan:
    refs = step.get("refs") or os.environ.get("REFS")
    if step["kind"] == "compare":
        res = Path(step["out"]) if Path(step["out"]).is_absolute() else out / step["out"]
        res.parent.mkdir(parents=True, exist_ok=True)
        for case in step["cases"]:
            for attempt in range(3):
                gate()
                t0 = time.time()
                run([BPY, "solutions/baselines/compare.py", "--task", step["task"],
                     "--solution", f"solutions/{step['task']}/{step['solution']}/solution.json", "--cases", case,
                     "--threads", step["threads"], "--warmups", 0, "--repeats", 1, "--timeout", step["timeout"],
                     "--refdir", f"{refs}/{step['task']}", "--out", res, "--python", BPY,
                     *(["--not-comparable", *step["not_comparable"]] if step.get("not_comparable") else [])])
                stats, why = thermal(t0, time.time())
                row = last_row(res)
                record(dict(file=str(res.relative_to(ROOT)) if res.is_relative_to(ROOT) else str(res), case=case,
                            created=row.get("created"), attempt=attempt + 1, retried=bool(why) and attempt < 2, **stats))
                if not why:
                    break
                if step.get("stop_on_sleep") and stats["slept"]:
                    log(f"host slept during {res.name} {case}: stopping the queue (stop_on_sleep)")
                    with (res.parent / "exclusions.jsonl").open("a") as fh:
                        fh.write(json.dumps(dict(file=res.name, case=case, created=row.get("created"), attempt=attempt + 1,
                                                 final=False, reason=why + "; queue stopped")) + "\n")
                    log("QUEUE STOPPED")
                    sys.exit(3)
                entry = dict(file=res.name, case=case, created=row.get("created"), attempt=attempt + 1,
                             final=attempt == 2, reason=why)
                with (res.parent / "exclusions.jsonl").open("a") as fh:     # next to the results file
                    fh.write(json.dumps(entry) + "\n")
                log(f"affected: {entry}")
                if attempt == 2:
                    break
                time.sleep(120)
            if step["threads"] > 1:
                time.sleep(60)
    elif step["kind"] == "evaluate":
        for attempt in range(3):
            gate()
            t0 = time.time()
            run([BPY if step.get("python") == "baseline" else PY, "-m", "harness", "evaluate", *step["args"]])
            stats, why = thermal(t0, time.time())
            record(dict(run=latest_run(), args=step["args"], attempt=attempt + 1, retried=bool(why) and attempt < 2, **stats))
            if not why:
                log(f"evaluate run {latest_run()}")
                break
            if step.get("stop_on_sleep") and stats["slept"]:
                with EXCL.open("a") as fh:
                    fh.write(json.dumps(dict(run=latest_run(), args=step["args"], attempt=attempt + 1, final=False,
                                             reason=why + "; queue stopped")) + "\n")
                log(f"host slept during evaluate run {latest_run()}: stopping the queue (stop_on_sleep)")
                log("QUEUE STOPPED")
                sys.exit(3)
            entry = dict(run=latest_run(), args=step["args"], attempt=attempt + 1, final=attempt == 2, reason=why)
            with EXCL.open("a") as fh:
                fh.write(json.dumps(entry) + "\n")
            log(f"affected: {entry}")
            if attempt == 2:
                break
            time.sleep(120)
    else:
        gate()
        run(step["args"], step.get("stdout"))
log("QUEUE DONE")
