"""The study's thermal gate and monitor (solutions/baselines/thermal_queue.py), importable: a sampler of
`pmset -g therm` (CPU_Speed_Limit), the xcpm CPU thermal level and the last sleep/wake times every 5 s,
a gate that waits for 60 s of unthrottled, cooled samples, and a summary of a time window."""
import json
import re
import subprocess
import threading
import time


def sample():
    therm = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True).stdout
    m = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", therm)
    speed = int(m.group(1)) if m else 100
    level = subprocess.run(["sysctl", "-n", "machdep.xcpm.cpu_thermal_level"], capture_output=True, text=True).stdout
    times = subprocess.run(["sysctl", "-n", "kern.sleeptime", "kern.waketime"], capture_output=True, text=True).stdout
    secs = [int(x) for x in re.findall(r"(?<![a-z])sec = (\d+)", times)]
    return time.time(), speed, int(level.strip() or 0), *(secs + [0, 0])[:2]


class Monitor:
    def __init__(self, log_path=None, period=5.0):
        self.samples, self.lock, self.log_path, self.period = [], threading.Lock(), log_path, period
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            s = sample()
            with self.lock:
                self.samples.append(s)
            if self.log_path:
                with open(self.log_path, "a") as fh:
                    fh.write(json.dumps(dict(t=round(s[0], 1), speed_limit=s[1], thermal_level=s[2],
                                             sleeptime=s[3], waketime=s[4])) + "\n")
            time.sleep(self.period)

    def gate(self, window=60.0):
        """Wait until the last `window` seconds of samples all show an unthrottled, cooled CPU."""
        t0 = time.time()
        while True:
            with self.lock:
                recent = [s for s in self.samples if s[0] >= time.time() - window]
            if len(recent) >= window / self.period - 2 and all(s[1] == 100 and s[2] <= 90 for s in recent):
                return time.time() - t0
            time.sleep(self.period)

    def summary(self, t_start, t_end):
        with self.lock:
            win = [s for s in self.samples if t_start - 5 <= s[0] <= t_end + 5]
        slow = [s for s in win if s[1] < 100]
        gaps = [b[0] - a[0] for a, b in zip(win, win[1:])]
        return dict(samples=len(win), throttled=len(slow), min_speed_limit=min((s[1] for s in win), default=None),
                    max_thermal_level=max((s[2] for s in win), default=None),
                    slept=bool(win) and (len({s[3] for s in win}) > 1 or any(t_start <= s[3] <= t_end for s in win)),
                    max_gap_s=round(max(gaps), 1) if gaps else None)
