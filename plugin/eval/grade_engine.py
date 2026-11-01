"""Grade the engine's own results on a kit with the kit's grader (reference/grade.py): the results that
bench_engine.py saved, written in the kit's result formats (task/TASK.md), then graded as grade_run.py
grades an agent's files.

usage: grade_engine.py KIT RESULTS_DIR OUT_DIR [--variant speed] [--tool NAME]
RESULTS_DIR: a bench_engine.py output directory (one saved result per size and call, and timings.jsonl).
OUT_DIR receives the results files and grade.txt, the grader's output. Both come from licensed data:
keep OUT_DIR outside the repository. Grade only a frozen engine, since the kit's reference/ holds the
answers.
"""
import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bench_engine import CALLS_002, KITS  # noqa: E402
from grade_run import PY, targets  # noqa: E402
from summarize import verdicts  # noqa: E402


def num(x):
    return "" if x is None or x != x else repr(float(x))


def write_001(res, secs, out, head):
    rows = ["tool,version,threads,column,size_pct,rep,seconds,N,b_lrrfsim01W,se_lrrfsim01W,b_lfcit01W,"
            "se_lfcit01W,r2"]
    for (size, call), d in sorted(res.items(), key=lambda kv: (-kv[0][0], kv[0][1])):
        c = d["coef"]
        val = lambda v, k: num(c.loc[v, k]) if call != 1 or v != "lfcit01W" else ""
        rows.append(",".join([*head, str(call), str(size), "1", num(secs[size, call]), str(d["nobs"]),
                              val("lrrfsim01W", "coef"), val("lrrfsim01W", "se"), val("lfcit01W", "coef"),
                              val("lfcit01W", "se"), num(d["r2"])]))
    (out / "results.csv").write_text("\n".join(rows) + "\n")


def params_002(d, call):
    """(parameter, b, analytic SE, bootstrap SE, critical value) of what the call reports."""
    if CALLS_002[call][5] == "gt":
        crit = d["crit"]["gt"]
        for r in d["gt"].itertuples():
            a, b = sorted((int(r.time), int(r.cohort) - 1))
            yield f"g{int(r.cohort)}:t_{a}_{b}", r.att, r.se, r.se_boot, crit
        return
    crit, av = d["crit"]["event_with_avgs"], d["averages"]
    for name, key in (("Pre_avg", "pre_avg"), ("Post_avg", "post_avg")):
        yield name, av.loc[key, "att"], av.loc[key, "se"], av.loc[key, "se_boot"], crit
    for r in d["event"].itertuples():
        e = int(r.e)
        yield (f"Tm{-e}" if e < 0 else f"Tp{e}"), r.att, r.se, r.se_boot, crit


def write_002(res, secs, out, head):
    est = ["tool,call,scale,parameter,b,se_analytic,se_boot,crit_uniform"]
    tim = ["tool,version,threads,call,scale,rep,seconds"]
    for (size, call), d in sorted(res.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        for p in params_002(d, call):
            est.append(",".join([head[0], str(call), str(size), p[0], *map(num, p[1:])]))
        tim.append(",".join([*head, str(call), str(size), "1", num(secs[size, call])]))
    (out / "results").mkdir(exist_ok=True)
    (out / f"results/{head[0]}_estimates.csv").write_text("\n".join(est) + "\n")
    (out / f"results/{head[0]}_timings.csv").write_text("\n".join(tim) + "\n")


def write_003(res, secs, out, head):
    import pandas as pd
    master = pd.read_parquet(KITS / "003/data/master.parquet")
    tim = ["tool,version,threads,size_pct,rep,seconds,n_master,pairs_kept,matched_master,score_sum"]
    (out / "results").mkdir(exist_ok=True)
    for (size, _), d in sorted(res.items()):
        rows = ["idm,idu,score"] + [f"{int(i)},{int(j)},{num(s)}" for i, j, s in zip(d["i"], d["j"], d["score"])]
        (out / f"results/{head[0]}_matches_{size}.csv").write_text("\n".join(rows) + "\n")
        n_master = len(master) if size == 100 else int((master[f"sub{size:02d}"] == 1).sum())
        tim.append(",".join([*head, str(size), "1", num(secs[size, 1]), str(n_master), str(d["i"].size),
                             str(len(set(d["i"].tolist()))), num(d["score"].sum())]))
    (out / f"results/{head[0]}_timings.csv").write_text("\n".join(tim) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kit", choices=["001", "002", "003"])
    ap.add_argument("results")
    ap.add_argument("out")
    ap.add_argument("--variant", default="speed")
    ap.add_argument("--tool")
    a = ap.parse_args()
    src, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res, secs, version, threads = {}, {}, None, None
    for line in (src / "timings.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["kit"] == a.kit and r["variant"] == a.variant and r["rep"] == 1:
            secs[r["size"], r["call"]] = r["seconds"]
            version, threads = r["version"], r["threads"]
    for size, call in secs:
        p = src / f"{a.kit}-{size}-{call}-{a.variant}.pkl"
        res[size, call] = pickle.loads(p.read_bytes())
    head = [a.tool or f"turbo-socialsci-{a.variant}", version, str(threads)]
    {"001": write_001, "002": write_002, "003": write_003}[a.kit](res, secs, out, head)
    kit_dir = KITS / a.kit
    if not (kit_dir / "reference/grade.py").exists():
        sys.exit(f"{kit_dir}/reference/grade.py missing")
    lines = []
    for f, extra in targets(a.kit, sorted(out.rglob("*.csv"))):
        r = subprocess.run([PY, "-I", "reference/grade.py", str(f), *extra], cwd=kit_dir, capture_output=True,
                           text=True, timeout=7200)
        lines.append(f"### {f.relative_to(out)} {' '.join(extra)} (exit {r.returncode})\n{r.stdout}{r.stderr}")
    text = "\n".join(lines)
    (out / "grade.txt").write_text(text)
    print(f"{a.kit} {a.variant} ({src.name}): {len(res)} results, {len(lines)} files graded: {verdicts(text)}"
          f" -> {out / 'grade.txt'}")


if __name__ == "__main__":
    main()
