"""Tables of the study report and of the baseline adapters' research notes (study tooling).

usage: tables.py RESULTS_DIR[,RESULTS_DIR...] RUN_DIR...                 # study report tables
       tables.py --per-baseline RESULTS_DIR[,RESULTS_DIR...] RUN_DIR...  # one table per baseline adapter

Run from the repository root. RESULTS_DIR holds compare.py's TASK-SOLUTION-tTHREADS.jsonl files
and optionally exclusions.json / exclusions.jsonl (rows left out, with the reason: file, case,
created; thermal_queue.py writes the latter, and an attempt marked final is kept); the run
directories are the candidates' evaluator runs (both thread counts). When a (task, adapter,
threads, case) was measured more than once, the last row that is not excluded is used.

Study tables: a summary over tasks, then per task (a) the candidate against the reference at one
and six threads, real and synthetic workloads separately; (b) the candidate against every
baseline at each thread count (baseline median and baseline / candidate for passing baselines, the
failure otherwise); (c) geometric means and ranges against the reference and against the fastest
passing baseline of each case.

Per-baseline tables: each case and thread count with the baseline's status, median, first call and
peak RSS, the candidate's and the reference's evaluator medians at the same thread count, and the
baseline / candidate ratio (in parentheses where the baseline fails the checks); failures are
listed below each table.
"""
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "solutions/baselines")
from summarize import from_run  # noqa: E402

args = sys.argv[1:]
per_baseline = args[:1] == ["--per-baseline"]
if per_baseline:
    args = args[1:]
resdirs, runs = [Path(d) for d in args[0].split(",")], [Path(a) for a in args[1:]]

cand = defaultdict(dict)  # task -> (threads, case) -> evaluator row
for r in runs:
    for row in from_run(r):
        cand[row["task"]][(row["threads"], row["case"])] = row

exclusions, base = [], defaultdict(lambda: defaultdict(dict))  # task -> solution -> (threads, case) -> record
for resdir in resdirs:
    excl = json.loads((resdir / "exclusions.json").read_text()) if (resdir / "exclusions.json").exists() else []
    if (resdir / "exclusions.jsonl").exists():  # written by thermal_queue.py (compare rows only)
        excl += [e for e in map(json.loads, (resdir / "exclusions.jsonl").read_text().splitlines())
                 if "file" in e and not e.get("final")]
    exclusions += excl
    excluded = {(e["file"], e["case"], e["created"]) for e in excl}
    for f in sorted(resdir.glob("*.jsonl")):
        m = re.match(r"(\d{3}b?)-(.+)-t(\d+)\.jsonl$", f.name)
        if not m:
            continue
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if (f.name, r["case"], r.get("created")) in excluded:
                continue
            r["_file"] = f"{resdir.name}/{f.name}"
            case = "104k20" if m.group(1) == "001" and r["case"] == "104" else r["case"]  # 104 = 20 regressors
            base[m.group(1)][m.group(2)][(r["threads"], case)] = r


def secs(v, big=",.0f"):
    if v is None:
        return "-"
    return f"{v:.3f} s" if v < 10 else (f"{v:.2f} s" if v < 100 else f"{v:{big}} s")


def order(case):
    if case.startswith("s"):
        return (1, int(case[1:]), case)
    return (0, int(case[:3]), case)


def suite(case):
    return "synthetic" if case.startswith("s") else "real"


def gmean(xs):
    xs = [x for x in xs if x]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def rng(xs):
    xs = [x for x in xs if x]
    return f"{min(xs):.1f}-{max(xs):.1f}x (geometric mean {gmean(xs):.1f}x, {len(xs)} cases)" if xs else "-"


def short(xs):
    xs = [x for x in xs if x]
    return f"{gmean(xs):.1f}x ({min(xs):.1f}-{max(xs):.1f}x, {len(xs)})" if xs else "-"


def bstatus(r):
    st = r["status"]
    if st == "COMPLETED":
        return "passed" if r.get("checks_passed") else "fails checks"
    return {"TIMEOUT": "timeout", "RUNTIME_ERROR": "error"}.get(st, st.lower())


def failure(r):
    if r.get("error"):
        # R puts the message on the line after "Error in call(...) :"; keep both, shorten the call
        text = " ".join(line.strip() for line in str(r["error"]).strip().splitlines())
        return re.sub(r"Error in ([\w.]+)\(.*?\s:\s", r"Error in \1(...): ", text)[:240]
    rel, other = [], []
    for ph in r.get("failed_checks") or []:
        if ph.get("phase") != "measured":
            continue
        for name, chk in (ph.get("failed") or {}).items():
            if "max_rel_error" in chk:
                rel.append(f"{name} {chk['max_rel_error']:.1e} ({chk.get('rtol'):g})")
            elif chk.get("reason"):
                other.append(f"{name}: {chk['reason']} ({chk.get('actual_shape')} vs {chk.get('expected_shape')})")
            else:
                other.append(f"{name}: {json.dumps(chk)[:160]}")
    return "; ".join((["max. rel. error (tolerance): " + ", ".join(rel)] if rel else []) + other)


def fastest_passing(task, th, case):
    """(solution, median) of the fastest baseline whose outputs passed the checks, or None."""
    best = None
    for n, recs in base.get(task, {}).items():
        r = recs.get((th, case))
        if r and bstatus(r) == "passed" and (best is None or r["median_s"] < best[1]):
            best = (n, r["median_s"])
    return best


def study():
    print("| Task | Threads | Candidate / reference, real | Candidate / reference, synthetic | "
          "Fastest passing baseline / candidate, real | Same, synthetic |")
    print("|---|---:|---:|---:|---:|---:|")
    for task in sorted(cand):
        C = cand[task]
        for th in (1, 6):
            cells = []
            for s in ("real", "synthetic"):
                cells.append(short([C[k]["speedup"] for k in C if k[0] == th and suite(k[1]) == s
                                    and C[k]["status"] == "PASSED"]))
            for s in ("real", "synthetic"):
                xs = []
                for (t, c), row in C.items():
                    fp = fastest_passing(task, th, c) if t == th and suite(c) == s else None
                    if fp and row.get("cand_median_s"):
                        xs.append(fp[1] / row["cand_median_s"])
                cells.append(short(xs))
            print(f"| {task} | {th} | " + " | ".join(cells) + " |")
    print("\nCells: geometric mean (range, number of cases) of per-case ratios of medians.")

    for task in sorted(cand):
        C = cand[task]
        cases = sorted({c for _, c in C}, key=order)
        print(f"\n## Task {task}\n")
        for s in ("real", "synthetic"):
            cs = [c for c in cases if suite(c) == s]
            print(f"\n{s.capitalize()} workloads, candidate against the reference (evaluator; medians):\n")
            print("| Case | Calls | Reference, 1 thread | Candidate, 1 thread | Speedup | Reference, 6 threads | "
                  "Candidate, 6 threads | Speedup | Peak RSS reference / candidate (1 thread) | Max. rel. error |")
            print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
            for c in cs:
                a, b = C.get((1, c), {}), C.get((6, c), {})
                calls = f"1+{a.get('warmups')}+{a.get('repeats')}"
                err = max(a.get("cand_max_rel_error") or 0, b.get("cand_max_rel_error") or 0)
                print(f"| {c} | {calls} | {secs(a.get('ref_median_s'))} | {secs(a.get('cand_median_s'))} | "
                      f"{a.get('speedup', 0):.1f}x | {secs(b.get('ref_median_s'))} | {secs(b.get('cand_median_s'))} | "
                      f"{b.get('speedup', 0):.1f}x | {a.get('ref_peak_rss_gib', 0):.2f} / "
                      f"{a.get('cand_peak_rss_gib', 0):.2f} GiB | {err:.1e} |")
            for th in (1, 6):
                sp = [C[(th, c)]["speedup"] for c in cs if (th, c) in C]
                st = {C[(th, c)]["status"] for c in cs if (th, c) in C}
                print(f"\n- {th} thread{'s' if th > 1 else ''}: against the reference {rng(sp)}; statuses {sorted(st)}")
            sp = [C[(1, c)]["ref_median_s"] / C[(6, c)]["cand_median_s"] for c in cs if (1, c) in C and (6, c) in C]
            print(f"- 6-thread candidate against the 1-thread reference: {rng(sp)}")

        sols = sorted(base.get(task, {}))
        for th in (1, 6):
            have = [n for n in sols if any(k[0] == th for k in base[task][n])]
            if not have:
                continue
            print(f"\nBaselines at {th} thread{'s' if th > 1 else ''} (first call + one measured call; "
                  "baseline median, and baseline / candidate median):\n")
            print("| Case | Candidate | " + " | ".join(have) + " |")
            print("|---|---:|" + "---:|" * len(have))
            best = {"real": [], "synthetic": []}
            allc = sorted({c for n in have for (t, c) in base[task][n] if t == th} | {c for (t, c) in C if t == th},
                          key=order)
            for c in allc:
                cm = (C.get((th, c)) or {}).get("cand_median_s")
                cells = []
                for n in have:
                    r = base[task][n].get((th, c))
                    if not r:
                        cells.append("")
                        continue
                    st = bstatus(r)
                    if st == "passed":
                        cells.append(f"{secs(r['median_s'])} ({r['median_s'] / cm:.1f}x)" if cm else secs(r["median_s"]))
                    elif st == "fails checks":
                        cells.append(f"{secs(r['median_s'])}, fails checks")
                    elif st == "timeout":
                        cells.append(f"timeout ({r['wall_s']:.0f} s, 2 calls)")
                    else:
                        cells.append(st)
                fp = fastest_passing(task, th, c)
                if fp and cm:
                    best[suite(c)].append((c, fp[0], fp[1] / cm))
                print(f"| {c} | {secs(cm)} | " + " | ".join(cells) + " |")
            for s in ("real", "synthetic"):
                if best[s]:
                    xs = [x for _, _, x in best[s]]
                    who = ", ".join(f"{c}: {n} {x:.1f}x" for c, n, x in best[s])
                    print(f"\n- {s}, against the fastest passing baseline of each case: {rng(xs)} [{who}]")


def baseline_tables():
    for task in sorted(base):
        for sol, recs in sorted(base[task].items()):
            print(f"\n#### {task} {sol}\n")
            print("Result files: " + ", ".join(sorted({f"`{r['_file']}`" for r in recs.values()})) + "\n")
            print("| Case | Threads | Status | Baseline median | First call | Peak RSS | Candidate median | "
                  "Baseline / candidate | Reference median |")
            print("|---|---:|---|---:|---:|---:|---:|---:|---:|")
            notes = []
            for (th, case), r in sorted(recs.items(), key=lambda kv: (kv[0][0], order(kv[0][1]))):
                c = cand.get(task, {}).get((th, case)) or {}
                st = bstatus(r)
                if st == "timeout":
                    st = f"timeout ({r['wall_s']:.0f} s for both calls)"
                ratio = (f"{r['median_s'] / c['cand_median_s']:.1f}x"
                         if r.get("median_s") and c.get("cand_median_s") else "-")
                if ratio != "-" and st != "passed":
                    ratio = f"({ratio})"  # not a valid comparison: the baseline's outputs fail the checks
                rss = f"{r['peak_rss_bytes'] / 2**30:.2f} GiB" if r.get("peak_rss_bytes") else "-"
                print(f"| {case} | {th} | {st} | {secs(r.get('median_s'), ',.1f')} | "
                      f"{secs(r.get('first_call_s'), ',.1f')} | {rss} | {secs(c.get('cand_median_s'), ',.1f')} | "
                      f"{ratio} | {secs(c.get('ref_median_s'), ',.1f')} |")
                fl = failure(r)
                if fl:
                    notes.append(f"- {case}, {th} thread{'s' if th > 1 else ''}: {fl}")
            if notes:
                print("\n" + "\n".join(notes))


if per_baseline:
    baseline_tables()
else:
    study()
for e in exclusions:
    print(f"\n- Excluded: `{e['file']}` case {e['case']} (row of {e['created']}): {e['reason']}")
for resdir in resdirs:
    if (resdir / "exclusions.jsonl").exists():
        for e in map(json.loads, (resdir / "exclusions.jsonl").read_text().splitlines()):
            if e.get("final") and "file" in e:
                print(f"\n- Kept although affected (third attempt): `{resdir.name}/{e['file']}` case {e['case']}: {e['reason']}")
