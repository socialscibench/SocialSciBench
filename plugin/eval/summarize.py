"""Collect the agent runs' records into one table (Markdown) and a JSON file.

usage: summarize.py [OUT_PREFIX]      (reads ~/SocialSciBench-kits/agent-runs/*.meta and the runs root)
Per run: the agent's wall time, turns, tokens and estimated cost; plugin use and the isolation audit; the
grader's verdicts for the agent's own results and for the rerun's; the rerun's wall time, the code's own
timings and peak memory.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "kits.json").read_text())


def metas():
    for base in (Path(CFG["storage"]).expanduser() / "agent-runs", Path(CFG["runs_root"])):
        if base.exists():
            yield from sorted(p for p in base.glob("*.meta") if (p / "run.json").exists())


def load(p):
    return json.loads(p.read_text()) if p.exists() else {}


def verdicts(text):
    """Cells passed / required, from the kits' graders: 001 and 002 print "N of M ... cells pass";
    003 prints "-> PASS" or "-> FAIL" for each size it grades (five sizes)."""
    m = re.findall(r"(\d+) of (\d+) [a-z ]*cells pass", text)
    if m:
        return f"{m[-1][0]}/{m[-1][1]}"
    sizes = re.findall(r"-> (PASS|FAIL)", text)
    return f"{sizes.count('PASS')}/5" if sizes else "none"


def session_totals(transcript):
    """A session that waits on background work emits one result event per extra turn: turns and tokens
    are summed over them; total_cost_usd is cumulative, so the last one is the session's."""
    turns = out = inp = 0
    cost = None
    for line in open(transcript):
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "result":
            u = ev.get("usage") or {}
            turns += ev.get("num_turns") or 0
            out += u.get("output_tokens") or 0
            inp += sum(u.get(k) or 0 for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
            cost = ev.get("total_cost_usd", cost)
    return dict(num_turns=turns, total_cost_usd=cost, usage=dict(output_tokens=out, input_tokens=inp))


def main():
    prefix = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    rows = []
    for m in metas():
        run, audit, rerun = load(m / "run.json"), load(m / "audit.json"), load(m / "rerun.json")
        res = session_totals(m / "transcript.jsonl")
        usage = res["usage"]
        row = dict(run=m.name[:-5], kit=run.get("kit"), arm=run.get("arm"), wall_min=round(run.get("wall_s", 0) / 60, 1),
                   stopped=run.get("stopped_at_limit"), turns=res.get("num_turns"),
                   cost_usd=res.get("total_cost_usd"),
                   tokens_in=usage.get("input_tokens"),
                   tokens_out=usage.get("output_tokens"), plugin_loaded="turbo-socialsci" in (run.get("plugins_loaded") or []),
                   plugin_mentions=audit.get("plugin_mentions"), skills=audit.get("skills_invoked"),
                   outside_paths=len(audit.get("outside_paths") or []), denials=audit.get("n_denials"),
                   rerun_wall_s=rerun.get("wall_s"), rerun_peak_gib=round(rerun["peak_rss_bytes"] / 2**30, 2)
                   if rerun.get("peak_rss_bytes") else None, rerun_exit=rerun.get("exit_code"))
        for w in ("agent", "rerun"):
            g = m / f"grade_{w}.txt"
            if g.exists():
                row[f"grade_{w}"] = verdicts(g.read_text())
        rows.append(row)
    cols = ["run", "wall_min", "stopped", "turns", "cost_usd", "tokens_out", "plugin_mentions", "outside_paths",
            "denials", "grade_agent", "grade_rerun", "rerun_wall_s", "rerun_peak_gib"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    print("\n".join(lines))
    if prefix:
        prefix.with_suffix(".json").write_text(json.dumps(rows, indent=2))
        prefix.with_suffix(".md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
