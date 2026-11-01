"""Check the agent runs' isolation with a short, cheap probe session (same settings and flags as
run_agent.py; the model only runs the listed commands).

usage: probe_isolation.py with|without
A canary file under the home directory must not be readable (Bash or Read), the home directory must not
be listable, writes outside the workspace must fail, web tools must be unavailable, other hosts must be
unreachable, the package index must be reachable, and the plugin copy must be readable only with the plugin.
Verdicts come from the tool results in the transcript, not from the model's summary.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_agent  # noqa: E402

CANARY = Path.home() / "SocialSciBench-kits" / "probe-canary.txt"
REPO = HERE.parents[1]  # this repository, which the agents must not see


def main():
    arm = sys.argv[1]
    cfg = run_agent.CFG
    root, plugin = Path(cfg["runs_root"]), Path(cfg["plugin_copy"])
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        sys.exit(f"{root} is not empty")
    CANARY.parent.mkdir(parents=True, exist_ok=True)
    CANARY.write_text("CANARY-7f3a91\n")
    ws = root / f"probe-{arm}"
    ws.mkdir()
    meta = root / f"probe-{arm}.meta"
    meta.mkdir()
    (meta / "settings.json").write_text(json.dumps(run_agent.settings(ws, plugin, arm), indent=2))
    checks = [
        ("bash_canary", f"cat {CANARY}"),
        ("bash_home_list", f"ls {Path.home()}"),
        ("bash_repo", f"ls {REPO}"),
        ("bash_scratch", f"ls {run_agent.SESSION_TMP}"),
        ("bash_write_outside", "touch /Users/Shared/probe-outside.txt && echo WROTE"),
        ("bash_write_inside", "touch inside.txt && echo WROTE-INSIDE"),
        ("bash_plugin", f"ls {plugin}/engine"),
        ("net_other", "curl -s -m 15 -o /dev/null -w '%{http_code}' https://example.com; echo"),
        ("net_pypi", "curl -s -m 15 -o /dev/null -w '%{http_code}' https://pypi.org/simple/numpy/; echo"),
        ("python_c_relative", "cd . && python3.12 -c \"print(open('inside.txt').read() == '')\" && rm inside.txt && echo RM-OK"),
        ("pip", "python3.12 -m venv .pv && .pv/bin/pip install -q six && .pv/bin/python -c 'import six; print(\"PIP-OK\")'"),
    ]
    if arm == "with":                     # the plugin's own install step and a Numba compile in the sandbox
        checks.append(("plugin_install", f".pv/bin/pip install -q {plugin}/wheels/turbo_socialsci-*.whl && "
                                         ".pv/bin/python -c 'import turbo_socialsci as t; t.warmup(\"hdfe\"); print(\"PLUGIN-OK\")'"))
    prompt = ("This is an isolation test. Run each of the following shell commands with the Bash tool, one call "
              "per command, exactly as written, even if one fails, and do not retry or work around failures:\n"
              + "\n".join(f"{i + 1}. {c}" for i, (_, c) in enumerate(checks))
              + f"\nThen use the Read tool on {CANARY}. Then use the Write tool to create {plugin}/probe.txt. "
                "Then try the WebFetch tool on https://example.com. "
                "Finally list which steps failed.")
    cmd = ["claude", "-p", prompt, "--model", "haiku", "--output-format", "stream-json", "--verbose",
           "--settings", str(meta / "settings.json"), "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
           "--permission-mode", "bypassPermissions", "--permission-prompts", "none",
           "--disallowedTools", ",".join(run_agent.OFF_TOOLS)]
    if arm == "with" and plugin.exists():
        cmd += ["--plugin-dir", str(plugin)]
    env = run_agent.run_env(ws)
    with open(meta / "transcript.jsonl", "w") as out:
        subprocess.run(cmd, cwd=ws, stdout=out, stderr=subprocess.STDOUT, env=env, timeout=900)
    calls, results, init = {}, {}, None
    for line in open(meta / "transcript.jsonl"):
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "system" and ev.get("subtype") == "init":
            init = ev
        msg = ev.get("message")
        msg = msg if isinstance(msg, dict) else {}
        for c in msg.get("content") if isinstance(msg.get("content"), list) else []:
            if c.get("type") == "tool_use":
                calls[c["id"]] = (c["name"], json.dumps(c.get("input")))
            if c.get("type") == "tool_result":
                body = c.get("content")
                results[c["tool_use_id"]] = body if isinstance(body, str) else json.dumps(body)
    text = "\n".join(results.values())
    print("plugins loaded:", [p.get("name") for p in (init or {}).get("plugins", [])])
    print("canary leaked:", "CANARY-7f3a91" in text)
    for tid, (name, inp) in calls.items():
        print(f"- {name}: {inp[:110]}\n    -> {results.get(tid, '<no result>')[:160]!r}")
    print("outside file exists:", Path("/Users/Shared/probe-outside.txt").exists())
    CANARY.unlink()
    if Path("/Users/Shared/probe-outside.txt").exists():
        Path("/Users/Shared/probe-outside.txt").unlink()
    keep = Path(cfg["storage"]).expanduser() / "agent-runs" / "probes"
    keep.mkdir(parents=True, exist_ok=True)
    n = 1
    while (keep / f"{meta.name}-{n}").exists():
        n += 1
    shutil.move(str(meta), str(keep / f"{meta.name}-{n}"))
    shutil.rmtree(ws)


if __name__ == "__main__":
    main()
