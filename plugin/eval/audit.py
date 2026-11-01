"""Audit an agent run's transcript: tool calls, any access outside its workspace, sandbox denials, and
whether (and how) the plugin was used.

usage: audit.py META_DIR      (the run's .meta directory, holding run.json and transcript.jsonl)
Writes META_DIR/audit.json and prints a summary. Path checks are heuristic for Bash commands (absolute
paths mentioned in the command text); file tools are checked exactly.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

HOME = str(Path.home())
TOOLCHAIN = [f"{HOME}/.local/bin", f"{HOME}/.local/share/uv", f"{HOME}/.cache/uv", "/usr/", "/bin/", "/opt/",
             "/Library/", "/System/", "/private/var/folders", "/var/folders", "/tmp", "/private/tmp", "/dev/"]
PATH_RE = re.compile(r"(?:(?<=\s)|(?<=^)|(?<=[\"'=(]))(~?/[A-Za-z0-9_.@%+:,\-/]+)")


def outside(path, ws, plugin):
    p = path.replace("~", HOME, 1) if path.startswith("~") else path
    if p.startswith(ws) or (plugin and p.startswith(plugin)):
        return False
    if p.startswith("/private/tmp/claude-502"):
        return True
    return not any(p.startswith(t) for t in TOOLCHAIN)


def main():
    meta = Path(sys.argv[1])
    info = json.loads((meta / "run.json").read_text())
    ws = info["workspace"]
    plugin = "/Users/Shared/turbo-socialsci-plugin" if info["arm"] == "with" else None
    tools, flags, denials, skills, plugin_uses, bash = Counter(), [], [], [], 0, 0
    for line in open(meta / "transcript.jsonl"):
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = ev.get("message")
        msg = msg if isinstance(msg, dict) else {}
        for c in msg.get("content") or [] if isinstance(msg.get("content"), list) else []:
            if c.get("type") == "tool_use":
                name, inp = c.get("name"), c.get("input") or {}
                tools[name] += 1
                text = json.dumps(inp)
                if "turbo_socialsci" in text or "turbo-socialsci-plugin" in text:   # not the runs' folder name
                    plugin_uses += 1
                if name == "Skill":
                    skills.append(inp.get("skill") or inp.get("name"))
                for key in ("file_path", "path", "notebook_path"):
                    if key in inp and isinstance(inp[key], str) and inp[key].startswith(("/", "~")) \
                            and outside(inp[key], ws, plugin):
                        flags.append(dict(tool=name, path=inp[key]))
                if name == "Bash":
                    bash += 1
                    for p in PATH_RE.findall(inp.get("command", "")):
                        if outside(p, ws, plugin) and len(p) > 1:
                            flags.append(dict(tool="Bash", path=p, command=inp.get("command", "")[:200]))
            if c.get("type") == "tool_result":
                body = c.get("content")
                body = json.dumps(body) if not isinstance(body, str) else body
                if re.search(r"Operation not permitted|sandbox|denied by", body, re.I):
                    denials.append(body[:300])
    out = dict(tools=dict(tools), bash_calls=bash, skills_invoked=skills, plugin_mentions=plugin_uses,
               outside_paths=flags, denial_snippets=denials[:50], n_denials=len(denials))
    (meta / "audit.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(dict(tools=dict(tools), skills_invoked=skills, plugin_mentions=plugin_uses,
                          outside_paths=len(flags), n_denials=len(denials)), indent=1))
    for f in flags[:20]:
        print("  outside:", f)


if __name__ == "__main__":
    main()
