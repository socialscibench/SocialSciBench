"""One agent run on a kit: the same Claude Code model, prompt and isolation with and without the plugin.

usage: run_agent.py KIT ARM RUN [--minutes 120]       ARM: with or without
The workspace /Users/Shared/turbo-socialsci-runs/KIT-ARM-RUN holds copies of only the kit's task/ and
data/ (APFS clones). The run's record (transcript, settings, timing) is written next to the workspace, in
KIT-ARM-RUN.meta/, and both are moved into the kits' storage by finish_run.py before the next run, so no
run can see another's files. Isolation (no container on this host):
- Claude Code's sandbox for Bash: writes only to the workspace and the package caches; reads denied for
  the home directory, this session's scratch area and the plugin copy (unless ARM is with), allowed for the
  workspace and the Python/uv toolchain; network only to package indexes;
- permission rules deny the Read and Edit tools the same paths; WebFetch, WebSearch and MCP servers off;
- the user's synced skills are turned off; ARM with adds --plugin-dir with the frozen plugin copy.
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
CFG = json.loads((HERE / "kits.json").read_text())
HOME = Path.home()
MODEL = "claude-opus-5-5"
SYNCED_SKILLS = ["built-in-browser", "chrome-browser", "computer-use", "deep-research", "docs", "docx",
                 "google-workspace", "import-memory", "morning", "pdf", "pptx", "skill-creator", "xlsx"]
DOMAINS = ["pypi.org", "files.pythonhosted.org", "github.com", "objects.githubusercontent.com",
           "release-assets.githubusercontent.com", "raw.githubusercontent.com", "cloud.r-project.org",
           "cran.r-project.org", "cran.rstudio.com", "conda.anaconda.org", "repo.anaconda.com"]


# this session's temporary files (scratch area with the kits' notes); Claude Code's own per-user temp
# directory around it stays usable, since the Bash tool keeps its working-directory file there
SESSION_TMP = f"/private/tmp/claude-{os.getuid()}/" + str(HERE.parents[1]).replace("/", "-")
# tools the runs do not get: web access, and contact with other sessions or schedules on this machine
OFF_TOOLS = ["WebFetch", "WebSearch", "SendMessage", "ListAgents", "RemoteTrigger", "PushNotification",
             "CronCreate", "CronDelete", "CronList", "ScheduleWakeup", "DesignSync"]


def settings(ws, plugin, arm):
    tool_paths = [HOME / ".local/bin", HOME / ".local/share/uv"]
    deny = [str(HOME), SESSION_TMP] + ([] if arm == "with" else [str(plugin)])
    allow_read = [str(ws)] + [str(p) for p in tool_paths] + ([str(plugin)] if arm == "with" else [])
    no_edit = deny + ([str(plugin)] if arm == "with" else [])          # the plugin copy is read-only
    deny_rules = [f"Read(/{p}/**)" for p in deny] + [f"Edit(/{p}/**)" for p in no_edit] + OFF_TOOLS
    return {
        "sandbox": {
            "enabled": True,
            "autoAllowBashIfSandboxed": True,
            "allowUnsandboxedCommands": False,
            "filesystem": {"denyRead": deny, "allowRead": allow_read, "allowWrite": [str(ws)]},
            # strictAllowlist: other hosts are refused in every permission mode (bypassPermissions would
            # otherwise allow them without a prompt)
            "network": {"allowedDomains": DOMAINS, "strictAllowlist": True},
        },
        # bypassPermissions: no approvals, which nobody could give in these runs (with Read() deny rules,
        # acceptEdits needed approval for any command whose file reads were not statically resolvable,
        # e.g. `python -c` or a relative path after `cd`). The OS sandbox still confines every Bash command
        # (allowUnsandboxedCommands false: no unsandboxed retry), and the deny rules still bind the file tools.
        "permissions": {"deny": deny_rules, "defaultMode": "bypassPermissions"},
        "skillOverrides": {s: "off" for s in SYNCED_SKILLS},
    }


def run_env(ws):
    """The runs' environment: no inherited session variables; package caches inside the workspace."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDE")}
    env.update(UV_CACHE_DIR=str(ws / ".cache/uv"), PIP_CACHE_DIR=str(ws / ".cache/pip"),
               XDG_CACHE_HOME=str(ws / ".cache"), MPLCONFIGDIR=str(ws / ".cache/matplotlib"),
               R_LIBS_USER=str(ws / ".Rlib"),
               # pip 26 verifies certificates through macOS's trust service, which the sandbox blocks
               # (OSStatus -26276); its bundled CA file works (uv cannot run in the sandbox at all: it
               # panics reading the system proxy configuration)
               PIP_USE_DEPRECATED="legacy-certs")
    return env


def main():
    kit, arm, run = sys.argv[1], sys.argv[2], sys.argv[3]
    minutes = float(sys.argv[sys.argv.index("--minutes") + 1]) if "--minutes" in sys.argv else 120.0
    assert arm in ("with", "without")
    store = Path(CFG["storage"]).expanduser() / kit
    root = Path(CFG["runs_root"])
    plugin = Path(CFG["plugin_copy"])
    root.mkdir(parents=True, exist_ok=True)
    others = [p for p in root.iterdir()]
    if others:
        sys.exit(f"{root} is not empty ({[p.name for p in others]}): finish the previous run first")
    if arm == "with" and not (plugin / ".claude-plugin/plugin.json").exists():
        sys.exit(f"no plugin copy at {plugin} (export_plugin.py)")
    name = f"{kit}-{arm}-{run}"
    ws, meta = root / name, root / f"{name}.meta"
    (ws / "task").mkdir(parents=True)
    meta.mkdir()
    for f in (store / "task").iterdir():
        shutil.copy2(f, ws / "task" / f.name)
    note = CFG["kits"][kit]["note"]
    if note:
        with open(ws / "task" / "TASK.md", "a") as fh:
            fh.write(f"\n\n{note}\n")
    (ws / "data").mkdir()
    for f in (store / "data").iterdir():
        subprocess.run(["cp", "-c", str(f), str(ws / "data" / f.name)], check=True)    # APFS clone
    st = settings(ws, plugin, arm)
    (meta / "settings.json").write_text(json.dumps(st, indent=2))
    cmd = ["claude", "-p", CFG["prompt"], "--model", MODEL, "--output-format", "stream-json", "--verbose",
           "--settings", str(meta / "settings.json"), "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
           "--permission-mode", "bypassPermissions", "--permission-prompts", "none",
           "--disallowedTools", ",".join(OFF_TOOLS)]
    if arm == "with":
        cmd += ["--plugin-dir", str(plugin)]
    env = run_env(ws)
    info = dict(kit=kit, arm=arm, run=run, model=MODEL, workspace=str(ws), cmd=cmd, minutes=minutes,
                start=time.time())
    (meta / "run.json").write_text(json.dumps(info, indent=2))
    with open(meta / "transcript.jsonl", "w") as out, open(meta / "stderr.txt", "w") as err:
        p = subprocess.Popen(cmd, cwd=ws, stdout=out, stderr=err, env=env, start_new_session=True)
        try:
            code = p.wait(timeout=minutes * 60)
            stopped = False
        except subprocess.TimeoutExpired:
            stopped = True
            os.killpg(p.pid, signal.SIGINT)
            try:
                code = p.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGTERM)
                code = p.wait()
    info.update(end=time.time(), exit_code=code, stopped_at_limit=stopped)
    info["wall_s"] = round(info["end"] - info["start"], 1)
    result, init = None, None
    for line in open(meta / "transcript.jsonl"):
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "system" and ev.get("subtype") == "init":
            init = ev
        if ev.get("type") == "result":
            result = ev
    if init:
        info["plugins_loaded"] = [p.get("name") for p in init.get("plugins", [])]
        info["plugin_errors"] = init.get("plugin_errors", [])
        info["skills"] = init.get("skills")
    if result:
        info["result"] = {k: result.get(k) for k in ("subtype", "is_error", "duration_ms", "duration_api_ms",
                                                      "num_turns", "total_cost_usd", "usage")}
        info["permission_denials"] = len(result.get("permission_denials") or [])
    (meta / "run.json").write_text(json.dumps(info, indent=2))
    print(json.dumps({k: info.get(k) for k in ("kit", "arm", "run", "wall_s", "exit_code", "stopped_at_limit",
                                               "plugins_loaded", "permission_denials")}))


if __name__ == "__main__":
    main()
