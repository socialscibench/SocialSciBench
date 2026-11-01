"""Copy the frozen plugin (plugin/turbo-socialsci at a git tag) to the location the agent runs load it from,
outside the home directory, and check it against the freeze manifest.

usage: export_plugin.py TAG MANIFEST.json      (repository root)
"""
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CFG = json.loads((Path(__file__).with_name("kits.json")).read_text())


def main():
    tag, manifest = sys.argv[1], json.loads(Path(sys.argv[2]).expanduser().read_text())
    dest = Path(CFG["plugin_copy"])
    if dest.exists():
        shutil.rmtree(dest)
    data = subprocess.run(["git", "archive", "--format=tar", tag, "plugin/turbo-socialsci"], cwd=ROOT,
                          check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            rel = Path(m.name).relative_to("plugin/turbo-socialsci")
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(tf.extractfile(m).read())
    files = {str(p.relative_to(dest)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(dest.rglob("*")) if p.is_file()}
    want = {k[len("plugin/turbo-socialsci/"):]: v for k, v in manifest["files"].items()
            if k.startswith("plugin/turbo-socialsci/")}
    if files != want:
        sys.exit(f"plugin copy differs from the freeze manifest: {sorted(set(files.items()) ^ set(want.items()))[:5]}")
    print(f"{dest}: {len(files)} files match the freeze manifest ({manifest['tag']}, {manifest['commit'][:12]})")


if __name__ == "__main__":
    main()
