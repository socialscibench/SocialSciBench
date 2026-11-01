"""Build the engine's wheel into plugin/turbo-socialsci/wheels/ and check that it holds exactly the source
tree's package files. The skills install this wheel (pip would otherwise build in the read-only plugin folder).

usage: build_wheel.py          (repository root; needs uv)
Timestamps are fixed (SOURCE_DATE_EPOCH), so rebuilding unchanged sources gives the same files.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "plugin/turbo-socialsci/engine"
WHEELS = ROOT / "plugin/turbo-socialsci/wheels"
EPOCH = "1793404800"            # 2026-10-26 00:00 UTC


def package_files():
    src = ENGINE / "src"
    return {str(p.relative_to(src)): p.read_bytes() for p in sorted((src / "turbo_socialsci").rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts}


def check(wheel):
    """Raise unless the wheel's package files equal the source tree's."""
    with zipfile.ZipFile(wheel) as z:
        inside = {n: z.read(n) for n in z.namelist() if not n.split("/")[0].endswith(".dist-info")}
    want = package_files()
    if inside != want:
        diff = sorted(set(inside) ^ set(want)) or [n for n in want if inside.get(n) != want[n]]
        raise SystemExit(f"{wheel.name} differs from the sources: {diff[:8]}")
    return len(want)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, SOURCE_DATE_EPOCH=EPOCH)
        subprocess.run(["uv", "build", "--wheel", "--out-dir", tmp, str(ENGINE)], check=True, env=env,
                       stdout=subprocess.DEVNULL)
        built = list(Path(tmp).glob("*.whl"))
        if len(built) != 1:
            sys.exit(f"expected one wheel, got {built}")
        if WHEELS.exists():
            shutil.rmtree(WHEELS)
        WHEELS.mkdir(parents=True)
        wheel = WHEELS / built[0].name
        shutil.copy2(built[0], wheel)
    for junk in list((ENGINE / "src").glob("*.egg-info")) + [ENGINE / "build"]:
        shutil.rmtree(junk, ignore_errors=True)
    print(f"{wheel.relative_to(ROOT)}: {check(wheel)} package files, identical to the sources")


if __name__ == "__main__":
    main()
