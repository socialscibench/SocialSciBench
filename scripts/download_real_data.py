"""Download the pinned, publicly shared real-data packages; keep binaries out of Git."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.parse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.storage import default_private_root, external_directory, private_package_file


class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.action = None
        self.fields = {}
        self.inside = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "form" and attrs.get("id") == "download-form":
            self.action = attrs.get("action")
            self.inside = True
        elif self.inside and tag == "input" and attrs.get("name"):
            self.fields[attrs["name"]] = attrs.get("value", "")

    def handle_endtag(self, tag):
        if tag == "form":
            self.inside = False


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def valid(path, entry):
    return (path.is_file() and path.stat().st_size == entry["size"]
            and ("sha256" not in entry or digest(path) == entry["sha256"]))


def download(entry, root, verify_only=False):
    target = root / entry["task"] / entry["path"]
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Package path escapes destination")
    label = f'{entry["task"]}/{entry["path"]}'
    if valid(target, entry):
        print(f"Verified {label}", flush=True)
        return True
    if verify_only:
        print(f"MISSING OR CHANGED {label}", flush=True)
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    for attempt in range(3):
        try:
            url = "https://drive.google.com/uc?" + urllib.parse.urlencode(
                {"export": "download", "id": entry["drive_id"]})
            def transfer(url):
                subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                                "--connect-timeout", "30", "--max-time", "900",
                                "--output", str(partial), url], check=True, capture_output=True)
            transfer(url)
            if partial.stat().st_size < 2_000_000 and partial.read_bytes()[:100].lstrip().lower().startswith(b"<!doctype html"):
                form = DownloadForm()
                form.feed(partial.read_text())
                parsed = urllib.parse.urlparse(form.action or "")
                if parsed.scheme != "https" or parsed.hostname != "drive.usercontent.google.com":
                    raise ValueError("No public download form; check Drive sharing or quota")
                transfer(form.action + "?" + urllib.parse.urlencode(form.fields))
            if not valid(partial, entry):
                raise ValueError("Downloaded bytes do not match pinned size/checksum")
            partial.replace(target)
            print(f"Downloaded {label} ({entry['size']:,} bytes)", flush=True)
            return True
        except Exception as exc:
            # URLs may contain temporary credentials; print only the exception type.
            print(f"Retry {attempt + 1}/3 {label}: {type(exc).__name__}", flush=True)
            if attempt < 2:
                time.sleep(2 ** attempt)
    print(f"FAILED {label}; check sharing/quota and retry", flush=True)
    return False


def relocate_private(entries, development_root, private_root):
    """Copy and verify the entire selection before removing any repository copy."""
    development_root = Path(development_root).resolve()
    private_root = external_directory(private_root, workspace=ROOT, other_forbidden=(development_root,))
    selected = [e for e in entries if private_package_file(e)]
    # Preflight every source/destination. Existing matching destinations allow an interrupted
    # relocation to resume; conflicting or unverified files are never overwritten/deleted.
    for entry in selected:
        source = development_root / entry["task"] / entry["path"]
        target = private_root / entry["task"] / entry["path"]
        if not source.resolve().is_relative_to(development_root):
            raise ValueError(f"Source escapes development root: {source}")
        if not target.resolve().is_relative_to(private_root):
            raise ValueError(f"Destination escapes private root: {target}")
        if source.exists() and not valid(source, entry):
            raise ValueError(f"Source checksum mismatch: {source}")
        if target.exists() and not valid(target, entry):
            raise ValueError(f"Destination checksum mismatch; refusing overwrite: {target}")
        if not source.exists() and not valid(target, entry):
            raise ValueError(f"Missing source and verified destination: {source}")
    private_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    for entry in selected:
        source = development_root / entry["task"] / entry["path"]
        target = private_root / entry["task"] / entry["path"]
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            created = False
            try:
                with source.open("rb") as src:
                    with target.open("xb") as dst:
                        created = True
                        shutil.copyfileobj(src, dst, length=1024 * 1024)
                        dst.flush()
                        os.fsync(dst.fileno())
            except BaseException:
                # The source still exists. Clean up only the new incomplete copy.
                if created:
                    target.unlink(missing_ok=True)
                raise
    for entry in selected:
        target = private_root / entry["task"] / entry["path"]
        if not valid(target, entry):
            raise ValueError(f"Destination verification failed; source copies retained: {target}")
    for entry in selected:
        source = development_root / entry["task"] / entry["path"]
        if source.exists():
            if not valid(source, entry):
                raise ValueError(f"Source changed during relocation; refusing removal: {source}")
            source.unlink()
    # Remove only empty directories; unknown files are retained for manual inspection.
    for task in {e["task"] for e in selected}:
        for name in ("heldout", "reference_outputs"):
            folder = development_root / task / "data" / name
            if folder.is_dir() and not any(folder.iterdir()):
                folder.rmdir()
    print(f"Relocated and verified {len(selected)} files ({sum(e['size'] for e in selected):,} bytes) to {private_root}")
    return len(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["001", "002", "003"], action="append")
    parser.add_argument("--out", type=Path, default=ROOT / "datasets")
    parser.add_argument("--private-root", type=Path, default=default_private_root(),
                        help="External root for held-out data/answers/logs (default: ~/SocialSciBench-private)")
    parser.add_argument("--complete-package", action="store_true",
                        help="Also download held-out data, answers and logs into --private-root")
    parser.add_argument("--relocate-private", action="store_true",
                        help="Move existing private files from --out to --private-root; no downloads")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 8:
        parser.error("--jobs must be between 1 and 8")
    index = json.loads((ROOT / "data_sources/packages.json").read_text())
    entries = [e for e in index["files"] if not args.task or e["task"] in args.task]
    try:
        if args.complete_package or args.relocate_private:
            args.private_root = external_directory(args.private_root, workspace=ROOT, other_forbidden=(args.out,))
        if args.complete_package and not args.verify_only:
            args.private_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if args.relocate_private:
            if args.verify_only or args.complete_package:
                parser.error("--relocate-private cannot be combined with --verify-only or --complete-package")
            relocate_private(entries, args.out, args.private_root)
            return 0
    except (ValueError, OSError) as error:
        parser.error(str(error))
    entries = [e for e in entries if args.complete_package or not private_package_file(e)]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(lambda e: download(e, args.private_root if private_package_file(e) else args.out,
                                                   args.verify_only), entries))
    print(f"{sum(results)}/{len(entries)} files verified")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
