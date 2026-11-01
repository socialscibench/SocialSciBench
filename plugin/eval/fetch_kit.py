"""Download parts of a speed-comparison kit from its shared Google Drive folders into the kits' storage
(outside the repository), and check the data files against the kit's manifest.

usage: fetch_kit.py KIT PART [PART ...]      PART: task, data or reference
Only `task` and `data` may be given to whoever implements a tool; fetch `reference` (answers and the
grader) only after the implementation being measured is frozen. Each part goes into a new, empty
directory; downloaded files are data and are never imported or executed from here.
"""
import hashlib
import html
import json
import re
import sys
import urllib.request
from pathlib import Path

CFG = json.loads((Path(__file__).with_name("kits.json")).read_text())
# the kits' Drive folder ids are kept outside the repository: their reference/ folders hold the answers
FOLDERS = json.loads(Path(CFG["folders_file"]).expanduser().read_text())
UA = {"User-Agent": "Mozilla/5.0"}


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600) as r:
        return r.read()


def listing(folder_id):
    """[(id, name, is_folder)] of a shared folder's page."""
    page = get(f"https://drive.google.com/drive/folders/{folder_id}?usp=sharing").decode("utf-8", "replace")
    out, seen = [], set()
    for m in re.finditer(r'data-id="([A-Za-z0-9_-]{20,60})"', page):
        fid = m.group(1)
        if fid in seen:
            continue
        seen.add(fid)
        label = ""
        for occ in re.finditer(re.escape(fid), page):      # the label follows one of the id's appearances
            lab = re.search(r'aria-label="([^"]+ Shared(?: folder)?)"', page[occ.end():occ.end() + 4000])
            if lab:
                label = html.unescape(lab.group(1))
                break
        if not label:
            raise RuntimeError(f"no name found for Drive item {fid}")
        folder = label.endswith("Shared folder")
        name = label[:-len(" Shared folder")] if folder else re.sub(r"\s+\S+(\s+\S+)?\s+Shared$", "", label)
        out.append((fid, name, folder))
    return out


def download(fid, dest):
    url = f"https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t"
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=3600) as r, open(tmp, "wb") as fh:
        while chunk := r.read(1 << 22):
            fh.write(chunk)
    head = tmp.read_bytes()[:200].lower()
    if b"<!doctype html" in head or b"<html" in head:
        tmp.unlink()
        raise RuntimeError(f"Drive returned a page instead of {dest.name}")
    tmp.rename(dest)


def fetch(folder_id, dest, names=None):
    dest.mkdir(parents=True, exist_ok=False)
    for fid, name, folder in listing(folder_id):
        if folder:
            fetch(fid, dest / name)
        else:
            download(fid, dest / name)
            print(f"  {dest / name}  {(dest / name).stat().st_size:,} bytes", flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def check_data(data_dir):
    man = data_dir / "manifest.json"
    if not man.exists():
        print("  no manifest.json")
        return
    text = man.read_text()
    found = 0
    for f in sorted(data_dir.iterdir()):
        if f.name == "manifest.json" or f.is_dir():
            continue
        digest = sha256(f)
        ok = digest in text
        found += ok
        print(f"  {f.name}: sha256 {digest[:16]}... {'matches the manifest' if ok else 'NOT IN THE MANIFEST'}")
    print(f"  {found} files verified")


def main():
    kit, parts = sys.argv[1], sys.argv[2:]
    root = Path(CFG["storage"]).expanduser() / kit
    for part in parts:
        if part not in FOLDERS[kit]:
            sys.exit(f"unknown part {part}")
        print(f"{kit}/{part}:")
        fetch(FOLDERS[kit][part], root / part)
        if part == "data":
            check_data(root / part)


if __name__ == "__main__":
    main()
