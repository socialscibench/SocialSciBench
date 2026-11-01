"""Stage BLS QCEW single files (2010-2024) as Parquet: county level, private ownership only.

Keeps rows with own_code 5 (private) and agglvl_code 71 (county total) or 75 (county x
3-digit NAICS), for real counties (5-digit FIPS, excluding statewide xx000 and
unknown-county xx999 codes). The ZIPs are not modified; SHA-256 checksums are written.

usage: python stage_qcew.py RAW_DIR OUT_DIR
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import duckdb


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


KEEP = ("own_code = '5' AND agglvl_code IN ('71', '75') AND length(area_fips) = 5 "
        "AND regexp_matches(area_fips, '^[0-9]{5}$') AND NOT area_fips LIKE '%000' AND NOT area_fips LIKE '%999'")
COLS = {
    "annual": "area_fips, industry_code, agglvl_code, year::INT AS year, disclosure_code, "
              "annual_avg_estabs::DOUBLE AS estabs, annual_avg_emplvl::DOUBLE AS emplvl, "
              "annual_avg_wkly_wage::DOUBLE AS wkly_wage",
    "qtrly": "area_fips, industry_code, agglvl_code, year::INT AS year, qtr::INT AS qtr, disclosure_code, "
             "qtrly_estabs::DOUBLE AS estabs, (month1_emplvl::DOUBLE + month2_emplvl::DOUBLE + month3_emplvl::DOUBLE) / 3 "
             "AS emplvl, avg_wkly_wage::DOUBLE AS wkly_wage",
}


def main(raw, out):
    raw, out = Path(raw), Path(out)
    tmp = out / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET memory_limit='8GB'; SET threads=8")
    sums = {}
    for kind in ("annual", "qtrly"):
        parts = []
        for z in sorted(raw.glob(f"*_{kind}_singlefile.zip")):
            sums[z.name] = dict(sha256=sha256(z), bytes=z.stat().st_size)
            subprocess.run(["unzip", "-o", "-q", str(z), "-d", str(tmp)], check=True)
            csv = next(tmp.glob("*.csv"))
            part = tmp / f"{z.stem}.parquet"
            con.execute(f"""COPY (SELECT {COLS[kind]} FROM read_csv('{csv}', all_varchar=true, header=true)
                                 WHERE {KEEP}) TO '{part}' (FORMAT parquet)""")
            csv.unlink()
            parts.append(part)
            n = con.execute(f"SELECT count(*) FROM '{part}'").fetchone()[0]
            print(f"{z.name}: {n:,} rows", flush=True)
        dst = out / f"qcew_county_private_{kind}.parquet"
        con.execute(f"COPY (SELECT * FROM read_parquet({[str(p) for p in parts]})) TO '{dst}' (FORMAT parquet, COMPRESSION zstd)")
        for p in parts:
            p.unlink()
    tmp.rmdir()
    (out / "checksums.json").write_text(json.dumps(sums, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:3])
