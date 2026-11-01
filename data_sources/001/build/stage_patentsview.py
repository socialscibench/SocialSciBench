"""Stage the PatentsView PVGPATDIS tables as typed Parquet files.

Each zipped TSV is extracted to a temporary file, read with DuckDB using explicit
column types, written to Parquet, and the temporary TSV is deleted. The original
ZIPs are not modified. SHA-256 checksums of the ZIPs are written to checksums.json.

usage: python stage_patentsview.py RAW_DIR OUT_DIR
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import duckdb

TABLES = {
    "g_patent": dict(patent_id="VARCHAR", patent_type="VARCHAR", patent_date="DATE", patent_title="VARCHAR",
                     wipo_kind="VARCHAR", num_claims="INTEGER", withdrawn="INTEGER", filename="VARCHAR"),
    "g_application": dict(application_id="VARCHAR", patent_id="VARCHAR", patent_application_type="VARCHAR",
                          filing_date="VARCHAR", series_code="VARCHAR", rule_47_flag="INTEGER"),
    "g_inventor_disambiguated": dict(patent_id="VARCHAR", inventor_sequence="INTEGER", inventor_id="VARCHAR",
                                     disambig_inventor_name_first="VARCHAR", disambig_inventor_name_last="VARCHAR",
                                     gender_code="VARCHAR", location_id="VARCHAR"),
    "g_assignee_disambiguated": dict(patent_id="VARCHAR", assignee_sequence="INTEGER", assignee_id="VARCHAR",
                                     disambig_assignee_individual_name_first="VARCHAR",
                                     disambig_assignee_individual_name_last="VARCHAR",
                                     disambig_assignee_organization="VARCHAR", assignee_type="INTEGER",
                                     location_id="VARCHAR"),
    "g_location_disambiguated": dict(location_id="VARCHAR", disambig_city="VARCHAR", disambig_state="VARCHAR",
                                     disambig_country="VARCHAR", latitude="DOUBLE", longitude="DOUBLE",
                                     county="VARCHAR", state_fips="VARCHAR", county_fips="VARCHAR"),
    "g_cpc_current": dict(patent_id="VARCHAR", cpc_sequence="INTEGER", cpc_section="VARCHAR", cpc_class="VARCHAR",
                          cpc_subclass="VARCHAR", cpc_group="VARCHAR", cpc_type="VARCHAR"),
    "g_us_patent_citation": dict(patent_id="VARCHAR", citation_sequence="INTEGER", citation_patent_id="VARCHAR",
                                 citation_date="VARCHAR", record_name="VARCHAR", wipo_kind="VARCHAR",
                                 citation_category="VARCHAR"),
}
# Columns kept in the staged Parquet (names and free text not needed by the workloads are dropped).
KEEP = {
    "g_patent": ["patent_id", "patent_type", "patent_date", "num_claims", "withdrawn"],
    "g_application": ["patent_id", "filing_date"],
    "g_inventor_disambiguated": ["patent_id", "inventor_sequence", "inventor_id", "gender_code", "location_id"],
    "g_assignee_disambiguated": ["patent_id", "assignee_sequence", "assignee_id", "assignee_type", "location_id"],
    "g_location_disambiguated": ["location_id", "disambig_state", "disambig_country"],
    "g_cpc_current": ["patent_id", "cpc_sequence", "cpc_section", "cpc_class", "cpc_subclass", "cpc_type"],
    "g_us_patent_citation": ["patent_id", "citation_patent_id", "citation_category"],
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main(raw, out):
    raw, out = Path(raw), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_tmp"
    tmp.mkdir(exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='10GB'; SET threads=10; SET temp_directory='{tmp}'")
    sums = {}
    for name, cols in TABLES.items():
        z = raw / f"{name}.tsv.zip"
        dst = out / f"{name}.parquet"
        sums[z.name] = dict(sha256=sha256(z), bytes=z.stat().st_size)
        if dst.exists():
            print(f"skip {name} (exists)", flush=True)
            continue
        t0 = time.time()
        subprocess.run(["unzip", "-o", "-q", str(z), "-d", str(tmp)], check=True)
        tsv = tmp / f"{name}.tsv"
        types = ", ".join(f"'{k}': '{v}'" for k, v in cols.items())
        keep = ", ".join(KEEP[name])
        con.execute(f"""
            COPY (SELECT {keep} FROM read_csv('{tsv}', delim='\t', quote='"', escape='"', header=true,
                                              columns={{{types}}}, strict_mode=false))
            TO '{dst}' (FORMAT parquet, COMPRESSION zstd)""")
        n = con.execute(f"SELECT count(*) FROM '{dst}'").fetchone()[0]
        tsv.unlink()
        print(f"{name}: {n:,} rows, {time.time() - t0:.0f}s", flush=True)
    (out / "checksums.json").write_text(json.dumps(sums, indent=2))
    tmp.rmdir()


if __name__ == "__main__":
    main(*sys.argv[1:3])
