"""Stage the SEC Insider Transactions Data Sets (Forms 3/4/5) as Parquet.

Reads every YYYYqN_form345.zip in RAW_DIR, keeps the columns the workloads need from
SUBMISSION, REPORTINGOWNER and NONDERIV_TRANS, parses dates, and writes one Parquet file
per table. The ZIPs are not modified. SHA-256 checksums go to checksums.json.

usage: python stage_sec.py RAW_DIR OUT_DIR
"""
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

import duckdb

COLS = {
    "SUBMISSION": ["ACCESSION_NUMBER", "FILING_DATE", "PERIOD_OF_REPORT", "DOCUMENT_TYPE", "ISSUERCIK"],
    "REPORTINGOWNER": ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE", "RPTOWNER_STATE"],
    "NONDERIV_TRANS": ["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES", "TRANS_PRICEPERSHARE",
                       "TRANS_ACQUIRED_DISP_CD", "SHRS_OWND_FOLWNG_TRANS"],
}
DATES = {"FILING_DATE", "PERIOD_OF_REPORT", "TRANS_DATE"}
NUMS = {"TRANS_SHARES", "TRANS_PRICEPERSHARE", "SHRS_OWND_FOLWNG_TRANS"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def expr(c):
    if c in DATES:
        return f"try_strptime({c}, '%d-%b-%Y')::DATE AS {c}"
    if c in NUMS:
        return f"try_cast({c} AS DOUBLE) AS {c}"
    return c


def main(raw, out):
    raw, out = Path(raw), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_tmp"
    zips = sorted(raw.glob("*_form345.zip"))
    sums = {z.name: dict(sha256=sha256(z), bytes=z.stat().st_size) for z in zips}
    # Extract the three tables of every quarter into tmp/<table>/<quarter>.tsv
    for z in zips:
        q = z.name.split("_")[0]
        with zipfile.ZipFile(z) as zf:
            for t in COLS:
                (tmp / t).mkdir(parents=True, exist_ok=True)
                with zf.open(f"{t}.tsv") as src, open(tmp / t / f"{q}.tsv", "wb") as dst:
                    shutil.copyfileobj(src, dst)
    con = duckdb.connect()
    con.execute("SET memory_limit='8GB'; SET threads=8")
    for t, cols in COLS.items():
        sel = ", ".join(expr(c) for c in cols)
        con.execute(f"""
            COPY (SELECT {sel}, regexp_extract(filename, '(\\d{{4}}q\\d)', 1) AS QUARTER
                  FROM read_csv('{tmp / t}/*.tsv', delim='\t', header=true, all_varchar=true, quote='',
                                union_by_name=true, filename=true, strict_mode=false, null_padding=true))
            TO '{out / (t.lower() + ".parquet")}' (FORMAT parquet, COMPRESSION zstd)""")
        n = con.execute(f"SELECT count(*) FROM '{out / (t.lower() + '.parquet')}'").fetchone()[0]
        print(f"{t}: {n:,} rows", flush=True)
    shutil.rmtree(tmp)
    (out / "checksums.json").write_text(json.dumps(sums, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:3])
