"""Stage the raw sources for the task-003 real-data workloads as Parquet.

Every source is reduced to the five linkage fields in the task's input format, plus the
attributes used to build workloads:
  name, addr, city, zip, state   uppercase ASCII, single spaces, trimmed; "" = missing;
                                 zip = first 5 digits; state = 2 letters
No name or address standardization is applied (task.md puts stnd_compname and
stnd_address out of scope): the real spelling variation is kept.

Sources and outputs (OUT_DIR/<source>.parquet):
  ppp    SBA Paycheck Protection Program loans (public_*_240930.csv; read as Latin-1, since some
         files are not valid UTF-8)
         + draw (PPP = first draw, PPS = second draw), business_type, naics, franchise, nonprofit
  bmf    IRS Exempt Organizations Business Master File (eo1.csv ... eo4.csv) + ein, ntee
  osha   OSHA inspections (OSHA_inspection.zip) + naics, owner_type, open_year
  frs    EPA Facility Registry Service national single file (national_single.zip) + registry_id

usage: python stage_sources.py RAW_ROOT OUT_DIR [ppp bmf osha frs]
RAW_ROOT holds the folders "PPP", "IRS EO BMF", "OSHA", "EPA FRS".
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


def clean(expr):
    """SQL: uppercase ASCII, punctuation kept, whitespace collapsed, trimmed, NULL -> ''."""
    return (f"trim(regexp_replace(regexp_replace(upper(strip_accents(coalesce({expr}, ''))), "
            f"'[^\\x20-\\x7E]', '', 'g'), '\\s+', ' ', 'g'))")


def zip5(expr):
    z = f"coalesce(regexp_extract(coalesce({expr}, ''), '^\\s*([0-9]{{5}})', 1), '')"
    return f"CASE WHEN {z} = '00000' THEN '' ELSE {z} END"


def state2(expr):
    return f"CASE WHEN regexp_matches(upper(trim(coalesce({expr}, ''))), '^[A-Z]{{2}}$') THEN upper(trim({expr})) ELSE '' END"


NA_TOKENS = "('N/A', 'NA', 'NOT AVAILABLE', 'EXEMPTION 6', 'NONE', 'UNKNOWN')"


def na(expr):
    """Treat placeholder strings as missing (e.g. 'N/A', 'EXEMPTION 6', EPA's '(UNKNOWN ADDRESS)')."""
    return f"CASE WHEN {expr} IN {NA_TOKENS} OR {expr} LIKE '(UNKNOWN%' THEN '' ELSE {expr} END"


def stage_ppp(con, raw, out):
    files = sorted((raw / "PPP").glob("public_*_240930.csv"))
    src = ", ".join(f"'{f}'" for f in files)
    con.execute(f"""
    COPY (
      SELECT LoanNumber AS id, ProcessingMethod AS draw,
             {na(clean('BorrowerName'))} AS name, {na(clean('BorrowerAddress'))} AS addr,
             {na(clean('BorrowerCity'))} AS city, {zip5('BorrowerZip')} AS zip,
             CASE WHEN {state2('BorrowerState')} <> '' THEN {state2('BorrowerState')} ELSE {state2('ProjectState')} END AS state,
             BusinessType AS business_type, NAICSCode AS naics, {clean('FranchiseName')} AS franchise,
             coalesce(NonProfit, '') = 'Y' AS nonprofit, DateApproved AS date_approved
      FROM read_csv([{src}], all_varchar=true, header=true, union_by_name=true, encoding='latin-1')
    ) TO '{out / "ppp.parquet"}' (FORMAT parquet, COMPRESSION zstd)""")
    return files


def stage_bmf(con, raw, out):
    files = sorted((raw / "IRS EO BMF").glob("eo[1-4].csv"))
    src = ", ".join(f"'{f}'" for f in files)
    con.execute(f"""
    COPY (
      SELECT EIN AS id, {clean('NAME')} AS name, {na(clean('STREET'))} AS addr, {clean('CITY')} AS city,
             {zip5('ZIP')} AS zip, {state2('STATE')} AS state, NTEE_CD AS ntee, SUBSECTION AS subsection
      FROM read_csv([{src}], all_varchar=true, header=true)
    ) TO '{out / "bmf.parquet"}' (FORMAT parquet, COMPRESSION zstd)""")
    return files


def unzip_csvs(zipfile, tmp):
    tmp.mkdir(parents=True, exist_ok=True)
    subprocess.run(["unzip", "-o", "-q", str(zipfile), "-d", str(tmp)], check=True)
    return sorted(p for p in tmp.rglob("*") if p.suffix.lower() == ".csv")


def stage_osha(con, raw, out):
    z = raw / "OSHA" / "OSHA_inspection.zip"
    csvs = unzip_csvs(z, out / "_tmp_osha")
    src = ", ".join(f"'{f}'" for f in csvs)
    con.execute(f"""
    COPY (
      SELECT activity_nr AS id, {na(clean('estab_name'))} AS name, {na(clean('site_address'))} AS addr,
             {na(clean('site_city'))} AS city, {zip5('site_zip')} AS zip, {state2('site_state')} AS state,
             naics_code AS naics, sic_code AS sic, owner_type, try_cast(left(open_date, 4) AS INT) AS open_year
      FROM read_csv([{src}], all_varchar=true, header=true, union_by_name=true, strict_mode=false, ignore_errors=true)
    ) TO '{out / "osha.parquet"}' (FORMAT parquet, COMPRESSION zstd)""")
    subprocess.run(["rm", "-rf", str(out / "_tmp_osha")], check=True)
    return [z]


def stage_frs(con, raw, out):
    z = raw / "EPA FRS" / "national_single.zip"
    csvs = unzip_csvs(z, out / "_tmp_frs")
    src = ", ".join(f"'{f}'" for f in csvs)
    con.execute(f"""
    COPY (
      SELECT REGISTRY_ID AS id, {na(clean('PRIMARY_NAME'))} AS name, {na(clean('LOCATION_ADDRESS'))} AS addr,
             {na(clean('CITY_NAME'))} AS city, {zip5('POSTAL_CODE')} AS zip, {state2('STATE_CODE')} AS state,
             NAICS_CODES AS naics
      FROM read_csv([{src}], all_varchar=true, header=true, union_by_name=true, strict_mode=false, ignore_errors=true)
    ) TO '{out / "frs.parquet"}' (FORMAT parquet, COMPRESSION zstd)""")
    subprocess.run(["rm", "-rf", str(out / "_tmp_frs")], check=True)
    return [z]


def main(raw, out, which):
    raw, out = Path(raw), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET memory_limit='10GB'; SET threads=10")
    sums_path = out / "checksums.json"
    sums = json.loads(sums_path.read_text()) if sums_path.exists() else {}
    for name in which or ["ppp", "bmf", "osha", "frs"]:
        files = {"ppp": stage_ppp, "bmf": stage_bmf, "osha": stage_osha, "frs": stage_frs}[name](con, raw, out)
        for f in files:
            sums[f"{f.parent.name}/{f.name}"] = dict(sha256=sha256(f), bytes=f.stat().st_size)
        n = con.execute(f"SELECT count(*) FROM '{out / (name + '.parquet')}'").fetchone()[0]
        print(f"{name}: {n:,} rows", flush=True)
    sums_path.write_text(json.dumps(sums, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
