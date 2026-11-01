"""Build the task-003 real-data workloads from the staged sources (stage_sources.py).

Each workload is a master file and a using (register) file with the task's five fields
(name, addr, city, zip, state), written to OUT_DIR/<group>/<id>_master.parquet and
<id>_using.parquet, plus manifest_<id>.json. Samples are deterministic: rows are ordered by
hash(id, seed) and the first n are kept, so a rebuild gives identical files.

Development (PPP loans; IRS nonprofit register for 103):
  101 small sample -> register       2,000 second-draw borrowers in Oregon -> Oregon first-draw borrowers
  102 panel across rounds             all second-draw borrowers in Vermont -> Vermont first-draw borrowers
  103 cross-source nonprofits         first-draw nonprofit borrowers in New England -> IRS EO BMF, New England
  104 dense big city                  500 second-draw borrowers in Chicago -> Illinois first-draw borrowers
  105 chains and franchises           2,000 second-draw franchise or accommodation/food (NAICS 72) borrowers in Tennessee -> Tennessee first-draw
  106 sparse fields                   as 102, with the master's street address and ZIP emptied
  107 person-named businesses         2,000 second-draw sole proprietors / self-employed / independent contractors in Louisiana -> Louisiana first-draw
  108 national                        5,000 second-draw borrowers nationwide -> all first-draw borrowers
Held out (OSHA-inspected private establishments, 2015-2025, one row per establishment -> EPA FRS):
  201 small sample -> register        2,000 establishments in Minnesota -> Minnesota FRS facilities
  202 dense big city                  500 establishments in Houston -> Texas FRS facilities
  203 multi-site manufacturing         2,000 manufacturing establishments in Ohio -> Ohio FRS facilities
  204 sparse fields                   2,000 establishments in Washington, street and ZIP emptied -> Washington FRS
  205 national                        5,000 establishments nationwide -> all FRS facilities

usage: python build_workloads.py STAGE_DIR OUT_DIR [ids ...]
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidates import count_candidates  # noqa: E402

FIELDS = ["name", "addr", "city", "zip", "state"]
NEW_ENGLAND = "('ME','NH','VT','MA','RI','CT')"
PERSON = "('Sole Proprietorship','Self-Employed Individuals','Independent Contractors')"


def sample(where, n, seed, table="PPP"):
    """Deterministic sample: first n rows ordered by hash(id, seed)."""
    lim = f"LIMIT {n}" if n else ""
    return f"SELECT * FROM {table} WHERE {where} ORDER BY hash(id || '-{seed}') {lim}"


DEV = {
    101: dict(case="A: small sample to a large register", master=sample("draw = 'PPS' AND state = 'OR'", 2000, 101),
              using="SELECT * FROM PPP WHERE draw = 'PPP' AND state = 'OR'",
              describe="2,000 second-draw PPP borrowers in Oregon -> all Oregon first-draw borrowers"),
    102: dict(case="B: panel linkage across rounds", master=sample("draw = 'PPS' AND state = 'VT'", None, 102),
              using="SELECT * FROM PPP WHERE draw = 'PPP' AND state = 'VT'",
              describe="all second-draw PPP borrowers in Vermont -> all Vermont first-draw borrowers"),
    103: dict(case="C + H: cross-source, nonprofits",
              master=sample(f"draw = 'PPP' AND nonprofit AND state IN {NEW_ENGLAND}", None, 103),
              using=f"SELECT * FROM BMF WHERE state IN {NEW_ENGLAND}",
              describe="first-draw PPP nonprofit borrowers in New England -> IRS EO BMF organizations in New England"),
    104: dict(case="D: dense big-city blocks", master=sample("draw = 'PPS' AND state = 'IL' AND city = 'CHICAGO'", 500, 104),
              using="SELECT * FROM PPP WHERE draw = 'PPP' AND state = 'IL'",
              describe="500 second-draw PPP borrowers in Chicago -> all Illinois first-draw borrowers"),
    105: dict(case="E: chains and franchises",
              master=sample("draw = 'PPS' AND state = 'TN' AND (franchise <> '' OR naics LIKE '72%')", 2000, 105),
              using="SELECT * FROM PPP WHERE draw = 'PPP' AND state = 'TN'",
              describe="2,000 second-draw PPP borrowers in Tennessee with a franchise name or in accommodation and food "
                       "services (NAICS 72) -> all Tennessee first-draw borrowers"),
    106: dict(case="F: sparse fields (street and ZIP missing)", master=sample("draw = 'PPS' AND state = 'VT'", None, 102),
              using="SELECT * FROM PPP WHERE draw = 'PPP' AND state = 'VT'", blank=["addr", "zip"],
              describe="as 102, with the master's street address and ZIP emptied (name, city and state only)"),
    107: dict(case="G: person-named businesses",
              master=sample(f"draw = 'PPS' AND state = 'LA' AND business_type IN {PERSON}", 2000, 107),
              using="SELECT * FROM PPP WHERE draw = 'PPP' AND state = 'LA'",
              describe="2,000 second-draw PPP sole proprietors, self-employed individuals and independent contractors "
                       "in Louisiana -> all Louisiana first-draw borrowers"),
    108: dict(case="I: national scale", master=sample("draw = 'PPS'", 5000, 108),
              using="SELECT * FROM PPP WHERE draw = 'PPP'",
              describe="5,000 second-draw PPP borrowers nationwide -> all first-draw borrowers nationwide"),
}

def osha(where, n, seed, blank=None):
    """Distinct private-sector establishments inspected 2015-2025 (latest inspection kept)."""
    return sample(where, n, seed, table="ESTAB")


HELDOUT = {
    201: dict(case="A + C: small sample to a register kept by another agency", master=osha("state = 'MN'", 2000, 201),
              using="SELECT * FROM FRS WHERE state = 'MN'",
              describe="2,000 OSHA-inspected establishments in Minnesota -> all EPA FRS facilities in Minnesota"),
    202: dict(case="D: dense big-city blocks", master=osha("state = 'TX' AND city = 'HOUSTON'", 500, 202),
              using="SELECT * FROM FRS WHERE state = 'TX'",
              describe="500 OSHA-inspected establishments in Houston -> all EPA FRS facilities in Texas"),
    203: dict(case="E: multi-site firms (manufacturing)", master=osha("state = 'OH' AND naics LIKE '3%'", 2000, 203),
              using="SELECT * FROM FRS WHERE state = 'OH'",
              describe="2,000 OSHA-inspected manufacturing establishments (NAICS 31-33) in Ohio -> all EPA FRS facilities in Ohio"),
    204: dict(case="F: sparse fields (street and ZIP missing)", master=osha("state = 'WA'", 2000, 204),
              using="SELECT * FROM FRS WHERE state = 'WA'", blank=["addr", "zip"],
              describe="2,000 OSHA-inspected establishments in Washington with street address and ZIP emptied "
                       "-> all EPA FRS facilities in Washington"),
    205: dict(case="I: national scale", master=osha("TRUE", 5000, 205), using="SELECT * FROM FRS",
              describe="5,000 OSHA-inspected establishments nationwide -> all EPA FRS facilities nationwide"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def estimates(n_master, n_using, pairs):
    """Rough times in seconds, scaled from the measured synthetic workloads (no method is run).
    Stata reclink2: each master record rescans the register (~0.09 us per register record) and
    scores its candidates (~10-20 us per pair). R port: ~5-8 us per pair. Python reference:
    ~1.0-1.6 us per pair."""
    scan = 9e-8 * n_master * n_using
    return dict(stata_reclink2_s=[round(scan + 10e-6 * pairs), round(scan + 20e-6 * pairs)],
                r_port_s=[round(5e-6 * pairs, 1), round(8e-6 * pairs, 1)],
                python_reference_s=[round(1.0e-6 * pairs, 1), round(1.6e-6 * pairs, 1)])


def build(con, wid, spec, group, out):
    con.execute(f"CREATE OR REPLACE TABLE M AS SELECT * FROM ({spec['master']})")
    for f in spec.get("blank", []):
        con.execute(f"UPDATE M SET {f} = ''")
    con.execute(f"CREATE OR REPLACE TABLE U AS SELECT * FROM ({spec['using']}) ORDER BY id")
    d = out / group
    d.mkdir(parents=True, exist_ok=True)
    cols = ", ".join(["id"] + FIELDS)
    con.execute(f"COPY (SELECT {cols} FROM M ORDER BY hash(id || '-order')) TO '{d / f'{wid}_master.parquet'}' (FORMAT parquet, COMPRESSION zstd)")
    con.execute(f"COPY (SELECT {cols} FROM U) TO '{d / f'{wid}_using.parquet'}' (FORMAT parquet, COMPRESSION zstd)")
    nm = con.execute("SELECT count(*) FROM M").fetchone()[0]
    nu = con.execute("SELECT count(*) FROM U").fetchone()[0]
    pairs, perfect = count_candidates(con, "M", "U")
    filled = {t: {f: round(con.execute(f"SELECT avg(({f} <> '')::INT) FROM {t}").fetchone()[0], 4) for f in FIELDS}
              for t in ("M", "U")}
    dup = con.execute(f"SELECT count(*) - count(DISTINCT ({', '.join(FIELDS)})) FROM U").fetchone()[0]
    big = con.execute("""SELECT city, state, count(*) FROM U WHERE city <> '' GROUP BY 1, 2 ORDER BY 3 DESC LIMIT 1""").fetchone()
    spec_out = dict(id=wid, group=group, case=spec["case"], design=spec["describe"], n_master=nm, n_using=nu,
                    candidate_pairs=pairs, perfect_matches=perfect, duplicate_using_records=int(dup),
                    largest_using_city=f"{big[0]}, {big[1]} ({big[2]:,})" if big else "",
                    share_nonempty=filled, blanked_master_fields=spec.get("blank", []),
                    estimated_seconds=estimates(nm, nu, pairs),
                    files={f"{wid}_master.parquet": sha256(d / f"{wid}_master.parquet"),
                           f"{wid}_using.parquet": sha256(d / f"{wid}_using.parquet")},
                    built=time.strftime("%Y-%m-%d"))
    (d / f"manifest_{wid}.json").write_text(json.dumps(spec_out, indent=2))
    print(f"  {group}/{wid}: {nm:,} x {nu:,}, {pairs:,} candidate pairs, {perfect:,} exact matches", flush=True)


def main(stage, out, ids):
    stage, out = Path(stage), Path(out)
    con = duckdb.connect()
    con.execute("SET memory_limit='10GB'; SET threads=10")
    con.execute(f"CREATE TABLE PPP AS SELECT * FROM '{stage / 'ppp.parquet'}' WHERE name <> '' AND state <> ''")
    con.execute(f"CREATE TABLE BMF AS SELECT * FROM '{stage / 'bmf.parquet'}' WHERE name <> '' AND state <> ''")
    have_heldout = (stage / "osha.parquet").exists() and (stage / "frs.parquet").exists()
    if have_heldout:
        con.execute(f"CREATE TABLE OSHA AS SELECT * FROM '{stage / 'osha.parquet'}' WHERE name <> '' AND state <> ''")
        con.execute(f"CREATE TABLE FRS AS SELECT * FROM '{stage / 'frs.parquet'}' WHERE name <> '' AND state <> ''")
        con.execute("""CREATE TABLE ESTAB AS          -- one row per establishment (latest inspection)
          SELECT * FROM OSHA WHERE owner_type = 'A' AND open_year BETWEEN 2015 AND 2025
          QUALIFY row_number() OVER (PARTITION BY name, addr, city, zip, state ORDER BY open_year DESC, id DESC) = 1""")
    for wid in ids or sorted(DEV) + sorted(HELDOUT):
        if wid in DEV:
            build(con, wid, DEV[wid], "development", out)
        elif wid in HELDOUT and have_heldout:
            build(con, wid, HELDOUT[wid], "heldout", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3:]])
