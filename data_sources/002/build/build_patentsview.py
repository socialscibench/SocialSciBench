"""Build the PatentsView workloads for task 002 (staggered DiD, csdid dripw).

Every workload is a balanced panel: units x periods outcome matrix `y`, baseline
covariates `X` measured before the panel starts, and first-treatment period `gvar`
(0 = never treated within the window). Units treated in or before the first period are
dropped (csdid cannot use always-treated units). Count outcomes are zero-filled: a year
(quarter, month) with no patents counts as zero.

Development (technology areas A-F, by each firm's or inventor's modal area):
  101 firm x year 2000-2019; treatment = first year the firm's state has >= 10 AI patent filings
  102 firm x year 1990-2019; treatment = first co-patent with another company (alliance proxy)
  103 firm x year 2000-2019; treatment = first patent in a technology section new to the firm; 10 covariates
  104 inventor x month 2015-2019; treatment = first filing for a firm other than the inventor's Dec-2014 firm
  105 firm x quarter 2015Q1-2025Q2 (grant dates); treatment = first AI patent (CPC G06N)
  106 = 101 without covariates (built by the loader from 101's file)
  107 inventor x year 2000-2019, binary outcome; treatment = first filing for a firm other than the 1999 firm
  108 3,000 largest firms x year 1990-2019; treatment = first patent with inventors in two or more countries
Held out (technology areas G-H):
  201 mirror of 107 with outcome log(1 + patents);  202 mirror of 105;  203 mirror of 102

Inputs:  the task-001 PatentsView work database (tables P, IP, TEAM, CIT, asg_all, firm_GROUP,
         inv_GROUP built by 001's build_patentsview.py) and the staged g_cpc_current.parquet.
usage:   python build_patentsview.py PV_WORK_DB CPC_PARQUET OUT_DIR [ids ...]
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

RELEASE = "PatentsView PVGPATDIS 2026-04-10"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def write_panel(con, units_sql, long_sql, T, xcols, transform, out, spec):
    """units_sql -> (key, gvar, x...) ; long_sql -> (key, t, value) with t in 1..T.
    Builds the balanced units x T matrix (missing cells = 0), applies `transform`, writes Parquet."""
    U = con.execute(units_sql).fetchnumpy()
    keys = U["key"]
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    L = con.execute(long_sql).fetchnumpy()
    idx = np.searchsorted(keys, L["key"])
    ok = (idx < keys.size) & (keys[np.minimum(idx, keys.size - 1)] == L["key"])
    Y = np.zeros((keys.size, T))
    np.add.at(Y, (idx[ok], L["t"][ok].astype(np.int64) - 1), L["value"][ok].astype(np.float64))
    Y = transform(Y)
    gvar = U["gvar"][order].astype(np.int64)
    cols = {"unit": np.arange(keys.size, dtype=np.int64), "gvar": gvar}
    for c in xcols:
        cols[c] = U[c][order].astype(np.float64)
    for t in range(T):
        cols[f"y{t + 1:03d}"] = Y[:, t]
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), out, compression="zstd")
    cohorts, sizes = np.unique(gvar[gvar > 0], return_counts=True)
    spec = dict(spec, n_units=int(keys.size), n_periods=T, n_covariates=len(xcols), covariates=xcols,
                n_cohorts=int(cohorts.size), n_never_treated=int((gvar == 0).sum()),
                share_treated=round(float((gvar > 0).mean()), 4),
                cohort_size_min=int(sizes.min()) if sizes.size else 0,
                cohort_size_median=float(np.median(sizes)) if sizes.size else 0,
                n_gt=int(cohorts.size * (T - 1)), source=RELEASE, file=out.name, sha256=sha256(out),
                built=time.strftime("%Y-%m-%d"))
    (out.parent / f"manifest_{out.stem}.json").write_text(json.dumps(spec, indent=2))
    print(f"  {out.parent.name}/{out.name}: {keys.size:,} units x {T}, {cohorts.size} cohorts, "
          f"{spec['share_treated']:.0%} treated", flush=True)


def log1p(Y):
    return np.log1p(Y)


def binary(Y):
    return (Y > 0).astype(np.float64)


def base(con, cpc):
    con.execute(f"""
    CREATE OR REPLACE TABLE AIPAT AS SELECT DISTINCT patent_id FROM '{cpc}' WHERE cpc_subclass = 'G06N';
    CREATE OR REPLACE TABLE SWPAT AS SELECT DISTINCT patent_id FROM '{cpc}' WHERE cpc_subclass = 'G06F';
    CREATE OR REPLACE TABLE COPAT AS SELECT patent_id FROM pv.asg_all GROUP BY 1 HAVING count(DISTINCT firm) >= 2;
    CREATE OR REPLACE TABLE FP AS          -- firm patents with filing year and attributes
      SELECT P.firm, P.patent_id, year(P.fdate) AS fy, P.fdate, P.gdate, P.gyear, P.sub, P.sec, P.fregion, P.ftype,
             ln(greatest(P.num_claims, 1)) AS lclaims, t.tsize, t.any_foreign, t.intl,
             (a.patent_id IS NOT NULL) AS ai, (s.patent_id IS NOT NULL) AS sw, (c.patent_id IS NOT NULL) AS cop
      FROM pv.P JOIN pv.TEAM t USING (patent_id)
      LEFT JOIN AIPAT a USING (patent_id) LEFT JOIN SWPAT s USING (patent_id) LEFT JOIN COPAT c USING (patent_id)
      WHERE P.firm IS NOT NULL AND P.fdate IS NOT NULL;
    CREATE OR REPLACE TABLE IPF AS         -- inventor patents with filing date and firm
      SELECT IP.inv, IP.patent_id, IP.firm, IP.sec, IP.g, P.fdate, year(P.fdate) AS fy, t.tsize
      FROM pv.IP JOIN pv.P USING (patent_id) JOIN pv.TEAM t USING (patent_id)
      WHERE P.fdate IS NOT NULL;
    """)


def firm_baseline(con, firms_table, B, name):
    """Firm covariates measured with patents filed up to year B (citations received by grant year B)."""
    con.execute(f"""
    CREATE OR REPLACE TABLE {name} AS
      WITH f AS (SELECT FP.* FROM FP JOIN {firms_table} USING (firm) WHERE FP.fy <= {B}),
      subs AS (SELECT firm, sub, count(*) AS n FROM f WHERE sub IS NOT NULL GROUP BY 1, 2),
      hhi AS (SELECT firm, 1 - sum(n * n) / power(sum(n), 2) AS breadth FROM subs GROUP BY 1),
      inv AS (SELECT FP.firm, count(DISTINCT IPF.inv) AS ninv FROM FP JOIN {firms_table} USING (firm)
              JOIN IPF USING (patent_id) WHERE FP.fy BETWEEN {B - 4} AND {B} GROUP BY 1),
      cit AS (SELECT c.dfirm AS firm, count(*) AS n FROM pv.CIT c JOIN {firms_table} t ON t.firm = c.dfirm
              WHERE c.cyear <= {B} GROUP BY 1)
      SELECT f.firm,
             ln(1 + count(*)) AS log_stock,
             ln(1 + sum(CASE WHEN f.fy >= {B - 4} THEN 1 ELSE 0 END)) AS log_recent5,
             ln(1 + coalesce(any_value(inv.ninv), 0)) AS log_inventors,
             coalesce(any_value(hhi.breadth), 0) AS breadth,
             avg(f.tsize) AS mean_team,
             avg(f.lclaims) AS mean_log_claims,
             avg(f.any_foreign) AS share_foreign_inventors,
             ln(1 + coalesce(any_value(cit.n), 0)) AS log_cites_received,
             {B} - min(f.fy) AS years_since_first,
             CASE WHEN mode(f.ftype) = 3 THEN 1.0 ELSE 0.0 END AS foreign_firm,
             avg(f.sw::INT) AS share_software
      FROM f LEFT JOIN inv USING (firm) LEFT JOIN hhi USING (firm) LEFT JOIN cit USING (firm)
      GROUP BY f.firm;
    """)


def inventor_baseline(con, invs_table, B_date, name):
    """Inventor covariates from patents filed on or before B_date."""
    con.execute(f"""
    CREATE OR REPLACE TABLE {name} AS
      SELECT i.inv,
             ln(1 + count(*)) AS log_prior_patents,
             date_diff('day', min(i.fdate), DATE '{B_date}') / 365.25 AS career_years,
             count(DISTINCT i.firm) AS prior_firms,
             count(DISTINCT i.sec) AS prior_sections,
             avg(i.tsize) AS mean_team,
             CASE WHEN mode(i.g) = 'F' THEN 1.0 ELSE 0.0 END AS female
      FROM IPF i JOIN {invs_table} USING (inv) WHERE i.fdate <= DATE '{B_date}' GROUP BY 1;
    """)


def firm_year_workload(con, wid, grp, y0, y1, active_sql, treat_sql, xcols, B, out, spec):
    """Generic firm x filing-year panel. active_sql selects firms; treat_sql gives (firm, tyear)."""
    T = y1 - y0 + 1
    con.execute(f"CREATE OR REPLACE TABLE W_FIRMS AS {active_sql}")
    firm_baseline(con, "W_FIRMS", B, "W_BASE")
    con.execute(f"""
    CREATE OR REPLACE TABLE W_UNITS AS
      SELECT f.firm AS key_s, coalesce(tr.tyear, 0) AS tyear, b.*
      FROM W_FIRMS f JOIN W_BASE b USING (firm) LEFT JOIN ({treat_sql}) tr USING (firm);
    CREATE OR REPLACE TABLE W_UNITS2 AS
      SELECT row_number() OVER (ORDER BY key_s) AS key,
             CASE WHEN tyear = 0 OR tyear > {y1} THEN 0 ELSE tyear - {y0} + 1 END AS gvar, *
      FROM W_UNITS WHERE tyear = 0 OR tyear > {y0};
    """)
    dropped = con.execute(f"SELECT count(*) FROM W_UNITS WHERE tyear <> 0 AND tyear <= {y0}").fetchone()[0]
    xs = ", ".join(xcols)
    write_panel(con, f"SELECT key, gvar{', ' + xs if xs else ''} FROM W_UNITS2",
                f"""SELECT u.key, FP.fy - {y0} + 1 AS t, count(*)::DOUBLE AS value
                    FROM FP JOIN W_UNITS2 u ON u.key_s = FP.firm WHERE FP.fy BETWEEN {y0} AND {y1} GROUP BY 1, 2""",
                T, xcols, log1p, out,
                dict(spec, id=wid, group=grp, periods=[y0, y1], frequency="year", baseline_year=B,
                     dropped_always_treated=int(dropped), y="log(1 + patents filed in year, later granted)"))


def build(con, wid, out):
    dev_firms = "SELECT firm FROM pv.firm_GROUP WHERE grp = '{g}'"
    if wid in (101,):
        x = ["log_stock", "log_inventors", "breadth", "share_foreign_inventors", "years_since_first"]
        con.execute("""
        CREATE OR REPLACE TABLE FSTATE AS SELECT firm, mode(fregion) AS state FROM FP
          WHERE fregion LIKE 'US-%' AND fregion <> 'US-NA' GROUP BY 1;
        CREATE OR REPLACE TABLE STATE_AI AS
          SELECT state, min(fy) AS tyear FROM (
            SELECT fregion AS state, fy, count(*) AS n FROM FP WHERE ai AND fregion LIKE 'US-%' GROUP BY 1, 2)
          WHERE n >= 10 GROUP BY 1;
        """)
        firm_year_workload(
            con, 101, "development", 2000, 2019,
            f"""SELECT f.firm FROM ({dev_firms.format(g='dev')}) f JOIN FSTATE USING (firm)
                JOIN (SELECT firm FROM FP WHERE fy BETWEEN 1990 AND 1999 GROUP BY 1 HAVING count(*) >= 3) a USING (firm)""",
            "SELECT f.firm, s.tyear FROM FSTATE f JOIN STATE_AI s USING (state)", x, 1999,
            out / "development" / "101.parquet",
            dict(unit="firm x year", sections="A-F", treatment="first year the firm's state has >= 10 AI (CPC G06N) "
                 "patent filings (state = modal assignee state)", sample="US firms with >= 3 patents filed 1990-1999"))
    elif wid in (102, 203):
        x = ["log_stock", "log_inventors", "breadth", "mean_team", "foreign_firm"]
        g = "dev" if wid == 102 else "held"
        firm_year_workload(
            con, wid, "development" if wid == 102 else "heldout", 1990, 2019,
            f"""SELECT f.firm FROM ({dev_firms.format(g=g)}) f
                JOIN (SELECT firm FROM FP WHERE fy BETWEEN 1980 AND 1989 GROUP BY 1 HAVING count(*) >= 5) a USING (firm)""",
            "SELECT firm, min(fy) AS tyear FROM FP WHERE cop GROUP BY 1", x, 1989,
            out / ("development" if wid == 102 else "heldout") / f"{wid}.parquet",
            dict(unit="firm x year", sections="A-F" if wid == 102 else "G-H",
                 treatment="first co-patent (patent with two or more company assignees)",
                 sample="firms with >= 5 patents filed 1980-1989"))
    elif wid == 103:
        x = ["log_stock", "log_recent5", "log_inventors", "breadth", "mean_team", "mean_log_claims",
             "share_foreign_inventors", "log_cites_received", "years_since_first", "foreign_firm"]
        firm_year_workload(
            con, 103, "development", 2000, 2019,
            f"""SELECT f.firm FROM ({dev_firms.format(g='dev')}) f
                JOIN (SELECT DISTINCT firm FROM FP WHERE fy BETWEEN 1995 AND 1999) a USING (firm)""",
            """SELECT n.firm, min(n.fy) AS tyear FROM FP n
               LEFT JOIN (SELECT DISTINCT firm, sec FROM FP WHERE fy <= 1999) o ON o.firm = n.firm AND o.sec = n.sec
               WHERE n.fy >= 2000 AND n.sec IS NOT NULL AND o.firm IS NULL GROUP BY 1""",
            x, 1999, out / "development" / "103.parquet",
            dict(unit="firm x year", sections="A-F", treatment="first patent whose main CPC section the firm had "
                 "not patented in through 1999 (diversification)", sample="firms with a patent filed 1995-1999"))
    elif wid in (105, 202):
        g = "dev" if wid == 105 else "held"
        x = ["log_stock", "log_inventors", "breadth", "share_software", "years_since_first"]
        con.execute(f"""
        CREATE OR REPLACE TABLE W_FIRMS AS SELECT f.firm FROM ({dev_firms.format(g=g)}) f
          JOIN (SELECT DISTINCT firm FROM FP WHERE gyear BETWEEN 2010 AND 2014) a USING (firm);
        """)
        firm_baseline(con, "W_FIRMS", 2014, "W_BASE")
        q = "(year({d}) - 2015) * 4 + quarter({d})"
        con.execute(f"""
        CREATE OR REPLACE TABLE W_UNITS AS
          SELECT f.firm AS key_s, coalesce(tr.tq, 0) AS tq, b.*
          FROM W_FIRMS f JOIN W_BASE b USING (firm)
          LEFT JOIN (SELECT firm, min({q.format(d='gdate')}) AS tq FROM FP WHERE ai GROUP BY 1) tr USING (firm);
        CREATE OR REPLACE TABLE W_UNITS2 AS
          SELECT row_number() OVER (ORDER BY key_s) AS key,
                 CASE WHEN tq = 0 OR tq > 42 THEN 0 ELSE tq END AS gvar, *
          FROM W_UNITS WHERE tq = 0 OR tq > 1;
        """)
        dropped = con.execute("SELECT count(*) FROM W_UNITS WHERE tq <> 0 AND tq <= 1").fetchone()[0]
        write_panel(con, f"SELECT key, gvar, {', '.join(x)} FROM W_UNITS2",
                    f"""SELECT u.key, {q.format(d='FP.gdate')} AS t, count(*)::DOUBLE AS value
                        FROM FP JOIN W_UNITS2 u ON u.key_s = FP.firm
                        WHERE {q.format(d='FP.gdate')} BETWEEN 1 AND 42 GROUP BY 1, 2""",
                    42, x, log1p, out / ("development" if wid == 105 else "heldout") / f"{wid}.parquet",
                    dict(id=wid, group="development" if wid == 105 else "heldout", unit="firm x quarter",
                         sections="A-F" if wid == 105 else "G-H", periods=["2015Q1", "2025Q2"], frequency="quarter",
                         baseline_year=2014, dropped_always_treated=int(dropped),
                         treatment="quarter of the firm's first granted AI patent (CPC G06N)",
                         y="log(1 + patents granted in quarter)", sample="firms with a patent granted 2010-2014"))
    elif wid == 108:
        x = ["log_stock", "log_inventors", "breadth", "foreign_firm", "years_since_first"]
        firm_year_workload(
            con, 108, "development", 1990, 2019,
            f"""SELECT firm FROM (
                  SELECT f.firm, count(*) AS n FROM ({dev_firms.format(g='dev')}) f JOIN FP USING (firm)
                  WHERE FP.gyear BETWEEN 1980 AND 2019 GROUP BY 1 ORDER BY n DESC, f.firm LIMIT 3000) top
                JOIN (SELECT DISTINCT firm FROM FP WHERE fy BETWEEN 1985 AND 1989) a USING (firm)""",
            "SELECT firm, min(fy) AS tyear FROM FP WHERE intl = 1 GROUP BY 1", x, 1989,
            out / "development" / "108.parquet",
            dict(unit="firm x year", sections="A-F", treatment="first patent with inventors in two or more countries",
                 sample="the 3,000 largest A-F firms (patents granted 1980-2019) with a patent filed 1985-1989"))
    elif wid in (104, 107, 201):
        g = "held" if wid == 201 else "dev"
        if wid == 104:
            lo, hi, B = "2010-01-01", "2014-12-31", "2014-12-31"
            T, per = 60, "(year({d}) - 2015) * 12 + month({d})"
            win = "i.fdate BETWEEN DATE '2015-01-01' AND DATE '2019-12-31'"
        else:
            lo, hi, B = "1995-01-01", "1999-12-31", "1999-12-31"
            T, per = 20, "year({d}) - 1999"
            win = "i.fdate BETWEEN DATE '2000-01-01' AND DATE '2019-12-31'"
        x = (["log_prior_patents", "career_years", "prior_firms", "prior_sections", "mean_team"] if wid == 104 else
             ["log_prior_patents", "career_years", "prior_firms", "prior_sections", "female"])
        con.execute(f"""
        CREATE OR REPLACE TABLE W_INVS AS
          SELECT inv FROM pv.inv_GROUP JOIN IPF USING (inv) WHERE grp = '{g}' AND firm IS NOT NULL GROUP BY 1
          HAVING count(*) >= 3 AND sum(CASE WHEN fdate BETWEEN DATE '{lo}' AND DATE '{hi}' THEN 1 ELSE 0 END) >= 1;
        CREATE OR REPLACE TABLE W_BASEFIRM AS   -- firm of the latest patent filed on or before the baseline date
          SELECT inv, arg_max(firm, fdate::VARCHAR || patent_id) AS bfirm
          FROM IPF JOIN W_INVS USING (inv) WHERE firm IS NOT NULL AND fdate <= DATE '{B}' GROUP BY 1;
        """)
        inventor_baseline(con, "W_INVS", B, "W_BASE")
        con.execute(f"""
        CREATE OR REPLACE TABLE W_UNITS AS
          SELECT v.inv AS key_s, coalesce(m.tp, 0) AS tp, b.*
          FROM W_INVS v JOIN W_BASEFIRM bf USING (inv) JOIN W_BASE b USING (inv)
          LEFT JOIN (SELECT i.inv, min({per.format(d='i.fdate')}) AS tp FROM IPF i JOIN W_BASEFIRM bf USING (inv)
                     WHERE i.firm IS NOT NULL AND i.firm <> bf.bfirm AND {win} GROUP BY 1) m USING (inv);
        CREATE OR REPLACE TABLE W_UNITS2 AS
          SELECT row_number() OVER (ORDER BY key_s) AS key, tp AS gvar, * FROM W_UNITS WHERE tp <> 1;
        """)
        dropped = con.execute("SELECT count(*) FROM W_UNITS WHERE tp = 1").fetchone()[0]
        write_panel(con, f"SELECT key, gvar, {', '.join(x)} FROM W_UNITS2",
                    f"""SELECT u.key, {per.format(d='i.fdate')} AS t, count(*)::DOUBLE AS value
                        FROM IPF i JOIN W_UNITS2 u ON u.key_s = i.inv WHERE {win} GROUP BY 1, 2""",
                    T, x, binary if wid == 107 else (log1p if wid == 201 else (lambda Y: Y)),
                    out / ("heldout" if wid == 201 else "development") / f"{wid}.parquet",
                    dict(id=wid, group="heldout" if wid == 201 else "development",
                         unit="inventor x month" if wid == 104 else "inventor x year",
                         sections="G-H" if wid == 201 else "A-F",
                         periods=["2015-01", "2019-12"] if wid == 104 else [2000, 2019],
                         frequency="month" if wid == 104 else "year", baseline_date=B,
                         dropped_always_treated=int(dropped),
                         treatment="first filing for a firm other than the inventor's firm at the baseline date",
                         y={104: "patents filed in month", 107: "1{any patent filed in year}",
                            201: "log(1 + patents filed in year)"}[wid],
                         sample="inventors with >= 3 patents and one filed in the five years before the panel"))


def main(pv_db, cpc, out, ids):
    out = Path(out)
    work = out / "_work_csdid_patentsview.duckdb"
    con = duckdb.connect(str(work))
    con.execute(f"SET memory_limit='11GB'; SET threads=10; SET temp_directory='{out / '_tmp'}'")
    con.execute(f"ATTACH '{pv_db}' AS pv (READ_ONLY)")
    if not con.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name = 'FP' AND database_name = "
                       f"'{work.stem}'").fetchone()[0]:
        base(con, cpc)
    for wid in ids or [101, 102, 103, 104, 105, 107, 108, 201, 202, 203]:
        t0 = time.time()
        build(con, wid, out)
        print(f"  ({wid} built in {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], [int(x) for x in sys.argv[4:]])
