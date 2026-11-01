"""Build held-out workload 301 for task 002 from the staged SEC insider data.

301 firm x year 2006-2025, firms with insider filings in every year (balanced).
    Treatment = first year a new CEO appears after a different CEO was observed at the firm
    (first CEO turnover); firms with no turnover are never treated.
    Outcome = net purchase ratio of the firm's insiders, (P shares - S shares) / (P + S), 0 if no trades.
    Covariates (2006): log insiders, share directors, share officers, log(1 + trades), mean boards per director.
Definitions of insiders, years and trades follow task 001's build_sec.py (original Forms 3/4/5;
directors and officers; present from first to last filing year at the firm).

usage: python build_sec.py SEC_STAGE_DIR OUT_DIR
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

T0, T1 = 2006, 2025


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main(stage, out):
    stage, out = Path(stage), Path(out)
    con = duckdb.connect()
    con.execute("SET memory_limit='8GB'; SET threads=10")
    con.execute(f"""
    CREATE TABLE FIL AS
      SELECT DISTINCT ON (s.ACCESSION_NUMBER, o.RPTOWNERCIK)
             s.ACCESSION_NUMBER AS acc, year(s.FILING_DATE) AS t, s.ISSUERCIK AS firm, o.RPTOWNERCIK AS insider,
             o.RPTOWNER_RELATIONSHIP ILIKE '%director%' AS is_dir,
             o.RPTOWNER_RELATIONSHIP ILIKE '%officer%' AS is_off,
             regexp_matches(coalesce(o.RPTOWNER_TITLE, ''), '(?i)(chief executive|\\bceo\\b)') AS is_ceo
      FROM '{stage}/submission.parquet' s JOIN '{stage}/reportingowner.parquet' o USING (ACCESSION_NUMBER)
      WHERE s.DOCUMENT_TYPE IN ('3', '4', '5') AND year(s.FILING_DATE) BETWEEN {T0} AND {T1}
        AND (o.RPTOWNER_RELATIONSHIP ILIKE '%director%' OR o.RPTOWNER_RELATIONSHIP ILIKE '%officer%');
    CREATE TABLE TR AS
      SELECT f.insider, f.firm,
             CASE WHEN year(n.TRANS_DATE) BETWEEN 2003 AND 2026 THEN year(n.TRANS_DATE) ELSE f.t END AS t,
             count(*) AS ntrans,
             sum(CASE WHEN n.TRANS_CODE = 'P' THEN coalesce(n.TRANS_SHARES, 0) ELSE 0 END) AS shbuy,
             sum(CASE WHEN n.TRANS_CODE = 'S' THEN coalesce(n.TRANS_SHARES, 0) ELSE 0 END) AS shsell
      FROM FIL f JOIN '{stage}/nonderiv_trans.parquet' n ON n.ACCESSION_NUMBER = f.acc GROUP BY 1, 2, 3;
    CREATE TABLE SPAN AS SELECT insider, firm, min(t) AS y0, max(t) AS y1 FROM FIL GROUP BY 1, 2;
    CREATE TABLE FLAGS AS
      SELECT insider, firm, t, bool_or(is_dir) AS is_dir, bool_or(is_off) AS is_off, bool_or(is_ceo) AS is_ceo
      FROM FIL GROUP BY 1, 2, 3;
    CREATE TABLE IFY AS
      SELECT g.insider, g.firm, g.t, fl.is_dir, fl.is_off, fl.is_ceo
      FROM (SELECT s.insider, s.firm, r.t FROM SPAN s, range(s.y0, s.y1 + 1) r(t)) g
      ASOF JOIN FLAGS fl ON fl.insider = g.insider AND fl.firm = g.firm AND g.t >= fl.t;
    CREATE TABLE BOARDS AS SELECT insider, t, count(DISTINCT firm) AS nboards FROM IFY GROUP BY 1, 2;
    CREATE TABLE FIRMY AS
      SELECT i.firm, i.t, count(*) AS ninsiders, avg(i.is_dir::INT) AS share_dir, avg(i.is_off::INT) AS share_off,
             coalesce(avg(CASE WHEN i.is_dir THEN b.nboards END), 0) AS mean_boards_dir,
             coalesce(sum(tr.ntrans), 0) AS ntrans,
             coalesce(sum(tr.shbuy), 0) AS shbuy, coalesce(sum(tr.shsell), 0) AS shsell
      FROM IFY i JOIN BOARDS b USING (insider, t)
      LEFT JOIN TR tr ON tr.insider = i.insider AND tr.firm = i.firm AND tr.t = i.t GROUP BY 1, 2;
    CREATE TABLE BAL AS SELECT firm FROM FIRMY WHERE t BETWEEN {T0} AND {T1} GROUP BY 1 HAVING count(*) = {T1 - T0 + 1};
    CREATE TABLE CEO_FIRST AS SELECT firm, insider, min(t) AS t0 FROM IFY WHERE is_ceo GROUP BY 1, 2;
    CREATE TABLE EVENT AS                  -- join form (not correlated EXISTS; see task 001 notes)
      SELECT c.firm, min(c.t0) AS event_year FROM CEO_FIRST c
      JOIN CEO_FIRST d ON d.firm = c.firm AND d.insider <> c.insider AND d.t0 < c.t0 GROUP BY 1;
    CREATE TABLE UNITS AS
      SELECT b.firm, coalesce(e.event_year, 0) AS ey,
             ln(f.ninsiders) AS log_insiders, f.share_dir AS share_directors, f.share_off AS share_officers,
             ln(1 + f.ntrans) AS log_trades, f.mean_boards_dir AS mean_boards_directors
      FROM BAL b JOIN FIRMY f ON f.firm = b.firm AND f.t = {T0} LEFT JOIN EVENT e ON e.firm = b.firm;
    """)
    xcols = ["log_insiders", "share_directors", "share_officers", "log_trades", "mean_boards_directors"]
    U = con.execute("SELECT * FROM UNITS WHERE ey = 0 OR ey > 2006 ORDER BY firm").df()
    dropped = con.execute("SELECT count(*) FROM UNITS WHERE ey <> 0 AND ey <= 2006").fetchone()[0]
    L = con.execute(f"""SELECT f.firm, f.t - {T0 - 1} AS t,
                        CASE WHEN f.shbuy + f.shsell > 0 THEN (f.shbuy - f.shsell) / (f.shbuy + f.shsell) ELSE 0 END AS v
                        FROM FIRMY f JOIN BAL USING (firm) WHERE f.t BETWEEN {T0} AND {T1}""").df()
    T = T1 - T0 + 1
    pos = {k: i for i, k in enumerate(U.firm)}
    Y = np.zeros((len(U), T))
    for firm, t, v in L.itertuples(index=False):
        if firm in pos:
            Y[pos[firm], t - 1] = v
    gvar = np.where(U.ey.to_numpy() == 0, 0, U.ey.to_numpy() - (T0 - 1)).astype(np.int64)
    cols = {"unit": np.arange(len(U), dtype=np.int64), "gvar": gvar}
    for c in xcols:
        cols[c] = U[c].to_numpy(np.float64)
    for t in range(T):
        cols[f"y{t + 1:03d}"] = Y[:, t]
    dst = out / "heldout" / "301.parquet"
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), dst, compression="zstd")
    cohorts, sizes = np.unique(gvar[gvar > 0], return_counts=True)
    spec = dict(id=301, group="heldout", unit="firm x year", periods=[T0, T1], frequency="year", baseline_year=T0,
                treatment="first CEO turnover (a new CEO appears after a different CEO was observed)",
                y="net purchase ratio of insiders, (P - S shares) / (P + S shares), 0 if no open-market trades",
                sample="firms with director or officer filings in every year 2006-2025",
                n_units=len(U), n_periods=T, n_covariates=len(xcols), covariates=xcols, n_cohorts=int(cohorts.size),
                n_never_treated=int((gvar == 0).sum()), share_treated=round(float((gvar > 0).mean()), 4),
                cohort_size_min=int(sizes.min()), cohort_size_median=float(np.median(sizes)),
                n_gt=int(cohorts.size * (T - 1)), dropped_always_treated=int(dropped),
                source="SEC Insider Transactions Data Sets, 2006q1-2026q3", file=dst.name, sha256=sha256(dst),
                built=time.strftime("%Y-%m-%d"))
    (dst.parent / "manifest_301.json").write_text(json.dumps(spec, indent=2))
    print(f"  heldout/301.parquet: {len(U):,} firms x {T}, {cohorts.size} cohorts, {spec['share_treated']:.0%} treated")


if __name__ == "__main__":
    main(*sys.argv[1:3])
