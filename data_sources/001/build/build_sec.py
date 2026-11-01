"""Build held-out workloads 301-304 from the staged SEC insider data (stage_sec.py).

Definitions (see patentsview_workload_design.md, section 4):
- Filings: original Forms 3, 4 and 5 (amendments 3/A, 4/A, 5/A are excluded); year = filing year.
- Insiders: reporting owners who are a director or an officer on the filing.
- An insider is present at a firm in every year from their first to their last filing there.
- Transactions: open-market purchases (TRANS_CODE 'P') and sales ('S'), dated by the
  transaction date when it falls in 2003-2026, otherwise by the filing date.
- Years 2006-2025 (2026 is incomplete).

usage: python build_sec.py STAGE_DIR OUT_DIR
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import duckdb

SOURCE = "SEC Insider Transactions Data Sets, 2006q1-2026q3"
T0, T1 = 2006, 2025


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def write(con, sql, out, spec):
    out.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY ({sql}) TO '{out}' (FORMAT parquet, COMPRESSION zstd)")
    n = con.execute(f"SELECT count(*) FROM '{out}'").fetchone()[0]
    spec = dict(spec, rows=n, source=SOURCE, file=out.name, sha256=sha256(out), built=time.strftime("%Y-%m-%d"))
    (out.parent / f"manifest_{out.stem}.json").write_text(json.dumps(spec, indent=2))
    print(f"  {out.parent.name}/{out.name}: {n:,} rows", flush=True)


def xcols(con, table):
    return ", ".join(c for c, in con.execute(
        f"SELECT column_name FROM (DESCRIBE {table}) WHERE column_name LIKE 'x%' ORDER BY 1").fetchall())


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
             o.RPTOWNER_RELATIONSHIP ILIKE '%tenpercent%' AS is_ten,
             regexp_matches(coalesce(o.RPTOWNER_TITLE, ''), '(?i)(chief executive|\\bceo\\b)') AS is_ceo,
             regexp_matches(coalesce(o.RPTOWNER_TITLE, ''), '(?i)(chief financial|\\bcfo\\b)') AS is_cfo,
             o.RPTOWNER_STATE AS state
      FROM '{stage}/submission.parquet' s JOIN '{stage}/reportingowner.parquet' o USING (ACCESSION_NUMBER)
      WHERE s.DOCUMENT_TYPE IN ('3', '4', '5') AND year(s.FILING_DATE) BETWEEN {T0} AND {T1}
        AND (o.RPTOWNER_RELATIONSHIP ILIKE '%director%' OR o.RPTOWNER_RELATIONSHIP ILIKE '%officer%');
    CREATE TABLE TR AS                    -- open-market trades and holdings per insider-firm-year
      SELECT f.insider, f.firm,
             CASE WHEN year(n.TRANS_DATE) BETWEEN 2003 AND 2026 THEN year(n.TRANS_DATE) ELSE f.t END AS t,
             count(*) AS ntrans,
             sum(CASE WHEN n.TRANS_CODE = 'P' THEN 1 ELSE 0 END) AS nbuy,
             sum(CASE WHEN n.TRANS_CODE = 'S' THEN 1 ELSE 0 END) AS nsell,
             sum(CASE WHEN n.TRANS_CODE = 'P' THEN coalesce(n.TRANS_SHARES, 0) ELSE 0 END) AS shbuy,
             sum(CASE WHEN n.TRANS_CODE = 'S' THEN coalesce(n.TRANS_SHARES, 0) ELSE 0 END) AS shsell,
             arg_max(n.SHRS_OWND_FOLWNG_TRANS, n.TRANS_DATE) AS holdings
      FROM FIL f JOIN '{stage}/nonderiv_trans.parquet' n ON n.ACCESSION_NUMBER = f.acc GROUP BY 1, 2, 3;
    CREATE TABLE SPAN AS
      SELECT insider, firm, min(t) AS y0, max(t) AS y1 FROM FIL GROUP BY 1, 2;
    CREATE TABLE FLAGS AS                 -- roles observed in the year's filings
      SELECT insider, firm, t, bool_or(is_dir) AS is_dir, bool_or(is_off) AS is_off, bool_or(is_ten) AS is_ten,
             bool_or(is_ceo) AS is_ceo, bool_or(is_cfo) AS is_cfo, count(*) AS nfil
      FROM FIL GROUP BY 1, 2, 3;
    CREATE TABLE IFY AS                   -- insider x firm x year, filled within each span; roles carried forward
      SELECT g.insider, g.firm, g.t, g.t - s.y0 AS tenure,
             fl.is_dir, fl.is_off, fl.is_ten, fl.is_ceo, fl.is_cfo,
             coalesce(cur.nfil, 0) AS nfil
      FROM (SELECT s.insider, s.firm, r.t FROM SPAN s, range(s.y0, s.y1 + 1) r(t)) g
      JOIN SPAN s USING (insider, firm)
      ASOF JOIN FLAGS fl ON fl.insider = g.insider AND fl.firm = g.firm AND g.t >= fl.t
      LEFT JOIN FLAGS cur ON cur.insider = g.insider AND cur.firm = g.firm AND cur.t = g.t;
    CREATE TABLE BOARDS AS SELECT insider, t, count(DISTINCT firm) AS nboards FROM IFY GROUP BY 1, 2;
    CREATE TABLE FIRMY AS                 -- firm x year aggregates
      SELECT i.firm, i.t, count(*) AS ninsiders,
             avg(i.is_dir::INT) AS share_dir, avg(i.is_off::INT) AS share_off, avg(i.is_ten::INT) AS share_ten,
             avg(CASE WHEN i.tenure = 0 THEN 1.0 ELSE 0.0 END) AS share_new, avg(i.tenure) AS mean_tenure,
             coalesce(avg(CASE WHEN i.is_dir THEN b.nboards END), 0) AS mean_boards_dir,
             sum(i.nfil) AS nfil,
             coalesce(sum(tr.ntrans), 0) AS ntrans,
             avg(CASE WHEN coalesce(tr.nbuy, 0) > 0 THEN 1.0 ELSE 0.0 END) AS share_buy,
             avg(CASE WHEN coalesce(tr.nsell, 0) > 0 THEN 1.0 ELSE 0.0 END) AS share_sell,
             coalesce(sum(tr.shbuy), 0) AS shbuy, coalesce(sum(tr.shsell), 0) AS shsell
      FROM IFY i JOIN BOARDS b USING (insider, t)
      LEFT JOIN TR tr ON tr.insider = i.insider AND tr.firm = i.firm AND tr.t = i.t
      GROUP BY 1, 2;
    CREATE TABLE FIRM0 AS
      SELECT firm, min(t) AS first_year, mode(state) AS state FROM FIL GROUP BY 1;
    """)

    # 301: insider x firm x year, y = 1{open-market purchase}
    con.execute("""
    CREATE TABLE W301 AS
      SELECT i.insider, i.firm, i.t,
             CASE WHEN coalesce(tr.nbuy, 0) > 0 THEN 1.0 ELSE 0.0 END AS y,
             i.is_dir::INT AS x01_director, i.is_off::INT AS x02_officer, i.is_ten::INT AS x03_tenpct_owner,
             i.is_ceo::INT AS x04_ceo, i.is_cfo::INT AS x05_cfo, i.tenure AS x06_tenure,
             b.nboards AS x07_n_boards, ln(1 + greatest(coalesce(l.holdings, 0), 0)) AS x08_log_holdings_lag,
             ln(1 + coalesce(l.ntrans, 0)) AS x09_log_trades_lag,
             CASE WHEN coalesce(l.nsell, 0) > 0 THEN 1 ELSE 0 END AS x10_sold_lag,
             ln(fy.ninsiders) AS x11_log_firm_insiders, fy.share_new AS x12_firm_turnover
      FROM IFY i JOIN BOARDS b USING (insider, t) JOIN FIRMY fy ON fy.firm = i.firm AND fy.t = i.t
      LEFT JOIN TR tr ON tr.insider = i.insider AND tr.firm = i.firm AND tr.t = i.t
      LEFT JOIN TR l ON l.insider = i.insider AND l.firm = i.firm AND l.t = i.t - 1;
    """)
    write(con, f"""SELECT y, {xcols(con, 'W301')},
                          dense_rank() OVER (ORDER BY insider) - 1 AS fe_insider,
                          dense_rank() OVER (ORDER BY firm) - 1 AS fe_firm, t - {T0} AS fe_year,
                          dense_rank() OVER (ORDER BY firm) - 1 AS cluster FROM W301""",
          out / "heldout" / "301.parquet",
          dict(id=301, role="heldout", unit="insider x firm x year", y="1{open-market purchase in year}",
               absorb=["fe_insider", "fe_firm", "fe_year"], cluster="cluster", years=[T0, T1]))

    # 302: firm x year governance panel, y = share of insiders who are new (firm's first year dropped)
    con.execute("""
    CREATE TABLE W302 AS
      SELECT fy.firm, fy.t, f0.state, fy.share_new AS y,
             ln(fy.ninsiders) AS x01_log_n_insiders, fy.share_dir AS x02_share_directors,
             fy.share_off AS x03_share_officers, fy.share_ten AS x04_share_tenpct,
             ln(1 + coalesce(l.ntrans, 0)) AS x05_log_trades_lag, coalesce(l.share_buy, 0) AS x06_share_buyers_lag,
             coalesce(l.share_sell, 0) AS x07_share_sellers_lag, fy.mean_tenure AS x08_mean_tenure,
             fy.mean_boards_dir AS x09_mean_boards_directors, ln(1 + coalesce(l.nfil, 0)) AS x10_log_filings_lag
      FROM FIRMY fy JOIN FIRM0 f0 USING (firm)
      LEFT JOIN FIRMY l ON l.firm = fy.firm AND l.t = fy.t - 1
      WHERE fy.t > f0.first_year;
    """)
    write(con, f"""SELECT y, {xcols(con, 'W302')},
                          dense_rank() OVER (ORDER BY firm) - 1 AS fe_firm, t - {T0} AS fe_year,
                          dense_rank() OVER (ORDER BY coalesce(state, 'NA'), t) - 1 AS fe_state_year,
                          dense_rank() OVER (ORDER BY firm) - 1 AS cluster FROM W302""",
          out / "heldout" / "302.parquet",
          dict(id=302, role="heldout", unit="firm x year", y="share of insiders who are new in the year",
               absorb=["fe_firm", "fe_year", "fe_state_year"], cluster="cluster", years=[T0 + 1, T1],
               note="state x year replaces industry x year (SIC codes are not in the insider data)"))

    # 303: board-interlock dyads, y = 1{firms share a director or officer in the year}
    con.execute("""
    CREATE TABLE SHARE AS
      SELECT a.firm AS i, b.firm AS j, a.t, count(*) AS nshared
      FROM IFY a JOIN IFY b ON a.insider = b.insider AND a.t = b.t AND a.firm < b.firm GROUP BY 1, 2, 3;
    CREATE TABLE FIRSTSHARE AS            -- first year each insider links the two firms
      SELECT a.firm AS i, b.firm AS j, a.insider, min(a.t) AS t0
      FROM IFY a JOIN IFY b ON a.insider = b.insider AND a.t = b.t AND a.firm < b.firm GROUP BY 1, 2, 3;
    CREATE TABLE PAIRS AS SELECT DISTINCT i, j FROM SHARE;
    CREATE TABLE FSCUM AS                 -- cumulative number of distinct insiders linking the pair
      SELECT i, j, t0, sum(count(*)) OVER (PARTITION BY i, j ORDER BY t0) AS n
      FROM FIRSTSHARE GROUP BY 1, 2, 3;
    CREATE TABLE W303 AS
      SELECT p.i, p.j, fi.t,
             CASE WHEN s.nshared IS NOT NULL THEN 1.0 ELSE 0.0 END AS y,
             -- Single-firm-by-year variables are absorbed by firm_i#year and firm_j#year, so the
             -- controls vary at the pair-year level (lagged ties and products of the two firms' traits).
             CASE WHEN s2.nshared IS NOT NULL THEN 1 ELSE 0 END AS x01_tie_lag2,
             ln(1 + coalesce(s1.nshared, 0)) AS x02_log_shared_lag,
             CASE WHEN coalesce(sb.n, 0) > 0 THEN 1 ELSE 0 END AS x03_ever_tied_before,
             ln(fi.ninsiders) * ln(fj.ninsiders) AS x04_insiders_product,
             fi.share_new * fj.share_new AS x05_turnover_product,
             fi.share_dir * fj.share_dir AS x06_director_share_product,
             ln(1 + coalesce(li.ntrans, 0)) * ln(1 + coalesce(lj.ntrans, 0)) AS x07_trades_lag_product,
             fi.mean_boards_dir * fj.mean_boards_dir AS x08_board_busyness_product,
             CASE WHEN s1.nshared IS NOT NULL THEN 1 ELSE 0 END AS x09_tie_lag,
             ln(1 + coalesce(sb.n, 0)) AS x10_log_shared_before
      FROM PAIRS p JOIN FIRMY fi ON fi.firm = p.i JOIN FIRMY fj ON fj.firm = p.j AND fj.t = fi.t
      ASOF LEFT JOIN FSCUM sb ON sb.i = p.i AND sb.j = p.j AND fi.t - 1 >= sb.t0
      LEFT JOIN SHARE s ON s.i = p.i AND s.j = p.j AND s.t = fi.t
      LEFT JOIN SHARE s1 ON s1.i = p.i AND s1.j = p.j AND s1.t = fi.t - 1
      LEFT JOIN SHARE s2 ON s2.i = p.i AND s2.j = p.j AND s2.t = fi.t - 2
      LEFT JOIN FIRMY li ON li.firm = p.i AND li.t = fi.t - 1
      LEFT JOIN FIRMY lj ON lj.firm = p.j AND lj.t = fi.t - 1;
    """)
    write(con, f"""SELECT y, {xcols(con, 'W303')},
                          dense_rank() OVER (ORDER BY i, t) - 1 AS fe_firm_i_year,
                          dense_rank() OVER (ORDER BY j, t) - 1 AS fe_firm_j_year,
                          dense_rank() OVER (ORDER BY i, j) - 1 AS fe_pair,
                          dense_rank() OVER (ORDER BY i, j) - 1 AS cluster FROM W303""",
          out / "heldout" / "303.parquet",
          dict(id=303, role="heldout", unit="firm pair x year (pairs that ever share an insider; years both firms present)",
               y="1{firms share a director or officer in the year}",
               absorb=["fe_firm_i_year", "fe_firm_j_year", "fe_pair"], cluster="cluster", years=[T0, T1]))

    # 304: event study around CEO turnover, y = net purchase ratio of the firm's insiders
    con.execute("""
    CREATE TABLE CEOY AS SELECT DISTINCT firm, insider, t FROM IFY WHERE is_ceo;
    CREATE TABLE CEO_FIRST AS SELECT firm, insider, min(t) AS t0 FROM CEOY GROUP BY 1, 2;
    -- First year a new CEO appears after a different CEO was observed. Written as a join:
    -- DuckDB 1.5.6 returned too few rows for the equivalent correlated EXISTS (905 vs 7,283 firms).
    CREATE TABLE EVENT AS
      SELECT c.firm, min(c.t0) AS event_year FROM CEO_FIRST c
      JOIN CEO_FIRST d ON d.firm = c.firm AND d.insider <> c.insider AND d.t0 < c.t0
      GROUP BY 1;
    """)
    dummies = []
    for k in (-5, -4, -3, -2, 0, 1, 2, 3, 4, 5):
        cond = f"rel <= {k}" if k == -5 else (f"rel >= {k}" if k == 5 else f"rel = {k}")
        dummies.append(f"CASE WHEN rel IS NOT NULL AND {cond} THEN 1 ELSE 0 END AS d_{'m' if k < 0 else 'p'}{abs(k)}")
    con.execute(f"""
    CREATE TABLE W304 AS
      SELECT fy.firm, fy.t,
             CASE WHEN fy.shbuy + fy.shsell > 0 THEN (fy.shbuy - fy.shsell) / (fy.shbuy + fy.shsell) ELSE 0 END AS y,
             {", ".join(dummies)},
             ln(fy.ninsiders) AS x01_log_n_insiders, fy.share_dir AS x02_share_directors,
             fy.share_off AS x03_share_officers, fy.mean_tenure AS x04_mean_tenure,
             ln(1 + coalesce(l.ntrans, 0)) AS x05_log_trades_lag, coalesce(l.share_buy, 0) AS x06_share_buyers_lag,
             fy.share_ten AS x07_share_tenpct, fy.mean_boards_dir AS x08_mean_boards_directors,
             ln(1 + fy.nfil) AS x09_log_filings, coalesce(l.share_new, 0) AS x10_turnover_lag
      FROM (SELECT FIRMY.*, FIRMY.t - e.event_year AS rel FROM FIRMY LEFT JOIN EVENT e USING (firm)) fy
      JOIN FIRM0 f0 USING (firm)
      LEFT JOIN FIRMY l ON l.firm = fy.firm AND l.t = fy.t - 1;
    """)
    ds = ", ".join(d.split(" AS ")[-1] for d in dummies)
    write(con, f"""SELECT y, {ds}, {xcols(con, 'W304')},
                          dense_rank() OVER (ORDER BY firm) - 1 AS fe_firm, t - {T0} AS fe_year,
                          dense_rank() OVER (ORDER BY firm) - 1 AS cluster FROM W304""",
          out / "heldout" / "304.parquet",
          dict(id=304, role="heldout", unit="firm x year", y="net purchase ratio (P - S shares) / (P + S shares)",
               absorb=["fe_firm", "fe_year"], cluster="cluster", event_dummies=ds.split(", "),
               event="first year a new CEO appears after a different CEO was observed", years=[T0, T1]))


if __name__ == "__main__":
    main(*sys.argv[1:3])
