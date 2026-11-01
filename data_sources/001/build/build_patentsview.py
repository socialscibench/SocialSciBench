"""Build the PatentsView workloads of patentsview_workload_design.md.

Development workloads 101-108 are written as ID structures only (fixed-effect and
cluster codes, plus real event-time dummies for 108); the loader adds synthetic y and X.
Held-out workloads 201-206 carry real outcomes and controls.

Inputs:  staged Parquet tables from stage_patentsview.py (STAGE_DIR)
Outputs: OUT_DIR/development/<id>.parquet, OUT_DIR/heldout/<id>.parquet, and a
         manifest_<id>.json next to each file.

usage: python build_patentsview.py STAGE_DIR OUT_DIR [step ...]
steps: base hist firmyear ip invyear fy dyad  (default: all)
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import duckdb
import numpy as np

RELEASE = "PVGPATDIS 2026-04-10"
DEV, HELD = ("A", "B", "C", "D", "E", "F"), ("G", "H")
Y0, Y1 = 1980, 2019          # grant years kept in workload rows (complete 5-year citation windows)
FY0, FY1 = 1980, 2015        # filing years for the firm-year panels


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def write(con, sql, out, spec):
    """Write a query to Parquet and a manifest next to it."""
    out.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY ({sql}) TO '{out}' (FORMAT parquet, COMPRESSION zstd)")
    n = con.execute(f"SELECT count(*) FROM '{out}'").fetchone()[0]
    spec = dict(spec, rows=n, source=RELEASE, file=out.name, sha256=sha256(out),
                built=time.strftime("%Y-%m-%d"))
    (out.parent / f"manifest_{out.stem}.json").write_text(json.dumps(spec, indent=2))
    print(f"  {out.parent.name}/{out.name}: {n:,} rows", flush=True)


def secs(group):
    return "(" + ",".join(f"'{s}'" for s in group) + ")"


def step_base(con, st):
    """Patents, firms, inventors, citations and per-patent attributes."""
    con.execute(f"""
    CREATE OR REPLACE TABLE loc AS
      SELECT location_id, disambig_country AS ctry,
             CASE WHEN disambig_country = 'US' THEN 'US-' || coalesce(disambig_state, 'NA')
                  ELSE coalesce(disambig_country, 'NA') END AS region
      FROM '{st}/g_location_disambiguated.parquet';
    CREATE OR REPLACE TABLE firm1 AS      -- first company assignee (US or foreign company)
      SELECT patent_id, arg_min(assignee_id, assignee_sequence) AS firm,
             arg_min(location_id, assignee_sequence) AS floc,
             arg_min(assignee_type, assignee_sequence) AS ftype
      FROM '{st}/g_assignee_disambiguated.parquet' WHERE assignee_type IN (2, 3) GROUP BY 1;
    CREATE OR REPLACE TABLE asg_all AS
      SELECT DISTINCT patent_id, assignee_id AS firm
      FROM '{st}/g_assignee_disambiguated.parquet' WHERE assignee_type IN (2, 3);
    CREATE OR REPLACE TABLE P AS
      SELECT p.patent_id, p.patent_date AS gdate, year(p.patent_date) AS gyear, p.num_claims,
             CASE WHEN f.fdate BETWEEN DATE '1900-01-01' AND p.patent_date THEN f.fdate END AS fdate,
             c.cpc_section AS sec, c.cpc_class AS cls, c.cpc_subclass AS sub, n.nsub,
             fm.firm, fm.ftype, lf.region AS fregion
      FROM '{st}/g_patent.parquet' p
      LEFT JOIN (SELECT patent_id, min(try_strptime(filing_date, '%Y-%m-%d')::DATE) AS fdate
                 FROM '{st}/g_application.parquet' GROUP BY 1) f USING (patent_id)
      LEFT JOIN (SELECT patent_id, cpc_section, cpc_class, cpc_subclass
                 FROM '{st}/g_cpc_current.parquet' WHERE cpc_sequence = 0) c USING (patent_id)
      LEFT JOIN (SELECT patent_id, count(DISTINCT cpc_subclass) AS nsub
                 FROM '{st}/g_cpc_current.parquet' GROUP BY 1) n USING (patent_id)
      LEFT JOIN firm1 fm USING (patent_id)
      LEFT JOIN loc lf ON fm.floc = lf.location_id
      WHERE p.patent_type = 'utility' AND p.withdrawn = 0;
    CREATE OR REPLACE TABLE BWD AS
      SELECT patent_id, count(*) AS nbwd,
             avg(CASE WHEN citation_category = 'cited by examiner' THEN 1.0 ELSE 0.0 END) AS exshare
      FROM '{st}/g_us_patent_citation.parquet' GROUP BY 1;
    CREATE OR REPLACE TABLE CIT AS        -- citations between utility patents in P
      SELECT c.patent_id AS citing, c.citation_patent_id AS cited,
             pc.gdate AS cdate, pc.gyear AS cyear, pc.firm AS cfirm,
             pd.gdate AS ddate, pd.firm AS dfirm
      FROM '{st}/g_us_patent_citation.parquet' c
      JOIN P pc ON c.patent_id = pc.patent_id
      JOIN P pd ON c.citation_patent_id = pd.patent_id;
    CREATE OR REPLACE TABLE FWD5 AS
      SELECT cited AS patent_id, count(*) AS fwd5 FROM CIT
      WHERE cdate >= ddate AND cdate <= ddate + INTERVAL 5 YEAR GROUP BY 1;
    CREATE OR REPLACE TABLE IP AS         -- inventor x patent
      SELECT DISTINCT ON (i.inventor_id, i.patent_id)
             i.inventor_id AS inv, i.patent_id, i.gender_code AS g, l.region, l.ctry,
             P.gdate, P.gyear, P.firm, P.sec, P.num_claims
      FROM '{st}/g_inventor_disambiguated.parquet' i JOIN P USING (patent_id)
      LEFT JOIN loc l USING (location_id)
      ORDER BY i.inventor_id, i.patent_id, i.inventor_sequence;
    """)


def step_hist(con):
    """Inventor history before each patent, and team attributes per patent."""
    con.execute("""
    CREATE OR REPLACE TABLE IPH AS
      SELECT *,
        count(*) OVER (PARTITION BY inv ORDER BY gdate
                       RANGE BETWEEN UNBOUNDED PRECEDING AND INTERVAL 1 DAY PRECEDING) AS prior_n,
        min(gdate) OVER (PARTITION BY inv) AS first_date,
        lag(firm) OVER (PARTITION BY inv ORDER BY gdate, patent_id) AS prev_firm,
        lag(gdate) OVER (PARTITION BY inv ORDER BY gdate, patent_id) AS prev_date
      FROM IP;
    CREATE OR REPLACE TABLE INVFIRM AS SELECT inv, firm, min(gdate) AS d, min(gyear) AS y
      FROM IP WHERE firm IS NOT NULL GROUP BY 1, 2;
    CREATE OR REPLACE TABLE INVSEC AS SELECT inv, sec, min(gdate) AS d, min(gyear) AS y
      FROM IP WHERE sec IS NOT NULL GROUP BY 1, 2;
    -- Joins rather than correlated subqueries (DuckDB 1.5.6 returned wrong results for some
    -- correlated subqueries with inequality predicates; see the SEC build notes).
    CREATE OR REPLACE TABLE IPX AS
      SELECT a.inv, a.patent_id, a.pfirms, b.psecs FROM
        (SELECT h.inv, h.patent_id, count(f.firm) AS pfirms
         FROM IPH h LEFT JOIN INVFIRM f ON f.inv = h.inv AND f.d < h.gdate GROUP BY 1, 2) a
      JOIN
        (SELECT h.inv, h.patent_id, count(s.sec) AS psecs
         FROM IPH h LEFT JOIN INVSEC s ON s.inv = h.inv AND s.d < h.gdate GROUP BY 1, 2) b
      USING (inv, patent_id);
    CREATE OR REPLACE TABLE TEAM AS
      SELECT patent_id, count(*) AS tsize, sum(prior_n) AS sum_prior,
             avg(CASE WHEN prior_n = 0 THEN 1.0 ELSE 0.0 END) AS share_new,
             count(DISTINCT region) AS nreg,
             CASE WHEN count(DISTINCT ctry) > 1 THEN 1 ELSE 0 END AS intl,
             coalesce(avg(CASE WHEN g = 'F' THEN 1.0 WHEN g = 'M' THEN 0.0 END), 0) AS share_f,
             max(CASE WHEN ctry IS DISTINCT FROM 'US' THEN 1 ELSE 0 END) AS any_foreign
      FROM IPH GROUP BY 1;
    """)


def step_firmyear(con):
    """Cumulative firm histories by grant year, for as-of lookups at any (firm, year)."""
    con.execute("""
    CREATE OR REPLACE TABLE FSY AS
      SELECT firm, sub, gyear AS year, count(*) AS n FROM P WHERE firm IS NOT NULL AND sub IS NOT NULL GROUP BY 1, 2, 3;
    CREATE OR REPLACE TABLE FSYD AS       -- change in sum of squared subclass counts
      SELECT firm, year, n,
             power(coalesce(sum(n) OVER w, 0) + n, 2) - power(coalesce(sum(n) OVER w, 0), 2) AS dsq
      FROM FSY WINDOW w AS (PARTITION BY firm, sub ORDER BY year ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING);
    CREATE OR REPLACE TABLE FC AS         -- cum_n counts all patents; cum_nsub / cum_sq only those with a CPC subclass
      SELECT firm, year, sum(n) OVER wf AS cum_n, sum(nsub) OVER wf AS cum_nsub, sum(dsq) OVER wf AS cum_sq
      FROM (SELECT a.firm, a.year, a.n, coalesce(b.n, 0) AS nsub, coalesce(b.dsq, 0) AS dsq
            FROM (SELECT firm, gyear AS year, count(*) AS n FROM P WHERE firm IS NOT NULL GROUP BY 1, 2) a
            LEFT JOIN (SELECT firm, year, sum(n) AS n, sum(dsq) AS dsq FROM FSYD GROUP BY 1, 2) b USING (firm, year))
      WINDOW wf AS (PARTITION BY firm ORDER BY year);
    CREATE OR REPLACE TABLE FCIT AS       -- citations received by the firm's patents, by citing year
      SELECT dfirm AS firm, cyear AS year, count(*) AS n,
             sum(count(*)) OVER (PARTITION BY dfirm ORDER BY cyear) AS cum_c
      FROM CIT WHERE dfirm IS NOT NULL GROUP BY 1, 2;
    CREATE OR REPLACE TABLE FINFO AS
      SELECT firm, min(gyear) AS first_year, mode(ftype) AS ftype FROM P WHERE firm IS NOT NULL GROUP BY 1;
    CREATE OR REPLACE TABLE FINV AS       -- distinct inventors on the firm's patents granted in a year
      SELECT firm, gyear AS year, count(DISTINCT inv) AS ninv FROM IP WHERE firm IS NOT NULL GROUP BY 1, 2;
    """)


def firm_features(con, target, out):
    """Firm features before year t for every (firm, t) in table `target`."""
    con.execute(f"""
    CREATE OR REPLACE TABLE {out} AS
      SELECT t.firm, t.t,
             coalesce(a.cum_n, 0) AS stock, coalesce(a.cum_n, 0) - coalesce(b.cum_n, 0) AS recent5,
             CASE WHEN coalesce(a.cum_nsub, 0) > 0 THEN 1 - a.cum_sq / power(a.cum_nsub, 2) ELSE 0 END AS breadth,
             coalesce(c.cum_c, 0) AS fcites,
             coalesce(c.cum_c, 0) - coalesce(d.cum_c, 0) AS fcites_lag1
      FROM {target} t
      ASOF LEFT JOIN FC a ON t.firm = a.firm AND t.t - 1 >= a.year
      ASOF LEFT JOIN FC b ON t.firm = b.firm AND t.t - 6 >= b.year
      ASOF LEFT JOIN FCIT c ON t.firm = c.firm AND t.t - 1 >= c.year
      ASOF LEFT JOIN FCIT d ON t.firm = d.firm AND t.t - 2 >= d.year;
    """)


def step_ip(con, out):
    """Inventor-patent rows: development 101-105 (IDs only) and held-out 201, 202, 206."""
    con.execute("CREATE OR REPLACE TABLE IPT AS SELECT DISTINCT firm, gyear AS t FROM P WHERE firm IS NOT NULL")
    firm_features(con, "IPT", "IPF")
    con.execute(f"""
    CREATE OR REPLACE TABLE IPR AS
      SELECT h.inv, h.patent_id, h.firm, h.gyear, P.sec, P.sub,
             ln(1 + coalesce(w.fwd5, 0)) AS y,
             ln(greatest(P.num_claims, 1)) AS x01_log_claims,
             ln(1 + coalesce(b.nbwd, 0)) AS x02_log_bwd_cites,
             coalesce(b.exshare, 0) AS x03_examiner_cite_share,
             coalesce(P.nsub, 1) AS x04_n_cpc_subclasses,
             date_diff('day', P.fdate, P.gdate) / 365.25 AS x05_grant_lag,
             tm.tsize AS x06_team_size,
             CASE WHEN tm.tsize > 1 THEN (tm.sum_prior - h.prior_n) / (tm.tsize - 1) ELSE 0 END AS x07_coinv_prior_mean,
             tm.share_new AS x08_share_first_time,
             tm.nreg AS x09_n_inv_regions,
             tm.intl AS x10_intl_team,
             tm.share_f AS x11_share_female,
             ln(1 + h.prior_n) AS x12_log_inv_prior,
             date_diff('day', h.first_date, h.gdate) / 365.25 AS x13_inv_career_yrs,
             x.pfirms AS x14_inv_prior_firms,
             CASE WHEN h.prev_firm IS NOT NULL AND h.prev_date < h.gdate AND h.prev_firm <> h.firm THEN 1 ELSE 0 END AS x15_inv_moved,
             x.psecs AS x16_inv_prior_sections,
             ln(1 + f.stock) AS x17_log_firm_stock,
             ln(1 + f.recent5) AS x18_log_firm_recent5,
             f.breadth AS x19_firm_breadth,
             ln(1 + f.fcites) AS x20_log_firm_prior_cites
      FROM IPH h JOIN P USING (patent_id)
      JOIN TEAM tm USING (patent_id) JOIN IPX x ON x.inv = h.inv AND x.patent_id = h.patent_id
      LEFT JOIN BWD b ON b.patent_id = h.patent_id LEFT JOIN FWD5 w ON w.patent_id = h.patent_id
      JOIN IPF f ON f.firm = h.firm AND f.t = h.gyear
      WHERE h.firm IS NOT NULL AND h.gyear BETWEEN {Y0} AND {Y1} AND P.fdate IS NOT NULL AND P.sec IS NOT NULL;
    """)
    xs = ", ".join(c for c, in con.execute(
        "SELECT column_name FROM (DESCRIBE IPR) WHERE column_name LIKE 'x%' ORDER BY 1").fetchall())
    codes = ("dense_rank() OVER (ORDER BY inv) - 1 AS fe_inventor, dense_rank() OVER (ORDER BY firm) - 1 AS fe_firm, "
             "gyear - {y0} AS fe_year, dense_rank() OVER (ORDER BY sub, gyear) - 1 AS fe_subclass_year, "
             "dense_rank() OVER (ORDER BY firm) - 1 AS cluster").format(y0=Y0)
    dev = [(101, ("D", "E")), (102, ("A",)), (103, ("C",)), (104, DEV)]
    for wid, group in dev:
        write(con, f"SELECT {codes} FROM IPR WHERE sec IN {secs(group)}", out / "development" / f"{wid}.parquet",
              dict(id=wid, role="development", unit="inventor x patent", sections=list(group),
                   absorb=["fe_inventor", "fe_firm", "fe_year"], cluster="cluster",
                   values="synthetic y and X added by the loader", grant_years=[Y0, Y1]))
    write(con, f"SELECT {codes} FROM IPR WHERE sec IN {secs(DEV)}", out / "development" / "105.parquet",
          dict(id=105, role="development", unit="inventor x patent", sections=list(DEV),
               absorb=["fe_inventor", "fe_firm", "fe_year", "fe_subclass_year"], cluster="cluster",
               values="synthetic y and X added by the loader", grant_years=[Y0, Y1]))
    for wid, group, extra in [(201, HELD, False), (202, HELD, True), (206, DEV + HELD, False)]:
        absorb = ["fe_inventor", "fe_firm", "fe_year"] + (["fe_subclass_year"] if extra else [])
        write(con, f"SELECT y, {xs}, {codes} FROM IPR WHERE sec IN {secs(group)}", out / "heldout" / f"{wid}.parquet",
              dict(id=wid, role="heldout", unit="inventor x patent", sections=list(group), y="log(1 + 5-year forward citations)",
                   absorb=absorb, cluster="cluster", grant_years=[Y0, Y1]))


def modal_group(con, table, key):
    """Assign each `key` (inventor or firm) to the section group of most of its patents."""
    con.execute(f"""
    CREATE OR REPLACE TABLE {key}_GROUP AS
      SELECT {key}, CASE WHEN sec IN {secs(DEV)} THEN 'dev' WHEN sec IN {secs(HELD)} THEN 'held' END AS grp
      FROM (SELECT {key}, sec, row_number() OVER (PARTITION BY {key} ORDER BY count(*) DESC, sec) AS r
            FROM {table} WHERE {key} IS NOT NULL AND sec IS NOT NULL AND gyear BETWEEN {Y0} AND {Y1} GROUP BY 1, 2)
      WHERE r = 1;
    """)


def step_invyear(con, out):
    """Inventor-year event-study panels: development 108 (IDs + real event dummies), held-out 205."""
    modal_group(con, "IP", "inv")
    con.execute(f"""
    CREATE OR REPLACE TABLE IY_ELIG AS
      SELECT inv, min(gyear) AS y0, max(gyear) AS y1 FROM IP
      WHERE firm IS NOT NULL AND gyear BETWEEN {Y0} AND {Y1} GROUP BY 1 HAVING count(*) >= 3;
    CREATE OR REPLACE TABLE IYG AS         -- inventor x year grid from first to last patent
      SELECT e.inv, t FROM IY_ELIG e, range(e.y0, e.y1 + 1) r(t);
    CREATE OR REPLACE TABLE IY_PAT AS      -- per inventor-year patent activity
      SELECT inv, gyear AS year, count(*) AS npat, sum(tm.tsize) AS team_sum,
             sum(CASE WHEN tm.tsize = 1 THEN 1 ELSE 0 END) AS solo, sum(ln(greatest(IP.num_claims, 1))) AS claims_sum,
             arg_max(firm, gdate::VARCHAR || patent_id) AS last_firm
      FROM IP JOIN TEAM tm USING (patent_id) WHERE firm IS NOT NULL GROUP BY 1, 2;
    CREATE OR REPLACE TABLE IY_CUM AS
      SELECT inv, year, npat, last_firm,
             sum(npat) OVER w AS c_pat, sum(team_sum) OVER w AS c_team, sum(solo) OVER w AS c_solo,
             sum(claims_sum) OVER w AS c_claims
      FROM IY_PAT WINDOW w AS (PARTITION BY inv ORDER BY year);
    CREATE OR REPLACE TABLE IY_CIT AS      -- citations received by the inventor's patents, by citing year
      SELECT inv, year, sum(n) OVER (PARTITION BY inv ORDER BY year) AS cum_c
      FROM (SELECT IP.inv, CIT.cyear AS year, count(*) AS n FROM CIT JOIN IP ON CIT.cited = IP.patent_id GROUP BY 1, 2);
    CREATE OR REPLACE TABLE IY AS
      SELECT g.inv, g.t, cur.last_firm AS firm,
             coalesce(n0.npat, 0) AS npat_t,
             coalesce(prev.c_pat, 0) AS prior_pat,
             coalesce(prev.c_team, 0) AS prior_team, coalesce(prev.c_solo, 0) AS prior_solo,
             coalesce(prev.c_claims, 0) AS prior_claims,
             coalesce(n1.npat, 0) AS npat_l1, coalesce(n2.npat, 0) AS npat_l2,
             coalesce(ct.cum_c, 0) AS prior_cites,
             pf.pfirms, ps.psecs,
             g.t - e.y0 AS career,
             g.t - coalesce(prev.year, g.t) AS since_last
      FROM IYG g JOIN IY_ELIG e USING (inv)
      JOIN (SELECT g.inv, g.t, count(f.firm) AS pfirms FROM IYG g
            LEFT JOIN INVFIRM f ON f.inv = g.inv AND f.y < g.t GROUP BY 1, 2) pf ON pf.inv = g.inv AND pf.t = g.t
      JOIN (SELECT g.inv, g.t, count(s.sec) AS psecs FROM IYG g
            LEFT JOIN INVSEC s ON s.inv = g.inv AND s.y < g.t GROUP BY 1, 2) ps ON ps.inv = g.inv AND ps.t = g.t
      ASOF LEFT JOIN IY_CUM cur ON g.inv = cur.inv AND g.t >= cur.year
      ASOF LEFT JOIN IY_CUM prev ON g.inv = prev.inv AND g.t - 1 >= prev.year
      LEFT JOIN IY_PAT n0 ON n0.inv = g.inv AND n0.year = g.t
      LEFT JOIN IY_PAT n1 ON n1.inv = g.inv AND n1.year = g.t - 1
      LEFT JOIN IY_PAT n2 ON n2.inv = g.inv AND n2.year = g.t - 2
      ASOF LEFT JOIN IY_CIT ct ON g.inv = ct.inv AND g.t - 1 >= ct.year;
    CREATE OR REPLACE TABLE IY_MOVE AS     -- first year the current firm differs from the previous year's
      SELECT inv, min(t) AS move_year FROM (
        SELECT inv, t, firm, lag(firm) OVER (PARTITION BY inv ORDER BY t) AS pf FROM IY)
      WHERE pf IS NOT NULL AND firm <> pf GROUP BY 1;
    CREATE OR REPLACE TABLE IYT AS SELECT DISTINCT firm, t FROM IY;
    """)
    firm_features(con, "IYT", "IYF")
    dummies = []
    for k in (-5, -4, -3, -2, 0, 1, 2, 3, 4, 5):
        cond = f"rel <= {k}" if k == -5 else (f"rel >= {k}" if k == 5 else f"rel = {k}")
        dummies.append(f"CASE WHEN rel IS NOT NULL AND {cond} THEN 1 ELSE 0 END AS d_{'m' if k < 0 else 'p'}{abs(k)}")
    con.execute(f"""
    CREATE OR REPLACE TABLE IYR AS
      SELECT IY.inv, IY.t, IY.firm, gr.grp,
             ln(1 + IY.npat_t) AS y,
             {", ".join(dummies)},
             -- Career length and firm age (year minus a first year) are exact combinations of the
             -- inventor/firm and year fixed effects, so they are not used as controls.
             IY.since_last AS x01_years_since_last_patent, IY.career * IY.career AS x02_career_sq,
             ln(1 + IY.prior_pat) AS x03_log_prior_pats, ln(1 + IY.npat_l1) AS x04_log_pats_lag1,
             ln(1 + IY.npat_l2) AS x05_log_pats_lag2, ln(1 + IY.prior_team) AS x06_log_collab_volume,
             IY.psecs AS x07_prior_sections, IY.pfirms AS x08_prior_firms,
             ln(1 + IY.prior_cites) AS x09_log_prior_cites,
             CASE WHEN IY.prior_pat > 0 THEN IY.prior_team / IY.prior_pat ELSE 0 END AS x10_mean_team_prior,
             CASE WHEN IY.prior_pat > 0 THEN IY.prior_solo / IY.prior_pat ELSE 0 END AS x11_share_solo_prior,
             CASE WHEN IY.prior_pat > 0 THEN IY.prior_claims / IY.prior_pat ELSE 0 END AS x12_mean_log_claims_prior,
             ln(1 + f.stock) AS x13_log_firm_stock, ln(1 + f.recent5) AS x14_log_firm_recent5,
             f.breadth AS x15_firm_breadth, ln(1 + f.fcites) AS x16_log_firm_prior_cites,
             ln(1 + coalesce(fi.ninv, 0)) AS x17_log_firm_inventors_lag1,
             IY.t - vf.y AS x18_years_at_current_firm,
             CASE WHEN fn.ftype = 3 THEN 1 ELSE 0 END AS x19_firm_foreign,
             ln(1 + f.fcites_lag1) AS x20_log_firm_cites_lag1
      FROM (SELECT IY.*, IY.t - m.move_year AS rel FROM IY LEFT JOIN IY_MOVE m USING (inv)) IY
      JOIN inv_GROUP gr USING (inv)
      JOIN IYF f ON f.firm = IY.firm AND f.t = IY.t
      JOIN FINFO fn ON fn.firm = IY.firm
      JOIN INVFIRM vf ON vf.inv = IY.inv AND vf.firm = IY.firm
      LEFT JOIN FINV fi ON fi.firm = IY.firm AND fi.year = IY.t - 1;
    """)
    ds = ", ".join(d.split(" AS ")[-1] for d in dummies)
    xs = ", ".join(c for c, in con.execute(
        "SELECT column_name FROM (DESCRIBE IYR) WHERE column_name LIKE 'x%' ORDER BY 1").fetchall())
    codes = ("dense_rank() OVER (ORDER BY inv) - 1 AS fe_inventor, t - {y0} AS fe_year, "
             "dense_rank() OVER (ORDER BY firm) - 1 AS fe_firm, dense_rank() OVER (ORDER BY inv) - 1 AS cluster").format(y0=Y0)
    write(con, f"SELECT {ds}, {codes} FROM IYR WHERE grp = 'dev'", out / "development" / "108.parquet",
          dict(id=108, role="development", unit="inventor x year", sections=list(DEV),
               absorb=["fe_inventor", "fe_year", "fe_firm"], cluster="cluster", event_dummies=ds.split(", "),
               values="real event-time dummies; synthetic y and 20 synthetic controls added by the loader"))
    write(con, f"SELECT y, {ds}, {xs}, {codes} FROM IYR WHERE grp = 'held'", out / "heldout" / "205.parquet",
          dict(id=205, role="heldout", unit="inventor x year", sections=list(HELD), y="log(1 + patents granted in year)",
               absorb=["fe_inventor", "fe_year", "fe_firm"], cluster="cluster", event_dummies=ds.split(", ")))


def step_fy(con, out):
    """Firm x filing-year panels: development 106 (IDs only), held-out 203."""
    modal_group(con, "P", "firm")
    con.execute(f"""
    CREATE OR REPLACE TABLE FPAT AS        -- the firm's patents by filing year, with attributes
      SELECT P.firm, year(P.fdate) AS fy, P.patent_id, P.sub, P.cls, P.fregion,
             ln(greatest(P.num_claims, 1)) AS lclaims, coalesce(b.exshare, 0) AS exshare,
             tm.tsize, tm.any_foreign, tm.nreg
      FROM P JOIN TEAM tm USING (patent_id) LEFT JOIN BWD b USING (patent_id)
      WHERE P.firm IS NOT NULL AND P.fdate IS NOT NULL;
    CREATE OR REPLACE TABLE FY_ELIG AS
      SELECT firm, min(fy) AS y0, max(fy) AS y1, mode(cls) AS mcls, mode(fregion) AS mregion
      FROM FPAT WHERE fy BETWEEN {FY0} AND {FY1} GROUP BY 1 HAVING count(*) >= 5;
    CREATE OR REPLACE TABLE FYG AS SELECT e.firm, t FROM FY_ELIG e, range(e.y0, e.y1 + 1) r(t);
    CREATE OR REPLACE TABLE FY_AGG AS
      SELECT firm, fy, count(*) AS n, avg(tsize) AS team, avg(any_foreign) AS foreign_share,
             avg(lclaims) AS claims, avg(exshare) AS exshare
      FROM FPAT GROUP BY 1, 2;
    CREATE OR REPLACE TABLE FY_INV AS      -- inventors on the firm's filings by year
      SELECT DISTINCT f.firm, f.fy, i.inv, i.region FROM FPAT f JOIN IP i USING (patent_id);
    CREATE OR REPLACE TABLE FY_INV1 AS     -- first filing year of each inventor at the firm
      SELECT firm, inv, min(fy) AS fy0 FROM FY_INV GROUP BY 1, 2;
    CREATE OR REPLACE TABLE FY_INVCUM AS   -- cumulative distinct inventors who have filed for the firm
      SELECT firm, fy0, sum(count(*)) OVER (PARTITION BY firm ORDER BY fy0) AS cum_inv
      FROM FY_INV1 GROUP BY 1, 2;
    CREATE OR REPLACE TABLE FY_INVAGG AS
      SELECT v.firm, v.fy, count(DISTINCT v.inv) AS ninv, count(DISTINCT v.region) AS nreg,
             avg(CASE WHEN f1.fy0 = v.fy THEN 1.0 ELSE 0.0 END) AS share_new_inv
      FROM FY_INV v JOIN FY_INV1 f1 USING (firm, inv) GROUP BY 1, 2;
    CREATE OR REPLACE TABLE FY_HIRE AS     -- share of filings with an inventor who moved in from another firm
      SELECT f.firm, f.fy, avg(CASE WHEN m.moved > 0 THEN 1.0 ELSE 0.0 END) AS hire_share
      FROM FPAT f JOIN (SELECT patent_id, max(CASE WHEN prev_firm IS NOT NULL AND prev_date < gdate
                                                  AND prev_firm <> firm THEN 1 ELSE 0 END) AS moved
                        FROM IPH GROUP BY 1) m USING (patent_id) GROUP BY 1, 2;
    CREATE OR REPLACE TABLE FY_SELF AS     -- self-citation share of backward citations within P
      SELECT f.firm, f.fy, avg(CASE WHEN c.dfirm = f.firm THEN 1.0 ELSE 0.0 END) AS self_share
      FROM FPAT f JOIN CIT c ON c.citing = f.patent_id GROUP BY 1, 2;
    CREATE OR REPLACE TABLE FY_SUB AS      -- diversification of filings up to each year
      SELECT firm, fy AS year, sum(n) OVER (PARTITION BY firm ORDER BY fy) AS cum_n,
             sum(dsq) OVER (PARTITION BY firm ORDER BY fy) AS cum_sq
      FROM (SELECT firm, fy, sum(n) AS n, sum(dsq) AS dsq FROM (
              SELECT firm, fy, n, power(coalesce(sum(n) OVER w, 0) + n, 2) - power(coalesce(sum(n) OVER w, 0), 2) AS dsq
              FROM (SELECT firm, sub, fy, count(*) AS n FROM FPAT WHERE sub IS NOT NULL GROUP BY 1, 2, 3)
              WINDOW w AS (PARTITION BY firm, sub ORDER BY fy ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING))
            GROUP BY 1, 2);
    CREATE OR REPLACE TABLE FYR AS
      SELECT g.firm, g.t, gr.grp, e.mcls, e.mregion,
             ln(1 + coalesce(a0.n, 0)) AS y,
             ln(1 + coalesce(s.cum_n, 0)) AS x01_log_stock_lag,
             ln(1 + coalesce(a1.n, 0)) AS x02_log_pats_lag,
             ln(1 + coalesce(v.ninv, 0)) AS x03_log_inventors_lag,
             coalesce(v.share_new_inv, 0) AS x04_share_new_inventors_lag,
             coalesce(h.hire_share, 0) AS x05_share_hire_lag,
             CASE WHEN coalesce(s.cum_n, 0) > 0 THEN 1 - s.cum_sq / power(s.cum_n, 2) ELSE 0 END AS x06_diversification_lag,
             coalesce(a1.team, 0) AS x07_team_size_lag,
             coalesce(a1.foreign_share, 0) AS x08_share_foreign_lag,
             ln(1 + coalesce(fc.n, 0)) AS x09_log_cites_received_lag,
             coalesce(sf.self_share, 0) AS x10_self_cite_share_lag,
             ln(1 + coalesce(ic.cum_inv, 0)) AS x11_log_cum_inventors_lag,   -- firm age would be collinear with firm + year FE
             coalesce(v.nreg, 0) AS x12_n_inv_regions_lag,
             coalesce(a1.claims, 0) AS x13_log_claims_lag,
             coalesce(a1.exshare, 0) AS x14_examiner_share_lag,
             CASE WHEN a1.n IS NULL THEN 1 ELSE 0 END AS x15_no_patents_lag
      FROM FYG g JOIN FY_ELIG e USING (firm) JOIN firm_GROUP gr USING (firm)
      LEFT JOIN FY_AGG a0 ON a0.firm = g.firm AND a0.fy = g.t
      LEFT JOIN FY_AGG a1 ON a1.firm = g.firm AND a1.fy = g.t - 1
      LEFT JOIN FY_INVAGG v ON v.firm = g.firm AND v.fy = g.t - 1
      LEFT JOIN FY_HIRE h ON h.firm = g.firm AND h.fy = g.t - 1
      LEFT JOIN FY_SELF sf ON sf.firm = g.firm AND sf.fy = g.t - 1
      LEFT JOIN FCIT fc ON fc.firm = g.firm AND fc.year = g.t - 1
      ASOF LEFT JOIN FY_SUB s ON s.firm = g.firm AND g.t - 1 >= s.year
      ASOF LEFT JOIN FY_INVCUM ic ON ic.firm = g.firm AND g.t - 1 >= ic.fy0;
    """)
    xs = ", ".join(c for c, in con.execute(
        "SELECT column_name FROM (DESCRIBE FYR) WHERE column_name LIKE 'x%' ORDER BY 1").fetchall())
    codes = ("dense_rank() OVER (ORDER BY firm) - 1 AS fe_firm, t - {y0} AS fe_year, "
             "dense_rank() OVER (ORDER BY mcls, t) - 1 AS fe_class_year, "
             "dense_rank() OVER (ORDER BY mregion, t) - 1 AS fe_region_year, "
             "dense_rank() OVER (ORDER BY firm) - 1 AS cluster").format(y0=FY0)
    write(con, f"SELECT {codes} FROM FYR WHERE grp = 'dev'", out / "development" / "106.parquet",
          dict(id=106, role="development", unit="firm x filing year", sections=list(DEV),
               absorb=["fe_firm", "fe_year"], absorb_variant=["fe_firm", "fe_year", "fe_class_year"],
               cluster="cluster", values="synthetic y and X added by the loader", filing_years=[FY0, FY1]))
    write(con, f"SELECT y, {xs}, {codes} FROM FYR WHERE grp = 'held'", out / "heldout" / "203.parquet",
          dict(id=203, role="heldout", unit="firm x filing year", sections=list(HELD),
               y="log(1 + patents filed in year, later granted)",
               absorb=["fe_firm", "fe_year", "fe_class_year", "fe_region_year"], cluster="cluster",
               filing_years=[FY0, FY1]))


def cosine_by_year(con, sql_counts, firms, years):
    """Cosine similarity of cumulative count vectors (before each year) between firms."""
    rows = con.execute(sql_counts).fetchall()          # (firm_idx, key, year, n)
    keys = {k: i for i, k in enumerate(sorted({r[1] for r in rows}))}
    F, K = len(firms), len(keys)
    by_year = {}
    for f, k, y, n in rows:
        by_year.setdefault(y, []).append((f, keys[k], n))
    cum = np.zeros((F, K))
    out = {}
    last = min(by_year) if by_year else years[0]
    for y in range(min(last, years[0]), years[-1] + 1):
        if y in years:
            norm = np.linalg.norm(cum, axis=1, keepdims=True)
            v = np.divide(cum, norm, out=np.zeros_like(cum), where=norm > 0)
            out[y] = v @ v.T
        for f, k, n in by_year.get(y, []):
            cum[f, k] += n
    return out


def step_dyad(con, out, top=1000, years=range(2000, 2020)):
    """Citing firm x cited firm x year dyads among the top firms: development 107 (IDs), held-out 204."""
    years = list(years)
    for grp, wid in (("dev", 107), ("held", 204)):
        con.execute(f"""
        CREATE OR REPLACE TABLE TOPF AS
          SELECT firm, row_number() OVER (ORDER BY count(*) DESC, firm) - 1 AS fi
          FROM P JOIN firm_GROUP gr USING (firm)
          WHERE gr.grp = '{grp}' AND gyear BETWEEN {Y0} AND {Y1} GROUP BY 1
          QUALIFY fi < {top};
        CREATE OR REPLACE TABLE DCIT AS
          SELECT a.fi AS i, b.fi AS j, c.cyear AS t, count(*) AS n
          FROM CIT c JOIN TOPF a ON a.firm = c.cfirm JOIN TOPF b ON b.firm = c.dfirm
          WHERE a.fi <> b.fi AND c.cyear <= {years[-1]} GROUP BY 1, 2, 3;
        CREATE OR REPLACE TABLE DCUM AS SELECT i, j, t, sum(n) OVER (PARTITION BY i, j ORDER BY t) AS cum FROM DCIT;
        CREATE OR REPLACE TABLE DPAIR AS
          SELECT DISTINCT i, j FROM DCIT WHERE t BETWEEN {years[0]} AND {years[-1]};
        CREATE OR REPLACE TABLE DGRID AS SELECT i, j, t FROM DPAIR, range({years[0]}, {years[-1] + 1}) r(t);
        """)
        if wid == 107:
            write(con, f"""SELECT (i * {len(years)} + (t - {years[0]})) AS fe_citing_year,
                                  (j * {len(years)} + (t - {years[0]})) AS fe_cited_year,
                                  dense_rank() OVER (ORDER BY i, j) - 1 AS fe_pair,
                                  dense_rank() OVER (ORDER BY i, j) - 1 AS cluster FROM DGRID""",
                  out / "development" / "107.parquet",
                  dict(id=107, role="development", unit="citing firm x cited firm x year", sections=list(DEV),
                       top_firms=top, years=[years[0], years[-1]],
                       absorb=["fe_citing_year", "fe_cited_year", "fe_pair"], cluster="cluster",
                       values="synthetic y and X added by the loader"))
            continue
        firms = [f for f, in con.execute("SELECT firm FROM TOPF ORDER BY fi").fetchall()]
        tech = cosine_by_year(con, """SELECT t.fi, P.sub, P.gyear, count(*) FROM P JOIN TOPF t USING (firm)
                                      WHERE P.sub IS NOT NULL GROUP BY 1, 2, 3""", firms, years)
        geo = cosine_by_year(con, """SELECT t.fi, IP.region, IP.gyear, count(*) FROM IP JOIN TOPF t USING (firm)
                                     WHERE IP.region IS NOT NULL GROUP BY 1, 2, 3""", firms, years)
        grid = con.execute("SELECT i, j, t FROM DGRID ORDER BY i, j, t").fetchnumpy()
        i, j, t = grid["i"], grid["j"], grid["t"]
        prox_tech, prox_geo = np.zeros(i.size), np.zeros(i.size)
        for y in years:
            m = t == y
            prox_tech[m] = tech[y][i[m], j[m]]
            prox_geo[m] = geo[y][i[m], j[m]]
        import pandas as pd
        con.register("PROX", pd.DataFrame(dict(i=i, j=j, t=t, prox_tech=prox_tech, prox_geo=prox_geo)))
        con.execute(f"""
        CREATE OR REPLACE TABLE DSH AS         -- inventors who patented for both firms, by year both links exist
          SELECT a.fi AS i, b.fi AS j, greatest(fa.y, fb.y) AS y, count(*) AS n
          FROM INVFIRM fa JOIN TOPF a ON a.firm = fa.firm
          JOIN INVFIRM fb ON fb.inv = fa.inv JOIN TOPF b ON b.firm = fb.firm
          WHERE a.fi <> b.fi GROUP BY 1, 2, 3;
        CREATE OR REPLACE TABLE DCO AS         -- co-assigned patents
          SELECT a.fi AS i, b.fi AS j, P.gyear AS y, count(*) AS n
          FROM asg_all x JOIN TOPF a ON a.firm = x.firm JOIN asg_all z ON z.patent_id = x.patent_id
          JOIN TOPF b ON b.firm = z.firm JOIN P ON P.patent_id = x.patent_id
          WHERE a.fi <> b.fi GROUP BY 1, 2, 3;
        CREATE OR REPLACE TABLE DT AS SELECT DISTINCT fi AS f, t FROM TOPF, range({years[0]}, {years[-1] + 1}) r(t);
        CREATE OR REPLACE TABLE DTF AS SELECT TOPF.firm, DT.t, DT.f FROM DT JOIN TOPF ON TOPF.fi = DT.f;
        """)
        firm_features(con, "DTF", "DFF")
        con.execute(f"""
        CREATE OR REPLACE TABLE DYR AS
          SELECT g.i, g.j, g.t, ln(1 + coalesce(c0.n, 0)) AS y,
                 p.prox_tech AS x01_tech_proximity, p.prox_geo AS x02_geo_proximity,
                 ln(1 + sh.n) AS x03_log_shared_inventors,
                 ln(1 + co.n) AS x04_log_copatents,
                 -- Firm-by-year variables are absorbed by citing#year and cited#year, so the
                 -- remaining controls vary at the pair-year level.
                 ln(1 + coalesce(cu.cum, 0)) AS x05_log_cum_cites_before,
                 ln(1 + coalesce(cr.cum, 0)) AS x06_log_cum_reverse_before,
                 ln(1 + shr.n) AS x07_log_shared_inventors_recent,
                 ln(1 + fi.stock) * ln(1 + fj.stock) AS x08_stock_product,
                 ln(1 + coalesce(c1.n, 0)) AS x09_log_cites_lag, ln(1 + coalesce(r1.n, 0)) AS x10_log_reverse_cites_lag
          FROM DGRID g JOIN PROX p USING (i, j, t)
          ASOF LEFT JOIN DCUM cu ON cu.i = g.i AND cu.j = g.j AND g.t - 2 >= cu.t
          ASOF LEFT JOIN DCUM cr ON cr.i = g.j AND cr.j = g.i AND g.t - 2 >= cr.t
          JOIN (SELECT g.i, g.j, g.t, coalesce(sum(s.n), 0) AS n FROM DGRID g
                LEFT JOIN DSH s ON s.i = g.i AND s.j = g.j AND s.y BETWEEN g.t - 3 AND g.t - 1 GROUP BY 1, 2, 3) shr
            ON shr.i = g.i AND shr.j = g.j AND shr.t = g.t
          JOIN (SELECT g.i, g.j, g.t, coalesce(sum(s.n), 0) AS n FROM DGRID g
                LEFT JOIN DSH s ON s.i = g.i AND s.j = g.j AND s.y < g.t GROUP BY 1, 2, 3) sh
            ON sh.i = g.i AND sh.j = g.j AND sh.t = g.t
          JOIN (SELECT g.i, g.j, g.t, coalesce(sum(o.n), 0) AS n FROM DGRID g
                LEFT JOIN DCO o ON o.i = g.i AND o.j = g.j AND o.y < g.t GROUP BY 1, 2, 3) co
            ON co.i = g.i AND co.j = g.j AND co.t = g.t
          JOIN (SELECT DFF.*, TOPF.fi FROM DFF JOIN TOPF USING (firm)) fi ON fi.fi = g.i AND fi.t = g.t
          JOIN (SELECT DFF.*, TOPF.fi FROM DFF JOIN TOPF USING (firm)) fj ON fj.fi = g.j AND fj.t = g.t
          LEFT JOIN DCIT c0 ON c0.i = g.i AND c0.j = g.j AND c0.t = g.t
          LEFT JOIN DCIT c1 ON c1.i = g.i AND c1.j = g.j AND c1.t = g.t - 1
          LEFT JOIN DCIT r1 ON r1.i = g.j AND r1.j = g.i AND r1.t = g.t - 1;
        """)
        xs = ", ".join(c for c, in con.execute(
            "SELECT column_name FROM (DESCRIBE DYR) WHERE column_name LIKE 'x%' ORDER BY 1").fetchall())
        write(con, f"""SELECT y, {xs}, (i * {len(years)} + (t - {years[0]})) AS fe_citing_year,
                              (j * {len(years)} + (t - {years[0]})) AS fe_cited_year,
                              dense_rank() OVER (ORDER BY i, j) - 1 AS fe_pair,
                              dense_rank() OVER (ORDER BY i, j) - 1 AS cluster FROM DYR""",
              out / "heldout" / "204.parquet",
              dict(id=204, role="heldout", unit="citing firm x cited firm x year", sections=list(HELD),
                   top_firms=top, years=[years[0], years[-1]], y="log(1 + citations from citing to cited firm in year)",
                   absorb=["fe_citing_year", "fe_cited_year", "fe_pair"], cluster="cluster"))


def main(stage, out, steps):
    stage, out = Path(stage), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(out / "_work_patentsview.duckdb"))
    con.execute(f"SET memory_limit='11GB'; SET threads=10; SET temp_directory='{out / '_tmp'}';"
                "SET preserve_insertion_order=false")
    steps = steps or ["base", "hist", "firmyear", "ip", "invyear", "fy", "dyad"]
    for s in steps:
        t0 = time.time()
        print(f"step {s}", flush=True)
        if s == "base":
            step_base(con, stage)
        elif s == "hist":
            step_hist(con)
        elif s == "firmyear":
            step_firmyear(con)
        elif s == "ip":
            step_ip(con, out)
        elif s == "invyear":
            step_invyear(con, out)
        elif s == "fy":
            step_fy(con, out)
        elif s == "dyad":
            step_dyad(con, out)
        print(f"  done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
