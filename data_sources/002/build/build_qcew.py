"""Build held-out workloads 401-403 for task 002 from QCEW and state minimum wages.

Treatment: the first period in which the state's minimum wage is above the federal $7.25
(in force since July 2009). States above $7.25 already in 2010 (period 1) are always
treated and dropped; states never above $7.25 in the window are never treated.
  Minimum wages: Vaghul & Zipperer v1.4.0 (state maximum within the year or quarter,
  2010-2022) and the U.S. Department of Labor's state minimum-wage history table
  (2023-2024, the rate shown for each year, applied from that year's first quarter).

401 county x year 2010-2024: food services and drinking places (NAICS 722), private sector;
    y = log employment; balanced = disclosed and positive in all 15 years.
402 county x 3-digit NAICS x year 2010-2024, private sector; same rules.
403 county x 3-digit NAICS x quarter 2010Q1-2024Q4 (60 quarters), private sector;
    y = log average monthly employment in the quarter.
Covariates (2010 annual): log employment, log average weekly wage, log establishments of the
unit, log total private employment of the county, and three Census-region dummies (West omitted).

usage: python build_qcew.py QCEW_STAGE_DIR MINWAGE_DIR OUT_DIR
"""
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

FED = 7.25
REGION = {"NE": "09 23 25 33 44 50 34 36 42", "MW": "17 18 26 39 55 19 20 27 29 31 38 46",
          "S": "10 11 12 13 24 37 45 51 54 01 21 28 47 05 22 40 48",
          "W": "02 04 06 08 15 16 30 32 35 41 49 53 56"}
REGION = {fips: r for r, s in REGION.items() for fips in s.split()}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def money(x):
    m = re.findall(r"\$?(\d+\.\d+)", str(x))
    return max(float(v) for v in m) if m else np.nan


def min_wages(mw_dir):
    """State x year and state x quarter minimum wage (max in period), 2010-2024, with source notes."""
    mw_dir = Path(mw_dir)
    a = pd.read_excel(mw_dir / "mw_state_excel" / "mw_state_annual.xlsx")
    a = a.rename(columns={"State FIPS Code": "fips", "Year": "year", "Annual State Maximum": "mw", "Name": "name"})
    a["fips"] = a.fips.astype(int).map("{:02d}".format)
    names = dict(zip(a.name.str.lower(), a.fips))
    annual = a[(a.year >= 2010) & (a.year <= 2022)][["fips", "year", "mw"]].copy()
    q = pd.read_excel(mw_dir / "mw_state_excel" / "mw_state_quarterly.xlsx")
    qcol = [c for c in q.columns if "State Maximum" in str(c)][0]
    q = q.rename(columns={"State FIPS Code": "fips", qcol: "mw"})
    q["fips"] = q.fips.astype(int).map("{:02d}".format)
    qdate = [c for c in q.columns if str(c).lower() in ("quarterly date", "date", "quarter")]
    if qdate and not pd.api.types.is_numeric_dtype(q[qdate[0]]):
        d = pd.PeriodIndex(q[qdate[0]].astype(str).str.replace(" ", "").str.upper(), freq="Q")
        q["year"], q["qtr"] = d.year, d.quarter
    else:
        q = q.rename(columns={"Year": "year", "Quarter": "qtr"})
    quarterly = q[(q.year >= 2010) & (q.year <= 2022)][["fips", "year", "qtr", "mw"]].copy()
    # DOL table for 2023-2024
    tables = pd.read_html(mw_dir / "dol_state_minimum_wage_history.html")
    rows = []
    for t in tables:
        t.columns = [str(c) for c in t.columns]
        for _, r in t.iterrows():
            st = re.sub(r"\s*\(.*?\)|\*", "", str(r.iloc[0])).strip().lower()
            if st in names:
                for c in t.columns[1:]:
                    y = re.match(r"(\d{4})", c)
                    if y and int(y.group(1)) in (2023, 2024):
                        rows.append((names[st], int(y.group(1)), money(r[c])))
    dol = pd.DataFrame(rows, columns=["fips", "year", "mw"])
    annual = pd.concat([annual, dol], ignore_index=True)
    dolq = dol.loc[dol.index.repeat(4)].assign(qtr=np.tile([1, 2, 3, 4], len(dol)))
    quarterly = pd.concat([quarterly, dolq[["fips", "year", "qtr", "mw"]]], ignore_index=True)
    for d in (annual, quarterly):
        d["mw"] = d["mw"].fillna(FED).clip(lower=FED)
    return annual, quarterly


def treatment(mw, keys, period):
    """First period with mw > FED per state; returns dict fips -> period index (1-based)."""
    mw = mw.assign(p=period(mw)).sort_values("p")
    first = mw[mw.mw > FED + 1e-9].groupby("fips").p.min()
    return {f: int(first.get(f, 0)) for f in keys}


def write(units, Y, xcols, out, spec):
    cols = {"unit": np.arange(len(units), dtype=np.int64), "gvar": units.gvar.to_numpy(np.int64)}
    for c in xcols:
        cols[c] = units[c].to_numpy(np.float64)
    for t in range(Y.shape[1]):
        cols[f"y{t + 1:03d}"] = Y[:, t]
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), out, compression="zstd")
    g = units.gvar.to_numpy()
    cohorts, sizes = np.unique(g[g > 0], return_counts=True)
    spec = dict(spec, n_units=len(units), n_periods=Y.shape[1], n_covariates=len(xcols), covariates=xcols,
                n_cohorts=int(cohorts.size), n_never_treated=int((g == 0).sum()),
                share_treated=round(float((g > 0).mean()), 4), cohort_size_min=int(sizes.min()),
                cohort_size_median=float(np.median(sizes)), n_gt=int(cohorts.size * (Y.shape[1] - 1)),
                source="BLS QCEW single files 2010-2024; Vaghul-Zipperer v1.4.0; U.S. DOL state minimum wage history",
                file=out.name, sha256=sha256(out), built=time.strftime("%Y-%m-%d"))
    (out.parent / f"manifest_{out.stem}.json").write_text(json.dumps(spec, indent=2))
    print(f"  {out.parent.name}/{out.name}: {len(units):,} units x {Y.shape[1]}, {cohorts.size} cohorts, "
          f"{spec['share_treated']:.0%} treated", flush=True)


def main(stage, mw_dir, out):
    stage, out = Path(stage), Path(out)
    annual_mw, quarterly_mw = min_wages(mw_dir)
    con = duckdb.connect()
    con.execute("SET memory_limit='8GB'; SET threads=10")
    con.execute(f"CREATE TABLE A AS SELECT * FROM '{stage}/qcew_county_private_annual.parquet'")
    con.execute(f"CREATE TABLE Q AS SELECT * FROM '{stage}/qcew_county_private_qtrly.parquet'")
    con.execute("""
    CREATE TABLE CTOT AS SELECT area_fips, ln(emplvl) AS log_county_emp FROM A
      WHERE agglvl_code = '71' AND industry_code = '10' AND year = 2010 AND emplvl > 0 AND coalesce(disclosure_code, '') <> 'N';
    CREATE TABLE BASE AS SELECT area_fips, industry_code, ln(emplvl) AS log_emp, ln(wkly_wage) AS log_wage,
                                ln(estabs) AS log_estabs FROM A
      WHERE agglvl_code = '75' AND year = 2010 AND emplvl > 0 AND wkly_wage > 0 AND estabs > 0 AND coalesce(disclosure_code, '') <> 'N';
    """)
    xcols = ["log_emp", "log_wage", "log_estabs", "log_county_emp", "region_northeast", "region_midwest", "region_south"]
    for wid, kind, industry in ((401, "annual", "722"), (402, "annual", None), (403, "qtrly", None)):
        T = 15 if kind == "annual" else 60
        tab = "A" if kind == "annual" else "Q"
        tcol = "year - 2009" if kind == "annual" else "(year - 2010) * 4 + qtr"
        ind = f"AND industry_code = '{industry}'" if industry else ""
        bal = con.execute(f"""
          SELECT area_fips, industry_code FROM {tab}
          WHERE agglvl_code = '75' {ind} AND year BETWEEN 2010 AND 2024 AND emplvl > 0 AND coalesce(disclosure_code, '') <> 'N'
          GROUP BY 1, 2 HAVING count(DISTINCT {tcol}) = {T}""").df()
        units = bal.merge(con.execute("SELECT * FROM BASE").df(), on=["area_fips", "industry_code"]) \
                   .merge(con.execute("SELECT * FROM CTOT").df(), on="area_fips")
        units["state"] = units.area_fips.str[:2]
        units = units[units.state.isin(REGION)].copy()
        for r, c in (("NE", "region_northeast"), ("MW", "region_midwest"), ("S", "region_south")):
            units[c] = (units.state.map(REGION) == r).astype(float)
        if kind == "annual":
            g = treatment(annual_mw, units.state.unique(), lambda d: d.year - 2009)
        else:
            g = treatment(quarterly_mw, units.state.unique(), lambda d: (d.year - 2010) * 4 + d.qtr)
        units["gvar"] = units.state.map(g).fillna(0).astype(int)
        dropped_states = sorted(s for s, v in g.items() if v == 1)
        n_before = len(units)
        units = units[units.gvar != 1].sort_values(["area_fips", "industry_code"]).reset_index(drop=True)
        key = units[["area_fips", "industry_code"]].assign(row=np.arange(len(units)))
        L = con.execute(f"""SELECT area_fips, industry_code, {tcol} AS t, ln(emplvl) AS v FROM {tab}
                            WHERE agglvl_code = '75' {ind} AND year BETWEEN 2010 AND 2024 AND emplvl > 0""").df()
        L = L.merge(key, on=["area_fips", "industry_code"])
        Y = np.full((len(units), T), np.nan)
        Y[L.row.to_numpy(), L.t.to_numpy().astype(int) - 1] = L.v.to_numpy()
        assert np.isfinite(Y).all(), "panel not balanced"
        write(units, Y, xcols, out / "heldout" / f"{wid}.parquet",
              dict(id=wid, group="heldout",
                   unit={401: "county (NAICS 722)", 402: "county x 3-digit NAICS", 403: "county x 3-digit NAICS"}[wid],
                   periods=["2010", "2024"] if kind == "annual" else ["2010Q1", "2024Q4"],
                   frequency="year" if kind == "annual" else "quarter", baseline_year=2010,
                   treatment="first period with the state minimum wage above the federal $7.25",
                   y="log employment" if kind == "annual" else "log average monthly employment in the quarter",
                   dropped_always_treated_states=dropped_states, dropped_always_treated_units=int(n_before - len(units))))


if __name__ == "__main__":
    main(*sys.argv[1:4])
