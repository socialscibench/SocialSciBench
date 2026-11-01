"""Brute-force check of the inventor-patent controls (workloads 201, 202, 206).

Samples rows from the build's IPR table and recomputes y and all 20 controls directly
from the base tables (patents, inventors, citations) with plain pandas, then reports the
largest absolute difference per variable.

usage: python validate_features.py WORK_DB [n_sample]
"""
import sys

import duckdb
import numpy as np
import pandas as pd


def main(db, n=300):
    con = duckdb.connect(db, read_only=True)
    s = con.execute(f"SELECT * FROM IPR USING SAMPLE {n} ROWS (reservoir, 7)").df()
    con.register("S_INV", pd.DataFrame(dict(v=s.inv.unique())))
    con.register("S_PAT", pd.DataFrame(dict(v=s.patent_id.unique())))
    con.register("S_FIRM", pd.DataFrame(dict(v=s.firm.unique())))
    ip_inv = con.execute("SELECT inv, patent_id, gdate, gyear, firm, sec FROM IP WHERE inv IN (SELECT v FROM S_INV)").df()
    team = con.execute("SELECT IP.* FROM IP WHERE patent_id IN (SELECT v FROM S_PAT)").df()
    con.register("S_CO", pd.DataFrame(dict(v=team.inv.unique())))
    ip_co = con.execute("SELECT inv, patent_id, gdate FROM IP WHERE inv IN (SELECT v FROM S_CO)").df()
    P = con.execute("SELECT patent_id, gdate, fdate, num_claims, nsub FROM P WHERE patent_id IN (SELECT v FROM S_PAT)").df().set_index("patent_id")
    fwd = con.execute("SELECT cited, cdate, ddate FROM CIT WHERE cited IN (SELECT v FROM S_PAT)").df()
    bwd = con.execute(f"SELECT patent_id, count(*) n, avg(CASE WHEN citation_category='cited by examiner' THEN 1.0 ELSE 0 END) ex "
                      f"FROM read_parquet('{sys.argv[3]}') WHERE patent_id IN (SELECT v FROM S_PAT) GROUP BY 1").df().set_index("patent_id") \
        if len(sys.argv) > 3 else None
    fp = con.execute("SELECT patent_id, firm, gyear, sub FROM P WHERE firm IN (SELECT v FROM S_FIRM)").df()
    fc = con.execute("SELECT dfirm AS firm, cyear FROM CIT WHERE dfirm IN (SELECT v FROM S_FIRM)").df()

    rows = []
    for r in s.itertuples():
        mine = ip_inv[ip_inv.inv == r.inv].sort_values(["gdate", "patent_id"])
        g = P.loc[r.patent_id, "gdate"]
        earlier = mine[mine.gdate < g]
        idx = list(mine.patent_id).index(r.patent_id)
        prev = mine.iloc[idx - 1] if idx > 0 else None
        moved = int(prev is not None and prev.gdate < g and pd.notna(prev.firm) and prev.firm != r.firm)
        tm = team[team.patent_id == r.patent_id]
        priors = [(ip_co[(ip_co.inv == v) & (ip_co.gdate < g)]).shape[0] for v in tm.inv]
        own = earlier.shape[0]
        k = len(priors)
        gend = tm.g[tm.g.isin(["F", "M"])]
        fr = fwd[fwd.cited == r.patent_id]
        f5 = int(((fr.cdate >= fr.ddate) & (pd.to_datetime(fr.cdate) <= pd.to_datetime(fr.ddate) + pd.DateOffset(years=5))).sum())
        fpat = fp[(fp.firm == r.firm) & (fp.gyear < r.gyear)]
        shares = fpat.dropna(subset=["sub"]).groupby("sub").size()
        breadth = 1 - ((shares / shares.sum()) ** 2).sum() if shares.sum() > 0 else 0.0
        out = dict(
            y=np.log1p(f5),
            x01_log_claims=np.log(max(P.loc[r.patent_id, "num_claims"], 1)),
            x04_n_cpc_subclasses=P.loc[r.patent_id, "nsub"],
            x05_grant_lag=(pd.Timestamp(g) - pd.Timestamp(P.loc[r.patent_id, "fdate"])).days / 365.25,
            x06_team_size=k,
            x07_coinv_prior_mean=(sum(priors) - own) / (k - 1) if k > 1 else 0.0,
            x08_share_first_time=np.mean([p == 0 for p in priors]),
            x09_n_inv_regions=tm.region.nunique(),
            x10_intl_team=int(tm.ctry.nunique() > 1),
            x11_share_female=(gend == "F").mean() if len(gend) else 0.0,
            x12_log_inv_prior=np.log1p(own),
            x13_inv_career_yrs=(pd.Timestamp(g) - pd.Timestamp(mine.gdate.min())).days / 365.25,
            x14_inv_prior_firms=earlier.firm.dropna().nunique(),
            x15_inv_moved=moved,
            x16_inv_prior_sections=earlier.sec.dropna().nunique(),
            x17_log_firm_stock=np.log1p(fpat.shape[0]),
            x18_log_firm_recent5=np.log1p(fpat[fpat.gyear >= r.gyear - 5].shape[0]),
            x19_firm_breadth=breadth,
            x20_log_firm_prior_cites=np.log1p(fc[(fc.firm == r.firm) & (fc.cyear < r.gyear)].shape[0]),
        )
        if bwd is not None:
            b = bwd.loc[r.patent_id] if r.patent_id in bwd.index else None
            out["x02_log_bwd_cites"] = np.log1p(b.n if b is not None else 0)
            out["x03_examiner_cite_share"] = b.ex if b is not None else 0.0
        rows.append({k2: abs(v - getattr(r, k2)) for k2, v in out.items()})
    diff = pd.DataFrame(rows)
    print(f"rows checked: {len(diff)}")
    print(diff.max().to_string())


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 300)
