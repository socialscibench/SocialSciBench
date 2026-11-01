"""Python tools for task 010_larger_than_memory: the reference, pandas, and DuckDB on Parquet.

usage: python bench_python.py TOOL RECIPE OUTPUT TIMEFILE WORKDIR MEMORY_MB FILE [FILE ...]
       TOOL: reference | pandas | duckdb | parquet (convert .dta to Parquet; OUTPUT is a directory)
       MEMORY_MB: the budget (0 = none); DuckDB's memory_limit is set to 60% of it.
Writes "time <seconds>" to TIMEFILE (imports and process start-up not included).
"""
import sys
import time

import numpy as np

import dta

FROM_DATE = "2015-01-01"
FROM_DAY = 20089                                                  # td(01jan2015)


# ---------------------------------------------------------------- pandas (in memory, usual practice)
def pandas_run(recipe, files, out, work, mem_mb):
    import pandas as pd

    def read(f, cols=None):
        return pd.read_stata(f, columns=cols, chunksize=100_000, convert_dates=False, convert_categoricals=False,
                             preserve_dtypes=True)

    if recipe == "extract":
        cols = ["user_id", "rcid", "startdate", "enddate", "seniority", "salary", "role_k1500_v2", "state", "country"]
        parts = []
        for f in files:
            for c in read(f, cols):
                keep = (c["country"] == "United States") & (c["startdate"].isna() | (c["startdate"] >= FROM_DAY))
                parts.append(c.loc[keep, cols[:-1]])
        d = pd.concat(parts, ignore_index=True)
        del parts
        d.insert(4, "year", pd.to_datetime(d["startdate"], unit="D", origin="1960-01-01").dt.year)
        labels = {}
        for name, src in (("role", "role_k1500_v2"), ("state", "state")):
            vocab = sorted(set(d[src].unique()) - {""})              # code-point order = byte order of UTF-8
            codes = pd.Categorical(d[src], categories=vocab).codes.astype("float64") + 1
            codes[codes == 0] = np.nan
            d[name] = codes
            labels[name] = {i + 1: s for i, s in enumerate(vocab)}
        d = d[["user_id", "rcid", "startdate", "enddate", "year", "seniority", "salary", "role", "state"]]
        d.to_stata(out, write_index=False, version=118, value_labels=labels)
    elif recipe == "aggregate":
        cols = ["rcid", "startdate", "seniority", "salary", "country"]
        parts = []
        for f in files:
            for c in read(f, cols):
                c = c[(c["country"] == "United States") & (c["startdate"].isna() | (c["startdate"] >= FROM_DAY))]
                dt = pd.to_datetime(c["startdate"], unit="D", origin="1960-01-01")
                g = pd.DataFrame({"rcid": c["rcid"], "quarter": (dt.dt.year - 1960) * 4 + (dt.dt.month - 1) // 3,
                                  "n_positions": 1, "n_salary": c["salary"].notna().astype(int),
                                  "sum_salary": c["salary"].fillna(0),
                                  "n_senior": ((c["seniority"] >= 4) & c["seniority"].notna()).astype(int)})
                parts.append(g.groupby(["rcid", "quarter"], dropna=False).sum().reset_index())
        d = pd.concat(parts).groupby(["rcid", "quarter"], dropna=False).sum().reset_index()
        d["mean_salary"] = np.where(d["n_salary"] > 0, d["sum_salary"] / d["n_salary"].where(d["n_salary"] > 0, 1),
                                    np.nan)
        d = d.sort_values(["rcid", "quarter"], na_position="last")
        d[["rcid", "quarter", "n_positions", "n_salary", "mean_salary", "n_senior"]].to_stata(
            out, write_index=False, version=118)
    else:
        d = pd.concat([pd.read_stata(f, convert_dates=False, convert_categoricals=False, preserve_dtypes=True)
                       for f in files], ignore_index=True)
        d.to_stata(out, write_index=False, version=118)


# ---------------------------------------------------------------- DuckDB on Parquet
def duckdb_run(recipe, files, out, work, mem_mb):
    import duckdb
    import pyarrow as pa
    import arrow_io

    con = duckdb.connect()
    con.execute("SET threads = 12")
    con.execute("SET preserve_insertion_order = false")
    con.execute(f"SET temp_directory = '{work}/duckdb_tmp'")
    if mem_mb:
        con.execute(f"SET memory_limit = '{int(mem_mb * 0.6)}MB'")
    src = "read_parquet([" + ", ".join(f"'{f}'" for f in files) + "])"
    filt = f"country = 'United States' AND (startdate >= DATE '{FROM_DATE}' OR startdate IS NULL)"
    if recipe == "extract":
        con.execute(f"""CREATE TEMP VIEW f AS SELECT user_id, rcid, startdate, enddate, seniority, salary,
                        role_k1500_v2 AS role_s, state AS state_s FROM {src} WHERE {filt}""")
        labels = {}
        for name, col in (("role", "role_s"), ("state", "state_s")):
            vocab = sorted((r[0] for r in con.execute(f"SELECT DISTINCT {col} FROM f WHERE {col} <> ''").fetchall()),
                           key=lambda s: s.encode())                      # byte order, as Stata's encode
            con.register(f"{name}_codes", pa.table({col: vocab, f"{name}_code": np.arange(1, len(vocab) + 1)}))
            labels[name] = {i + 1: s for i, s in enumerate(vocab)}
        reader = con.execute("""SELECT user_id, rcid, startdate, enddate, year(startdate) AS year, seniority, salary,
                                       role_code AS role, state_code AS state
                                FROM f LEFT JOIN role_codes USING (role_s) LEFT JOIN state_codes USING (state_s)
                             """).fetch_record_batch(100_000)
        variables = [dict(name="user_id", type=dta.DOUBLE), dict(name="rcid", type=dta.LONG),
                     dict(name="startdate", type=dta.INT, format="%td"), dict(name="enddate", type=dta.INT, format="%td"),
                     dict(name="year", type=dta.INT), dict(name="seniority", type=dta.BYTE),
                     dict(name="salary", type=dta.DOUBLE), dict(name="role", type=dta.LONG, vlname="role"),
                     dict(name="state", type=dta.LONG, vlname="state")]
        arrow_io.batches_to_dta(reader, out, variables, labels)
    elif recipe == "aggregate":
        reader = con.execute(f"""
            SELECT rcid, (year(startdate) - 1960) * 4 + (month(startdate) - 1) // 3 AS quarter,
                   count(*) AS n_positions, count(salary) AS n_salary, sum(salary) / count(salary) AS mean_salary,
                   count(*) FILTER (WHERE seniority >= 4) AS n_senior
            FROM {src} WHERE {filt}
            GROUP BY ALL ORDER BY rcid, quarter NULLS LAST""").fetch_record_batch(100_000)
        variables = [dict(name="rcid", type=dta.LONG), dict(name="quarter", type=dta.INT, format="%tq"),
                     dict(name="n_positions", type=dta.LONG), dict(name="n_salary", type=dta.LONG),
                     dict(name="mean_salary", type=dta.DOUBLE), dict(name="n_senior", type=dta.LONG)]
        arrow_io.batches_to_dta(reader, out, variables)
    else:
        import json
        import pyarrow.parquet as pq
        meta = json.loads(pq.read_schema(files[0]).metadata[b"stata"])
        variables = [dict(name=n, type=t, format=f, varlabel=l) for n, t, f, l in
                     zip(meta["names"], meta["types"], meta["formats"], meta["varlabels"])]
        reader = con.execute(f"SELECT * FROM {src}").fetch_record_batch(100_000)
        arrow_io.batches_to_dta(reader, out, variables)


def main():
    tool, recipe, out, timefile, work, mem_mb, *files = sys.argv[1:]
    mem_mb = float(mem_mb)
    if tool == "duckdb":
        import duckdb  # noqa: F401  (import before timing, like the other tools' start-up)
    if tool == "pandas":
        import pandas  # noqa: F401
    t0 = time.perf_counter()
    if tool == "reference":
        import reference
        reference.RECIPES[recipe](files, out, work)
    elif tool == "pandas":
        pandas_run(recipe, files, out, work, mem_mb)
    elif tool == "duckdb":
        duckdb_run(recipe, files, out, work, mem_mb)
    elif tool == "parquet":
        import arrow_io
        arrow_io.to_parquet(files, out)
    t1 = time.perf_counter()
    with open(timefile, "w") as f:
        f.write(f"time {t1 - t0:.5f}\n")


if __name__ == "__main__":
    main()
