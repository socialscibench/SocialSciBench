"""Reference implementation for task 010_larger_than_memory.

Three data-preparation recipes on a set of yearly position files (.dta), defined by what Stata
computes (Stata's missing-value rules included):

extract    keep the US positions that started in 2015 or later, 9 variables, roles and states as codes:
               keep if country == "United States" & startdate >= td(01jan2015)
                   (Stata rule: a missing startdate is larger than any date, so those rows are kept)
               gen int year = year(startdate)
               encode role_k1500_v2, gen(role)     encode state, gen(state)
                   (codes 1, 2, ... in byte order of the distinct non-empty values in the extract;
                    "" becomes missing)
           output: user_id rcid startdate enddate year seniority salary role state
aggregate  firm-quarter panel of the same rows:
               quarter = qofd(startdate) (missing if startdate is missing)
               collapse (count) n_positions n_salary=salary (mean) mean_salary=salary
                        (sum) n_senior = (seniority >= 4 & seniority < .), by(rcid quarter)
           output: rcid quarter n_positions n_salary mean_salary n_senior, sorted by rcid quarter
append     all variables of all files, stacked (Stata's append rules for differing types)

Row order of extract and append outputs is not part of the specification (checksums are
order-independent); aggregate is sorted.

Streaming: memory use is one chunk of rows plus small tables (distinct roles and states, firm-quarter
cells), whatever the size of the input.

Deliberately simple: NumPy and the dta module. Correct, readable, not optimized.
"""
import os
from pathlib import Path

import numpy as np

import dta

US = b"United States"
FROM_DAY = 20089                                                  # td(01jan2015)
EXTRACT_VARS = ["user_id", "rcid", "startdate", "enddate", "year", "seniority", "salary", "role", "state"]


def _filter(rows, h):
    t = dict(zip(h["names"], h["types"]))
    start = rows["startdate"]
    keep_date = (start >= FROM_DAY) | dta.is_missing(start, t["startdate"])     # Stata: missing > any date
    return (rows["country"] == US) & keep_date


def _year_quarter(days, t):
    """year(d) and qofd(d) for Stata dates; missing where d is missing (returned as a mask)."""
    miss = dta.is_missing(days, t)
    d = np.where(miss, 0, days).astype("int64")
    ym = (np.datetime64("1960-01-01") + d.astype("timedelta64[D]")).astype("datetime64[M]").astype(np.int64)
    year = 1970 + ym // 12
    month = ym % 12
    return year, (year - 1960) * 4 + month // 3, miss


def extract(paths, out, workdir, chunk_rows=100_000):
    headers = [dta.read_header(p) for p in paths]
    h0 = headers[0]
    t = dict(zip(h0["names"], h0["types"]))
    # pass 1: filter, keep the needed variables (strings still as text) in a temporary file
    tmp_vars = [dict(name=v, type=t[v]) for v in ["user_id", "rcid", "startdate", "enddate", "seniority", "salary",
                                                     "role_k1500_v2", "state"]]
    tmp_path = Path(workdir) / f"extract_tmp_{os.getpid()}.dta"
    tmp = dta.Writer(tmp_path, tmp_vars)
    roles, states = set(), set()
    for p, h in zip(paths, headers):
        for rows in dta.iter_rows(p, chunk_rows, header=h):
            sel = rows[_filter(rows, h)]
            tmp.write({v["name"]: sel[v["name"]] for v in tmp_vars})
            roles.update(np.unique(sel["role_k1500_v2"]).tolist())
            states.update(np.unique(sel["state"]).tolist())
    tmp.close()
    roles = np.array(sorted(roles - {b""}), dtype=f"S{t['role_k1500_v2']}")     # byte order, as Stata sorts
    states = np.array(sorted(states - {b""}), dtype=f"S{t['state']}")
    # pass 2: codes and year
    out_vars = [dict(name="user_id", type=dta.DOUBLE, format="%12.0f"), dict(name="rcid", type=dta.LONG),
                dict(name="startdate", type=dta.INT, format="%td"), dict(name="enddate", type=dta.INT, format="%td"),
                dict(name="year", type=dta.INT), dict(name="seniority", type=dta.BYTE),
                dict(name="salary", type=dta.DOUBLE, format="%9.2f"),
                dict(name="role", type=dta.LONG, vlname="role"), dict(name="state", type=dta.LONG, vlname="state")]
    w = dta.Writer(out, out_vars)
    th = dta.read_header(tmp_path)
    for rows in dta.iter_rows(tmp_path, chunk_rows, header=th):
        year, _, miss = _year_quarter(rows["startdate"], dta.INT)
        cols = {v: rows[v] for v in ["user_id", "rcid", "startdate", "enddate", "seniority", "salary"]}
        cols["year"] = np.where(miss, dta.missing_value(dta.INT), year).astype(np.int16)
        for name, src, vocab in (("role", "role_k1500_v2", roles), ("state", "state", states)):
            x = rows[src]
            code = np.searchsorted(vocab, x) + 1
            cols[name] = np.where(x == b"", dta.missing_value(dta.LONG), code).astype(np.int32)
        w.write(cols)
    w.close({"role": {i + 1: s.decode() for i, s in enumerate(roles)},
             "state": {i + 1: s.decode() for i, s in enumerate(states)}})
    tmp_path.unlink()


def aggregate(paths, out, workdir=None, chunk_rows=100_000):
    parts = []
    for p in paths:
        h = dta.read_header(p)
        t = dict(zip(h["names"], h["types"]))
        for rows in dta.iter_rows(p, chunk_rows, header=h):
            sel = rows[_filter(rows, h)]
            _, quarter, qmiss = _year_quarter(sel["startdate"], t["startdate"])
            quarter = np.where(qmiss, 2 ** 31 - 1, quarter)                # missing quarter sorts last
            sal_ok = ~dta.is_missing(sel["salary"], t["salary"])
            sen = sel["seniority"]
            senior = (sen >= 4) & ~dta.is_missing(sen, t["seniority"])
            keys = np.stack([sel["rcid"].astype(np.int64), quarter], axis=1)
            u, inv = np.unique(keys, axis=0, return_inverse=True)
            inv = inv.ravel()
            parts.append(np.column_stack([u, np.bincount(inv, minlength=len(u)),
                                          np.bincount(inv, sal_ok, len(u)),
                                          np.bincount(inv, np.where(sal_ok, sel["salary"], 0.0), len(u)),
                                          np.bincount(inv, senior, len(u))]))
            if len(parts) >= 20:                                          # keep the partial tables small
                parts = [_combine(parts)]
    tot = _combine(parts)
    quarter = tot[:, 1].astype(np.int64)
    n_sal = tot[:, 3]
    w = dta.Writer(out, [dict(name="rcid", type=dta.LONG), dict(name="quarter", type=dta.INT, format="%tq"),
                         dict(name="n_positions", type=dta.LONG), dict(name="n_salary", type=dta.LONG),
                         dict(name="mean_salary", type=dta.DOUBLE), dict(name="n_senior", type=dta.LONG)])
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n_sal > 0, tot[:, 4] / np.where(n_sal > 0, n_sal, 1), dta.missing_value(dta.DOUBLE))
    w.write({"rcid": tot[:, 0].astype(np.int32),
             "quarter": np.where(quarter == 2 ** 31 - 1, dta.missing_value(dta.INT), quarter).astype(np.int16),
             "n_positions": tot[:, 2].astype(np.int32), "n_salary": n_sal.astype(np.int32), "mean_salary": mean,
             "n_senior": tot[:, 5].astype(np.int32)})
    w.close()


def _combine(parts):
    a = np.concatenate(parts)
    u, inv = np.unique(a[:, :2].astype(np.int64), axis=0, return_inverse=True)        # sorted by rcid, quarter
    inv = inv.ravel()
    return np.column_stack([u] + [np.bincount(inv, a[:, c], len(u)) for c in range(2, a.shape[1])])


def append(paths, out, workdir=None, chunk_rows=100_000):
    headers = [dta.read_header(p) for p in paths]
    spec = {}
    for h in headers:
        for j, n in enumerate(h["names"]):
            if n in spec:
                spec[n]["type"] = dta.promote(spec[n]["type"], h["types"][j])
            else:
                spec[n] = dict(name=n, type=h["types"][j], format=h["formats"][j], vlname=h["vlnames"][j],
                               varlabel=h["varlabels"][j])
    out_vars = list(spec.values())
    w = dta.Writer(out, out_vars, label=headers[0]["label"], characteristics=headers[0]["characteristics"])
    for p, h in zip(paths, headers):
        same = h["dtype"] == w.dtype
        for rows in dta.iter_rows(p, chunk_rows, header=h):
            if same:
                w.write(rows)                                             # identical layout: copy the rows
                continue
            t = dict(zip(h["names"], h["types"]))
            cols = {}
            for v in out_vars:
                if v["name"] in t:
                    cols[v["name"]] = dta.convert(rows[v["name"]], t[v["name"]], v["type"])
                else:
                    cols[v["name"]] = (np.full(len(rows), b"") if v["type"] <= dta.STR_MAX
                                       else np.full(len(rows), dta.missing_value(v["type"])))
            w.write(cols)
    labels = {}
    for h in headers:
        for k, v in h["value_labels"].items():
            labels.setdefault(k, v)
    w.close(labels)


RECIPES = dict(extract=extract, aggregate=aggregate, append=append)
