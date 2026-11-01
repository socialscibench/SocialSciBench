"""Synthetic stand-in for the yearly position files (task 010_larger_than_memory).

Writes seven yearly .dta files (2020-2026) with the same 24 variables, storage types and row width
(457 bytes) as a commercial workforce dataset of job positions (one row per position a person holds),
at any total size. Rows are grouped by firm, as in the real files. Missing-value rates, numbers of
distinct values and string lengths follow coarse statistics of a 300,000-row sample of the real files;
all names are made up except "United States" and US state names. Generation is deterministic and
streams in chunks, so it needs little memory.

usage: python generate.py --gb 8 --out DIR
"""
import argparse
import datetime as dt
from pathlib import Path

import numpy as np

import dta

YEARS = list(range(2020, 2027))
YEAR_SHARE = np.array([16.10, 17.35, 18.28, 18.01, 17.96, 17.80, 16.43])    # relative file sizes (real files)
ROW_BYTES = 457
N_FIRMS = 3_216
SEED = 10_000

SCHEMA = [  # name, type, format, label
    ("seniority", dta.BYTE, "%9.0f", "Seniority level (1-7)"),
    ("rcid", dta.LONG, "%9.0f", "Company ID"),
    ("startdate", dta.INT, "%td", "Position start date"),
    ("user_id", dta.DOUBLE, "%9.0f", "Person ID"),
    ("position_id", dta.DOUBLE, "%9.0f", "Position ID"),
    ("country", 40, "%-9s", "Country"),
    ("region", 33, "%-9s", "Region"),
    ("state", 40, "%-9s", "State"),
    ("metro_area", 53, "%-9s", "Metropolitan area"),
    ("msa", 53, "%-9s", "Metropolitan statistical area"),
    ("enddate", dta.INT, "%td", "Position end date"),
    ("city", 55, "%-9s", "City"),
    ("remote_suitability", dta.DOUBLE, "%9.0f", "Remote suitability (0-1)"),
    ("weight", dta.DOUBLE, "%9.0f", "Sampling weight"),
    ("role_k1500_v2", 37, "%-9s", "Role (1,500 categories)"),
    ("start_salary", dta.DOUBLE, "%9.2f", "Salary at start date"),
    ("role_k17000_v3", 49, "%-9s", "Role (17,000 categories)"),
    ("end_salary", dta.DOUBLE, "%9.2f", "Salary at end date"),
    ("salary", dta.DOUBLE, "%9.2f", "Predicted salary"),
    ("position_number", dta.INT, "%9.0f", "Position number within person"),
    ("total_compensation", dta.DOUBLE, "%9.2f", "Total compensation"),
    ("additional_compensation", dta.DOUBLE, "%9.2f", "Additional compensation"),
    ("onet_code", 10, "%-9s", "Occupation code"),
    ("ultimate_parent_rcid", dta.LONG, "%9.0f", "Ultimate parent company ID"),
]
VARIABLES = [dict(name=n, type=t, format=f, varlabel=l) for n, t, f, l in SCHEMA]

US_STATES = ["California", "Texas", "New York", "Florida", "Illinois", "Massachusetts", "Washington", "New Jersey",
             "Pennsylvania", "Georgia", "Virginia", "North Carolina", "Ohio", "Michigan", "Colorado", "Arizona",
             "Minnesota", "Maryland", "Tennessee", "Wisconsin", "Missouri", "Indiana", "Oregon", "Connecticut",
             "Utah", "South Carolina", "Alabama", "Louisiana", "Kentucky", "Oklahoma", "Iowa", "Kansas", "Nevada",
             "Arkansas", "Nebraska", "District of Columbia", "Mississippi", "New Mexico", "Idaho", "New Hampshire",
             "Delaware", "Rhode Island", "Hawaii", "Maine", "West Virginia", "Montana", "Vermont", "South Dakota",
             "North Dakota", "Alaska", "Wyoming"]


def stata_day(y, m, d):
    return (dt.date(y, m, d) - dt.date(1960, 1, 1)).days


def _words(rng, n, lo, hi):
    vowels, cons = np.array(list("aeiou")), np.array(list("bcdfghklmnprstvwz"))
    out = set()
    while len(out) < n:
        k = rng.integers(2, 6)
        w = "".join(c + v for c, v in zip(rng.choice(cons, k), rng.choice(vowels, k)))[: rng.integers(lo, hi)]
        out.add(w.capitalize())
    return sorted(out)


def _phrases(rng, words, n, max_words, max_len):
    out = set()
    while len(out) < n:
        p = " ".join(rng.choice(words, rng.integers(1, max_words + 1)))[:max_len]
        out.add(p)
    return np.array(sorted(out), dtype=f"S{max_len}")


def _zipf(n, s):
    p = 1 / np.arange(1, n + 1) ** s
    return p / p.sum()


class Vocabulary:
    """Made-up names with the real files' numbers of distinct values and string lengths."""

    def __init__(self):
        rng = np.random.default_rng(SEED)
        words = _words(rng, 30_000, 3, 10)
        countries = ["United States"] + list(rng.choice(words, 210, replace=False))
        regions = ["Northern America"] + [f"{w} Region" for w in rng.choice(words, 15, replace=False)]
        country_region = [0] + list(rng.integers(1, 16, 210))
        # locations: (country, region, state, metro, msa, city) tuples with weights
        foreign_states = [f"{w} Province" for w in rng.choice(words, 1_300, replace=False)]
        metros = [f"{w} Metropolitan Area" for w in rng.choice(words, 830, replace=False)]
        msas = [f"{a}-{b}, {c[:2].upper()} MSA" for a, b, c in zip(rng.choice(words, 410), rng.choice(words, 410),
                                                                   rng.choice(words, 410))]
        cities = list(rng.choice(words, 11_500, replace=False))
        n_loc = 25_000
        is_us = rng.random(n_loc) < 0.60
        ctry = np.where(is_us, 0, 1 + np.minimum(rng.zipf(1.6, n_loc), 210) - 1)
        state = np.where(is_us, rng.choice(51, n_loc, p=_zipf(51, 0.9)), rng.integers(0, 1_300, n_loc))
        loc = dict(
            country=np.array([countries[c] for c in ctry]),
            region=np.array([regions[country_region[c]] for c in ctry]),
            state=np.array([US_STATES[s] if u else foreign_states[s] for s, u in zip(state, is_us)]),
            metro_area=np.array(metros)[rng.choice(830, n_loc, p=_zipf(830, 0.8))],
            msa=np.array(msas)[rng.choice(410, n_loc, p=_zipf(410, 0.8))],
            city=np.array(cities)[rng.choice(11_500, n_loc, p=_zipf(11_500, 0.9))],
        )
        no_ctry = rng.random(n_loc) < 0.013                         # missing location
        for k, share in [("state", 0.12), ("metro_area", 0.13), ("msa", 0.50), ("city", 0.20)]:
            loc[k][(rng.random(n_loc) < share) | no_ctry] = ""
        for k in ("country", "region"):
            loc[k][no_ctry] = ""
        widths = dict(country=40, region=33, state=40, metro_area=53, msa=53, city=55)
        self.loc = {k: v.astype(f"S{widths[k]}") for k, v in loc.items()}
        self.loc_p = _zipf(n_loc, 0.7)
        self.role1 = _phrases(rng, words, 1_496, 2, 37)
        self.role2 = _phrases(rng, words, 11_600, 3, 49)
        self.onet = np.array(sorted({f"{a:02d}-{b:04d}.{c:02d}" for a, b, c in
                                     zip(rng.integers(11, 54, 2000), rng.integers(1000, 9999, 2000),
                                         rng.integers(0, 4, 2000) * 10)})[:900], dtype="S10")
        self.firm_id = np.sort(rng.choice(np.arange(1, 110_000_000), N_FIRMS, replace=False)).astype(np.int32)
        self.firm_size = rng.lognormal(0, 1.5, N_FIRMS)


def write_year(path, year, n_rows, voc, chunk_rows=500_000):
    rng = np.random.default_rng([SEED, year])
    counts = rng.multinomial(n_rows, voc.firm_size / voc.firm_size.sum())    # rows per firm, in firm order
    bounds = np.concatenate([[0], np.cumsum(counts)])
    p_end_missing = 0.48 + (year - 2020) / 6 * (0.96 - 0.48)
    mid, last = stata_day(year, 7, 1), stata_day(2026, 10, 1)
    w = dta.Writer(path, VARIABLES, label=b"Synthetic positions")
    for chunk, start in enumerate(range(0, n_rows, chunk_rows)):
        m = min(chunk_rows, n_rows - start)
        r = np.random.default_rng([SEED, year, chunk])
        rows = np.zeros(m, dtype=w.dtype)
        firm = np.searchsorted(bounds, np.arange(start, start + m), side="right") - 1
        rows["rcid"] = voc.firm_id[firm]
        other = voc.firm_id[r.integers(0, N_FIRMS, m)]
        rows["ultimate_parent_rcid"] = np.where(r.random(m) < 0.92, voc.firm_id[firm], other)
        # people: runs of 1-6 consecutive positions share a user_id
        new_person = r.random(m) < 0.4
        new_person[0] = True
        pid = np.cumsum(new_person)
        rows["user_id"] = r.integers(1_000_000, 2_400_000_000, pid[-1] + 1)[pid].astype(np.float64)
        rows["position_id"] = r.integers(-9 * 10 ** 18, 9 * 10 ** 18, m).astype(np.float64)
        rows["seniority"] = r.choice(np.arange(1, 8), m, p=[.05, .25, .30, .18, .12, .07, .03])
        start_day = np.minimum(mid - np.floor(r.exponential(900, m)), stata_day(year, 12, 31)).clip(9_500)
        start_missing = r.random(m) < 0.16
        rows["startdate"] = np.where(start_missing, dta.missing_value(dta.INT), start_day).astype(np.int16)
        end_day = np.where(start_missing, mid + r.integers(-180, 180, m), start_day + np.floor(r.exponential(700, m)))
        rows["enddate"] = np.where(r.random(m) < p_end_missing, dta.missing_value(dta.INT),
                                   np.minimum(end_day, last)).astype(np.int16)
        loc = r.choice(len(voc.loc_p), m, p=voc.loc_p)
        for k, arr in voc.loc.items():
            rows[k] = arr[loc]
        rows["remote_suitability"] = np.where(r.random(m) < 0.3, 0.5, r.beta(2, 2, m))
        rows["weight"] = 1 + r.lognormal(-2.5, 1, m)
        salary = np.round(r.lognormal(np.log(85_000), 0.7, m), 2)
        rows["salary"] = salary
        rows["end_salary"] = salary
        rows["start_salary"] = np.round(salary * r.uniform(0.6, 0.95, m), 2)
        total = np.round(salary * (1 + r.lognormal(-2.2, 0.8, m)), 2)
        rows["total_compensation"] = total
        rows["additional_compensation"] = np.round(total - salary, 2)
        rows["position_number"] = np.minimum(1 + r.geometric(0.22, m), 60)
        rows["role_k1500_v2"] = voc.role1[r.choice(len(voc.role1), m, p=_zipf(len(voc.role1), 0.9))]
        rows["role_k17000_v3"] = voc.role2[r.choice(len(voc.role2), m, p=_zipf(len(voc.role2), 0.8))]
        rows["onet_code"] = voc.onet[r.choice(len(voc.onet), m, p=_zipf(len(voc.onet), 0.9))]
        w.write(rows)
    w.close()


def generate(gb, out):
    """Seven yearly files totalling about gb * 1e9 bytes of data. Returns their paths."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    n_total = int(gb * 1e9 / ROW_BYTES)
    n = np.floor(n_total * YEAR_SHARE / YEAR_SHARE.sum()).astype(int)
    voc = Vocabulary()
    paths = []
    for y, ny in zip(YEARS, n):
        p = out / f"Positions_{y}_synthetic.dta"
        if not p.exists():
            write_year(p, y, int(ny), voc)
        paths.append(p)
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gb", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for p in generate(a.gb, a.out):
        print(p)
