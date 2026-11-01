"""Order-independent checksums of a .dta file, used to compare every tool's output with the reference.

Per variable: the number of non-missing values and their sum (numeric) or total length in bytes
(string). Variables with a value label are also summarized through their labels: the total length
of the label text over all rows, so codes must map to the same text. Read in chunks.

usage: python checksums.py file.dta
"""
import json
import sys

import numpy as np

import dta


def checksums(path, chunk_rows=1_000_000):
    h = dta.read_header(path)
    t = dict(zip(h["names"], h["types"]))
    vl = {n: h["value_labels"].get(l) for n, l in zip(h["names"], h["vlnames"]) if l}
    out = {"N": h["N"]}
    n = {v: 0 for v in t}
    s = {v: 0.0 for v in t}
    lab = {v: 0 for v in vl}
    lookup = {}
    for v, table in vl.items():
        if table:
            codes = np.array(sorted(table))
            lookup[v] = (codes, np.array([len(table[c].encode()) for c in codes]))
    for rows in dta.iter_rows(path, chunk_rows, header=h):
        for v, ty in t.items():
            x = rows[v]
            if ty <= dta.STR_MAX:
                L = dta.str_len(x)
                n[v] += int((L > 0).sum())
                s[v] += float(L.sum())
            else:
                ok = ~dta.is_missing(x, ty)
                n[v] += int(ok.sum())
                s[v] += float(x[ok].astype(np.float64).sum())
                if v in lookup:
                    codes, lens = lookup[v]
                    c = x[ok].astype(np.int64)
                    i = np.searchsorted(codes, c)
                    found = (i < len(codes)) & (codes[np.minimum(i, len(codes) - 1)] == c)
                    lab[v] += int(lens[i[found]].sum())
    for v in t:
        out[f"n_{v}"] = n[v]
        out[f"sum_{v}"] = s[v]
    for v in vl:
        out[f"labeltext_{v}"] = lab[v]
    return out


def compare(a, b, rtol=1e-10):
    """Keys that differ: counts and label totals must be equal, sums within rtol."""
    bad = []
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            bad.append(k)
        elif k.startswith("sum_"):
            if abs(a[k] - b[k]) > rtol * max(1.0, abs(b[k])):
                bad.append(k)
        elif a[k] != b[k]:
            bad.append(k)
    return bad


if __name__ == "__main__":
    print(json.dumps(checksums(sys.argv[1]), indent=1))
