"""Conversions between .dta columns and Arrow, for the DuckDB baseline.

to_parquet: stream .dta files into Parquet files (one per input), with Stata missing values as nulls,
%td dates as dates, and the Stata storage types and formats kept in the schema metadata.
batches_to_dta: write Arrow record batches (e.g. a DuckDB query result) into a .dta file.
"""
import json
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import dta

DATE_SHIFT = 3653                                                 # days from 1960-01-01 to 1970-01-01


def fixed_to_arrow(x):
    """str# column (NumPy S dtype) -> Arrow string array, text up to the first NUL byte."""
    w = x.dtype.itemsize
    u = np.ascontiguousarray(x).view(np.uint8).reshape(len(x), w)
    zero = u == 0
    lens = np.where(zero.any(axis=1), zero.argmax(axis=1), w)
    keep = np.arange(w)[None, :] < lens[:, None]
    offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    return pa.LargeStringArray.from_buffers(len(x), pa.py_buffer(offsets), pa.py_buffer(u[keep].tobytes()))


def arrow_to_fixed(a, w):
    """Arrow string array -> NumPy S{w} array (nulls become empty strings)."""
    if isinstance(a, pa.ChunkedArray):
        a = a.combine_chunks()
    a = a.cast(pa.large_string()).fill_null("")
    n = len(a)
    offs = np.frombuffer(a.buffers()[1], dtype=np.int64)[a.offset:a.offset + n + 1]
    data = np.frombuffer(a.buffers()[2], dtype=np.uint8) if a.buffers()[2] is not None else np.zeros(0, np.uint8)
    lens = np.minimum(np.diff(offs), w)
    out = np.zeros((n, w), dtype=np.uint8)
    rows = np.repeat(np.arange(n), lens)
    cols = np.arange(int(lens.sum())) - np.repeat(np.cumsum(lens) - lens, lens)
    out[rows, cols] = data[np.repeat(offs[:-1], lens) + cols]
    return out.view(f"S{w}").ravel()


def column_to_arrow(x, t, fmt):
    if t <= dta.STR_MAX:
        return fixed_to_arrow(x)
    miss = dta.is_missing(x, t)
    if fmt.startswith("%td"):
        return pa.array(np.where(miss, 0, x).astype(np.int32) - DATE_SHIFT, type=pa.date32(), mask=miss)
    return pa.array(x, mask=miss)


def to_parquet(paths, out_dir, chunk_rows=200_000):
    out = []
    for p in paths:
        h = dta.read_header(p)
        meta = {"stata": json.dumps(dict(types=h["types"], names=h["names"], formats=h["formats"],
                                         varlabels=h["varlabels"]))}
        target = f"{out_dir}/{h['path'].rsplit('/', 1)[-1].replace('.dta', '.parquet')}"
        writer = None
        for rows in dta.iter_rows(p, chunk_rows, header=h):
            cols = [column_to_arrow(rows[n], t, f) for n, t, f in zip(h["names"], h["types"], h["formats"])]
            tbl = pa.Table.from_arrays(cols, names=h["names"])
            if writer is None:
                writer = pq.ParquetWriter(target, tbl.schema.with_metadata(meta), compression="zstd")
            writer.write_table(tbl)
        writer.close()
        out.append(target)
    return out


def arrow_to_column(a, t, is_date=False):
    """Arrow array -> NumPy column of Stata type t, nulls as Stata missing."""
    if isinstance(a, pa.ChunkedArray):
        a = a.combine_chunks()
    if t <= dta.STR_MAX:
        return arrow_to_fixed(a, t)
    miss = a.is_null().to_numpy(zero_copy_only=False)
    if is_date or pa.types.is_date(a.type):
        v = a.cast(pa.int32()).fill_null(0).to_numpy() + DATE_SHIFT
    else:
        v = a.fill_null(0).to_numpy()
    v = v.astype(dta.NUMERIC[t])
    if miss.any():
        v[miss] = dta.missing_value(t)
    return v


def batches_to_dta(reader, out, variables, value_labels=None):
    """reader: iterable of Arrow record batches whose columns match variables (dicts with name, type)."""
    w = dta.Writer(out, variables)
    for b in reader:
        w.write({v["name"]: arrow_to_column(b.column(v["name"]), v["type"], v.get("format", "").startswith("%td"))
                 for v in variables})
    w.close(value_labels)


if __name__ == "__main__":
    print(to_parquet(sys.argv[2:], sys.argv[1]))
