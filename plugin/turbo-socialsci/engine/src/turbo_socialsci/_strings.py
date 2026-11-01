"""Text columns in the forms the linkage cores read, from Arrow's buffers (offsets and characters), so
pandas' Arrow-backed strings are read in place and Python str objects are converted once by Arrow,
without Python loops:
- byte_rows: rows of ASCII bytes (2-D uint8, each row NUL-padded to the longest value), copied;
- var_rows: the characters and offsets themselves (the "memory" variant's using file)."""
import numpy as np
import numba as nb

NON_ASCII, NUL = 1, 2


@nb.njit(cache=True, parallel=True)
def _fill_rows(off, data, out, nparts):
    """Copies value r's bytes data[off[r]:off[r + 1]] to out[r]; flags non-ASCII bytes and NULs."""
    n = out.shape[0]
    flags = np.zeros(nparts, np.int64)
    for p in nb.prange(nparts):
        f = 0
        for r in range(n * p // nparts, n * (p + 1) // nparts):
            a = off[r]
            for k in range(off[r + 1] - a):
                c = data[a + k]
                if c >= 128:
                    f |= NON_ASCII
                elif c == 0:
                    f |= NUL
                out[r, k] = c
        flags[p] = f
    res = 0
    for p in range(nparts):
        res |= flags[p]
    return res


@nb.njit(cache=True, parallel=True)
def _scan(data, lo, hi, nparts):
    """NON_ASCII | NUL as found in data[lo:hi]."""
    flags = np.zeros(nparts, np.int64)
    n = hi - lo
    for p in nb.prange(nparts):
        mx = np.uint8(0)
        zero = False
        for i in range(lo + n * p // nparts, lo + n * (p + 1) // nparts):
            c = data[i]
            mx = max(mx, c)
            zero |= c == 0
        flags[p] = (NON_ASCII if mx >= 128 else 0) | (NUL if zero else 0)
    res = 0
    for p in range(nparts):
        res |= flags[p]
    return res


def _arrow(series):
    """The Series as one Arrow string array without nulls (missing values: empty strings)."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.compute as pc
    s = series
    if pd.api.types.is_integer_dtype(s.dtype):
        s = s.astype(object).where(s.notna(), "").astype(str)
    try:
        arr = pa.array(s, type=pa.large_string(), from_pandas=True)
    except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError):
        arr = pa.array(s.astype(object).where(s.notna(), "").astype(str), type=pa.large_string(),
                       from_pandas=True)
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    if not (pa.types.is_string(arr.type) or pa.types.is_large_string(arr.type)):
        arr = arr.cast(pa.large_string())
    if arr.null_count:
        arr = pc.fill_null(arr, "")
    return arr


def _buffers(arr):
    """(offsets, characters) of an Arrow string array, as NumPy views of its buffers."""
    import pyarrow as pa
    n = len(arr)
    bufs = arr.buffers()
    otype = np.int64 if pa.types.is_large_string(arr.type) else np.int32
    off = np.frombuffer(bufs[1], dtype=otype)[arr.offset:arr.offset + n + 1] if n else np.zeros(1, otype)
    data = np.frombuffer(bufs[2], dtype=np.uint8) if (n and bufs[2] is not None and bufs[2].size) else \
        np.zeros(1, np.uint8)
    return off, data


def var_rows(series):
    """(characters, offsets, longest length, flags) of a pandas Series of strings (missing values
    count as empty strings): the characters read in place where the Series is one Arrow array,
    offsets as int32 (both read-only); None if the characters exceed 2**31 - 1 bytes."""
    arr = _arrow(series)
    off, data = _buffers(arr)
    if off[-1] >= 2**31:
        return None
    off = off.astype(np.int32)
    off.flags.writeable = False
    data = data.view()
    data.flags.writeable = False
    n = off.size - 1
    width = max(1, int(np.diff(off).max())) if n else 1
    lo, hi = int(off[0]), int(off[-1])
    flags = _scan(data, lo, hi, max(1, min(nb.get_num_threads(), (hi - lo) // 1048576 + 1))) if hi > lo else 0
    return data, off, width, int(flags)


def byte_rows(series):
    """(rows, flags) of a pandas Series of strings (missing values count as empty strings);
    flags: NON_ASCII | NUL as found."""
    arr = _arrow(series)
    n = len(arr)
    off, data = _buffers(arr)
    width = max(1, int(np.diff(off).max())) if n else 1
    out = np.zeros((n, width), np.uint8)
    flags = _fill_rows(off, data, out, max(1, min(nb.get_num_threads(), n // 65536 + 1))) if n else 0
    return out, int(flags)
