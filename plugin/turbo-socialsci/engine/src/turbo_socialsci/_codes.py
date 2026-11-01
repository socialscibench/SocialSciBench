"""Integer codes of grouping variables (fixed effects, clusters, panel units) without hashing where
possible.

Integer-valued columns (numpy integers, floats holding integers, pandas' nullable integers) are coded
through a bitmap of the values present: one bit per value in [min, max], so the codes are the ranks
of the values (codes in increasing value order, as np.unique numbers them) and the work is linear in
the rows. The columns are read in place. Other columns, and integer ranges too wide for a bitmap,
go through pandas.factorize. Codes are int32 (-1 for missing).
"""
import numpy as np
import numba as nb
from numba import types
from numba.extending import intrinsic

MAX_BITS_PER_ROW = 64           # bitmap of at most 64 bits per row (8 bytes per row)
NO_MISS = np.zeros(0, np.bool_)


@intrinsic
def _popcount(typingctx, x):
    sig = types.uint64(types.uint64)

    def codegen(context, builder, signature, args):
        fn = builder.module.declare_intrinsic("llvm.ctpop", [args[0].type])
        return builder.call(fn, args)
    return sig, codegen


@nb.njit(cache=True, parallel=True)
def _int_range(vals, miss, nparts):
    """Smallest and largest value (miss: missing flags, or empty for none)."""
    n = vals.size
    has = miss.size > 0
    los = np.empty(nparts, np.int64)
    his = np.empty(nparts, np.int64)
    for p in nb.prange(nparts):
        lo = np.iinfo(np.int64).max
        hi = np.iinfo(np.int64).min
        for i in range(n * p // nparts, n * (p + 1) // nparts):
            if not (has and miss[i]):
                v = np.int64(vals[i])
                lo = min(lo, v)
                hi = max(hi, v)
        los[p] = lo
        his[p] = hi
    return los.min(), his.max()


@nb.njit(cache=True, parallel=True)
def _bitmap_codes(vals, miss, lo, span, nparts):
    """Ranks of vals - lo among the distinct values present (missing: -1)."""
    n = vals.size
    has = miss.size > 0
    nw = (span >> 6) + 1
    bits = np.zeros(nw, np.uint64)
    one = np.uint64(1)
    for i in range(n):                                  # one writer: no lost bits
        if not (has and miss[i]):
            v = np.int64(vals[i]) - lo
            bits[v >> 6] |= one << np.uint64(v & 63)
    rank = np.empty(nw, np.int64)
    acc = 0
    for w in range(nw):
        rank[w] = acc
        acc += np.int64(_popcount(bits[w]))
    out = np.empty(n, np.int32)
    for p in nb.prange(nparts):
        for i in range(n * p // nparts, n * (p + 1) // nparts):
            if has and miss[i]:
                out[i] = -1
            else:
                v = np.int64(vals[i]) - lo
                below = bits[v >> 6] & ((one << np.uint64(v & 63)) - one)
                out[i] = rank[v >> 6] + np.int64(_popcount(below))
    return out, acc


@nb.njit(cache=True)
def _float_ints(vals, miss):
    """Whether every non-missing float is an integer within +-2**62 (missing: NaN, marked in miss)."""
    for i in range(vals.size):
        v = vals[i]
        if v != v:
            miss[i] = True
        elif not (abs(v) < 4.6e18 and v == np.floor(v)):
            return False
    return True


def _readonly(a):
    v = np.ascontiguousarray(a).view()
    v.flags.writeable = False
    return v


def _from_ints(vals, miss):
    """Bitmap codes of an integer array (any integer dtype; miss: missing flags or None), or None
    when the range of values is too wide."""
    vals = _readonly(vals)              # read in place; read-only views: one compiled version per dtype
    miss = _readonly(NO_MISS if miss is None else np.asarray(miss, dtype=np.bool_))
    if vals.size >= 2**31 - 1:
        return None
    nparts = max(1, min(nb.get_num_threads(), vals.size // 65536 + 1))
    lo, hi = _int_range(vals, miss, nparts)
    if hi < lo:                                         # nothing present
        return np.full(vals.size, -1, np.int32), 0
    span = int(hi) - int(lo)
    if span <= MAX_BITS_PER_ROW * max(vals.size, 1024):
        out, levels = _bitmap_codes(vals, miss, np.int64(lo), np.int64(span), nparts)
        return out, int(levels)
    return None


def codes(series):
    """(codes int32 with -1 for missing, number of levels) of a pandas Series."""
    import pandas as pd
    dt = series.dtype
    vals = miss = None
    if isinstance(dt, np.dtype) and dt.kind in "iub":
        vals = series.to_numpy()
        if dt.kind == "b":
            vals = vals.view(np.uint8)
        elif dt.kind == "u" and dt.itemsize == 8 and vals.size and vals.max() >= 2**63:
            vals = None                                      # beyond int64: hashing
    elif isinstance(dt, np.dtype) and dt.kind == "f":
        f = np.array(series.to_numpy(), dtype=np.float64)    # a writable float64 copy
        miss = np.zeros(f.size, np.bool_)
        if _float_ints(f, miss):
            vals = np.where(miss, 0, f).astype(np.int64)
    elif pd.api.types.is_integer_dtype(dt) or pd.api.types.is_bool_dtype(dt):     # nullable
        miss = series.isna().to_numpy(dtype=np.bool_)
        vals = series.to_numpy(dtype=np.int64, na_value=0)
    if vals is not None:
        out = _from_ints(vals, miss)
        if out is not None:
            return out
    c, u = pd.factorize(series, use_na_sentinel=True)
    return c.astype(np.int32), len(u)


def combine(parts):
    """Codes of the interaction of several coded variables (missing if any part is missing)."""
    out, L = parts[0]
    for c, Lc in parts[1:]:
        miss = (out < 0) | (c < 0)
        combo = np.where(miss, 0, out.astype(np.int64) * max(Lc, 1) + c)
        res = _from_ints(combo, miss)
        if res is None:
            import pandas as pd
            combo[miss] = -1
            f, u = pd.factorize(combo)
            keep = u >= 0
            remap = np.cumsum(keep) - 1
            res = np.where(miss, -1, remap[f]).astype(np.int32), int(keep.sum())
        out, L = res
    return out, L


def codes_of(df, spec):
    """Codes of one variable or an interaction 'a#b#c' (missing in any part -> -1), and the levels."""
    parts = [p.strip() for p in str(spec).split("#")]
    return combine([codes(df[p]) for p in parts])
