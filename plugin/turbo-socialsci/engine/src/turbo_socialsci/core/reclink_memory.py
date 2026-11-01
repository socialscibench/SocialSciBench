# turbo-socialsci core: the frozen SocialSciBench candidate solutions/003/block-lean (freeze of 2026-10-26)
# with the plugin's changes (plugin/README.md; diff: plugin/tools/patches/reclink_memory.diff).
"""Task 003 candidate "block-lean": the "block-index" computation (reclink2 record linkage: perfect
matches, required + or-blocking, reclink's bigram comparator with the Winkler-type prefix bonus,
float32 match and non-match weights, every tied best candidate, as computed by reference.run),
for a small peak memory.

Memory (what differs from block-index): the Numba kernels are loaded from Numba's on-disk cache,
built ahead of the evaluation by build_cache.py (data-independent compiled code), so the measured
process does not run the LLVM compiler; this module is imported under a fixed name by
solution.py because cached code refers to its module by name. Free pages of the C allocator
(for example those the input loading released) are returned to the OS at the start of the call
and after the dictionaries are built (malloc_zone_pressure_relief on macOS, malloc_trim with
glibc), since freed memory otherwise stays resident and counts in the peak RSS.

The method is block-index's:
1. Dictionaries of the master values only. Every distinct master value of every variable gets an
   id; every using value is looked up once (a 64-bit hash of its characters, verified by
   comparison) and gets the id of the equal master value or -1. For long variables a one-bit
   filter on the first eight characters rejects most using values without reading the rest.
   Perfect matches, blocking and "equal strings score 1" only compare using values with master
   values, so these ids are all that the using file contributes outside the candidates.
2. Perfect matches: master and using records whose ids agree on every variable (all duplicate
   combinations); both sides are removed from the search.
3. Blocking: candidates of a master record are the using records with its required() values and
   the same value of at least one other variable (empty master values skipped). The other
   variable with the most candidate pairs (the city in practice) is the "block" variable: the
   using records sharing a master's required and block values form a block, scored together.
   Candidates found through the other variables outside the block are "extras".
4. Scoring a block: for every scored variable except the block variable, the block's distinct
   bigrams of each using value are inverted into lists, so the number of a master value's
   bigrams (with multiplicity) occurring in each using value is a sum over the master's bigrams'
   lists. The comparator and weights then follow the reference operation by operation (float64
   arithmetic, float32 rounding wherever the reference rounds), over chunks of candidates.
5. Best candidates: the maximum score of each master record and all candidates attaining it,
   if at least minscore.

Strings must be ASCII (the reference requires it too). Threads: Numba parallel loops over using
records, blocks and master records (NUMBA_NUM_THREADS, set by the evaluator's --threads).
"""
import ctypes
import ctypes.util

import operator

import numpy as np
import numba as nb
from llvmlite import ir
from numba import types
from numba.core import cgutils
from numba.core.extending import overload, overload_attribute
from numba.experimental import structref
from numba.extending import intrinsic

PREFIX = 8                     # characters hashed by the using-value filter
FILTER_BITS = 22               # filter size (bits) for long variables
CHUNK = 128                    # candidates per vectorized scoring chunk
U64 = np.uint64


# ----------------------------------------------------------------------------------------------
# Variable-length rows (turbo-socialsci 0.2): the using file's strings read in place, from Arrow's
# characters and offsets, instead of a fixed-width copy (rows as wide as the longest value). A[r, k]
# and A.shape behave as for the 2-D arrays: characters past a value's length read as NUL.
# ----------------------------------------------------------------------------------------------

@structref.register
class VarRowsType(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)


class VarRows(structref.StructRefProxy):
    """Strings as characters (uint8 array), offsets (int32, one more than the values) and the
    longest length."""
    def __new__(cls, data, off, width):
        return structref.StructRefProxy.__new__(cls, data, off, width)

    def __len__(self):
        return _rows_shape_of(self)[0]

    @property
    def shape(self):
        return _rows_shape_of(self)


structref.define_proxy(VarRows, VarRowsType, ["data", "off", "width"])


@nb.njit(cache=True)
def _rows_shape_of(a):
    return (a.off.size - 1, a.width)


@overload(operator.getitem, inline="always")
def _rows_getitem(a, idx):
    if isinstance(a, VarRowsType) and isinstance(idx, types.BaseTuple) and len(idx) == 2:
        def impl(a, idx):
            r, k = idx
            s = a.off[r]
            if k < a.off[r + 1] - s:
                return a.data[s + k]
            return np.uint8(0)
        return impl


@overload_attribute(VarRowsType, "shape")
def _rows_shape(a):
    def get(a):
        return (a.off.size - 1, a.width)
    return get


# ----------------------------------------------------------------------------------------------
# Hashing and dictionaries
# ----------------------------------------------------------------------------------------------

@nb.njit(inline="always", cache=True)
def _fmix(h):
    h ^= h >> U64(33)
    h *= U64(0xFF51AFD7ED558CCD)
    h ^= h >> U64(33)
    h *= U64(0xC4CEB9FE1A85EC53)
    h ^= h >> U64(33)
    return h


@nb.njit(inline="always", cache=True)
def _word(A, r, k0):
    """Characters k0..k0+7 of row r of a UCS4 array packed into 64 bits (8 bits each; positions
    past the width are NUL); bit 7 of a byte set or a character above 255 marks non-ASCII."""
    w = A.shape[1]
    x = U64(0)
    hi = np.uint32(0)
    if k0 + 8 <= w:
        for t in range(8):
            c = A[r, k0 + t]
            hi |= c
            x |= U64(c & np.uint32(255)) << U64(8 * t)
    else:
        for t in range(w - k0):
            c = A[r, k0 + t]
            hi |= c
            x |= U64(c & np.uint32(255)) << U64(8 * t)
    return x, hi


@nb.njit(inline="always", cache=True)
def _hash_row(A, r):
    """Hash of row r of a UCS4 array (its characters up to the terminating NUL), 8 per step;
    second value: whether a non-ASCII character was seen."""
    h = U64(0x9E3779B97F4A7C15)
    w = A.shape[1]
    bad = False
    for k0 in range(0, w, 8):
        x, hi = _word(A, r, k0)
        if hi > 127:
            bad = True
        if x == U64(0):
            break
        h = (h ^ x) * U64(0xFF51AFD7ED558CCD)
        h ^= h >> U64(31)
    return _fmix(h), bad


@nb.njit(inline="always", cache=True)
def _prefix_hash(A, r):
    x, hi = _word(A, r, 0)
    return _fmix(x ^ U64(0xC2B2AE3D27D4EB4F))


@nb.njit(inline="always", cache=True)
def _equal_rows(A, r, Bm, s):
    """Row r of UCS4 array A equals row s of UCS4 array Bm (NUL-terminated or full width)."""
    w = max(A.shape[1], Bm.shape[1])
    for k0 in range(0, w, 8):
        xa, ha = _word(A, r, k0) if k0 < A.shape[1] else (U64(0), np.uint32(0))
        xb, hb = _word(Bm, s, k0) if k0 < Bm.shape[1] else (U64(0), np.uint32(0))
        if xa != xb:
            return False
        if xa == U64(0):
            return True
    return True


@nb.njit(cache=True)
def _build_table(Mv, plen):
    """Open-addressing table of the distinct master values (rows of Mv) and the prefix filter."""
    n = Mv.shape[0]
    size = 16
    while size < 2 * n + 2:
        size *= 2
    keys = np.zeros(size, np.uint64)
    vals = np.full(size, -1, np.int32)
    mask = U64(size - 1)
    filt = np.zeros((1 << FILTER_BITS) // 64, np.uint64)
    for v in range(n):
        h, _ = _hash_row(Mv, v)
        idx = h & mask
        while vals[idx] >= 0:
            idx = (idx + U64(1)) & mask
        keys[idx] = h
        vals[idx] = v
        if plen > 0:
            b = _prefix_hash(Mv, v) >> U64(64 - FILTER_BITS)
            filt[b >> U64(6)] |= U64(1) << (b & U64(63))
    return keys, vals, filt


@nb.njit(parallel=True, cache=True)
def _lookup(U, Mv, keys, vals, filt, plen, out, bad):
    """Id of the equal master value for every using value (row of U), or -1."""
    n = U.shape[0]
    mask = U64(keys.size - 1)
    nparts = min(n, 1024)
    for part in nb.prange(nparts):
        lo = n * part // nparts
        hi = n * (part + 1) // nparts
        for j in range(lo, hi):
            if plen > 0:
                b = _prefix_hash(U, j) >> U64(64 - FILTER_BITS)
                if (filt[b >> U64(6)] >> (b & U64(63))) & U64(1) == U64(0):
                    out[j] = -1
                    continue
            h, nonascii = _hash_row(U, j)
            if nonascii:
                bad[0] = 1
            idx = h & mask
            res = -1
            while vals[idx] >= 0:
                if keys[idx] == h and _equal_rows(U, j, Mv, vals[idx]):
                    res = vals[idx]
                    break
                idx = (idx + U64(1)) & mask
            out[j] = res


@nb.njit(cache=True)
def _key_table(a, b, nb_):
    """Ids of distinct pairs (a[i], b[i]) (both >= 0) with a lookup table; -1 where b < 0."""
    n = a.size
    size = 16
    while size < 2 * n + 2:
        size *= 2
    keys = np.full(size, -1, np.int64)
    vals = np.full(size, -1, np.int32)
    mask = size - 1
    ids = np.full(n, -1, np.int32)
    nk = 0
    for i in range(n):
        if a[i] < 0 or b[i] < 0:
            continue
        key = np.int64(a[i]) * nb_ + b[i]
        idx = np.int64(_fmix(U64(key)) & U64(mask))
        while vals[idx] >= 0 and keys[idx] != key:
            idx = (idx + 1) & mask
        if vals[idx] < 0:
            keys[idx] = key
            vals[idx] = nk
            nk += 1
        ids[i] = vals[idx]
    return keys, vals, ids, nk


@nb.njit(parallel=True, cache=True)
def _key_lookup(keys, vals, a, b, nb_, out):
    n = a.size
    mask = keys.size - 1
    for j in nb.prange(n):
        out[j] = -1
        if a[j] < 0 or b[j] < 0:
            continue
        key = np.int64(a[j]) * nb_ + b[j]
        idx = np.int64(_fmix(U64(key)) & U64(mask))
        while vals[idx] >= 0:
            if keys[idx] == key:
                out[j] = vals[idx]
                break
            idx = (idx + 1) & mask


@nb.njit(cache=True)
def _group(key, nkeys, skip):
    """CSR lists of the indices i with key[i] >= 0 (and not skip[i]), by key."""
    cnt = np.zeros(nkeys + 1, np.int64)
    for i in range(key.size):
        if key[i] >= 0 and not skip[i]:
            cnt[key[i] + 1] += 1
    for k in range(nkeys):
        cnt[k + 1] += cnt[k]
    out = np.empty(cnt[nkeys], np.int32)         # record numbers (turbo-socialsci 0.2: 32 bits)
    pos = cnt[:-1].copy()
    for i in range(key.size):
        if key[i] >= 0 and not skip[i]:
            out[pos[key[i]]] = i
            pos[key[i]] += 1
    return cnt, out


# ----------------------------------------------------------------------------------------------
# The comparator (reclink's bigram1) and the weights, as in the reference
# ----------------------------------------------------------------------------------------------

@nb.njit(inline="always", cache=True)
def _f32(x):
    return np.float64(np.float32(x))


@intrinsic
def _bits(typingctx, x):
    def codegen(context, builder, sig, args):
        return builder.bitcast(args[0], context.get_value_type(types.int64))
    return types.int64(types.float64), codegen


MEMO = 4096
MARGIN = 1e-5                 # candidates this close to the approximate best are rescored exactly


@intrinsic
def _f32_bits(typingctx, x):
    def codegen(context, builder, sig, args):
        return builder.bitcast(args[0], context.get_value_type(types.int32))
    return types.int32(types.float32), codegen


@intrinsic
def _bits_f32(typingctx, x):
    def codegen(context, builder, sig, args):
        return builder.bitcast(args[0], context.get_value_type(types.float32))
    return types.float32(types.int32), codegen


@nb.njit(inline="always", error_model="numpy", cache=True)
def _cbrt(x):
    """x^(1/3) for 2e-8 <= x <= 1, branch-free (vectorizes): a float32 bit estimate (FreeBSD
    cbrtf's constant) and three Halley steps (relative error near double rounding)."""
    i = _f32_bits(np.float32(x))
    y = np.float64(_bits_f32(np.int32(i // np.int32(3) + np.int32(709958130))))
    for _ in range(3):
        y3 = y * y * y
        y = y * (y3 + 2.0 * x) / (2.0 * y3 + x)
    return y
PREFETCH = 8                  # rows ahead when reading a block's scattered using records
DIRECT_MAX = 3                # blocks with fewer master records are counted without inverted lists


def _prefetch_at(context, builder, aryty, ary_value, index):
    ary = context.make_array(aryty)(context, builder, ary_value)
    ptr = cgutils.get_item_pointer(context, builder, aryty, ary, index)
    i8p = ir.IntType(8).as_pointer()
    fnty = ir.FunctionType(ir.VoidType(), [i8p, ir.IntType(32), ir.IntType(32), ir.IntType(32)])
    fn = cgutils.get_or_insert_function(builder.module, fnty, "llvm.prefetch.p0i8")
    i32 = ir.IntType(32)
    builder.call(fn, [builder.bitcast(ptr, i8p), ir.Constant(i32, 0), ir.Constant(i32, 3), ir.Constant(i32, 1)])
    return context.get_dummy_value()


@intrinsic
def _prefetch_2d(typingctx, arr, row):
    """Prefetch the start of row `row` of a C-contiguous 2-D array (llvm.prefetch)."""
    def codegen(context, builder, sig, args):
        return _prefetch_at(context, builder, sig.args[0], args[0], [args[1], context.get_constant(types.intp, 0)])
    return types.void(arr, types.intp), codegen


@intrinsic
def _prefetch_1d(typingctx, arr, i):
    """Prefetch element i of a 1-D array (llvm.prefetch)."""
    def codegen(context, builder, sig, args):
        return _prefetch_at(context, builder, sig.args[0], args[0], [args[1]])
    return types.void(arr, types.intp), codegen


def _prefetch_row(arr, row):
    """Prefetch the start of a row (2-D array or variable-length rows)."""


@overload(_prefetch_row, inline="always")
def _prefetch_row_impl(arr, row):
    if isinstance(arr, types.Array):
        return lambda arr, row: _prefetch_2d(arr, row)
    if isinstance(arr, VarRowsType):
        return lambda arr, row: _prefetch_1d(arr.data, arr.off[row])


@nb.njit(inline="always", cache=True)
def _bonus(b, memo_key, memo_val, minbigram, biscale, third):
    """biscale (b - minbigram)^(1/3), memoized by the value of b (b takes few distinct values)."""
    k = _bits(b)
    h = np.int64((U64(k) * U64(0x9E3779B97F4A7C15)) >> U64(52))      # MEMO = 2^12 slots
    if memo_key[h] != k:
        memo_key[h] = k
        memo_val[h] = biscale * (b - minbigram) ** third
    return memo_val[h]


@nb.njit(inline="always", cache=True)
def _wink(lm, ls, cp):
    w = 0.0
    for i in range(1, 5):
        a = min(i, ls)
        if a == min(i, lm) and a <= cp:
            w += 1.0
    return w


@nb.njit(cache=True)
def _contains(A, r, la, Bm, s, lb):
    """Row s of Bm (length lb) occurs in row r of A (length la)."""
    if lb == 0:
        return True
    for st in range(la - lb + 1):
        ok = True
        for k in range(lb):
            if A[r, st + k] != Bm[s, k]:
                ok = False
                break
        if ok:
            return True
    return False


@nb.njit(inline="always", cache=True)
def _bigram_value(count, lm, ls, cp, eq, contained):
    """reclink's bigram1 for one pair, with the reference's rounding."""
    g = 0.0
    if lm > 2:
        g = _f32(count * 2.0 / (ls + lm - 2))
    if lm < 3 or ls < 3:
        if contained:
            g = _f32(min(lm, ls) / (lm + ls - 1)) if lm + ls > 1 else 0.0
        else:
            g = 0.0
    g = _f32(g + _wink(lm, ls, cp) * (1.0 - g) / 10.0)
    if eq:
        g = 1.0
    if ls == 0:
        g = 0.0
    return g


@nb.njit(inline="always", cache=True)
def _accumulate(M, NM, bi, wm, wnm, m_empty, u_empty, minbigram, lo_pen, biscale, third):
    """One scored variable's contribution to M and NM (reference.run's update order)."""
    if not m_empty:
        bonus = biscale * (bi - minbigram) ** third if bi > minbigram else 0.0
        M = _f32(M + wm * (0.5 * bi ** 2 + bonus))
    if not (m_empty or u_empty):
        if bi < lo_pen:
            pen = 1.0
        elif bi < minbigram:
            pen = 1.0 - bi ** 2
        else:
            pen = (1.0 - bi) ** 2
        NM = _f32(NM + wnm * pen)
    if m_empty != u_empty:
        NM = _f32(NM + wnm * 0.3)
    return M, NM


# ----------------------------------------------------------------------------------------------
# Value encodings: lengths, packed first four characters, distinct bigram codes
# ----------------------------------------------------------------------------------------------

@nb.njit(inline="always", cache=True)
def _length(A, r):
    w = A.shape[1]
    for k in range(w):
        if A[r, k] == 0:
            return k
    return w


@nb.njit(inline="always", cache=True)
def _prefix4(A, r, ln):
    p = np.uint32(0)
    for k in range(min(4, ln)):
        p |= np.uint32(A[r, k]) << np.uint32(8 * k)
    return p


@nb.njit(inline="always", cache=True)
def _common_prefix(pm, ps, lm):
    """Number of leading characters (at most 4, master characters non-empty) equal in both."""
    x = pm ^ ps
    cp = 0
    for k in range(4):
        if (x >> np.uint32(8 * k)) & np.uint32(255) != 0:
            break
        cp += 1
    return min(cp, lm)


@nb.njit(cache=True)
def _invert(codes_off, codes, n, ucodes, lst_off, lst, hist):
    """Inverted lists: for each distinct code (sorted, in ucodes) the block positions holding it."""
    m = codes_off[n]
    for i in range(m):
        hist[codes[i]] += 1
    lst_off[0] = 0
    nu = 0
    for c in range(hist.size):
        if hist[c] > 0:
            ucodes[nu] = c
            lst_off[nu + 1] = lst_off[nu] + hist[c]
            hist[c] = lst_off[nu]            # running insert position
            nu += 1
    for r in range(n):
        for i in range(codes_off[r], codes_off[r + 1]):
            c = codes[i]
            lst[hist[c]] = r
            hist[c] += 1
    for i in range(nu):
        hist[ucodes[i]] = 0
    return nu



@nb.njit(cache=True)
def _encode_block(Uf, uidf, f_uid, brecs, s0, n, lens, pre, bid, stamp, tag0):
    """Lengths, packed prefixes and value ids of the block's using records (rows prefetched)."""
    tot = 0
    for r in range(n):
        if r + PREFETCH < n:
            _prefetch_row(Uf, brecs[s0 + r + PREFETCH])
        j = brecs[s0 + r]
        ln = _length(Uf, j)
        lens[r] = ln
        pre[r] = _prefix4(Uf, j, ln)
        bid[r] = uidf[j]
        if ln > 1:
            tot += ln - 1
    return tot


@nb.njit(cache=True)
def _direct_counts(Uf, uidf, brecs, s0, n, lens, pre, bid, table, cnt, stamp, tag0, encode):
    """For blocks with few master records: per using record, the number of the master value's
    bigrams (table, with multiplicity) occurring in it, read directly from its characters; with
    encode, also its length, packed prefix and value id."""
    for r in range(n):
        if r + PREFETCH < n:
            _prefetch_row(Uf, brecs[s0 + r + PREFETCH])
        j = brecs[s0 + r]
        if encode:
            ln = _length(Uf, j)
            lens[r] = ln
            pre[r] = _prefix4(Uf, j, ln)
            bid[r] = uidf[j]
        else:
            ln = lens[r]
        t = tag0 + r
        c_ = 0
        for k in range(ln - 1):
            c = (np.int64(Uf[j, k]) << 7) | np.int64(Uf[j, k + 1])
            if c < (1 << 14) and stamp[c] != t:
                stamp[c] = t
                c_ += table[c]
        cnt[r] = c_


@nb.njit(cache=True)
def _block_codes(Uf, brecs, s0, n, lens, codes, loc, stamp, tag0):
    """Distinct bigram codes of each record (CSR: loc), from the block's using records."""
    pos = 0
    bad = False
    for r in range(n):
        if r + PREFETCH < n:
            _prefetch_row(Uf, brecs[s0 + r + PREFETCH])
        j = brecs[s0 + r]
        loc[r] = pos
        t = tag0 + r
        for k in range(lens[r] - 1):
            c = (np.int64(Uf[j, k]) << 7) | np.int64(Uf[j, k + 1])
            if c >= (1 << 14):
                bad = True
                continue
            if stamp[c] != t:
                stamp[c] = t
                codes[pos] = c
                pos += 1
    loc[n] = pos
    return pos, bad


@nb.njit(inline="always", cache=True)
def _find(ucodes, lo, hi, c):
    """Position of code c in the sorted ucodes[lo:hi], or -1."""
    while lo < hi:
        mid = (lo + hi) >> 1
        if ucodes[mid] < c:
            lo = mid + 1
        else:
            hi = mid
    return lo


@nb.njit(cache=True)
def _master_bigrams(Mv, v, ln, table, sign):
    """Add (sign=1) or remove (sign=-1) the bigram counts of master value v to table."""
    for k in range(ln - 1):
        c = (np.int64(Mv[v, k]) << 7) | np.int64(Mv[v, k + 1])
        if c < (1 << 14):
            table[c] += sign


@nb.njit(cache=True)
def _direct_count(U, j, ls, table, stamp, tag):
    """Number of bigrams of the master value (counted in table, with multiplicity) that occur in
    using value j: the sum of table over j's distinct bigrams."""
    cnt = 0
    for k in range(ls - 1):
        c = (np.int64(U[j, k]) << 7) | np.int64(U[j, k + 1])
        if c < (1 << 14) and stamp[c] != tag:
            stamp[c] = tag
            cnt += table[c]
    return cnt


@nb.njit(error_model="numpy", cache=True)
def _exact_block_score(i, r, Utup, Mtup, mid, mlen, mpre, kind, fidx, wm, wnm, brecs, s0,
                       counts, lens, pre, bid, minbigram, lo_pen, biscale, third):
    """Exact score (reference operations, libm pow for the bonus) of block record r for master
    record i, from the block's encodings and bigram counts."""
    nv = mid.shape[0]
    M = 0.0
    NM = 0.0
    for f in range(nv):
        vm = mid[f, i]
        lm = mlen[f, vm]
        if kind[f] == 0:
            if lm > 0:
                M = _f32(M + wm[f])
            continue
        if kind[f] == 1:
            M, NM = _accumulate(M, NM, 1.0, wm[f], wnm[f], False, False, minbigram, lo_pen, biscale, third)
            continue
        fi = fidx[f]
        ls = lens[fi, r]
        cp = _common_prefix(mpre[f, vm], pre[fi, r], lm)
        cont = False
        if lm < 3 or ls < 3:
            j = brecs[s0 + r]
            cont = (_contains(Mtup[f], vm, lm, Utup[f], j, ls) if lm >= ls
                    else _contains(Utup[f], j, ls, Mtup[f], vm, lm))
        b_ = _bigram_value(counts[fi, r] if lm > 2 else 0, lm, ls, cp, bid[fi, r] == vm, cont)
        M, NM = _accumulate(M, NM, b_, wm[f], wnm[f], lm == 0, ls == 0, minbigram, lo_pen, biscale, third)
    return _f32(M / (M + NM))


@nb.njit(error_model="numpy", cache=True)
def _task(task, Utup, Mtup, mid, uid, mlen, mpre, kind, fidx, idxf, wm, wnm,
          order, task_off, mblock, ubk, bcnt, brecs,
          mkey, kcnt, krecs, koff, ortype,
          minscore, minbigram, lo_pen, biscale, third, maxn, cap, out_i, out_j, out_s, out_n, bad):
    """Best candidates of the master records order[task_off[task]:task_off[task + 1]] (ordered
    by block). A block is encoded when its first master record arrives: for every indexed
    variable the lengths, packed prefixes and value ids of its using records and the inverted
    lists of their distinct bigram codes."""
    nv = mid.shape[0]
    nidx = idxf.size
    ni = max(nidx, 1)
    counts = np.zeros((ni, maxn), np.int32)
    sc = np.empty(maxn, np.float64)
    memo_key = np.full(MEMO, -1, np.int64)
    memo_val = np.zeros(MEMO)
    lens = np.zeros((ni, maxn), np.int32)
    pre = np.zeros((ni, maxn), np.uint32)
    bid = np.zeros((ni, maxn), np.int32)
    loc = np.zeros(maxn + 1, np.int64)
    ucodes = np.zeros((ni, 1 << 14), np.int64)
    lst_off = np.zeros((ni, (1 << 14) + 1), np.int64)
    nuc = np.zeros(ni, np.int64)
    tots = np.zeros(ni, np.int64)
    capl = 1024
    lst = np.zeros((ni, capl), np.int32)
    codes = np.zeros(capl, np.uint16)                # bigram codes (< 2^14; turbo-socialsci 0.2: 16 bits)
    hist = np.zeros(1 << 14, np.int64)
    bstamp = np.full(1 << 14, -1, np.int64)
    btag = 0
    cmap = np.full((ni, 1 << 14), -1, np.int64)      # code -> list of the current block
    cur_bk = -1
    direct = False
    tables = np.zeros((nv, 1 << 14), np.int32)
    M = np.empty(CHUNK)
    NM = np.empty(CHUNK)
    bi = np.empty(CHUNK)
    bonus = np.empty(CHUNK)
    stamp = np.full(1 << 14, -1, np.int64)
    ex = np.empty(64, np.int64)
    exsc = np.empty(64, np.float64)
    near = np.empty(64, np.int64)
    nsc = np.empty(64, np.float64)
    tag = 0
    nout = 0
    for oi in range(task_off[task], task_off[task + 1]):
        i = order[oi]
        bk = mblock[i]
        best = -1.0
        n = 0
        s0 = 0
        nnear = 0
        if bk >= 0:
            s0 = bcnt[bk]
            n = bcnt[bk + 1] - s0
            first = bk != cur_bk
            if first:
                group = 0                         # master records of this block in the task
                for o2 in range(oi, task_off[task + 1]):
                    if mblock[order[o2]] != bk:
                        break
                    group += 1
                direct = group < DIRECT_MAX
            if direct:
                # few master records: count each master's bigrams directly in the records
                for fi in range(nidx):
                    f = idxf[fi]
                    vm = mid[f, i]
                    lm = mlen[f, vm]
                    if first or lm >= 3:
                        _master_bigrams(Mtup[f], vm, lm, tables[f], 1)
                        _direct_counts(Utup[f], uid[f], brecs, s0, n, lens[fi], pre[fi], bid[fi],
                                       tables[f], counts[fi], bstamp, btag, first)
                        btag += n
                        _master_bigrams(Mtup[f], vm, lm, tables[f], -1)
                if first:
                    for fi in range(nidx):
                        for q in range(nuc[fi]):
                            cmap[fi, ucodes[fi, q]] = -1
                        nuc[fi] = 0
                    cur_bk = bk
            elif bk != cur_bk:                    # encode the block, invert its bigram codes
                for fi in range(nidx):
                    f = idxf[fi]
                    tots[fi] = _encode_block(Utup[f], uid[f], 0, brecs, s0, n, lens[fi], pre[fi],
                                             bid[fi], bstamp, 0)
                tmax = 0
                for fi in range(nidx):
                    tmax = max(tmax, tots[fi])
                if tmax > capl:
                    capl = tmax + tmax // 4          # turbo-socialsci 0.2: 1.25x, not 2x
                    lst = np.zeros((ni, capl), np.int32)
                    codes = np.zeros(capl, np.uint16)
                for fi in range(nidx):
                    f = idxf[fi]
                    for q in range(nuc[fi]):
                        cmap[fi, ucodes[fi, q]] = -1
                    npos, bad_ = _block_codes(Utup[f], brecs, s0, n, lens[fi], codes, loc, bstamp, btag)
                    if bad_:
                        bad[0] = 1
                    btag += n
                    nuc[fi] = _invert(loc, codes[:npos], n, ucodes[fi], lst_off[fi], lst[fi], hist)
                    for q in range(nuc[fi]):
                        cmap[fi, ucodes[fi, q]] = q
                cur_bk = bk
            # bigram counts of the block's values through the inverted lists
            for f in range(nv):
                if kind[f] != 2 or direct:
                    continue
                fi = fidx[f]
                vm = mid[f, i]
                lm = mlen[f, vm]
                if lm < 3:
                    continue
                cnt = counts[fi]
                for r in range(n):
                    cnt[r] = 0
                Mv = Mtup[f]
                for k in range(lm - 1):
                    c = (np.int64(Mv[vm, k]) << 7) | np.int64(Mv[vm, k + 1])
                    if c >= (1 << 14):
                        continue
                    q = cmap[fi, c]
                    if q >= 0:
                        for t in range(lst_off[fi, q], lst_off[fi, q + 1]):
                            cnt[lst[fi, t]] += 1
            # scores of the block's records, in chunks
            for c0 in range(0, n, CHUNK):
                m = min(CHUNK, n - c0)
                p0 = s0 + c0
                for q in range(m):
                    M[q] = 0.0
                    NM[q] = 0.0
                for f in range(nv):
                    vm = mid[f, i]
                    lm = mlen[f, vm]
                    w = wm[f]
                    wn = wnm[f]
                    if kind[f] == 0:
                        if lm > 0:
                            for q in range(m):
                                M[q] = _f32(M[q] + w)
                        continue
                    if kind[f] == 1:
                        # equal, non-empty values: bigram 1, bonus constant, penalty 0
                        add = w * (0.5 * (1.0 * 1.0) + biscale * (1.0 - minbigram) ** third)
                        pen0 = wn * ((1.0 - 1.0) * (1.0 - 1.0))
                        for q in range(m):
                            M[q] = _f32(M[q] + add)
                        for q in range(m):
                            NM[q] = _f32(NM[q] + pen0)
                        continue
                    fi = fidx[f]
                    lsv = lens[fi, c0:c0 + m]
                    if lm == 0:
                        for q in range(m):
                            NM[q] = _f32(NM[q] + wn * 0.3) if lsv[q] > 0 else NM[q]
                        continue
                    prv = pre[fi, c0:c0 + m]
                    idv = bid[fi, c0:c0 + m]
                    nshort = 0
                    for q in range(m):
                        nshort += lsv[q] < 3
                    if lm > 2:
                        cv = counts[fi, c0:c0 + m]
                        pm = mpre[f, vm]
                        for q in range(m):
                            ls = lsv[q]
                            x = pm ^ prv[q]
                            cp = (((x & np.uint32(0xFF)) == 0) + ((x & np.uint32(0xFFFF)) == 0)
                                  + ((x & np.uint32(0xFFFFFF)) == 0) + (x == 0))
                            cp = min(cp, lm)
                            g = _f32(cv[q] * 2.0 / (ls + lm - 2))
                            g = _f32(g + cp * (1.0 - g) / 10.0)    # cp is the prefix bonus count
                            g = 1.0 if idv[q] == vm else g
                            bi[q] = 0.0 if ls == 0 else g
                    # strings shorter than 3 characters: containment rule (exact, scalar)
                    Uf = Utup[f]
                    for q in range(m if (lm < 3 or nshort > 0) else 0):
                        ls = lsv[q]
                        if lm < 3 or ls < 3:
                            j = brecs[p0 + q]
                            cp = _common_prefix(mpre[f, vm], prv[q], lm)
                            cont = (_contains(Mtup[f], vm, lm, Uf, j, ls) if lm >= ls
                                    else _contains(Uf, j, ls, Mtup[f], vm, lm))
                            bi[q] = _bigram_value(0, lm, ls, cp, idv[q] == vm, cont)
                    # M and NM, as in the reference
                    for q in range(m):              # approximate bonus; near-best candidates are rescored
                        b_ = bi[q]
                        bonus[q] = biscale * _cbrt(max(b_ - minbigram, 1e-9)) if b_ > minbigram else 0.0
                    for q in range(m):
                        b_ = bi[q]
                        M[q] = _f32(M[q] + w * (0.5 * (b_ * b_) + bonus[q]))
                    for q in range(m):
                        b_ = bi[q]
                        pen = 1.0 if b_ < lo_pen else ((1.0 - b_ * b_) if b_ < minbigram else (1.0 - b_) * (1.0 - b_))
                        NM[q] = _f32(NM[q] + wn * pen) if lsv[q] > 0 else _f32(NM[q] + wn * 0.3)
                for q in range(m):
                    sc[c0 + q] = _f32(M[q] / (M[q] + NM[q]))
            best_a = -1.0
            for r in range(n):
                if sc[r] > best_a:
                    best_a = sc[r]
            nnear = 0
            for r in range(n):
                if sc[r] >= best_a - MARGIN:
                    s_ = _exact_block_score(i, r, Utup, Mtup, mid, mlen, mpre, kind, fidx, wm, wnm, brecs, s0,
                                            counts, lens, pre, bid, minbigram, lo_pen, biscale, third)
                    if nnear == near.size:
                        near2 = np.empty(2 * near.size, np.int64)
                        near2[:nnear] = near[:nnear]
                        near = near2
                        nsc2 = np.empty(2 * nsc.size, np.float64)
                        nsc2[:nnear] = nsc[:nnear]
                        nsc = nsc2
                    near[nnear] = r
                    nsc[nnear] = s_
                    nnear += 1
                    if s_ > best:
                        best = s_
        # extras: candidates through the other or-blocking variables, outside the block
        nex = 0
        for f in range(nv):
            if not ortype[f]:
                continue
            k_ = mkey[f, i]
            if k_ < 0:
                continue
            base = koff[f]
            for t in range(kcnt[f, k_], kcnt[f, k_ + 1]):
                j = krecs[base + t]
                if bk >= 0 and ubk[j] == bk:
                    continue
                if nex == ex.size:
                    ex2 = np.empty(2 * ex.size, np.int64)
                    ex2[:nex] = ex[:nex]
                    ex = ex2
                ex[nex] = j
                nex += 1
        if nex > 0:
            exs = np.sort(ex[:nex])
            nu_ = 0
            for t in range(nex):
                if t == 0 or exs[t] != exs[t - 1]:
                    ex[nu_] = exs[t]
                    nu_ += 1
            nex = nu_
            if exsc.size < nex:
                exsc = np.empty(nex, np.float64)
            for f in range(nv):
                if kind[f] != 0:
                    vm = mid[f, i]
                    _master_bigrams(Mtup[f], vm, mlen[f, vm], tables[f], 1)
            for t in range(nex):
                j = ex[t]
                Mx = 0.0
                NMx = 0.0
                for f in range(nv):
                    vm = mid[f, i]
                    lm = mlen[f, vm]
                    if kind[f] == 0:
                        if lm > 0:
                            Mx = _f32(Mx + wm[f])
                        continue
                    Uf = Utup[f]
                    ls = _length(Uf, j)
                    eq = uid[f, j] == vm
                    cp = _common_prefix(mpre[f, vm], _prefix4(Uf, j, ls), lm)
                    tag += 1
                    cnt_ = _direct_count(Uf, j, ls, tables[f], stamp, tag) if lm > 2 else 0
                    cont = False
                    if lm < 3 or ls < 3:
                        cont = (_contains(Mtup[f], vm, lm, Uf, j, ls) if lm >= ls
                                else _contains(Uf, j, ls, Mtup[f], vm, lm))
                    b_ = _bigram_value(cnt_, lm, ls, cp, eq, cont)
                    Mx, NMx = _accumulate(Mx, NMx, b_, wm[f], wnm[f], lm == 0, ls == 0,
                                          minbigram, lo_pen, biscale, third)
                s_ = _f32(Mx / (Mx + NMx))
                exsc[t] = s_
                if s_ > best:
                    best = s_
            for f in range(nv):
                if kind[f] != 0:
                    vm = mid[f, i]
                    _master_bigrams(Mtup[f], vm, mlen[f, vm], tables[f], -1)
        # all candidates with the best score
        if best >= minscore:
            for t in range(nnear):
                if nsc[t] == best:
                    if nout < cap:
                        out_i[task, nout] = i
                        out_j[task, nout] = brecs[s0 + near[t]]
                        out_s[task, nout] = best
                    nout += 1
            for t in range(nex):
                if exsc[t] == best:
                    if nout < cap:
                        out_i[task, nout] = i
                        out_j[task, nout] = ex[t]
                        out_s[task, nout] = best
                    nout += 1
    out_n[task] = nout


@nb.njit(parallel=True, cache=True)
def _search(Utup, Mtup, mid, uid, mlen, mpre, kind, fidx, idxf, wm, wnm,
            order, task_off, mblock, ubk, bcnt, brecs,
            mkey, kcnt, krecs, koff, ortype,
            minscore, minbigram, lo_pen, biscale, third, maxn, cap, bad):
    """Best candidates of every master record in order, in tasks (task_off). kind: 0 required,
    1 block variable, 2 indexed (scored through the block's inverted lists); fidx maps indexed
    variables to their encoding row (idxf the inverse); ortype marks the or-blocking variables
    whose matches outside the block are scored directly."""
    ntask = task_off.size - 1
    out_i = np.empty((ntask, cap), np.int64)
    out_j = np.empty((ntask, cap), np.int64)
    out_s = np.empty((ntask, cap), np.float64)
    out_n = np.zeros(ntask, np.int64)
    for task in nb.prange(ntask):
        _task(task, Utup, Mtup, mid, uid, mlen, mpre, kind, fidx, idxf, wm, wnm, order, task_off,
              mblock, ubk, bcnt, brecs, mkey, kcnt, krecs, koff, ortype,
              minscore, minbigram, lo_pen, biscale, third, maxn, cap, out_i, out_j, out_s, out_n, bad)
    return out_i, out_j, out_s, out_n

@nb.njit(cache=True)
def _perfect(tm, tu, ntm):
    """All pairs (i, j) with master tuple tm[i] == using tuple tu[j] (>= 0)."""
    cnt, grp = _group(tm, ntm, np.zeros(tm.size, np.bool_))
    total = 0
    for j in range(tu.size):
        if tu[j] >= 0:
            total += cnt[tu[j] + 1] - cnt[tu[j]]
    pi = np.empty(total, np.int64)
    pj = np.empty(total, np.int64)
    pos = 0
    for j in range(tu.size):
        k = tu[j]
        if k >= 0:
            for t in range(cnt[k], cnt[k + 1]):
                pi[pos] = grp[t]
                pj[pos] = j
                pos += 1
    return pi, pj


@nb.njit(cache=True)
def _lengths_of(Mv):
    out = np.empty(Mv.shape[0], np.int32)
    for v in range(Mv.shape[0]):
        out[v] = _length(Mv, v)
    return out


def _releaser():
    """Return free pages of the C allocator to the OS (macOS malloc_zone_pressure_relief, glibc
    malloc_trim): freed memory otherwise stays resident and counts in the peak RSS."""
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"))
    except OSError:
        return lambda: None
    if hasattr(libc, "malloc_zone_pressure_relief"):
        f = libc.malloc_zone_pressure_relief
        f.argtypes, f.restype = [ctypes.c_void_p, ctypes.c_size_t], ctypes.c_size_t
        return lambda: f(None, 0)
    if hasattr(libc, "malloc_trim"):
        f = libc.malloc_trim
        f.argtypes, f.restype = [ctypes.c_size_t], ctypes.c_int
        return lambda: f(0)
    return lambda: None


_release = _releaser()


def _ucs4(a):
    if isinstance(a, VarRows):
        return a                       # turbo-socialsci 0.2: variable-length rows, read in place
    if isinstance(a, np.ndarray) and a.ndim == 2 and a.dtype == np.uint8:
        return a                       # turbo-socialsci 0.2: ASCII bytes, one NUL-padded row per value
    a = np.ascontiguousarray(np.asarray(a, dtype=str))
    w = max(a.dtype.itemsize // 4, 1)
    if a.size == 0:
        return np.zeros((0, w), np.uint32)
    return a.view(np.uint32).reshape(a.shape[0], w)


def _combine(am, au, bm, bu, nb_):
    """Ids of distinct pairs of master ids (am, bm) and the matching using pairs (au, bu)."""
    keys, vals, ids, nk = _key_table(am, bm, nb_)
    out = np.empty(au.size, np.int32)
    _key_lookup(keys, vals, au, bu, nb_, out)
    return ids, out, nk


def run(master, using, vars, wmatch, wnomatch, required, minscore=0.6, minbigram=0.6):
    vars = list(vars)
    nv = len(vars)
    if nv <= 3:
        raise ValueError("this candidate implements or-blocking, which reclink2 uses with four or more variables")
    req = np.array([v in set(required) for v in vars])
    wm = np.asarray(wmatch, dtype=np.float64)
    wnm = np.asarray(wnomatch, dtype=np.float64)
    nm = len(master[vars[0]])
    nu = len(using[vars[0]])
    threads = nb.get_num_threads()
    bad = np.zeros(1, np.int64)
    _release()                       # pages freed before the call (e.g. while loading the inputs)

    # 1. Dictionaries of the master values; ids of the using values.
    Mtup, Utup = [], []
    mid = np.empty((nv, nm), np.int32)
    uid = np.empty((nv, nu), np.int32)
    nvals = np.empty(nv, np.int64)
    lens_m = []
    for f, v in enumerate(vars):
        mv = master[v]
        if isinstance(mv, np.ndarray) and mv.ndim == 2 and mv.dtype == np.uint8:
            # turbo-socialsci 0.2: ASCII bytes (checked by the caller); the distinct rows in byte
            # order, which for ASCII is the order of the strings
            w = mv.shape[1]
            vals, inv = np.unique(np.ascontiguousarray(mv).view(f"S{w}").reshape(-1), return_inverse=True)
            Mv = np.ascontiguousarray(vals).view(np.uint8).reshape(-1, w)
            ln = _lengths_of(Mv)
        else:
            vals, inv = np.unique(np.asarray(mv, dtype=str), return_inverse=True)
            Mv = _ucs4(vals)
            ln = _lengths_of(Mv)
            if Mv.size and (Mv.max() > 127 or not np.array_equal(ln, np.char.str_len(vals))):
                raise ValueError("master values must be ASCII strings without NUL characters")
        Mtup.append(Mv)
        lens_m.append(ln)
        mid[f] = inv.reshape(-1)
        nvals[f] = len(vals)
        Uf = _ucs4(using[v])
        Utup.append(Uf)
        plen = PREFIX if Uf.shape[1] > PREFIX else 0
        keys, tvals, filt = _build_table(Mv, plen)
        _lookup(Uf, Mv, keys, tvals, filt, plen, uid[f], bad)
    mlen = np.zeros((nv, max(int(nvals.max()), 1)), np.int32)
    mpre = np.zeros((nv, max(int(nvals.max()), 1)), np.uint32)
    for f in range(nv):
        mlen[f, :nvals[f]] = lens_m[f]
        for k in range(min(4, Mtup[f].shape[1])):        # rows are NUL-padded past their length
            mpre[f, :nvals[f]] |= Mtup[f][:, k].astype(np.uint32) << np.uint32(8 * k)
    if bad[0]:
        raise ValueError("using values must be ASCII strings")
    _release()

    # 2. Required values: one key per distinct combination.
    rq = np.flatnonzero(req)
    if rq.size == 0:
        rk_m, rk_u = np.zeros(nm, np.int32), np.zeros(nu, np.int32)
    else:
        rk_m, rk_u = mid[rq[0]].copy(), uid[rq[0]].copy()
        for f in rq[1:]:
            rk_m, rk_u, _ = _combine(rk_m, rk_u, mid[f], uid[f], nvals[f])

    # 3. Perfect matches: equal on every variable (all duplicate combinations).
    tm, tu = mid[0].copy(), uid[0].copy()
    ntm = nvals[0]
    for f in range(1, nv):
        tm, tu, ntm = _combine(tm, tu, mid[f], uid[f], nvals[f])
    pi, pj = _perfect(tm, tu, ntm)
    perfect_m = np.zeros(nm, np.bool_)
    perfect_m[pi] = True
    perfect_u = np.zeros(nu, np.bool_)
    perfect_u[pj] = True

    # 4. Or-blocking keys (required values + one other value, empty master values skipped).
    orv = np.flatnonzero(~req)
    mkey = np.full((nv, nm), -1, np.int32)
    ukey = np.full((nv, nu), -1, np.int32)
    kcnts, krec_list, pairs = {}, {}, {}
    for f in orv:
        ok = ~perfect_m & (mlen[f, mid[f]] > 0) & (rk_m >= 0)
        a = np.where(ok, rk_m, -1).astype(np.int32)
        keys, kv, ids, nk = _key_table(a, mid[f], nvals[f])
        mkey[f] = ids
        _key_lookup(keys, kv, rk_u, uid[f], nvals[f], ukey[f])
        kcnt, krecs = _group(ukey[f], nk, perfect_u)
        kcnts[f], krec_list[f] = kcnt, krecs
        sizes = np.diff(kcnt)
        pairs[f] = int(sizes[ids[ids >= 0]].sum())
    B = max(orv, key=lambda f: pairs[f])                      # the block variable

    # 5. Blocks of the block variable; the other scored variables are encoded per block.
    bcnt, brecs = kcnts[B], krec_list[B]
    nblk = bcnt.size - 1
    bsize = np.diff(bcnt)
    maxn = int(bsize.max()) if nblk else 0
    kind = np.where(req, 0, 2).astype(np.int64)
    kind[B] = 1
    idxf = np.array([f for f in range(nv) if kind[f] == 2], np.int64)
    fidx = np.full(nv, -1, np.int64)
    fidx[idxf] = np.arange(idxf.size)

    # 6. Search, masters ordered by block.
    mblock = mkey[B].copy()
    ubk = ukey[B]
    todo = np.flatnonzero(~perfect_m)
    order = todo[np.argsort(mblock[todo], kind="stable")]
    # tasks of about equal work (candidates in the block plus a constant per master record)
    work = np.where(mblock[order] >= 0, bsize[np.maximum(mblock[order], 0)] if nblk else 0, 0) + 64
    ntask = max(1, min(order.size, 1 if threads == 1 else 4 * threads))
    cum = np.concatenate([[0], np.cumsum(work)])
    task_off = np.searchsorted(cum, np.linspace(0, cum[-1], ntask + 1)).astype(np.int64)
    task_off[0], task_off[-1] = 0, order.size
    ortype = np.zeros(nv, np.bool_)
    for f in orv:
        if f != B:
            ortype[f] = True
    koff = np.zeros(nv, np.int64)
    kc = np.zeros((nv, nm + 1), np.int64)
    parts = []
    pos = 0
    for f in range(nv):
        if ortype[f]:
            koff[f] = pos
            kc[f, :kcnts[f].size] = kcnts[f]
            parts.append(krec_list[f])
            pos += krec_list[f].size
    krecs_all = np.concatenate(parts) if parts else np.zeros(0, np.int32)
    del parts
    for f in list(krec_list):          # turbo-socialsci 0.2: the lists are in krecs_all now
        if f != B:
            krec_list[f] = None
    third = 1 / 3
    biscale = .5 / (1 - minbigram) ** third
    lo_pen = minbigram - .2
    args = (tuple(Utup), tuple(Mtup), mid, uid, mlen, mpre, kind, fidx, idxf, wm, wnm, order, task_off,
            mblock, ubk, bcnt, brecs, mkey, kc, krecs_all, koff,
            ortype, float(minscore), float(minbigram), lo_pen, biscale, third, max(maxn, 1))
    cap = 4 * int(np.diff(task_off).max(initial=0)) + 64      # room for ties; rerun if exceeded
    oi, oj, os_, on = _search(*args, cap, bad)
    if on.max(initial=0) > cap:
        oi, oj, os_, on = _search(*args, int(on.max()), bad)
    if bad[0]:
        raise ValueError("using values must be ASCII strings")
    keep = np.arange(oi.shape[1])[None, :] < on[:, None]
    i = np.concatenate([pi, oi[keep]])
    j = np.concatenate([pj, oj[keep]])
    sc = np.concatenate([np.ones(pi.size), os_[keep]])
    o = np.lexsort((j, i))
    return i[o], j[o], sc[o], int(pi.size)
