# turbo-socialsci core, new in 0.2 (not from a benchmark candidate).
"""Sums of influence functions times Rademacher multipliers (the multiplier bootstrap's draws), from
tables of signed sums: the "four Russians" method for a product with a +1/-1 matrix.

The multipliers are those of _draws.rademacher: the bits of SplitMix64 words keyed by (seed, unit,
draw), so every tiling of units and draws gives the same multipliers. For each 8 consecutive rows
of the influence functions, the 8 rows' words are transposed (as an 8 x 8 matrix of bytes, then of
bits) into one byte per draw, the pattern of the 8 multipliers, and a table holds the 256 signed
sums of the 8 rows' values. A draw's sum is then one table row per 8 rows. The additions run as
4-double vectors (LLVM intrinsics: Numba leaves LLVM's SLP vectorizer off), 4 tables at a time.
"""
import numpy as np
import numba as nb
from llvmlite import ir
from numba import types
from numba.core import cgutils
from numba.extending import intrinsic

from ._draws import GAMMA, M1, M2

_G64 = np.uint64(GAMMA)
_M164 = np.uint64(M1)
_M264 = np.uint64(M2)
SIGN_DRAWS = 8192             # draws per task
BATCH = 4096                  # rows whose multipliers' patterns are held at once (13 MB at 25,000 draws)
_LOWBIT = np.array([0] + [(s & -s).bit_length() - 1 for s in range(1, 256)], dtype=np.int64)


@nb.njit(cache=True, inline="always")
def _mix64(key, u, nw, word):
    z = key + (u * nw + np.uint64(word) + np.uint64(1)) * _G64
    z = (z ^ (z >> np.uint64(30))) * _M164
    z = (z ^ (z >> np.uint64(27))) * _M264
    return z ^ (z >> np.uint64(31))


@nb.njit(cache=True, inline="always")
def _transpose8(x):
    """Transpose of the 8 x 8 bit matrix with row i in byte i (bit j of the byte: column j)."""
    t = (x ^ (x >> np.uint64(7))) & np.uint64(0x00AA00AA00AA00AA)
    x = x ^ t ^ (t << np.uint64(7))
    t = (x ^ (x >> np.uint64(14))) & np.uint64(0x0000CCCC0000CCCC)
    x = x ^ t ^ (t << np.uint64(14))
    t = (x ^ (x >> np.uint64(28))) & np.uint64(0x00000000F0F0F0F0)
    return x ^ t ^ (t << np.uint64(28))


@nb.njit(cache=True, inline="always")
def _swap_bytes(h, i, j, shift, mask):
    t = ((h[i] >> np.uint64(shift)) ^ h[j]) & np.uint64(mask)
    h[j] ^= t
    h[i] ^= t << np.uint64(shift)


@nb.njit(cache=True, parallel=True)
def _sign_patterns(rows, key, nwords, B, P64, nparts):
    """Byte b of row g of P64 (viewed as bytes): the multipliers of rows 8g..8g+7 for draw b as bits
    (bit i: row 8g + i, 1 for +1), from _draws.rademacher's words: the 8 rows' 64-bit words transposed
    as an 8 x 8 matrix of bytes, then each byte block as an 8 x 8 matrix of bits."""
    R = rows.size
    ngrp = P64.shape[0]
    nw = np.uint64(nwords)
    nwb = (B + 63) // 64
    for part in nb.prange(nparts):
        h = np.zeros(8, np.uint64)
        for g in range(ngrp * part // nparts, ngrp * (part + 1) // nparts):
            r0 = 8 * g
            m = max(0, min(8, R - r0))
            for w in range(nwb):
                for i in range(8):
                    h[i] = _mix64(key, np.uint64(rows[r0 + i]), nw, w) if i < m else np.uint64(0)
                for i in (0, 2, 4, 6):
                    _swap_bytes(h, i, i + 1, 8, 0x00FF00FF00FF00FF)
                for i in (0, 1, 4, 5):
                    _swap_bytes(h, i, i + 2, 16, 0x0000FFFF0000FFFF)
                for i in (0, 1, 2, 3):
                    _swap_bytes(h, i, i + 4, 32, 0x00000000FFFFFFFF)
                for k in range(8):                # byte i of h[k]: byte k of row i's word
                    P64[g, 8 * w + k] = _transpose8(h[k])


def _vec4(context, builder, aryty, ary_value, index):
    """Pointer to the 4 doubles of a 1-D float64 array starting at index (no wraparound)."""
    ary = context.make_array(aryty)(context, builder, ary_value)
    ptr = cgutils.get_item_pointer(context, builder, aryty, ary, [index])
    return builder.bitcast(ptr, ir.VectorType(ir.DoubleType(), 4).as_pointer())


@intrinsic
def _add4(typingctx, dst, d, src, e, inc, i):
    """dst[d:d+4] = src[e:e+4] + inc[i:i+4], as one 4-double vector operation (Numba leaves LLVM's SLP
    vectorizer off, so four scalar statements stay scalar)."""
    sig = types.void(dst, types.intp, src, types.intp, inc, types.intp)

    def codegen(context, builder, signature, args):
        a = builder.load(_vec4(context, builder, signature.args[2], args[2], args[3]), align=8)
        b = builder.load(_vec4(context, builder, signature.args[4], args[4], args[5]), align=8)
        builder.store(builder.fadd(a, b), _vec4(context, builder, signature.args[0], args[0], args[1]), align=8)
        return context.get_dummy_value()
    return sig, codegen


@intrinsic
def _acc4x4(typingctx, acc, a, T, k0, k1, k2, k3):
    """acc[a:a+4] += (T[k0:k0+4] + T[k1:k1+4]) + (T[k2:k2+4] + T[k3:k3+4]), as 4-double vectors."""
    sig = types.void(acc, types.intp, T, types.intp, types.intp, types.intp, types.intp)

    def codegen(context, builder, signature, args):
        accty, Tty = signature.args[0], signature.args[2]
        ld = lambda j: builder.load(_vec4(context, builder, Tty, args[2], args[j]), align=8)
        s = builder.fadd(builder.fadd(ld(3), ld(4)), builder.fadd(ld(5), ld(6)))
        pa = _vec4(context, builder, accty, args[0], args[1])
        builder.store(builder.fadd(builder.load(pa, align=8), s), pa, align=8)
        return context.get_dummy_value()
    return sig, codegen


@nb.njit(cache=True, parallel=True)
def _sign_lookup(buf, P, B, out):
    """out[q, b] += sum over 8-row groups g of the table row P[g, b] selects (signed sums of the
    group's values in column q), in tasks of 4 columns x SIGN_DRAWS draws; each draw adds 4 tables'
    rows at a time to its sums, as 4-double vectors."""
    R, Q = buf.shape
    ngrp = P.shape[0]                                   # a multiple of 4 (extra groups: zero)
    nchunk = (Q + 3) // 4
    nblk = (B + SIGN_DRAWS - 1) // SIGN_DRAWS
    for task in nb.prange(nchunk * nblk):
        q0 = (task // nblk) * 4
        qn = min(4, Q - q0)
        b0 = (task % nblk) * SIGN_DRAWS
        nd = min(SIGN_DRAWS, B - b0)
        acc = np.zeros(4 * SIGN_DRAWS)
        T = np.zeros(4 * 1024)                          # T[(gi * 256 + s) * 4 + q]
        two = np.zeros(32)
        for g0 in range(0, ngrp, 4):
            for gi in range(4):                     # T[gi, s] = sum_i (2 bit_i(s) - 1) buf[row i of the group]
                r0 = 8 * (g0 + gi)
                t0 = gi * 1024
                for q in range(4):
                    T[t0 + q] = 0.0
                for i in range(8):
                    for q in range(4):
                        v = buf[r0 + i, q0 + q] if (r0 + i < R and q < qn) else 0.0
                        two[4 * i + q] = 2.0 * v
                        T[t0 + q] -= v
                for s in range(1, 256):
                    _add4(T, t0 + 4 * s, T, t0 + 4 * (s & (s - 1)), two, 4 * _LOWBIT[s])
            for b in range(b0, b0 + nd):
                _acc4x4(acc, 4 * (b - b0), T, 4 * np.int64(P[g0, b]), 1024 + 4 * np.int64(P[g0 + 1, b]),
                        2048 + 4 * np.int64(P[g0 + 2, b]), 3072 + 4 * np.int64(P[g0 + 3, b]))
        for q in range(qn):
            for b in range(nd):
                out[q0 + q, b0 + b] += acc[4 * b + q]


def signed_sums(buf, rows, key, nwords, B):
    """buf.T @ V for the multipliers V[r, b] of rows[r] and draws b < B (_draws.rademacher's), without
    forming V: for each 8 consecutive rows, a table holds the 256 signed sums of their values, and
    each draw adds the row its 8 multipliers (as bits) select: one addition per 8 rows instead of 8
    multiply-adds. Rows are taken BATCH at a time, so the patterns take B bytes per 8 rows of a batch."""
    buf = np.ascontiguousarray(buf, dtype=np.float64)
    rows = np.ascontiguousarray(rows, dtype=np.int64)
    R = buf.shape[0]
    out = np.zeros((buf.shape[1], B))
    nwb = (B + 63) // 64
    for r0 in range(0, R, BATCH):
        r1 = min(R, r0 + BATCH)
        ngrp = -(-((r1 - r0 + 7) // 8) // 4) * 4
        P64 = np.zeros((ngrp, 8 * nwb), np.uint64)
        _sign_patterns(rows[r0:r1], np.uint64(key), nwords, B, P64, max(1, min(nb.get_num_threads(), ngrp // 64 + 1)))
        _sign_lookup(buf[r0:r1], P64.view(np.uint8), B, out)
        del P64
    return out
