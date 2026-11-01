"""Task 008 candidate: reclink2 record linkage with master-side dictionaries, city blocks and
inverted bigram lists (Numba, one thread).

Same computation as reference.py, reorganized:

1. Exact matching uses dictionaries of the master values only (n_master is much smaller than
   n_using). Every using value is hashed once and looked up (most values that are absent from the
   master file are rejected by a one-bit filter); absent values get id -1. Perfect matches, the
   blocking joins and the "equal strings score 1" rule only ever compare a using value with a
   master value, so these ids suffice.
2. Eligible using records (not perfectly matched, required() values present in the master file)
   are sorted by (required values, layout field), the or-block field whose agreeing records form
   the most candidate pairs (city). Most candidates of a master record are one block of records
   that agree with it on the layout field and the required fields. Master records are processed
   in the same order, so consecutive master records share a block.
3. When a block is first needed, its values are encoded once (length, first four characters,
   distinct bigram codes) together with an inverted list per bigram. A master record's bigram
   counts against the whole block are then sums over the lists of its own bigrams. Candidates
   found through the other or-block fields are encoded per master record and counted directly.
4. The rest of the comparator, the weights and the float32 rounding after every operation follow
   the reference operation by operation, vectorized over chunks of candidates, so the scores are
   bitwise identical. (bi - minbigram)^(1/3) is memoized by value.
5. Or-blocking without duplicate pairs: a candidate found through or-block field v is skipped if
   it also agrees with the master record on an or-block field processed before v.

Inputs must be ASCII without embedded NUL characters (the reference itself fails on non-ASCII
strings); other inputs raise ValueError.
"""
import numpy as np
from llvmlite import ir
from numba import njit, types
from numba.core import cgutils
from numba.extending import intrinsic

BISCALE_POWER = 1 / 3


def _multipliers(n):
    """Odd 32-bit hash multipliers per character position (data-independent constants)."""
    m = np.random.default_rng(8008).integers(0, 2**32, size=(2, max(n, 4096)), dtype=np.uint64)
    return (m | np.uint64(1)).astype(np.uint32)


_P = _multipliers(4096)


@intrinsic
def _prefetch(typingctx, arr, flat):
    """Prefetch element `flat` of a C-contiguous array (llvm.prefetch: read, high locality)."""
    def codegen(context, builder, sig, args):
        ary = context.make_array(sig.args[0])(context, builder, args[0])
        ptr = builder.gep(ary.data, [context.cast(builder, args[1], sig.args[1], types.intp)])
        i8p = ir.IntType(8).as_pointer()
        fn = cgutils.get_or_insert_function(
            builder.module, ir.FunctionType(ir.VoidType(), [i8p] + [ir.IntType(32)] * 3), "llvm.prefetch.p0i8")
        builder.call(fn, [builder.bitcast(ptr, i8p), ir.Constant(ir.IntType(32), 0),
                          ir.Constant(ir.IntType(32), 3), ir.Constant(ir.IntType(32), 1)])
        return context.get_dummy_value()
    return types.void(arr, flat), codegen


@njit(cache=False, inline="always")
def _fmix(h):
    h ^= h >> np.uint64(33)
    h *= np.uint64(0xFF51AFD7ED558CCD)
    h ^= h >> np.uint64(33)
    h *= np.uint64(0xC4CEB9FE1A85EC53)
    h ^= h >> np.uint64(33)
    return h


@njit(cache=False, error_model="numpy", inline="always")
def _hash_row(buf, r, P):
    """Length and hash of row r of a fixed-width UCS4 array, over the full width (padding is zero,
    so it adds nothing); status 1 = embedded NUL, 2 = non-ASCII."""
    h1 = np.uint32(0)
    h2 = np.uint32(0)
    bad = np.uint32(0)
    nz = 0
    last = -1
    for k in range(buf.shape[1]):
        x = buf[r, k]
        h1 += x * P[0, k]
        h2 += x * P[1, k]
        bad |= x
        nz += np.int64(x != 0)
        last = max(last, k if x != 0 else -1)
    h = _fmix((np.uint64(h1) << np.uint64(32)) | np.uint64(h2))
    return nz, h, (1 if last + 1 != nz else 0) | (2 if bad >= 128 else 0)


@njit(cache=False, error_model="numpy")
def _master_dict(buf, P, lens, hashes, seen):
    """Ids of the distinct master values of one field (open addressing), lengths, hashes and the
    characters used. Returns ids, representative rows, slots, status."""
    n = buf.shape[0]
    size = 16
    while size < 2 * n:
        size *= 2
    mask = np.uint64(size - 1)
    slots = np.full(size, -1, np.int32)
    rep = np.empty(n, np.int32)
    ids = np.empty(n, np.int32)
    status = 0
    nid = 0
    for r in range(n):
        row = buf[r]
        L, h, st = _hash_row(buf, r, P)
        lens[r] = L
        hashes[r] = h
        status |= st
        for k in range(L):
            seen[row[k] & np.uint32(127)] = True
        s = h & mask
        while True:
            t = slots[s]
            if t < 0:
                slots[s] = nid
                rep[nid] = r
                ids[r] = nid
                nid += 1
                break
            u = rep[t]
            if hashes[u] == h and lens[u] == L:
                same = True
                for k in range(L):
                    if buf[u, k] != row[k]:
                        same = False
                        break
                if same:
                    ids[r] = t
                    break
            s = (s + np.uint64(1)) & mask
    return ids, rep[:nid].copy(), slots, status


@njit(cache=False, error_model="numpy")
def _probe_table(slots, rep, hashes, bits):
    """The slots with hash tags ((hash >> 32) << 32 | id + 1, 0 = empty) and a one-bit filter
    (2^bits bits) of the master value hashes."""
    tab = np.zeros(slots.shape[0], np.uint64)
    for s in range(slots.shape[0]):
        t = slots[s]
        if t >= 0:
            tab[s] = ((hashes[rep[t]] >> np.uint64(32)) << np.uint64(32)) | np.uint64(t + 1)
    m = (np.uint64(1) << np.uint64(bits)) - np.uint64(1)
    bloom = np.zeros(((1 << bits) + 63) // 64, np.uint64)
    for t in range(rep.shape[0]):
        bit = (hashes[rep[t]] >> np.uint64(20)) & m
        bloom[bit >> np.uint64(6)] |= np.uint64(1) << (bit & np.uint64(63))
    return tab, bloom


@njit(cache=False, error_model="numpy")
def _using_ids(buf, P, tab, bloom, bits, rep, mbuf, mlens, ids):
    """Master-dictionary id of every using value of one field (-1 if absent from the master
    file); returns the status."""
    mask = np.uint64(tab.shape[0] - 1)
    m = (np.uint64(1) << np.uint64(bits)) - np.uint64(1)
    status = 0
    for r in range(buf.shape[0]):
        L, h, st = _hash_row(buf, r, P)
        status |= st
        idv = -1
        bit = (h >> np.uint64(20)) & m
        if bloom[bit >> np.uint64(6)] & (np.uint64(1) << (bit & np.uint64(63))):
            tag = h >> np.uint64(32)
            s = h & mask
            while True:
                e = tab[s]
                if e == 0:
                    break
                if (e >> np.uint64(32)) == tag:
                    t = np.int64(e & np.uint64(0xFFFFFFFF)) - 1
                    u = np.uint64(rep[t])
                    if mlens[u] == L:
                        same = True
                        for k in range(L):
                            if mbuf[u, k] != buf[r, k]:
                                same = False
                                break
                        if same:
                            idv = t
                            break
                s = (s + np.uint64(1)) & mask
        ids[r] = idv
    return status


@njit(cache=False)
def _tuple_ids(mid, uid, fields, P):
    """Ids of the distinct master tuples of mid[fields]; using tuples get the matching id or -1."""
    nm = mid.shape[1]
    nu = uid.shape[1]
    size = 16
    while size < 2 * nm:
        size *= 2
    mask = np.uint64(size - 1)
    slots = np.full(size, -1, np.int32)
    rep = np.empty(nm, np.int32)
    tm = np.empty(nm, np.int32)
    tu = np.empty(nu, np.int32)
    nid = 0
    for r in range(nm):
        h = np.uint64(0)
        for k in range(fields.shape[0]):
            h += np.uint64(mid[fields[k], r] + 1) * np.uint64(P[0, k])
        h = _fmix(h)
        s = h & mask
        while True:
            t = slots[s]
            if t < 0:
                slots[s] = nid
                rep[nid] = r
                tm[r] = nid
                nid += 1
                break
            u = rep[t]
            same = True
            for k in range(fields.shape[0]):
                if mid[fields[k], u] != mid[fields[k], r]:
                    same = False
                    break
            if same:
                tm[r] = t
                break
            s = (s + np.uint64(1)) & mask
    for r in range(nu):
        tu[r] = -1
        h = np.uint64(0)
        bad = False
        for k in range(fields.shape[0]):
            x = uid[fields[k], r]
            if x < 0:
                bad = True
                break
            h += np.uint64(x + 1) * np.uint64(P[0, k])
        if bad:
            continue
        h = _fmix(h)
        s = h & mask
        while True:
            t = slots[s]
            if t < 0:
                break
            u = rep[t]
            same = True
            for k in range(fields.shape[0]):
                if mid[fields[k], u] != uid[fields[k], r]:
                    same = False
                    break
            if same:
                tu[r] = t
                break
            s = (s + np.uint64(1)) & mask
    return tm, tu, nid


@njit(cache=False)
def _perfect(tm, tu, ntup):
    """All (master, using) pairs with equal tuples, and the removal flags."""
    nm = tm.shape[0]
    nu = tu.shape[0]
    ptr = np.zeros(ntup + 1, np.int64)
    for r in range(nm):
        ptr[tm[r] + 1] += 1
    for k in range(ntup):
        ptr[k + 1] += ptr[k]
    rows = np.empty(nm, np.int64)
    fill = ptr[:-1].copy()
    for r in range(nm):
        rows[fill[tm[r]]] = r
        fill[tm[r]] += 1
    npairs = 0
    for j in range(nu):
        if tu[j] >= 0:
            npairs += ptr[tu[j] + 1] - ptr[tu[j]]
    pi = np.empty(npairs, np.int64)
    pj = np.empty(npairs, np.int64)
    rm = np.zeros(nm, np.bool_)
    ru = np.zeros(nu, np.bool_)
    k = 0
    for j in range(nu):
        t = tu[j]
        if t < 0:
            continue
        ru[j] = True
        for e in range(ptr[t], ptr[t + 1]):
            pi[k] = rows[e]
            pj[k] = j
            rm[rows[e]] = True
            k += 1
    return pi, pj, rm, ru


@njit(cache=False, error_model="numpy")
def _sort2(items, k1, K1, k2, K2):
    """Stable sort of items by (k2[item], k1[item]) in two counting-sort passes. Returns the
    sorted items, their k1 values and the start of each k2 value."""
    n = items.shape[0]
    c1 = np.zeros(K1 + 1, np.int64)
    for t in range(n):
        c1[k1[items[t]] + 1] += 1
    for k in range(K1):
        c1[k + 1] += c1[k]
    tmp = np.empty(n, np.int64)
    t1 = np.empty(n, np.int64)
    t2 = np.empty(n, np.int64)
    for t in range(n):
        x = items[t]
        d = c1[k1[x]]
        tmp[d] = x
        t1[d] = k1[x]
        t2[d] = k2[x]
        c1[k1[x]] += 1
    c2 = np.zeros(K2 + 1, np.int64)
    for t in range(n):
        c2[t2[t] + 1] += 1
    for k in range(K2):
        c2[k + 1] += c2[k]
    start = c2.copy()
    out = np.empty(n, np.int64)
    s1 = np.empty(n, np.int64)
    for t in range(n):
        d = c2[t2[t]]
        out[d] = tmp[t]
        s1[d] = t1[t]
        c2[t2[t]] += 1
    return out, s1, start


@njit(cache=False, error_model="numpy")
def _value_lists(ids, empty, nids):
    """For each master-dictionary id other than the empty string's, the positions p (ascending)
    with ids[p] == id."""
    ptr = np.zeros(nids + 1, np.int64)
    for p in range(ids.shape[0]):
        if ids[p] >= 0 and ids[p] != empty:
            ptr[ids[p] + 1] += 1
    for k in range(nids):
        ptr[k + 1] += ptr[k]
    lst = np.empty(ptr[nids], np.int64)
    fill = ptr[:-1].copy()
    for p in range(ids.shape[0]):
        k = ids[p]
        if k >= 0 and k != empty:
            lst[fill[k]] = p
            fill[k] += 1
    return ptr, lst


@njit(cache=False, error_model="numpy")
def _encode_master(buf, lens, remap, A, s, codes, off, pre):
    """Bigram codes (A * char1 + char2 in the field alphabet), with multiplicity, and the first
    four characters of every master value of field s."""
    for r in range(buf.shape[0]):
        row = buf[r]
        L = lens[r]
        x = np.uint32(0)
        for k in range(min(L, 4)):
            x |= row[k] << np.uint32(8 * k)
        pre[s, r] = x
        q = off[s, r]
        if L >= 2:
            a = remap[row[0] & np.uint32(127)]
            for k in range(1, L):
                b = remap[row[k] & np.uint32(127)]
                codes[np.uint64(q)] = a * A + b
                a = b
                q += 1


@njit(cache=False)
def _lower_bound(a, lo, hi, x):
    while lo < hi:
        mid = (lo + hi) >> 1
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=False)
def _grow(a, cap):
    b = np.empty(cap, a.dtype)
    b[:a.shape[0]] = a
    return b


@njit(cache=False, inline="always")
def _f32(x):
    return np.float64(np.float32(x))


@njit(cache=False, error_model="numpy")
def _candidates(i, gs, ge, mid, mlen, scored, uid, order, skey, ob, ptr, ptr_base, lst, cand):
    """Candidates of master record i: positions [p0, p1) of order that agree with it on the
    layout field (the whole group without or-blocking), and nx other using rows in cand."""
    if ob.shape[0] == 0:
        return gs, ge, 0
    p0 = 0
    p1 = 0
    nx = 0
    f0 = scored[ob[0]]
    if mlen[f0, i] > 0:
        k = np.int64(mid[f0, i])
        p0 = _lower_bound(skey, gs, ge, k)
        p1 = _lower_bound(skey, p0, ge, k + 1)
    for o in range(1, ob.shape[0]):
        f = scored[ob[o]]
        if mlen[f, i] == 0:
            continue
        k = mid[f, i]
        end = ptr[ptr_base[o] + k + 1]
        a = _lower_bound(lst, ptr[ptr_base[o] + k], end, gs)
        b = _lower_bound(lst, a, end, ge)
        for e in range(a, b):
            p = lst[e]
            if p0 <= p < p1:                             # agrees on the layout field
                continue
            j = order[p]
            dup = False
            for o2 in range(1, o):
                f2 = scored[ob[o2]]
                if mlen[f2, i] > 0 and uid[f2, j] == mid[f2, i]:
                    dup = True
                    break
            if not dup:
                cand[nx] = j
                nx += 1
    return p0, p1, nx


@njit(cache=False, error_model="numpy")
def _fill(buf, uidf, rows, b0, s, A, remap, clen, cid, cpre, boff, bcodes, stamp, serial):
    """Length, master-dictionary id, first four characters and distinct bigram codes of the using
    values buf[rows[k]], stored at candidate position b0 + k of field s."""
    bo = boff[s]
    bc = bcodes[s]
    st = stamp[s]
    q = bo[b0]
    w = buf.shape[1]
    n = rows.shape[0]
    for k in range(n):
        if k + 8 < n:                                    # the rows are scattered: prefetch ahead
            jn = rows[k + 8]
            _prefetch(buf, jn * w)
            if w > 16:
                _prefetch(buf, jn * w + 16)
            _prefetch(uidf, jn)
        j = rows[k]
        row = buf[j]
        L = 0
        while L < w:
            if row[np.uint64(L)] == 0:
                break
            L += 1
        b = np.uint64(b0 + k)
        clen[s, b] = L
        cid[s, b] = uidf[j]
        x = np.uint32(0)
        for k2 in range(min(L, 4)):
            x |= row[k2] << np.uint32(8 * k2)
        cpre[s, b] = x
        if L >= 2:
            serial += 1
            a = remap[row[0] & np.uint32(127)]
            for k2 in range(1, L):
                c2 = remap[row[k2] & np.uint32(127)]
                code = a * A + c2
                a = c2
                if st[code] != serial:
                    st[code] = serial
                    bc[np.uint64(q)] = code
                    q += 1
        bo[b + np.uint64(1)] = q
    return serial


@njit(cache=False, error_model="numpy")
def _invert(boff, bcodes, s, nb, A2, iptr, irows):
    """Inverted lists of candidates [0, nb) for field s: irows[s, iptr[s, c]:iptr[s, c + 1]] are
    the candidates whose value contains bigram c."""
    bo = boff[s]
    bc = bcodes[s]
    ip = iptr[s]
    ir = irows[s]
    for c in range(A2 + 1):
        ip[c] = 0
    for q in range(bo[nb]):
        ip[np.uint64(bc[q]) + np.uint64(1)] += 1
    for c in range(A2):
        ip[c + 1] += ip[c]
    for b in range(nb):
        for q in range(bo[b], bo[b + 1]):
            c = bc[np.uint64(q)]
            ir[np.uint64(ip[c])] = b
            ip[c] += 1
    for c in range(A2, 0, -1):
        ip[c] = ip[c - 1]
    ip[0] = 0


@njit(cache=False, error_model="numpy")
def _search(morder, gm, mid, mlen, mpre, moff, mcodes, kind, scored, ubufs, uid, order, skey,
            gstart, gend, ob, ptr, ptr_base, lst, A, remap, Amax2,
            wm, wnm, wnm03, termM1, minscore, minbigram, lo_thr, biscale, third):
    CH = 128
    F = kind.shape[0]
    S = scored.shape[0]
    lay = ob[0] if ob.shape[0] > 0 else -1
    wmax = 1
    for s in range(S):
        wmax = max(wmax, ubufs[s].shape[1])
    T = np.zeros((S, Amax2), np.int32)
    # candidate-local values: the current block at [0, nb), then the master record's other candidates
    ncap = 1024
    qcap = ncap * wmax
    clen = np.zeros((S, ncap), np.int32)
    cid = np.zeros((S, ncap), np.int32)
    cpre = np.zeros((S, ncap), np.uint32)
    ccnt = np.zeros((S, ncap), np.int32)
    boff = np.zeros((S, ncap + 1), np.int64)
    bcodes = np.zeros((S, qcap), np.uint16)
    irows = np.zeros((S, qcap), np.uint32)
    iptr = np.zeros((S, Amax2 + 1), np.int64)
    stamp = np.zeros((S, Amax2), np.int64)
    serial = 0
    cand = np.empty(max(order.shape[0], 1), np.int64)
    win = np.empty(ncap, np.int64)
    num = np.empty(CH)
    den = np.empty(CH)
    kw = np.empty(CH)
    uef = np.empty(CH)
    bi = np.empty(CH)
    bon = np.empty(CH)
    Mw = np.empty(CH)
    NMw = np.empty(CH)
    score = np.empty(CH)
    key = np.full(4096, -1.0)                    # memo of biscale * (bi - minbigram)^(1/3) by bi
    val = np.empty(4096)
    cap = 1024
    oi = np.empty(cap, np.int64)
    oj = np.empty(cap, np.int64)
    osc = np.empty(cap, np.float64)
    n_out = 0
    blk0 = -1
    blk1 = -1
    for t in range(morder.shape[0]):
        i = morder[t]
        g = gm[i]
        p0, p1, nx = _candidates(i, gstart[g], gend[g], mid, mlen, scored, uid, order, skey, ob, ptr,
                                 ptr_base, lst, cand)
        nb = p1 - p0
        nc = nb + nx
        if nc == 0:
            continue
        if nc > ncap or nc * wmax > qcap:
            while nc > ncap:
                ncap *= 2
            qcap = max(qcap, ncap * wmax)
            clen = np.zeros((S, ncap), np.int32)
            cid = np.zeros((S, ncap), np.int32)
            cpre = np.zeros((S, ncap), np.uint32)
            ccnt = np.zeros((S, ncap), np.int32)
            boff = np.zeros((S, ncap + 1), np.int64)
            bcodes = np.zeros((S, qcap), np.uint16)
            irows = np.zeros((S, qcap), np.uint32)
            win = np.empty(ncap, np.int64)
            blk0 = -1
        if p0 != blk0 or p1 != blk1:                     # a new block: encode it, invert it
            blk0 = p0
            blk1 = p1
            for s in range(S):
                if s == lay:                             # agrees with every master record of the block
                    k = mid[scored[s], i]
                    for b in range(nb):
                        clen[s, b] = 1
                        cid[s, b] = k
                        boff[s, b] = 0
                    boff[s, nb] = 0
                    continue
                boff[s, 0] = 0
                serial = _fill(ubufs[s], uid[scored[s]], order[p0:p1], 0, s, A[s], remap[s], clen, cid,
                               cpre, boff, bcodes, stamp, serial)
                _invert(boff, bcodes, s, nb, A[s] * A[s], iptr, irows)
        if nx > 0:
            for s in range(S):
                serial = _fill(ubufs[s], uid[scored[s]], cand[:nx], nb, s, A[s], remap[s], clen, cid,
                               cpre, boff, bcodes, stamp, serial)
        # bigram count tables of the master values; bigram counts of the candidates
        for s in range(S):
            f = scored[s]
            Ts = T[s]
            for c in mcodes[moff[s, i]:moff[s, i + 1]]:
                Ts[c] += 1
            cs = ccnt[s]
            for b in range(nc):
                cs[b] = 0
            if mlen[f, i] < 3:
                continue
            if s != lay and nb > 0:
                ip = iptr[s]
                ir = irows[s]
                for c in mcodes[moff[s, i]:moff[s, i + 1]]:
                    for r in ir[ip[c]:ip[np.uint64(c) + np.uint64(1)]]:
                        cs[r] += 1
            bo = boff[s]
            bc = bcodes[s]
            for b in range(nb, nc):
                c = 0
                for q in range(bo[b], bo[b + 1]):
                    c += Ts[bc[np.uint64(q)]]
                cs[b] = c
        best = -1.0
        nw = 0
        for c0 in range(0, nc, CH):
            cn = min(CH, nc - c0)
            for b in range(cn):
                Mw[b] = 0.0
                NMw[b] = 0.0
            # score of every candidate (reference.run step 3), one field at a time
            for f in range(F):
                s = kind[f]
                lm = mlen[f, i]
                if s < 0:                                # required(): agrees by construction
                    if lm > 0:
                        w = wm[f]
                        for b in range(cn):
                            Mw[b] = _f32(Mw[b] + w)
                    continue
                lb = clen[s, c0:c0 + cn]
                for b in range(cn):
                    uef[b] = np.float64(lb[b] == 0)
                w03 = wnm03[f]
                if lm == 0:                              # empty master value: no M or NM term
                    for b in range(cn):
                        a2 = _f32(NMw[b] + w03)
                        NMw[b] = NMw[b] if uef[b] > 0.5 else a2
                    continue
                if s == lay and c0 + cn <= nb:           # the block agrees on the layout field
                    c1 = termM1[f]
                    for b in range(cn):
                        Mw[b] = _f32(Mw[b] + c1)
                    continue
                mp = mpre[s, i]
                mi = mid[f, i]
                pb = cpre[s, c0:c0 + cn]
                ib = cid[s, c0:c0 + cn]
                cb = ccnt[s, c0:c0 + cn]
                # bigram1 as g = f32(num / den), then f32(g + k (1 - g) / 10) with k the common-
                # prefix count; equal strings get (1, 1, k = 0) and empty using values (0, 1, 0)
                nshort = 0
                lm2 = np.int32(lm - 2)
                for b in range(cn):
                    ls = lb[b]
                    x = mp ^ pb[b]
                    ki = np.int32(np.int32((x & np.uint32(0xFF)) == 0) + np.int32((x & np.uint32(0xFFFF)) == 0)
                                  + np.int32((x & np.uint32(0xFFFFFF)) == 0) + np.int32(x == 0))
                    ne = ib[b] != mi
                    lg = ls >= 3
                    ef = np.float64(not ne)
                    lgf = np.float64(lg)
                    # exact 0/1 blends: x * 1 = x, x * 0 = 0, x + 0 = x
                    num[b] = ef + (1.0 - ef) * (lgf * (cb[b] * 2.0))
                    den[b] = ef + (1.0 - ef) * (lgf * np.float64(np.int32(ls + lm2)) + (1.0 - lgf))
                    kw[b] = np.float64(np.int32(ne & (ls > 0)) * ki)
                    nshort += np.int32(ne & (ls > 0) & (ls < 3))
                if lm < 3 or nshort > 0:                 # strings shorter than 3 characters
                    Ts = T[s]
                    As = A[s]
                    bo = boff[s]
                    bc = bcodes[s]
                    m0 = moff[s, i]
                    m1 = moff[s, i + 1]
                    for b in range(cn):
                        ls = lb[b]
                        if (ls >= 3 and lm >= 3) or ls == 0 or ib[b] == mi:
                            continue
                        q0 = bo[c0 + b]
                        q1 = bo[c0 + b + 1]
                        contains = False
                        if lm == 1:
                            xc = remap[s, mp & np.uint32(127)]
                            for q in range(q0, q1):
                                if bc[q] // As == xc or bc[q] % As == xc:
                                    contains = True
                        elif ls == 1:
                            xc = remap[s, pb[b] & np.uint32(127)]
                            for q in range(m0, m1):
                                if mcodes[q] // As == xc or mcodes[q] % As == xc:
                                    contains = True
                        elif lm == 2:
                            for q in range(q0, q1):
                                if Ts[bc[q]] > 0:
                                    contains = True
                        else:
                            contains = Ts[bc[q0]] > 0
                        num[b] = min(lm, ls) if contains else 0.0
                        den[b] = lm + ls - 1 if contains else 1
                for b in range(cn):
                    gg = _f32(num[b] / den[b])
                    bi[b] = _f32(gg + kw[b] * (1.0 - gg) / 10.0)
                for b in range(cn):
                    v = bi[b]
                    if v > minbigram:
                        h = np.int64(v * 16777216.0) & 4095
                        if key[h] != v:
                            key[h] = v
                            val[h] = biscale * (v - minbigram) ** third
                        bon[b] = val[h]
                    else:
                        bon[b] = 0.0
                wmf = wm[f]
                wnmf = wnm[f]
                for b in range(cn):
                    v = bi[b]
                    Mw[b] = _f32(Mw[b] + wmf * (.5 * (v * v) + bon[b]))
                    pen = 1.0 if v < lo_thr else (1.0 - v * v if v < minbigram else (1.0 - v) * (1.0 - v))
                    a1 = _f32(NMw[b] + wnmf * pen)
                    a2 = _f32(NMw[b] + w03)
                    NMw[b] = a2 if uef[b] > 0.5 else a1
            for b in range(cn):
                score[b] = _f32(Mw[b] / (Mw[b] + NMw[b]))
            for b in range(cn):
                if score[b] > best:
                    best = score[b]
                    win[0] = c0 + b
                    nw = 1
                elif score[b] == best:
                    win[nw] = c0 + b
                    nw += 1
        for s in range(S):
            for c in mcodes[moff[s, i]:moff[s, i + 1]]:
                T[s, c] = 0
        if best < minscore:
            continue
        if n_out + nw > cap:
            while n_out + nw > cap:
                cap *= 2
            oi = _grow(oi, cap)
            oj = _grow(oj, cap)
            osc = _grow(osc, cap)
        for w in range(nw):
            b = win[w]
            oi[n_out] = i
            oj[n_out] = order[p0 + b] if b < nb else cand[b - nb]
            osc[n_out] = best
            n_out += 1
    return oi[:n_out], oj[:n_out], osc[:n_out]


def _ucs4(a):
    """A string array as a read-only (n, width) array of UCS4 code points (no copy if possible)."""
    a = np.asarray(a)
    if a.dtype.kind != "U":
        a = a.astype(str)
    a = np.ascontiguousarray(a)
    w = a.dtype.itemsize // 4
    v = np.zeros((a.shape[0], 1), np.uint32) if w == 0 else a.view(np.uint32).reshape(a.shape[0], w)
    v.flags.writeable = False
    return v


def run(master, using, vars, wmatch, wnomatch, required, minscore=0.6, minbigram=0.6):
    vars = list(vars)
    F = len(vars)
    nm, nu = len(master[vars[0]]), len(using[vars[0]])
    MB = [_ucs4(master[v]) for v in vars]
    UB = [_ucs4(using[v]) for v in vars]
    width = max(b.shape[1] for b in MB + UB) + F + 1
    P = _P if width <= _P.shape[1] else _multipliers(width)
    scored = np.array([f for f in range(F) if vars[f] not in required], np.int64)
    S = len(scored)

    # 1. master dictionaries; ids of the using values, field by field
    mid = np.empty((F, nm), np.int32)
    uid = np.empty((F, nu), np.int32)
    mlen = np.empty((F, nm), np.int64)
    nids = np.empty(F, np.int64)
    empty = np.full(F, -1, np.int64)                 # master-dictionary id of the empty string
    seen = np.zeros((F, 128), np.bool_)
    for f in range(F):
        mh = np.empty(nm, np.uint64)
        ids, rep, slots, st = _master_dict(MB[f], P, mlen[f], mh, seen[f])
        mid[f] = ids
        nids[f] = len(rep)
        e = np.flatnonzero(mlen[f] == 0)
        if len(e):
            empty[f] = ids[e[0]]
        bits = int(min(max(np.ceil(np.log2(16 * max(len(rep), 1))), 10), 28))
        tab, bloom = _probe_table(slots, rep, mh, bits)
        st |= _using_ids(UB[f], P, tab, bloom, bits, rep, MB[f], mlen[f], uid[f])
        if st:
            raise ValueError(f"{vars[f]}: only ASCII strings without NUL characters are supported")

    # 2. groups of required() values, perfect matches
    req = [f for f in range(F) if vars[f] in required]
    if len(req) == 1:
        gm, gu, ngroups = mid[req[0]], uid[req[0]], int(nids[req[0]])
    elif req:
        gm, gu, ngroups = _tuple_ids(mid, uid, np.array(req, np.int64), P)
    else:
        gm, gu, ngroups = np.zeros(nm, np.int32), np.zeros(nu, np.int32), 1
    ngroups = max(ngroups, 1)
    tm, tu, ntup = _tuple_ids(mid, uid, np.arange(F, dtype=np.int64), P)
    pi, pj, removed_m, removed_u = _perfect(tm, tu, ntup)

    # 3. eligible using records by group, then by the layout field: the or-block field whose
    #    agreeing records form the most candidate pairs (counted ignoring groups)
    elig = ~removed_u & (gu >= 0)
    items = np.flatnonzero(elig)
    mrest = np.flatnonzero(~removed_m)
    ob = list(range(S)) if F > 3 else []
    lk, K1, mk = np.zeros(nu, np.int64), 1, np.zeros(nm, np.int64)
    if ob:
        pairs = []
        for s in ob:
            f = scored[s]
            cnt = np.bincount(uid[f][elig & (uid[f] >= 0) & (uid[f] != empty[f])], minlength=nids[f])
            pairs.append(int(cnt[mid[f][mrest[mlen[f][mrest] > 0]]].sum()))
        lay = ob[int(np.argmax(pairs))]
        ob = [lay] + [s for s in ob if s != lay]
        f = scored[lay]
        lk, K1, mk = np.where(uid[f] >= 0, uid[f], nids[f]).astype(np.int64), int(nids[f]) + 1, mid[f].astype(np.int64)
    order, skey, gstart = _sort2(items, lk, K1, gu.astype(np.int64), ngroups)
    gend, gstart = gstart[1:], gstart[:-1]
    morder, _, _ = _sort2(mrest, mk, K1, gm.astype(np.int64), ngroups)

    # field alphabets: the characters of the master values, plus one code for all others
    A = np.empty(S, np.int64)
    remap = np.empty((S, 128), np.int64)
    for s in range(S):
        chars = np.flatnonzero(seen[scored[s]])
        A[s] = len(chars) + 1
        remap[s] = len(chars)
        remap[s, chars] = np.arange(len(chars))
    Amax2 = int((A ** 2).max()) if S else 1

    # 4. bigram codes of the master values; or-block candidate lists; the search
    moff = np.empty((S, nm + 1), np.int64)
    base = 0
    for s in range(S):
        moff[s, 0] = base
        np.cumsum(np.maximum(mlen[scored[s]] - 1, 0), out=moff[s, 1:])
        moff[s, 1:] += base
        base = int(moff[s, -1])
    mcodes = np.empty(max(base, 1), np.uint16)
    mpre = np.empty((S, nm), np.uint32)
    for s in range(S):
        _encode_master(MB[scored[s]], mlen[scored[s]], remap[s], A[s], s, mcodes, moff, mpre)
    ptrs, lsts, ptr_base = [np.zeros(1, np.int64)], [], [0]
    nbase = 0
    for s in ob[1:]:
        ptr, lst = _value_lists(uid[scored[s]][order], int(empty[scored[s]]), int(nids[scored[s]]))
        ptr_base.append(sum(len(x) for x in ptrs))
        ptrs.append(ptr + nbase)
        lsts.append(lst)
        nbase += len(lst)
    ptr_all = np.concatenate(ptrs)
    lst_all = np.concatenate(lsts) if lsts else np.zeros(0, np.int64)
    kind = np.full(F, -1, np.int64)
    kind[scored] = np.arange(S)
    biscale = .5 / (1 - minbigram) ** BISCALE_POWER
    wnm03 = np.array([w * 0.3 for w in wnomatch], np.float64)
    termM1 = np.array([w * (.5 * 1.0 ** 2 + biscale * (1.0 - minbigram) ** BISCALE_POWER) for w in wmatch])
    ubufs = tuple(UB[f] for f in scored) or (_ucs4(np.array([""])),)   # placeholder if all are required
    i, j, sc = _search(morder, gm.astype(np.int64), mid, mlen, mpre, moff, mcodes, kind, scored, ubufs,
                       uid, order, skey, gstart, gend, np.array(ob, np.int64), ptr_all,
                       np.array(ptr_base, np.int64), lst_all, A, remap, Amax2,
                       np.asarray(wmatch, np.float64), np.asarray(wnomatch, np.float64), wnm03, termM1,
                       float(minscore), float(minbigram), float(minbigram - .2), float(biscale),
                       BISCALE_POWER)
    i = np.concatenate([pi, i])
    j = np.concatenate([pj, j])
    s = np.concatenate([np.ones(len(pi)), sc])
    o = np.lexsort((j, i))
    return i[o], j[o], s[o], int(len(pi))
