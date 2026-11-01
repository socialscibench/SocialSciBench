# turbo-socialsci core, new in 0.2 (not from a benchmark candidate).
"""OLS with one absorbed fixed effect and one-way clustered standard errors, exactly and without an
n x k floating-point matrix.

With one fixed effect the partialled data are deviations from group means, so the partialled cross
products are sums over groups,
    X~'X~ = sum_g sum_{i in g} (x_i - xbar_g)(x_i - xbar_g)',
and need no iterative solver. Columns holding only 0 and 1 (dummies, in integer or boolean dtypes)
are stored as lists of each row's ones, so they cost work in proportion to their ones:
- dummy x dummy: the group's co-occurrence count minus S_a S_b / n_g (S: the group's counts of
  ones), which is exactly 0 for a dummy that is constant within its groups;
- dummy x other: the sum of the other column's within-group deviations over the dummy's ones;
- other columns (y first): two-pass within-group deviations.
Residuals are e_i = r_i - mean_g(r) with r = y - x'b, and the cluster scores sum (x_i - xbar_g) e_i.
The sample (missing values, singleton groups), the collinearity rule, the small-sample factor, df_a
and the fit statistics are hdfe_speed's (reghdfe's conventions).

Memory: the other columns of the estimation sample are copied once, in their own floating-point
type, with the dummies' lists (about 2 bytes per one plus 8 per row). With gather=True (the "speed"
variant) they are copied again in the fixed effect's group order, in float64, so the passes over
groups read memory in order.
"""
import numpy as np
import numba as nb

from ._ols import COLLINEAR_TOL, independent

BLOCK = 4096                 # rows per block when scanning the dummy columns
SPAN = 16                    # rows tested at once for being all 0
SCORE_PARTS_BYTES = 1 << 26  # per-part cluster-score sums up to this size, else parts own their clusters


@nb.njit(cache=True)
def _zero_one(a):
    """Whether every value of the float column a is 0 or 1 (stops at the first other value)."""
    for i in range(a.size):
        v = a[i]
        if v != 0.0 and v != 1.0:
            return False
    return True


@nb.njit(cache=True, parallel=True)
def _complete(Z, fe, cl, nparts):
    """1 for rows with a fixed-effect level, a cluster and finite values in every column of Z."""
    n = fe.size
    ok = np.empty(n, np.uint8)
    for p in nb.prange(nparts):
        lo, hi = n * p // nparts, n * (p + 1) // nparts
        for i in range(lo, hi):
            ok[i] = 1 if (fe[i] >= 0 and cl[i] >= 0) else 0
        for d in range(len(Z)):
            a = Z[d]
            for i in range(lo, hi):
                if not np.isfinite(a[i]):
                    ok[i] = 0
    return ok


@nb.njit(cache=True, parallel=True)
def _groups(fe, ok, G0, cl, Lc0, nparts):
    """Drops the rows of singleton groups (one pass suffices with one fixed effect); codes of the
    groups and clusters left; each group's cluster and whether every group lies in one cluster."""
    n0 = fe.size
    hist = np.zeros((nparts, G0), np.int32)
    for p in nb.prange(nparts):
        for i in range(n0 * p // nparts, n0 * (p + 1) // nparts):
            if ok[i]:
                hist[p, fe[i]] += 1
    gnew = np.full(G0, -1, np.int64)
    G = 0
    for g in range(G0):
        c = 0
        for p in range(nparts):
            c += hist[p, g]
        if c >= 2:
            gnew[g] = G
            G += 1
    seen = np.zeros((nparts, Lc0), np.uint8)
    pgcl = np.full((nparts, G), -1, np.int64)
    pn = np.zeros(nparts, np.int64)
    pnested = np.ones(nparts, np.bool_)
    for p in nb.prange(nparts):
        for i in range(n0 * p // nparts, n0 * (p + 1) // nparts):
            if ok[i]:
                g = gnew[fe[i]]
                if g < 0:
                    ok[i] = 0
                    continue
                pn[p] += 1
                c = cl[i]
                seen[p, c] = 1
                if pgcl[p, g] < 0:
                    pgcl[p, g] = c
                elif pgcl[p, g] != c:
                    pnested[p] = False
    nested = True
    for p in range(nparts):
        nested = nested and pnested[p]
    gcl = np.full(G, -1, np.int64)
    for g in range(G):
        for p in range(nparts):
            c = pgcl[p, g]
            if c >= 0:
                if gcl[g] < 0:
                    gcl[g] = c
                elif gcl[g] != c:
                    nested = False
    cnew = np.full(Lc0, -1, np.int64)
    Lc = 0
    for c in range(Lc0):
        any_ = False
        for p in range(nparts):
            if seen[p, c]:
                any_ = True
        if any_:
            cnew[c] = Lc
            Lc += 1
    for g in range(G):
        gcl[g] = cnew[gcl[g]]
    return gnew, G, cnew, Lc, gcl, nested, pn.sum()


@nb.njit(cache=True)
def _order_by_cluster(gnew, gcl, G, Lc):
    """Relabels the groups so that the groups of each cluster are consecutive (stable)."""
    start = np.zeros(Lc + 1, np.int64)
    for g in range(G):
        start[gcl[g] + 1] += 1
    for c in range(Lc):
        start[c + 1] += start[c]
    pos = np.empty(G, np.int64)
    for g in range(G):
        pos[g] = start[gcl[g]]
        start[gcl[g]] += 1
    out = np.empty(G, np.int64)
    for g in range(G):
        out[pos[g]] = gcl[g]
    for h in range(gnew.size):
        if gnew[h] >= 0:
            gnew[h] = pos[gnew[h]]
    return out


@nb.njit(cache=True)
def _compact_starts(ok, nblocks):
    """Index in the estimation sample of the first kept row of every block of BLOCK rows."""
    n0 = ok.size
    start = np.zeros(nblocks + 1, np.int64)
    for b in range(nblocks):
        c = 0
        for i in range(b * BLOCK, min(n0, (b + 1) * BLOCK)):
            c += ok[i]
        start[b + 1] = start[b] + c
    return start


@nb.njit(cache=True, parallel=True)
def _compact(Z, ok, cstart, fe, gnew, cl, cnew, Zc, gc, cc):
    """The kept rows' columns (rows of Zc), group codes and cluster codes, in the original order."""
    n0 = ok.size
    nblocks = cstart.size - 1
    kd1 = len(Z)
    for b in nb.prange(nblocks):
        lo, hi = b * BLOCK, min(n0, (b + 1) * BLOCK)
        for d in range(kd1):
            a = Z[d]
            c = cstart[b]
            for i in range(lo, hi):
                if ok[i]:
                    Zc[c, d] = a[i]
                    c += 1
        c = cstart[b]
        for i in range(lo, hi):
            if ok[i]:
                gc[c] = gnew[fe[i]]
                cc[c] = cnew[cl[i]]
                c += 1


@nb.njit(cache=True, parallel=True)
def _dummy_counts(D, ok, cstart, cnt):
    """Ones per kept row over the dummy columns, and per column the bitwise OR of its values (> 1:
    not a dummy). Runs of SPAN rows that are all 0 are skipped after one test."""
    n0 = ok.size
    kb = len(D)
    nblocks = cstart.size - 1
    bor = np.zeros((nblocks, kb), np.uint8)
    for b in nb.prange(nblocks):
        lo, hi = b * BLOCK, min(n0, (b + 1) * BLOCK)
        local = np.zeros(hi - lo, np.uint16)
        for j in range(kb):
            a = D[j]
            o = np.uint8(0)
            i = lo
            while i + SPAN <= hi:
                x = np.uint8(0)
                for t in range(SPAN):
                    x |= a[i + t]
                if x != 0:
                    o |= x
                    for t in range(SPAN):
                        local[i - lo + t] += a[i + t]
                i += SPAN
            while i < hi:
                o |= a[i]
                local[i - lo] += a[i]
                i += 1
            bor[b, j] = o
        c = cstart[b]
        for i in range(lo, hi):
            if ok[i]:
                cnt[c] = local[i - lo]
                c += 1
    orv = np.zeros(kb, np.uint8)
    for b in range(nblocks):
        for j in range(kb):
            orv[j] |= bor[b, j]
    return orv


@nb.njit(cache=True, parallel=True)
def _dummy_fill(D, ok, cstart, ptr, ids):
    """Each kept row's dummy columns with a one, in column order (CSR over the estimation sample);
    runs of SPAN zeros are skipped after one test."""
    n0 = ok.size
    kb = len(D)
    nblocks = cstart.size - 1
    for b in nb.prange(nblocks):
        lo, hi = b * BLOCK, min(n0, (b + 1) * BLOCK)
        pos = np.full(hi - lo, -1, np.int64)
        c = cstart[b]
        for i in range(lo, hi):
            if ok[i]:
                pos[i - lo] = ptr[c]
                c += 1
        for j in range(kb):
            a = D[j]
            jj = np.uint16(j)
            i = lo
            while i + SPAN <= hi:
                x = np.uint8(0)
                for t in range(SPAN):
                    x |= a[i + t]
                if x != 0:
                    for r in range(i, i + SPAN):
                        if a[r] != 0:
                            q = pos[r - lo]
                            if q >= 0:
                                ids[q] = jj
                                pos[r - lo] = q + 1
                i += SPAN
            for r in range(i, hi):
                if a[r] != 0:
                    q = pos[r - lo]
                    if q >= 0:
                        ids[q] = jj
                        pos[r - lo] = q + 1


@nb.njit(cache=True, parallel=True)
def _group_index(gc, G, nparts):
    """Counting sort of the estimation sample by group: offsets and rows (stable)."""
    n = gc.size
    hist = np.zeros((nparts, G), np.int64)
    for p in nb.prange(nparts):
        for i in range(n * p // nparts, n * (p + 1) // nparts):
            hist[p, gc[i]] += 1
    off = np.empty(G + 1, np.int64)
    start = np.empty((nparts, G), np.int64)
    acc = 0
    for g in range(G):
        off[g] = acc
        for p in range(nparts):
            start[p, g] = acc
            acc += hist[p, g]
    off[G] = acc
    src = np.empty(n, np.int32)
    for p in nb.prange(nparts):
        pos = start[p]
        for i in range(n * p // nparts, n * (p + 1) // nparts):
            g = gc[i]
            src[pos[g]] = i
            pos[g] += 1
    return off, src


@nb.njit(cache=True, parallel=True)
def _gather(Zc, ptr, ids, cc, src, nparts):
    """The sample in group order: float64 columns, the dummy lists and the cluster codes."""
    n = src.size
    kd1 = Zc.shape[1]
    Zg = np.empty((n, kd1))
    cg = np.empty(n, np.int32)
    cntg = np.empty(n + 1, np.int64)
    cntg[0] = 0
    for p in nb.prange(nparts):
        for r in range(n * p // nparts, n * (p + 1) // nparts):
            i = src[r]
            for d in range(kd1):
                Zg[r, d] = Zc[i, d]
            cg[r] = cc[i]
            cntg[r + 1] = ptr[i + 1] - ptr[i]
    for r in range(n):
        cntg[r + 1] += cntg[r]
    idsg = np.empty(ids.size, ids.dtype)
    for p in nb.prange(nparts):
        for r in range(n * p // nparts, n * (p + 1) // nparts):
            i = src[r]
            q = cntg[r]
            for t in range(ptr[i], ptr[i + 1]):
                idsg[q] = ids[t]
                q += 1
    return Zg, cntg, idsg, cg


@nb.njit(cache=True, parallel=True)
def _cross(Zc, ptr, ids, src, off, gpart, kb, XXp):
    """Per part of the groups, the lower triangle of the partialled cross products of [y, x]
    (other columns first, then the dummies)."""
    kd1 = Zc.shape[1]
    ident = src.size == 0
    for p in nb.prange(gpart.size - 1):
        XX = XXp[p]
        zbar = np.zeros(kd1)
        w = np.zeros(kd1)
        sa = np.zeros(max(kb, 1), np.int64)
        act = np.empty(max(kb, 1), np.int64)
        for g in range(gpart[p], gpart[p + 1]):
            r0, r1 = off[g], off[g + 1]
            ng = r1 - r0
            for d in range(kd1):
                zbar[d] = 0.0
            for r in range(r0, r1):
                i = r if ident else src[r]
                for d in range(kd1):
                    zbar[d] += Zc[i, d]
            for d in range(kd1):
                zbar[d] /= ng
            na = 0
            for r in range(r0, r1):
                i = r if ident else src[r]
                for d in range(kd1):
                    w[d] = Zc[i, d] - zbar[d]
                for d in range(kd1):
                    wd = w[d]
                    for e in range(d + 1):
                        XX[d, e] += wd * w[e]
                q0, q1 = ptr[i], ptr[i + 1]
                for q in range(q0, q1):
                    a = np.int64(ids[q])
                    if sa[a] == 0:
                        act[na] = a
                        na += 1
                    sa[a] += 1
                    ca = kd1 + a
                    for d in range(kd1):
                        XX[ca, d] += w[d]
                    for q2 in range(q0, q + 1):              # ids ascending: ids[q2] <= a
                        XX[ca, kd1 + np.int64(ids[q2])] += 1.0
            for u in range(na):
                a = act[u]
                fa = sa[a] / ng
                for v in range(na):
                    b = act[v]
                    if b <= a:
                        XX[kd1 + a, kd1 + b] -= fa * sa[b]
            for u in range(na):
                sa[act[u]] = 0


@nb.njit(cache=True, parallel=True)
def _scores(Zc, ptr, ids, src, cc, off, gpart, gcl, nested, own, bd, bb, kb, Sp, rssp):
    """Residuals e = r - mean_g(r), r = y - x'b, their sum of squares, and the cluster sums of
    (x - xbar_g) e (per part, or written directly when each part owns its clusters)."""
    kd1 = Zc.shape[1]
    kd = kd1 - 1
    ident = src.size == 0
    for p in nb.prange(gpart.size - 1):
        S = Sp[0] if own else Sp[p]
        zbar = np.zeros(kd1)
        sa = np.zeros(max(kb, 1), np.int64)
        act = np.empty(max(kb, 1), np.int64)
        rss = 0.0
        for g in range(gpart[p], gpart[p + 1]):
            r0, r1 = off[g], off[g + 1]
            ng = r1 - r0
            for d in range(kd1):
                zbar[d] = 0.0
            na = 0
            sr = 0.0
            for r in range(r0, r1):
                i = r if ident else src[r]
                ri = Zc[i, 0]
                for d in range(1, kd1):
                    v = Zc[i, d]
                    zbar[d] += v
                    ri -= bd[d - 1] * v
                for q in range(ptr[i], ptr[i + 1]):
                    a = np.int64(ids[q])
                    ri -= bb[a]
                    if sa[a] == 0:
                        act[na] = a
                        na += 1
                    sa[a] += 1
                sr += ri
            m = sr / ng
            for d in range(1, kd1):
                zbar[d] /= ng
            se = 0.0
            for r in range(r0, r1):
                i = r if ident else src[r]
                ri = Zc[i, 0]
                for d in range(1, kd1):
                    ri -= bd[d - 1] * Zc[i, d]
                for q in range(ptr[i], ptr[i + 1]):
                    ri -= bb[np.int64(ids[q])]
                e = ri - m
                rss += e * e
                se += e
                c = cc[i]
                for d in range(1, kd1):
                    S[c, d - 1] += (Zc[i, d] - zbar[d]) * e
                for q in range(ptr[i], ptr[i + 1]):
                    S[c, kd + np.int64(ids[q])] += e
                if not nested:
                    for u in range(na):
                        a = act[u]
                        S[c, kd + a] -= sa[a] / ng * e
            if nested:
                c = gcl[g]
                for u in range(na):
                    a = act[u]
                    S[c, kd + a] -= sa[a] / ng * se
            for u in range(na):
                sa[act[u]] = 0
        rssp[p] = rss


@nb.njit(cache=True)
def _centered_ss(Zc, n):
    """Sums of squared deviations from the sample mean of each column of Zc (two passes)."""
    kd1 = Zc.shape[1]
    mean = np.zeros(kd1)
    for i in range(n):
        for d in range(kd1):
            mean[d] += Zc[i, d]
    for d in range(kd1):
        mean[d] /= max(n, 1)
    ss = np.zeros(kd1)
    for i in range(n):
        for d in range(kd1):
            v = Zc[i, d] - mean[d]
            ss[d] += v * v
    return ss


def _parts(off, nparts, allowed=None):
    """Group ranges of about equal numbers of rows; cut only at the groups in `allowed` if given."""
    G = off.size - 1
    targets = off[-1] * np.arange(1, nparts) / nparts
    if allowed is None:
        cuts = np.searchsorted(off, targets)
    else:
        cuts = allowed[np.minimum(np.searchsorted(off[allowed], targets), allowed.size - 1)]
    return np.unique(np.concatenate([[0], np.clip(cuts, 0, G), [G]])).astype(np.int64)


def run(Z, D, fe, G0, cl, Lc0, xmap, collinear_tol=COLLINEAR_TOL, gather=True):
    """Z: typed list of the outcome and the other regressors (one float dtype; Z[0] = y); D: typed
    list of uint8 columns expected to hold 0/1 (checked; a column that does not is reported in
    "not_dummies" and nothing is computed); fe, cl: codes (-1 missing) with G0 and Lc0 levels;
    xmap[j]: the column of regressor j (caller's order) in [Z[1:], D]."""
    nthreads = nb.get_num_threads()
    n0 = fe.size
    nparts = max(1, min(nthreads, n0 // 65536 + 1))
    ok = _complete(Z, fe, cl, nparts)
    n_complete = int(ok.sum())
    gnew, G, cnew, Lc, gcl, nested, n = _groups(fe, ok, G0, cl, Lc0, nparts)
    n = int(n)
    kd1, kb = len(Z), len(D)
    kd = kd1 - 1
    k = kd + kb
    if n < 2 or G < 1:
        return dict(empty=True, nobs=int(n), n_complete=n_complete)
    if nested:
        gcl = _order_by_cluster(gnew, gcl, G, Lc)
    nblocks = (n0 + BLOCK - 1) // BLOCK
    cstart = _compact_starts(ok, nblocks)
    Zc = np.empty((n, kd1), dtype=Z[0].dtype)
    gc = np.empty(n, np.int32)
    cc = np.empty(n, np.int32)
    _compact(Z, ok, cstart, fe, gnew, cl, cnew, Zc, gc, cc)
    if kb:
        ptr = np.zeros(n + 1, np.int64)
        orv = _dummy_counts(D, ok, cstart, ptr[1:])         # counts, then their running sums
        if (orv > 1).any():
            return dict(not_dummies=np.flatnonzero(orv > 1), n_complete=n_complete)
        np.cumsum(ptr[1:], out=ptr[1:])
        ids = np.empty(int(ptr[-1]), np.uint16)
        _dummy_fill(D, ok, cstart, ptr, ids)
        ones = np.bincount(ids, minlength=kb).astype(np.int64)
    else:
        ones = np.zeros(0, np.int64)
        ptr = np.zeros(n + 1, np.int64)
        ids = np.zeros(0, np.uint16)
    del ok
    off, src = _group_index(gc, G, max(1, min(nthreads, n // 262144 + 1)))
    del gc

    # sums of squares before partialling (collinearity tolerance) and the total sum of squares
    ss_z = _centered_ss(Zc, n)
    ss_d = ones - ones.astype(np.float64) ** 2 / n
    if gather:
        Zw, ptrw, idsw, ccw = _gather(Zc, ptr, ids, cc, src, max(1, min(nthreads, n // 65536 + 1)))
        del Zc, ptr, ids, cc
        srcw = np.zeros(0, np.int32)
    else:
        Zw, ptrw, idsw, ccw, srcw = Zc, ptr, ids, cc, src

    # partialled cross products, per part of the groups
    nparts = max(1, min(4 * nthreads, G, n // 4096 + 1))
    gpart = _parts(off, nparts)
    K = kd1 + kb
    XXp = np.zeros((gpart.size - 1, K, K))
    _cross(Zw, ptrw, idsw, srcw, off, gpart, kb, XXp)
    L = XXp.sum(axis=0)
    del XXp
    XX = np.tril(L) + np.tril(L, -1).T
    # caller's order: [y, x_1..x_k]
    order = np.concatenate([[0], 1 + np.asarray(xmap, dtype=np.int64)])
    W = XX[np.ix_(order, order)]
    ss = np.concatenate([ss_z[1:], ss_d])[np.asarray(xmap, dtype=np.int64)]
    keep_x = independent(W[1:, 1:], ss, collinear_tol)
    kept = np.flatnonzero(keep_x)
    XtX_inv = np.linalg.inv(W[1:, 1:][np.ix_(kept, kept)]) if kept.size else np.zeros((0, 0))
    b_kept = XtX_inv @ W[1:, 0][kept]
    b_all = np.zeros(k)
    b_all[kept] = b_kept
    b_int = np.zeros(k)                                  # coefficients in [Z[1:], D] order
    b_int[np.asarray(xmap, dtype=np.int64)] = b_all

    # residuals and cluster scores
    own = False
    if (gpart.size - 1) * Lc * k * 8 > SCORE_PARTS_BYTES:
        if not nested:
            nparts_s = max(1, min(nthreads, SCORE_PARTS_BYTES // max(Lc * k * 8, 1)))
            gpart = _parts(off, nparts_s)
        else:                                            # parts cut at cluster boundaries
            first = np.flatnonzero(np.diff(np.concatenate([[-1], gcl])) != 0)
            gpart = _parts(off, max(1, min(4 * nthreads, Lc)), np.append(first, G))
            own = True
    Sp = np.zeros((1 if own else gpart.size - 1, Lc, k))
    rssp = np.zeros(gpart.size - 1)
    _scores(Zw, ptrw, idsw, srcw, ccw, off, gpart, gcl, nested, own, b_int[:kd], b_int[kd:], kb, Sp, rssp)
    S_int = Sp.sum(axis=0)
    del Sp
    scores = S_int[:, np.asarray(xmap, dtype=np.int64)]
    df_a = 0 if nested else G - 1
    Kdf = kept.size + 1 + df_a
    q = (n - 1) / (n - Kdf) * Lc / (Lc - 1) if Lc > 1 else np.nan
    meat = scores[:, kept].T @ scores[:, kept]
    beta = np.full(k, np.nan)
    beta[kept] = b_kept
    vcov = np.full((k, k), np.nan)
    vcov[np.ix_(kept, kept)] = q * XtX_inv @ meat @ XtX_inv
    rss = float(rssp.sum())
    tss = float(ss_z[0])
    r2 = 1.0 - rss / tss if tss > 0 else float("nan")
    yyw = float(W[0, 0])
    r2_within = 1.0 - rss / yyw if yyw > 0 else float("nan")
    return dict(beta=beta, vcov=vcov, nobs=int(n), df_a=int(df_a), n_clusters=int(Lc), kept=keep_x,
                dropped=np.flatnonzero(~keep_x), rss=rss, tss=tss, r2=float(r2), r2_within=float(r2_within),
                n_complete=n_complete)
