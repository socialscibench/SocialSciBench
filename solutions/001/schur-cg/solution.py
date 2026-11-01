"""Task 001 candidate "schur-cg": OLS with high-dimensional fixed effects and one-way clustered
standard errors, as computed by reference.run (reghdfe conventions).

Method
------
1. Singleton rows are peeled until every level of every fixed effect has two rows (the fixed
   point the reference's repeated passes reach), and levels are recoded.
2. The fixed effects are partialled out of y and all regressors together. A fixed effect whose
   levels are unions of another's (year within technology-subclass x year) is skipped: its
   indicators lie in the other's span. The normal equations for the fixed-effect coefficients
   are solved with one fixed effect (the one with the most levels, e.g. the inventor)
   eliminated exactly: its coefficients are group means given the others', so the remaining
   system is the Schur complement
       S = A_oo - N_ob C_b^-1 N_bo
   over the levels of the other fixed effects (A_oo: their cross-tabulation; N_bo: counts of
   rows of each eliminated level at each other level; C_b: the eliminated levels' sizes).
   S is formed densely for small systems, assembled as a sparse matrix when it has few entries,
   and otherwise applied through A_oo and N_bo, forming and scattering back each eliminated
   level's row in turn. Conjugate gradients solve S for all columns at once (each sparse entry
   updates a row of all columns), preconditioned block by block: exact pseudo-inverses of small
   blocks, a randomized approximate Cholesky factorization of large sparse blocks' Laplacians
   (Kyng & Sachdeva 2016; Gao, Kyng & Spielman 2023), otherwise the diagonal.
3. The demeaned data are the rows minus their fixed-effect coefficients; OLS, the cluster
   scores and the sandwich follow, with the small-sample factor and df_a computed as in the
   reference (nesting in the cluster variable, connected components of the first two
   non-nested fixed effects, nesting among the others).

Threads: Numba parallel loops (NUMBA_NUM_THREADS, set by the evaluator's --threads) and BLAS.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import numba as nb

TOL = 1e-9           # relative residual of the reduced normal equations, per column
MAXITER = 100_000
DENSE_MAX = 2000     # blocks with at most this many levels are inverted exactly (dense)
DENSE_S = 4000       # reduced systems with at most this many levels are formed as dense matrices
SMALL_BLOCK = 512    # rows of blocks this small are stored densely in the reduced operator
SPARSE_PAIRS = 64    # larger blocks get approximate Cholesky if their Laplacian is this sparse


# ----------------------------------------------------------------------------------------------
# Recoding and singletons
# ----------------------------------------------------------------------------------------------

@nb.njit(parallel=True)
def _dense_codes(ids, size):
    seen = np.zeros(size, np.int64)
    for i in nb.prange(ids.size):
        seen[ids[i]] = 1
    nl = 0
    for v in range(size):
        if seen[v]:
            seen[v] = nl
            nl += 1
    out = np.empty(ids.size, np.int64)
    for i in nb.prange(ids.size):
        out[i] = seen[ids[i]]
    return out, nl


def _recode(ids):
    """Codes 0..L-1 of arbitrary integer ids (L levels)."""
    ids = np.ascontiguousarray(ids)
    if ids.size == 0:
        return np.zeros(0, np.int64), 0
    lo, hi = int(ids.min()), int(ids.max())
    if lo >= 0 and hi < 4 * ids.size + 4096:
        return _dense_codes(ids, hi + 1)
    u, inv = np.unique(ids, return_inverse=True)
    return inv.astype(np.int64), u.size


@nb.njit
def _peel(codes, nlev):
    """Rows kept after repeatedly dropping rows that are alone in a level of some fixed effect
    (the fixed point the reference's passes reach; a sweep also drops rows that become
    singletons during it)."""
    n, G = codes.shape
    start = np.zeros(G + 1, np.int64)
    for g in range(G):
        start[g + 1] = start[g] + nlev[g]
    counts = np.zeros(start[G], np.int32)          # all fixed effects' level counts, compact
    for i in range(n):
        for g in range(G):
            counts[start[g] + codes[i, g]] += 1
    keep = np.ones(n, np.bool_)
    changed = True
    while changed:
        changed = False
        for i in range(n):
            if not keep[i]:
                continue
            single = False
            for g in range(G):
                if counts[start[g] + codes[i, g]] == 1:
                    single = True
                    break
            if single:
                keep[i] = False
                changed = True
                for g in range(G):
                    counts[start[g] + codes[i, g]] -= 1
    return keep


@nb.njit(parallel=True)
def _group_rows_par(code, L, nparts):
    n = code.size
    hist = np.zeros((nparts, L + 1), np.int64)
    for p in nb.prange(nparts):
        for i in range(n * p // nparts, n * (p + 1) // nparts):
            hist[p, code[i]] += 1
    off = np.zeros(L + 1, np.int64)
    start = np.empty((nparts, L), np.int64)
    acc = 0
    for l in range(L):
        off[l] = acc
        for p in range(nparts):
            start[p, l] = acc
            acc += hist[p, l]
    off[L] = acc
    rows = np.empty(n, np.int64)
    for p in nb.prange(nparts):
        pos = start[p]
        for i in range(n * p // nparts, n * (p + 1) // nparts):
            c = code[i]
            rows[pos[c]] = i
            pos[c] += 1
    return off, rows


@nb.njit
def _group_rows_serial(code, L):
    n = code.size
    off = np.zeros(L + 1, np.int64)
    for i in range(n):
        off[code[i] + 1] += 1
    for l in range(L):
        off[l + 1] += off[l]
    rows = np.empty(n, np.int64)
    pos = off[:-1].copy()
    for i in range(n):
        rows[pos[code[i]]] = i
        pos[code[i]] += 1
    return off, rows


def _group_rows(code, L):
    """Counting sort of rows by level (stable): (offsets, rows)."""
    nparts = min(nb.get_num_threads(), code.size // 262144 + 1)
    if nparts <= 1:
        return _group_rows_serial(code, L)
    return _group_rows_par(code, L, nparts)


# ----------------------------------------------------------------------------------------------
# Structures of the reduced system
# ----------------------------------------------------------------------------------------------

@nb.njit(parallel=True)
def _incidence(off_b, rows_b, ocodes, Lo):
    """For every level of the eliminated fixed effect: its distinct levels of the other fixed
    effects (stacked index) with row counts (CSR)."""
    Lb = off_b.size - 1
    nG = ocodes.shape[1]
    nnz = np.zeros(Lb + 1, np.int64)
    nparts = max(1, min(Lb, 4 * nb.get_num_threads()))
    for part in nb.prange(nparts):
        mark = np.full(Lo, -1, np.int64)
        for b in range(Lb * part // nparts, Lb * (part + 1) // nparts):
            c = 0
            for t in range(off_b[b], off_b[b + 1]):
                r = rows_b[t]
                for g in range(nG):
                    o = ocodes[r, g]
                    if mark[o] != b:
                        mark[o] = b
                        c += 1
            nnz[b + 1] = c
    for b in range(Lb):
        nnz[b + 1] += nnz[b]
    idx = np.empty(nnz[Lb], np.int64)
    cnt = np.zeros(nnz[Lb], np.float64)
    for part in nb.prange(nparts):
        mark = np.full(Lo, -1, np.int64)
        slot = np.zeros(Lo, np.int64)
        for b in range(Lb * part // nparts, Lb * (part + 1) // nparts):
            p = nnz[b]
            for t in range(off_b[b], off_b[b + 1]):
                r = rows_b[t]
                for g in range(nG):
                    o = ocodes[r, g]
                    if mark[o] != b:
                        mark[o] = b
                        slot[o] = p
                        idx[p] = o
                        p += 1
                    cnt[slot[o]] += 1.0
    return nnz, idx, cnt


@nb.njit
def _transpose(ptr, idx, val, ncols):
    nrow = ptr.size - 1
    tp = np.zeros(ncols + 1, np.int64)
    for t in range(ptr[nrow]):
        tp[idx[t] + 1] += 1
    for c in range(ncols):
        tp[c + 1] += tp[c]
    ti = np.empty(ptr[nrow], np.int64)
    tv = np.empty(ptr[nrow], np.float64)
    pos = tp[:-1].copy()
    for r in range(nrow):
        for t in range(ptr[r], ptr[r + 1]):
            c = idx[t]
            ti[pos[c]] = r
            tv[pos[c]] = val[t]
            pos[c] += 1
    return tp, ti, tv


@nb.njit(parallel=True)
def _crosstab(ocodes, Lo, offs, groups_off, groups_rows):
    """Cross-tabulation A_oo of the other fixed effects (stacked levels), CSR including the
    diagonal counts. groups_* hold the rows of every stacked level."""
    n, nG = ocodes.shape
    nnz = np.zeros(Lo + 1, np.int64)
    nparts = max(1, min(Lo, 4 * nb.get_num_threads()))
    for part in nb.prange(nparts):
        mark = np.full(Lo, -1, np.int64)
        for l in range(Lo * part // nparts, Lo * (part + 1) // nparts):
            c = 1
            mark[l] = l
            for t in range(groups_off[l], groups_off[l + 1]):
                r = groups_rows[t]
                for g in range(nG):
                    o = ocodes[r, g]
                    if mark[o] != l:
                        mark[o] = l
                        c += 1
            nnz[l + 1] = c
    for l in range(Lo):
        nnz[l + 1] += nnz[l]
    idx = np.empty(nnz[Lo], np.int64)
    val = np.zeros(nnz[Lo], np.float64)
    for part in nb.prange(nparts):
        mark = np.full(Lo, -1, np.int64)
        slot = np.zeros(Lo, np.int64)
        for l in range(Lo * part // nparts, Lo * (part + 1) // nparts):
            p = nnz[l]
            idx[p] = l
            mark[l] = l
            slot[l] = p
            p += 1
            for t in range(groups_off[l], groups_off[l + 1]):
                r = groups_rows[t]
                for g in range(nG):
                    o = ocodes[r, g]
                    if mark[o] != l:
                        mark[o] = l
                        slot[o] = p
                        idx[p] = o
                        p += 1
                    if o == l:
                        continue
                    val[slot[o]] += 1.0
            val[nnz[l]] = groups_off[l + 1] - groups_off[l]
    return nnz, idx, val


# ----------------------------------------------------------------------------------------------
# Group sums, the reduced operator and conjugate gradients (all columns at once)
# ----------------------------------------------------------------------------------------------

@nb.njit(parallel=True)
def _group_sums(y, X, idx, off, rows, out):
    """Per level, the sums of y and of every regressor over its rows (kept rows idx)."""
    L = off.size - 1
    k = X.shape[1]
    for l in nb.prange(L):
        for j in range(k + 1):
            out[l, j] = 0.0
        for t in range(off[l], off[l + 1]):
            r = idx[rows[t]]
            out[l, 0] += y[r]
            for j in range(k):
                out[l, 1 + j] += X[r, j]


@nb.njit(parallel=True)
def _spmm(ptr, idx, val, X, Y, beta):
    """Y = beta * Y + A X for CSR A and row-major X, Y (rows x K)."""
    nrow = ptr.size - 1
    K = X.shape[1]
    for r in nb.prange(nrow):
        if beta == 0.0:
            for k in range(K):
                Y[r, k] = 0.0
        else:
            for k in range(K):
                Y[r, k] *= beta
        for t in range(ptr[r], ptr[r + 1]):
            c = idx[t]
            a = val[t]
            for k in range(K):
                Y[r, k] += a * X[c, k]


@nb.njit(parallel=True)
def _scale_rows(T, s):
    for r in nb.prange(T.shape[0]):
        for k in range(T.shape[1]):
            T[r, k] *= s[r]


@nb.njit
def _order_levels(ptr, idx, Lo):
    """Eliminated levels sorted by their smallest other level (counting sort), so that levels
    processed one after another share rows of the operand. Every level has an entry."""
    Lb = ptr.size - 1
    key = np.empty(Lb, np.int64)
    pos = np.zeros(Lo + 1, np.int64)
    for b in range(Lb):
        m = Lo - 1
        for t in range(ptr[b], ptr[b + 1]):
            if idx[t] < m:
                m = idx[t]
        key[b] = m
        pos[m + 1] += 1
    for l in range(Lo):
        pos[l + 1] += pos[l]
    order = np.empty(Lb, np.int64)
    for b in range(Lb):
        order[pos[key[b]]] = b
        pos[key[b]] += 1
    return order


@nb.njit(parallel=True)
def _permute_rows(ptr, idx, val, order):
    """The CSR rows order[0], order[1], ..."""
    nptr = np.empty(order.size + 1, np.int64)
    nptr[0] = 0
    for i in range(order.size):
        nptr[i + 1] = nptr[i] + ptr[order[i] + 1] - ptr[order[i]]
    nidx = np.empty(idx.size, idx.dtype)
    nval = np.empty(val.size, val.dtype)
    for i in nb.prange(order.size):
        s = nptr[i]
        for t in range(ptr[order[i]], ptr[order[i] + 1]):
            nidx[s] = idx[t]
            nval[s] = val[t]
            s += 1
    return nptr, nidx, nval


@nb.njit
def _fused_range(ptr, idx, cnt, cinv, P, Y, lo, hi):
    """Y -= N_ob C^-1 N_bo P over the eliminated levels [lo, hi): each level's row of
    C^-1 N_bo P is formed and scattered back at once, so those rows are never stored."""
    K = P.shape[1]
    t = np.empty(K)
    for b in range(lo, hi):
        for k in range(K):
            t[k] = 0.0
        for e in range(ptr[b], ptr[b + 1]):
            c = idx[e]
            w = cnt[e]
            for k in range(K):
                t[k] += w * P[c, k]
        s = cinv[b]
        for k in range(K):
            t[k] *= s
        for e in range(ptr[b], ptr[b + 1]):
            c = idx[e]
            w = cnt[e]
            for k in range(K):
                Y[c, k] -= w * t[k]


@nb.njit(parallel=True)
def _fused(ptr, idx, cnt, cinv, P, Y, bounds, Yp):
    """Y -= N_ob C^-1 N_bo P, the eliminated levels split into parts by entries: part 0
    scatters into Y, the others into their own buffers Yp, which are then added to Y."""
    nparts = bounds.size - 1
    for part in nb.prange(nparts):
        if part == 0:
            _fused_range(ptr, idx, cnt, cinv, P, Y, bounds[0], bounds[1])
        else:
            Z = Yp[part - 1]
            for r in range(Z.shape[0]):
                for k in range(Z.shape[1]):
                    Z[r, k] = 0.0
            _fused_range(ptr, idx, cnt, cinv, P, Z, bounds[part], bounds[part + 1])
    for r in nb.prange(Y.shape[0]):
        for p in range(nparts - 1):
            for k in range(Y.shape[1]):
                Y[r, k] += Yp[p, r, k]


@nb.njit(parallel=True)
def _colsums2(A, B, out):
    """out[k] = sum_r A[r, k] * B[r, k], in parallel partial sums."""
    n, K = A.shape
    nparts = min(max(n // 4096, 1), 64)
    part_s = np.zeros((nparts, K))
    for part in nb.prange(nparts):
        lo = n * part // nparts
        hi = n * (part + 1) // nparts
        for r in range(lo, hi):
            for k in range(K):
                part_s[part, k] += A[r, k] * B[r, k]
    for k in range(K):
        s = 0.0
        for part in range(nparts):
            s += part_s[part, k]
        out[k] = s


@nb.njit(parallel=True)
def _axpy_cols(X, a, P, active):
    for r in nb.prange(X.shape[0]):
        for k in range(X.shape[1]):
            if active[k]:
                X[r, k] += a[k] * P[r, k]


@nb.njit(parallel=True)
def _precond(R, dinv, Z):
    for r in nb.prange(R.shape[0]):
        d = dinv[r]
        for k in range(R.shape[1]):
            Z[r, k] = R[r, k] * d


@nb.njit(parallel=True)
def _precond_range(R, dinv, Z, lo, hi):
    for r in nb.prange(lo, hi):
        d = dinv[r]
        for k in range(R.shape[1]):
            Z[r, k] = R[r, k] * d


@nb.njit(parallel=True)
def _update_p(P, Z, b, active):
    for r in nb.prange(P.shape[0]):
        for k in range(P.shape[1]):
            if active[k]:
                P[r, k] = Z[r, k] + b[k] * P[r, k]


def _pcg(B, apply_S, precond, tol, maxiter):
    """Preconditioned conjugate gradients for S X = B, every column at once (from X = 0); a
    column stops when its residual falls below tol times its right-hand side."""
    Lo, K = B.shape
    X = np.zeros((Lo, K))
    R = B.copy()
    Z = np.empty_like(B)
    precond(R, Z)
    P = Z.copy()
    Y = np.empty_like(B)
    rz = np.empty(K)
    _colsums2(R, Z, rz)
    b2 = np.empty(K)
    _colsums2(B, B, b2)
    target = (tol ** 2) * b2
    active = b2 > 0
    pap = np.empty(K)
    rr = np.empty(K)
    rz_new = np.empty(K)
    it = 0
    while active.any() and it < maxiter:
        it += 1
        apply_S(P, Y)
        _colsums2(P, Y, pap)
        alpha = np.where(active & (pap > 0), rz / np.where(pap > 0, pap, 1.0), 0.0)
        _axpy_cols(X, alpha, P, active)
        _axpy_cols(R, -alpha, Y, active)
        _colsums2(R, R, rr)
        active &= rr > target
        precond(R, Z)
        _colsums2(R, Z, rz_new)
        beta = np.where(active, rz_new / np.where(rz > 0, rz, 1.0), 0.0)
        rz[:] = rz_new
        _update_p(P, Z, beta, active)
    return X, it


@nb.njit(parallel=True)
def _residuals(y, X, idx, codes, alpha_b, bcol, alpha_o, ocols, offs, out):
    """Demeaned rows: [y, X] of the kept rows minus their fixed-effect coefficients."""
    n = idx.size
    k = X.shape[1]
    nG = ocols.size
    for r in nb.prange(n):
        i = idx[r]
        b = codes[r, bcol]
        out[r, 0] = y[i] - alpha_b[b, 0]
        for j in range(k):
            out[r, 1 + j] = X[i, j] - alpha_b[b, 1 + j]
        for g in range(nG):
            o = offs[g] + codes[r, ocols[g]]
            for j in range(k + 1):
                out[r, j] -= alpha_o[o, j]


@nb.njit(parallel=True)
def _cluster_scores(W, e, off, rows, out):
    """Cluster sums of x_r e_r (x_r = W[r, 1:])."""
    G = off.size - 1
    k = W.shape[1] - 1
    for g in nb.prange(G):
        for j in range(k):
            out[g, j] = 0.0
        for t in range(off[g], off[g + 1]):
            r = rows[t]
            er = e[r]
            for j in range(k):
                out[g, j] += W[r, 1 + j] * er


@nb.njit
def _diag_correction(diag, ptr, idx, cnt, cinv):
    for b in range(ptr.size - 1):
        for t in range(ptr[b], ptr[b + 1]):
            diag[idx[t]] -= cnt[t] * cnt[t] * cinv[b]


# ----------------------------------------------------------------------------------------------
# Preconditioner: approximate Cholesky of a block's graph Laplacian (Kyng & Sachdeva 2016; the
# "AC" sampling of Gao, Kyng & Spielman 2023), dense pseudo-inverse for small blocks
# ----------------------------------------------------------------------------------------------

@nb.njit
def _rand(state):
    x = state[0]
    x ^= x << np.uint64(13)
    x ^= x >> np.uint64(7)
    x ^= x << np.uint64(17)
    state[0] = x
    return np.float64(x >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@nb.njit(nogil=True)
def _approx_chol(n, ei, ej, ew, seed):
    """Randomized approximate Cholesky factorization L D L' of the Laplacian of the multigraph
    (ei, ej, ew). Vertices are eliminated in order of increasing degree; eliminating v replaces
    its star by a sampled sparse approximation of the clique the exact elimination would add
    (unbiased, one edge per neighbor). Returns the order, factor columns and D."""
    m = ei.size
    cap = 2 * m + 16
    nxt = np.empty(cap, np.int64)
    nbr = np.empty(cap, np.int64)
    wt = np.empty(cap)
    head = np.full(n, -1, np.int64)
    deg0 = np.zeros(n, np.int64)
    top = 0
    for t in range(m):
        a, b, w = ei[t], ej[t], ew[t]
        nbr[top] = b
        wt[top] = w
        nxt[top] = head[a]
        head[a] = top
        top += 1
        nbr[top] = a
        wt[top] = w
        nxt[top] = head[b]
        head[b] = top
        top += 1
        deg0[a] += 1
        deg0[b] += 1
    state = np.array([np.uint64(seed) | np.uint64(1)])
    key = np.empty(n)
    for v in range(n):
        key[v] = deg0[v] + _rand(state)
    order = np.argsort(key)
    elim = np.zeros(n, np.bool_)
    slot = np.full(n, -1, np.int64)
    us = np.empty(n, np.int64)
    ws = np.empty(n)
    fptr = np.zeros(n + 1, np.int64)
    fcap = 2 * m + 16
    fidx = np.empty(fcap, np.int64)
    fval = np.empty(fcap)
    D = np.zeros(n)
    fpos = 0
    for oi in range(n):
        v = order[oi]
        d = 0
        e = head[v]
        while e >= 0:
            u = nbr[e]
            if not elim[u]:
                if slot[u] < 0:
                    slot[u] = d
                    us[d] = u
                    ws[d] = 0.0
                    d += 1
                ws[slot[u]] += wt[e]
            e = nxt[e]
        for t in range(d):
            slot[us[t]] = -1
        elim[v] = True
        W = 0.0
        for t in range(d):
            W += ws[t]
        D[v] = W
        if fpos + d > fcap:
            fcap = 2 * (fpos + d)
            fi2 = np.empty(fcap, np.int64)
            fv2 = np.empty(fcap)
            fi2[:fpos] = fidx[:fpos]
            fv2[:fpos] = fval[:fpos]
            fidx = fi2
            fval = fv2
        for t in range(d):
            fidx[fpos] = us[t]
            fval[fpos] = ws[t] / W
            fpos += 1
        fptr[oi + 1] = fpos
        if d >= 2:
            o = np.argsort(ws[:d])
            su = us[:d][o].copy()
            sw = ws[:d][o].copy()
            suf = np.empty(d + 1)
            suf[d] = 0.0
            for t in range(d - 1, -1, -1):
                suf[t] = suf[t + 1] + sw[t]
            if top + 2 * d > cap:
                cap = 2 * (top + 2 * d)
                n2 = np.empty(cap, np.int64)
                b2 = np.empty(cap, np.int64)
                w2 = np.empty(cap)
                n2[:top] = nxt[:top]
                b2[:top] = nbr[:top]
                w2[:top] = wt[:top]
                nxt = n2
                nbr = b2
                wt = w2
            for i in range(d - 1):
                target = _rand(state) * suf[i + 1]          # neighbor j > i, probability sw[j] / suf[i+1]
                lo = i + 1
                hi = d - 1
                while lo < hi:
                    mid = (lo + hi) // 2
                    if suf[i + 1] - suf[mid + 1] > target:
                        hi = mid
                    else:
                        lo = mid + 1
                w_new = sw[i] * suf[i + 1] / W
                a = su[i]
                b = su[lo]
                nbr[top] = b
                wt[top] = w_new
                nxt[top] = head[a]
                head[a] = top
                top += 1
                nbr[top] = a
                wt[top] = w_new
                nxt[top] = head[b]
                head[b] = top
                top += 1
    # relabel the factor in elimination order (solves then sweep rows sequentially)
    pos = np.empty(n, np.int64)
    for oi in range(n):
        pos[order[oi]] = oi
    fidx2 = np.empty(fpos, np.int64)
    for t in range(fpos):
        fidx2[t] = pos[fidx[t]]
    D2 = np.empty(n)
    for oi in range(n):
        D2[oi] = D[order[oi]]
    return order, fptr, fidx2, fval[:fpos].copy(), D2


@nb.njit
def _ac_solve_chunk(order, fptr, fidx, fval, D, Wc, R, Z, o0, k0, k1):
    """Columns [k0, k1) (at most 8) of Z[o0:o0+n] = (L D L')^+ R[o0:o0+n]. The factor is
    labelled in elimination order; Wc (n x 8) holds the block's rows in that order, padded
    with zero columns, so a row's 8 entries are one cache line and the sweeps run in registers."""
    n = order.size
    w = k1 - k0
    for i in range(n):
        v = o0 + order[i]
        for k in range(8):
            Wc[i, k] = R[v, k0 + k] if k < w else 0.0
    for i in range(n):
        a0, a1, a2, a3 = Wc[i, 0], Wc[i, 1], Wc[i, 2], Wc[i, 3]
        a4, a5, a6, a7 = Wc[i, 4], Wc[i, 5], Wc[i, 6], Wc[i, 7]
        for t in range(fptr[i], fptr[i + 1]):
            u = fidx[t]
            c = fval[t]
            Wc[u, 0] += c * a0
            Wc[u, 1] += c * a1
            Wc[u, 2] += c * a2
            Wc[u, 3] += c * a3
            Wc[u, 4] += c * a4
            Wc[u, 5] += c * a5
            Wc[u, 6] += c * a6
            Wc[u, 7] += c * a7
    for i in range(n):
        dv = D[i]
        for k in range(8):
            Wc[i, k] = Wc[i, k] / dv if dv > 0 else 0.0
    for i in range(n - 1, -1, -1):
        a0, a1, a2, a3 = Wc[i, 0], Wc[i, 1], Wc[i, 2], Wc[i, 3]
        a4, a5, a6, a7 = Wc[i, 4], Wc[i, 5], Wc[i, 6], Wc[i, 7]
        for t in range(fptr[i], fptr[i + 1]):
            u = fidx[t]
            c = fval[t]
            a0 += c * Wc[u, 0]
            a1 += c * Wc[u, 1]
            a2 += c * Wc[u, 2]
            a3 += c * Wc[u, 3]
            a4 += c * Wc[u, 4]
            a5 += c * Wc[u, 5]
            a6 += c * Wc[u, 6]
            a7 += c * Wc[u, 7]
        Wc[i, 0], Wc[i, 1], Wc[i, 2], Wc[i, 3] = a0, a1, a2, a3
        Wc[i, 4], Wc[i, 5], Wc[i, 6], Wc[i, 7] = a4, a5, a6, a7
    for i in range(n):
        v = o0 + order[i]
        for k in range(w):
            Z[v, k0 + k] = Wc[i, k]


@nb.njit(parallel=True)
def _ac_solve(order, fptr, fidx, fval, D, W, R, Z, o0):
    """Z[o0:o0+n] = (L D L')^+ R[o0:o0+n] for every column, in chunks of 8 columns (in
    parallel); W is a (chunks, n, 8) buffer."""
    K = R.shape[1]
    for ch in nb.prange(W.shape[0]):
        _ac_solve_chunk(order, fptr, fidx, fval, D, W[ch], R, Z, o0, 8 * ch, min(K, 8 * ch + 8))


@nb.njit
def _block_pairs(ptr, idx, cnt, cinv, lo, hi, dense):
    """Contributions sum_b n_bl n_bl' / c_b of the eliminated levels to the block [lo, hi):
    with dense, the full (hi-lo)^2 matrix; otherwise the off-diagonal pairs as an edge list."""
    nb_ = ptr.size - 1
    nl = hi - lo
    if dense:
        Mx = np.zeros((nl, nl))
        for b in range(nb_):
            for t in range(ptr[b], ptr[b + 1]):
                a = idx[t]
                if a < lo or a >= hi:
                    continue
                ca = cnt[t] * cinv[b]
                for u in range(ptr[b], ptr[b + 1]):
                    c = idx[u]
                    if c >= lo and c < hi:
                        Mx[a - lo, c - lo] += ca * cnt[u]
        return Mx, np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
    m = 0
    for b in range(nb_):
        k = 0
        for t in range(ptr[b], ptr[b + 1]):
            if idx[t] >= lo and idx[t] < hi:
                k += 1
        m += k * (k - 1) // 2
    ei = np.empty(m, np.int64)
    ej = np.empty(m, np.int64)
    ew = np.empty(m)
    p = 0
    for b in range(nb_):
        for t in range(ptr[b], ptr[b + 1]):
            a = idx[t]
            if a < lo or a >= hi:
                continue
            for u in range(t + 1, ptr[b + 1]):
                c = idx[u]
                if c >= lo and c < hi:
                    ei[p] = a - lo
                    ej[p] = c - lo
                    ew[p] = cnt[t] * cnt[u] * cinv[b]
                    p += 1
    return np.zeros((0, 0)), ei, ej, ew


@nb.njit
def _assemble_dense(ptr, idx, cnt, cinv, Lo):
    """sum_b n_b n_b' / c_b as a dense Lo x Lo matrix."""
    Mx = np.zeros((Lo, Lo))
    for b in range(ptr.size - 1):
        cb_ = cinv[b]
        for t in range(ptr[b], ptr[b + 1]):
            a = idx[t]
            ca = cnt[t] * cb_
            for u in range(ptr[b], ptr[b + 1]):
                Mx[a, idx[u]] += ca * cnt[u]
    return Mx


@nb.njit
def _add_csr_dense(ptr, idx, val, Mx):
    for r in range(ptr.size - 1):
        for t in range(ptr[r], ptr[r + 1]):
            Mx[r, idx[t]] += val[t]


@nb.njit
def _split_S(ptr, idx, val, small_rows, is_small):
    """Rows of S in small_rows as a dense (len(small_rows) x Lo) array; the rest of S without
    the columns of small blocks (those products use the dense rows' transpose) as CSR."""
    Lo = ptr.size - 1
    ns = small_rows.size
    Dn = np.zeros((ns, Lo))
    for a in range(ns):
        r = small_rows[a]
        for t in range(ptr[r], ptr[r + 1]):
            Dn[a, idx[t]] = val[t]
    nnz = np.zeros(Lo + 1, np.int64)
    for r in range(Lo):
        c = 0
        if not is_small[r]:
            for t in range(ptr[r], ptr[r + 1]):
                if not is_small[idx[t]]:
                    c += 1
        nnz[r + 1] = nnz[r] + c
    i2 = np.empty(nnz[Lo], np.int64)
    v2 = np.empty(nnz[Lo])
    for r in range(Lo):
        if is_small[r]:
            continue
        p = nnz[r]
        for t in range(ptr[r], ptr[r + 1]):
            if not is_small[idx[t]]:
                i2[p] = idx[t]
                v2[p] = val[t]
                p += 1
    return Dn, nnz, i2, v2


@nb.njit
def _pair_counts(ptr, idx, offs):
    """P[g, h] = sum over eliminated levels of (its levels in block g) x (its levels in block h)."""
    nblk = offs.size - 1
    out = np.zeros((nblk, nblk), np.int64)
    loc = np.zeros(nblk, np.int64)
    for b in range(ptr.size - 1):
        for t in range(ptr[b], ptr[b + 1]):
            o = idx[t]
            g = 0
            while o >= offs[g + 1]:
                g += 1
            loc[g] += 1
        for g in range(nblk):
            for h in range(nblk):
                out[g, h] += loc[g] * loc[h]
        for g in range(nblk):
            loc[g] = 0
    return out


@nb.njit(parallel=True, nogil=True)
def _assemble_S(a_ptr, a_idx, a_val, nb_ptr, nb_idx, nb_cnt, no_ptr, no_idx, no_cnt, cinv):
    """S = A_oo - N_ob C^-1 N_bo as CSR (Gustavson row by row)."""
    Lo = a_ptr.size - 1
    nnz = np.zeros(Lo + 1, np.int64)
    nparts = max(1, min(Lo, 4 * nb.get_num_threads()))
    for part in nb.prange(nparts):
        mark = np.full(Lo, -1, np.int64)
        for l in range(Lo * part // nparts, Lo * (part + 1) // nparts):
            c = 0
            for t in range(a_ptr[l], a_ptr[l + 1]):
                o = a_idx[t]
                if mark[o] != l:
                    mark[o] = l
                    c += 1
            for t in range(no_ptr[l], no_ptr[l + 1]):
                b = no_idx[t]
                for u in range(nb_ptr[b], nb_ptr[b + 1]):
                    o = nb_idx[u]
                    if mark[o] != l:
                        mark[o] = l
                        c += 1
            nnz[l + 1] = c
    for l in range(Lo):
        nnz[l + 1] += nnz[l]
    idx = np.empty(nnz[Lo], np.int64)
    val = np.zeros(nnz[Lo])
    for part in nb.prange(nparts):
        mark = np.full(Lo, -1, np.int64)
        slot = np.zeros(Lo, np.int64)
        for l in range(Lo * part // nparts, Lo * (part + 1) // nparts):
            p = nnz[l]
            for t in range(a_ptr[l], a_ptr[l + 1]):
                o = a_idx[t]
                if mark[o] != l:
                    mark[o] = l
                    slot[o] = p
                    idx[p] = o
                    p += 1
                val[slot[o]] += a_val[t]
            for t in range(no_ptr[l], no_ptr[l + 1]):
                b = no_idx[t]
                c = no_cnt[t] * cinv[b]
                for u in range(nb_ptr[b], nb_ptr[b + 1]):
                    o = nb_idx[u]
                    if mark[o] != l:
                        mark[o] = l
                        slot[o] = p
                        idx[p] = o
                        p += 1
                    val[slot[o]] -= c * nb_cnt[u]
    return nnz, idx, val


# ----------------------------------------------------------------------------------------------
# Degrees of freedom absorbed by the fixed effects (reghdfe conventions, as in the reference)
# ----------------------------------------------------------------------------------------------

@nb.njit
def _nested(a, La, b):
    """Every level of a lies within one level of b."""
    first = np.full(La, -1, np.int64)
    for i in range(a.size):
        if first[a[i]] < 0:
            first[a[i]] = b[i]
        elif first[a[i]] != b[i]:
            return False
    return True


@nb.njit
def _components(a, La, b, Lb):
    parent = np.arange(La + Lb)
    for i in range(a.size):
        x = a[i]
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        y = La + b[i]
        while parent[y] != y:
            parent[y] = parent[parent[y]]
            y = parent[y]
        if x != y:
            parent[x] = y
    nc = 0
    for v in range(La + Lb):
        if parent[v] == v:
            nc += 1
    return nc


def _absorbed_df(codes, L, cl, Lc):
    df_a = 0
    seen = []
    for g in range(codes.shape[1]):
        fe = np.ascontiguousarray(codes[:, g])
        if _nested(fe, L[g], cl):
            continue
        if not seen:
            df_a += L[g]
        elif len(seen) == 1:
            h = seen[0]
            df_a += L[g] - _components(np.ascontiguousarray(codes[:, h]), L[h], fe, L[g])
        else:
            nested_levels = [L[h] for h in seen if _nested(fe, L[g], np.ascontiguousarray(codes[:, h]))]
            df_a += L[g] - max([1] + nested_levels)
        seen.append(g)
    return df_a - 1 if seen else 0


# ----------------------------------------------------------------------------------------------

def run(y, X, fe_ids, cluster_ids, tol=1e-8, maxiter=100_000):
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64).reshape(len(y), -1)
    fe_ids = np.asarray(fe_ids).reshape(len(y), -1)
    n0, G = fe_ids.shape
    k = X.shape[1]

    # 1. Singletons, then levels of the kept rows.
    codes0 = np.empty((n0, G), np.int64)
    L0 = np.empty(G, np.int64)
    for g in range(G):
        codes0[:, g], L0[g] = _recode(fe_ids[:, g])
    keep = _peel(codes0, L0)
    idx = np.flatnonzero(keep)
    n = idx.size
    codes = np.empty((n, G), np.int64)
    L = np.empty(G, np.int64)
    for g in range(G):
        codes[:, g], L[g] = _recode(codes0[idx, g])
    cl, Lc = _recode(np.asarray(cluster_ids)[idx])
    K = k + 1

    # 2. Partial out the fixed effects: eliminate the one with most levels, CG on the rest. A
    # fixed effect whose levels are unions of another's levels adds nothing to the projection
    # (its indicators are sums of the other's): it is skipped here but counts in df_a.
    use = []
    for g in np.argsort(-L, kind="stable"):
        fe = np.ascontiguousarray(codes[:, g])
        if not any(_nested(np.ascontiguousarray(codes[:, h]), L[h], fe) for h in use):
            use.append(int(g))
    bcol = use[0]
    # other fixed effects, those with few levels last
    ocols = np.array(sorted(use[1:], key=lambda g: (L[g] <= SMALL_BLOCK, g)), np.int64)
    offs = np.concatenate([[0], np.cumsum(L[ocols])]).astype(np.int64)
    Lo = int(offs[-1])
    off_b, rows_b = _group_rows(codes[:, bcol], L[bcol])
    cb = np.diff(off_b).astype(np.float64)
    Sb = np.empty((L[bcol], K))
    _group_sums(y, X, idx, off_b, rows_b, Sb)
    if Lo == 0:
        alpha_b = Sb / cb[:, None]
        alpha_o = np.zeros((0, K))
    else:
        ocodes = codes[:, ocols] + offs[:-1][None, :]
        stacked = ocodes.ravel()                              # rows of every stacked level
        g_off, g_rows_flat = _group_rows(stacked, Lo)
        g_rows = g_rows_flat // len(ocols)
        nb_ptr, nb_idx, nb_cnt = _incidence(off_b, rows_b, ocodes, Lo)
        no_ptr, no_idx, no_cnt = _transpose(nb_ptr, nb_idx, nb_cnt, Lo)
        a_ptr, a_idx, a_val = _crosstab(ocodes, Lo, offs, g_off, g_rows)
        cinv = 1.0 / cb
        So = np.empty((Lo, K))
        _group_sums(y, X, idx, g_off, g_rows, So)
        # right-hand side: D_o'V - N_ob C^-1 D_b'V
        Tb = Sb * cinv[:, None]
        B = So.copy()
        _spmm(no_ptr, no_idx, -no_cnt, Tb, B, 1.0)
        # diagonal of S
        diag = np.diff(g_off).astype(np.float64)
        _diag_correction(diag, nb_ptr, nb_idx, nb_cnt, cinv)
        dinv = np.where(diag > 1e-9 * np.diff(g_off), 1.0 / np.where(diag > 0, diag, 1.0), 0.0)
        # preconditioner, block by block (each other fixed effect's levels): dense pseudo-inverse
        # for few levels, approximate Cholesky of the block's Laplacian when it is sparse,
        # otherwise the diagonal. With several threads the factorizations run in the background
        # while the operator is assembled.
        P2 = _pair_counts(nb_ptr, nb_idx, offs)
        sizes = np.diff(offs)
        pairs = np.diag(P2).copy()
        pool = ThreadPoolExecutor(1) if nb.get_num_threads() > 1 else None
        blocks = []
        for g in range(len(ocols)):
            lo, hi = int(offs[g]), int(offs[g + 1])
            if hi - lo <= DENSE_MAX:
                Mx, _, _, _ = _block_pairs(nb_ptr, nb_idx, nb_cnt, cinv, lo, hi, True)
                Sg = np.diag(np.diff(g_off)[lo:hi].astype(np.float64)) - Mx
                blocks.append(["dense", lo, hi, np.linalg.pinv(Sg, rtol=1e-10, hermitian=True)])
            elif pairs[g] <= SPARSE_PAIRS * (hi - lo):
                _, ei, ej, ew = _block_pairs(nb_ptr, nb_idx, nb_cnt, cinv, lo, hi, False)
                job = (hi - lo, ei, ej, ew, 20261022)
                blocks.append(["ac", lo, hi, pool.submit(_approx_chol, *job) if pool else _approx_chol(*job)])
            else:
                blocks.append(["diag", lo, hi, None])
        # operator: S itself when its entries are few, else through A_oo and N_bo
        nthreads = nb.get_num_threads()
        if pool is not None:
            nb.set_num_threads(nthreads - 1)          # one thread is factorizing
        implicit_cost = 2 * nb_idx.size + a_idx.size              # per column and iteration
        explicit_nnz = int(np.minimum(P2, np.outer(sizes, sizes)).sum()) + a_idx.size
        assembly = int(P2.sum())
        if Lo <= DENSE_S and assembly <= 64 * implicit_cost:
            Sd = _assemble_dense(nb_ptr, nb_idx, nb_cnt, cinv, Lo)
            Sd = -Sd
            _add_csr_dense(a_ptr, a_idx, a_val, Sd)

            def apply_S(P, Y):
                np.matmul(Sd, P, out=Y)
        elif explicit_nnz <= 2 * implicit_cost:      # the implicit route also streams N_bo C^-1 N_bo
            s_ptr, s_idx, s_val = _assemble_S(a_ptr, a_idx, a_val, nb_ptr, nb_idx, nb_cnt,
                                              no_ptr, no_idx, no_cnt, cinv)
            small = sizes <= SMALL_BLOCK                  # small blocks come last (see ocols)
            if small.any() and not small.all():
                is_small = np.repeat(small, sizes)
                nL = int(np.argmax(is_small))
                Dn, s_ptr, s_idx, s_val = _split_S(s_ptr, s_idx, s_val, np.arange(nL, Lo), is_small)
                DnL_T = Dn[:, :nL].T
                Tmp = np.empty((nL, K))

                def apply_S(P, Y):
                    _spmm(s_ptr, s_idx, s_val, P, Y, 0.0)       # large rows x large columns
                    np.matmul(DnL_T, P[nL:], out=Tmp)            # large rows x small columns
                    Y[:nL] += Tmp
                    np.matmul(Dn, P, out=Y[nL:])                 # rows of the small blocks
            else:
                def apply_S(P, Y):
                    _spmm(s_ptr, s_idx, s_val, P, Y, 0.0)
        else:
            # eliminated levels in an order that keeps the operand's rows in cache, split into
            # one part per thread (each part but the first scatters into its own buffer)
            order = _order_levels(nb_ptr, nb_idx, Lo)
            f_ptr, f_idx, f_cnt = _permute_rows(nb_ptr, nb_idx, nb_cnt, order)
            f_cinv = cinv[order]
            nparts = int(min(nthreads, L[bcol], max(1, 2**30 // (8 * Lo * K))))
            cuts = np.searchsorted(f_ptr[1:], f_ptr[-1] * np.arange(1, nparts) / nparts)
            bounds = np.concatenate([[0], cuts, [L[bcol]]]).astype(np.int64)
            Yp = np.empty((nparts - 1, Lo, K))

            def apply_S(P, Y):
                _spmm(a_ptr, a_idx, a_val, P, Y, 0.0)
                _fused(f_ptr, f_idx, f_cnt, f_cinv, P, Y, bounds, Yp)
        if pool is not None:
            for blk in blocks:
                if blk[0] == "ac":
                    blk[3] = blk[3].result()
            pool.shutdown()
            nb.set_num_threads(nthreads)
        for blk in blocks:
            if blk[0] == "ac":
                blk[3] = blk[3] + (np.empty(((K + 7) // 8, blk[2] - blk[1], 8)),)      # solve buffer

        def precond(R, Z):
            for kind, lo, hi, data in blocks:
                if kind == "dense":
                    Z[lo:hi] = data @ R[lo:hi]
                elif kind == "ac":
                    _ac_solve(*data, R, Z, lo)
                else:
                    _precond_range(R, dinv, Z, lo, hi)
        alpha_o, _ = _pcg(B, apply_S, precond, TOL, MAXITER)
        # eliminated coefficients: C^-1 (D_b'V - N_bo alpha_o)
        T = np.empty((L[bcol], K))
        _spmm(nb_ptr, nb_idx, nb_cnt, alpha_o, T, 0.0)
        alpha_b = (Sb - T) * cinv[:, None]
    W = np.empty((n, K))
    _residuals(y, X, idx, codes, alpha_b, bcol, alpha_o, ocols, offs, W)

    # 3. OLS on the demeaned data and the one-way cluster-robust sandwich.
    WW = W.T @ W
    XtX_inv = np.linalg.inv(WW[1:, 1:])
    beta = XtX_inv @ WW[1:, 0]
    e = W @ np.concatenate([[1.0], -beta])
    c_off, c_rows = _group_rows(cl, Lc)
    scores = np.empty((Lc, k))
    _cluster_scores(W, e, c_off, c_rows, scores)
    meat = scores.T @ scores
    df_a = _absorbed_df(codes, L, cl, Lc)
    Kdf = k + 1 + df_a
    q = (n - 1) / (n - Kdf) * Lc / (Lc - 1)
    vcov = q * XtX_inv @ meat @ XtX_inv
    return dict(beta=beta, vcov=vcov, nobs=int(n), df_a=int(df_a), n_clusters=int(Lc))
