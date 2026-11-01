"""Task 001 candidate: exact elimination of the largest fixed effect, then a solve of
the reduced normal equations in coefficient space.

The fixed effect with the most levels has a diagonal block in the normal equations, so
it is eliminated in closed form. The remaining system (a weighted graph Laplacian on the
other fixed effects' levels) is solved for y and every regressor at once, with a solver
chosen by its structure: dense for small systems, Jacobi-preconditioned CG for
well-connected graphs, and an exact grounded sparse factorization for forest-like graphs
(low worker mobility). The data are read in two passes: per-level sums and the Gram
matrix, then cluster scores from residuals rebuilt on the fly.

`tol` and `maxiter` are accepted for interface compatibility; the reduced system is
solved to near machine precision regardless. Numba kernels compile on the first call.
"""
import time

import numba as nb
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components

SMALL = 100        # FEs with at most this many levels form a dense border of the reduced system
DENSE_MAX = 4000   # reduced systems up to this size are factorized densely


# ---------------------------------------------------------------- preprocessing

def recode(ids):
    """Codes 0..L-1, in the order of the original values (like np.unique)."""
    lo, hi = int(ids.min()), int(ids.max())
    if lo >= 0 and hi < 4 * ids.size + 1024:
        present = np.zeros(hi + 1, dtype=bool)
        present[ids] = True
        if present.all():
            return ids.astype(np.int64, copy=False), hi + 1
        mapping = np.cumsum(present) - 1
        return mapping[ids], int(present.sum())
    u, inv = np.unique(ids, return_inverse=True)
    return inv.astype(np.int64), u.size


def drop_singletons(codes, levels):
    """Mask of kept observations, or None if no level has a single observation."""
    G = codes.shape[1]
    if all(np.bincount(codes[:, g], minlength=levels[g]).min() >= 2 for g in range(G)):
        return None
    keep = np.ones(codes.shape[0], dtype=bool)
    while True:
        changed = False
        for g in range(G):
            counts = np.bincount(codes[keep, g], minlength=levels[g])
            single = keep & (counts[codes[:, g]] == 1)
            if single.any():
                keep &= ~single
                changed = True
        if not changed:
            return keep


def nested_in(a, La, b):
    """True if every level of a lies inside exactly one level of b."""
    m = np.empty(La, dtype=b.dtype)
    m[a] = b
    return bool((m[a] == b).all())


def distinct_pairs(a, La, b, Lb):
    """Distinct (a, b) pairs, without sorting when the product space is small."""
    if La * Lb <= 50_000_000:
        key = np.flatnonzero(np.bincount(a * Lb + b, minlength=La * Lb))
    else:
        key = np.unique(a * Lb + b)
    return key // Lb, key % Lb


def absorbed_df(codes, levels, cl, G):
    """reghdfe's df_a rule, as in reference.absorbed_df, with O(n) nesting checks."""
    df_a, seen = 0, []
    for g in range(codes.shape[1]):
        fe, L = codes[:, g], levels[g]
        if nested_in(fe, L, cl):
            continue
        if not seen:
            df_a += L
        elif len(seen) == 1:
            a, La = seen[0]
            pa, pb = distinct_pairs(a, La, fe, L)
            graph = sp.coo_matrix((np.ones(pa.size), (pa, pb + La)), shape=(La + L, La + L))
            n_comp, _ = connected_components(graph, directed=False)
            df_a += L - n_comp
        else:
            nl = [Lh for h, Lh in seen if nested_in(fe, L, h)]
            df_a += L - max([1] + nl)
        seen.append((fe, L))
    return df_a - 1 if seen else 0


# ---------------------------------------------------------------- the two passes

@nb.njit(cache=False, fastmath=True)
def level_sums(y, X, codes, off, L):
    """sums[off[g] + codes[i, g]] += [y_i, x_i] for every fixed effect g."""
    n, k = X.shape
    G = codes.shape[1]
    out = np.zeros((L, k + 1))
    for i in range(n):
        yi = y[i]
        xi = X[i]
        for g in range(G):
            row = out[off[g] + codes[i, g]]
            row[0] += yi
            for j in range(k):
                row[j + 1] += xi[j]
    return out


@nb.njit(cache=False, fastmath=True)
def cluster_scores(y, X, codes, off, A, beta, cl, n_cl):
    """scores[c] = sum_{i in c} xt_i * et_i, with xt, et rebuilt from the FE coefficients A."""
    n, k = X.shape
    G = codes.shape[1]
    out = np.zeros((n_cl, k))
    zt = np.empty(k + 1)
    for i in range(n):
        xi = X[i]
        zt[0] = y[i]
        for j in range(k):
            zt[j + 1] = xi[j]
        for g in range(G):
            a = A[off[g] + codes[i, g]]
            for j in range(k + 1):
                zt[j] -= a[j]
        e = zt[0]
        for j in range(k):
            e -= zt[j + 1] * beta[j]
        row = out[cl[i]]
        for j in range(k):
            row[j] += zt[j + 1] * e
    return out


# ---------------------------------------------------------------- reduced system

@nb.njit(cache=False)
def crosstab_rows(codes, off, Le):
    """CSR of W = D_e'D_R: for each level of the eliminated FE (column 0), the distinct
    levels of the other FEs among its observations, with counts (O(n), counting sort)."""
    n, G = codes.shape
    start = np.zeros(Le + 1, np.int64)
    for i in range(n):
        start[codes[i, 0] + 1] += 1
    for l in range(Le):
        start[l + 1] += start[l]
    pos = start[:-1].copy()
    order = np.empty(n, np.int64)
    for i in range(n):
        l = codes[i, 0]
        order[pos[l]] = i
        pos[l] += 1
    gmax = 0
    for l in range(Le):
        gmax = max(gmax, start[l + 1] - start[l])
    buf = np.empty(gmax * (G - 1), np.int64)
    indptr = np.zeros(Le + 1, np.int64)
    indices = np.empty(n * (G - 1), np.int64)
    data = np.empty(n * (G - 1))
    nnz = 0
    for l in range(Le):
        m = 0
        for p in range(start[l], start[l + 1]):
            i = order[p]
            for g in range(1, G):
                buf[m] = off[g] - off[1] + codes[i, g]
                m += 1
        for a in range(1, m):  # insertion sort: groups are small
            v = buf[a]
            b = a - 1
            while b >= 0 and buf[b] > v:
                buf[b + 1] = buf[b]
                b -= 1
            buf[b + 1] = v
        j = 0
        while j < m:
            v = buf[j]
            c = 0
            while j < m and buf[j] == v:
                c += 1
                j += 1
            indices[nnz] = v
            data[nnz] = c
            nnz += 1
        indptr[l + 1] = nnz
    return indptr, indices[:nnz].copy(), data[:nnz].copy()


@nb.njit(cache=False)
def unique_rows(indptr, indices, data, ne):
    """Group identical rows of W (same levels, same counts). Returns one representative
    row per group and the summed weights sum(1/n_l): S's correction depends only on these."""
    Le = indptr.size - 1
    h = np.empty(Le, np.uint64)
    for l in range(Le):
        x = np.uint64(1469598103934665603)
        for a in range(indptr[l], indptr[l + 1]):
            x = (x ^ np.uint64(indices[a])) * np.uint64(1099511628211)
            x = (x ^ np.uint64(np.int64(data[a]))) * np.uint64(1099511628211)
        h[l] = x
    order = np.argsort(h)
    rep = np.empty(Le, np.int64)
    wt = np.empty(Le)
    used = np.zeros(Le, np.bool_)
    m = 0
    i = 0
    while i < Le:
        j = i
        while j < Le and h[order[j]] == h[order[i]]:
            j += 1
        # rows order[i:j] share a hash; split them into exact-equality groups
        for a in range(i, j):
            la = order[a]
            if used[la]:
                continue
            used[la] = True
            w = 1.0 / ne[la]
            for b in range(a + 1, j):
                lb = order[b]
                if used[lb]:
                    continue
                same = indptr[la + 1] - indptr[la] == indptr[lb + 1] - indptr[lb]
                if same:
                    for t in range(indptr[la + 1] - indptr[la]):
                        if indices[indptr[la] + t] != indices[indptr[lb] + t] or data[indptr[la] + t] != data[indptr[lb] + t]:
                            same = False
                            break
                if same:
                    used[lb] = True
                    w += 1.0 / ne[lb]
            rep[m] = la
            wt[m] = w
            m += 1
        i = j
    return rep[:m].copy(), wt[:m].copy()


@nb.njit(cache=False)
def schur_terms(indptr, indices, data, rep, wt, small, Ls, sidx):
    """Correction -sum_l w_l w_l' / n_l, split into sparse big-big triplets and dense
    border blocks (big-small and small-small), for small FEs given by `small`."""
    LR = small.size
    # count big-big pairs
    m = 0
    for l in rep:
        nb_ = 0
        for a in range(indptr[l], indptr[l + 1]):
            if not small[indices[a]]:
                nb_ += 1
        m += nb_ * nb_
    rows = np.empty(m, np.int64)
    cols = np.empty(m, np.int64)
    vals = np.empty(m)
    border = np.zeros((LR, Ls))
    t = 0
    for q in range(rep.size):
        l = rep[q]
        for a in range(indptr[l], indptr[l + 1]):
            ia = indices[a]
            wa = data[a] * wt[q]
            for b in range(indptr[l], indptr[l + 1]):
                ib = indices[b]
                if small[ib]:
                    border[ia, sidx[ib]] -= wa * data[b]
                elif not small[ia]:
                    rows[t] = ia
                    cols[t] = ib
                    vals[t] = -wa * data[b]
                    t += 1
    return rows, cols, vals, border


@nb.njit(cache=False)
def schur_dense(indptr, indices, data, rep, wt, LR):
    S = np.zeros((LR, LR))
    for q in range(rep.size):
        l = rep[q]
        for a in range(indptr[l], indptr[l + 1]):
            ia = indices[a]
            wa = data[a] * wt[q]
            for b in range(indptr[l], indptr[l + 1]):
                S[ia, indices[b]] -= wa * data[b]
    return S


def crosstab_R(codes, off, levels):
    """D_R'D_R as a sparse matrix in R-space (diagonal counts plus pairwise cross-tabs)."""
    G = codes.shape[1]
    base = off[1]
    rows, cols, vals = [], [], []
    for g in range(1, G):
        cnt = np.bincount(codes[:, g], minlength=levels[g]).astype(float)
        r = np.arange(levels[g]) + off[g] - base
        rows.append(r), cols.append(r), vals.append(cnt)
        for h in range(g + 1, G):
            Lg, Lh = levels[g], levels[h]
            key = codes[:, g] * Lh + codes[:, h]
            if Lg * Lh <= 50_000_000:
                c = np.bincount(key, minlength=Lg * Lh)
                nz = np.flatnonzero(c)
                c = c[nz].astype(float)
            else:
                nz, c = np.unique(key, return_counts=True)
                c = c.astype(float)
            a, b = nz // Lh + off[g] - base, nz % Lh + off[h] - base
            rows += [a, b]
            cols += [b, a]
            vals += [c, c]
    LR = int(off[-1] - base)
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(LR, LR))


class Implicit:
    """S = D_R'D_R - W' diag(1/n_e) W, applied without forming S."""

    def __init__(self, DRtDR, W, ne):
        self.DRtDR, self.W, self.Wt, self.inv = DRtDR, W, W.T.tocsr(), 1.0 / ne

    def __matmul__(self, P):
        return self.DRtDR @ P - self.Wt @ ((self.W @ P) * self.inv[:, None])


class Restricted:
    """Diagonally scaled restriction of an operator to the active levels."""

    def __init__(self, S, idx, s, L):
        self.S, self.idx, self.s, self.L = S, idx, s, L

    def __matmul__(self, P):
        full = np.zeros((self.L, P.shape[1]))
        full[self.idx] = P * self.s[:, None]
        return (self.S @ full)[self.idx] * self.s[:, None]


def pcg(S, B, precond, tol=1e-10, maxiter=20_000, floor=1e-8, patience=30):
    """Block preconditioned CG on a consistent PSD system, one scalar step per column.

    Stops at relative residual tol, or once the residual is below `floor` and has
    stopped improving: rounding leaves a small inconsistent part in B that a
    singular system cannot fit.
    """
    Xs = np.zeros_like(B)
    R = B.copy()
    Zr = precond(R)
    P = Zr.copy()
    rz = np.einsum("ij,ij->j", R, Zr)
    bnorm = np.linalg.norm(B, axis=0)
    bnorm[bnorm == 0] = 1.0
    best, since, Xbest = np.inf, 0, Xs
    for it in range(1, maxiter + 1):
        Q = S @ P
        pq = np.einsum("ij,ij->j", P, Q)
        alpha = np.divide(rz, pq, out=np.zeros_like(rz), where=pq > 0)
        Xs += P * alpha
        R -= Q * alpha
        rel = float(np.max(np.linalg.norm(R, axis=0) / bnorm))
        if rel <= tol:
            return Xs, it, rel
        if rel < 0.5 * best:
            best, since, Xbest = rel, 0, Xs.copy()
        else:
            since += 1
            if best <= floor and since >= patience:
                return Xbest, it, best
        Zr = precond(R)
        rz_new = np.einsum("ij,ij->j", R, Zr)
        beta = np.divide(rz_new, rz, out=np.zeros_like(rz), where=rz > 0)
        P = Zr + P * beta
        rz = rz_new
    raise RuntimeError(f"PCG did not converge in {maxiter} iterations")


def giant_share(W, big, Le):
    """Largest share of big-FE levels connected through the eliminated FE."""
    if not big.any():
        return 1.0  # only small FEs remain: a small, well-connected system
    Wc = W.tocoo()
    keep = big[Wc.col]
    LR = big.size
    g = sp.coo_matrix((np.ones(keep.sum()), (Wc.row[keep], Wc.col[keep] + Le)), shape=(Le + LR, Le + LR))
    _, lab = connected_components(g, directed=False)
    counts = np.bincount(lab[Le:][big])
    return counts.max() / max(big.sum(), 1)


def solve_forest(S, B, small, info):
    """Exact solve of [[S_bb, S_bs], [S_sb, S_ss]] A = B when the big-FE block S_bb is a
    Laplacian-like matrix of many small components: ground one level per component,
    factorize (no fill on forests), and fold the small FEs in through a dense Schur complement."""
    bi, si = np.flatnonzero(~small), np.flatnonzero(small)
    Sbb = S[bi][:, bi].tocsr()
    Sbs = S[bi][:, si].toarray()
    Sss = S[si][:, si].toarray()
    C = Sbb.tocoo()
    strong = (C.row != C.col) & (np.abs(C.data) > 1e-12)
    g = sp.coo_matrix((np.ones(strong.sum()), (C.row[strong], C.col[strong])), shape=Sbb.shape)
    ncomp, lab = connected_components(g, directed=False)
    ground = np.unique(lab, return_index=True)[1]          # one level per component set to 0
    free = np.ones(bi.size, dtype=bool)
    free[ground] = False
    fi = np.flatnonzero(free)
    lu = spla.splu(Sbb[fi][:, fi].tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                   options=dict(SymmetricMode=True))
    info["fill"] = round((lu.L.nnz + lu.U.nnz) / max(Sbb.nnz, 1), 2)
    info["components"] = int(ncomp)

    def pinv_bb(R):
        out = np.zeros((bi.size, R.shape[1]))
        out[fi] = lu.solve(np.ascontiguousarray(R[fi]))
        return out

    UV = pinv_bb(np.hstack([B[bi], Sbs]))
    U, V = UV[:, :B.shape[1]], UV[:, B.shape[1]:]
    A = np.zeros_like(B)
    if si.size:
        T = Sss - Sbs.T @ V                               # dense Schur complement on small FEs
        rs = B[si] - Sbs.T @ U
        w, Q = np.linalg.eigh(T)
        keep = w > 1e-10 * max(w.max(), 1e-300)
        a_s = Q[:, keep] @ ((Q[:, keep].T @ rs) / w[keep][:, None])
        A[si] = a_s
        A[bi] = U - V @ a_s
    else:
        A[bi] = U
    res = np.linalg.norm(S @ A - B, axis=0) / np.maximum(np.linalg.norm(B, axis=0), 1e-300)
    info["solver"] = f"grounded sparse LU + Schur ({ncomp} components, res {res.max():.0e})"
    return A


def solve_reduced(codes, off, levels, ne, W, B, info):
    """Solve S A = B, S = D_R'D_R - W' diag(1/n_e) W, PSD and possibly singular."""
    LR = B.shape[0]
    DRtDR = crosstab_R(codes, off, levels)
    diag0 = DRtDR.diagonal()
    d = diag0 - (W.multiply(W)).T @ (1.0 / ne)
    idx = np.flatnonzero(d > 1e-9 * diag0)     # levels with no remaining variation are free: 0
    s = 1.0 / np.sqrt(d[idx])
    Ba = B[idx] * s[:, None]
    info["active"] = idx.size
    small = np.repeat([L <= SMALL for L in levels[1:]], levels[1:])
    big = ~small

    if LR <= DENSE_MAX:
        rep, wt = unique_rows(W.indptr, W.indices, W.data, ne)
        info["unique_rows"] = rep.size
        S = DRtDR.toarray() + schur_dense(W.indptr, W.indices, W.data, rep, wt, LR)
        Sa = S[np.ix_(idx, idx)] * np.outer(s, s)
        try:  # well-conditioned systems: a few Jacobi-PCG iterations beat a factorization
            sol, it, rel = pcg(Sa, Ba, lambda R: R, maxiter=30)
            info["solver"] = f"dense Jacobi PCG ({it} it, res {rel:.0e})"
        except RuntimeError:
            cf = sla.cho_factor(Sa + 1e-10 * np.eye(idx.size), lower=True, check_finite=False)
            sol, it, rel = pcg(Sa, Ba, lambda R: sla.cho_solve(cf, R, check_finite=False))
            info["solver"] = f"dense Cholesky + PCG ({it} it, res {rel:.0e})"
    else:
        giant = giant_share(W, big, ne.size)
        info["giant"] = round(float(giant), 3)
        nbb = int((np.add.reduceat(big[W.indices].astype(np.int64), W.indptr[:-1]) ** 2).sum())
        explicit = nbb <= 8 * max(W.nnz, 1) or giant <= 0.5
        if explicit:
            sidx = np.cumsum(small) - 1
            Ls = int(small.sum())
            rep, wt = unique_rows(W.indptr, W.indices, W.data, ne)
            info["unique_rows"] = rep.size
            r, c, v, border = schur_terms(W.indptr, W.indices, W.data, rep, wt, small, Ls, sidx)
            sc = np.flatnonzero(small)
            br, bc = np.nonzero(border)
            Bm = sp.csr_matrix((border[br, bc], (br, sc[bc])), shape=(LR, LR))  # (all, small)
            Bbig = sp.diags(big.astype(float)) @ Bm                              # (big, small)
            Sx = (DRtDR + sp.csr_matrix((v, (r, c)), shape=(LR, LR)) + Bm + Bbig.T).tocsr()
            S = Sx[idx][:, idx]
            S = (sp.diags(s) @ S @ sp.diags(s)).tocsr()
            info["nnz_S"] = S.nnz
        else:
            S = Restricted(Implicit(DRtDR, W, ne), idx, s, LR)
        if giant > 0.5:
            # well connected (high mobility): Jacobi-preconditioned CG, tens of iterations
            sol, it, rel = pcg(S, Ba, lambda R: R)
            info["solver"] = f"{'explicit' if explicit else 'implicit'} Jacobi PCG ({it} it, res {rel:.0e})"
        else:
            # forest-like (low mobility): many small components, factorized exactly
            sol = solve_forest(S, Ba, small[idx], info)
    A = np.zeros_like(B)
    A[idx] = sol * s[:, None]
    return A


# ---------------------------------------------------------------- estimator

def run(y, X, fe_ids, cluster_ids, tol=1e-8, maxiter=100_000, info=None):
    """
    y           [n_obs]                float64
    X           [n_obs, n_regressors]  float64
    fe_ids      [n_obs, n_fe]          int64   (FE level codes, any integers)
    cluster_ids [n_obs]                int64
    info        optional dict that receives solver diagnostics and stage timings
    returns dict(beta [k], vcov [k, k], nobs, df_a, n_clusters)
    """
    info = {} if info is None else info
    t = {"start": time.perf_counter()}
    n0, G = fe_ids.shape
    codes, levels = np.empty((n0, G), dtype=np.int64), []
    for g in range(G):
        codes[:, g], L = recode(fe_ids[:, g])
        levels.append(L)
    keep = drop_singletons(codes, levels)
    cl, n_cl = recode(cluster_ids)
    if keep is not None:
        y, X, cl = y[keep], X[keep], cl[keep]
        codes = codes[keep]
        for g in range(G):
            codes[:, g], levels[g] = recode(codes[:, g])
        cl, n_cl = recode(cl)
    y, X = np.ascontiguousarray(y, dtype=float), np.ascontiguousarray(X, dtype=float)
    n, k = X.shape
    df_a = absorbed_df(codes, levels, cl, n_cl)
    t["prep"] = time.perf_counter()

    # eliminated FE (most levels) first, then the rest
    e = int(np.argmax(levels))
    order = [e] + [g for g in range(G) if g != e]
    codes = np.ascontiguousarray(codes[:, order])
    levels = [levels[g] for g in order]
    off = np.concatenate([[0], np.cumsum(levels)]).astype(np.int64)
    Le = levels[0]

    # pass 1: level sums of [y, X] for every FE, and the Gram matrix
    sums = level_sums(y, X, codes, off, int(off[-1]))
    ZtZ = np.empty((k + 1, k + 1))
    ZtZ[0, 0] = y @ y
    ZtZ[0, 1:] = ZtZ[1:, 0] = X.T @ y
    ZtZ[1:, 1:] = X.T @ X
    t["pass1"] = time.perf_counter()

    ne = np.bincount(codes[:, 0], minlength=Le).astype(float)
    Se = sums[:Le]
    if G > 1:
        LR = int(off[-1] - off[1])
        indptr, indices, data = crosstab_rows(codes, off, Le)
        W = sp.csr_matrix((data, indices, indptr), shape=(Le, LR))   # D_e'D_R
        B = sums[Le:] - W.T @ (Se / ne[:, None])
        t["assemble"] = time.perf_counter()
        info["L_R"] = LR
        AR = solve_reduced(codes, off, levels, ne, W, B, info)
        t["solve"] = time.perf_counter()
        A = np.vstack([(Se - W @ AR) / ne[:, None], AR])
    else:
        A = Se / ne[:, None]
        t["assemble"] = t["solve"] = time.perf_counter()

    # demeaned Gram matrix and OLS: Z'MZ = Z'Z - (D'Z)'A
    ZtMZ = ZtZ - sums.T @ A
    XtX, Xty = ZtMZ[1:, 1:], ZtMZ[1:, 0]
    beta = np.linalg.solve(XtX, Xty)

    # pass 2: cluster scores from the rebuilt residuals
    scores = cluster_scores(y, X, codes, off, A, beta, cl, n_cl)
    t["pass2"] = time.perf_counter()
    XtX_inv = np.linalg.inv(XtX)
    meat = scores.T @ scores
    K = k + 1 + df_a
    q = (n - 1) / (n - K) * n_cl / (n_cl - 1)
    vcov = q * XtX_inv @ meat @ XtX_inv
    vcov = (vcov + vcov.T) / 2
    t["end"] = time.perf_counter()
    keys = list(t)
    info["times"] = {b: round(t[b] - t[a], 4) for a, b in zip(keys, keys[1:])}
    return dict(beta=beta, vcov=vcov, nobs=int(n), df_a=int(df_a), n_clusters=int(n_cl))
