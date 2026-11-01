"""Task 002 candidate "cohort-fused": Callaway & Sant'Anna (2021) ATT(g,t) with the doubly robust
IPW estimator of Sant'Anna & Zhao (2020), the simple and event-study aggregates, and their
influence-function standard errors, as computed by reference.run.

The reference estimates every (cohort, period) cell separately. In each cell the propensity score
is a logit of cohort membership on X among the cohort and the never-treated units, and the outcome
model is an OLS of the outcome change on X among the never-treated units. Neither depends on the
period, so this candidate fits them once:

- one logit per cohort, by Newton-Raphson from zero with the reference's stopping rule. A cohort
  whose iteration does not converge (e.g. perfect separation) is recomputed cell by cell with the
  reference's own NumPy code, so that the reference's non-converged results are reproduced;
- one least-squares fit of every period's outcome on X among the never-treated units. The fit is
  linear in the outcome, so the residual of the change y_t - y_b is E_t - E_b with E = y - X B.

Each cell's influence function is then a short expression per unit in E, X, the cohort's
propensity scores and a few cell constants (sums over units, computed with BLAS). The reference
stores all cells' influence functions in an n_units x n_cells matrix and aggregates it; here one
fused pass over the units evaluates every cell's influence function for the unit, accumulates the
sums that the cell standard errors need, and forms the unit's simple, event-time and pre/post
aggregated influence functions (linear combinations of the cells' plus the cohort-weight terms),
so no n_units x n_cells array is formed.

Threads: Numba parallel loops (NUMBA_NUM_THREADS, set by the evaluator's --threads) over the
cohorts' logits and over units; BLAS follows the evaluator's thread limit.
"""
import numpy as np
import numba as nb
from numba import types
from numba.extending import intrinsic

TRIM = 1.0                   # no trimming: csdid passes pscoretrim(1) to drdid (task 002 since 2026-10-23)
LOGIT_TOL = 1e-12
FAST_MAXITER = 60
COND_MAX = 1e8               # fast-path logits must also be well conditioned
ZERO_ATT = 1e-9              # a cell ATT this small relative to its terms is recomputed exactly
BLOCK = 512


# ----------------------------------------------------------------------------------------------
# Reference code paths, used only for cohorts whose logit does not converge
# ----------------------------------------------------------------------------------------------

def _logit_reference(D, X, tol=LOGIT_TOL, maxiter=100):
    """reference.logit (identical NumPy operations)."""
    b = np.zeros(X.shape[1])
    for _ in range(maxiter):
        p = 1.0 / (1.0 + np.exp(-X @ b))
        H = (X * (p * (1 - p))[:, None]).T @ X
        step = np.linalg.solve(H, X.T @ (D - p))
        b += step
        if np.max(np.abs(step)) < tol:
            break
    p = 1.0 / (1.0 + np.exp(-X @ b))
    return b, np.linalg.inv((X * (p * (1 - p))[:, None]).T @ X)


def _drdid_cell(dy, X, D, ps_b, ps_V, trim=TRIM):
    """reference.drdid_panel after its logit (identical NumPy operations)."""
    n = dy.size
    ps_raw = 1.0 / (1.0 + np.exp(-X @ ps_b))
    ps = np.minimum(ps_raw, 1 - 1e-6)
    keep = ~((D == 0) & (ps_raw > trim))
    reg_b = np.linalg.lstsq(X[D == 0], dy[D == 0], rcond=None)[0]
    resid = dy - X @ reg_b
    w1 = D.astype(float)
    w0 = ps * (1 - D) / (1 - ps) * keep
    att_treat, att_cont = w1 * resid, w0 * resid
    eta_treat = att_treat.mean() / w1.mean()
    eta_cont = att_cont.mean() / w0.mean()
    att = eta_treat - eta_cont
    w_ols = 1.0 - D
    lin_ols = (w_ols * resid)[:, None] * X @ (np.linalg.inv((X * w_ols[:, None]).T @ X) * n)
    lin_ps = ((D - ps)[:, None] * X) @ (ps_V * n)
    inf_treat = (att_treat - w1 * eta_treat - lin_ols @ (w1[:, None] * X).mean(0)) / w1.mean()
    inf_cont = (att_cont - w0 * eta_cont
                + lin_ps @ ((w0 * (resid - eta_cont))[:, None] * X).mean(0)
                - lin_ols @ (w0[:, None] * X).mean(0)) / w0.mean()
    return att + inf_treat - inf_cont


# ----------------------------------------------------------------------------------------------
# Vectorizable kernels
# ----------------------------------------------------------------------------------------------

@intrinsic
def _as_f64(typingctx, x):
    def codegen(context, builder, sig, args):
        return builder.bitcast(args[0], context.get_value_type(types.float64))
    return types.float64(types.int64), codegen


@intrinsic
def _as_i64(typingctx, x):
    def codegen(context, builder, sig, args):
        return builder.bitcast(args[0], context.get_value_type(types.int64))
    return types.int64(types.float64), codegen


@nb.njit(inline="always", error_model="numpy")
def _exp(x):
    """exp(x) to about 1 ulp for |x| <= 708 (clamped outside), branch-free so loops vectorize:
    Cody-Waite reduction by ln 2, degree-13 Taylor polynomial, exponent set from integer bits."""
    x = min(max(x, -708.0), 708.0)
    t = x * 1.4426950408889634 + 6755399441055744.0       # 1.5 * 2^52: round(x / ln 2) in low bits
    n = t - 6755399441055744.0
    r = x - n * 6.93147180369123816490e-01 - n * 1.90821492927058770002e-10
    p = 1.0 / 6227020800.0
    p = p * r + 1.0 / 479001600.0
    p = p * r + 1.0 / 39916800.0
    p = p * r + 1.0 / 3628800.0
    p = p * r + 1.0 / 362880.0
    p = p * r + 1.0 / 40320.0
    p = p * r + 1.0 / 5040.0
    p = p * r + 1.0 / 720.0
    p = p * r + 1.0 / 120.0
    p = p * r + 1.0 / 24.0
    p = p * r + 1.0 / 6.0
    p = p * r + 0.5
    p = p * r + 1.0
    p = p * r + 1.0
    return p * _as_f64((_as_i64(t) + 1023) << 52)


@nb.njit(nogil=True, error_model="numpy", fastmath={"contract"})
def _logistic(wk, k1, treated):
    """Row k1 of wk holds the linear index; replace it by the score residual D - p and put the
    weight p (1 - p) in row k1 + 1. No reassociation here: _exp relies on exact rounding."""
    for u in range(BLOCK):
        p = 1.0 / (1.0 + _exp(-wk[k1, u]))
        wk[k1 + 1, u] = p * (1.0 - p)
        wk[k1, u] = (1.0 - p) if treated else -p


@nb.njit(nogil=True, error_model="numpy", fastmath={"reassoc", "contract"})
def _newton_pass(Xb, b, treated, wk, acc):
    """Add the logit gradient (row k1 of acc) and Hessian (rows 0..k1-1, lower triangle)
    contributions of the units in Xb (blocks x k1 x BLOCK, zero columns as padding, which
    contribute nothing). Reassociation is allowed here so that the sums over units vectorize."""
    k1 = Xb.shape[1]
    for blk in range(Xb.shape[0]):
        xb = Xb[blk]
        for u in range(BLOCK):
            wk[k1, u] = 0.0
        for k in range(k1):
            bk = b[k]
            for u in range(BLOCK):
                wk[k1, u] += xb[k, u] * bk
        _logistic(wk, k1, treated)
        for k in range(k1):
            for u in range(BLOCK):
                wk[k, u] = wk[k1 + 1, u] * xb[k, u]
        for k in range(k1):
            t = 0.0
            for u in range(BLOCK):
                t += wk[k1, u] * xb[k, u]
            acc[k1, k] += t
            for l in range(k + 1):
                t = 0.0
                for u in range(BLOCK):
                    t += wk[k, u] * xb[l, u]
                acc[k, l] += t


@nb.njit(nogil=True, error_model="numpy")
def _logit_cohort(XbC, XbT, tol, maxiter):
    """Logit of membership (units of XbT) against the controls (XbC) by Newton-Raphson from
    zero, with the reference's stopping rule; returns (b, inverse Hessian at b, converged)."""
    k1 = XbC.shape[1]
    b = np.zeros(k1)
    H = np.zeros((k1, k1))
    g = np.zeros(k1)
    wk = np.empty((k1 + 2, BLOCK))
    acc = np.empty((k1 + 1, k1))
    converged = False
    for it in range(maxiter + 1):
        acc[:, :] = 0.0
        _newton_pass(XbC, b, False, wk, acc)
        _newton_pass(XbT, b, True, wk, acc)
        for k in range(k1):
            g[k] = acc[k1, k]
            for l in range(k + 1):
                H[k, l] = acc[k, l]
                H[l, k] = acc[k, l]
        if converged or it == maxiter:
            break
        step = np.linalg.solve(H, g)
        b += step
        if np.max(np.abs(step)) < tol:
            converged = True                 # one more pass for the Hessian at the final b
    ev = np.linalg.eigvalsh(H)
    well = ev[0] > 0.0 and ev[-1] <= COND_MAX * ev[0]
    return b, np.linalg.inv(H), converged and well


@nb.njit(parallel=True, error_model="numpy")
def _all_logits(XbC, XbT, boff, tol, maxiter):
    ncoh = boff.size - 1
    k1 = XbC.shape[1]
    B = np.zeros((ncoh, k1))
    V = np.zeros((ncoh, k1, k1))
    ok = np.zeros(ncoh, np.bool_)
    for j in nb.prange(ncoh):
        b, v, c = _logit_cohort(XbC, XbT[boff[j]:boff[j + 1]], tol, maxiter)
        B[j] = b
        V[j] = v
        ok[j] = c
    return B, V, ok


@nb.njit(parallel=True, error_model="numpy")
def _scores(XtC, XtT, toff, B, psc, w0c, pst):
    """Clipped propensity scores (and weights ps/(1-ps), zero for scores above TRIM; none at TRIM = 1)
    of all controls for every cohort (rows of psc, w0c) and of every treated unit for its own cohort."""
    ncoh, k1 = B.shape
    nC = XtC.shape[1]
    nblk = (nC + BLOCK - 1) // BLOCK
    for task in nb.prange(ncoh * nblk):
        j = task // nblk
        lo = (task % nblk) * BLOCK
        m = min(BLOCK, nC - lo)
        eta = np.zeros(m)
        for k in range(k1):
            xk = XtC[k, lo:lo + m]
            bk = B[j, k]
            for u in range(m):
                eta[u] += xk[u] * bk
        pr = psc[j, lo:lo + m]
        wr = w0c[j, lo:lo + m]
        for u in range(m):
            raw = 1.0 / (1.0 + _exp(-eta[u]))
            ps = min(raw, 1.0 - 1e-6)
            pr[u] = ps
            wr[u] = 0.0 if raw > TRIM else ps / (1.0 - ps)
    for j in range(ncoh):
        for c in range(toff[j], toff[j + 1]):
            eta = 0.0
            for k in range(k1):
                eta += XtT[k, c] * B[j, k]
            pst[c] = min(1.0 / (1.0 + _exp(-eta)), 1.0 - 1e-6)


FB = 64                     # units per block (vector lanes) in the passes over units
FAST = {"reassoc", "contract"}


@nb.njit(error_model="numpy")
def _ranges(n, parts):
    return np.linspace(0, n, max(1, min(n, parts)) + 1).astype(np.int64)


@nb.njit(parallel=True, error_model="numpy", fastmath=FAST)
def _ctrl_sums(EB, XB, W0B, nparts):
    """Per cohort j and period t: Sw[j, t] = sum_C w0 E[:, t] and SXw[j, k, t] = sum_C w0 x_k E[:, t]
    over the controls, stored in blocks of FB units: EB (blocks x T x FB), XB (blocks x k1 x FB),
    W0B (cohorts x blocks x FB; zero for padding)."""
    nblk, T, B = EB.shape
    k1 = XB.shape[1]
    ncoh = W0B.shape[0]
    edges = _ranges(nblk, nparts)
    np_ = edges.size - 1
    Sw = np.zeros((np_, ncoh, T))
    SXw = np.zeros((np_, ncoh, k1, T))
    for part in nb.prange(np_):
        wx = np.empty(B)
        for blk in range(edges[part], edges[part + 1]):
            for j in range(ncoh):
                for t in range(T):
                    acc = 0.0
                    for b in range(B):
                        acc += W0B[j, blk, b] * EB[blk, t, b]
                    Sw[part, j, t] += acc
                for k in range(k1):
                    for b in range(B):
                        wx[b] = W0B[j, blk, b] * XB[blk, k, b]
                    for t in range(T):
                        acc = 0.0
                        for b in range(B):
                            acc += wx[b] * EB[blk, t, b]
                        SXw[part, j, k, t] += acc
    return Sw.sum(axis=0), SXw.sum(axis=0)


@nb.njit(error_model="numpy")
def _all(flags, j):
    for u in range(flags.shape[1]):
        if not flags[j, u]:
            return False
    return True


@nb.njit(error_model="numpy", fastmath=FAST)
def _finish_block(out, osim, up, A1, A2, part, nE, npre):
    """Add the block's units' aggregated influence values (out: events x FB, osim: simple) and
    their pre/post averages to the sums and sums of squares."""
    B = osim.size
    for e in range(nE):
        s = 0.0
        s2 = 0.0
        for b in range(B):
            v = out[e, b]
            s += v
            s2 += v * v
        A1[part, e] += s
        A2[part, e] += s2
    s = 0.0
    s2 = 0.0
    for b in range(B):
        s += osim[b]
        s2 += osim[b] * osim[b]
    A1[part, nE] += s
    A2[part, nE] += s2
    for which in range(2):
        e0 = 0 if which == 0 else npre
        e1 = npre if which == 0 else nE
        if e1 == e0:
            continue
        for b in range(B):
            up[b] = 0.0
        for e in range(e0, e1):
            for b in range(B):
                up[b] += out[e, b]
        s = 0.0
        s2 = 0.0
        inv = 1.0 / (e1 - e0)
        for b in range(B):
            v = up[b] * inv
            s += v
            s2 += v * v
        A1[part, nE + 1 + which] += s
        A2[part, nE + 1 + which] += s2


@nb.njit(error_model="numpy", fastmath=FAST)
def _start_block(out, osim, Fb, blk, hasF, nE):
    B = osim.size
    for e in range(nE):
        for b in range(B):
            out[e, b] = Fb[blk, e, b] if hasF else 0.0
    for b in range(B):
        osim[b] = Fb[blk, nE, b] if hasF else 0.0


@nb.njit(parallel=True, error_model="numpy", fastmath=FAST)
def _fused_ctrl(EB, XB, PSB, W0B, Fb, gper, exact, rmw1, rmw0, a1, a2, A3, etaC, piE, piS,
                emin, nE, npre, nparts):
    """Pass over the controls (blocks of FB units). For each unit and cell, d = r - att with r the
    reference's drdid_panel output; accumulates per cell sum d and sum d^2, and per aggregate
    the sums of the units' aggregated influence values and their squares."""
    nblk, T, B = EB.shape
    nc = T - 1
    k1 = XB.shape[1]
    ncoh = gper.size
    hasF = Fb.shape[0] > 0
    edges = _ranges(nblk, nparts)
    np_ = edges.size - 1
    S1 = np.zeros((np_, ncoh, nc))
    S2 = np.zeros((np_, ncoh, nc))
    A1 = np.zeros((np_, nE + 3))
    A2 = np.zeros((np_, nE + 3))
    for part in nb.prange(np_):
        out = np.zeros((nE, B))
        osim = np.zeros(B)
        up = np.zeros(B)
        al = np.zeros(B)
        be = np.zeros(B)
        ga = np.zeros(B)
        z = np.zeros(B)
        d = np.zeros(B)
        for blk in range(edges[part], edges[part + 1]):
            _start_block(out, osim, Fb, blk, hasF, nE)
            for j in range(ncoh):
                if _all(exact, j):
                    continue
                for b in range(B):
                    al[b] = 0.0
                    be[b] = 0.0
                for k in range(k1):
                    c1 = a1[j, k]
                    c2 = a2[j, k]
                    for b in range(B):
                        al[b] += c1 * XB[blk, k, b]
                        be[b] += c2 * XB[blk, k, b]
                r1 = rmw1[j]
                r0 = rmw0[j]
                for b in range(B):
                    w0 = W0B[j, blk, b]
                    al[b] = -al[b] * r1 + (be[b] - w0) * r0
                    be[b] = w0 * r0
                    ga[b] = PSB[j, blk, b] * r0
                g = gper[j]
                off = 2 - g - emin
                for u in range(nc):
                    if exact[j, u]:
                        continue
                    bc = u if u < g - 2 else g - 2          # base column of the cell
                    ec = etaC[j, u]
                    a = A3[j, 0, u]
                    for b in range(B):
                        z[b] = a * XB[blk, 0, b]
                    for k in range(1, k1):
                        a = A3[j, k, u]
                        for b in range(B):
                            z[b] += a * XB[blk, k, b]
                    s1 = 0.0
                    s2 = 0.0
                    for b in range(B):
                        dv = al[b] * (EB[blk, u + 1, b] - EB[blk, bc, b]) + be[b] * ec + ga[b] * z[b]
                        d[b] = dv
                        s1 += dv
                        s2 += dv * dv
                    S1[part, j, u] += s1
                    S2[part, j, u] += s2
                    pe = piE[j, u]
                    pse = piS[j, u]
                    e = off + u
                    for b in range(B):
                        out[e, b] += pe * d[b]
                        osim[b] += pse * d[b]
            _finish_block(out, osim, up, A1, A2, part, nE, npre)
    return S1.sum(axis=0), S2.sum(axis=0), A1.sum(axis=0), A2.sum(axis=0)


@nb.njit(parallel=True, error_model="numpy", fastmath=FAST)
def _fused_treated(EB, XB, PSB, VB, bcoh, Fb, gper, exact, rmw1, rmw0, A3, etaT, piE, piS,
                   kapE, kapS, emin, nE, npre, nparts):
    """As _fused_ctrl for the treated units, in blocks that each hold units of one cohort
    (bcoh); VB is 1 for units and 0 for padding."""
    nblk, T, B = EB.shape
    nc = T - 1
    k1 = XB.shape[1]
    ncoh = gper.size
    hasF = Fb.shape[0] > 0
    edges = _ranges(nblk, nparts)
    np_ = edges.size - 1
    S1 = np.zeros((np_, ncoh, nc))
    S2 = np.zeros((np_, ncoh, nc))
    A1 = np.zeros((np_, nE + 3))
    A2 = np.zeros((np_, nE + 3))
    for part in nb.prange(np_):
        out = np.zeros((nE, B))
        osim = np.zeros(B)
        up = np.zeros(B)
        c1 = np.zeros(B)
        c2 = np.zeros(B)
        z = np.zeros(B)
        d = np.zeros(B)
        for blk in range(edges[part], edges[part + 1]):
            _start_block(out, osim, Fb, blk, hasF, nE)
            j = bcoh[blk]
            if not _all(exact, j):
                r1 = rmw1[j]
                r0 = rmw0[j]
                for b in range(B):
                    c1[b] = r1 * VB[blk, b]
                    c2[b] = (1.0 - PSB[blk, b]) * r0 * VB[blk, b]
                g = gper[j]
                off = 2 - g - emin
                for u in range(nc):
                    if exact[j, u]:
                        continue
                    bc = u if u < g - 2 else g - 2
                    et = etaT[j, u]
                    a = A3[j, 0, u]
                    for b in range(B):
                        z[b] = a * XB[blk, 0, b]
                    for k in range(1, k1):
                        a = A3[j, k, u]
                        for b in range(B):
                            z[b] += a * XB[blk, k, b]
                    s1 = 0.0
                    s2 = 0.0
                    for b in range(B):
                        dv = (EB[blk, u + 1, b] - EB[blk, bc, b] - et) * c1[b] - c2[b] * z[b]
                        d[b] = dv
                        s1 += dv
                        s2 += dv * dv
                    S1[part, j, u] += s1
                    S2[part, j, u] += s2
                    pe = piE[j, u]
                    pse = piS[j, u]
                    ke = kapE[j, u]
                    e = off + u
                    for b in range(B):
                        out[e, b] += pe * d[b] + ke * VB[blk, b]
                        osim[b] += pse * d[b]
                ks = kapS[j]
                for b in range(B):
                    osim[b] += ks * VB[blk, b]
            _finish_block(out, osim, up, A1, A2, part, nE, npre)
    return S1.sum(axis=0), S2.sum(axis=0), A1.sum(axis=0), A2.sum(axis=0)


def _lanes(M, groups=None):
    """Rows of M (n x m) in blocks of FB lanes: (blocks x m x FB), zero padded. With groups
    (offsets of consecutive row groups), every group starts a new block; returns the blocks,
    the group of each block and the validity of each lane."""
    n, m = M.shape
    if groups is None:
        groups = np.array([0, n])
    sizes = np.diff(groups)
    nblk = (sizes + FB - 1) // FB
    rows = np.full(int(nblk.sum()) * FB, -1, np.int64)
    pos = 0
    for gi in range(sizes.size):
        rows[pos:pos + sizes[gi]] = np.arange(groups[gi], groups[gi + 1])
        pos += nblk[gi] * FB
    valid = rows >= 0
    out = np.zeros((rows.size, m))
    out[valid] = M[rows[valid]]
    blocks = np.ascontiguousarray(out.reshape(-1, FB, m).transpose(0, 2, 1))
    return blocks, np.repeat(np.arange(sizes.size), nblk), valid.reshape(-1, FB).astype(np.float64)


def _blocks(Xt, lo, hi):
    """Columns lo:hi of Xt (k1 x n) as zero-padded blocks (nblocks x k1 x BLOCK)."""
    k1 = Xt.shape[0]
    nb_ = (hi - lo + BLOCK - 1) // BLOCK
    out = np.zeros((nb_, BLOCK, k1)).reshape(nb_ * BLOCK, k1)
    out[:hi - lo] = Xt[:, lo:hi].T
    return np.ascontiguousarray(out.reshape(nb_, BLOCK, k1).transpose(0, 2, 1))


def _edges(n, parts):
    return np.linspace(0, n, max(1, min(n, parts)) + 1).astype(np.int64)


def run(y, X, gvar):
    y = np.ascontiguousarray(y, dtype=np.float64)
    N, T = y.shape
    nc = T - 1
    Xc = np.column_stack([np.asarray(X, dtype=np.float64).reshape(N, -1), np.ones(N)])
    k1 = Xc.shape[1]
    gvar = np.asarray(gvar)
    cohorts = np.unique(gvar[gvar > 0])
    ncoh = cohorts.size
    gper = cohorts.astype(np.int64)
    cohort_of = np.searchsorted(cohorts, gvar).astype(np.int64)
    cohort_of[gvar == 0] = -1
    ctrl = np.flatnonzero(gvar == 0)
    nC = ctrl.size
    tr = np.flatnonzero(gvar > 0)
    tr = tr[np.argsort(cohort_of[tr], kind="stable")]          # treated units grouped by cohort
    n_gi = np.bincount(cohort_of[tr], minlength=ncoh)
    toff = np.concatenate([[0], np.cumsum(n_gi)]).astype(np.int64)
    n_g = n_gi.astype(np.float64)
    n_s = nC + n_g
    slot = np.empty(N, np.int64)
    slot[ctrl] = np.arange(nC)
    slot[tr] = np.arange(tr.size)
    threads = nb.get_num_threads()

    # 1. Propensity scores: one logit per cohort.
    X0 = Xc[ctrl]
    XtC = np.ascontiguousarray(X0.T)
    XtT = np.ascontiguousarray(Xc[tr].T)
    XbC = _blocks(XtC, 0, nC)
    tblocks = [_blocks(XtT, toff[j], toff[j + 1]) for j in range(ncoh)]
    boff = np.concatenate([[0], np.cumsum([t.shape[0] for t in tblocks])]).astype(np.int64)
    XbT = np.concatenate(tblocks) if tblocks else np.zeros((0, k1, BLOCK))
    B, V, ok = _all_logits(XbC, XbT, boff, LOGIT_TOL, FAST_MAXITER)
    exact = ~ok                       # cohorts computed with the reference's own code
    psc = np.empty((ncoh, nC))
    w0c = np.empty((ncoh, nC))
    pst = np.empty(tr.size)
    _scores(XtC, XtT, toff, B, psc, w0c, pst)

    # 2. Outcome model: OLS of every period's outcome on X among the never-treated units.
    Q, R = np.linalg.qr(X0)                                    # the reference's lstsq solution
    Bols = np.linalg.solve(R, Q.T @ y[ctrl])                   # (X0 has full column rank there)
    E = y - Xc @ Bols
    EC = E[ctrl]
    XtX0_inv = np.linalg.inv(X0.T @ X0)

    # 3. Cell constants. Columns u = 0..T-2 are periods t = u + 2; base column of a cell is
    #    t - 2 before the cohort's first period and g - 2 from it on.
    EBC, _, _ = _lanes(EC)
    XBC, _, _ = _lanes(X0)
    nbC = EBC.shape[0]
    PSB = np.zeros((ncoh, nbC * FB))
    PSB[:, :nC] = psc
    PSB = PSB.reshape(ncoh, nbC, FB)
    W0B = np.zeros((ncoh, nbC * FB))
    W0B[:, :nC] = w0c
    W0B = W0B.reshape(ncoh, nbC, FB)
    Sw, SXw = _ctrl_sums(EBC, XBC, W0B, 8 * threads)
    sw0 = w0c.sum(axis=1)
    sw0x = w0c @ X0                                            # (ncoh, k1)
    Sg = np.add.reduceat(E[tr], toff[:-1], axis=0) if tr.size else np.zeros((0, T))
    Sx = np.add.reduceat(Xc[tr], toff[:-1], axis=0) if tr.size else np.zeros((0, k1))
    mw1 = n_g / n_s
    mw0 = sw0 / n_s
    a1 = Sx @ XtX0_inv                                         # A M1 with A = inv(X0'X0) n_s
    a2 = sw0x @ XtX0_inv                                       # A M2
    cols = np.arange(1, T)                                     # data columns of the cells
    basec = np.where(cols[None, :] + 1 < gper[:, None], cols[None, :] - 1, gper[:, None] - 2)
    rows = np.arange(ncoh)[:, None]
    etaT = (Sg[:, 1:] - Sg[rows, basec]) / n_g[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        etaC = (Sw[:, 1:] - Sw[rows, basec]) / sw0[:, None]
    att = etaT - etaC
    # Cells computed with the reference's own NumPy code: all cells of cohorts whose logit is not
    # well determined, and cells whose ATT is zero up to rounding (the reference's rounding
    # residue then decides the reported value).
    # (rounding in E - which has the outcome's level - is relative to the outcome's level)
    yscale = np.abs(y).mean(axis=0)
    xcell = exact[:, None] | ~(np.abs(att) > ZERO_ATT * (yscale[1:][None, :] + yscale[basec]))
    psi, att_x, se_x = {}, {}, {}
    for j in np.flatnonzero(xcell.any(axis=1)):
        g = cohorts[j]
        s = (gvar == 0) | (gvar == g)
        D = (gvar[s] == g).astype(float)
        Xs = Xc[s]
        b, v = _logit_reference(D, Xs)
        base = 1
        for t in range(2, T + 1):
            if xcell[j, t - 2]:
                dy = y[s, t - 1] - y[s, base - 1]
                r_ = _drdid_cell(dy, Xs, D, b, v)
                full = np.zeros(N)
                full[s] = (r_ - r_.mean()) * N / s.sum()
                rif = full + r_.mean()
                # the reference reduces the n_units x n_cells matrix over axis 0 (a running sum
                # down each column); a two-column array gives the same order of additions
                two = np.column_stack([rif, rif])
                m = two.mean(0)[0]
                psi[j, t - 2] = rif - m
                att_x[j, t - 2] = m
                se_x[j, t - 2] = np.sqrt(((two - two.mean(0)) ** 2).sum(0))[0] / N
                att[j, t - 2] = m
            if t < g:
                base = t
    M3 = ((SXw[:, :, 1:] - SXw[rows[:, :, None], np.arange(k1)[None, :, None], basec[:, None, :]])
          - etaC[:, None, :] * sw0x[:, :, None]) / n_s[:, None, None]
    A3 = np.ascontiguousarray(np.einsum("jkl,jlt->jkt", V * n_s[:, None, None], M3))

    # 4. Aggregation weights (cohort sizes) and the influence of the estimated weights.
    ev = (cols[None, :] + 1) - gper[:, None]                  # event time of each cell
    events = np.unique(ev)
    emin = int(events[0])
    nE = events.size
    npre = int((events < 0).sum())
    eidx = ev - emin
    cell_ng = np.broadcast_to(n_g[:, None], ev.shape)
    post = ev >= 0
    ng_post = cell_ng[post].sum()
    ng_event = np.bincount(eidx.ravel(), weights=cell_ng.ravel(), minlength=nE)
    att_s0 = (att * cell_ng)[post].sum() / ng_post
    att_e0 = np.bincount(eidx.ravel(), weights=(att * cell_ng).ravel(), minlength=nE) / ng_event
    piE = np.ascontiguousarray(cell_ng / ng_event[eidx] * N / n_s[:, None])
    piS = np.ascontiguousarray(np.where(post, cell_ng / ng_post * N / n_s[:, None], 0.0))
    kapE = np.ascontiguousarray((att - att_e0[eidx]) * N / ng_event[eidx])
    kapS_cell = np.where(post, (att - att_s0) * N / ng_post, 0.0)
    kapS = np.where(xcell, 0.0, kapS_cell).sum(axis=1)       # added in the pass over units

    # Fallback cohorts: their cells' influence columns and cohort-weight terms, per unit.
    F = np.zeros((N if psi else 0, nE + 1))
    for (j, u), col in psi.items():
        F[:, eidx[j, u]] += n_g[j] / ng_event[eidx[j, u]] * col
        if post[j, u]:
            F[:, nE] += n_g[j] / ng_post * col
        F[cohort_of == j, eidx[j, u]] += kapE[j, u]
        F[cohort_of == j, nE] += kapS_cell[j, u]

    # 5. Passes over the units (controls, then the treated units grouped by cohort).
    rmw1, rmw0 = 1.0 / mw1, 1.0 / mw0
    Ftr = F[tr] if psi else np.zeros((0, nE + 1))
    FbC = _lanes(F[ctrl])[0] if psi else np.zeros((0, nE + 1, FB))
    S1, S2, A1, A2 = _fused_ctrl(EBC, XBC, PSB, W0B, FbC, gper, xcell, rmw1, rmw0, a1, a2, A3,
                                 etaC, piE, piS, emin, nE, npre, 8 * threads)
    EBT, bcoh, VB = _lanes(E[tr], toff)
    XBT = _lanes(Xc[tr], toff)[0]
    PSBT = _lanes(pst[:, None], toff)[0][:, 0, :]
    FbT = _lanes(Ftr, toff)[0] if psi else np.zeros((0, nE + 1, FB))
    t1, t2, t3, t4 = _fused_treated(EBT, XBT, PSBT, VB, bcoh, FbT, gper, xcell, rmw1, rmw0, A3,
                                    etaT, piE, piS, kapE, kapS, emin, nE, npre, 8 * threads)
    S1, S2, A1, A2 = S1 + t1, S2 + t2, A1 + t3, A2 + t4

    # 6. Outputs.
    nsb = n_s[:, None]
    att_gt = att + S1 / nsb
    se_gt = np.sqrt(np.maximum(S2 - S1 * S1 / nsb, 0.0)) / nsb
    for (j, u), m in att_x.items():
        att_gt[j, u] = m
        se_gt[j, u] = se_x[j, u]
    # aggregates use the reported cell ATTs, as the reference does
    att_s = (att_gt * cell_ng)[post].sum() / ng_post
    att_e = np.bincount(eidx.ravel(), weights=(att_gt * cell_ng).ravel(), minlength=nE) / ng_event
    mean = A1 / N
    se = np.sqrt(np.maximum(A2 - N * mean * mean, 0.0)) / N
    att_event = att_e + mean[:nE]
    gt = np.column_stack([np.repeat(gper, nc), np.tile(np.arange(2, T + 1), ncoh)]).astype(np.int64)
    return dict(gt=gt, att_gt=att_gt.ravel(), se_gt=se_gt.ravel(),
                att_simple=float(att_s + mean[nE]), se_simple=float(se[nE]),
                event_times=events.astype(np.int64), att_event=att_event, se_event=se[:nE].copy(),
                att_pre_avg=float(att_event[:npre].mean()) if npre else float("nan"),
                se_pre_avg=float(se[nE + 1]) if npre else float("nan"),
                att_post_avg=float(att_event[npre:].mean()) if nE > npre else float("nan"),
                se_post_avg=float(se[nE + 2]) if nE > npre else float("nan"))
