"""Task 002 candidate "cell-fused" (version 3, for the combined task of 2026-10-25): Callaway &
Sant'Anna (2021) staggered difference-in-differences as reference.run computes it under csdid
1.81's or 2.0.1's rules: never or not-yet-treated controls, sampling weights, unbalanced panels
with pair-balanced cells, the dripw, reg, ipw, stdipw and drimp estimators, varying or universal
base periods, propensity-score trimming, covariates fixed or varying over time, csdid 2.0.1's
overlap check, the simple, group, calendar and event-study aggregates, event-time subsets, the
event-study covariance, influence-function standard errors and the multiplier bootstrap with the
given +1/-1 multipliers.

The reference estimates every (cohort, period) cell as a separate small problem, stores every
cell's full-sample influence function and cohort-weight column in two n_units x n_cells arrays,
aggregates them, and multiplies all the influence functions by the multipliers. This candidate
computes the same quantities in four passes:

1. Cell list and counts (parallel over cells): which units each cell uses (cohort plus
   comparison units observed in both of the cell's periods), the cell sizes and weight sums.
2. Estimation (parallel over cohorts, the cohort's cells in order): csdid 2.0.1's overlap check
   (an unweighted logit; with unit weights it is the propensity fit itself), the propensity score
   (weighted logit, or inverse probability tilting for drimp) and the outcome regression, by
   Newton-Raphson with the reference's stopping rule. In a balanced panel, consecutive cells of a
   cohort often use exactly the same units and covariates (with never-treated controls and fixed
   covariates, every cell of the cohort does) and then share one fit; other fits start from the
   previous cell's solution. Each cell is reduced to a few vectors and scalars from which any
   unit's influence function is a short expression (a few dot products with its covariates).
   Cells left unestimated by the overlap check are removed; the aggregation weights follow.
3. One fused pass over units (parallel over blocks of 64 units of one cohort): for each unit,
   every cell it belongs to contributes its influence-function value to the cell's mean and sum
   of squares and, with the aggregation weights, to the unit's influence functions of the simple,
   group, calendar and event-study aggregates (including the terms for the estimated cohort
   weights). No n_units x n_cells array is formed.
4. Bootstrap (with multipliers): the pass over units runs in chunks of one cohort's units; each
   chunk's cell influence functions (only the cells the cohort belongs to) and aggregate influence
   functions are multiplied by the chunk's multipliers with BLAS and accumulated; bootstrap
   standard errors and critical values from order statistics (selection, not sorting).

Cells whose systems are singular, whose fits do not converge, whose scores lie within 1e-9 of
the trimming or overlap cut, or whose overlap fit is extreme are recomputed with the reference's
own NumPy operations (the reference raises where its own fits do not converge).

Threads: Numba parallel loops (NUMBA_NUM_THREADS, set by the evaluator's --threads) and BLAS (the
evaluator's limits), in separate phases.
"""
import math
from statistics import NormalDist

import numpy as np
import numba as nb
from numba import types
from numba.extending import intrinsic

CAP = 1.0 - 1e-6             # drdid caps the propensity score below 1
TOL = 1e-10                  # the reference's Newton stopping rule (largest step)
METHODS = ("dripw", "reg", "ipw", "stdipw", "drimp")
RULES = ("csdid181", "csdid201")
DRIPW, REG, IPW, STDIPW, DRIMP = range(5)
OK, EXACT, SKIP = 0, 1, 3    # cell states: estimated here, recomputed exactly, not estimated (overlap)
OVERLAP = 0.999              # csdid 2.0.1's overlap cut on the unweighted logit score
NEAR = 1e-9                  # a score this close to the trimming or overlap cut: exact path
BIG_INDEX = 25.0             # a warm-started overlap fit this extreme is checked from the reference's start
BLK = 256                    # units per block in the vectorized passes
LANE = 64                    # units per block (vector lanes) in the pass over units
CHUNK = 32                   # blocks of units per bootstrap product
FAST = {"reassoc", "contract"}
IQR_NORM = NormalDist().inv_cdf(0.75) - NormalDist().inv_cdf(0.25)


# ----------------------------------------------------------------------------------------------
# Reference operations (NumPy), for the rare cells recomputed exactly
# ----------------------------------------------------------------------------------------------

def _r_expit(v):
    return 0.5 * (1.0 + np.tanh(0.5 * v))


def _r_solve(A, b):
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A) @ b


def _r_inv(A):
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A)


def _r_logit(D, X, w, tol=1e-10, maxiter=100):
    b = np.zeros(X.shape[1])
    for _ in range(maxiter):
        p = _r_expit(X @ b)
        H = (X * (w * p * (1 - p))[:, None]).T @ X
        step = _r_solve(H, X.T @ (w * (D - p)))
        b += step
        if np.max(np.abs(step)) < tol:
            break
    else:
        raise RuntimeError("logit did not converge")
    p = _r_expit(X @ b)
    return X @ b, _r_inv((X * (w * p * (1 - p))[:, None]).T @ X)


def _r_ipt(D, X, w, tol=1e-10, maxiter=200):
    b = np.zeros(X.shape[1])
    b[-1] = np.log(np.sum(w * D) / np.sum(w * (1 - D)))
    obj = lambda v: np.sum(w * (D * (X @ v) - (1 - D) * np.exp(X @ v)))
    for _ in range(maxiter):
        e = np.exp(X @ b)
        g = X.T @ (w * (D - (1 - D) * e))
        H = (X * (w * (1 - D) * e)[:, None]).T @ X
        step = _r_solve(H, g)
        f0, s = obj(b), 1.0
        with np.errstate(over="ignore", invalid="ignore"):
            while not obj(b + s * step) >= f0 - 1e-12 * abs(f0) and s > 1e-8:
                s /= 2
        b += s * step
        if np.max(np.abs(s * step)) < tol:
            break
    else:
        raise RuntimeError("inverse probability tilting did not converge")
    return X @ b


def _r_wls(dy, X, sw):
    XtWX = (X * sw[:, None]).T @ X
    inv = _r_inv(XtWX)
    return inv @ (X.T @ (sw * dy)), inv


def _r_cell_rif(dy, X, D, w, method, trim):
    """reference.cell_rif (identical NumPy operations)."""
    n = dy.size
    w1 = w * D
    if method == "reg":
        b, inv = _r_wls(dy[D == 0], X[D == 0], w[D == 0])
        xb = X @ b
        eta_t, eta_c = np.mean(w1 * dy) / w1.mean(), np.mean(w1 * xb) / w1.mean()
        lin_ols = ((w * (1 - D) * (dy - xb))[:, None] * X) @ (inv * n)
        inf_t = (w1 * dy - w1 * eta_t) / w1.mean()
        inf_c = (w1 * xb - w1 * eta_c + lin_ols @ (w1[:, None] * X).mean(0)) / w1.mean()
        return (eta_t - eta_c) + inf_t - inf_c
    if method == "drimp":
        index = _r_ipt(D, X, w)
    else:
        index, V = _r_logit(D, X, w)
    ps_raw = _r_expit(index)
    ps = np.minimum(ps_raw, CAP)
    keep = ~((D == 0) & (ps_raw > trim))
    if method == "drimp":
        w0r = (1 - D) * ps / (1 - ps) * w
        b, _ = _r_wls(dy[D == 0], X[D == 0], w0r[D == 0])
        resid = dy - X @ b
        summand = keep * (1 - (1 - D) / (1 - ps)) * resid
        att = np.mean(w * summand) / w1.mean()
        return att + w * keep * (summand - D * att) / w1.mean()
    w0 = w * ps * (1 - D) / (1 - ps) * keep
    lin_ps = ((w * (D - ps))[:, None] * X) @ (V * n)
    if method == "ipw":
        att = np.mean((w1 - w0) * dy) / w1.mean()
        lin2 = lin_ps @ ((w0 * dy)[:, None] * X).mean(0)
        return att + (w1 * dy - w0 * dy - lin2 - w1 * att) / w1.mean()
    if method == "stdipw":
        eta_t, eta_c = np.mean(w1 * dy) / w1.mean(), np.mean(w0 * dy) / w0.mean()
        inf_t = (w1 * dy - w1 * eta_t) / w1.mean()
        inf_c = (w0 * dy - w0 * eta_c + lin_ps @ ((w0 * (dy - eta_c))[:, None] * X).mean(0)) / w0.mean()
        return (eta_t - eta_c) + inf_t - inf_c
    wo = w * (1 - D)
    b, inv = _r_wls(dy[D == 0], X[D == 0], w[D == 0])
    r = dy - X @ b
    eta_t, eta_c = np.mean(w1 * r) / w1.mean(), np.mean(w0 * r) / w0.mean()
    lin_ols = ((wo * r)[:, None] * X) @ (inv * n)
    inf_t = (w1 * r - w1 * eta_t - lin_ols @ (w1[:, None] * X).mean(0)) / w1.mean()
    inf_c = (w0 * r - w0 * eta_c + lin_ps @ ((w0 * (r - eta_c))[:, None] * X).mean(0)
             - lin_ols @ (w0[:, None] * X).mean(0)) / w0.mean()
    return (eta_t - eta_c) + inf_t - inf_c


def _r_overlap_fails(D, X, cut=OVERLAP):
    """reference.overlap_fails (identical NumPy operations)."""
    if X.shape[1] == 1:
        return D.mean() >= cut
    try:
        index, _ = _r_logit(D, X, np.ones(D.size))
    except RuntimeError:
        return True
    return _r_expit(index).max() >= cut


def _sample(y, gvar, control):
    """reference.sample: units kept, observed mask [kept units, T]."""
    obs = np.isfinite(y) & ((gvar == 0) | (gvar > 1))[:, None]
    seen = np.flatnonzero(obs[gvar == 0].any(0))
    cut = seen.max() + 1 if seen.size else 0
    if control == "notyet":
        cut = max(cut, int(gvar[obs.any(1)].max()))
    obs[:, cut:] = False
    units = obs.any(1)
    return units, obs[units]


# ----------------------------------------------------------------------------------------------
# Scalar kernels
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
    Cody-Waite reduction by ln 2, degree-13 Taylor polynomial, exponent set from integer bits.
    Must not be compiled with reassociation (the rounding trick relies on exact rounding)."""
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


@nb.njit(inline="always", error_model="numpy")
def _expit(v):
    return 1.0 / (1.0 + _exp(-v))


@nb.njit(inline="always", error_model="numpy")
def _member(Gi, g, m, notyet):
    return Gi == g or Gi == 0 or (notyet and Gi > m)


# ----------------------------------------------------------------------------------------------
# Small dense linear algebra (k + 1 <= a dozen unknowns)
# ----------------------------------------------------------------------------------------------

@nb.njit(error_model="numpy")
def _chol(A, L):
    """Cholesky factor of the symmetric matrix A into L; False unless A is numerically positive definite."""
    k = A.shape[0]
    for j in range(k):
        s = A[j, j]
        for q in range(j):
            s -= L[j, q] * L[j, q]
        if not (s > 1e-12 * abs(A[j, j])) or not math.isfinite(s):   # near-singular: exact path instead
            return False
        L[j, j] = math.sqrt(s)
        for i in range(j + 1, k):
            s = A[i, j]
            for q in range(j):
                s -= L[i, q] * L[j, q]
            L[i, j] = s / L[j, j]
    return True


@nb.njit(error_model="numpy")
def _chol_solve(L, b, x):
    k = L.shape[0]
    for i in range(k):
        s = b[i]
        for q in range(i):
            s -= L[i, q] * x[q]
        x[i] = s / L[i, i]
    for i in range(k - 1, -1, -1):
        s = x[i]
        for q in range(i + 1, k):
            s -= L[q, i] * x[q]
        x[i] = s / L[i, i]


@nb.njit(error_model="numpy")
def _chol_inv(L, out):
    k = L.shape[0]
    e = np.zeros(k)
    col = np.empty(k)
    for j in range(k):
        e[:] = 0.0
        e[j] = 1.0
        _chol_solve(L, e, col)
        for i in range(k):
            out[i, j] = col[i]


@nb.njit(error_model="numpy")
def _matvec(A, v, out):
    k = A.shape[0]
    for i in range(k):
        s = 0.0
        for j in range(k):
            s += A[i, j] * v[j]
        out[i] = s


# ----------------------------------------------------------------------------------------------
# Pass 1: cells and their sizes
# ----------------------------------------------------------------------------------------------

@nb.njit(parallel=True, error_model="numpy")
def _count_cells(G, obs, w, cg, ct, cb, cm, notyet):
    C, N = cg.size, G.size
    n = np.zeros(C, np.int64)
    n1 = np.zeros(C, np.int64)
    sw = np.zeros(C)
    sw1 = np.zeros(C)
    for c in nb.prange(C):
        g, t, b, m = cg[c], ct[c] - 1, cb[c] - 1, cm[c]
        cn, cn1, s, s1 = 0, 0, 0.0, 0.0
        for i in range(N):
            Gi = G[i]
            if _member(Gi, g, m, notyet) and obs[i, b] and obs[i, t]:
                cn += 1
                s += w[i]
                if Gi == g:
                    cn1 += 1
                    s1 += w[i]
        n[c], n1[c], sw[c], sw1[c] = cn, cn1, s, s1
    return n, n1, sw, sw1


def _cell_counts(gvar, w, obs8, cg, ct, cb, cm, notyet):
    """Cell sizes and weight sums (all units, cohort units). Cells whose two periods are observed
    for every unit (or for none) are counted from cohort tallies; the others unit by unit."""
    full, none = obs8.all(0), ~obs8.any(0)
    vals, inv = np.unique(gvar, return_inverse=True)
    cnt = np.bincount(inv, minlength=vals.size).astype(np.int64)
    wsum = np.bincount(inv, weights=w, minlength=vals.size)
    c0 = int(cnt[0]) if vals[0] == 0 else 0
    w0 = float(wsum[0]) if vals[0] == 0 else 0.0
    suf_n = np.append(np.cumsum(cnt[::-1])[::-1], 0)    # units of cohort values from index i on
    suf_w = np.append(np.cumsum(wsum[::-1])[::-1], 0.0)
    gi = np.searchsorted(vals, cg)
    n1 = cnt[gi]
    sw1 = wsum[gi]
    n = n1 + c0
    sw = sw1 + w0
    if notyet:
        mi = np.searchsorted(vals, cm, side="right")    # first value above the threshold
        inside = cg > cm                                # the cohort itself is above the threshold
        n = n + suf_n[mi] - np.where(inside, n1, 0)
        sw = sw + suf_w[mi] - np.where(inside, sw1, 0.0)
    tb_full = full[ct - 1] & full[cb - 1]
    tb_none = none[ct - 1] | none[cb - 1]
    n, n1, sw, sw1 = (np.where(tb_full, v, 0) for v in (n, n1, sw, sw1))
    rest = np.flatnonzero(~tb_full & ~tb_none)
    if rest.size:
        rn, rn1, rsw, rsw1 = _count_cells(gvar, obs8, w, cg[rest], ct[rest], cb[rest], cm[rest], notyet)
        n[rest], n1[rest], sw[rest], sw1[rest] = rn, rn1, rsw, rsw1
    return n.astype(np.int64), n1.astype(np.int64), sw.astype(np.float64), sw1.astype(np.float64)


# ----------------------------------------------------------------------------------------------
# Pass 2: estimation
# ----------------------------------------------------------------------------------------------

@nb.njit(error_model="numpy")
def _gather(G, obs, y, X3, w, g, t, b, p, m, notyet, wbar, idx, Xg, D, ws, dy):
    N = G.size
    k1 = X3.shape[2]
    j = 0
    for i in range(N):
        Gi = G[i]
        if _member(Gi, g, m, notyet) and obs[i, b] and obs[i, t]:
            idx[j] = i
            for k in range(k1):
                Xg[k, j] = X3[p, i, k]
            D[j] = 1.0 if Gi == g else 0.0
            ws[j] = w[i] / wbar
            dy[j] = y[i, t] - y[i, b]
            j += 1


@nb.njit(error_model="numpy")
def _regather_dy(y, idx, t, b, dy):
    for j in range(idx.size):
        i = idx[j]
        dy[j] = y[i, t] - y[i, b]


@nb.njit(error_model="numpy", fastmath=FAST)
def _index(Xg, beta, lo, m, eta):
    k1 = Xg.shape[0]
    for j in range(m):
        eta[j] = 0.0
    for k in range(k1):
        bk = beta[k]
        xk = Xg[k, lo:lo + m]
        for j in range(m):
            eta[j] += xk[j] * bk


@nb.njit(error_model="numpy")
def _logit_weights(eta, D, ws, lo, m, wp, rr):
    for j in range(m):
        p = _expit(eta[j])
        wj = ws[lo + j]
        wp[j] = wj * p * (1.0 - p)
        rr[j] = wj * (D[lo + j] - p)


@nb.njit(error_model="numpy")
def _ipt_weights(eta, D, ws, lo, m, wp, rr):
    for j in range(m):
        e = _exp(eta[j])
        wj = ws[lo + j]
        c = 1.0 - D[lo + j]
        wp[j] = wj * c * e
        rr[j] = wj * (D[lo + j] - c * e)


@nb.njit(error_model="numpy", fastmath=FAST)
def _accumulate(Xg, lo, m, wp, rr, H, g):
    k1 = Xg.shape[0]
    for a in range(k1):
        xa = Xg[a, lo:lo + m]
        s = 0.0
        for j in range(m):
            s += rr[j] * xa[j]
        g[a] += s
        for c in range(a + 1):
            xc = Xg[c, lo:lo + m]
            s = 0.0
            for j in range(m):
                s += wp[j] * xa[j] * xc[j]
            H[a, c] += s


@nb.njit(error_model="numpy")
def _hess_grad(Xg, D, ws, beta, tilt, H, g, eta, wp, rr):
    """Hessian (positive definite form) and gradient of the weighted logit (tilt False) or of the
    inverse-probability-tilting objective sum w [D x'b - (1 - D) exp(x'b)] (tilt True)."""
    k1, n = Xg.shape
    H[:, :] = 0.0
    g[:] = 0.0
    for lo in range(0, n, BLK):
        m = min(BLK, n - lo)
        _index(Xg, beta, lo, m, eta)
        if tilt:
            _ipt_weights(eta, D, ws, lo, m, wp, rr)
        else:
            _logit_weights(eta, D, ws, lo, m, wp, rr)
        _accumulate(Xg, lo, m, wp, rr, H, g)
    for a in range(k1):
        for c in range(a):
            H[c, a] = H[a, c]


@nb.njit(error_model="numpy")
def _ipt_objective(Xg, D, ws, v, eta):
    k1, n = Xg.shape
    f = 0.0
    for lo in range(0, n, BLK):
        m = min(BLK, n - lo)
        _index(Xg, v, lo, m, eta)
        for j in range(m):
            d = D[lo + j]
            f += ws[lo + j] * (d * eta[j] - (1.0 - d) * math.exp(eta[j]))
    return f


@nb.njit(error_model="numpy")
def _newton(Xg, D, ws, beta, tilt, maxiter, V):
    """Newton-Raphson from beta (updated in place) with the reference's stopping rule (largest step
    below TOL); for tilting, the reference's step halving. Returns 0 (converged), 1 (not converged)
    or 2 (singular or non-finite). For the logit, V receives the inverse Hessian of the last
    iteration (evaluated less than TOL from the solution; the reference evaluates it at the
    solution, a relative difference of order TOL)."""
    k1, n = Xg.shape
    H = np.empty((k1, k1))
    L = np.empty((k1, k1))
    g = np.empty(k1)
    step = np.empty(k1)
    trial = np.empty(k1)
    eta = np.empty(BLK)
    wp = np.empty(BLK)
    rr = np.empty(BLK)
    status = 1
    for _ in range(maxiter):
        _hess_grad(Xg, D, ws, beta, tilt, H, g, eta, wp, rr)
        if not _chol(H, L):
            return 2
        _chol_solve(L, g, step)
        s = 1.0
        if tilt:
            f0 = _ipt_objective(Xg, D, ws, beta, eta)
            while True:
                for k in range(k1):
                    trial[k] = beta[k] + s * step[k]
                f = _ipt_objective(Xg, D, ws, trial, eta)
                if f >= f0 - 1e-12 * abs(f0) or not s > 1e-8:
                    break
                s /= 2
        mx = 0.0
        for k in range(k1):
            if not math.isfinite(step[k]):
                return 2
            beta[k] += s * step[k]
            mx = max(mx, abs(s * step[k]))
        if mx < TOL:
            status = 0
            break
    if status == 0 and not tilt:
        _chol_inv(L, V)
    return status


@nb.njit(error_model="numpy", fastmath=FAST)
def _wls_matrix(Xg, D, sw, A):
    """A = sum over comparison units (D == 0) of sw x x'."""
    k1, n = Xg.shape
    A[:, :] = 0.0
    cw = np.empty(n)
    for j in range(n):
        cw[j] = (1.0 - D[j]) * sw[j]
    for a in range(k1):
        xa = Xg[a]
        for c in range(a + 1):
            xc = Xg[c]
            s = 0.0
            for j in range(n):
                s += cw[j] * xa[j] * xc[j]
            A[a, c] = s
            A[c, a] = s


@nb.njit(error_model="numpy", fastmath=FAST)
def _wls_rhs(Xg, D, sw, dy, rhs):
    k1, n = Xg.shape
    cw = np.empty(n)
    for j in range(n):
        cw[j] = (1.0 - D[j]) * sw[j] * dy[j]
    for a in range(k1):
        xa = Xg[a]
        s = 0.0
        for j in range(n):
            s += cw[j] * xa[j]
        rhs[a] = s


@nb.njit(error_model="numpy")
def _scores(Xg, bp, D, trim, ps, keep):
    """Capped propensity scores and trimming flags (comparison units scoring above trim) of the
    gathered units. Returns the largest raw score, the largest absolute index, and whether a
    comparison unit's raw score lies within NEAR of the trimming cut."""
    k1, n = Xg.shape
    eta = np.empty(BLK)
    mx = 0.0
    big = 0.0
    near = False
    for lo in range(0, n, BLK):
        m = min(BLK, n - lo)
        _index(Xg, bp, lo, m, eta)
        for j in range(m):
            raw = _expit(eta[j])
            mx = max(mx, raw)
            big = max(big, abs(eta[j]))
            ps[lo + j] = min(raw, CAP)
            kj = 1.0
            if D[lo + j] == 0.0 and trim < 1.0:
                if raw > trim:
                    kj = 0.0
                if abs(raw - trim) <= NEAR:
                    near = True
            keep[lo + j] = kj
    return mx, big, near


@nb.njit(error_model="numpy")
def _overlap_state(mx):
    if abs(mx - OVERLAP) <= NEAR:
        return EXACT
    return SKIP if mx >= OVERLAP else OK


@nb.njit(error_model="numpy")
def _share_state(D):
    """Overlap check without covariates: the cohort's share of the cell's units."""
    s = 0.0
    for j in range(D.size):
        s += D[j]
    return SKIP if s / D.size >= OVERLAP else OK


@nb.njit(error_model="numpy")
def _overlap_fit(Xg, D, bo, warm, Vo):
    """csdid 2.0.1's overlap check by its own unweighted logit (warm-started from bo, updated on
    success). Returns (state, fitted). A fit that fails here, or a warm-started fit whose index is
    extreme and that does not converge from the reference's start, is left to the exact path."""
    k1, n = Xg.shape
    ones = np.ones(n)
    trial = bo.copy()
    if not warm:
        trial[:] = 0.0
    status = _newton(Xg, D, ones, trial, False, 50 if warm else 100, Vo)
    cold = not warm
    if status != 0 and warm:
        trial[:] = 0.0
        status = _newton(Xg, D, ones, trial, False, 100, Vo)
        cold = True
    if status != 0:
        return EXACT, False
    ps = np.empty(n)
    keep = np.empty(n)
    mx, big, _ = _scores(Xg, trial, D, 1.0, ps, keep)
    if big > BIG_INDEX and not cold:
        check = np.zeros(k1)
        if _newton(Xg, D, ones, check, False, 100, Vo) != 0:
            return EXACT, False
    bo[:] = trial
    return _overlap_state(mx), True


@nb.njit(error_model="numpy", fastmath=FAST)
def _constants(method, Xg, D, ws, dy, ps, keep, br, inv, V, U, V2, sc):
    """Cell constants of the influence function (see _units). sc: a (ATT), eta_t, eta_c, mean w1,
    mean w0. Per-unit quantities first, then sums over units for each covariate (contiguous, so
    the loops vectorize). keep: 0 for trimmed comparison units."""
    k1, n = Xg.shape
    xb = np.zeros(n)
    if method == REG or method == DRIPW or method == DRIMP:
        for k in range(k1):
            bk = br[k]
            xk = Xg[k]
            for j in range(n):
                xb[j] += xk[j] * bk
    w1 = np.empty(n)
    w0 = np.empty(n)
    v = np.empty(n)
    s_w1 = s_w0 = s_a = s_b = 0.0
    for j in range(n):
        d = D[j]
        wj = ws[j]
        w1j = wj * d
        w1[j] = w1j
        if method == REG:
            w0j = 0.0
            vj = dy[j]
            s_a += w1j * vj
            s_b += w1j * xb[j]
        elif method == DRIMP:
            w0j = 0.0
            vj = dy[j] - xb[j]
            s_a += w1j * vj
            s_b += keep[j] * (1.0 - d) * wj * ps[j] / (1.0 - ps[j]) * vj
        else:
            w0j = keep[j] * wj * ps[j] * (1.0 - d) / (1.0 - ps[j])
            vj = dy[j] - xb[j] if method == DRIPW else dy[j]
            s_a += w1j * vj
            s_b += w0j * vj
        w0[j] = w0j
        v[j] = vj
        s_w1 += w1j
        s_w0 += w0j
    M1 = np.zeros(k1)
    M3 = np.zeros(k1)
    Q = np.zeros(k1)
    if method == REG or method == DRIPW:
        for k in range(k1):
            xk = Xg[k]
            acc = 0.0
            for j in range(n):
                acc += w1[j] * xk[j]
            M1[k] = acc
    if method == IPW or method == STDIPW or method == DRIPW:
        w0v = w0 * v
        for k in range(k1):
            xk = Xg[k]
            a3 = aq = 0.0
            for j in range(n):
                a3 += w0[j] * xk[j]
                aq += w0v[j] * xk[j]
            M3[k] = a3
            Q[k] = aq
    mw1, mw0 = s_w1 / n, s_w0 / n
    U[:] = 0.0
    V2[:] = 0.0
    if method == REG or method == DRIMP or method == IPW:
        eta_t, eta_c = s_a / s_w1, s_b / s_w1
    else:
        eta_t, eta_c = s_a / s_w1, s_b / s_w0
    if method == REG:
        _matvec(inv, M1, U)
    elif method == IPW:
        _matvec(V, Q, V2)
    elif method == STDIPW or method == DRIPW:
        tmp = np.empty(k1)
        for k in range(k1):
            tmp[k] = Q[k] - eta_c * M3[k]
        _matvec(V, tmp, V2)
        if method == DRIPW:
            v1 = np.empty(k1)
            v3 = np.empty(k1)
            _matvec(inv, M1, v1)
            _matvec(inv, M3, v3)
            for k in range(k1):
                U[k] = v3[k] / mw0 - v1[k] / mw1
    sc[0], sc[1], sc[2], sc[3], sc[4] = eta_t - eta_c, eta_t, eta_c, mw1, mw0


@nb.njit(error_model="numpy", fastmath=FAST)
def _invariants(method, Xg, D, ws, ps, keep, w1, w0, M1, M3, Mxx):
    """For cells sharing one fit: the per-unit weights w1 = ws D and w0 (ws ps / (1 - ps) for the
    comparison units kept after trimming: the comparison weights, or drimp's comparison factor)
    and the sums that do not depend on the outcome change: returns (sum w1, sum w0) and fills
    M1 = sum w1 x, M3 = sum w0 x and, for dripw, Mxx = sum w0 x x'."""
    k1, n = Xg.shape
    s1 = s0 = 0.0
    for j in range(n):
        d = D[j]
        wj = ws[j]
        w1[j] = wj * d
        w0[j] = 0.0 if method == REG else keep[j] * wj * ps[j] * (1.0 - d) / (1.0 - ps[j])
        s1 += w1[j]
        s0 += w0[j]
    for a in range(k1):
        xa = Xg[a]
        m1 = m3 = 0.0
        for j in range(n):
            m1 += w1[j] * xa[j]
            m3 += w0[j] * xa[j]
        M1[a] = m1
        M3[a] = m3
        if method == DRIPW:
            for c in range(a + 1):
                xc = Xg[c]
                acc = 0.0
                for j in range(n):
                    acc += w0[j] * xa[j] * xc[j]
                Mxx[a, c] = acc
                Mxx[c, a] = acc
    return s1, s0


@nb.njit(error_model="numpy", fastmath=FAST)
def _dy_sums(Xg, D, sw, w1, w0, dy, q, rhs, t0, t1):
    """The sums that depend on the outcome change: returns (sum w1 dy, sum w0 dy) and fills
    q = sum w0 dy x and rhs = sum over comparison units of sw dy x (the regression's right side)."""
    k1, n = Xg.shape
    a1 = a0 = 0.0
    for j in range(n):
        dj = dy[j]
        a1 += w1[j] * dj
        a0 += w0[j] * dj
        t0[j] = w0[j] * dj
        t1[j] = (1.0 - D[j]) * sw[j] * dj
    for k in range(k1):
        xk = Xg[k]
        sq = sr = 0.0
        for j in range(n):
            sq += t0[j] * xk[j]
            sr += t1[j] * xk[j]
        q[k] = sq
        rhs[k] = sr
    return a1, a0


@nb.njit(error_model="numpy")
def _shared_constants(method, n, s1, s0, M1, M3, Mxx, a1, a0, q, br, inv, V, U, V2, sc):
    """_constants from the shared sums (the same quantities: v = dy - x'br is expanded, e.g.
    sum w0 v x = q - Mxx br)."""
    k1 = M1.size
    mw1, mw0 = s1 / n, s0 / n
    U[:] = 0.0
    V2[:] = 0.0
    m1b = m3b = 0.0
    for k in range(k1):
        m1b += M1[k] * br[k]
        m3b += M3[k] * br[k]
    tmp = np.empty(k1)
    if method == REG:
        eta_t, eta_c = a1 / s1, m1b / s1
        _matvec(inv, M1, U)
    elif method == IPW:
        eta_t, eta_c = a1 / s1, a0 / s1
        _matvec(V, q, V2)
    elif method == STDIPW:
        eta_t, eta_c = a1 / s1, a0 / s0
        for k in range(k1):
            tmp[k] = q[k] - eta_c * M3[k]
        _matvec(V, tmp, V2)
    elif method == DRIMP:
        eta_t, eta_c = (a1 - m1b) / s1, (a0 - m3b) / s1
    else:
        eta_t, eta_c = (a1 - m1b) / s1, (a0 - m3b) / s0
        for k in range(k1):
            acc = q[k] - eta_c * M3[k]
            for c in range(k1):
                acc -= Mxx[k, c] * br[c]
            tmp[k] = acc
        _matvec(V, tmp, V2)
        v1 = np.empty(k1)
        v3 = np.empty(k1)
        _matvec(inv, M1, v1)
        _matvec(inv, M3, v3)
        for k in range(k1):
            U[k] = v3[k] / mw0 - v1[k] / mw1
    sc[0], sc[1], sc[2], sc[3], sc[4] = eta_t - eta_c, eta_t, eta_c, mw1, mw0


@nb.njit(parallel=True, error_model="numpy")
def _estimate(G, obs, y, X3, w, cg, ct, cb, cp, cm, cn, csw, key, coh_ptr, method, notyet, trim, overlap,
              unweighted):
    """Cell constants for every cell; cells are processed per cohort, in order, so that cells with
    the same units and covariates (equal key) share their fits and other fits start from the
    previous solution. flag: OK, EXACT (recompute with the reference's operations) or SKIP (csdid
    2.0.1's overlap check fails)."""
    C = cg.size
    k1 = X3.shape[2]
    BR = np.zeros((C, k1))
    BP = np.zeros((C, k1))
    U = np.zeros((C, k1))
    V2 = np.zeros((C, k1))
    SC = np.zeros((C, 7))
    flag = np.zeros(C, np.int64)
    tilt = method == DRIMP
    for h in nb.prange(coh_ptr.size - 1):
        bp = np.zeros(k1)
        bo = np.zeros(k1)
        V = np.zeros((k1, k1))
        Vo = np.zeros((k1, k1))
        A = np.zeros((k1, k1))
        L = np.zeros((k1, k1))
        inv = np.zeros((k1, k1))
        rhs = np.zeros(k1)
        br = np.zeros(k1)
        warm = False
        warm_o = False
        prev_key = -1
        have = False
        state = OK
        idx = np.empty(0, np.int64)
        Xg = np.empty((k1, 0))
        D = np.empty(0)
        ws = np.empty(0)
        dy = np.empty(0)
        ps = np.empty(0)
        keep = np.empty(0)
        sw0r = np.empty(0)
        hoist = False
        w1a = np.empty(0)
        w0a = np.empty(0)
        t0 = np.empty(0)
        t1 = np.empty(0)
        M1 = np.zeros(k1)
        M3 = np.zeros(k1)
        Mxx = np.zeros((k1, k1))
        q = np.zeros(k1)
        s1h = s0h = 0.0
        for c in range(coh_ptr[h], coh_ptr[h + 1]):
            n = cn[c]
            t, b = ct[c] - 1, cb[c] - 1
            shared = have and key[c] >= 0 and key[c] == prev_key
            prev_key = key[c]
            if shared:
                if state != OK:
                    flag[c] = state
                    continue
                _regather_dy(y, idx, t, b, dy)
            else:
                have = True
                idx = np.empty(n, np.int64)
                Xg = np.empty((k1, n))
                D = np.empty(n)
                ws = np.empty(n)
                dy = np.empty(n)
                ps = np.ones(n)
                keep = np.ones(n)
                sw0r = np.empty(n)
                _gather(G, obs, y, X3, w, cg[c], t, b, cp[c], cm[c], notyet, csw[c] / n, idx, Xg, D, ws, dy)
                state = OK
                if overlap and (tilt or not unweighted):      # the overlap check needs its own logit
                    if k1 == 1:
                        state = _share_state(D)
                    else:
                        state, fitted = _overlap_fit(Xg, D, bo, warm_o, Vo)
                        if fitted:
                            warm_o = True
                if state == OK and method != REG:
                    trial = bp.copy()
                    s1 = s0 = 0.0
                    if tilt:
                        for j in range(n):
                            s1 += ws[j] * D[j]
                            s0 += ws[j] * (1.0 - D[j])
                    if not warm:
                        trial[:] = 0.0
                        if tilt:
                            trial[k1 - 1] = math.log(s1 / s0)
                    status = _newton(Xg, D, ws, trial, tilt, 50 if warm else (200 if tilt else 100), V)
                    cold = not warm
                    if status != 0 and warm:            # retry from the reference's starting point
                        trial[:] = 0.0
                        if tilt:
                            trial[k1 - 1] = math.log(s1 / s0)
                        status = _newton(Xg, D, ws, trial, tilt, 200 if tilt else 100, V)
                        cold = True
                    if status != 0:
                        state = EXACT
                    else:
                        bp[:] = trial
                        warm = True
                        mx, big, near = _scores(Xg, bp, D, trim, ps, keep)
                        if near:
                            state = EXACT
                        elif overlap and not tilt and unweighted:   # the fit is the overlap logit
                            if k1 == 1:
                                state = _share_state(D)
                            else:
                                if big > BIG_INDEX and not cold:
                                    check = np.zeros(k1)
                                    if _newton(Xg, D, ws, check, False, 100, Vo) != 0:
                                        state = EXACT
                                if state == OK:
                                    state = _overlap_state(mx)
                if state == OK and method != IPW and method != STDIPW:
                    if method == DRIMP:
                        for j in range(n):
                            sw0r[j] = ps[j] / (1.0 - ps[j]) * ws[j]
                        _wls_matrix(Xg, D, sw0r, A)
                    else:
                        _wls_matrix(Xg, D, ws, A)
                    if not _chol(A, L):
                        state = EXACT
                    else:
                        _chol_inv(L, inv)
                # the next cell shares this fit: hoist the sums that do not depend on the outcome
                hoist = state == OK and c + 1 < coh_ptr[h + 1] and key[c] >= 0 and key[c + 1] == key[c]
                if hoist:
                    w1a = np.empty(n)
                    w0a = np.empty(n)
                    t0 = np.empty(n)
                    t1 = np.empty(n)
                    s1h, s0h = _invariants(method, Xg, D, ws, ps, keep, w1a, w0a, M1, M3, Mxx)
            if state != OK:
                flag[c] = state
                continue
            if hoist:
                a1, a0 = _dy_sums(Xg, D, sw0r if method == DRIMP else ws, w1a, w0a, dy, q, rhs, t0, t1)
                if method != IPW and method != STDIPW:
                    _chol_solve(L, rhs, br)
                _shared_constants(method, n, s1h, s0h, M1, M3, Mxx, a1, a0, q, br, inv, V, U[c], V2[c], SC[c])
            else:
                if method != IPW and method != STDIPW:
                    _wls_rhs(Xg, D, sw0r if method == DRIMP else ws, dy, rhs)
                    _chol_solve(L, rhs, br)
                _constants(method, Xg, D, ws, dy, ps, keep, br, inv, V, U[c], V2[c], SC[c])
            SC[c, 6] = csw[c] / n
            BR[c] = br
            BP[c] = bp
    return BR, BP, U, V2, SC, flag


# ----------------------------------------------------------------------------------------------
# Pass 3: influence functions of every cell and every aggregate, one pass over units
# ----------------------------------------------------------------------------------------------

@nb.njit(parallel=True, error_model="numpy")
def _units(b0, b1, GB, XB, YB, OB, WB, VB, brow0, bval, cg, ct, cb, cp, cm, method, notyet, r201, trim,
           BR, BP, U, V2, SC, exrow, EXD, agg, alpha, beta, inv_s, Dm, n_agg, S1, S2, gcol, wcol, inv_n, buf):
    """For the blocks b0 .. b1 - 1 (LANE lanes of one cohort each) and every cell the block's cohort
    can belong to: each member unit's influence-function value r (reference.cell_rif, term by
    term; for cells recomputed exactly, the exact values in EXD), summed per cell around the
    cell's closed-form ATT a (sum and sum of squares), and the units' aggregate influence
    functions (the first n_agg columns of their rows of Dm, minus the aggregate ATTs): the sum
    over their cells of alpha (r - a) plus, for their own cohort's cells, beta w / S (under
    2.0.1's rules every unit of the cohort carries the cohort weight, observed in the cell or
    not). Consecutive cells with the same propensity fit and covariates reuse the lanes' scores.
    With a bootstrap buffer, the member cells' values (r - a) / n are also written to the chunk's
    rows (column gcol[c]) and, for the cohort's own cells when wcol[c] >= 0, the units' weights
    if observed in the cell. Lanes outside a cell, and padding lanes, contribute exactly zero."""
    P, k1, B = XB.shape[1], XB.shape[2], XB.shape[3]
    C = cg.size
    boot = buf.shape[0] > 0
    for blk in nb.prange(b0, b1):
        Gb = GB[blk]
        nv = bval[blk]
        r0 = brow0[blk]
        q0 = r0 - brow0[b0]
        acc = np.zeros((n_agg, B))
        xb = np.empty(B)
        xp = np.empty(B)
        xu = np.empty(B)
        xv = np.empty(B)
        ckp = np.empty(B)        # trimming flag
        cw0 = np.empty(B)        # kp (1 - d) ps / (1 - ps)
        cdp = np.empty(B)        # d - ps
        cim = np.empty(B)        # 1 - (1 - d) / (1 - ps)
        dev = np.empty(B)
        mem = np.empty(B)
        last = -1
        for c in range(C):
            g = cg[c]
            if not _member(Gb, g, cm[c], notyet):
                continue
            d = 1.0 if Gb == g else 0.0
            t, b = ct[c] - 1, cb[c] - 1
            cnt = 0.0
            for u in range(B):
                mem[u] = VB[blk, u] * OB[blk, u, t] * OB[blk, u, b]
                cnt += mem[u]
            if cnt == 0.0:
                if r201 and d == 1.0:
                    for q in range(4):
                        col = agg[c, q]
                        if col >= 0:
                            be = beta[c, q] * inv_s
                            for u in range(B):
                                acc[col, u] += be * WB[blk, u]
                if boot and gcol[c] >= 0:
                    j = gcol[c]
                    for u in range(nv):
                        buf[q0 + u, j] = 0.0
                if boot and wcol[c] >= 0:
                    j = wcol[c]
                    for u in range(nv):
                        buf[q0 + u, j] = 0.0
                continue
            e = exrow[c]
            if e >= 0:
                for u in range(B):
                    dev[u] = EXD[e, blk * B + u]
            else:
                p = cp[c] if P > 1 else 0
                same = last >= 0 and cg[last] == g and cp[last] == cp[c]
                if same:
                    for k in range(k1):
                        if BP[c, k] != BP[last, k] or U[c, k] != U[last, k]:
                            same = False
                if not same:
                    for u in range(B):
                        xp[u] = xu[u] = 0.0
                    for k in range(k1):
                        bpk, uk = BP[c, k], U[c, k]
                        for u in range(B):
                            x = XB[blk, p, k, u]
                            xp[u] += x * bpk
                            xu[u] += x * uk
                    if method != REG:
                        for u in range(B):
                            raw = _expit(xp[u])
                            ps = min(raw, CAP)
                            kp = 0.0 if (d == 0.0 and raw > trim) else 1.0
                            ckp[u] = kp
                            cw0[u] = kp * (1.0 - d) * ps / (1.0 - ps)
                            cdp[u] = d - ps
                            cim[u] = 1.0 - (1.0 - d) / (1.0 - ps)
                    last = c
                for u in range(B):
                    xb[u] = xv[u] = 0.0
                for k in range(k1):
                    brk, vk = BR[c, k], V2[c, k]
                    for u in range(B):
                        x = XB[blk, p, k, u]
                        xb[u] += x * brk
                        xv[u] += x * vk
                a, eta_t, eta_c, mw1, mw0, wbar = SC[c, 0], SC[c, 1], SC[c, 2], SC[c, 3], SC[c, 4], SC[c, 6]
                for u in range(B):
                    wsi = WB[blk, u] / wbar
                    dyi = YB[blk, u, t] - YB[blk, u, b]
                    w1 = wsi * d
                    if method == REG:
                        r = (a + (w1 * dyi - w1 * eta_t) / mw1
                             - (w1 * xb[u] - w1 * eta_c + wsi * (1.0 - d) * (dyi - xb[u]) * xu[u]) / mw1)
                    elif method == DRIMP:
                        kp = ckp[u]
                        summand = kp * cim[u] * (dyi - xb[u])
                        r = a + wsi * kp * (summand - d * a) / mw1
                    else:
                        w0 = wsi * cw0[u]
                        lin = wsi * cdp[u] * xv[u]
                        if method == IPW:
                            r = a + (w1 * dyi - w0 * dyi - lin - w1 * a) / mw1
                        elif method == STDIPW:
                            r = a + (w1 * dyi - w1 * eta_t) / mw1 - (w0 * dyi - w0 * eta_c + lin) / mw0
                        else:
                            res = dyi - xb[u]
                            r = (a + (w1 * res - w1 * eta_t) / mw1 - (w0 * res - w0 * eta_c + lin) / mw0
                                 + wsi * (1.0 - d) * res * xu[u])
                    dev[u] = (r - a) if mem[u] > 0.0 else 0.0
            s1 = s2 = 0.0
            for u in range(B):
                s1 += dev[u]
                s2 += dev[u] * dev[u]
            S1[blk, c] += s1
            S2[blk, c] += s2
            for q in range(4):
                col = agg[c, q]
                if col >= 0:
                    al, be = alpha[c, q], beta[c, q] * d * inv_s
                    if r201:
                        for u in range(B):
                            acc[col, u] += al * dev[u] + be * WB[blk, u]
                    else:
                        for u in range(B):
                            acc[col, u] += al * dev[u] + be * WB[blk, u] * mem[u]
            if boot and gcol[c] >= 0:
                j = gcol[c]
                sc = inv_n[c]
                for u in range(nv):
                    buf[q0 + u, j] = dev[u] * sc
            if boot and wcol[c] >= 0:
                j = wcol[c]
                for u in range(nv):
                    buf[q0 + u, j] = WB[blk, u] * mem[u]
        for col in range(n_agg):
            for u in range(nv):
                Dm[r0 + u, col] = acc[col, u]


def _blocks(gvar, X3, y, obs8, w):
    """Units sorted by cohort (never treated first) into blocks of LANE lanes, one cohort per block
    (the last block of a cohort partial, zero padded): cohort of each block, covariates (blocks x
    P x k1 x LANE), outcomes and observed flags (blocks x LANE x T), weights and lane validity
    (blocks x LANE), each block's first row in cohort order and its number of units, and the
    units in that order."""
    order = np.argsort(gvar, kind="stable")
    vals, starts, counts = np.unique(gvar[order], return_index=True, return_counts=True)
    nb_per = (counts + LANE - 1) // LANE
    nblk = int(nb_per.sum())
    pos = np.full(nblk * LANE, -1, np.int64)
    brow0 = np.empty(nblk, np.int64)
    bval = np.empty(nblk, np.int64)
    p = bi = 0
    for st, cnt, nbk in zip(starts, counts, nb_per):
        pos[p:p + cnt] = order[st:st + cnt]
        for j in range(nbk):
            brow0[bi] = st + j * LANE
            bval[bi] = min(LANE, cnt - j * LANE)
            bi += 1
        p += nbk * LANE
    valid = pos >= 0
    P, k1, T = X3.shape[0], X3.shape[2], y.shape[1]
    XB = np.zeros((nblk * LANE, P, k1))
    XB[valid] = X3[:, pos[valid], :].transpose(1, 0, 2)
    XB[~np.isfinite(XB)] = 0.0                       # covariates of unobserved periods are never used
    YB = np.zeros((nblk * LANE, T))
    YB[valid] = np.where(obs8[pos[valid]] > 0, y[pos[valid]], 0.0)
    OB = np.zeros((nblk * LANE, T))
    OB[valid] = obs8[pos[valid]]
    WB = np.zeros(nblk * LANE)
    WB[valid] = w[pos[valid]]
    return (np.repeat(vals, nb_per).astype(np.int64),
            np.ascontiguousarray(XB.reshape(nblk, LANE, P, k1).transpose(0, 2, 3, 1)),
            YB.reshape(nblk, LANE, T), OB.reshape(nblk, LANE, T), WB.reshape(nblk, LANE),
            valid.astype(np.float64).reshape(nblk, LANE), brow0, bval, order)


# ----------------------------------------------------------------------------------------------
# Pass 4 helpers: column statistics and bootstrap order statistics
# ----------------------------------------------------------------------------------------------

@nb.njit(parallel=True, error_model="numpy")
def _col_stats(Dm, ncol):
    """Column means and centered sums of squares of the first ncol columns (two passes over rows)."""
    N = Dm.shape[0]
    nch = 64
    part = np.zeros((nch, ncol))
    for h in nb.prange(nch):
        acc = np.zeros(ncol)
        for i in range(h * N // nch, (h + 1) * N // nch):
            for j in range(ncol):
                acc[j] += Dm[i, j]
        part[h] = acc
    mean = np.zeros(ncol)
    for h in range(nch):
        for j in range(ncol):
            mean[j] += part[h, j]
    for j in range(ncol):
        mean[j] /= N
    for h in nb.prange(nch):
        acc = np.zeros(ncol)
        for i in range(h * N // nch, (h + 1) * N // nch):
            for j in range(ncol):
                z = Dm[i, j] - mean[j]
                acc[j] += z * z
        part[h] = acc
    css = np.zeros(ncol)
    for h in range(nch):
        for j in range(ncol):
            css[j] += part[h, j]
    return mean, css


@nb.njit(parallel=True, error_model="numpy")
def _rows_add(G, rows, P):
    """G[rows[j]] += P[j] (distinct rows)."""
    for j in nb.prange(rows.size):
        g = G[rows[j]]
        pj = P[j]
        for r in range(g.size):
            g[r] += pj[r]


@nb.njit(error_model="numpy")
def _select(a, lo, hi, k):
    """Quickselect: rearranges a[lo:hi] so that a[k] is its (k - lo)-th smallest value, with
    a[lo:k] <= a[k] <= a[k + 1:hi] (Hoare partitions, median-of-three pivots)."""
    while hi - lo > 1:
        x, y, z = a[lo], a[(lo + hi - 1) // 2], a[hi - 1]
        pivot = max(min(x, y), min(max(x, y), z))
        i, j = lo, hi - 1
        while i <= j:
            while a[i] < pivot:
                i += 1
            while a[j] > pivot:
                j -= 1
            if i <= j:
                a[i], a[j] = a[j], a[i]
                i += 1
                j -= 1
        if k <= j:
            hi = j + 1
        elif k >= i:
            lo = i
        else:
            return


@nb.njit(parallel=True, error_model="numpy")
def _quartiles(Dr, i25, i75):
    """Order statistics i25 <= i75 (0-based) of every row (selection on a copy of each row)."""
    K, B = Dr.shape
    q25 = np.empty(K)
    q75 = np.empty(K)
    for k in nb.prange(K):
        a = Dr[k].copy()
        _select(a, 0, B, i75)
        q75[k] = a[i75]
        if i25 < i75:
            _select(a, 0, i75, i25)
        q25[k] = a[i25]
    return q25, q75


@nb.njit(error_model="numpy")
def _kth(x, k):
    a = x.copy()
    _select(a, 0, a.size, k)
    return a[k]


@nb.njit(parallel=True, error_model="numpy")
def _tmax(Dr, se):
    """max over rows k of |Dr[k, r] / se[k]| for every draw r, NaN if any ratio is NaN (np.max)."""
    K, B = Dr.shape
    out = np.zeros(B)
    nan = np.zeros(B, np.bool_)
    nblk = (B + 255) // 256
    for ib in nb.prange(nblk):
        lo, hi = ib * 256, min(B, ib * 256 + 256)
        for k in range(K):
            s = se[k]
            for r in range(lo, hi):
                q = abs(Dr[k, r] / s)
                if q != q:
                    nan[r] = True
                elif q > out[r]:
                    out[r] = q
    for r in range(B):
        if nan[r]:
            out[r] = np.nan
    return out


def _order_index(B, p, rules):
    """0-based index of reference.order_stat: ceil(p B) (2.0.1) or ceil(p (B + 1)) (1.81), clamped."""
    k = np.ceil(p * B) if rules == "csdid201" else np.ceil(p * (B + 1))
    return min(max(int(k), 1), B) - 1


def _boot_stats(Dr, alpha, floor, rules):
    """reference.boot_stats for draws stored one row per column of the table ([K, B])."""
    B = Dr.shape[1]
    q25, q75 = _quartiles(Dr, _order_index(B, 0.25, rules), _order_index(B, 0.75, rules))
    se_b = (q75 - q25) / IQR_NORM
    point = NormalDist().inv_cdf(1 - alpha / 2)
    tm = _tmax(Dr, se_b)
    tm = tm[np.isfinite(tm)]
    if tm.size == 0:                 # as the reference: no finite draw of the maximum (degenerate table)
        raise IndexError("index -1 is out of bounds for axis 0 with size 0")
    crit = float(_kth(tm, _order_index(tm.shape[0], 1 - alpha, rules)))
    if rules == "csdid201" and floor and not crit >= point:
        crit = point
    return se_b, crit


# ----------------------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------------------

def _exact_cell(gvar, obs, y, X3, w, g, t, b, p, m, notyet, method, trim, N, overlap):
    """One cell with the reference's own NumPy operations (None if csdid 2.0.1's overlap check
    fails): its units, influence function, and its ATT and standard error computed from the
    full-sample column as the reference does."""
    grp = (gvar == 0) | (gvar == g) | ((gvar > m) if notyet else False)
    s = grp & obs[:, b - 1] & obs[:, t - 1]
    D = (gvar[s] == g).astype(np.float64)
    Xs = X3[p][s]
    if overlap and _r_overlap_fails(D, Xs):
        return None
    ws = w[s] / w[s].mean()
    r = _r_cell_rif(y[s, t - 1] - y[s, b - 1], Xs, D, ws, method, trim)
    full = np.zeros(N)
    full[s] = (r - r.mean()) * N / s.sum()
    # the reference reduces its n_units x n_cells matrix over axis 0 (a running sum down each
    # column); a two-column array gives the same order of additions (a 1-D sum is pairwise)
    two = np.column_stack([full + r.mean()] * 2)
    mu = two.mean(0)
    return np.flatnonzero(s), r, float(mu[0]), float(np.sqrt(((two - mu) ** 2).sum(0))[0] / N)


def run(y, X, gvar, w=None, control="never", method="dripw", event_window=(-5, 5),
        base_period="varying", trim=1.0, boot_weights=None, alpha=0.05, event_sets=None, rules="csdid181"):
    if control not in ("never", "notyet"):
        raise ValueError("control must be 'never' or 'notyet'")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if base_period not in ("varying", "universal"):
        raise ValueError("base_period must be 'varying' or 'universal'")
    if rules not in RULES:
        raise ValueError(f"rules must be one of {RULES}")
    y = np.asarray(y, dtype=np.float64)
    gvar = np.asarray(gvar, dtype=np.int64)
    unweighted = w is None
    w = np.ones(y.shape[0]) if w is None else np.asarray(w, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    varying = X.ndim == 3
    if varying:                      # a missing covariate makes the unit-period unobserved
        y = np.where(np.isfinite(X).all(2), y, np.nan)
    units, obs = _sample(y, gvar, control)
    y = np.ascontiguousarray(y[units])
    gvar, w = gvar[units], w[units]
    X = X[units]
    N = int(units.sum())
    T = obs.shape[1]
    k = X.shape[-1]
    k1 = k + 1
    if varying:                      # covariates of each period, with a constant: [T, N, k1]
        X3 = np.empty((T, N, k1))
        X3[:, :, :k] = X.transpose(1, 0, 2)
    else:
        X3 = np.empty((1, N, k1))
        X3[0, :, :k] = X
    X3[:, :, k] = 1.0
    w = w / w.mean()
    unweighted = unweighted or bool(np.all(w == 1.0))
    notyet = control == "notyet"
    r201 = rules == "csdid201"
    overlap = r201 and method != "reg"
    mcode = METHODS.index(method)
    obs8 = np.ascontiguousarray(obs, dtype=np.uint8)
    cohorts = np.unique(gvar[gvar > 0])

    # 1. cells in csdid's order, and their sizes
    cg, ct, cb = [], [], []
    for g in cohorts:
        if base_period == "universal":
            for t in range(1, T + 1):
                if t != g - 1:
                    cg.append(g)
                    ct.append(t)
                    cb.append(g - 1)
        else:
            base = 1
            for t in range(2, T + 1):
                cg.append(g)
                ct.append(t)
                cb.append(base)
                if t < g:
                    base = t
    cg, ct, cb = (np.array(v, dtype=np.int64) for v in (cg, ct, cb))
    # comparison threshold: not-yet-treated units are those first treated after max(g, t) or, under
    # 2.0.1's rules with varying base periods, after t (untreated in both of the cell's periods)
    cm = ct.copy() if (notyet and r201 and base_period == "varying") else np.maximum(cg, ct)
    cp = (np.minimum(cb, ct) - 1) if varying else np.zeros(cg.size, np.int64)
    n, n1, sw, sw1 = _cell_counts(gvar, w, obs8, cg, ct, cb, cm, notyet)
    keep = (n1 > 0) & (n1 < n)
    cg, ct, cb, cm, cp, n, sw, sw1 = (v[keep] for v in (cg, ct, cb, cm, cp, n, sw, sw1))

    # 2. estimation; cells of a cohort with identical units and covariates (balanced panel) share fits
    C = cg.size
    balanced = bool(obs8[:, :int(max(ct.max(initial=1), cb.max(initial=1)))].all()) if C else True
    order = np.searchsorted(cohorts, cg)
    coh_ptr = np.searchsorted(order, np.arange(cohorts.size + 1))
    if balanced:
        key = np.searchsorted(cohorts, cm, side="right") if notyet else np.zeros(C, np.int64)
        key = key * (T + 1) + cp
    else:
        key = -np.ones(C, np.int64)
    BR, BP, U, V2, SC, flag = _estimate(gvar, obs8, y, X3, w, cg, ct, cb, cp, cm, n, sw,
                                        key.astype(np.int64), coh_ptr.astype(np.int64), mcode, notyet,
                                        float(trim), overlap, unweighted)
    exact = {}
    for c in np.flatnonzero(flag == EXACT):
        res = _exact_cell(gvar, obs, y, X3, w, cg[c], ct[c], cb[c], cp[c], cm[c], notyet, method, trim, N, overlap)
        if res is None:
            flag[c] = SKIP
        else:
            exact[c] = res
            SC[c, 0] = res[1].mean()
    est = flag != SKIP                                   # cells left unestimated by 2.0.1's overlap check
    newpos = np.cumsum(est) - 1
    exact = {int(newpos[c]): v for c, v in exact.items()}
    cg, ct, cb, cm, cp, n, sw, sw1, BR, BP, U, V2, SC, flag = (
        v[est] for v in (cg, ct, cb, cm, cp, n, sw, sw1, BR, BP, U, V2, SC, flag))
    C = cg.size
    e = ct - cg
    post = e >= 0
    if r201:                                             # cohort weights count all of the cohort's units
        cw = np.bincount(np.searchsorted(cohorts, gvar[gvar > 0]), weights=w[gvar > 0], minlength=cohorts.size)
        sw1 = cw[np.searchsorted(cohorts, cg)]

    # aggregation layout: [simple, groups..., calendar periods..., event times...]; event columns
    # cover the window and any other event time named in event_sets
    groups = np.array([g for g in cohorts if (post & (cg == g)).any()], dtype=np.int64)
    periods = np.array(sorted({int(t) for t in ct[post]}), dtype=np.int64)
    lo, hi = event_window
    all_events = np.unique(e)
    events = np.array([kk for kk in all_events if lo <= kk <= hi], dtype=np.int64)
    sets = [np.asarray(sorted(set(int(kk) for kk in es)), dtype=np.int64) for es in (event_sets or [])]
    ev_cols = np.array(sorted(set(events.tolist()) | {int(kk) for s in sets for kk in s if kk in all_events}),
                       dtype=np.int64)
    G1, P1 = 1 + groups.size, 1 + groups.size + periods.size
    n_agg = P1 + ev_cols.size
    n_w = groups.size if r201 else 0
    n_ext = n_agg + n_w + 1                              # + 2.0.1's unit-level cohort weights + ones
    agg = np.full((C, 4), -1, np.int64)
    agg[post, 0] = 0
    agg[post, 1] = 1 + np.searchsorted(groups, cg[post])
    agg[post, 2] = G1 + np.searchsorted(periods, ct[post])
    inev = np.isin(e, ev_cols)
    agg[inev, 3] = P1 + np.searchsorted(ev_cols, e[inev])
    S = sw1.sum() / N                                   # wgt /= wgt.mean(0).sum()
    mw = sw1 / N / S                                    # wgt.mean(0)
    MA = np.zeros(n_agg)
    for q in range(4):
        sel = agg[:, q] >= 0
        np.add.at(MA, agg[sel, q], mw[sel])
    alpha_c = np.zeros((C, 4))
    for q in range(4):
        sel = agg[:, q] >= 0
        alpha_c[sel, q] = mw[sel] / MA[agg[sel, q]] * N / n[sel]

    # 3. aggregate weights (beta: estimated cohort weights) and the pass over units
    a = SC[:, 0]
    attA = np.zeros(n_agg)
    for q in range(4):
        sel = agg[:, q] >= 0
        np.add.at(attA, agg[sel, q], a[sel] * mw[sel])
    attA /= np.where(MA > 0, MA, 1.0)
    beta_c = np.zeros((C, 4))
    for q in range(4):
        sel = agg[:, q] >= 0
        beta_c[sel, q] = (a[sel] - attA[agg[sel, q]]) / MA[agg[sel, q]]
    GB, XB, YB, OB, WB, VB, brow0, bval, unit_order = _blocks(gvar, X3, y, obs8, w)
    nblk = GB.size
    exrow = -np.ones(C, np.int64)
    EXD = np.zeros((len(exact), nblk * LANE))
    if exact:
        slot = np.empty(N, np.int64)                    # block slot of each unit
        for blk in range(nblk):
            slot[unit_order[brow0[blk]:brow0[blk] + bval[blk]]] = blk * LANE + np.arange(bval[blk])
        for j, (c, (idx, r, _, _)) in enumerate(sorted(exact.items())):
            exrow[c] = j
            EXD[j, slot[idx]] = r - a[c]
    Dm = np.zeros((N, n_ext))
    gu = gvar[unit_order]
    if n_w:
        gi = np.searchsorted(groups, gu)
        ing = (gi < groups.size) & (groups[np.minimum(gi, groups.size - 1)] == gu)
        Dm[np.flatnonzero(ing), n_agg + gi[ing]] = w[unit_order][ing]
    Dm[:, n_ext - 1] = 1.0
    S1 = np.zeros((nblk, C))
    S2 = np.zeros((nblk, C))
    inv_n = 1.0 / n
    args = (GB, XB, YB, OB, WB, VB, brow0, bval, cg, ct, cb, cp, cm, mcode, notyet, r201, float(trim),
            BR, BP, U, V2, SC, exrow, EXD, agg, alpha_c, beta_c, 1.0 / S, Dm, n_agg, S1, S2)
    boot = boot_weights is not None
    if not boot:
        _units(0, nblk, *args, -np.ones(C, np.int64), -np.ones(C, np.int64), inv_n, np.zeros((0, 0)))
    else:
        # per chunk of one cohort's units: the influence functions of the cells they belong to (and the
        # units' weights, for the cohort weights' terms, and ones) times the chunk's multipliers (BLAS)
        v = np.asarray(boot_weights)
        rows_all = np.flatnonzero(units)[unit_order]     # multiplier row of each unit in cohort order
        Bd = v.shape[1]
        Ggt = np.zeros((C, Bd))                          # sum over each cell's units of v (r - a) / n
        per_cell = not (r201 or balanced)                # 1.81, unbalanced: weights of the units in the cell
        WVc = np.zeros((C if per_cell else 0, Bd))       # sum of v w over the cell's (or cohort's) units
        WVg = np.zeros((cohorts.size, Bd))
        Gones = np.zeros(Bd)                             # sum of v over all units
        bstart = np.searchsorted(GB, np.unique(GB))
        bend = np.append(bstart[1:], nblk)
        for s0, s1 in zip(bstart, bend):                 # the blocks of one cohort value
            Gb = GB[s0]
            member = np.flatnonzero((cg == Gb) | (Gb == 0) | ((cm < Gb) if notyet else False))
            own = np.flatnonzero(cg == Gb) if Gb > 0 else np.zeros(0, np.int64)
            m = member.size
            nw = (own.size if per_cell else 1) if Gb > 0 else 0
            gcol = -np.ones(C, np.int64)
            gcol[member] = np.arange(m)
            wcol = -np.ones(C, np.int64)
            if per_cell:
                wcol[own] = m + 1 + np.arange(own.size)
            contiguous = m > 0 and member[-1] - member[0] + 1 == m
            for c0 in range(s0, s1, CHUNK):
                c1 = min(s1, c0 + CHUNK)
                r0, r1 = brow0[c0], brow0[c1 - 1] + bval[c1 - 1]
                buf = np.empty((r1 - r0, m + 1 + nw))
                buf[:, m] = 1.0
                if nw and not per_cell:
                    buf[:, m + 1] = w[unit_order[r0:r1]]
                _units(c0, c1, *args, gcol, wcol, inv_n, buf)
                prod = buf.T @ v[rows_all[r0:r1]].astype(np.float64)
                if contiguous:
                    Ggt[member[0]:member[-1] + 1] += prod[:m]
                elif m:
                    _rows_add(Ggt, member, prod[:m])
                Gones += prod[m]
                if nw and per_cell:
                    _rows_add(WVc, own, prod[m + 1:])
                elif nw:
                    WVg[np.searchsorted(cohorts, Gb)] += prod[m + 1]
    S1, S2 = S1.sum(axis=0), S2.sum(axis=0)

    att_gt = a + S1 / n
    se_gt = np.sqrt(np.maximum(S2 - S1 * S1 / n, 0.0)) / n
    # A cell whose influence function is constant up to rounding has a standard error that is
    # rounding residue (relative to the size of its values); report the reference's residue.
    rms = np.sqrt(np.maximum((S2 + 2.0 * a * S1) / n + a * a, 0.0))
    for c in np.flatnonzero((flag == OK) & ~(se_gt > 1e-11 * rms)):
        att_gt[c], se_gt[c] = _exact_cell(gvar, obs, y, X3, w, cg[c], ct[c], cb[c], cp[c], cm[c], notyet,
                                          method, trim, N, False)[2:]
    for c, (_, _, att, se_c) in exact.items():
        att_gt[c], se_gt[c] = att, se_c

    # aggregates: ATT from the cells' ATTs and weights (csdid's aggte), SE from the unit IFs
    att_agg = np.zeros(n_agg)
    for q in range(4):
        sel = agg[:, q] >= 0
        np.add.at(att_agg, agg[sel, q], att_gt[sel] * mw[sel])
    att_agg /= np.where(MA > 0, MA, 1.0)
    mean, css = _col_stats(Dm, n_ext - 1)
    se_cols = np.sqrt(css) / N

    def comb(coef):                                     # a linear combination of the Dm columns
        x = Dm @ np.append(coef, 0.0)
        return x, float(np.sqrt(((x - x.mean()) ** 2).sum()) / N)

    gsel = np.arange(1, G1)
    psel = np.arange(G1, P1)
    esel = P1 + np.searchsorted(ev_cols, events)
    pre, pst = events < 0, events >= 0
    att_group = att_agg[gsel]
    c_ga = np.zeros(n_ext - 1)
    if r201:   # cohort shares estimated from unit-level membership (aggte over the group columns)
        m_w = cw[np.searchsorted(cohorts, groups)] / N
        att_ga = float((att_group * m_w).sum() / m_w.sum())
        c_ga[gsel] = m_w / m_w.sum()
        c_ga[n_agg:n_agg + n_w] = (att_group - att_ga) / m_w.sum()
    else:      # cohort shares held fixed at the mean cohort weight
        gw = np.array([mw[post & (cg == g)].mean() for g in groups])
        gamma = gw / gw.sum()
        att_ga = float(att_group @ gamma)
        c_ga[gsel] = gamma
    c_cal = np.zeros(n_ext - 1)
    c_cal[psel] = 1.0 / psel.size
    c_pre = np.zeros(n_ext - 1)
    c_post = np.zeros(n_ext - 1)
    if pre.any():
        c_pre[esel[pre]] = 1.0 / pre.sum()
    if pst.any():
        c_post[esel[pst]] = 1.0 / pst.sum()
    out = dict(gt=np.column_stack([cg, ct]).astype(np.int64), att_gt=att_gt, se_gt=se_gt,
               group_ids=groups, att_group=att_group, se_group=se_cols[gsel],
               calendar_periods=periods, att_calendar=att_agg[psel], se_calendar=se_cols[psel],
               event_times=events, att_event=att_agg[esel], se_event=se_cols[esel])
    Ec = Dm[:, esel] - mean[esel]
    out["cov_event"] = (Ec.T @ Ec) / N ** 2
    del Ec
    out["event_sets"] = [s.tolist() for s in sets]
    att_sets, se_sets = [], []
    for s in sets:
        present = np.array([kk for kk in s if kk in all_events], dtype=np.int64)
        coef = np.zeros(n_ext - 1)
        coef[P1 + np.searchsorted(ev_cols, present)] = 1.0 / present.size
        att_sets.append(float(att_agg[P1 + np.searchsorted(ev_cols, present)].mean()))
        se_sets.append(comb(coef)[1])
    out["att_event_sets"] = np.array(att_sets, dtype=np.float64)
    out["se_event_sets"] = np.array(se_sets, dtype=np.float64)
    out["att_simple"], out["se_simple"] = float(att_agg[0]), float(se_cols[0])
    out["att_group_avg"], out["se_group_avg"] = att_ga, comb(c_ga)[1]
    out["att_calendar_avg"], out["se_calendar_avg"] = float(att_agg[psel].mean()), comb(c_cal)[1]
    out["att_pre_avg"], out["se_pre_avg"] = ((float(att_agg[esel][pre].mean()), comb(c_pre)[1]) if pre.any()
                                             else (float("nan"), float("nan")))
    out["att_post_avg"], out["se_post_avg"] = ((float(att_agg[esel][pst].mean()), comb(c_post)[1]) if pst.any()
                                               else (float("nan"), float("nan")))

    # 4. bootstrap statistics: draws of every table (one row per column) and of the single estimates
    if boot:
        # sums over units of v times each aggregate column of Dm: the cells' sums with the aggregation
        # weights alpha, plus the cohort weights' terms (beta / S times sums of v w), and the 2.0.1
        # unit-level cohort weights; then centered as the reference centers every column
        A = np.zeros((n_agg, C))
        Bw = np.zeros((n_agg, C))
        for q in range(4):
            sel = np.flatnonzero(agg[:, q] >= 0)
            A[agg[sel, q], sel] += alpha_c[sel, q] * n[sel]
            Bw[agg[sel, q], sel] += beta_c[sel, q] / S
        if per_cell:
            Graw = A @ Ggt + Bw @ WVc
        else:                                            # the cohort weights' terms by cohort
            Bg = np.zeros((n_agg, cohorts.size))
            np.add.at(Bg.T, np.searchsorted(cohorts, cg), Bw.T)
            Graw = A @ Ggt + Bg @ WVg
        if n_w:
            Graw = np.vstack([Graw, WVg[np.searchsorted(cohorts, groups)]])
        Dagg = (Graw - mean[:, None] * Gones) / N                          # centered columns
        tables = dict(gt=Ggt, group=Dagg[gsel], calendar=Dagg[psel], event=Dagg[esel])
        for name, Dr in tables.items():
            out[f"se_boot_{name}"], out[f"crit_{name}"] = _boot_stats(np.ascontiguousarray(Dr), alpha,
                                                                       name != "gt", rules)
        c_simple = np.zeros(n_ext - 1)
        c_simple[0] = 1.0
        singles = dict(simple=c_simple, group_avg=c_ga, calendar_avg=c_cal,
                       pre_avg=c_pre if pre.any() else None, post_avg=c_post if pst.any() else None)
        for name, coef in singles.items():
            if coef is None:
                out[f"se_boot_{name}"] = np.nan
            else:
                out[f"se_boot_{name}"] = float(_boot_stats((coef @ Dagg)[None, :], alpha, True, rules)[0][0])
    return out
