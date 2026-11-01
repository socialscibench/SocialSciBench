"""Task 002b candidate "cell-fused": Callaway & Sant'Anna (2021) ATT(g,t) with csdid's common
options (never or not-yet-treated controls, sampling weights, unbalanced panels with pair-balanced
cells; the dripw, reg, ipw, stdipw and drimp estimators) and the simple, group, calendar and
event-study aggregates with their influence-function standard errors, as computed by
reference.run.

The reference estimates every (cohort, period) cell as a separate small problem, stores every
cell's full-sample influence function and cohort-weight column in two n_units x n_cells arrays,
and aggregates them. This candidate computes the same quantities in three passes:

1. Cell list and counts (parallel over cells): which units each cell uses (cohort plus
   comparison units observed in both of the cell's periods), the cell sizes and weight sums.
   The aggregation weights depend only on these, so they are known before any estimation.
2. Estimation (parallel over cohorts, the cohort's cells in order): the propensity score
   (weighted logit, or inverse probability tilting for drimp) and the outcome regression, by
   Newton-Raphson with the reference's stopping rule. In a balanced panel, consecutive cells of a
   cohort often use exactly the same units (with never-treated controls, every cell of the cohort
   does), and then share one fit; other fits start from the previous cell's solution. Each cell
   is reduced to a few vectors and scalars from which any unit's influence function is a short
   expression (a few dot products with the unit's covariates).
3. One fused pass over units (parallel over blocks of units): for each unit, every cell it
   belongs to contributes its influence-function value to the cell's mean and sum of squares and,
   with the aggregation weights, to the unit's influence function for the simple, group, calendar
   and event-study aggregates (including the terms for the estimated cohort weights). No
   n_units x n_cells array is formed.

Cells whose ATT is within rounding of zero relative to its terms, or whose systems are singular
or do not converge from the warm start and from the reference's start, are recomputed with the
reference's own NumPy operations (the reference raises where its own fits do not converge).

Threads: Numba parallel loops (NUMBA_NUM_THREADS, set by the evaluator's --threads).
"""
import math

import numpy as np
import numba as nb
from numba import types
from numba.extending import intrinsic

CAP = 1.0 - 1e-6             # drdid caps the propensity score below 1; nothing is trimmed (pscoretrim(1))
TOL = 1e-10                  # the reference's Newton stopping rule (largest step)
METHODS = ("dripw", "reg", "ipw", "stdipw", "drimp")
DRIPW, REG, IPW, STDIPW, DRIMP = range(5)
ZERO_ATT = 1e-7              # a cell ATT this small relative to its terms is recomputed exactly
BLK = 256                    # units per block in the vectorized passes
LANE = 64                    # units per block (vector lanes) in the pass over units
FAST = {"reassoc", "contract"}


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


def _r_cell_rif(dy, X, D, w, method):
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
    keep = ~((D == 0) & (ps_raw > 1.0))
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


# ----------------------------------------------------------------------------------------------
# Pass 2: estimation
# ----------------------------------------------------------------------------------------------

@nb.njit(error_model="numpy")
def _gather(G, obs, y, Xr, w, g, t, b, m, notyet, n, wbar, idx, Xg, D, ws, dy):
    N, k1 = Xr.shape
    j = 0
    for i in range(N):
        Gi = G[i]
        if _member(Gi, g, m, notyet) and obs[i, b] and obs[i, t]:
            idx[j] = i
            for k in range(k1):
                Xg[k, j] = Xr[i, k]
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
    or 2 (singular or non-finite). For the logit, V receives the inverse Hessian at the solution."""
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
        _hess_grad(Xg, D, ws, beta, tilt, H, g, eta, wp, rr)
        if not _chol(H, L):
            return 2
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
def _scores(Xg, bp, ps):
    """Capped propensity scores of the gathered units."""
    k1, n = Xg.shape
    eta = np.empty(BLK)
    for lo in range(0, n, BLK):
        m = min(BLK, n - lo)
        _index(Xg, bp, lo, m, eta)
        for j in range(m):
            ps[lo + j] = min(_expit(eta[j]), CAP)


@nb.njit(error_model="numpy", fastmath=FAST)
def _constants(method, Xg, D, ws, dy, ps, br, inv, V, U, V2, sc):
    """Cell constants of the influence function (see _units). sc: a (ATT), eta_t, eta_c, mean w1,
    mean w0, and the magnitude of the terms whose difference is the ATT plus the mean absolute
    outcome change (the scale of the near-zero test). Per-unit quantities first, then sums over
    units for each covariate (contiguous, so the loops vectorize)."""
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
    s_w1 = s_w0 = s_a = s_b = s_dy = 0.0
    for j in range(n):
        d = D[j]
        wj = ws[j]
        s_dy += wj * abs(dy[j])
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
            s_b += (1.0 - d) * wj * ps[j] / (1.0 - ps[j]) * vj
        else:
            w0j = wj * ps[j] * (1.0 - d) / (1.0 - ps[j])
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
    a, scale = eta_t - eta_c, abs(eta_t) + abs(eta_c)
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
    sc[0], sc[1], sc[2], sc[3], sc[4], sc[5] = a, eta_t, eta_c, mw1, mw0, scale + s_dy / n


@nb.njit(parallel=True, error_model="numpy")
def _estimate(G, obs, y, Xr, w, cg, ct, cb, cm, cn, csw, key, coh_ptr, method, notyet):
    """Cell constants for every cell; cells are processed per cohort, in order, so that cells with
    the same units (equal key) share their fits and other fits start from the previous solution."""
    C = cg.size
    k1 = Xr.shape[1]
    BR = np.zeros((C, k1))
    BP = np.zeros((C, k1))
    U = np.zeros((C, k1))
    V2 = np.zeros((C, k1))
    SC = np.zeros((C, 7))
    flag = np.zeros(C, np.int64)
    for h in nb.prange(coh_ptr.size - 1):
        bp = np.zeros(k1)
        V = np.zeros((k1, k1))
        A = np.zeros((k1, k1))
        L = np.zeros((k1, k1))
        inv = np.zeros((k1, k1))
        rhs = np.zeros(k1)
        br = np.zeros(k1)
        warm = False
        prev_key = -1
        fit_ok = False
        idx = np.empty(0, np.int64)
        Xg = np.empty((k1, 0))
        D = np.empty(0)
        ws = np.empty(0)
        dy = np.empty(0)
        ps = np.empty(0)
        sw0r = np.empty(0)
        for c in range(coh_ptr[h], coh_ptr[h + 1]):
            n = cn[c]
            t, b = ct[c] - 1, cb[c] - 1
            shared = key[c] >= 0 and key[c] == prev_key and fit_ok
            if shared:
                _regather_dy(y, idx, t, b, dy)
            else:
                idx = np.empty(n, np.int64)
                Xg = np.empty((k1, n))
                D = np.empty(n)
                ws = np.empty(n)
                dy = np.empty(n)
                ps = np.empty(n)
                sw0r = np.empty(n)
                _gather(G, obs, y, Xr, w, cg[c], t, b, cm[c], notyet, n, csw[c] / n, idx, Xg, D, ws, dy)
                fit_ok = True
                if method != REG:
                    tilt = method == DRIMP
                    start = bp.copy()
                    if not warm:
                        start[:] = 0.0
                        if tilt:
                            s1 = s0 = 0.0
                            for j in range(n):
                                s1 += ws[j] * D[j]
                                s0 += ws[j] * (1.0 - D[j])
                            start[k1 - 1] = math.log(s1 / s0)
                    trial = start.copy()
                    status = _newton(Xg, D, ws, trial, tilt, 50 if warm else (200 if tilt else 100), V)
                    if status != 0 and warm:            # retry from the reference's starting point
                        trial[:] = 0.0
                        if tilt:
                            s1 = s0 = 0.0
                            for j in range(n):
                                s1 += ws[j] * D[j]
                                s0 += ws[j] * (1.0 - D[j])
                            trial[k1 - 1] = math.log(s1 / s0)
                        status = _newton(Xg, D, ws, trial, tilt, 100 if not tilt else 200, V)
                    if status != 0:
                        fit_ok = False
                    else:
                        bp[:] = trial
                        warm = True
                        _scores(Xg, bp, ps)
                if fit_ok and method != IPW and method != STDIPW:
                    if method == DRIMP:
                        for j in range(n):
                            sw0r[j] = ps[j] / (1.0 - ps[j]) * ws[j]
                        _wls_matrix(Xg, D, sw0r, A)
                    else:
                        _wls_matrix(Xg, D, ws, A)
                    if not _chol(A, L):
                        fit_ok = False
                    else:
                        _chol_inv(L, inv)
            prev_key = key[c]
            if not fit_ok:
                flag[c] = 1
                continue
            if method != IPW and method != STDIPW:
                _wls_rhs(Xg, D, sw0r if method == DRIMP else ws, dy, rhs)
                _chol_solve(L, rhs, br)
            _constants(method, Xg, D, ws, dy, ps, br, inv, V, U[c], V2[c], SC[c])
            SC[c, 6] = csw[c] / n
            BR[c] = br
            BP[c] = bp
            if not abs(SC[c, 0]) >= ZERO_ATT * SC[c, 5]:
                flag[c] = 2
    return BR, BP, U, V2, SC, flag


# ----------------------------------------------------------------------------------------------
# Pass 3: influence functions of every cell and every aggregate, one pass over units
# ----------------------------------------------------------------------------------------------

@nb.njit(parallel=True, error_model="numpy")
def _units(GB, XB, YB, OB, WB, VB, cg, ct, cb, cm, method, notyet, BR, BP, U, V2, SC, flag, agg, alpha, beta,
           inv_s, n_agg):
    """For every block of units (one cohort per block, LANE lanes) and every cell the block's cohort can
    belong to: each member unit's influence-function value r (reference.cell_rif, term by term),
    summed per cell around the cell's closed-form ATT a (sum and sum of squares), and the units'
    aggregate influence functions (minus the aggregate ATTs): the sum over their cells of
    alpha (r - a) plus, for their own cohort's cells, beta w / S. Lanes outside a cell, and padding
    lanes, contribute exactly zero (their outcome changes may be NaN)."""
    nblk, k1, B = XB.shape
    C = cg.size
    Dm = np.zeros((nblk, B, n_agg))
    S1 = np.zeros((nblk, C))
    S2 = np.zeros((nblk, C))
    for blk in nb.prange(nblk):
        Gb = GB[blk]
        xb = np.empty(B)
        xp = np.empty(B)
        xu = np.empty(B)
        xv = np.empty(B)
        dev = np.empty(B)
        mem = np.empty(B)
        for c in range(C):
            if flag[c] != 0:
                continue
            g = cg[c]
            if not _member(Gb, g, cm[c], notyet):
                continue
            t, b = ct[c] - 1, cb[c] - 1
            cnt = 0.0
            for u in range(B):
                mem[u] = VB[blk, u] * OB[blk, u, t] * OB[blk, u, b]
                cnt += mem[u]
            if cnt == 0.0:
                continue
            for u in range(B):
                xb[u] = xp[u] = xu[u] = xv[u] = 0.0
            for k in range(k1):
                br, bp, uk, vk = BR[c, k], BP[c, k], U[c, k], V2[c, k]
                for u in range(B):
                    x = XB[blk, k, u]
                    xb[u] += x * br
                    xp[u] += x * bp
                    xu[u] += x * uk
                    xv[u] += x * vk
            d = 1.0 if Gb == g else 0.0
            a, eta_t, eta_c, mw1, mw0, wbar = SC[c, 0], SC[c, 1], SC[c, 2], SC[c, 3], SC[c, 4], SC[c, 6]
            for u in range(B):
                wsi = WB[blk, u] / wbar
                dyi = YB[blk, u, t] - YB[blk, u, b]
                w1 = wsi * d
                if method == REG:
                    r = (a + (w1 * dyi - w1 * eta_t) / mw1
                         - (w1 * xb[u] - w1 * eta_c + wsi * (1.0 - d) * (dyi - xb[u]) * xu[u]) / mw1)
                else:
                    ps = min(_expit(xp[u]), CAP)
                    if method == DRIMP:
                        summand = (1.0 - (1.0 - d) / (1.0 - ps)) * (dyi - xb[u])
                        r = a + wsi * (summand - d * a) / mw1
                    else:
                        w0 = wsi * ps * (1.0 - d) / (1.0 - ps)
                        lin = wsi * (d - ps) * xv[u]
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
                    for u in range(B):
                        Dm[blk, u, col] += al * dev[u] + be * WB[blk, u] * mem[u]
    return Dm.reshape(nblk * B, n_agg), S1.sum(axis=0), S2.sum(axis=0)


def _blocks(gvar, Xr, y, obs8, w):
    """Units sorted by cohort (never treated first) into blocks of LANE lanes, one cohort per block,
    zero padded: cohort of each block, covariates (blocks x k1 x LANE), outcomes and observed flags
    (blocks x LANE x T), weights and lane validity (blocks x LANE), and each unit's position."""
    order = np.argsort(gvar, kind="stable")
    vals, starts, counts = np.unique(gvar[order], return_index=True, return_counts=True)
    nb_per = (counts + LANE - 1) // LANE
    nblk = int(nb_per.sum())
    pos = np.full(nblk * LANE, -1, np.int64)
    p = 0
    for st, cnt, nbk in zip(starts, counts, nb_per):
        pos[p:p + cnt] = order[st:st + cnt]
        p += nbk * LANE
    valid = pos >= 0
    k1, T = Xr.shape[1], y.shape[1]
    XB = np.zeros((nblk * LANE, k1))
    XB[valid] = Xr[pos[valid]]
    YB = np.zeros((nblk * LANE, T))
    YB[valid] = np.where(obs8[pos[valid]] > 0, y[pos[valid]], 0.0)
    OB = np.zeros((nblk * LANE, T))
    OB[valid] = obs8[pos[valid]]
    WB = np.zeros(nblk * LANE)
    WB[valid] = w[pos[valid]]
    slot = np.empty(gvar.size, np.int64)
    slot[pos[valid]] = np.flatnonzero(valid)
    return (np.repeat(vals, nb_per).astype(np.int64),
            np.ascontiguousarray(XB.reshape(nblk, LANE, k1).transpose(0, 2, 1)),
            YB.reshape(nblk, LANE, T), OB.reshape(nblk, LANE, T), WB.reshape(nblk, LANE),
            valid.astype(np.float64).reshape(nblk, LANE), slot)


# ----------------------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------------------

def _exact_cell(gvar, obs, y, Xc, w, g, t, b, notyet, method, N):
    """One cell with the reference's own NumPy operations: its units, D, influence function, and its
    ATT and standard error computed from the full-sample column as the reference does."""
    grp = (gvar == 0) | (gvar == g) | ((gvar > max(g, t)) if notyet else False)
    s = grp & obs[:, b - 1] & obs[:, t - 1]
    D = (gvar[s] == g).astype(np.float64)
    ws = w[s] / w[s].mean()
    r = _r_cell_rif(y[s, t - 1] - y[s, b - 1], Xc[s], D, ws, method)
    full = np.zeros(N)
    full[s] = (r - r.mean()) * N / s.sum()
    # the reference reduces its n_units x n_cells matrix over axis 0 (a running sum down each
    # column); a two-column array gives the same order of additions (a 1-D sum is pairwise)
    two = np.column_stack([full + r.mean()] * 2)
    m = two.mean(0)
    return np.flatnonzero(s), D, r, float(m[0]), float(np.sqrt(((two - m) ** 2).sum(0))[0] / N)


def run(y, X, gvar, w=None, control="never", method="dripw", event_window=(-5, 5)):
    if control not in ("never", "notyet"):
        raise ValueError("control must be 'never' or 'notyet'")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    y = np.asarray(y, dtype=np.float64)
    gvar = np.asarray(gvar, dtype=np.int64)
    w = np.ones(y.shape[0]) if w is None else np.asarray(w, dtype=np.float64)
    units, obs = _sample(y, gvar, control)
    y = np.ascontiguousarray(y[units])
    gvar, w = gvar[units], w[units]
    X = np.asarray(X, dtype=np.float64)[units]
    N = int(units.sum())
    Xr = np.ascontiguousarray(np.column_stack([X, np.ones(N)]))
    k1 = Xr.shape[1]
    w = w / w.mean()
    notyet = control == "notyet"
    mcode = METHODS.index(method)
    T = obs.shape[1]
    obs8 = np.ascontiguousarray(obs, dtype=np.uint8)
    cohorts = np.unique(gvar[gvar > 0])

    # 1. cells in csdid's order, and their sizes
    cg, ct, cb = [], [], []
    for g in cohorts:
        base = 1
        for t in range(2, T + 1):
            cg.append(g)
            ct.append(t)
            cb.append(base)
            if t < g:
                base = t
    cg, ct, cb = (np.array(v, dtype=np.int64) for v in (cg, ct, cb))
    cm = np.maximum(cg, ct)
    n, n1, sw, sw1 = _count_cells(gvar, obs8, w, cg, ct, cb, cm, notyet)
    keep = (n1 > 0) & (n1 < n)
    cg, ct, cb, cm, n, sw, sw1 = (v[keep] for v in (cg, ct, cb, cm, n, sw, sw1))
    C = cg.size
    e = ct - cg
    post = e >= 0

    # aggregation layout: [simple, groups..., calendar periods..., event times...]
    groups = np.array([g for g in cohorts if (post & (cg == g)).any()], dtype=np.int64)
    periods = np.array(sorted({int(t) for t in ct[post]}), dtype=np.int64)
    lo, hi = event_window
    events = np.array([k for k in np.unique(e) if lo <= k <= hi], dtype=np.int64)
    G1, P1 = 1 + groups.size, 1 + groups.size + periods.size
    n_agg = P1 + events.size
    agg = np.full((C, 4), -1, np.int64)
    agg[post, 0] = 0
    agg[post, 1] = 1 + np.searchsorted(groups, cg[post])
    agg[post, 2] = G1 + np.searchsorted(periods, ct[post])
    inwin = (e >= lo) & (e <= hi)
    agg[inwin, 3] = P1 + np.searchsorted(events, e[inwin])
    S = sw1.sum() / N                                   # wgt /= wgt.mean(0).sum()
    mw = sw1 / N / S                                    # wgt.mean(0)
    MA = np.zeros(n_agg)
    for q in range(4):
        sel = agg[:, q] >= 0
        np.add.at(MA, agg[sel, q], mw[sel])
    alpha = np.zeros((C, 4))
    for q in range(4):
        sel = agg[:, q] >= 0
        alpha[sel, q] = mw[sel] / MA[agg[sel, q]] * N / n[sel]

    # 2. estimation; cells of a cohort with identical units (balanced panel) share fits
    balanced = bool(obs8[:, :int(ct.max(initial=1))].all()) if C else True
    order = np.searchsorted(cohorts, cg)
    coh_ptr = np.searchsorted(order, np.arange(cohorts.size + 1))
    if balanced:
        key = np.searchsorted(cohorts, cm, side="right") if notyet else np.zeros(C, np.int64)
    else:
        key = -np.ones(C, np.int64)
    BR, BP, U, V2, SC, flag = _estimate(gvar, obs8, y, Xr, w, cg, ct, cb, cm, n, sw, key.astype(np.int64),
                                        coh_ptr.astype(np.int64), mcode, notyet)

    # cells recomputed with the reference's operations
    exact = {}
    for c in np.flatnonzero(flag):
        exact[c] = _exact_cell(gvar, obs, y, Xr, w, cg[c], ct[c], cb[c], notyet, method, N)
        SC[c, 0] = exact[c][2].mean()

    # 3. aggregate weights (beta: estimated cohort weights) and the pass over units
    a = SC[:, 0]
    attA = np.zeros(n_agg)
    for q in range(4):
        sel = agg[:, q] >= 0
        np.add.at(attA, agg[sel, q], a[sel] * mw[sel])
    attA /= np.where(MA > 0, MA, 1.0)
    beta = np.zeros((C, 4))
    for q in range(4):
        sel = agg[:, q] >= 0
        beta[sel, q] = (a[sel] - attA[agg[sel, q]]) / MA[agg[sel, q]]
    GB, XB, YB, OB, WB, VB, slot = _blocks(gvar, Xr, y, obs8, w)
    Dm, S1, S2 = _units(GB, XB, YB, OB, WB, VB, cg, ct, cb, cm, mcode, notyet, BR, BP, U, V2, SC, flag, agg,
                        alpha, beta, 1.0 / S, n_agg)
    Dm = Dm[VB.ravel() > 0]                 # one row per unit (in block order; only sums over units are used)

    att_gt = a + S1 / n
    se_gt = np.sqrt(np.maximum(S2 - S1 * S1 / n, 0.0)) / n
    # A cell whose influence function is constant up to rounding has a standard error that is
    # rounding residue (relative to the size of its values); report the reference's residue.
    rms = np.sqrt(np.maximum((S2 + 2.0 * a * S1) / n + a * a, 0.0))
    for c in np.flatnonzero((flag == 0) & ~(se_gt > 1e-11 * rms)):
        att_gt[c], se_gt[c] = _exact_cell(gvar, obs, y, Xr, w, cg[c], ct[c], cb[c], notyet, method, N)[3:]
    rows = np.cumsum(VB.ravel() > 0) - 1     # row of Dm for each block slot
    for c, (idx, D, r, att, se_c) in exact.items():
        att_gt[c], se_gt[c] = att, se_c
        dev = r - a[c]
        wb = w[idx] * D / S
        for q in range(4):
            col = agg[c, q]
            if col >= 0:
                Dm[rows[slot[idx]], col] += alpha[c, q] * dev + beta[c, q] * wb

    # aggregates: ATT from the cells' ATTs and weights (csdid's aggte), SE from the unit IFs
    att_agg = np.zeros(n_agg)
    for q in range(4):
        sel = agg[:, q] >= 0
        np.add.at(att_agg, agg[sel, q], att_gt[sel] * mw[sel])
    att_agg /= np.where(MA > 0, MA, 1.0)

    def se(d):
        return float(np.sqrt(((d - d.mean()) ** 2).sum()) / N)

    se_cols = np.sqrt(((Dm - Dm.mean(0)) ** 2).sum(0)) / N
    gsel = np.arange(1, G1)
    gw = np.array([mw[post & (cg == g)].mean() for g in groups])
    gamma = gw / gw.sum()
    psel = np.arange(G1, P1)
    esel = np.arange(P1, n_agg)
    pre, pst = events < 0, events >= 0
    out = dict(gt=np.column_stack([cg, ct]).astype(np.int64), att_gt=att_gt, se_gt=se_gt,
               group_ids=groups, att_group=att_agg[gsel], se_group=se_cols[gsel],
               calendar_periods=periods, att_calendar=att_agg[psel], se_calendar=se_cols[psel],
               event_times=events, att_event=att_agg[esel], se_event=se_cols[esel])
    out["att_simple"], out["se_simple"] = float(att_agg[0]), float(se_cols[0])
    out["att_group_avg"], out["se_group_avg"] = float(att_agg[gsel] @ gamma), se(Dm[:, gsel] @ gamma)
    out["att_calendar_avg"], out["se_calendar_avg"] = float(att_agg[psel].mean()), se(Dm[:, psel].mean(1))
    if pre.any():
        out["att_pre_avg"], out["se_pre_avg"] = float(att_agg[esel][pre].mean()), se(Dm[:, esel[pre]].mean(1))
    else:
        out["att_pre_avg"], out["se_pre_avg"] = float("nan"), float("nan")
    if pst.any():
        out["att_post_avg"], out["se_post_avg"] = float(att_agg[esel][pst].mean()), se(Dm[:, esel[pst]].mean(1))
    else:
        out["att_post_avg"], out["se_post_avg"] = float("nan"), float("nan")
    return out
