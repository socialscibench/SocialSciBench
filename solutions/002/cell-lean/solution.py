"""Task 002 candidate "cell-lean" (combined task of 2026-10-25): the computation of the "cell-fused"
candidate (Callaway & Sant'Anna 2021 staggered DiD as reference.run computes it under csdid 1.81's
or 2.0.1's rules, with every option of the task) written for a small peak memory: NumPy only (no
JIT compiler in the process), bounded working arrays, and no n_units x n_aggregates or
n_units x n_cells arrays.

1. Cells and their sizes (cohort tallies on balanced panels, masks otherwise).
2. Estimation, one cohort at a time, its cells in order: csdid 2.0.1's overlap check, the
   propensity fit (logit or tilting) and the outcome regression by Newton's method with the
   reference's stopping rule, shared by consecutive cells with the same units and covariates
   (with the sums that do not depend on the outcome change) and warm-started otherwise.
3. Pass over units in blocks of one cohort's units (at most 512 KB per working matrix): the
   block's influence functions for all the cells it belongs to as one units x cells matrix, the
   cells' sums and sums of squares, and the block's aggregate influence functions, whose sums and
   cross-products (a small Gram matrix) give every aggregate's standard error and the event-study
   covariance; the per-unit aggregate influence functions are never stored.
4. Bootstrap: the same blocks' cell influence functions times their multipliers (BLAS), for tiles
   of cells and draws of at most about 4 MB of draws (several passes over the units when a
   table is larger); aggregate draws from the cell draws; standard errors and critical values from
   order statistics (np.partition), with the critical value's maximum accumulated over tiles.

Freed working arrays are returned to the OS after every 64 MB or so of them (malloc_zone_pressure_relief
on macOS, malloc_trim with glibc), since the allocator otherwise keeps them resident.

Threads: BLAS only (the evaluator's limits); everything else runs in one thread.
"""
import ctypes
import ctypes.util
from statistics import NormalDist

import numpy as np

CAP = 1.0 - 1e-6             # drdid caps the propensity score below 1
TOL = 1e-10                  # the reference's Newton stopping rule (largest step)
METHODS = ("dripw", "reg", "ipw", "stdipw", "drimp")
RULES = ("csdid181", "csdid201")
OK, EXACT, SKIP = 0, 1, 3    # cell states: estimated here, recomputed exactly, not estimated (overlap)
OVERLAP = 0.999              # csdid 2.0.1's overlap cut on the unweighted logit score
NEAR = 1e-9                  # a score this close to the trimming or overlap cut: exact path
BIG_INDEX = 25.0             # a warm-started overlap fit this extreme is checked from the reference's start
WORK = 1 << 16               # elements (512 KB) per working matrix of the pass over units
DRAWS = 1 << 19              # elements (4 MB) of bootstrap draws held at a time
IQR_NORM = NormalDist().inv_cdf(0.75) - NormalDist().inv_cdf(0.25)


def _releaser():
    """Ask the C allocator to return free pages to the OS (macOS malloc_zone_pressure_relief, glibc
    malloc_trim); freed working arrays otherwise stay resident and raise the peak RSS."""
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


_release_now = _releaser()
_churn = [0]


def _release(nbytes=0):
    """Count about nbytes of freed working arrays (only large ones, 4 MB or more: the allocator reuses
    smaller blocks); return free pages to the OS every 64 MB (each call also makes the next
    allocations fault their pages again, so it is not done more often)."""
    if nbytes >= 1 << 22:
        _churn[0] += nbytes
    if _churn[0] >= 1 << 26:
        _churn[0] = 0
        _release_now()


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
# Fits
# ----------------------------------------------------------------------------------------------

def _expit(v):
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-v))


def _wgram(X, wv):
    """X' diag(wv) X, in chunks of rows (temporaries of at most 2^16 elements)."""
    k1 = X.shape[1]
    step = max(1, (1 << 16) // k1)
    H = np.zeros((k1, k1))
    for r0 in range(0, X.shape[0], step):
        Xs = X[r0:r0 + step]
        H += (Xs * wv[r0:r0 + step, None]).T @ Xs
    return H


def _chol(H):
    """Cholesky factor, or None unless H is numerically positive definite (pivots above 1e-12 of
    the diagonal)."""
    try:
        L = np.linalg.cholesky(H)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(L)) or not np.all(np.diag(L) ** 2 > 1e-12 * np.abs(np.diag(H))):
        return None
    return L


def _chol_solve(L, b):
    return np.linalg.solve(L.T, np.linalg.solve(L, b))


def _newton(X, D, w, b, tilt, maxiter):
    """Newton-Raphson from b (copied) with the reference's stopping rule; for tilting, the
    reference's step halving. Returns (b, status, V): status 0 (converged), 1 (not converged) or
    2 (singular or non-finite); V the inverse Hessian of the last iteration (logit)."""
    b = b.copy()
    for _ in range(maxiter):
        eta = X @ b
        if tilt:
            e = np.exp(np.minimum(eta, 708.0))
            c = 1.0 - D
            H = _wgram(X, w * c * e)
            g = X.T @ (w * (D - c * e))
        else:
            p = _expit(eta)
            H = _wgram(X, w * p * (1.0 - p))
            g = X.T @ (w * (D - p))
        L = _chol(H)
        if L is None:
            return b, 2, None
        step = _chol_solve(L, g)
        if not np.all(np.isfinite(step)):
            return b, 2, None
        s = 1.0
        if tilt:
            obj = lambda v: np.sum(w * (D * (X @ v) - (1 - D) * np.exp(X @ v)))
            with np.errstate(over="ignore", invalid="ignore"):
                f0 = obj(b)
                while not obj(b + s * step) >= f0 - 1e-12 * abs(f0) and s > 1e-8:
                    s /= 2
        b = b + s * step
        if np.max(np.abs(s * step)) < TOL:
            return b, 0, (None if tilt else _chol_solve(L, np.eye(L.shape[0])))
    return b, 1, None


def _fit(X, D, w, warm, b0, tilt):
    """Warm-started fit, retried from the reference's start (zero; for tilting, the constant-only
    solution) if it fails. Returns (b, status, V, cold)."""
    start = np.zeros(X.shape[1])
    if tilt:
        start[-1] = np.log(np.sum(w * D) / np.sum(w * (1 - D)))
    if warm:
        b, status, V = _newton(X, D, w, b0, tilt, 50)
        if status == 0:
            return b, 0, V, False
    b, status, V = _newton(X, D, w, start, tilt, 200 if tilt else 100)
    return b, status, V, True


def _overlap_state(mx):
    if abs(mx - OVERLAP) <= NEAR:
        return EXACT
    return SKIP if mx >= OVERLAP else OK


def _overlap(X, D, state):
    """csdid 2.0.1's overlap check by its own unweighted logit, warm-started from state['bo']."""
    if X.shape[1] == 1:
        return SKIP if D.mean() >= OVERLAP else OK
    ones = np.ones(D.size)
    b, status, _, cold = _fit(X, D, ones, state["bo"] is not None, state["bo"], False)
    if status != 0:
        return EXACT
    eta = X @ b
    if np.abs(eta).max() > BIG_INDEX and not cold:
        if _newton(X, D, ones, np.zeros(X.shape[1]), False, 100)[1] != 0:
            return EXACT
    state["bo"] = b
    return _overlap_state(_expit(eta).max())


def _invariants(method, X, D, ws, ps, keep):
    """Per-unit weights and the sums over the cell's units that do not depend on the outcome change
    (shared by cells with the same units and covariates): w1 = ws D; w0 = ws ps / (1 - ps) for the
    comparison units kept after trimming (the comparison weights, or drimp's comparison factor)."""
    w1 = ws * D
    w0 = np.zeros_like(ws) if method == "reg" else keep * ws * (ps / (1.0 - ps)) * (1.0 - D)
    inv = dict(w1=w1, w0=w0, s1=w1.sum(), s0=w0.sum(), M1=X.T @ w1, M3=X.T @ w0)
    inv["Mxx"] = _wgram(X, w0) if method == "dripw" else None
    return inv


def _cell_constants(method, X, D, sw0, dy, I, L, inv, V):
    """Cell constants of the influence function, (a, eta_t, eta_c, mean w1, mean w0), U and V2, from the
    fit's invariants I and the sums that depend on the outcome change (the regression residual
    v = dy - x'br is expanded: e.g. sum w0 v x = q - Mxx br)."""
    k1 = X.shape[1]
    n = dy.size
    s1, s0, M1, M3 = I["s1"], I["s0"], I["M1"], I["M3"]
    a1, a0 = I["w1"] @ dy, I["w0"] @ dy
    Z = X.T @ np.column_stack([I["w0"] * dy, (1.0 - D) * sw0 * dy])
    q = Z[:, 0]
    br = _chol_solve(L, Z[:, 1]) if method in ("dripw", "reg", "drimp") else np.zeros(k1)
    U = np.zeros(k1)
    V2 = np.zeros(k1)
    m1b, m3b = M1 @ br, M3 @ br
    if method == "reg":
        eta_t, eta_c = a1 / s1, m1b / s1
        U = inv @ M1
    elif method == "ipw":
        eta_t, eta_c = a1 / s1, a0 / s1
        V2 = V @ q
    elif method == "stdipw":
        eta_t, eta_c = a1 / s1, a0 / s0
        V2 = V @ (q - eta_c * M3)
    elif method == "drimp":
        eta_t, eta_c = (a1 - m1b) / s1, (a0 - m3b) / s1
    else:
        eta_t, eta_c = (a1 - m1b) / s1, (a0 - m3b) / s0
        V2 = V @ (q - I["Mxx"] @ br - eta_c * M3)
        U = inv @ M3 / (s0 / n) - inv @ M1 / (s1 / n)
    return (eta_t - eta_c, eta_t, eta_c, s1 / n, s0 / n), U, V2, br


# ----------------------------------------------------------------------------------------------
# Cells
# ----------------------------------------------------------------------------------------------

def _cell_counts(gvar, w, obs, cg, ct, cb, cm, notyet):
    """Cell sizes and weight sums (all units, cohort units): from cohort tallies where both of a
    cell's periods are observed for every unit (or for none), from masks otherwise."""
    full, none = obs.all(0), ~obs.any(0)
    vals, inv = np.unique(gvar, return_inverse=True)
    cnt = np.bincount(inv, minlength=vals.size).astype(np.int64)
    wsum = np.bincount(inv, weights=w, minlength=vals.size)
    c0 = int(cnt[0]) if vals[0] == 0 else 0
    w0 = float(wsum[0]) if vals[0] == 0 else 0.0
    suf_n = np.append(np.cumsum(cnt[::-1])[::-1], 0)
    suf_w = np.append(np.cumsum(wsum[::-1])[::-1], 0.0)
    gi = np.searchsorted(vals, cg)
    n1, sw1 = cnt[gi], wsum[gi]
    n, sw = n1 + c0, sw1 + w0
    if notyet:
        mi = np.searchsorted(vals, cm, side="right")
        inside = cg > cm
        n = n + suf_n[mi] - np.where(inside, n1, 0)
        sw = sw + suf_w[mi] - np.where(inside, sw1, 0.0)
    tb_full = full[ct - 1] & full[cb - 1]
    tb_none = none[ct - 1] | none[cb - 1]
    n, n1, sw, sw1 = (np.where(tb_full, v, 0) for v in (n, n1, sw, sw1))
    for c in np.flatnonzero(~tb_full & ~tb_none):
        s = _members(gvar, obs, cg[c], ct[c], cb[c], cm[c], notyet)
        n[c], n1[c] = s.sum(), (s & (gvar == cg[c])).sum()
        sw[c], sw1[c] = w[s].sum(), w[s & (gvar == cg[c])].sum()
    return n.astype(np.int64), n1.astype(np.int64), sw.astype(np.float64), sw1.astype(np.float64)


def _members(gvar, obs, g, t, b, m, notyet):
    grp = (gvar == 0) | (gvar == g) | ((gvar > m) if notyet else False)
    return grp & obs[:, b - 1] & obs[:, t - 1]


def _exact_cell(gvar, obs, y, Xp, w, g, t, b, p, m, notyet, method, trim, N, overlap):
    """One cell with the reference's own NumPy operations (None if csdid 2.0.1's overlap check
    fails): its units, influence function, ATT and standard error as the reference computes them."""
    s = _members(gvar, obs, g, t, b, m, notyet)
    D = (gvar[s] == g).astype(np.float64)
    Xs = Xp(p)[s]
    if overlap and _r_overlap_fails(D, Xs):
        return None
    ws = w[s] / w[s].mean()
    r = _r_cell_rif(y[s, t - 1] - y[s, b - 1], Xs, D, ws, method, trim)
    full = np.zeros(N)
    full[s] = (r - r.mean()) * N / s.sum()
    two = np.column_stack([full + r.mean()] * 2)
    mu = two.mean(0)
    return np.flatnonzero(s), r, float(mu[0]), float(np.sqrt(((two - mu) ** 2).sum(0))[0] / N)


def _estimate(gvar, obs, y, Xp, w, cg, ct, cb, cp, cm, n, sw, key, method, notyet, trim, overlap, unweighted):
    """Cell constants for every cell (cohorts in turn, their cells in order; cells with equal
    key share the fits). Returns BR, BP, U, V2, SC and flag (OK, EXACT or SKIP)."""
    C = cg.size
    k1 = Xp(0).shape[1]
    BR, BP, U, V2 = (np.zeros((C, k1)) for _ in range(4))
    SC = np.zeros((C, 6))
    flag = np.zeros(C, np.int64)
    tilt = method == "drimp"
    c = 0
    while c < C:
        g = cg[c]
        fits = dict(bp=None, bo=None)
        prev = None
        while c < C and cg[c] == g:
            t, b = ct[c], cb[c]
            if prev is not None and key[c] >= 0 and key[c] == prev["key"]:
                cell = prev
            else:
                prev = cell = None
                s = _members(gvar, obs, g, t, b, cm[c], notyet)
                idx = np.flatnonzero(s)
                X = Xp(cp[c])[idx]
                D = (gvar[idx] == g).astype(np.float64)
                ws = w[idx] / (sw[c] / n[c])
                cell = dict(key=key[c], idx=idx, X=X, D=D, ws=ws, state=OK)
                state = OK
                if overlap and (tilt or not unweighted):
                    state = _overlap(X, D, fits)
                ps = np.ones(idx.size)
                keep = np.ones(idx.size)
                bp = np.zeros(k1)
                V = None
                if state == OK and method != "reg":
                    bp, status, V, cold = _fit(X, D, ws, fits["bp"] is not None, fits["bp"], tilt)
                    if status != 0:
                        state = EXACT
                    else:
                        fits["bp"] = bp
                        eta = X @ bp
                        raw = _expit(eta)
                        ps = np.minimum(raw, CAP)
                        if trim < 1.0:
                            keep = np.where((D == 0) & (raw > trim), 0.0, 1.0)
                            if np.any((D == 0) & (np.abs(raw - trim) <= NEAR)):
                                state = EXACT
                        if state == OK and overlap and not tilt and unweighted:
                            if k1 == 1:
                                state = SKIP if D.mean() >= OVERLAP else OK
                            else:
                                if np.abs(eta).max() > BIG_INDEX and not cold:
                                    if _newton(X, D, ws, np.zeros(k1), False, 100)[1] != 0:
                                        state = EXACT
                                if state == OK:
                                    state = _overlap_state(raw.max())
                L = inv = None
                sw0 = (ps / (1.0 - ps) * ws) if tilt else ws
                if state == OK and method in ("dripw", "reg", "drimp"):
                    A = _wgram(X, sw0 * (1.0 - D))
                    L = _chol(A)
                    if L is None:
                        state = EXACT
                    else:
                        inv = _chol_solve(L, np.eye(k1))
                cell.update(state=state, bp=bp, V=V, L=L, inv=inv, sw0=sw0,
                            I=_invariants(method, X, D, ws, ps, keep) if state == OK else None)
            prev = cell
            state = cell["state"]
            if state != OK:
                flag[c] = state
                c += 1
                continue
            dy = y[cell["idx"], t - 1] - y[cell["idx"], b - 1]
            sc, U[c], V2[c], br = _cell_constants(method, cell["X"], cell["D"], cell["sw0"], dy, cell["I"],
                                                  cell["L"], cell["inv"], cell["V"])
            SC[c, :5] = sc
            SC[c, 5] = sw[c] / n[c]
            BR[c], BP[c] = br, cell["bp"]
            c += 1
        if prev is not None:
            _release(prev["X"].nbytes * 8)
        prev = cell = None
    return BR, BP, U, V2, SC, flag


# ----------------------------------------------------------------------------------------------
# Pass over units
# ----------------------------------------------------------------------------------------------

def _block_dev(method, Xb, Yb, Ob, wb, d, cols, ct, cb, cp, BR, BP, U, V2, SC, trim, Xpb):
    """Influence-function values minus the cell ATTs (r - a) of a block of units for the cells
    `cols` ([n_block, n_cells]; zero outside a cell), and the membership mask."""
    tt, bb = ct[cols] - 1, cb[cols] - 1
    mem = Ob[:, tt] & Ob[:, bb]
    dev = np.zeros(mem.shape)
    for p in np.unique(cp[cols]):
        sel = np.flatnonzero(cp[cols] == p)
        X = Xb if Xpb is None else Xpb(p)
        cc = cols[sel]
        a, eta_t, eta_c, mw1, mw0, wbar = (SC[cc, j] for j in range(6))
        dd = d[sel]
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            dy = Yb[:, tt[sel]] - Yb[:, bb[sel]]
            wsi = wb[:, None] / wbar
            w1 = wsi * dd
            xb = X @ BR[cc].T
            if method == "reg":
                xu = X @ U[cc].T
                r = (a + (w1 * dy - w1 * eta_t) / mw1
                     - (w1 * xb - w1 * eta_c + wsi * (1.0 - dd) * (dy - xb) * xu) / mw1)
            else:
                raw = _expit(X @ BP[cc].T)
                ps = np.minimum(raw, CAP)
                kp = np.where((dd == 0.0) & (raw > trim), 0.0, 1.0)
                if method == "drimp":
                    summand = kp * (1.0 - (1.0 - dd) / (1.0 - ps)) * (dy - xb)
                    r = a + wsi * kp * (summand - dd * a) / mw1
                else:
                    w0 = kp * wsi * ps * (1.0 - dd) / (1.0 - ps)
                    lin = wsi * (dd - ps) * (X @ V2[cc].T)
                    if method == "ipw":
                        r = a + (w1 * dy - w0 * dy - lin - w1 * a) / mw1
                    elif method == "stdipw":
                        r = a + (w1 * dy - w1 * eta_t) / mw1 - (w0 * dy - w0 * eta_c + lin) / mw0
                    else:
                        res = dy - xb
                        r = (a + (w1 * res - w1 * eta_t) / mw1 - (w0 * res - w0 * eta_c + lin) / mw0
                             + wsi * (1.0 - dd) * res * (X @ U[cc].T))
            dev[:, sel] = np.where(mem[:, sel], r - a, 0.0)
    return dev, mem


def _order_index(B, p, rules):
    """0-based index of reference.order_stat: ceil(p B) (2.0.1) or ceil(p (B + 1)) (1.81), clamped."""
    k = np.ceil(p * B) if rules == "csdid201" else np.ceil(p * (B + 1))
    return min(max(int(k), 1), B) - 1


def _se_boot(Dr, rules):
    """Bootstrap SEs (interquartile range / 1.349) of draws stored one row per column [K, B]."""
    B = Dr.shape[1]
    i25, i75 = _order_index(B, 0.25, rules), _order_index(B, 0.75, rules)
    q = np.partition(Dr, [i25, i75], axis=1)
    return (q[:, i75] - q[:, i25]) / IQR_NORM


def _tmax_update(tm, Dr, se):
    """tm = max(tm, max_k |Dr[k] / se[k]|) with np.max's NaN propagation."""
    with np.errstate(invalid="ignore", divide="ignore"):
        cur = np.max(np.abs(Dr / se[:, None]), axis=0)
    return np.where(np.isnan(tm) | np.isnan(cur), np.nan, np.maximum(tm, cur))


def _crit(tm, alpha, floor, rules):
    tm = tm[np.isfinite(tm)]
    i = _order_index(tm.shape[0], 1 - alpha, rules)
    crit = float(np.partition(tm, i)[i])
    if rules == "csdid201" and floor and not crit >= NormalDist().inv_cdf(1 - alpha / 2):
        crit = NormalDist().inv_cdf(1 - alpha / 2)
    return crit


# ----------------------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------------------

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
    if varying:
        y = np.where(np.isfinite(X).all(2), y, np.nan)
    units, obs = _sample(y, gvar, control)
    if not units.all():
        y, gvar, w, X = y[units], gvar[units], w[units], X[units]
    N = int(units.sum())
    T = obs.shape[1]
    k = X.shape[-1]
    k1 = k + 1
    if varying:                                         # covariates of period p (0-based), with a constant
        def Xp(p, rows=slice(None)):
            out = np.empty((X[rows, 0, 0].size, k1))
            out[:, :k] = X[rows, p, :]
            out[:, k] = 1.0
            return out
    else:
        Xc = np.empty((N, k1))
        Xc[:, :k] = X
        Xc[:, k] = 1.0
        Xp = lambda p, rows=slice(None): Xc[rows]
    w = w / w.mean()
    unweighted = unweighted or bool(np.all(w == 1.0))
    notyet = control == "notyet"
    r201 = rules == "csdid201"
    overlap = r201 and method != "reg"
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
    cm = ct.copy() if (notyet and r201 and base_period == "varying") else np.maximum(cg, ct)
    cp = (np.minimum(cb, ct) - 1) if varying else np.zeros(cg.size, np.int64)
    n, n1, sw, sw1 = _cell_counts(gvar, w, obs, cg, ct, cb, cm, notyet)
    keep = (n1 > 0) & (n1 < n)
    cg, ct, cb, cm, cp, n, sw, sw1 = (v[keep] for v in (cg, ct, cb, cm, cp, n, sw, sw1))

    # 2. estimation
    C = cg.size
    balanced = bool(obs[:, :int(max(ct.max(initial=1), cb.max(initial=1)))].all()) if C else True
    if balanced:
        key = np.searchsorted(cohorts, cm, side="right") if notyet else np.zeros(C, np.int64)
        key = key * (T + 1) + cp
    else:
        key = -np.ones(C, np.int64)
    BR, BP, U, V2, SC, flag = _estimate(gvar, obs, y, Xp, w, cg, ct, cb, cp, cm, n, sw, key, method, notyet,
                                        float(trim), overlap, unweighted)
    exact = {}
    for c in np.flatnonzero(flag == EXACT):
        res = _exact_cell(gvar, obs, y, Xp, w, cg[c], ct[c], cb[c], cp[c], cm[c], notyet, method, trim, N, overlap)
        if res is None:
            flag[c] = SKIP
        else:
            exact[c] = res
            SC[c, 0] = res[1].mean()
    est = flag != SKIP
    newpos = np.cumsum(est) - 1
    exact = {int(newpos[c]): v for c, v in exact.items()}
    cg, ct, cb, cm, cp, n, sw, sw1, BR, BP, U, V2, SC, flag = (
        v[est] for v in (cg, ct, cb, cm, cp, n, sw, sw1, BR, BP, U, V2, SC, flag))
    C = cg.size
    e = ct - cg
    post = e >= 0
    if r201:
        cw = np.bincount(np.searchsorted(cohorts, gvar[gvar > 0]), weights=w[gvar > 0], minlength=cohorts.size)
        sw1 = cw[np.searchsorted(cohorts, cg)]

    # aggregation layout: [simple, groups..., calendar periods..., event times...] (+ 2.0.1's
    # unit-level cohort weights, for the cohort average's standard error)
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
    n_ext = n_agg + n_w
    agg = np.full((C, 4), -1, np.int64)
    agg[post, 0] = 0
    agg[post, 1] = 1 + np.searchsorted(groups, cg[post])
    agg[post, 2] = G1 + np.searchsorted(periods, ct[post])
    inev = np.isin(e, ev_cols)
    agg[inev, 3] = P1 + np.searchsorted(ev_cols, e[inev])
    S = sw1.sum() / N
    mw = sw1 / N / S
    MA = np.zeros(n_agg)
    for q in range(4):
        sel = agg[:, q] >= 0
        np.add.at(MA, agg[sel, q], mw[sel])
    a = SC[:, 0]
    attA = np.zeros(n_agg)
    for q in range(4):
        sel = agg[:, q] >= 0
        np.add.at(attA, agg[sel, q], a[sel] * mw[sel])
    attA /= np.where(MA > 0, MA, 1.0)
    Alpha = np.zeros((C, n_agg))                        # dev -> aggregate influence functions
    Beta = np.zeros((C, n_agg))                         # cohort weights' terms (times w)
    for q in range(4):
        sel = np.flatnonzero(agg[:, q] >= 0)
        Alpha[sel, agg[sel, q]] += mw[sel] / MA[agg[sel, q]] * N / n[sel]
        Beta[sel, agg[sel, q]] += (a[sel] - attA[agg[sel, q]]) / MA[agg[sel, q]] / S
    EX = {c: np.zeros(N) for c in exact}                # exact cells' r - a
    for c, (idx, r, _, _) in exact.items():
        EX[c][idx] = r - a[c]

    # 3./4. pass over units in blocks of one cohort value; bootstrap tiles of cells and draws
    boot = boot_weights is not None
    v = np.asarray(boot_weights) if boot else None
    Bd = v.shape[1] if boot else 0
    rows_all = np.flatnonzero(units) if boot else None
    order = np.argsort(gvar, kind="stable")
    vals, starts, counts = np.unique(gvar[order], return_index=True, return_counts=True)
    per_cell = boot and not (r201 or balanced)
    S1 = np.zeros(C)
    S2 = np.zeros(C)
    Gram = np.zeros((n_ext, n_ext))
    Esum = np.zeros(n_ext)
    ctile = max(1, DRAWS // max(Bd, 1))                 # cells per bootstrap tile
    tiles = [np.arange(c0, min(C, c0 + ctile)) for c0 in range(0, C, ctile)] if boot else [None]
    tm_gt = np.zeros(Bd)
    Graw = np.zeros((n_agg, Bd))
    Gones = np.zeros(Bd)
    WVg = np.zeros((cohorts.size, Bd))
    WVc = np.zeros((C if per_cell else 0, Bd))
    se_boot_gt = np.zeros(C)
    first = True
    for tile in tiles:
        Ggt = np.zeros((tile.size, Bd)) if boot else None
        intile = np.zeros(C, bool)
        if boot:
            intile[tile] = True
        for Gb, st, cnt in zip(vals, starts, counts):
            members = np.flatnonzero((cg == Gb) | (Gb == 0) | ((cm < Gb) if notyet else False))
            if not first:
                members = members[intile[members]]
            if members.size == 0 and not (first and Gb > 0 and r201):
                continue
            d = (cg[members] == Gb).astype(np.float64)
            own = np.flatnonzero(d == 1.0)
            exc = [(j, c) for j, c in enumerate(members) if c in EX]
            if first:
                Am, Bm = Alpha[members], Beta[members]
                wcol = n_agg + np.searchsorted(groups, Gb) if (n_w and Gb > 0 and Gb in groups) else -1
                gi = np.searchsorted(cohorts, Gb) if Gb > 0 else -1
            if boot:
                sel = np.flatnonzero(intile[members])
                rows_t = members[sel] - tile[0]
                trow = slice(rows_t[0], rows_t[-1] + 1) if (sel.size and rows_t[-1] - rows_t[0] + 1 == sel.size) \
                    else rows_t
                inv_n = 1.0 / n[members[sel]]
            nbk = max(1, min(cnt, WORK // max(members.size, 1)))
            for r0 in range(st, st + cnt, nbk):
                ub = order[r0:min(st + cnt, r0 + nbk)]       # sample units of this block
                Ob = obs[ub]
                Yb = y[ub]
                wb = w[ub]
                Xb = Xc[ub] if not varying else None
                Xpb = (lambda p, rows=ub: Xp(p, rows)) if varying else None
                dev, mem = _block_dev(method, Xb, Yb, Ob, wb, d, members, ct, cb, cp, BR, BP, U, V2, SC,
                                      float(trim), Xpb)
                for j, c in exc:
                    dev[:, j] = np.where(mem[:, j], EX[c][ub], 0.0)
                if first:
                    S1[members] += dev.sum(0)
                    S2[members] += (dev * dev).sum(0)
                    E = np.zeros((ub.size, n_ext))
                    E[:, :n_agg] = dev @ Am
                    if own.size:
                        wm = mem[:, own] * wb[:, None] if not r201 else np.broadcast_to(wb[:, None], (ub.size, own.size))
                        E[:, :n_agg] += wm @ Bm[own]
                    if wcol >= 0:
                        E[:, wcol] = wb
                    Gram += E.T @ E
                    Esum += E.sum(0)
                    del E
                if boot:
                    vb = v[rows_all[ub]]
                    dt = dev[:, sel] * inv_n if sel.size else None
                    piece = max(1, DRAWS // max(ub.size, sel.size, 1))
                    for c0 in range(0, Bd, piece):            # draws in pieces of bounded size
                        vf = vb[:, c0:c0 + piece].astype(np.float64)
                        c1 = c0 + vf.shape[1]
                        if first:
                            Gones[c0:c1] += vf.sum(0)
                            if gi >= 0 and not per_cell:
                                WVg[gi, c0:c1] += wb @ vf
                            elif gi >= 0 and own.size:
                                WVc[members[own], c0:c1] += (wb[:, None] * mem[:, own]).T @ vf
                        if dt is not None:
                            Ggt[trow, c0:c1] += dt.T @ vf
                        del vf
                _release(dev.nbytes)
                del dev, mem
        if boot:
            se_boot_gt[tile] = _se_boot(Ggt, rules)
            tm_gt = _tmax_update(tm_gt, Ggt, se_boot_gt[tile])
            Graw += Alpha[tile].T @ (Ggt * n[tile][:, None])
            _release(Ggt.nbytes * 4)
            del Ggt
        first = False

    att_gt = a + S1 / n
    se_gt = np.sqrt(np.maximum(S2 - S1 * S1 / n, 0.0)) / n
    rms = np.sqrt(np.maximum((S2 + 2.0 * a * S1) / n + a * a, 0.0))
    for c in np.flatnonzero((flag == OK) & ~(se_gt > 1e-11 * rms)):
        att_gt[c], se_gt[c] = _exact_cell(gvar, obs, y, Xp, w, cg[c], ct[c], cb[c], cp[c], cm[c], notyet,
                                          method, trim, N, False)[2:]
    for c, (_, _, att, se_c) in exact.items():
        att_gt[c], se_gt[c] = att, se_c

    # aggregates: ATT from the cells' ATTs and weights, SE from the Gram matrix of the units'
    # aggregate influence functions (centered)
    att_agg = np.zeros(n_agg)
    for q in range(4):
        sel = agg[:, q] >= 0
        np.add.at(att_agg, agg[sel, q], att_gt[sel] * mw[sel])
    att_agg /= np.where(MA > 0, MA, 1.0)
    mean = Esum / N
    Gc = Gram - N * np.outer(mean, mean)

    def se_of(coef):
        return float(np.sqrt(max(coef @ Gc @ coef, 0.0)) / N)

    gsel = np.arange(1, G1)
    psel = np.arange(G1, P1)
    esel = P1 + np.searchsorted(ev_cols, events)
    pre, pst = events < 0, events >= 0
    att_group = att_agg[gsel]
    c_ga = np.zeros(n_ext)
    if r201:
        m_w = cw[np.searchsorted(cohorts, groups)] / N
        att_ga = float((att_group * m_w).sum() / m_w.sum())
        c_ga[gsel] = m_w / m_w.sum()
        c_ga[n_agg:] = (att_group - att_ga) / m_w.sum()
    else:
        gw = np.array([mw[post & (cg == g)].mean() for g in groups])
        gamma = gw / gw.sum()
        att_ga = float(att_group @ gamma)
        c_ga[gsel] = gamma
    unit = lambda j: np.eye(1, n_ext, j)[0]
    c_cal = np.zeros(n_ext)
    c_cal[psel] = 1.0 / psel.size
    c_pre = np.zeros(n_ext)
    c_post = np.zeros(n_ext)
    if pre.any():
        c_pre[esel[pre]] = 1.0 / pre.sum()
    if pst.any():
        c_post[esel[pst]] = 1.0 / pst.sum()
    diag = np.sqrt(np.maximum(np.diag(Gc), 0.0)) / N
    out = dict(gt=np.column_stack([cg, ct]).astype(np.int64), att_gt=att_gt, se_gt=se_gt,
               group_ids=groups, att_group=att_group, se_group=diag[gsel],
               calendar_periods=periods, att_calendar=att_agg[psel], se_calendar=diag[psel],
               event_times=events, att_event=att_agg[esel], se_event=diag[esel],
               cov_event=Gc[np.ix_(esel, esel)] / N ** 2)
    out["event_sets"] = [s.tolist() for s in sets]
    att_sets, se_sets = [], []
    for s in sets:
        present = np.array([kk for kk in s if kk in all_events], dtype=np.int64)
        cols = P1 + np.searchsorted(ev_cols, present)
        coef = np.zeros(n_ext)
        coef[cols] = 1.0 / present.size
        att_sets.append(float(att_agg[cols].mean()))
        se_sets.append(se_of(coef))
    out["att_event_sets"] = np.array(att_sets, dtype=np.float64)
    out["se_event_sets"] = np.array(se_sets, dtype=np.float64)
    out["att_simple"], out["se_simple"] = float(att_agg[0]), float(diag[0])
    out["att_group_avg"], out["se_group_avg"] = att_ga, se_of(c_ga)
    out["att_calendar_avg"], out["se_calendar_avg"] = float(att_agg[psel].mean()), se_of(c_cal)
    out["att_pre_avg"], out["se_pre_avg"] = ((float(att_agg[esel][pre].mean()), se_of(c_pre)) if pre.any()
                                             else (float("nan"), float("nan")))
    out["att_post_avg"], out["se_post_avg"] = ((float(att_agg[esel][pst].mean()), se_of(c_post)) if pst.any()
                                               else (float("nan"), float("nan")))

    if boot:
        if per_cell:
            Graw += Beta.T @ WVc
        else:
            Bg = np.zeros((cohorts.size, n_agg))
            np.add.at(Bg, np.searchsorted(cohorts, cg), Beta)
            Graw += Bg.T @ WVg
        if n_w:
            Graw = np.vstack([Graw, WVg[np.searchsorted(cohorts, groups)]])
        Dagg = (Graw - mean[:, None] * Gones) / N
        out["se_boot_gt"] = se_boot_gt
        if not np.isfinite(tm_gt).any():
            raise IndexError("index -1 is out of bounds for axis 0 with size 0")
        out["crit_gt"] = _crit(tm_gt, alpha, False, rules)
        for name, sel in (("group", gsel), ("calendar", psel), ("event", esel)):
            Dr = Dagg[sel]
            se_b = _se_boot(Dr, rules)
            out[f"se_boot_{name}"] = se_b
            out[f"crit_{name}"] = _crit(_tmax_update(np.zeros(Bd), Dr, se_b), alpha, True, rules)
        c_simple = unit(0)
        for name, coef in (("simple", c_simple), ("group_avg", c_ga), ("calendar_avg", c_cal),
                           ("pre_avg", c_pre if pre.any() else None), ("post_avg", c_post if pst.any() else None)):
            out[f"se_boot_{name}"] = (np.nan if coef is None else float(_se_boot((coef @ Dagg)[None, :], rules)[0]))
    return out
