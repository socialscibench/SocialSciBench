"""Task 002 candidate: doubly robust staggered DiD with shared nuisance fits.

The reference fits, for every (cohort, period) cell, a logit of cohort membership and an OLS of
the outcome change among never-treated units. Neither depends on the period:

- the logit's sample (never-treated plus cohort g), outcome and covariates are fixed per cohort,
  so it is fitted once per cohort;
- the OLS design (covariates of all never-treated units) is the same in every cell, and the
  outcome change is linear in y, so one regression of every period on X gives each cell's
  residuals as R[:, t] - R[:, base].

Every term of a cell's influence function is then a per-unit expression in those residuals
and a few per-cohort sums. One compiled pass over the units evaluates it for every cell and
accumulates the cell variances and the simple and event-time aggregate influence functions
directly, without storing the n_units x n_cells matrix.
"""
import numpy as np
from numba import njit

TRIM = 0.995


@njit(cache=False)
def logit_terms(X0, XG, b, H, grad):
    """Hessian X'WX and score X'(D - p) of the cohort logit (controls D=0, members D=1),
    accumulated over row blocks with BLAS."""
    p = b.size
    H[:] = 0.0
    grad[:] = 0.0
    blk = 2048
    for part in range(2):
        A = X0 if part == 0 else XG
        n = A.shape[0]
        for s in range(0, n, blk):
            Ab = A[s:min(s + blk, n)]
            pr = 1.0 / (1.0 + np.exp(-(Ab @ b)))
            wA = Ab * (pr * (1.0 - pr))[:, None]
            H += wA.T @ Ab
            grad += Ab.T @ (part - pr)


def logit(X0, XG, b, tol=1e-12, maxiter=100):
    """Newton-Raphson logit as in the reference, from start value b. Returns b and inv(H).

    The reference recomputes the Hessian after the final step; that step is below 1e-12,
    so the Hessian from the last iteration is used instead.
    """
    p = b.size
    H, grad = np.empty((p, p)), np.empty(p)
    for _ in range(maxiter):
        logit_terms(X0, XG, b, H, grad)
        step = np.linalg.solve(H, grad)
        b = b + step
        if np.max(np.abs(step)) < tol:
            break
    return b, np.linalg.inv(H)


@njit(cache=False)
def control_pass(RC, X0, ctrl, w0, psC, aC, bC, G, g2, off, eta_c, inv_m1, inv_m0,
                 c_simple, c_event, S1, S2, Dsimple, Devent):
    """Never-treated units: each belongs to every cell. For cohort c and period index t,
    with r = R[t] - R[base] and xv = X_i . G[c, :, t] (the propensity-score correction),
      psi - att = -r a/m1 - (w0 (r - eta_c) - ps xv - r b)/m0.
    The base is t - 1 up to period index g2 = g - 2 and g2 afterwards; cell (c, t) has
    event-time column off[c] + t.
    """
    C, p, T = G.shape
    xv = np.empty(T)
    for j in range(ctrl.size):
        i = ctrl[j]
        acc = 0.0
        for c in range(C):
            xv[:] = 0.0
            for k in range(p):
                xk = X0[j, k]
                for t in range(T):
                    xv[t] += xk * G[c, k, t]
            w, ps = w0[c, j], psC[c, j]
            m0 = inv_m0[c]
            alpha = -aC[j, c] * inv_m1[c] + (bC[j, c] - w) * m0
            wm, pm = w * m0, ps * m0
            o = off[c]
            gb = g2[c]
            for t in range(1, T):
                r = RC[j, t] - RC[j, t - 1 if t <= gb else gb]
                d = r * alpha + wm * eta_c[c, t] + pm * xv[t]
                S1[c, t] += d
                S2[c, t] += d * d
                acc += c_simple[c, t] * d
                Devent[i, o + t] += c_event[c, t] * d
        Dsimple[i] += acc


@njit(cache=False)
def member_pass(R, Xc, members, cohort, psG, G, g2, off, eta_t, inv_m1, inv_m0,
                c_simple, c_event, S1, S2, Dsimple, Devent):
    """Treated units, each in its own cohort's cells only:
      psi - att = (r - eta_t)/m1 - (1 - ps) xv/m0.
    """
    C, p, T = G.shape
    for j in range(members.size):
        i = members[j]
        c = cohort[j]
        ps = psG[j]
        m1, m0 = inv_m1[c], inv_m0[c]
        acc = 0.0
        for t in range(1, T):
            s = 0.0
            for k in range(p):
                s += Xc[i, k] * G[c, k, t]
            r = R[i, t] - R[i, t - 1 if t <= g2[c] else g2[c]]
            d = (r - eta_t[c, t]) * m1 - (1.0 - ps) * s * m0
            S1[c, t] += d
            S2[c, t] += d * d
            acc += c_simple[c, t] * d
            Devent[i, off[c] + t] += c_event[c, t] * d
        Dsimple[i] += acc


@njit(cache=False)
def finalize_event(Devent, att_k, wt_event, unit_cohort, pre):
    """Complete the event-time influence functions in place (add each event's ATT and, for
    treated units, the estimated-weight term of their cohort) and return their means,
    centered sums of squares, and the pre- and post-period averages per unit."""
    n, m = Devent.shape
    n_pre = 0
    for k in range(m):
        n_pre += pre[k]
    mean = np.zeros(m)
    rif_pre = np.zeros(n)
    rif_post = np.zeros(n)
    for i in range(n):
        c = unit_cohort[i]
        sp, sq = 0.0, 0.0
        for k in range(m):
            v = Devent[i, k] + att_k[k]
            if c >= 0:
                v += wt_event[c, k]
            Devent[i, k] = v
            mean[k] += v
            if pre[k]:
                sp += v
            else:
                sq += v
        rif_pre[i] = sp / n_pre if n_pre > 0 else np.nan
        rif_post[i] = sq / (m - n_pre) if m > n_pre else np.nan
    mean /= n
    ss = np.zeros(m)
    for i in range(n):
        for k in range(m):
            d = Devent[i, k] - mean[k]
            ss[k] += d * d
    return mean, ss, rif_pre, rif_post


def run(y, X, gvar):
    y = np.asarray(y, dtype=np.float64)
    gvar = np.asarray(gvar)
    N, T = y.shape
    Xc = np.column_stack([X, np.ones(N)])
    p = Xc.shape[1]
    cohorts = np.unique(gvar[gvar > 0])
    C = cohorts.size
    ctrl = np.flatnonzero(gvar == 0)
    nC = ctrl.size
    treated = gvar > 0
    members = np.flatnonzero(treated)
    cohort_of = np.searchsorted(cohorts, gvar[members])
    n_g = np.bincount(cohort_of, minlength=C).astype(np.float64)

    # Outcome model, shared by all cells: OLS of every period on X among never-treated units
    X0 = Xc[ctrl]
    M0inv = np.linalg.inv(X0.T @ X0)
    R = y - Xc @ (M0inv @ (X0.T @ y[ctrl]))
    RC = np.ascontiguousarray(R[ctrl])

    # Cell layout and aggregation weights (cohort size shares; the common scale cancels)
    gt = np.array([(g, t) for g in cohorts for t in range(2, T + 1)], dtype=np.int64)
    e = gt[:, 1] - gt[:, 0]
    events = np.unique(e)
    ev_cell = np.searchsorted(events, e)
    m_w = np.repeat(n_g / N, T - 1)
    post = e >= 0
    W_post = m_w[post].sum()
    W_event = np.bincount(ev_cell, weights=m_w, minlength=events.size)

    # Base period index for each (cohort, period index), period index 0 unused
    tt = np.arange(T)
    base = np.where(tt[None, :] + 1 < cohorts[:, None], tt[None, :] - 1, cohorts[:, None] - 2)
    base[:, 0] = 0

    psC = np.empty((C, nC))
    w0 = np.empty((C, nC))
    psG = np.empty(members.size)
    U = np.empty((p, C))
    U0 = np.empty((p, C))
    G = np.empty((C, p, T))
    eta_t = np.zeros((C, T))
    eta_c = np.zeros((C, T))
    inv_m1 = np.empty(C)
    inv_m0 = np.empty(C)
    b = np.zeros(p)
    for c, g in enumerate(cohorts):
        sel = members[cohort_of == c]
        XG = Xc[sel]
        n = nC + sel.size

        # Propensity score, once per cohort (Newton started from the previous cohort's fit)
        b, V = logit(X0, XG, b)
        ps_raw = 1.0 / (1.0 + np.exp(-(X0 @ b)))
        ps = np.minimum(ps_raw, 1 - 1e-6)
        psC[c] = ps
        w0[c] = ps / (1 - ps) * (ps_raw <= TRIM)
        psG[cohort_of == c] = np.minimum(1.0 / (1.0 + np.exp(-(XG @ b))), 1 - 1e-6)

        # Per-cohort sums; X's last column is the constant, so Q[-1] = sum w0 R
        Q = (X0 * w0[c][:, None]).T @ RC           # sum_C w0 X R, [p, T]
        SwX = w0[c] @ X0
        W0 = SwX[-1]
        SG = R[sel].sum(0)
        bc = base[c]
        eta_t[c] = (SG - SG[bc]) / sel.size
        eta_c[c] = (Q[-1] - Q[-1][bc]) / W0
        G[c] = V @ (Q - Q[:, bc] - SwX[:, None] * eta_c[c][None, :])
        U[:, c] = M0inv @ XG.sum(0)
        U0[:, c] = M0inv @ SwX
        inv_m1[c] = n / sel.size
        inv_m0[c] = n / W0
    att = eta_t - eta_c

    # Per-(cohort, period) coefficients of the aggregate influence functions, incl. N/n
    scale = N / (nC + n_g)
    cell = np.arange(C)[:, None] * (T - 1) + tt[None, 1:] - 1
    c_simple = np.zeros((C, T))
    c_event = np.zeros((C, T))
    c_simple[:, 1:] = np.where(post[cell], m_w[cell] / W_post, 0.0) * scale[:, None]
    g2 = cohorts - 2
    off = ev_cell[cell[:, 0]] - 1                # event column of (c, t) is off[c] + t
    c_event[:, 1:] = m_w[cell] / W_event[ev_cell[cell]] * scale[:, None]

    S1 = np.zeros((C, T))
    S2 = np.zeros((C, T))
    Dsimple = np.zeros(N)
    Devent = np.zeros((N, events.size))
    control_pass(RC, X0, ctrl, w0, psC, X0 @ U, X0 @ U0, G, g2, off, eta_c, inv_m1, inv_m0,
                 c_simple, c_event, S1, S2, Dsimple, Devent)
    member_pass(R, Xc, members, cohort_of, psG, G, g2, off, eta_t, inv_m1, inv_m0,
                c_simple, c_event, S1, S2, Dsimple, Devent)

    n_cell = (nC + n_g)[:, None]
    mean_d = S1[:, 1:] / n_cell
    att_gt = (att[:, 1:] + mean_d).ravel()
    se_gt = (np.sqrt(np.maximum(S2[:, 1:] - n_cell * mean_d ** 2, 0.0)) / n_cell).ravel()

    # Estimated-weight terms of the aggregates depend only on the unit's cohort
    cell_cohort = np.repeat(np.arange(C), T - 1)
    att_s = (att_gt[post] * m_w[post]).sum() / W_post
    wt = np.bincount(cell_cohort[post], weights=att_gt[post] - att_s, minlength=C) / W_post
    rif_simple = Dsimple + att_s
    rif_simple[members] += wt[cohort_of]

    att_k = np.bincount(ev_cell, weights=att_gt * m_w, minlength=events.size) / W_event
    wt_event = np.zeros((C, events.size))
    np.add.at(wt_event, (cell_cohort, ev_cell), (att_gt - att_k[ev_cell]) / W_event[ev_cell])
    unit_cohort = np.full(N, -1, dtype=np.int64)
    unit_cohort[members] = cohort_of
    att_event, ss_event, rif_pre, rif_post = finalize_event(
        Devent, att_k, wt_event, unit_cohort, events < 0)

    def se(v):
        return float(np.sqrt(((v - v.mean()) ** 2).sum()) / N)

    return dict(
        gt=gt, att_gt=att_gt, se_gt=se_gt,
        att_simple=float(rif_simple.mean()), se_simple=se(rif_simple),
        event_times=events.astype(np.int64), att_event=att_event, se_event=np.sqrt(ss_event) / N,
        att_pre_avg=float(rif_pre.mean()), se_pre_avg=se(rif_pre),
        att_post_avg=float(rif_post.mean()), se_post_avg=se(rif_post),
    )
