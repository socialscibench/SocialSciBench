"""Reference implementation for task 009_melogit.

Maximum likelihood for the two-level random-intercept logit fitted by Stata's

    melogit y x1 ... xk || group:

(defaults: 7 integration points, vce(oim)). The likelihood of group j integrates the random
intercept out,

    L_j(b, s) = Int prod_{i in j} Lambda(x_ij'b + u)^y_ij (1 - Lambda(x_ij'b + u))^(1-y_ij) phi(u; 0, s^2) du,

and is approximated by adaptive Gauss-Hermite quadrature with Q = 7 points centred at the mode m_j
of the integrand and scaled by its curvature t_j (u_jq = m_j + sqrt(2) t_j z_q), as in lme4 with
nAGQ = 7. The estimate maximizes this approximation over (b, ln s).

The gradient is exact: it includes the movement of the nodes (m_j, t_j) with the parameters
(implicit-function derivatives of the mode condition). Newton-type steps use the Hessian with the
nodes held fixed; the final Hessian, for the standard errors (OIM), is the central-difference
derivative of the exact gradient. var(_cons) = s^2 with a delta-method standard error.

Deliberately simple: dense NumPy arrays, no tricks. Correct, readable, not optimized.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import expit, log_expit, logsumexp

Q = 7
Z, OMEGA = np.polynomial.hermite.hermgauss(Q)          # nodes and weights for exp(-z^2)


def _logit_start(y, X):
    """Pooled logit (Newton), used as starting values for b."""
    b = np.zeros(X.shape[1])
    for _ in range(50):
        p = expit(X @ b)
        step = np.linalg.solve((X * (p * (1 - p))[:, None]).T @ X, X.T @ (y - p))
        b += step
        if np.max(np.abs(step)) < 1e-12:
            break
    return b


def _modes(y, eta, g, G, s2, m):
    """Mode m_j and curvature scale t_j of each group's integrand (Newton, vectorized over groups)."""
    for _ in range(100):
        p = expit(eta + m[g])
        d1 = np.bincount(g, y - p, G) - m / s2
        d2 = -np.bincount(g, p * (1 - p), G) - 1 / s2
        step = -d1 / d2
        m = m + step
        if np.max(np.abs(step)) < 1e-13:
            break
    p = expit(eta + m[g])
    t = 1 / np.sqrt(np.bincount(g, p * (1 - p), G) + 1 / s2)
    return m, t


def evaluate(y, X, g, C, theta, m0, hessian=True):
    """Log likelihood, exact gradient and (optionally) the fixed-node Hessian at theta = (b, ln s).

    C is the [G, N] 0/1 matrix of group membership (sums over the observations of each group).
    """
    N, K = X.shape
    G = C.shape[0]
    b, lns = theta[:K], theta[K]
    s2 = np.exp(2 * lns)
    eta = X @ b
    m, t = _modes(y, eta, g, G, s2, m0)
    nodes = m[:, None] + np.sqrt(2) * t[:, None] * Z[None, :]                      # [G, Q]
    la = np.empty((G, Q))                                                           # log of each node's term
    D = np.empty((G, Q, K + 1))                                                     # d/dtheta, nodes fixed
    Fp = np.empty((G, Q))                                                           # d/du of the log integrand
    vbar = np.zeros(N) if hessian else None
    p_q = []
    for q in range(Q):
        e = eta + nodes[g, q]
        p = expit(e)
        la[:, q] = np.bincount(g, y * log_expit(e) + (1 - y) * log_expit(-e), G)
        D[:, q, :K] = C @ ((y - p)[:, None] * X)
        Fp[:, q] = np.bincount(g, y - p, G) - nodes[:, q] / s2
        p_q.append(p)
    la += (np.log(np.sqrt(2) * t)[:, None] + np.log(OMEGA)[None, :] + Z[None, :] ** 2
           - 0.5 * np.log(2 * np.pi) - lns - nodes ** 2 / (2 * s2))
    D[:, :, K] = -1 + nodes ** 2 / s2
    lse = logsumexp(la, axis=1)
    P = np.exp(la - lse[:, None])                                                   # weight of each node
    Dbar = np.einsum("jq,jqa->ja", P, D)

    # movement of the nodes: m solves F'(m) = 0, t = (-F''(m))^(-1/2)
    pm = expit(eta + m[g])
    vm = pm * (1 - pm)
    dm = np.empty((G, K + 1))
    dm[:, :K] = -(t ** 2)[:, None] * (C @ (vm[:, None] * X))
    dm[:, K] = t ** 2 * 2 * m / s2
    w3 = vm * (1 - 2 * pm)
    dS = np.empty((G, K + 1))                                                       # d(-F''(m))/dtheta
    dS[:, :K] = C @ (w3[:, None] * X) + np.bincount(g, w3, G)[:, None] * dm[:, :K]
    dS[:, K] = np.bincount(g, w3, G) * dm[:, K] - 2 / s2
    dt = -0.5 * (t ** 3)[:, None] * dS
    A = (P * Fp).sum(axis=1)
    B = 1 / t + (P * Fp * np.sqrt(2) * Z[None, :]).sum(axis=1)
    grad = (Dbar + A[:, None] * dm + B[:, None] * dt).sum(axis=0)

    H = None
    if hessian:
        for q in range(Q):
            vbar += P[g, q] * p_q[q] * (1 - p_q[q])
        H = np.einsum("jq,jqa,jqb->ab", P, D, D) - Dbar.T @ Dbar
        H[:K, :K] -= (X * vbar[:, None]).T @ X
        H[K, K] -= np.sum(P * 2 * nodes ** 2 / s2)
    return lse.sum(), grad, H, m


def run(y, X, group, tol=1e-10, max_iter=200):
    """y: [N] 0/1, X: [N, k] regressors (a constant is added last), group: [N] ids 0..G-1."""
    y = y.astype(float)
    X = np.column_stack([X, np.ones(len(y))])
    N, K = X.shape
    g, G = group, group.max() + 1
    C = csr_matrix((np.ones(N), (g, np.arange(N))), shape=(G, N))
    theta = np.append(_logit_start(y, X), 0.0)
    ll, grad, H, m = evaluate(y, X, g, C, theta, np.zeros(G))
    for it in range(max_iter):
        step = np.linalg.solve(-H, grad)
        if float(grad @ step) < tol:                                    # scaled gradient, as in Stata's ml
            break
        for _ in range(40):                                             # step halving
            ll_new, grad_new, H_new, m_new = evaluate(y, X, g, C, theta + step, m)
            if ll_new >= ll - 1e-12 * abs(ll):
                break
            step /= 2
        theta, ll, grad, H, m = theta + step, ll_new, grad_new, H_new, m_new

    # OIM: central differences of the exact gradient
    Hx = np.empty((K + 1, K + 1))
    for a in range(K + 1):
        h = 1e-5 * max(1.0, abs(theta[a]))
        e = np.zeros(K + 1)
        e[a] = h
        gp = evaluate(y, X, g, C, theta + e, m, hessian=False)[1]
        gm = evaluate(y, X, g, C, theta - e, m, hessian=False)[1]
        Hx[:, a] = (gp - gm) / (2 * h)
    Hx = (Hx + Hx.T) / 2
    V = np.linalg.inv(-Hx)
    s2 = np.exp(2 * theta[K])
    return dict(beta=theta[:K], se=np.sqrt(np.diag(V)[:K]), var_u=s2, se_var_u=2 * s2 * np.sqrt(V[K, K]), ll=ll,
                iterations=it + 1, nobs=N, ngroups=G)
