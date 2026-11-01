"""turbo-socialsci: fast, exact implementations of three social-science computations.

- feols:   OLS with absorbed fixed effects and one-way cluster-robust SEs (Stata reghdfe's conventions)
- att_gt:  Callaway & Sant'Anna (2021) staggered difference-in-differences (Stata csdid 1.81 / 2.0.1)
- reclink: name and address record linkage (Stata reclink2)

Each function raises NotApplicable, before any heavy computation, when a request is outside what it
implements; the caller then does the task another way. Nothing here imports Numba until a compiled
core is used; att_gt's "memory" core is NumPy only except for the bootstrap's kernel (0.2).
"""
from .errors import NotApplicable

__version__ = "0.2.0"
__all__ = ["NotApplicable", "feols", "att_gt", "reclink", "warmup", "__version__"]


def feols(*args, **kwargs):
    from .hdfe import feols as f
    return f(*args, **kwargs)


def att_gt(*args, **kwargs):
    from .did import att_gt as f
    return f(*args, **kwargs)


def reclink(*args, **kwargs):
    from .reclink import reclink as f
    return f(*args, **kwargs)


def warmup(*args, **kwargs):
    from ._warmup import warmup as f
    return f(*args, **kwargs)


for _f, _m in ((feols, "hdfe"), (att_gt, "did"), (reclink, "reclink")):
    _f.__doc__ = f"See turbo_socialsci.{_m}.{_f.__name__}."
