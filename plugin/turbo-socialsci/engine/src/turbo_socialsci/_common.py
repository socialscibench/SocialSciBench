"""Shared helpers: reading tables, thread limits, memory estimates."""
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .errors import NotApplicable

VARIANTS = ("auto", "speed", "memory")


def read_table(data, columns=None):
    """A pandas DataFrame from a DataFrame or a .parquet, .csv, .dta or .feather path (only `columns`)."""
    import pandas as pd
    if isinstance(data, pd.DataFrame):
        missing = [c for c in (columns or []) if c not in data.columns]
        if missing:
            raise KeyError(f"columns not in the data: {missing}")
        return data
    path = Path(data)
    suffix = path.suffix.lower()
    cols = list(dict.fromkeys(columns)) if columns else None
    if suffix == ".parquet":
        return pd.read_parquet(path, columns=cols)
    if suffix == ".csv":
        return pd.read_csv(path, usecols=cols)
    if suffix == ".dta":
        return pd.read_stata(path, columns=cols, convert_categoricals=False)
    if suffix == ".feather":
        return pd.read_feather(path, columns=cols)
    raise ValueError(f"unsupported file type {suffix!r}: use .parquet, .csv, .dta or .feather, or a DataFrame")


def check_variant(variant):
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")


def total_memory():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def choose(variant, speed_bytes):
    """'auto' picks the speed core unless its estimated peak exceeds half of physical memory."""
    if variant != "auto":
        return variant
    mem = total_memory()
    return "memory" if (mem and speed_bytes > 0.5 * mem) else "speed"


@contextmanager
def thread_limit(threads, numba=True):
    """Limit BLAS (threadpoolctl) and, for the Numba cores, Numba's threads during a call; None keeps
    the libraries' settings (and the NUMBA_NUM_THREADS / OMP_NUM_THREADS environment variables)."""
    if threads is None:
        yield
        return
    threads = int(threads)
    if threads < 1:
        raise ValueError("threads must be at least 1")
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=threads):
        if numba:
            import numba as nb
            old = nb.get_num_threads()
            nb.set_num_threads(min(threads, nb.config.NUMBA_NUM_THREADS))
            try:
                yield
            finally:
                nb.set_num_threads(old)
        else:
            yield


def codes_of(df, spec):
    """Integer codes of one variable or an interaction 'a#b#c' (missing in any part -> -1)."""
    import pandas as pd
    parts = [p.strip() for p in spec.split("#")]
    if len(parts) == 1:
        codes, _ = pd.factorize(df[parts[0]], use_na_sentinel=True)
        return codes.astype(np.int64)
    sub = [pd.factorize(df[p], use_na_sentinel=True)[0].astype(np.int64) for p in parts]
    bad = np.zeros(len(df), dtype=bool)
    for s in sub:
        bad |= s < 0
    combo = np.zeros(len(df), dtype=np.int64)
    for s in sub:                       # mixed radix over the parts' levels, re-factorized to stay small
        combo = combo * (int(s.max(initial=0)) + 1) + np.maximum(s, 0)
        combo = pd.factorize(combo)[0].astype(np.int64)
    combo[bad] = -1
    return combo


def not_numeric(df, cols):
    import pandas as pd
    return [c for c in cols if not (pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]))]


def require(cond, reason):
    if not cond:
        raise NotApplicable(reason)
