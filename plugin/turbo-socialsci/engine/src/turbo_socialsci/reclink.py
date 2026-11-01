"""Name and address record linkage as Stata's reclink2 computes it, on the SocialSciBench task-003
solvers."""
import numpy as np

from ._common import check_variant, read_table, require, thread_limit


def _strings(df, col, side, in_place=False):
    """The column as rows of ASCII bytes (2-D, copied) or, in_place, as the variable-length rows of the
    memory core (characters and offsets read where the table holds them); NotApplicable for other text."""
    import pandas as pd
    from ._strings import NON_ASCII, NUL, byte_rows, var_rows
    s = df[col]
    if pd.api.types.is_float_dtype(s):
        raise ValueError(f"column {col!r} is numeric with decimals; convert it to strings first")
    if not (pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s)
            or pd.api.types.is_integer_dtype(s)):
        raise ValueError(f"column {col!r} must hold strings")
    got = var_rows(s) if in_place else None
    if got is not None:
        from .core.reclink_memory import VarRows
        data, off, width, flags = got
        rows = VarRows(data, off, width)
    else:
        rows, flags = byte_rows(s)
    require(not flags & NON_ASCII, f"non-ASCII text in {side} variable {col!r} (reclink2's comparator is "
                                   "implemented for ASCII)")
    require(not flags & NUL, f"NUL characters in {side} variable {col!r}")
    return rows


def reclink(master, using, *, idmaster, idusing, vars, wmatch, wnomatch, required=(), minscore=0.6,
            minbigram=0.6, variant="speed", threads=None, check_only=False):
    """reclink2 VARS using USING, idmaster() idusing() wmatch() wnomatch() required() minscore() minbigram().

    master, using: DataFrames or paths; both have the columns in vars (four or more) and their id column.
    Returns a DataFrame [idmaster, idusing, score], one row per kept pair: perfect matches (equal on
    every variable, score 1, every duplicate combination), then for the other master records the best
    candidate or candidates (ties kept) with score >= minscore. df.attrs["n_perfect"] counts the
    perfect pairs. Strings are compared exactly as given (reclink2 does not standardize them);
    missing values count as empty strings. variant "memory" reads the using file's strings where the
    table holds them (pandas' Arrow-backed strings; Python str objects are converted to Arrow once)
    instead of copying them into fixed-width rows, and returns free memory to the OS as it goes.
    Raises NotApplicable for fewer than four variables (reclink2's other blocking rule) or non-ASCII text.
    """
    import pandas as pd
    check_variant(variant)
    vars = [vars] if isinstance(vars, str) else list(vars)
    require(len(vars) >= 4, "fewer than four match variables: reclink2 then blocks differently, "
                            "which is not implemented")
    require(len(wmatch) == len(vars) and len(wnomatch) == len(vars), "wmatch and wnomatch need one weight per variable")
    required = [required] if isinstance(required, str) else list(required)
    require(all(r in vars for r in required), "required() variables must be among the match variables")
    M = read_table(master, vars + [idmaster])
    U = read_table(using, vars + [idusing])
    mdict = {v: _strings(M, v, "master") for v in vars}
    udict = {v: _strings(U, v, "using", in_place=(variant == "memory")) for v in vars}
    if len({type(a) for a in udict.values()}) > 1:          # over 2 GB of characters in a column
        udict = {v: _strings(U, v, "using") for v in vars}
    if check_only:
        return dict(applies=True, master=len(M), using=len(U))
    if variant in ("speed", "auto"):
        from .core import reclink_speed as core
        use = "speed"
    else:
        from .core import reclink_memory as core
        use = "memory"
    with thread_limit(threads):
        i, j, score, n_perfect = core.run(mdict, udict, vars, list(map(float, wmatch)), list(map(float, wnomatch)),
                                          required, minscore=float(minscore), minbigram=float(minbigram))
    out = pd.DataFrame({idmaster: M[idmaster].to_numpy()[np.asarray(i, dtype=np.int64)],
                        idusing: U[idusing].to_numpy()[np.asarray(j, dtype=np.int64)],
                        "score": np.asarray(score, dtype=np.float64)})
    if idmaster == idusing:
        out.columns = [f"{idmaster}_master", f"{idusing}_using", "score"]
    out.attrs["n_perfect"] = int(n_perfect)
    out.attrs["variant"] = use
    return out
