---
name: record-linkage
description: Links two files of firm or person records by name and address the way Stata's reclink2 does (perfect matches on every variable, blocking on the required() variables plus at least one other variable, reclink2's bigram comparator with match and non-match weights in single precision, and every best-scoring candidate, ties kept, at or above minscore), fast on registers of millions of records. Use for reclink or reclink2 linkages, matching company or establishment records to a register by name, street, city, ZIP and state, and fuzzy name-and-address merges meant to reproduce reclink2.
compatibility: Python 3.12 or 3.13. Installs numpy 2.3.5, numba 0.62.1, scipy, pandas and pyarrow from PyPI.
---

# Name and address record linkage (reclink2-style)

## Does it apply?

Use it for reclink2's computation with **four or more match variables** and ASCII text.

Not covered:
- fewer than four variables (reclink2 then blocks differently);
- non-ASCII text;
- reclink2's other options (`npairs()`, `manytoone`, `exactstr()`, `orblock()`);
- other comparators (Jaro-Winkler, Levenshtein);
- probabilistic linkage (Fellegi-Sunter, Splink, fastLink).

For these, do the task with your usual tools: `reclink` raises `turbo_socialsci.NotApplicable` (CLI
exit code 3) before computing anything.

## Install (once per Python environment)

```bash
python3.12 -m venv .venv          # Python 3.12 or 3.13
.venv/bin/pip install -q "${CLAUDE_PLUGIN_ROOT}"/wheels/turbo_socialsci-*.whl
```

Then run your scripts with `.venv/bin/python`. Installing pulls NumPy, Numba, SciPy, pandas and pyarrow from
PyPI; `uv pip install` works too where uv runs.

## Use

```python
from turbo_socialsci import reclink, warmup

warmup("reclink")              # loads (first time after install: compiles) the kernels; untimed
m = reclink("master.parquet", "register.parquet", idmaster="idm", idusing="idu",
            vars=["name", "addr", "city", "zip", "state"],
            wmatch=[10, 8, 5, 5, 5], wnomatch=[2, 3, 4, 4, 4], required=["state"],
            minscore=0.6, minbigram=0.6, threads=4)
m                      # DataFrame idm, idu, score: one row per kept pair
m.attrs["n_perfect"]   # perfect pairs (equal on every variable; score 1; every duplicate combination)
```

- `master` and `using` are DataFrames or `.parquet`, `.csv`, `.dta` or `.feather` paths. Pass string columns as they are: pandas' default (Arrow-backed) strings are read in place, and Python `str` objects (`object` dtype) are converted once.
- Both files have every variable in `vars` and their own id column.
- To link part of the master file, filter the DataFrame first. As in reclink2, using records that perfectly match a master record are removed from the other master records' candidates, so results depend on the master file as given.
- Equivalent to Stata's `reclink2 vars using file, idmaster() idusing() gen(score) wmatch() wnomatch() required() minscore() minbigram()`.
- Strings are compared exactly as given; there is no standardization. Missing values count as empty strings.
- For master records without a perfect match, the result keeps the best-scoring candidates, all of them when tied, if the best score is at least `minscore`.
- `variant="memory"` reads the using file's strings where the table holds them instead of copying them into fixed-width rows, and returns freed memory to the OS as it goes. It needs about 40% of the speed variant's memory and is up to 1.25× slower; the results are identical.
- `threads` sets Numba threads for the call.
- The first call in a fresh process loads the compiled kernels from a cache, which takes about a second. The first call after installation compiles them, which takes about 80 seconds (70 more for the memory variant). Call `warmup("reclink")`, or `warmup("reclink", variant="memory")` for the memory variant, before timing anything.
- `check_only=True` checks coverage without linking.

## Command line

```bash
turbo-socialsci reclink --master m.parquet --using u.parquet --idmaster idm --idusing idu \
    --vars name addr city zip state --wmatch 10 8 5 5 5 --wnomatch 2 3 4 4 4 --required state \
    --threads 4 --out matches.csv       # exit 3 = not applicable
```
