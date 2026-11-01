"""Reference implementation for task 005_merge (Python, pandas).

Merge a master table with a using table on key variables, keeping all rows and recording
where each row came from, following Stata's

    merge 1:1 keys using using.dta     (or m:1)

which keeps master-only (_merge = 1), using-only (_merge = 2) and matched (_merge = 3) rows.
Like Stata, the merge checks that the keys are unique where the merge type requires it
(both tables for 1:1, the using table for m:1) and stops otherwise.
"""
import pandas as pd

VALIDATE = {"1:1": "one_to_one", "m:1": "many_to_one", "1:m": "one_to_many"}
MERGE_CODE = {"left_only": 1, "right_only": 2, "both": 3}


def run(master, using, keys, kind):
    """
    master  DataFrame with key columns and master variables
    using   DataFrame with key columns and using variables
    keys    list of key column names
    kind    "1:1", "m:1" or "1:m"
    returns the merged DataFrame with a _merge column (1, 2, 3)
    """
    out = pd.merge(master, using, on=keys, how="outer", indicator="_merge", validate=VALIDATE[kind])
    out["_merge"] = out["_merge"].map(MERGE_CODE).astype("int8")
    return out


def checksums(out, keys):
    """Row count, _merge counts, and the count and sum of every non-key numeric column."""
    num = [c for c in out.columns if c not in keys and c != "_merge"]
    return dict(rows=len(out), merge={k: int((out["_merge"] == k).sum()) for k in (1, 2, 3)},
                cols={c: (int(out[c].notna().sum()), float(out[c].sum())) for c in num})
