"""Task 008 competitor: the 14 October 2026 study's Numba comparator in the reference pipeline.

The study's `record_numba_full` (research/2026-10-14/tasks007008/implementations.py in the companion
repository) replaces only reference.bigram1 by a compiled per-pair comparator (`full_bigram`, a
sorted-CSR merge of bigram counts plus the short-string and Winkler rules); joins, float32 rounding,
perfect matches and tie selection are the reference's. The study patched the imported reference
module; candidates may not import the reference at runtime, so this file carries a copy of the
reference pipeline (as the evaluator's starter does) with the study's comparator. One thread.
"""
import numpy as np
import pandas as pd
from numba import njit
from scipy.sparse import csr_matrix

BISCALE_POWER = 1 / 3


def f32(x):
    return np.asarray(x, dtype=np.float64).astype(np.float32).astype(np.float64)


class Strings:
    """Distinct values of one variable (master and using pooled): lengths, bigram matrices, prefixes."""

    def __init__(self, values):
        self.values, self.inverse = np.unique(np.asarray(values, dtype=str), return_inverse=True)
        self.len = np.char.str_len(self.values).astype(np.int64)
        L = max(2, int(self.len.max()))
        c = np.frombuffer(self.values.astype(f"S{L}").tobytes(), dtype=np.uint8).reshape(-1, L).astype(np.int64)
        on = np.arange(L - 1)[None, :] < (self.len - 1)[:, None]
        rows = np.nonzero(on)[0]
        cols = (c[:, :-1] * 256 + c[:, 1:])[on]                 # bigram code = 256 * char1 + char2
        n = len(self.values)
        self.pos = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, 65536))  # bigram counts
        self.has = self.pos.copy()                                 # bigram occurs at least once
        self.has.data[:] = 1.0
        self.prefix = c[:, :4] if L >= 4 else np.pad(c, ((0, 0), (0, 4 - L)))


@njit(cache=False)
def full_bigram(a, b, indptr, indices, data, lens, chars):
    out = np.zeros(len(a), np.float64)
    for r in range(len(a)):
        m, s = a[r], b[r]
        lm, ls = lens[m], lens[s]
        if ls == 0:
            continue
        if m == s:
            out[r] = 1.
            continue
        p, pe = indptr[m], indptr[m + 1]
        q, qe = indptr[s], indptr[s + 1]
        cnt = 0.
        while p < pe and q < qe:
            if indices[p] == indices[q]:
                cnt += data[p]
                p += 1
                q += 1
            elif indices[p] < indices[q]:
                p += 1
            else:
                q += 1
        g = 0.
        if lm > 2:
            g = float(np.float32(cnt * 2 / (ls + lm - 2)))
        if lm < 3 or ls < 3:
            shorter, longer = (m, s) if lm <= ls else (s, m)
            low, high = min(lm, ls), max(lm, ls)
            contains = False
            for start in range(high - low + 1):
                equal = True
                for k in range(low):
                    if chars[shorter, k] != chars[longer, start + k]:
                        equal = False
                        break
                if equal:
                    contains = True
                    break
            if contains:
                # Empty master with one-character using yields NaN in reference.
                g = float(np.float32(low / (lm + ls - 1))) if lm + ls != 1 else np.nan
            else:
                g = 0.
        wink = 0
        for k in range(1, 5):
            ml, sl = min(k, lm), min(k, ls)
            if ml != sl:
                continue
            equal = True
            for j in range(ml):
                if chars[m, j] != chars[s, j]:
                    equal = False
                    break
            if equal:
                wink += 1
        out[r] = float(np.float32(g + wink * (1 - g) / 10))
    return out


def bigram1(S, a, b):
    """The study's compiled_full_bigram1: reference bigram1 scores of value ids a (master), b (using)."""
    S.pos.sort_indices()
    L = max(2, int(S.len.max()))
    chars = np.frombuffer(S.values.astype(f'S{L}').tobytes(), dtype=np.uint8).reshape(-1, L)
    return full_bigram(a, b, S.pos.indptr, S.pos.indices, S.pos.data, S.len, chars)

def run(master, using, vars, wmatch, wnomatch, required, minscore=0.6, minbigram=0.6, chunk=20_000):
    """master, using: dicts of equal-length string arrays (empty string = missing).

    Returns arrays (i, j, score): master row, using row, score, one row per retained pair.
    """
    nm, nu = len(master[vars[0]]), len(using[vars[0]])
    S = {v: Strings(np.concatenate([master[v], using[v]])) for v in vars}
    mid = {v: S[v].inverse[:nm] for v in vars}                  # value ids, comparable across files
    uid = {v: S[v].inverse[nm:] for v in vars}
    empty = {v: S[v].len == 0 for v in vars}
    biscale = .5 / (1 - minbigram) ** BISCALE_POWER

    # 1. perfect matches
    M = pd.DataFrame({v: mid[v] for v in vars}).assign(i=np.arange(nm))
    U = pd.DataFrame({v: uid[v] for v in vars}).assign(j=np.arange(nu))
    perf = M.merge(U, on=vars)[["i", "j"]].to_numpy()
    out_i, out_j, out_s = [perf[:, 0]], [perf[:, 1]], [np.ones(len(perf))]
    rest = np.setdiff1d(np.arange(nm), perf[:, 0])
    U = U[~np.isin(U.j.to_numpy(), perf[:, 1])]

    # 2. blocking: required variables, and at least one or-block variable
    orblock = [v for v in vars if v not in required] if len(vars) > 3 else []
    for start in range(0, len(rest), chunk):
        Mc = M.iloc[rest[start:start + chunk]]
        if orblock:
            parts = [Mc[~empty[v][Mc[v].to_numpy()]].merge(U[[v, *required, "j"]], on=[v, *required])[["i", "j"]]
                     for v in orblock]
        else:
            parts = [Mc.merge(U[[*required, "j"]], on=required)[["i", "j"]]]
        pairs = pd.concat(parts).drop_duplicates().to_numpy()
        if len(pairs) == 0:
            continue
        i, j = pairs[:, 0], pairs[:, 1]

        # 3. score
        Mw = np.zeros(len(i))
        NMw = np.zeros(len(i))
        for v, wm, wnm in zip(vars, wmatch, wnomatch):
            a, b = mid[v][i], uid[v][j]
            m_empty, u_empty = empty[v][a], empty[v][b]
            if v in required:
                Mw = np.where(m_empty, Mw, f32(Mw + wm))
            else:
                bi = bigram1(S[v], a, b)
                bonus = np.where(bi > minbigram, biscale * np.clip(bi - minbigram, 0, None) ** BISCALE_POWER, 0.0)
                Mw = np.where(m_empty, Mw, f32(Mw + wm * (.5 * bi ** 2 + bonus)))
                pen = np.where(bi < minbigram - .2, 1.0, np.where(bi < minbigram, 1 - bi ** 2, (1 - bi) ** 2))
                NMw = np.where(m_empty | u_empty, NMw, f32(NMw + wnm * pen))
            NMw = np.where(m_empty != u_empty, f32(NMw + wnm * 0.3), NMw)
        score = f32(Mw / (Mw + NMw))

        # 4. best candidate(s) per master record
        best = pd.Series(score).groupby(i).transform("max").to_numpy()
        keep = (score == best) & (score >= minscore)
        out_i.append(i[keep])
        out_j.append(j[keep])
        out_s.append(score[keep])
    i, j, s = np.concatenate(out_i), np.concatenate(out_j), np.concatenate(out_s)
    o = np.lexsort((j, i))
    return i[o], j[o], s[o], len(perf)

