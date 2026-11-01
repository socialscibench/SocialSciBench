"""Reference implementation for task 003 (record linkage).

Record linkage as done by Stata's

    reclink2 name addr city zip state using using.dta, idmaster(idm) idusing(idu) gen(score) ///
        wmatch(10 8 5 5 5) wnomatch(2 3 4 4 4) required(state)

(Wasi & Flaaen's version of Blasnik's reclink, default minscore(.6) minbigram(.6) npairs(1)):

1. Perfect matches: master records equal to a using record on every variable get score 1; both
   are removed from the rest of the search.
2. Candidates ("blocking"): for each remaining master record, the using records that agree
   exactly on every required() variable and on at least one of the other variables (or-blocking,
   the default with four or more variables). Empty master values are ignored.
3. Score of each candidate pair, variable by variable:
     bi  = bigram1(master value, using value)                       (string comparator, 0-1)
     M  += wmatch   * (.5 bi^2 + (bi > .6 ? biscale (bi - .6)^(1/3) : 0)),  biscale = .5/.4^(1/3)
     NM += wnomatch * (bi < .4 ? 1 : bi < .6 ? 1 - bi^2 : (1 - bi)^2)     (if the using value is not empty)
     NM += .3 wnomatch                     if exactly one of the two values is empty
   required() variables add wmatch to M. score = M / (M + NM).
4. Keep, for each master record, the candidate(s) with the highest score if score >= .6 (all ties).

bigram1(m, s): (number of the len(m)-1 bigrams of m that occur anywhere in s) * 2 / (len(s)+len(m)-2);
strings shorter than 3 characters score min(len)/(len(m)+len(s)-1) if one contains the other,
else 0; then a Winkler-type bonus of k(1-score)/10, k = number of i in 1..4 with
s[:i] == m[:i]; equal strings score 1, an empty using value 0.

reclink2 stores bi, M, NM and score as Stata floats (single precision), so every intermediate
result is rounded to float32 here too; otherwise ties and the .6 cutoffs come out differently.

Deliberately simple: pandas joins for blocking, sparse bigram matrices for the comparator.
Correct, readable, not optimized.
"""
import numpy as np
import pandas as pd
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


def bigram1(S, a, b):
    """Stata reclink's bigram1 score for value ids a (master) and b (using); float32-rounded."""
    lm, ls = S.len[a], S.len[b]
    count = np.concatenate([np.asarray(S.pos[a[k:k + 1_000_000]].multiply(S.has[b[k:k + 1_000_000]]).sum(axis=1))
                            .ravel() for k in range(0, len(a), 1_000_000)])
    g = np.zeros(len(a))
    long_m = lm > 2
    g[long_m] = f32(count[long_m] * 2 / (ls[long_m] + lm[long_m] - 2))
    short = (lm < 3) | (ls < 3)
    if short.any():
        idx = np.flatnonzero(short)
        cont = np.array([S.values[b[k]] in S.values[a[k]] or S.values[a[k]] in S.values[b[k]] for k in idx])
        g[idx] = np.where(cont, f32(np.minimum(lm[idx], ls[idx]) / (lm[idx] + ls[idx] - 1)), 0.0)
    # Winkler bonus: number of i in 1..4 with s[:i] == m[:i]
    eq = S.prefix[a] == S.prefix[b]
    cp = np.cumprod(eq & (S.prefix[a] > 0), axis=1).sum(axis=1)     # common prefix length, at most 4
    wink = np.zeros(len(a))
    for i in range(1, 5):
        wink += (np.minimum(i, ls) == np.minimum(i, lm)) & (np.minimum(i, ls) <= cp)
    g = f32(g + wink * (1 - g) / 10)
    g[a == b] = 1.0
    g[ls == 0] = 0.0
    return g


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


def checksums(i, j, s, n_perfect, truth):
    return dict(pairs=int(len(i)), matched_master=int(len(np.unique(i))), perfect=int(n_perfect),
                score_sum=float(s.sum()), id_master_sum=int((i + 1).sum()), id_using_sum=int((j + 1).sum()),
                true_links_found=int((truth[i] == j).sum()))
