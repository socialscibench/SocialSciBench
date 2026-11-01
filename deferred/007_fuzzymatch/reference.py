"""Reference implementation for task 007_fuzzymatch.

Fuzzy name matching between two lists, following the defaults of Stata's

    matchit idm namem using using.dta, idusing(idu) txtusing(nameu)

(similmethod(bigram), score(jaccard), threshold(.5)): each name is split into overlapping
2-character pieces ("bigrams", counted with repetition, spaces and punctuation included,
case-sensitive); the score of a pair is

    sum_g c1(g) c2(g) / sqrt( sum_g c1(g)^2 * sum_g c2(g)^2 )

i.e. the cosine similarity of the two bigram-count vectors (matchit calls it "jaccard"), and
every pair scoring at least the threshold is returned. Names of 2 characters or fewer count
as a single piece, as in matchit.

Deliberately simple: sparse count matrices multiplied block by block. Correct, readable, not
optimized.
"""
import numpy as np
from scipy.sparse import csr_matrix


def bigrams(s):
    return [s[j:j + 2] for j in range(len(s) - 1)] if len(s) > 2 else [s]


def count_matrix(names, vocab):
    rows, cols = [], []
    for i, s in enumerate(names):
        for g in bigrams(s):
            rows.append(i)
            cols.append(vocab.setdefault(g, len(vocab)))
    return rows, cols


def run(master, using, threshold=0.5, block=None):
    """
    master  [n_master] str;  using [n_using] str;  threshold: keep pairs with score >= threshold
    returns dict(i [n_pairs] master row, j [n_pairs] using row, score [n_pairs])
    """
    vocab = {}
    rm, cm = count_matrix(master, vocab)
    ru, cu = count_matrix(using, vocab)
    V = len(vocab)
    A = csr_matrix((np.ones(len(rm)), (rm, cm)), shape=(len(master), V))   # duplicates are summed
    B = csr_matrix((np.ones(len(ru)), (ru, cu)), shape=(len(using), V))
    na = np.asarray(A.multiply(A).sum(1)).ravel()                          # sum of squared counts
    nb = np.asarray(B.multiply(B).sum(1)).ravel()
    BT = B.T.tocsc()
    block = block or max(1, int(5e7 // max(len(using), 1)))
    out_i, out_j, out_s = [], [], []
    for start in range(0, len(master), block):
        N = (A[start:start + block] @ BT).tocoo()                         # shared-bigram products
        score = N.data / np.sqrt(na[start + N.row] * nb[N.col])
        keep = score >= threshold
        out_i.append(start + N.row[keep])
        out_j.append(N.col[keep])
        out_s.append(score[keep])
    return dict(i=np.concatenate(out_i), j=np.concatenate(out_j), score=np.concatenate(out_s))


def checksums(res, truth=None):
    """Order-independent summaries: number of pairs, sums of scores and of 1-based ids, and
    (if true links are known) how many true links scored at or above the threshold."""
    out = dict(pairs=int(res["i"].size), score_sum=float(res["score"].sum()),
               id_master_sum=int((res["i"] + 1).sum()), id_using_sum=int((res["j"] + 1).sum()))
    if truth is not None:
        found = set(zip(res["i"].tolist(), res["j"].tolist()))
        links = [(int(m), j) for j, m in enumerate(truth) if m >= 0]
        out["true_links_found"] = sum((m, j) in found for m, j in links)
    return out
