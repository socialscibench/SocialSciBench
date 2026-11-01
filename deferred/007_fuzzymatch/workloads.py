"""Deterministic synthetic workloads for task 007_fuzzymatch.

Two lists of company names. The using list contains noisy versions of some master names
(typos, a different legal suffix, dropped or added words, punctuation) plus unrelated names,
as when linking patent assignees, startups or job-posting employers to a firm database.
Names are uppercase ASCII, as after the usual cleaning step.
"""
import numpy as np

# Numbered 1-5 from small to large. Each workload has a fixed seed.
WORKLOADS = [
    dict(id=1, seed=7001, design="sanity check", n_master=500, n_using=500, overlap=0.5),
    dict(id=2, seed=7002, design="firm database to firm database (e.g. Compustat to ExecuComp)",
         n_master=5_000, n_using=5_000, overlap=0.6),
    dict(id=3, seed=7003, design="patent assignees to public firms", n_master=50_000, n_using=10_000, overlap=0.3),
    dict(id=4, seed=7004, design="startups to startups (e.g. Crunchbase to PitchBook)", n_master=50_000,
         n_using=50_000, overlap=0.5),
    dict(id=5, seed=7005, design="job-posting employers to firm database", n_master=200_000, n_using=20_000,
         overlap=0.5),
]
THRESHOLD = 0.5

SUFFIXES = ["INC", "CORP", "CORPORATION", "LLC", "LTD", "CO", "HOLDINGS", "GROUP", "TECHNOLOGIES", "SYSTEMS",
            "INTERNATIONAL", "PARTNERS", "LABS", "SOLUTIONS", "ENTERPRISES"]
LETTERS = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))


def _vocabulary(rng, size):
    """Pronounceable pseudo-words, 3 to 9 letters."""
    vowels, cons = np.array(list("AEIOU")), np.array(list("BCDFGHKLMNPRSTVWZ"))
    words = set()
    while len(words) < size:
        n = rng.integers(2, 6)
        words.add("".join(c + v for c, v in zip(rng.choice(cons, n), rng.choice(vowels, n)))[: rng.integers(3, 10)])
    return np.array(sorted(words))


def _name(rng, vocab):
    words = list(rng.choice(vocab, rng.integers(1, 4)))
    return " ".join(words + [rng.choice(SUFFIXES)])


def _noisy(rng, name):
    """A plausible variant of a name in another database."""
    words = name.split()
    kind = rng.integers(0, 5)
    if kind == 0 and len(words) > 1:                       # different legal suffix
        words[-1] = rng.choice(SUFFIXES)
    elif kind == 1:                                        # typo: replace one letter
        i = rng.integers(0, len(words[0]))
        words[0] = words[0][:i] + rng.choice(LETTERS) + words[0][i + 1:]
    elif kind == 2 and len(words) > 2:                     # drop a middle word
        del words[1]
    elif kind == 3:                                        # punctuation / spacing
        words[-1] = words[-1] + "."
        words[0] = words[0] + ","
    else:                                                  # swap two adjacent letters
        w = words[0]
        if len(w) > 3:
            i = rng.integers(0, len(w) - 1)
            words[0] = w[:i] + w[i + 1] + w[i] + w[i + 2:]
    return " ".join(words)


def generate(w):
    rng = np.random.default_rng(w["seed"])
    vocab = _vocabulary(rng, max(2_000, w["n_master"] // 10))
    master = np.array([_name(rng, vocab) for _ in range(w["n_master"])])
    n_linked = int(w["n_using"] * w["overlap"])
    src = rng.choice(w["n_master"], n_linked, replace=False)
    using = np.array([_noisy(rng, master[i]) for i in src] +
                     [_name(rng, vocab) for _ in range(w["n_using"] - n_linked)])
    perm = rng.permutation(w["n_using"])
    using = using[perm]
    truth = np.full(w["n_using"], -1)
    truth[np.argsort(perm)[:n_linked]] = src                 # true master row of each linked using row
    return dict(master=master, using=using, threshold=THRESHOLD, truth=truth)


def axes(w, d):
    return dict(n_master=w["n_master"], n_using=w["n_using"], pairs=w["n_master"] * w["n_using"],
                true_links=int((d["truth"] >= 0).sum()), threshold=THRESHOLD)
