"""Deterministic synthetic workloads for task 003 (record linkage).

Firm records with a name, street address, city, ZIP code and state. The using file is a register
of unique firms (e.g. a business register or an establishment database). The master file holds
records of some of those firms as they appear in another source (exact copies, or with a changed
name suffix, a typo, an abbreviated or missing address, a missing ZIP code...) plus firms that are
not in the register, so the true links are known. Strings are uppercase ASCII, as after the usual
cleaning step (e.g. stnd_compname / stnd_address).
"""
import numpy as np

# Numbered 1-5 from small to large. Each workload has a fixed seed.
WORKLOADS = [
    dict(id=1, seed=8001, design="sanity check", n_master=500, n_using=2_000, overlap=0.6),
    dict(id=2, seed=8002, design="survey employers to business register", n_master=2_000, n_using=20_000,
         overlap=0.6),
    dict(id=3, seed=8003, design="startups to state business registrations", n_master=10_000,
         n_using=100_000, overlap=0.5),
    dict(id=4, seed=8004, design="loan recipients to establishment database", n_master=50_000,
         n_using=500_000, overlap=0.7),
    dict(id=5, seed=8005, design="establishments to establishment database (national)", n_master=100_000,
         n_using=2_000_000, overlap=0.7),
]

# reclink2 name addr city zip state using ..., wmatch(10 8 5 5 5) wnomatch(2 3 4 4 4) required(state)
VARS = ["name", "addr", "city", "zip", "state"]
WMATCH = [10, 8, 5, 5, 5]
WNOMATCH = [2, 3, 4, 4, 4]
REQUIRED = ["state"]
MINSCORE = 0.6
MINBIGRAM = 0.6
FIRMS_PER_CITY = 300

STATES = ["CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI", "NJ", "VA", "WA", "AZ", "TN", "MA", "IN",
          "MD", "MO", "WI", "CO", "MN", "SC", "AL", "LA", "KY", "OR", "OK", "CT", "UT", "IA", "NV", "AR", "KS",
          "MS", "NM", "NE", "ID", "WV", "HI", "NH", "ME", "MT", "RI", "DE", "SD", "ND", "AK", "VT", "WY"]
SUFFIXES = ["INC", "CORP", "CORPORATION", "LLC", "LTD", "CO", "HOLDINGS", "GROUP", "TECHNOLOGIES", "SYSTEMS",
            "INTERNATIONAL", "PARTNERS", "LABS", "SOLUTIONS", "ENTERPRISES"]
STREET_TYPES = {"ST": "STREET", "AVE": "AVENUE", "RD": "ROAD", "BLVD": "BOULEVARD", "DR": "DRIVE", "LN": "LANE",
                "WAY": "WAY", "CT": "COURT", "PL": "PLACE", "PKWY": "PARKWAY"}
LETTERS = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))


def _words(rng, size, lo=3, hi=10):
    """Pronounceable pseudo-words of lo to hi-1 letters."""
    vowels, cons = np.array(list("AEIOU")), np.array(list("BCDFGHKLMNPRSTVWZ"))
    words = set()
    while len(words) < size:
        n = rng.integers(2, 6)
        words.add("".join(c + v for c, v in zip(rng.choice(cons, n), rng.choice(vowels, n)))[: rng.integers(lo, hi)])
    return np.array(sorted(words))


def _typo(rng, w):
    i = rng.integers(0, len(w))
    return w[:i] + rng.choice(LETTERS) + w[i + 1:]


def _noisy_name(rng, name):
    words = name.split()
    kind = rng.integers(0, 5)
    if kind == 0 and len(words) > 1:                       # different legal suffix
        words[-1] = rng.choice(SUFFIXES)
    elif kind == 1:                                        # typo
        words[0] = _typo(rng, words[0])
    elif kind == 2 and len(words) > 2:                     # drop a middle word
        del words[1]
    elif kind == 3:                                        # punctuation
        words[-1] = words[-1] + "."
        words[0] = words[0] + ","
    else:                                                  # swap two adjacent letters
        w = words[0]
        if len(w) > 3:
            i = rng.integers(0, len(w) - 1)
            words[0] = w[:i] + w[i + 1] + w[i] + w[i + 2:]
    return " ".join(words)


def _noisy_addr(rng, addr):
    num, *street, typ = addr.split()
    kind = rng.integers(0, 4)
    if kind == 0:                                          # street type spelled out
        typ = STREET_TYPES[typ]
    elif kind == 1:                                        # house-number typo
        i = rng.integers(0, len(num))
        num = num[:i] + str(rng.integers(0, 10)) + num[i + 1:]
    elif kind == 2:                                        # street type dropped
        typ = ""
    else:                                                  # suite added
        typ = f"{typ} STE {rng.integers(1, 500)}"
    return " ".join([num, *street, typ]).strip()


def generate(w):
    rng = np.random.default_rng(w["seed"])
    n_u = w["n_using"]
    # cities: lognormal sizes, each in one state (state weights roughly follow population)
    n_cities = max(5, n_u // FIRMS_PER_CITY)
    city_names = _words(rng, n_cities, 4, 11)
    city_state = rng.choice(len(STATES), n_cities, p=(w_ := 1 / np.arange(1, 51) ** 0.8) / w_.sum())
    city_size = rng.lognormal(0, 0.8, n_cities)
    n_zips = np.maximum(1, np.round(city_size * FIRMS_PER_CITY / 400)).astype(int)
    zip_start = 10000 + np.concatenate([[0], np.cumsum(n_zips)[:-1]]) * 7
    p_city = city_size / city_size.sum()

    name_vocab = _words(rng, max(2_000, n_u // 20))
    street_vocab = _words(rng, max(500, n_u // 200), 4, 10)

    def firms(c):
        """Random firms in cities c (vectorized)."""
        n = len(c)
        k = rng.integers(1, 4, n)
        w3 = rng.choice(name_vocab, (n, 3))
        sfx = rng.choice(SUFFIXES, n)
        names = [" ".join(list(w3[i, :k[i]]) + [sfx[i]]) for i in range(n)]
        nums, streets, types = rng.integers(1, 10000, n), rng.choice(street_vocab, n), rng.choice(list(STREET_TYPES), n)
        addrs = [f"{a} {b} {t}" for a, b, t in zip(nums, streets, types)]
        zips = [f"{z:05d}" for z in zip_start[c] + 7 * rng.integers(0, n_zips[c])]
        return np.column_stack([np.array(names, dtype=object), np.array(addrs, dtype=object),
                                city_names[c].astype(object), np.array(zips, dtype=object),
                                np.array(STATES, dtype=object)[city_state[c]], c.astype(object)])

    # using: unique firms
    cand = firms(rng.choice(n_cities, int(n_u * 1.02) + 10, p=p_city))
    _, first = np.unique(np.array(["|".join(r[:5]) for r in cand]), return_index=True)
    using = cand[np.sort(first)[:n_u]]
    assert len(using) == n_u
    miss = rng.random(n_u) < 0.03                          # a few register records lack a ZIP
    using[miss, 3] = ""

    # master: noisy copies of linked firms, plus firms not in the register
    n_linked = int(w["n_master"] * w["overlap"])
    src = rng.choice(n_u, n_linked, replace=False)
    master = []
    for i in src:
        name, addr, city, z, st, c = using[i]
        if rng.random() >= 0.25:                           # 25% are exact copies (perfect matches)
            if rng.random() < 0.7:
                name = _noisy_name(rng, name)
            u = rng.random()
            addr = "" if u < 0.08 else _noisy_addr(rng, addr) if u < 0.6 else addr
            if rng.random() < 0.05:
                city = _typo(rng, city)
            u = rng.random()
            z = "" if u < 0.10 else f"{zip_start[c] + 7 * rng.integers(0, n_zips[c]):05d}" if u < 0.15 else z
        master.append([name, addr, city, z, st])
    new = firms(rng.choice(n_cities, w["n_master"] - n_linked, p=p_city))[:, :5]
    new[rng.random(len(new)) < 0.05, 1] = ""
    master = np.vstack([np.array(master, dtype=object).reshape(-1, 5), new])
    perm = rng.permutation(w["n_master"])
    master = master[perm]
    truth = np.full(w["n_master"], -1)
    truth[np.argsort(perm)[:n_linked]] = src                # true using row of each linked master row
    return dict(master={v: master[:, j].astype(str) for j, v in enumerate(VARS)},
                using={v: using[:, j].astype(str) for j, v in enumerate(VARS)}, truth=truth)


def axes(w, d):
    return dict(n_master=w["n_master"], n_using=w["n_using"], true_links=int((d["truth"] >= 0).sum()),
                cities=max(5, w["n_using"] // FIRMS_PER_CITY))
