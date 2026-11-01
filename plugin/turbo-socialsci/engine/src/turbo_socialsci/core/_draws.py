"""Rademacher multipliers drawn inside the call (plugin addition; the benchmark's loader supplied them).

v[u, b] = +1 or -1 for unit u (its row in the caller's arrays) and draw b, from a counter-based
generator: bit b % 64 of SplitMix64's output for the counter u * n_words + b // 64 + 1 (n_words =
ceil(B / 64)), offset by a key derived from the seed. Each multiplier is therefore a function of
(seed, unit, draw) alone: any tiling of units and draws gives the same multipliers, and the speed
variant's Numba version (did_speed._rademacher_nb) gives exactly these. NumPy only, so the memory
variant keeps the JIT compiler out of its process. Bit order assumes a little-endian host.
"""
import numpy as np

GAMMA = 0x9E3779B97F4A7C15
M1 = 0xBF58476D1CE4E5B9
M2 = 0x94D049BB133111EB


def _mix(z):
    z = (z ^ (z >> np.uint64(30))) * np.uint64(M1)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(M2)
    return z ^ (z >> np.uint64(31))


def seed_key(seed):
    """64-bit key of a non-negative integer seed."""
    if int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    with np.errstate(over="ignore"):
        z = np.array([int(seed) % (1 << 64)], dtype=np.uint64) * np.uint64(GAMMA) + np.uint64(0x2545F4914F6CDD1D)
        return int(_mix(z)[0])


def n_words(B):
    return (int(B) + 63) // 64


def rademacher(key, units, nwords, b0, b1):
    """float64 multipliers [len(units), b1 - b0] of draws b0 .. b1 - 1 for the given units."""
    w0, w1 = b0 // 64, (b1 - 1) // 64 + 1
    u = np.asarray(units, dtype=np.uint64).reshape(-1, 1)
    w = np.arange(w0, w1, dtype=np.uint64).reshape(1, -1)
    with np.errstate(over="ignore"):
        z = _mix(np.uint64(key) + (u * np.uint64(nwords) + w + np.uint64(1)) * np.uint64(GAMMA))
    bits = np.unpackbits(np.ascontiguousarray(z).view(np.uint8), axis=1, bitorder="little")
    out = bits[:, b0 - w0 * 64:b1 - w0 * 64].astype(np.float64)
    out *= 2.0
    out -= 1.0
    return out


def rademacher_int8(key, n_units, B):
    """The whole [n_units, B] int8 matrix (tests: the task's boot_weights form of these multipliers)."""
    v = rademacher(key, np.arange(n_units), n_words(B), 0, B)
    return v.astype(np.int8)
