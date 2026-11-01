"""Task 008 competitor: the 14 October 2026 study's complete C++17 pipeline, one thread.

linkage.cpp is the study's source (research/2026-10-14/native-extension/task008/linkage.cpp in the
companion repository, SHA-256 537a9294...), copied unchanged; this bridge follows the study's
native.py. It specializes the task's five ASCII fields, weights and required(state). The library
is compiled from this snapshot's linkage.cpp at import time, with the study's compiler flags,
into a cache in the environment (data-independent; import is excluded from timing). Input packing,
the native call and the copied outputs are inside the timed call. One thread, as the study's
one-thread configuration.
"""
import ctypes as c
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FLAGS = ["-std=c++17", "-O3", "-ffp-contract=off", "-fno-fast-math", "-pthread", "-dynamiclib"]
CACHE = Path(os.environ.get("SOCIALSCIBENCH_BUILD_CACHE", Path(sys.prefix) / "build-cache"))
VARS = ["name", "addr", "city", "zip", "state"]
WMATCH = [10, 8, 5, 5, 5]
WNOMATCH = [2, 3, 4, 4, 4]
REQUIRED = ["state"]


def _library():
    src = (HERE / "linkage.cpp").read_bytes()
    lib = CACHE / f"liblinkage-{hashlib.sha256(src + ' '.join(FLAGS).encode()).hexdigest()[:16]}.dylib"
    if not lib.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = lib.with_name(lib.name + f".{os.getpid()}.tmp")
        subprocess.run(["clang++", *FLAGS, str(HERE / "linkage.cpp"), "-o", str(tmp)], check=True)
        tmp.replace(lib)
    return c.CDLL(str(lib))


class Output(c.Structure):
    _fields_ = [(n, c.c_int64) for n in ("count", "perfect", "candidates")] + [
        ("i", c.POINTER(c.c_int64)), ("j", c.POINTER(c.c_int64)), ("score", c.POINTER(c.c_double))
    ] + [(n, c.c_double) for n in ("feature_s", "index_s", "search_s", "output_s")] + [("error", c.c_char_p)]


lib = _library()
lib.linkage_run.argtypes = [c.POINTER(c.c_char_p), c.POINTER(c.c_int64), c.c_int64, c.c_int64,
                            c.POINTER(c.c_double), c.POINTER(c.c_double), c.c_double, c.c_double, c.c_double,
                            c.c_int]
lib.linkage_run.restype = c.POINTER(Output)
lib.linkage_free.argtypes = [c.POINTER(Output)]


def run(master, using, vars, wmatch, wnomatch, required, minscore=0.6, minbigram=0.6):
    if (list(vars), list(wmatch), list(wnomatch), list(required)) != (VARS, WMATCH, WNOMATCH, REQUIRED):
        raise ValueError("this pipeline is specialized to the task's fields, weights and required(state)")
    columns = []
    for data in (master, using):
        for v in VARS:
            text = np.asarray(data[v], dtype=str)
            # This specialization has the same ASCII scope as the benchmark.
            if any("\0" in s for s in text):
                raise ValueError("Embedded NUL is unsupported")
            columns.append(np.ascontiguousarray(text.astype("S")))
    buffers = (c.c_char_p * 10)(*[c.cast(a.ctypes.data, c.c_char_p) for a in columns])
    widths = (c.c_int64 * 10)(*[a.dtype.itemsize for a in columns])
    wm = (c.c_double * 5)(*WMATCH)
    wnm = (c.c_double * 5)(*WNOMATCH)
    result = lib.linkage_run(buffers, widths, len(columns[0]), len(columns[5]), wm, wnm,
                             minscore, minbigram, .5 / (1 - minbigram) ** (1 / 3), 1)
    try:
        r = result.contents
        if r.error:
            raise RuntimeError(r.error.decode())
        out = tuple(np.ctypeslib.as_array(x, shape=(r.count,)).copy() if r.count else np.empty(0, dtype=d)
                    for x, d in ((r.i, np.int64), (r.j, np.int64), (r.score, np.float64)))
        return out + (int(r.perfect),)
    finally:
        lib.linkage_free(result)
