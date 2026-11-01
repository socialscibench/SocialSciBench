"""Minimal streaming reader and writer for Stata .dta files (format 118, little-endian), shared by the
reference, the generator, the DuckDB pipeline and the output checker.

Reading never loads more than one chunk of rows; writing streams rows and fills in N and the section
map when the file is closed. Supported: numeric types, str# (no strL), value-label tables.
"""
import struct

import numpy as np

STR_MAX = 2045
STRL, DOUBLE, FLOAT, LONG, INT, BYTE = 32768, 65526, 65527, 65528, 65529, 65530
NUMERIC = {BYTE: "i1", INT: "<i2", LONG: "<i4", FLOAT: "<f4", DOUBLE: "<f8"}
INT_MAX = {BYTE: 100, INT: 32740, LONG: 2147483620}             # largest non-missing value; . = max + 1
FLOAT_MAX = np.float32(1.7014117e38)
DOUBLE_MAX = 8.988465674311579e307
TYPE_NAME = {BYTE: "byte", INT: "int", LONG: "long", FLOAT: "float", DOUBLE: "double"}


def type_name(t):
    return f"str{t}" if t <= STR_MAX else TYPE_NAME[t]


def _expect(f, tag):
    got = f.read(len(tag))
    if got != tag:
        raise ValueError(f"expected {tag!r}, found {got!r}")


def _cstr(b):
    return b.split(b"\0", 1)[0].decode("utf-8")


def read_header(path):
    """Variables, types, metadata, N, the data offset and the value-label tables of a .dta file."""
    with open(path, "rb") as f:
        _expect(f, b"<stata_dta><header><release>")
        release = f.read(3)
        if release != b"118":
            raise ValueError(f"{path}: format {release!r}; only 118 is supported")
        _expect(f, b"</release><byteorder>")
        if f.read(3) != b"LSF":
            raise ValueError("only little-endian files are supported")
        _expect(f, b"</byteorder><K>")
        K = struct.unpack("<H", f.read(2))[0]
        _expect(f, b"</K><N>")
        N = struct.unpack("<Q", f.read(8))[0]
        _expect(f, b"</N><label>")
        label = f.read(struct.unpack("<H", f.read(2))[0])
        _expect(f, b"</label><timestamp>")
        f.read(struct.unpack("<B", f.read(1))[0])
        _expect(f, b"</timestamp></header><map>")
        off = struct.unpack("<14Q", f.read(14 * 8))
        f.seek(off[2] + len(b"<variable_types>"))
        types = list(struct.unpack(f"<{K}H", f.read(2 * K)))
        f.seek(off[3] + len(b"<varnames>"))
        names = [_cstr(f.read(129)) for _ in range(K)]
        f.seek(off[5] + len(b"<formats>"))
        formats = [_cstr(f.read(57)) for _ in range(K)]
        f.seek(off[6] + len(b"<value_label_names>"))
        vlnames = [_cstr(f.read(129)) for _ in range(K)]
        f.seek(off[7] + len(b"<variable_labels>"))
        varlabels = [_cstr(f.read(321)) for _ in range(K)]
        f.seek(off[8])
        characteristics = f.read(off[9] - off[8])
        f.seek(off[10])
        strls = f.read(off[11] - off[10])
        f.seek(off[11])
        value_labels = _parse_value_labels(f.read(off[12] - off[11]))
    if STRL in types or strls != b"<strls></strls>":
        raise ValueError("strL variables are not supported")
    return dict(path=str(path), K=K, N=N, label=label, types=types, names=names, formats=formats, vlnames=vlnames,
                varlabels=varlabels, characteristics=characteristics, value_labels=value_labels,
                data_offset=off[9] + len(b"<data>"), dtype=row_dtype(types, names))


def _parse_value_labels(b):
    """{label name: {code: text}} from the '<value_labels>...</value_labels>' section."""
    out, pos = {}, len(b"<value_labels>")
    while b[pos:pos + 5] == b"<lbl>":
        n_bytes = struct.unpack("<i", b[pos + 5:pos + 9])[0]
        name = _cstr(b[pos + 9:pos + 9 + 129])
        t = pos + 9 + 129 + 3
        n, txtlen = struct.unpack("<ii", b[t:t + 8])
        offs = np.frombuffer(b[t + 8:t + 8 + 4 * n], "<i4")
        vals = np.frombuffer(b[t + 8 + 4 * n:t + 8 + 8 * n], "<i4")
        txt = b[t + 8 + 8 * n:t + 8 + 8 * n + txtlen]
        out[name] = {int(v): _cstr(txt[o:]) for v, o in zip(vals, offs)}
        pos = t + n_bytes + len(b"</lbl>")
    return out


def row_dtype(types, names):
    return np.dtype([(n, f"S{t}" if t <= STR_MAX else NUMERIC[t]) for n, t in zip(names, types)])


def iter_rows(path, chunk_rows=500_000, nrows=None, header=None):
    """Yield the rows of a .dta file as structured arrays of at most chunk_rows rows."""
    h = header or read_header(path)
    n = h["N"] if nrows is None else min(nrows, h["N"])
    dt = h["dtype"]
    with open(path, "rb") as f:
        f.seek(h["data_offset"])
        for start in range(0, n, chunk_rows):
            m = min(chunk_rows, n - start)
            yield np.frombuffer(f.read(m * dt.itemsize), dtype=dt)


def missing_index(x, t):
    """-1 where not missing, else 0 for ., 1 for .a, ..., 26 for .z (numeric types)."""
    if t in INT_MAX:
        v = x.astype(np.int64)
        return np.where(v > INT_MAX[t], v - INT_MAX[t] - 1, -1)
    if t == FLOAT:
        bits = x.view(np.uint32).astype(np.int64)
        return np.where(x > FLOAT_MAX, (bits - 0x7F000000) >> 11, -1)
    bits = x.view(np.uint64)
    return np.where(x > DOUBLE_MAX, ((bits - np.uint64(0x7FE0000000000000)) >> np.uint64(40)).astype(np.int64), -1)


def str_len(x):
    """Length of each str# value up to its first NUL byte. Bytes after the NUL are not part of the
    value: Stata's own saved files can contain leftover bytes there."""
    w = x.dtype.itemsize
    u = np.ascontiguousarray(x).view(np.uint8).reshape(len(x), w)
    zero = u == 0
    return np.where(zero.any(axis=1), zero.argmax(axis=1), w)


def is_missing(x, t):
    if t <= STR_MAX:
        return str_len(x) == 0
    if t in INT_MAX:
        return x > INT_MAX[t]
    return x > (FLOAT_MAX if t == FLOAT else DOUBLE_MAX)


def missing_value(t, k=0):
    """Stored value of missing code k (0 = ., 1 = .a, ...) in numeric type t."""
    if t in INT_MAX:
        return INT_MAX[t] + 1 + k
    if t == FLOAT:
        return np.array(0x7F000000 + (k << 11), dtype=np.uint32).view(np.float32)
    return np.array(0x7FE0000000000000 + (k << 40), dtype=np.uint64).view(np.float64)


def promote(a, b):
    """Stata's storage-type promotion for append."""
    if (a <= STR_MAX) != (b <= STR_MAX):
        raise TypeError("variable is string in one file and numeric in another")
    if a <= STR_MAX:
        return max(a, b)
    if a == b:
        return a
    rank = {BYTE: 0, INT: 1, LONG: 2, FLOAT: 3, DOUBLE: 4}
    lo, hi = sorted((a, b), key=rank.get)
    return DOUBLE if hi == DOUBLE or (lo == LONG and hi == FLOAT) else hi


def convert(x, t_from, t_to):
    """Convert a column between Stata storage types, keeping missing codes."""
    if t_from == t_to:
        return x
    if t_from <= STR_MAX:
        return x.astype(f"S{t_to}")
    k = missing_index(x, t_from)
    y = x.astype(NUMERIC[t_to])
    for code in np.unique(k[k >= 0]):
        y[k == code] = missing_value(t_to, int(code))
    return y


class Writer:
    """Stream rows into a new .dta file.

    variables: list of dicts with name, type, and optionally format, varlabel, vlname.
    value_labels: {label name: {code: text}}, written when the file is closed.
    """

    def __init__(self, path, variables, label=b"", characteristics=b"<characteristics></characteristics>"):
        self.vars = variables
        self.dtype = row_dtype([v["type"] for v in variables], [v["name"] for v in variables])
        self.f = open(path, "wb")
        self.n = 0
        f, K = self.f, len(variables)
        self.off = [0] * 14
        f.write(b"<stata_dta><header><release>118</release><byteorder>LSF</byteorder>")
        f.write(b"<K>" + struct.pack("<H", K) + b"</K><N>")
        self.n_pos = f.tell()
        f.write(struct.pack("<Q", 0) + b"</N><label>" + struct.pack("<H", len(label)) + label + b"</label>")
        ts = b"01 Jan 2026 00:00"                                     # fixed, so outputs are reproducible
        f.write(b"<timestamp>" + struct.pack("<B", len(ts)) + ts + b"</timestamp></header>")
        self.off[1] = f.tell()
        f.write(b"<map>" + b"\0" * 112 + b"</map>")

        def fixed(s, n):
            b = s.encode("utf-8")[:n - 1]
            return b + b"\0" * (n - len(b))

        def default_format(t):
            return f"%{min(t, 2045)}s" if t <= STR_MAX else {BYTE: "%8.0g", INT: "%8.0g", LONG: "%12.0g",
                                                              FLOAT: "%9.0g", DOUBLE: "%10.0g"}[t]

        sections = [
            ("variable_types", struct.pack(f"<{K}H", *[v["type"] for v in variables])),
            ("varnames", b"".join(fixed(v["name"], 129) for v in variables)),
            ("sortlist", b"\0\0" * (K + 1)),
            ("formats", b"".join(fixed(v.get("format") or default_format(v["type"]), 57) for v in variables)),
            ("value_label_names", b"".join(fixed(v.get("vlname", ""), 129) for v in variables)),
            ("variable_labels", b"".join(fixed(v.get("varlabel", ""), 321) for v in variables)),
        ]
        for i, (tag, body) in enumerate(sections, start=2):
            self.off[i] = f.tell()
            f.write(f"<{tag}>".encode() + body + f"</{tag}>".encode())
        self.off[8] = f.tell()
        f.write(characteristics)
        self.off[9] = f.tell()
        f.write(b"<data>")

    def write(self, rows):
        """rows: a structured array with this file's dtype, or a dict of columns."""
        if not isinstance(rows, np.ndarray) or rows.dtype != self.dtype:
            cols = rows
            m = len(next(iter(cols.values()))) if isinstance(cols, dict) else len(cols)
            rows = np.empty(m, dtype=self.dtype)
            for v in self.vars:
                rows[v["name"]] = cols[v["name"]]
        self.f.write(rows.tobytes())
        self.n += len(rows)

    def close(self, value_labels=None):
        f = self.f
        f.write(b"</data>")
        self.off[10] = f.tell()
        f.write(b"<strls></strls>")
        self.off[11] = f.tell()
        f.write(b"<value_labels>")
        for name, table in (value_labels or {}).items():
            codes = sorted(table)
            txt, offs = b"", []
            for c in codes:
                offs.append(len(txt))
                txt += table[c].encode("utf-8") + b"\0"
            body = struct.pack("<ii", len(codes), len(txt)) + np.array(offs, "<i4").tobytes() + \
                np.array(codes, "<i4").tobytes() + txt
            f.write(b"<lbl>" + struct.pack("<i", len(body)) + name.encode().ljust(129, b"\0") + b"\0" * 3 + body
                    + b"</lbl>")
        f.write(b"</value_labels>")
        self.off[12] = f.tell()
        f.write(b"</stata_dta>")
        self.off[13] = f.tell()
        f.seek(self.n_pos)
        f.write(struct.pack("<Q", self.n))
        f.seek(self.off[1] + len(b"<map>"))
        f.write(struct.pack("<14Q", *self.off))
        f.close()
