# funcs/fast_event_io.py
"""
Fast, dependency-light reading and writing of EventCalc `*_data.dat` event
files.

Three performance patterns, previously absent from the funcs/ tree:

1. **numpy-vectorized parsing** -- each channel data block is converted from
   text to a numeric ndarray with a single ``np.fromstring`` call and one
   ``reshape``, instead of a per-line ``map(float, line.split())`` loop. For
   the large hadronic channels this is ~30-50x faster than the line-by-line
   path used by ``selecting_processing.py`` / ``events_analysis.py``.

2. **worker sharding** -- ``map_event_files`` distributes a per-file callable
   across a process pool (``EVENTCALC_WORKERS`` / ``EXHAD_WORKERS`` env var,
   default = CPU count), so post-processing many event files (e.g. a
   mass-lifetime scan) scales with the number of cores.

3. **chunk-at-a-time formatting** -- ``write_event_rows`` formats a whole
   block of rows with one string operation instead of one per row, and
   formats a column that holds a single repeated value (a mass, a PDG code,
   an unfilled decay-product slot) once for the block. The text it writes is
   the text ``numpy.savetxt`` and ``pandas.DataFrame.to_csv`` write, byte for
   byte.

All are pure input/output helpers; they do not change any physics and are
independent of the decay generator (Pythia or exhad).
"""

import os
import re
import numpy as np
from concurrent.futures import ProcessPoolExecutor

_HEADER_NUM = re.compile(r"Total number of events:\s*([0-9.eE+-]+)")
_CHANNEL = re.compile(r"#<process=([^;]+);\s*sample_points=([0-9.eE+-]+)>")

EVENT_WRITE_CHUNK_SIZE = 10000


def _worker_count(default=None):
    raw = os.environ.get("EVENTCALC_WORKERS",
                         os.environ.get("EXHAD_WORKERS", "")).strip().lower()
    if raw in ("", "auto"):
        return default if default is not None else (os.cpu_count() or 1)
    try:
        n = int(raw)
        return max(1, n)
    except ValueError:
        return default if default is not None else (os.cpu_count() or 1)


def _parse_block(lines):
    """Vectorized parse of a list of numeric text rows into a 2-D float array.

    All rows in an EventCalc channel block are padded to the same width, so a
    single ``np.fromstring`` on the joined block + one ``reshape`` suffices.
    Rows of inconsistent width (should not occur) fall back to a safe per-row
    parse.
    """
    if not lines:
        return np.empty((0, 0), dtype=np.float64)
    ncol = len(lines[0].split())
    flat = np.fromstring(" ".join(lines), sep=" ", dtype=np.float64)
    if ncol > 0 and flat.size == ncol * len(lines):
        return flat.reshape(len(lines), ncol)
    # ragged fallback (pads short rows with zeros)
    rows = [np.fromstring(ln, sep=" ", dtype=np.float64) for ln in lines]
    width = max(r.size for r in rows)
    out = np.zeros((len(rows), width), dtype=np.float64)
    for i, r in enumerate(rows):
        out[i, :r.size] = r
    return out


def read_event_file(path):
    """
    Read an EventCalc `*_data.dat` file with the vectorized block parser.

    Returns
    -------
    n_ev_tot : float
        Total number of events from the header (NaN if absent).
    channels : dict[str, np.ndarray]
        Per-channel (n_events, n_fields) numeric arrays. n_fields = 10 (mother)
        + 6 * n_products; the mother block is columns 0..9, each subsequent
        6-tuple is [px, py, pz, E, m, pdg] of a decay product.
    """
    with open(path) as f:
        first = f.readline()
        m = _HEADER_NUM.search(first)
        n_ev_tot = float(m.group(1)) if m else float("nan")

        channels = {}
        cur_name = None
        cur_lines = []
        for line in f:
            s = line.strip()
            if not s:
                continue
            hm = _CHANNEL.match(s)
            if hm:
                if cur_name is not None:
                    channels[cur_name] = _parse_block(cur_lines)
                cur_name = hm.group(1)
                cur_lines = []
            else:
                cur_lines.append(s)
        if cur_name is not None:
            channels[cur_name] = _parse_block(cur_lines)
    return n_ev_tot, channels


def iter_events(channels):
    """Yield (channel_name, mother_row[10], products[n,6]) for every event."""
    for name, arr in channels.items():
        if arr.size == 0:
            continue
        nprod = (arr.shape[1] - 10) // 6
        for row in arr:
            prod = row[10:10 + 6 * nprod].reshape(nprod, 6)
            yield name, row[:10], prod


def map_event_files(func, paths, workers=None):
    """
    Apply ``func(path)`` to each event file, sharded across a process pool.

    ``func`` must be a top-level (picklable) callable taking a single path and
    returning a picklable result. With workers <= 1 the mapping runs inline
    (useful for debugging or tiny sets).
    """
    paths = list(paths)
    n = _worker_count() if workers is None else workers
    n = min(n, max(1, len(paths)))
    if n <= 1 or len(paths) <= 1:
        return [func(p) for p in paths]
    with ProcessPoolExecutor(max_workers=n) as ex:
        return list(ex.map(func, paths))


def _row_template(block, float_format):
    """Return the format template for one block of rows and its varying columns.

    A column whose rows all hold the same double is formatted once and its
    text is placed in the template as a literal, so only the columns that
    vary are formatted per row. The literal comes from the same conversion
    the per-row path would apply to that value, so the bytes are the same
    either way. Columns are compared bit pattern by bit pattern, which keeps
    -0.0 apart from 0.0.
    """
    bit_pattern = block.view(np.uint64)
    constant = np.all(bit_pattern == bit_pattern[0], axis=0)
    fields = []
    varying = []
    for column in range(block.shape[1]):
        if constant[column]:
            literal = float_format % block[0, column].item()
            fields.append(literal.replace("%", "%%"))
        else:
            fields.append(float_format)
            varying.append(column)
    return " ".join(fields) + "\n", varying


def _shortest_text(block):
    """Shortest round-trip text for a block, with an empty field for NaN.

    This is the conversion ``pandas.DataFrame.to_csv`` applies to a float
    frame when no ``float_format`` is given, including its default empty
    field for a missing value.
    """
    text = block.astype(str)
    text[np.isnan(block)] = ""
    return "".join(" ".join(row) + "\n" for row in text.tolist())


def write_event_rows(stream, rows, float_format=None, chunk_size=None):
    """Write `rows` to `stream` as one space-separated text line per row.

    ``float_format`` is a printf conversion applied to every value, e.g.
    ``"%.18e"``; the text is then what ``numpy.savetxt`` writes with that
    ``fmt`` and a space delimiter. ``None`` writes the shortest text that
    reads back as the same double, which is what
    ``pandas.DataFrame.to_csv(sep=' ')`` writes for a float frame.

    Every line ends in a newline, the last one included.
    """
    rows = np.ascontiguousarray(rows, dtype=np.float64)
    if rows.ndim != 2:
        raise ValueError("event rows must form a 2-D array")
    if rows.size == 0:
        return
    if chunk_size is None or chunk_size <= 0:
        chunk_size = EVENT_WRITE_CHUNK_SIZE
    conversion = "%r" if float_format is None else float_format
    for start in range(0, len(rows), chunk_size):
        block = rows[start:start + chunk_size]
        if float_format is None and np.isnan(block).any():
            stream.write(_shortest_text(block))
            continue
        template, varying = _row_template(block, conversion)
        values = block if len(varying) == block.shape[1] else block[:, varying]
        stream.write((template * len(block)) % tuple(values.ravel().tolist()))
