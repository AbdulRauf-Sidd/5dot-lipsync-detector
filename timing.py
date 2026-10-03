"""Lightweight per-stage wall-clock timing for profiling the pipeline.

Timings live in a plain dict of {stage_name: seconds}. Top-level stages are
meant to be non-overlapping so they sum to (roughly) the wall time; a name
containing "/" (e.g. "face_detection/forward") is a sub-stage that breaks down
its parent and is excluded from the sum.
"""

from __future__ import annotations

import time
from contextlib import contextmanager


@contextmanager
def timed(timings: dict | None, name: str):
    if timings is None:
        yield
        return
    # Register the key on entry so parents are ordered before their sub-stages.
    timings.setdefault(name, 0.0)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timings[name] += time.perf_counter() - t0


def merge_timings(dst: dict, src: dict) -> dict:
    for name, secs in src.items():
        dst[name] = dst.get(name, 0.0) + secs
    return dst


def format_timings(timings: dict, wall: float | None = None) -> str:
    accounted = sum(secs for name, secs in timings.items() if "/" not in name)
    base = wall if wall else accounted
    width = max([len(n) for n in timings] + [len("unaccounted")]) + 2

    lines = []
    for name, secs in timings.items():
        if "/" in name:
            label = "  " + name.split("/", 1)[1]
        else:
            label = name
        pct = 100.0 * secs / base if base else 0.0
        lines.append(f"  {label:<{width}} {secs:9.2f}s  {pct:5.1f}%")
    if wall:
        other = wall - accounted
        lines.append(f"  {'unaccounted':<{width}} {other:9.2f}s  {100.0 * other / wall:5.1f}%")
        lines.append(f"  {'TOTAL (wall)':<{width}} {wall:9.2f}s")
    return "\n".join(lines)
