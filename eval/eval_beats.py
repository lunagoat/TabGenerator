#!/usr/bin/env python3
"""Beat / downbeat F-measure (70 ms) of the decoders on the beat test sets."""
import sys
import time

import env  # noqa
import numpy as np

import beat_data
from tabgen import beats as B


def minimal(b, d):
    """beat_this's own peak picking (what the pipeline used before)."""
    def peaks(x):
        out = []
        for i in range(len(x)):
            lo, hi = max(0, i - 3), min(len(x), i + 4)
            if x[i] > 0 and x[i] == x[lo:hi].max():
                out.append(i)
        ded = []
        for i in out:
            if ded and i - ded[-1] <= 1:
                continue
            ded.append(i)
        return np.array(ded) / 50
    bt, dt = peaks(b), peaks(d)
    if len(bt):
        dt = np.unique([bt[np.argmin(np.abs(bt - x))] for x in dt])
    return bt, dt


def dp(b, d, p=None):
    bt = B.decode_beats(b, p)
    return bt, bt[B.decode_bars(bt, d, p)]


def run(method, sets, verbose=False):
    res = {}
    for name, rows in sets.items():
        fb, fd = [], []
        for key, b, d, beats, downs in rows:
            bt, dt = method(b, d)
            fb.append(beat_data.fmeasure(bt, beats))
            fd.append(beat_data.fmeasure(dt, downs))
            if verbose:
                print(f"   {key:50s} beat {fb[-1]:.3f} down {fd[-1]:.3f}")
        res[name] = (float(np.mean(fb)), float(np.mean(fd)))
    return res


def irregular(bt):
    """Share of beat intervals more than 25% off the local median interval."""
    ibi = np.diff(bt)
    if len(ibi) < 9:
        return 0.0
    med = np.array([np.median(ibi[max(0, i - 4): i + 5]) for i in range(len(ibi))])
    return float(np.mean(np.abs(ibi / med - 1) > 0.25))


if __name__ == "__main__":
    sets = {"mdb": beat_data.mdb(), "babyslakh": beat_data.babyslakh(), "guitarset": beat_data.guitarset()}
    for label, m in (("beat_this peaks", minimal), ("dp", dp)):
        t = time.time()
        r = run(m, sets, "-v" in sys.argv)
        print(f"{label:16s}", "  ".join(f"{k}: beat {v[0]:.3f} down {v[1]:.3f}" for k, v in r.items()), f"({time.time() - t:.0f}s)")
    for key, b, d, _, _ in beat_data.user_songs():
        a, da = minimal(b, d)
        c, dc = dp(b, d)
        gaps = lambda x, dx: dict(zip(*[v.tolist() for v in np.unique(np.diff(np.searchsorted(x, dx)), return_counts=True)]))
        print(f"{key[5:35]:30s} peaks: {len(a)} beats, irregular {irregular(a):.3f}, bars {gaps(a, da)} | dp: {len(c)} beats, irregular {irregular(c):.3f}, bars {gaps(c, dc)}")
