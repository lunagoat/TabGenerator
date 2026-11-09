"""Beats and bar lines from beat_this's frame-wise activations.

beat_this outputs, 50 times a second, how likely a beat and a downbeat is.
Its own peak picking takes every peak above 50%, which leaves holes where
the band drops out (intros, breaks), doubles beats on busy fills and gives
bar lines that jump around. Here the same activations are decoded with two
small dynamic programs instead:

1. Beats: the most likely beat sequence under a tempo-continuity prior. The
   tempo may drift or change, but each change costs, so quiet stretches are
   bridged at the surrounding tempo and stray peaks are skipped.
2. Bars: every beat gets a position in its bar; the bar length may change
   (and odd bars may occur), but each change costs, so a few missing or
   extra downbeat peaks don't move the bar lines.
"""

from dataclasses import dataclass

import numpy as np

FPS = 50


@dataclass
class BeatParams:
    min_bpm: float = 40.0
    max_bpm: float = 240.0
    tempo_change: float = 100.0  # cost per squared log-ratio of successive beat lengths
    beat_floor: float = 2.0  # activations (logits) are clipped to +-this for beats
    skip_ceiling: float = 4.0  # ... and for frames passed over
    guard: int = 2  # frames around a beat that don't count as passed over
    # Bars
    meters: tuple = (3, 4, 5, 6, 7)
    meter_prior: tuple = (0.6, 0.0, 2.5, 2.5, 3.5)  # cost per bar of that length
    meter_change: float = 12.0  # switching to another bar length
    odd_bar: float = 9.0  # a bar cut short (a longer bar counts as two meter changes)
    down_clip: float = 3.0


def _log_sigmoid(x):
    return -np.logaddexp(0.0, -x)


def decode_beats(beat_logits: np.ndarray, p: BeatParams | None = None) -> np.ndarray:
    """Beat times in seconds."""
    p = p or BeatParams()
    b = np.asarray(beat_logits, dtype=np.float64)
    T = len(b)
    lmin = max(2, int(np.floor(FPS * 60.0 / p.max_bpm)))
    lmax = int(np.ceil(FPS * 60.0 / p.min_bpm))
    lags = np.arange(lmin, lmax + 1)
    nl = len(lags)
    if T < 2 * lmax:
        return _peaks(b)
    # A peak is rarely more than one frame wide, and the true beat may sit
    # between two frames: let a beat take the best of itself and its neighbours.
    pooled = np.maximum(b, np.maximum(np.r_[b[1:], -99], np.r_[-99, b[:-1]]) - 1.0)
    hit = _log_sigmoid(np.clip(pooled, -p.beat_floor, p.beat_floor))
    miss = _log_sigmoid(-np.clip(b, -p.skip_ceiling, p.skip_ceiling))
    cum = np.concatenate([[0.0], np.cumsum(miss)])  # cum[i] = sum(miss[:i])

    def skipped(a, c):
        """Cost of having no beat on frames a+guard+1 .. c-guard-1."""
        lo = np.minimum(a + p.guard + 1, c)
        hi = np.maximum(c - p.guard, lo)
        return cum[hi] - cum[lo]

    loglag = np.log(lags)
    pen = p.tempo_change * (loglag[:, None] - loglag[None, :]) ** 2  # [from, to]
    NEG = -1e18
    S = np.full((T, nl), NEG)
    M = np.full((T, nl), NEG)  # M[t, l] = max over l' of S[t, l'] - pen[l', l]
    back = np.zeros((T, nl), dtype=np.int16)
    first = np.zeros((T, nl), dtype=bool)
    for t in range(T):
        prev = t - lags
        ok = prev >= 0
        s = np.full(nl, NEG)
        if ok.any():
            pv = prev[ok]
            s[ok] = M[pv, np.nonzero(ok)[0]] + skipped(pv, t)
        if t < lmax:
            # First beat: nothing before it, any tempo.
            start = cum[max(0, t - p.guard)]
            first[t] = start > s
            s = np.maximum(s, start)
        S[t] = s + hit[t]
        cand = S[t][:, None] - pen
        back[t] = cand.argmax(axis=0)
        M[t] = cand.max(axis=0)
    # Last beat: close to the end.
    tail = np.arange(max(0, T - lmax), T)
    end_score = S[tail].max(axis=1) + (cum[T] - cum[np.minimum(tail + p.guard + 1, T)])
    t = int(tail[end_score.argmax()])
    l = int(S[t].argmax())
    frames = [t]
    while not first[t, l]:
        prev = t - lags[l]
        if prev < 0 or M[prev, l] <= NEG / 2:
            break
        l = int(back[prev, l])
        t = int(prev)
        frames.append(t)
    frames = np.array(frames[::-1])
    return _refine(b, frames) / FPS


def _refine(b: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """Sub-frame beat positions: move to the local peak (within a frame) and
    interpolate it with a parabola."""
    out = frames.astype(float)
    T = len(b)
    for i, f in enumerate(frames):
        lo, hi = max(1, f - 1), min(T - 2, f + 1)
        if hi < lo:
            continue
        c = lo + int(np.argmax(b[lo: hi + 1]))
        if b[c] < -2.0:
            continue  # a bridged beat, no real peak to refine
        y0, y1, y2 = b[c - 1], b[c], b[c + 1]
        denom = y0 - 2 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if denom < -1e-9 else 0.0
        out[i] = c + float(np.clip(delta, -0.5, 0.5))
    # Keep strictly increasing.
    for i in range(1, len(out)):
        if out[i] <= out[i - 1] + 1:
            out[i] = out[i - 1] + 1
    return out


def _peaks(b: np.ndarray) -> np.ndarray:
    """Fallback for very short audio: plain peak picking."""
    out = [i for i in range(1, len(b) - 1) if b[i] > 0 and b[i] >= b[i - 1] and b[i] > b[i + 1]]
    return np.array(out, dtype=float) / FPS


def decode_bars(
    beats: np.ndarray, downbeat_logits: np.ndarray, p: BeatParams | None = None
) -> np.ndarray:
    """Indexes (into `beats`) of the beats that start a bar."""
    p = p or BeatParams()
    n = len(beats)
    if n == 0:
        return np.zeros(0, dtype=int)
    d = np.asarray(downbeat_logits, dtype=np.float64)
    fr = np.clip(np.round(beats * FPS).astype(int), 0, len(d) - 1)
    act = np.maximum(d[fr], np.maximum(d[np.clip(fr - 1, 0, len(d) - 1)], d[np.clip(fr + 1, 0, len(d) - 1)]))
    act = np.clip(act, -p.down_clip, p.down_clip)
    is_down = _log_sigmoid(act)
    not_down = _log_sigmoid(-act)

    # States: (meter index, position in bar).
    states = [(mi, pos) for mi, m in enumerate(p.meters) for pos in range(m)]
    index = {s: i for i, s in enumerate(states)}
    ns = len(states)
    emit = np.zeros((n, ns))
    for i, (mi, pos) in enumerate(states):
        emit[:, i] = is_down if pos == 0 else not_down
    # Transitions into each state: list of (from state, cost).
    into: list[list[tuple[int, float]]] = [[] for _ in range(ns)]
    for (mi, pos), i in index.items():
        m = p.meters[mi]
        if pos > 0:
            into[i].append((index[(mi, pos - 1)], 0.0))
            continue
        # A new bar starts: after a full bar of this meter ...
        into[i].append((index[(mi, m - 1)], p.meter_prior[mi]))
        # ... after a full bar of another meter ...
        for mj, m2 in enumerate(p.meters):
            if mj != mi:
                into[i].append((index[(mj, m2 - 1)], p.meter_prior[mi] + p.meter_change))
        # ... or after this meter's bar was cut short / the bar line moved.
        for pos2 in range(1, m - 1):  # (never a one-beat bar)
            into[i].append((index[(mi, pos2)], p.meter_prior[mi] + p.odd_bar))
    score = emit[0].copy()
    back = np.zeros((n, ns), dtype=np.int32)
    for k in range(1, n):
        new = np.full(ns, -1e18)
        for i in range(ns):
            best, arg = -1e18, 0
            for j, c in into[i]:
                v = score[j] - c
                if v > best:
                    best, arg = v, j
            new[i] = best + emit[k, i]
            back[k, i] = arg
        score = new
    s = int(score.argmax())
    path = [s]
    for k in range(n - 1, 0, -1):
        s = int(back[k, s])
        path.append(s)
    path = path[::-1]
    return np.array([k for k, s in enumerate(path) if states[s][1] == 0], dtype=int)


def track(mono: np.ndarray, sr: int, device: str, p: BeatParams | None = None):
    """(beat times, indexes of the beats that start a bar) for a song."""
    from beat_this.inference import Audio2Frames

    # "final0" rather than "small0": the small model emits spurious beats
    # before the first downbeat (per MuScriptor's notes).
    model = Audio2Frames(checkpoint_path="final0", device=device)
    beat_logits, down_logits = model(mono.astype(np.float32), sr)
    beat_logits, down_logits = beat_logits.cpu().numpy(), down_logits.cpu().numpy()
    beats = decode_beats(beat_logits, p)
    return beats, decode_bars(beats, down_logits, p)
