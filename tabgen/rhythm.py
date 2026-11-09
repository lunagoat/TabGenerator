"""Beats, bars, and snapping note times onto a readable rhythmic grid.

Beat and downbeat times come from beat_this. Unlike a fixed-tempo grid, every
beat keeps its own tracked time, so songs that speed up or were played without
a click still line up. Each beat then gets the subdivision (8ths, 16ths,
triplets, 32nds...) that best explains the onsets inside it.
"""

from dataclasses import dataclass, field

import numpy as np

TICKS_PER_QUARTER = 960

# Per-beat subdivisions to choose from, with how "expensive" each one is to
# read. Simple time: 16ths are the everyday grid, triplets need evidence.
SIMPLE_SUBDIVISIONS = {1: 0.0, 2: 0.3, 4: 0.8, 8: 3.5, 3: 2.0, 6: 4.0}
# Compound time (6/8, 12/8): the beat is a dotted quarter split in three.
COMPOUND_SUBDIVISIONS = {1: 0.0, 3: 0.3, 6: 0.9, 12: 4.0, 2: 3.0}
ONSET_SIGMA_S = 0.025  # typical timing error of a transcribed onset
FAMILY_SWITCH = 3.0  # cost of going from straight to triplet subdivisions or back
CLUSTER_S = 0.035  # onsets closer than this are one rhythmic event (strums, flams)


def track_beats(mono: np.ndarray, sr: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Beat times (seconds) of the full mix, and the indexes of the beats
    that start a bar. See beats.py."""
    from . import beats

    times, bar_starts = beats.track(mono, sr, device)
    return np.asarray(times, dtype=float), np.asarray(bar_starts, dtype=int)


@dataclass
class Bar:
    first_beat: int  # index into BeatMap.beats
    n_beats: int


@dataclass
class BeatMap:
    """Beat times covering the whole song, grouped into bars."""

    beats: np.ndarray  # strictly increasing, seconds; extrapolated past the tracked span
    bars: list[Bar]
    compound: bool = False  # beats are dotted quarters split in three (6/8, 12/8)
    subdivisions: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))

    @property
    def beat_ticks(self) -> int:
        return TICKS_PER_QUARTER * 3 // 2 if self.compound else TICKS_PER_QUARTER

    def time_to_beat(self, t):
        return np.interp(t, self.beats, np.arange(len(self.beats)))

    def beat_to_time(self, b):
        return np.interp(b, np.arange(len(self.beats)), self.beats)

    def bar_tempo(self, bar: Bar) -> float:
        """Tempo of a bar in quarter notes per minute."""
        start = self.beats[bar.first_beat]
        end = self.beats[min(bar.first_beat + bar.n_beats, len(self.beats) - 1)]
        n = min(bar.n_beats, len(self.beats) - 1 - bar.first_beat)
        if n <= 0 or end <= start:
            return 120.0
        per_beat = (end - start) / n
        quarters_per_beat = 1.5 if self.compound else 1.0
        return 60.0 * quarters_per_beat / per_beat

    def time_signature(self, bar: Bar) -> tuple[int, int]:
        if self.compound:
            return bar.n_beats * 3, 8
        return bar.n_beats, 4


def _fill_edges(beats: np.ndarray, duration: float) -> tuple[np.ndarray, int]:
    """Extend the tracked beats to cover [0, duration] at the local tempo.

    Returns the extended beats and how many were added at the start.
    """
    if len(beats) < 2:
        period = 0.5
        beats = np.array([0.0, period]) if len(beats) == 0 else np.array([beats[0], beats[0] + period])
    head_period = float(np.median(np.diff(beats[:9])))
    tail_period = float(np.median(np.diff(beats[-9:])))
    head = []
    t = beats[0] - head_period
    while t > -head_period:  # keep one beat at or before 0
        head.append(t)
        t -= head_period
    tail = []
    t = beats[-1] + tail_period
    while t < duration + 2 * tail_period:
        tail.append(t)
        t += tail_period
    return np.concatenate([np.array(head[::-1]), beats, np.array(tail)]), len(head)


def smooth_beats(beats: np.ndarray, half_window: int = 4) -> np.ndarray:
    """Remove frame-level jitter (beat_this works at 50 fps) with a local
    straight-line fit around each beat; real tempo changes survive because
    the window is only a couple of bars wide. Beats that deviate strongly
    from their neighbours (tempo jumps, fermatas) are left alone."""
    n = len(beats)
    if n < 2 * half_window + 1:
        return beats
    out = beats.copy()
    idx = np.arange(n)
    for i in range(n):
        lo, hi = max(0, i - half_window), min(n, i + half_window + 1)
        x, y = idx[lo:hi], beats[lo:hi]
        slope, icpt = np.polyfit(x, y, 1)
        resid = y - (icpt + slope * x)
        if np.abs(resid).max() > 0.08 * slope:
            continue  # not locally steady: trust the tracker
        out[i] = icpt + slope * i
    return out


def build_beatmap(
    beats: np.ndarray,
    bar_starts: np.ndarray,
    duration: float,
    compound: bool = False,
) -> BeatMap:
    """`bar_starts` are indexes into `beats` of the beats that start a bar."""
    n_tracked = len(beats)
    beats = smooth_beats(np.asarray(beats, dtype=float))
    full, head = _fill_edges(beats, duration)
    starts = sorted({int(i) + head for i in bar_starts if 0 <= int(i) < n_tracked})
    meter = _dominant_meter(starts)
    # Pad beats before the song so the first bar line starts a bar and the
    # pickup is a full bar with leading rests.
    first = starts[0] if starts else 0
    pad = (-first) % meter
    if pad:
        period = full[1] - full[0]
        full = np.concatenate([full[0] - period * np.arange(pad, 0, -1), full])
        starts = [d + pad for d in starts]
        first += pad
    bars = [Bar(b, meter) for b in range(0, first, meter)]
    bars += [Bar(a, c - a) for a, c in zip(starts, starts[1:])]
    # After the last tracked bar line: the meter that was in force there.
    tail_meter = bars[-1].n_beats if len(starts) > 1 else meter
    b = starts[-1] if starts else 0
    while b < len(full) - 1:
        bars.append(Bar(b, tail_meter))
        b += tail_meter
    # Make the beats cover the last bar completely.
    extra = b + 1 - len(full)
    if extra > 0:
        period = full[-1] - full[-2]
        full = np.concatenate([full, full[-1] + period * np.arange(1, extra + 1)])
    return BeatMap(beats=full, bars=bars, compound=compound)


def _dominant_meter(starts: list[int]) -> int:
    if len(starts) < 2:
        return 4
    vals, counts = np.unique(np.diff(starts), return_counts=True)
    return int(vals[counts.argmax()])


# ----------------------------------------------------------------------------
# Quantization


@dataclass
class QNote:
    """A note snapped to the grid. Ticks count from the start of beat 0."""

    onset_tick: int
    end_tick: int
    pitch: int
    instrument: str
    is_drum: bool
    onset: float  # original times in seconds, kept for fingering
    offset: float


def _cluster(times: np.ndarray) -> np.ndarray:
    """Representative time per cluster of near-simultaneous onsets."""
    if not len(times):
        return times
    times = np.sort(times)
    out = [[times[0]]]
    for t in times[1:]:
        if t - out[-1][0] < CLUSTER_S:
            out[-1].append(t)
        else:
            out.append([t])
    return np.array([min(c) for c in out])


SHUFFLE_BIAS = {3: -1.9, 6: -2.5, 4: 1.0, 8: 1.0}


def choose_subdivisions(bm: BeatMap, onsets: np.ndarray, shuffle: bool | None = None) -> np.ndarray:
    """Pick a subdivision per beat that best explains the onsets in it.

    `shuffle`: whether triplets are the song's everyday grid (see
    `is_shuffle`); None decides from these onsets alone."""
    table = COMPOUND_SUBDIVISIONS if bm.compound else SIMPLE_SUBDIVISIONS
    n = len(bm.beats) - 1
    events = _cluster(np.asarray(onsets, dtype=float))
    pos = bm.time_to_beat(events)
    beat_len = np.diff(bm.beats)
    per_beat: list[list[float]] = [[] for _ in range(n)]
    for p in pos:
        i = int(np.floor(p))
        frac = p - i
        if 0 <= i < n:
            per_beat[i].append(frac)

    subs = list(table)
    triplet = np.array([(s in (3, 6)) and not bm.compound for s in subs])

    def costs_for(fracs, beat_len_i, bias):
        """Cost of each subdivision for one beat (lower is better)."""
        out = np.zeros(len(subs))
        if not fracs:
            return out
        f = np.array(fracs)
        for j, s in enumerate(subs):
            grid = np.round(f * s) / s
            err = (f - grid) * beat_len_i
            cost = float(np.sum((err / ONSET_SIGMA_S) ** 2)) + table[s] + bias.get(s, 0.0)
            # Two separate onsets on one grid point merge into one: penalise.
            slots = np.round(f * s).astype(int)
            cost += 6.0 * (len(slots) - len(set(slots.tolist())))
            out[j] = cost
        return out

    def decode(bias):
        """Best subdivision per beat. Straight and triplet beats rarely
        alternate, so changing between the two families costs extra: a
        lone triplet beat in a straight passage needs clear evidence."""
        cost = np.array([costs_for(per_beat[i], beat_len[i], bias) for i in range(n)])
        if not n:
            return np.ones(0, dtype=int)
        switch = FAMILY_SWITCH * (triplet[:, None] != triplet[None, :])
        # Beats without onsets carry the family along at no cost.
        empty = np.array([not per_beat[i] for i in range(n)])
        score = cost[0].copy()
        back = np.zeros((n, len(subs)), dtype=int)
        for i in range(1, n):
            tot = score[:, None] + switch
            back[i] = tot.argmin(axis=0)
            score = tot.min(axis=0) + cost[i]
        j = int(score.argmin())
        path = [j]
        for i in range(n - 1, 0, -1):
            j = int(back[i][j])
            path.append(j)
        path = path[::-1]
        return np.array([1 if empty[i] else subs[path[i]] for i in range(n)], dtype=int)

    if shuffle is None:
        chosen = decode({})
        shuffle = not bm.compound and _mostly_triplets(chosen, per_beat)
        if not shuffle:
            return chosen
    return decode(SHUFFLE_BIAS if shuffle and not bm.compound else {})


def _mostly_triplets(chosen, per_beat) -> bool:
    busy = [chosen[i] for i in range(len(chosen)) if len(per_beat[i]) >= 2]
    return bool(busy) and sum(1 for s in busy if s in (3, 6)) / len(busy) > 0.5


def is_shuffle(bm: BeatMap, onsets: np.ndarray) -> bool:
    """Whether most busy beats are triplet-based (shuffle, swing): triplets
    are then the everyday grid and straight 16ths need the evidence."""
    if bm.compound:
        return False
    chosen = choose_subdivisions(bm, onsets, shuffle=False)
    pos = bm.time_to_beat(_cluster(np.asarray(onsets, dtype=float)))
    counts = np.bincount(np.floor(pos).astype(int).clip(0, len(chosen) - 1), minlength=len(chosen))
    busy = chosen[counts >= 2]
    return len(busy) > 0 and float(np.mean(np.isin(busy, (3, 6)))) > 0.5


def detect_compound(bm: BeatMap, onsets: np.ndarray) -> bool:
    """Whether beats divide in three throughout (6/8, 12/8 feel)."""
    events = _cluster(np.asarray(onsets, dtype=float))
    pos = bm.time_to_beat(events)
    frac = pos - np.floor(pos)
    off = frac[(frac > 0.08) & (frac < 0.92)]
    if len(off) < 30:
        return False
    # Share of off-beat onsets near 1/3 or 2/3 vs near 1/4, 1/2, 3/4.
    d3 = np.minimum(np.abs(off - 1 / 3), np.abs(off - 2 / 3))
    d2 = np.min(np.abs(off[:, None] - np.array([0.25, 0.5, 0.75])[None, :]), axis=1)
    return float(np.mean(d3 < d2)) > 0.75


def quantize(bm: BeatMap, notes, subdivisions: np.ndarray) -> list[QNote]:
    """Snap note on/off times to the per-beat grid."""
    bt = bm.beat_ticks
    n = len(subdivisions)

    def snap(t: float) -> tuple[int, int]:
        """(tick, subdivision step in ticks at that point)."""
        p = float(bm.time_to_beat(t))
        i = int(np.clip(np.floor(p), 0, n - 1))
        s = int(subdivisions[i])
        k = round((p - i) * s)
        return i * bt + k * bt // s, bt // s

    out = []
    for nt in notes:
        on, step = snap(nt.onset)
        if nt.is_drum:
            end = on + step
        else:
            end, _ = snap(nt.offset)
            end = max(end, on + step)
        out.append(QNote(on, end, nt.pitch, nt.instrument, nt.is_drum, nt.onset, nt.offset))
    # Same pitch twice on one tick (in one part) is one note.
    seen = {}
    for q in out:
        key = (q.onset_tick, q.pitch, q.instrument, q.is_drum)
        if key not in seen or q.end_tick > seen[key].end_tick:
            seen[key] = q
    return sorted(seen.values(), key=lambda q: (q.onset_tick, q.pitch))
