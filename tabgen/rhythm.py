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
CLUSTER_S = 0.035  # onsets closer than this are one rhythmic event (strums, flams)


def track_beats(mono: np.ndarray, sr: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Beat and downbeat times (seconds) of the full mix."""
    from beat_this.inference import Audio2Beats

    # "final0" rather than "small0": the small model emits spurious beats
    # before the first downbeat (per MuScriptor's notes).
    tracker = Audio2Beats(checkpoint_path="final0", device=device, dbn=False)
    # Returns (beats, downbeats) despite beat_this's own naming.
    beats, downbeats = tracker(mono.astype(np.float32), sr)
    return np.asarray(beats, dtype=float), np.asarray(downbeats, dtype=float)


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
    downbeats: np.ndarray,
    duration: float,
    compound: bool = False,
) -> BeatMap:
    beats = smooth_beats(np.unique(np.round(beats, 4)))
    full, _ = _fill_edges(beats, duration)
    down_idx = sorted({int(np.argmin(np.abs(full - d))) for d in downbeats})
    meter = _dominant_meter(down_idx)
    # Pad beats before the song so the first downbeat starts a bar and the
    # pickup bar is a full bar with leading rests.
    first_down = down_idx[0] if down_idx else 0
    pad = (-first_down) % meter
    if pad:
        period = full[1] - full[0]
        full = np.concatenate([full[0] - period * np.arange(pad, 0, -1), full])
        down_idx = [d + pad for d in down_idx]
    bars = _lay_bars(down_idx, meter, len(full))
    return BeatMap(beats=full, bars=bars, compound=compound)


def _dominant_meter(down_idx: list[int]) -> int:
    if len(down_idx) < 2:
        return 4
    gaps = np.diff(down_idx)
    gaps = gaps[(gaps >= 2) & (gaps <= 12)]
    if not len(gaps):
        return 4
    vals, counts = np.unique(gaps, return_counts=True)
    return int(vals[counts.argmax()])


def _lay_bars(down_idx: list[int], meter: int, n_beats: int) -> list[Bar]:
    """Bars of `meter` beats from beat 0, re-anchored when the tracked
    downbeats consistently say the bar line moved (an odd bar, a dropped
    beat...). A lone stray downbeat is ignored."""
    downs = set(down_idx)
    bars: list[Bar] = []
    b = 0
    while b < n_beats - 1:
        nxt = b + meter
        if b >= (down_idx[0] if down_idx else 0) and nxt not in downs:
            best = None
            for j in range(b + 2, b + 2 * meter):
                if j == nxt or j not in downs:
                    continue
                support = sum((j + k * meter) in downs for k in range(1, 4))
                if support >= 3 and (best is None or abs(j - nxt) < abs(best - nxt)):
                    best = j
            if best is not None:
                nxt = best
        bars.append(Bar(b, nxt - b))
        b = nxt
    return bars


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


def choose_subdivisions(bm: BeatMap, onsets: np.ndarray) -> np.ndarray:
    """Pick a subdivision per beat that best explains the onsets in it."""
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

    def best_for(fracs, beat_len_i, bias):
        if not fracs:
            return 1
        f = np.array(fracs)
        best = None
        for s, cost_s in table.items():
            grid = np.round(f * s) / s
            err = (f - grid) * beat_len_i
            cost = float(np.sum((err / ONSET_SIGMA_S) ** 2)) + cost_s + bias.get(s, 0.0)
            # Two separate onsets on one grid point merge into one: penalise.
            slots = np.round(f * s).astype(int)
            cost += 6.0 * (len(slots) - len(set(slots.tolist())))
            if best is None or cost < best[0]:
                best = (cost, s)
        return best[1]

    chosen = np.ones(n, dtype=int)
    bias: dict[int, float] = {}
    for rnd in range(2):
        for i in range(n):
            chosen[i] = best_for(per_beat[i], beat_len[i], bias)
        if rnd == 0 and not bm.compound:
            # If most busy beats are triplet-based (shuffle), triplets become
            # the everyday grid and straight 16ths need the evidence instead.
            busy = [chosen[i] for i in range(n) if len(per_beat[i]) >= 2]
            if busy:
                trip = sum(1 for s in busy if s in (3, 6)) / len(busy)
                if trip > 0.5:
                    bias = {3: -1.9, 6: -2.5, 4: 1.0, 8: 1.0}
                else:
                    break
    return chosen


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
