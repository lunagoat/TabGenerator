"""Tunings, and turning pitches into playable string/fret positions.

Every note has several places it can be played on the neck. We pick one per
note by running Viterbi over the song: each chord gets a set of candidate
fingerings with a cost for how awkward the shape is, and moving between
consecutive shapes costs according to how far the fretting hand travels.
"""

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def note_name(pitch: int) -> str:
    return NOTE_NAMES[pitch % 12]


@dataclass(frozen=True)
class Tuning:
    name: str
    strings: tuple[int, ...]  # MIDI pitch of each open string, lowest string first
    prior: float = 0.0  # how unlikely this tuning is a priori (added per song)

    @property
    def lowest(self) -> int:
        return self.strings[0]

    def labels(self) -> list[str]:
        """Names for the string lines, lowest string first (top string lowercase if E)."""
        names = [note_name(p) for p in self.strings]
        if names[-1] == "E":
            names[-1] = "e"
        return names


GUITAR_TUNINGS = [
    Tuning("E Standard", (40, 45, 50, 55, 59, 64), 0.0),
    Tuning("Eb Standard", (39, 44, 49, 54, 58, 63), 0.3),
    Tuning("Drop D", (38, 45, 50, 55, 59, 64), 0.2),
    Tuning("D Standard", (38, 43, 48, 53, 57, 62), 0.5),
    Tuning("Drop C#", (37, 44, 49, 54, 58, 63), 0.5),
    Tuning("C# Standard", (37, 42, 47, 52, 56, 61), 0.8),
    Tuning("Drop C", (36, 43, 48, 53, 57, 62), 0.5),
    Tuning("C Standard", (36, 41, 46, 51, 55, 60), 0.8),
    Tuning("Drop B", (35, 42, 47, 52, 56, 61), 0.8),
    Tuning("B Standard", (35, 40, 45, 50, 55, 59), 1.0),
    Tuning("7-String B Standard", (35, 40, 45, 50, 55, 59, 64), 0.6),
    Tuning("Drop A", (33, 40, 45, 50, 55, 59), 1.2),
    Tuning("7-String Drop A", (33, 40, 45, 50, 55, 59, 64), 1.0),
]

BASS_TUNINGS = [
    Tuning("E Standard", (28, 33, 38, 43), 0.0),
    Tuning("Eb Standard", (27, 32, 37, 42), 0.3),
    Tuning("Drop D", (26, 33, 38, 43), 0.2),
    Tuning("D Standard", (26, 31, 36, 41), 0.5),
    Tuning("Drop C#", (25, 32, 37, 42), 0.5),
    Tuning("Drop C", (24, 31, 36, 41), 0.5),
    Tuning("C Standard", (24, 29, 34, 39), 0.8),
    Tuning("5-String B Standard", (23, 28, 33, 38, 43), 0.5),
    Tuning("Drop B", (23, 30, 35, 40), 1.0),
    Tuning("5-String Drop A", (21, 28, 33, 38, 43), 1.0),
]


def _tuning_key(name: str) -> str:
    """Spelling-insensitive key: "Drop D", "drop_d", "DropD", "drop-d" all match."""
    key = name.lower()
    for ch in " -_":
        key = key.replace(ch, "")
    return key.replace("standard", "std")


def find_tuning(name: str, bass: bool) -> Tuning:
    table = BASS_TUNINGS if bass else GUITAR_TUNINGS
    for t in table:
        if _tuning_key(t.name) == _tuning_key(name):
            return t
    choices = ", ".join(t.name for t in table)
    raise ValueError(f"Unknown {'bass' if bass else 'guitar'} tuning {name!r}. Choose from: {choices}")


@dataclass
class FingeringWeights:
    """Costs used by the fingering optimizer, tuned on GuitarSet."""

    # Accuracy on GuitarSet (string chosen by the real player): 73%, vs
    # 54.5% for "lowest fret" and 68.9% without hand-position tracking.
    span: float = 0.0  # per fret of stretch beyond a comfortable 3-fret span
    span_hard: int = 5  # anything wider is not considered
    stretch: float = 1.0  # note one fret past the 4-fret box under the hand
    height: float = 0.02  # per fret of hand position (0.01 scores 75% on GuitarSet
    # but its jazz solos sit high on the neck; rock tabs favour low positions)
    open_bonus: float = 0.0  # open strings need no finger
    open_high_mix: float = 0.0  # open string mixed with a fretted note at fret >= 7
    string_gap: float = 1.2  # per unused string inside a 3+ note chord
    move: float = 1.5  # per fret the hand moves
    shift: float = 0.0  # any hand movement at all
    ease_time: float = 0.35  # seconds of gap that halve the cost of moving
    string_jump: float = 0.36  # per string crossed between single notes
    ringing_clash: float = 3.0  # new note on a string that is still ringing
    top_k: int = 16


@dataclass
class Event:
    """Notes starting together. `pitches` sorted ascending."""

    time: float
    pitches: list[int]
    ends: list[float] = field(default_factory=list)  # note-off time per pitch


@dataclass
class Fingering:
    strings: tuple[int, ...]  # string index per kept pitch (0 = lowest string)
    frets: tuple[int, ...]
    pitches: tuple[int, ...]
    dropped: tuple[int, ...] = ()  # pitches that could not be placed


def _chord_candidates(pitches, tuning: Tuning, max_fret: int, w: FingeringWeights):
    """All ways to put these pitches on distinct strings, cheapest first.

    Returns a list of (static_cost, strings, frets).
    """
    n_strings = len(tuning.strings)
    options = []
    for p in pitches:
        opts = [(s, p - o) for s, o in enumerate(tuning.strings) if 0 <= p - o <= max_fret]
        options.append(opts)
    results = []

    def rec(i, used, strings, frets, lo, hi):
        if i == len(pitches):
            results.append((tuple(strings), tuple(frets)))
            return
        for s, f in options[i]:
            if used & (1 << s):
                continue
            nlo, nhi = lo, hi
            if f > 0:
                nlo, nhi = min(lo, f), max(hi, f)
                if nhi - nlo > w.span_hard:
                    continue
            # Keep pitch order and string order consistent: a higher pitch
            # on a lower string is possible but rare in chords.
            strings.append(s)
            frets.append(f)
            rec(i + 1, used | (1 << s), strings, frets, nlo, nhi)
            strings.pop()
            frets.pop()

    rec(0, 0, [], [], 99, -1)
    scored = [(_static_cost(s, f, n_strings, w), s, f) for s, f in results]
    scored.sort(key=lambda x: x[0])
    return scored[: w.top_k]


def _static_cost(strings, frets, n_strings, w: FingeringWeights) -> float:
    fretted = [f for f in frets if f > 0]
    cost = 0.0
    if fretted:
        lo, hi = min(fretted), max(fretted)
        span = hi - lo
        # Frets get narrower up the neck, so the same span is easier higher up.
        comfortable = 3 if lo < 8 else 4
        cost += w.span * max(0, span - comfortable)
        # More than four fingers needed? Notes on the lowest fret can be barred.
        fingers = 1 + sum(1 for f in fretted if f != lo)
        if fingers > 4:
            cost += 3.0 * (fingers - 4)
        if len(fretted) < len(frets) and hi >= 7:
            cost += w.open_high_mix
    cost -= w.open_bonus * (len(frets) - len(fretted))
    if len(strings) >= 3:
        used = sorted(strings)
        gaps = (used[-1] - used[0] + 1) - len(used)
        cost += w.string_gap * gaps
        # Pitch order should follow string order in a chord.
    if any(strings[i] > strings[i + 1] for i in range(len(strings) - 1)):
        cost += 0.8
    return cost


def _placeable_subset(pitches, tuning: Tuning, max_fret: int, w: FingeringWeights):
    """Drop as few notes as possible so the chord fits on the strings.

    Prefers dropping octave doublings, then the notes that would be
    doubled anyway, then outer notes.
    """
    n = len(pitches)
    for k in range(min(n, len(tuning.strings)), 0, -1):
        best = None
        for keep in combinations(range(n), k):
            sub = [pitches[i] for i in keep]
            cands = _chord_candidates(sub, tuning, max_fret, w)
            if not cands:
                continue
            dropped = [pitches[i] for i in range(n) if i not in keep]
            pcs = {p % 12 for p in sub}
            # Losing a pitch class is worse than losing a doubling.
            lost_pc = sum(1 for p in dropped if p % 12 not in pcs)
            score = cands[0][0] + 2.0 * lost_pc
            if best is None or score < best[0]:
                best = (score, sub, cands, dropped)
        if best is not None:
            return best[1], best[2], best[3]
    return [], [], list(pitches)


class _Layer:
    """DP states for one event: (fingering candidate, hand position) pairs.

    The hand position is the fret under the index finger. Tracking it
    explicitly means open strings don't make the optimizer forget where the
    hand is, and a passage tends to stay in one position like a player would.
    """

    def __init__(self, event: Event, tuning: Tuning, max_fret: int, w: FingeringWeights):
        pitches = event.pitches
        cands = _chord_candidates(pitches, tuning, max_fret, w)
        dropped: list[int] = []
        kept = list(pitches)
        if not cands:
            kept, cands, dropped = _placeable_subset(pitches, tuning, max_fret, w)
        self.kept = kept
        self.dropped = dropped
        self.cands = cands
        self.time = event.time
        if not cands:
            return
        # Which kept notes are still sounding later on (for ringing clashes).
        ends = dict(zip(event.pitches, event.ends)) if event.ends else {}
        self.kept_ends = np.array([ends.get(p, event.time) for p in kept], dtype=float)

        max_hand = max(1, max_fret - 3)
        cand_idx, hand, static, mask, single = [], [], [], [], []
        for ci, (cost, strings, frets) in enumerate(cands):
            fretted = [f for f in frets if f > 0]
            m = 0
            for s in strings:
                m |= 1 << s
            if fretted:
                lo, hi = min(fretted), max(fretted)
                hands = range(max(1, hi - 4), lo + 1) if hi - 4 <= lo else [lo]
            else:
                hands = range(1, max_hand + 1)
            for h in hands:
                stretch = max(0, hi - (h + 3)) if fretted else 0
                cand_idx.append(ci)
                hand.append(h)
                static.append(cost + w.stretch * stretch + w.height * h)
                mask.append(m)
                single.append(strings[0] if len(strings) == 1 else -1)
        self.cand_idx = np.array(cand_idx)
        self.hand = np.array(hand, dtype=float)
        self.static = np.array(static)
        self.mask = np.array(mask, dtype=np.int64)
        self.single = np.array(single, dtype=float)

    def ringing_mask(self, t: float) -> np.ndarray:
        """Per state: bitmask of strings still ringing at time t."""
        ringing = self.kept_ends > t + 0.05
        out = np.zeros(len(self.cand_idx), dtype=np.int64)
        if not ringing.any():
            return out
        per_cand = np.zeros(len(self.cands), dtype=np.int64)
        for ci, (_, strings, _) in enumerate(self.cands):
            m = 0
            for s, r in zip(strings, ringing):
                if r:
                    m |= 1 << s
            per_cand[ci] = m
        return per_cand[self.cand_idx]


def _popcount(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    c = np.zeros_like(x)
    while x.any():
        c += x & 1
        x >>= 1
    return c


def _transition(a: _Layer, b: _Layer, w: FingeringWeights) -> np.ndarray:
    """Cost matrix [states of a, states of b] for moving from a to b."""
    dt = max(0.0, b.time - a.time)
    # A long gap gives time to move the hand for free-ish.
    ease = 1.0 / (1.0 + dt / max(w.ease_time, 1e-3))
    dist = np.abs(b.hand[None, :] - a.hand[:, None])
    move = (w.move * dist + w.shift * (dist > 0)) * ease
    # Single-note lines like to stay near the same string.
    single = (a.single[:, None] >= 0) & (b.single[None, :] >= 0)
    jump = np.where(single, w.string_jump * np.abs(b.single[None, :] - a.single[:, None]), 0.0)
    ring = a.ringing_mask(b.time)
    clash = _popcount(ring[:, None] & b.mask[None, :])
    return move + jump + w.ringing_clash * clash


def assign_frets(
    events: list[Event],
    tuning: Tuning,
    max_fret: int = 24,
    weights: FingeringWeights | None = None,
) -> tuple[list[Fingering], float]:
    """Choose a string/fret for every note. Returns (fingerings, total cost)."""
    w = weights or FingeringWeights()
    layers = [_Layer(e, tuning, max_fret, w) for e in events]
    # Viterbi over events that have at least one candidate.
    idx = [i for i, l in enumerate(layers) if l.cands]
    back: dict[int, np.ndarray] = {}
    total = 0.0
    prev = None
    cost = None
    for i in idx:
        layer = layers[i]
        if prev is None:
            cost = layer.static.copy()
        else:
            trans = _transition(layers[prev], layer, w)
            tot = cost[:, None] + trans
            back[i] = tot.argmin(axis=0)
            cost = tot.min(axis=0) + layer.static
        prev = i
    choice: dict[int, int] = {}
    if prev is not None:
        c = int(cost.argmin())
        total = float(cost[c])
        for j in reversed(range(len(idx))):
            i = idx[j]
            choice[i] = int(layers[i].cand_idx[c])
            if i in back:
                c = int(back[i][c])
    out = []
    for i, layer in enumerate(layers):
        if i in choice:
            _, strings, frets = layer.cands[choice[i]]
            out.append(Fingering(strings, frets, tuple(layer.kept), tuple(layer.dropped)))
        else:
            out.append(Fingering((), (), (), tuple(events[i].pitches)))
    n_dropped = sum(len(f.dropped) for f in out)
    return out, total + 4.0 * n_dropped


def robust_lowest(pitches: list[int], min_count: int = 3, min_frac: float = 0.004) -> int | None:
    """Lowest pitch that occurs often enough not to be a stray transcription error."""
    if not pitches:
        return None
    vals, counts = np.unique(np.asarray(pitches), return_counts=True)
    need = max(min_count, int(min_frac * len(pitches)))
    for v, c in zip(vals, counts):
        if c >= need:
            return int(v)
    return int(vals[0])


def credible_low_pitches(events: list[Event]) -> list[int]:
    """Pitches that can vouch for how low the instrument goes.

    A note doubled an octave up in the same chord doesn't count: distorted
    power chords produce real energy an octave below their root (the
    difference tone of root and fifth), which transcription picks up as a
    note. Real low strings also get played on their own; those phantoms
    never do. Worst case a genuine low root is only ever played doubled, and
    the tab keeps just its octave.
    """
    return [p for e in events for p in e.pitches if p + 12 not in e.pitches]


def fit_to_range(events: list[Event], tuning: Tuning) -> int:
    """Handle notes below the lowest string: drop them when the octave above
    is already in the chord (difference tones, octave doublings), otherwise
    move them up an octave. Returns how many notes were changed."""
    changed = 0
    for e in events:
        pitches, ends = [], []
        for p, end in zip(e.pitches, e.ends or [e.time] * len(e.pitches)):
            if p < tuning.lowest:
                changed += 1
                if p + 12 in e.pitches or p + 12 in pitches or p + 12 < tuning.lowest:
                    continue
                p += 12
            pitches.append(p)
            ends.append(end)
        order = sorted(range(len(pitches)), key=lambda i: pitches[i])
        e.pitches = [pitches[i] for i in order]
        if e.ends:
            e.ends = [ends[i] for i in order]
    return changed


def choose_tuning(
    events: list[Event],
    bass: bool,
    max_fret: int = 24,
    weights: FingeringWeights | None = None,
    band_tuning: str | None = None,
) -> Tuning:
    """Pick the tuning that makes the part most natural to play.

    The lowest regularly played note narrows it down to tunings whose lowest
    string sits at (or just below) that note; the fingering cost decides
    between the rest, e.g. Drop D vs D Standard.

    `band_tuning` is the guitar's tuning: bands tune together, so the bass
    gets the same one unless that leaves more than 5% of its notes too low.
    """
    if band_tuning:
        table = BASS_TUNINGS if bass else GUITAR_TUNINGS
        same = next((t for t in table if t.name == band_tuning), None)
        pitches = [p for e in events for p in e.pitches]
        if same and pitches and sum(p < same.lowest for p in pitches) <= 0.05 * len(pitches):
            return same
    table = BASS_TUNINGS if bass else GUITAR_TUNINGS
    low = robust_lowest(credible_low_pitches(events))
    if low is None:
        return table[0]
    fits = [t for t in table if t.lowest <= low]
    if not fits:
        return min(table, key=lambda t: t.lowest)
    top = max(t.lowest for t in fits)
    # Candidates: tunings whose lowest string is the closest one at/below `low`
    # (plus E standard, which is always worth trying if it fits).
    cands = [t for t in fits if t.lowest == top]
    if table[0] in fits and table[0] not in cands:
        cands.append(table[0])
    if len(cands) == 1:
        return cands[0]
    n_notes = max(1, sum(len(e.pitches) for e in events))
    best = None
    for t in cands:
        _, cost = assign_frets(events, t, max_fret, weights)
        # Normalise per note so the prior means the same for every song length.
        score = cost / n_notes + t.prior
        if best is None or score < best[0]:
            best = (score, t)
    return best[1]
