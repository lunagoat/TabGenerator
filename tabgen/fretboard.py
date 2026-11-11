"""Tunings, and turning pitches into playable string/fret positions.

Every note has several places it can be played on the neck. We pick one per
note by running Viterbi over the song: each chord gets a set of candidate
fingerings with a cost for how awkward the shape is, and moving between
consecutive shapes costs according to how far the fretting hand travels.
"""

from dataclasses import dataclass, field, replace
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
    """Costs used by the fingering optimizer.

    Tuned with eval/train_fingering.py on 578 human-made rock tabs: on 132
    held-out tabs the string the tabber chose is picked for 77.6% of notes
    (68.2% before learning from tabs), and on GuitarSet for 77.5% (74.4%).
    """

    span: float = 0.3  # per fret of stretch beyond a comfortable 3-fret span
    span_hard: int = 5  # anything wider is not considered
    stretch: float = 1.0  # note one fret past the 4-fret box under the hand
    height: float = 0.02  # per fret of hand position; retain a low-position prior
    open_bonus: float = 0.0  # open strings need no finger
    open_high_mix: float = 0.0  # open string mixed with a fretted note at fret >= 7
    string_gap: float = 1.2  # per unused string inside a 3+ note chord
    move: float = 3.0  # per fret the hand moves
    shift: float = 0.0  # any hand movement at all
    ease_time: float = 0.35  # seconds of gap that halve the cost of moving
    string_jump: float = 0.36  # per string crossed between single notes
    ringing_clash: float = 3.0  # new note on a string that is still ringing
    top_k: int = 16
    run_move: float = 4.0  # extra per-fret movement cost between single notes
    open_anchor: float = 0.0  # per fret of hand position above 2 while an open string is played
    index_reach: float = 0.0  # per fret between the index finger and the lowest fretted note
    rest_ease: float = 0.5  # blend onset spacing (0) with actual note-off rest (1)
    # Learned from human-made tabs (see FingeringPrior): how strongly to
    # prefer chord shapes, and strings for single notes, that tabbers use.
    shape_prior: float = 1.0
    string_prior: float = 1.0
    prior: "FingeringPrior | None" = None  # None: the one shipped in tabgen/data


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


class FingeringPrior:
    """How often human tabbers put notes where: counts learned from Guitar
    Pro files (eval/train_fingering.py).

    - chord shapes: for a set of pitches (as intervals above the lowest), how
      often each shape (strings and frets relative to the lowest string and
      fret) was used, per tuning layout (the intervals between the strings)
      with a fallback over all layouts;
    - register: for a pitch at a given height above the lowest open string,
      how often each string was used.

    Only ratios between the options for one chord matter, so a cost is
    -log(count + smoothing).
    """

    SMOOTH = 0.5

    def __init__(self, shapes=None, shapes_any=None, strings=None, tunings=None):
        self.shapes = shapes or {}  # (layout, intervals) -> {shape: count}
        self.shapes_any = shapes_any or {}  # intervals -> {shape: count}
        self.strings = strings or {}  # (n_strings, height) -> {string: count}
        self.tunings = tunings or {}  # tuning name -> tracks tabbed in it

    @staticmethod
    def layout(tuning: Tuning) -> str:
        return ",".join(str(b - a) for a, b in zip(tuning.strings, tuning.strings[1:]))

    @staticmethod
    def shape_key(pitches, strings, frets):
        """(intervals, shape) as strings, both relative so they transpose."""
        order = sorted(range(len(pitches)), key=lambda i: pitches[i])
        p0, s0 = pitches[order[0]], min(strings)
        fretted = [f for f in frets if f > 0]
        f0 = min(fretted) if fretted else 0
        intervals = ".".join(str(pitches[i] - p0) for i in order)
        shape = ".".join(f"{strings[i] - s0}:{frets[i] - f0 if frets[i] > 0 else 'o'}" for i in order)
        return intervals, shape

    def add(self, tuning: Tuning, pitches, strings, frets):
        if len(pitches) == 1:
            key = f"{len(tuning.strings)}|{pitches[0] - tuning.lowest}"
            d = self.strings.setdefault(key, {})
            d[str(strings[0])] = d.get(str(strings[0]), 0) + 1
            return
        intervals, shape = self.shape_key(pitches, strings, frets)
        for table, key in ((self.shapes, f"{self.layout(tuning)}|{intervals}"), (self.shapes_any, intervals)):
            d = table.setdefault(key, {})
            d[shape] = d.get(shape, 0) + 1
        # The lowest note's string says where on the neck the chord sits.
        low = min(range(len(pitches)), key=lambda i: pitches[i])
        key = f"{len(tuning.strings)}|{pitches[low] - tuning.lowest}"
        d = self.strings.setdefault(key, {})
        d[str(strings[low])] = d.get(str(strings[low]), 0) + 1

    def costs(self, tuning: Tuning, pitches, strings, frets) -> tuple[float, float]:
        """(shape cost, register cost) of one candidate."""
        import math

        low = min(range(len(pitches)), key=lambda i: pitches[i])
        reg = self.strings.get(f"{len(tuning.strings)}|{pitches[low] - tuning.lowest}", {})
        register = -math.log(reg.get(str(strings[low]), 0) + self.SMOOTH)
        if len(pitches) == 1:
            return 0.0, register
        intervals, shape = self.shape_key(pitches, strings, frets)
        table = self.shapes.get(f"{self.layout(tuning)}|{intervals}")
        if not table or sum(table.values()) < 5:
            table = self.shapes_any.get(intervals, {})
        return -math.log(table.get(shape, 0) + self.SMOOTH), register

    def to_json(self) -> dict:
        return {"shapes": self.shapes, "shapes_any": self.shapes_any, "strings": self.strings, "tunings": self.tunings}

    @classmethod
    def from_json(cls, d: dict) -> "FingeringPrior":
        return cls(d["shapes"], d["shapes_any"], d["strings"], d.get("tunings"))

    def pruned(self, min_count: int = 2) -> "FingeringPrior":
        """Without shapes seen fewer than `min_count` times (smaller file)."""
        def prune(t):
            out = {}
            for k, d in t.items():
                d2 = {s: c for s, c in d.items() if c >= min_count}
                if d2:
                    out[k] = d2
            return out
        return FingeringPrior(prune(self.shapes), prune(self.shapes_any), self.strings, self.tunings)


_DEFAULT_PRIOR = None


def default_prior() -> "FingeringPrior | None":
    """The prior shipped in tabgen/data/fingering_prior.json, if present."""
    global _DEFAULT_PRIOR
    if _DEFAULT_PRIOR is None:
        import json
        from pathlib import Path

        path = Path(__file__).with_name("data") / "fingering_prior.json"
        _DEFAULT_PRIOR = FingeringPrior.from_json(json.loads(path.read_text())) if path.is_file() else False
    return _DEFAULT_PRIOR or None


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
    if w.prior is not None and (w.shape_prior or w.string_prior):
        rescored = []
        for cost, s, f in scored:
            shape, register = w.prior.costs(tuning, pitches, s, f)
            rescored.append((cost + w.shape_prior * shape + w.string_prior * register, s, f))
        scored = rescored
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
                # Open strings are played with the hand near the nut: a riff
                # that mixes them with fretted notes sits in open position.
                anchor = w.open_anchor * max(0, h - 2) if len(fretted) < len(frets) else 0.0
                # The index finger usually takes the lowest fretted note:
                # nobody chugs a 4th-fret note with the hand parked at the nut.
                reach = w.index_reach * (lo - h) if fretted else 0.0
                static.append(cost + w.stretch * stretch + w.height * h + anchor + reach)
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
    # Time spent holding a note is not all available for shifting. Blend
    # onset spacing with actual silence; missing note-offs preserve the old
    # onset-based behavior. A rest still gives the hand time to relocate.
    rest = max(0.0, b.time - float(a.kept_ends.max())) if len(a.kept_ends) else dt
    dt = dt * (1.0 - w.rest_ease) + rest * w.rest_ease
    ease = 1.0 / (1.0 + dt / max(w.ease_time, 1e-3))
    dist = np.abs(b.hand[None, :] - a.hand[:, None])
    move = (w.move * dist + w.shift * (dist > 0)) * ease
    # Single-note runs favour a stable position box more strongly than
    # chord changes do. This uses the sounding notes, not an instrument,
    # tuning, genre or dataset-specific register assumption.
    single = (a.single[:, None] >= 0) & (b.single[None, :] >= 0)
    move += single * w.run_move * dist * ease
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
    if w.prior is None and (w.shape_prior or w.string_prior):
        w = replace(w, prior=default_prior())
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


TUNING_RARITY = 0.2  # weight of -log(how common a tuning is) against fingering cost per note
TUNING_MAX_EVENTS = 1500  # enough of a song to judge its tuning


def choose_tuning(
    events: list[Event],
    bass: bool,
    max_fret: int = 24,
    weights: FingeringWeights | None = None,
    band_tuning: str | None = None,
) -> Tuning:
    """Pick the tuning the part was most likely played in.

    Every tuning the part fits on (its lowest regularly played note is not
    below the lowest string) competes: how natural the part is to play in it
    (the fingering cost per note, which includes how human tabbers place
    notes and chord shapes), plus how common the tuning is. A half-step-down
    song that never touches its lowest string still sits on the open strings
    and shapes of Eb tuning, which the fingering cost notices. On held-out
    human tabs this picks the tab's tuning for 91.6% of tracks (86.6% when
    only the tunings with the closest lowest string were tried).

    `band_tuning` is the guitar's tuning: bands tune together, so the bass
    gets the same one unless that leaves more than 5% of its notes too low.
    """
    import math

    table = BASS_TUNINGS if bass else GUITAR_TUNINGS
    if band_tuning:
        same = next((t for t in table if t.name == band_tuning), None)
        pitches = [p for e in events for p in e.pitches]
        if same and pitches and sum(p < same.lowest for p in pitches) <= 0.05 * len(pitches):
            return same
    low = robust_lowest(credible_low_pitches(events))
    if low is None:
        return table[0]
    fits = [t for t in table if t.lowest <= low]
    if not fits:
        return min(table, key=lambda t: t.lowest)
    prior = (weights.prior if weights and weights.prior else None) or default_prior()
    counts = prior.tunings if prior and not bass else {}
    if not counts:
        # No statistics (bass, or no prior file): the closest tunings only,
        # with hand-set priors.
        top = max(t.lowest for t in fits)
        fits = [t for t in fits if t.lowest == top] + ([table[0]] if table[0] in fits and table[0].lowest != top else [])
    if len(fits) == 1:
        return fits[0]
    total = sum(counts.values())
    ev = [Event(e.time, list(e.pitches), list(e.ends)) for e in events[:TUNING_MAX_EVENTS]]
    n_notes = max(1, sum(len(e.pitches) for e in ev))
    best = None
    for t in fits:
        _, cost = assign_frets([Event(e.time, list(e.pitches), list(e.ends)) for e in ev], t, max_fret, weights)
        if counts:
            rarity = TUNING_RARITY * -math.log((counts.get(t.name, 0) + 1) / (total + len(table)))
        else:
            rarity = t.prior
        score = cost / n_notes + rarity
        if best is None or score < best[0]:
            best = (score, t)
    return best[1]
