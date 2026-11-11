"""Turning a transcription into something a guitarist would write down.

A transcription model reports what it hears, note by note, and is right
about most notes and wrong about some. Its mistakes differ every time a riff
comes round, so the same two bars come out slightly different in every
repeat; overtones of distorted low strings show up as extra high notes; and
a second guitar's line lands in the middle of the riff. A player reads all
of that as a mess that nobody could play. This module tidies it up:

1. `split_layers`: a line that another guitar plays on top of a riff is
   taken out of the riff's part.
2. `unify_repeats`: bars that are clearly the same riff are compared. A note
   only one of them has, and that not every transcription pass heard, is
   dropped; a note nearly all of them have is restored where it went
   missing.
3. `remove_overtones`: high notes that are a harmonic of a lower note in the
   same strum (distorted guitar only).
4. `check_playable` (after string/fret positions are chosen): what one hand
   cannot do is counted, and notes that need the hand somewhere else
   entirely for a single strum are left out.
"""

from collections import Counter
from dataclasses import dataclass, replace

import numpy as np

from .rhythm import BeatMap, QNote

SAME_RIFF = 0.5  # bars sharing this much (Jaccard) are repeats of one riff
MIN_REPEATS = 3
# Measured against Official tabs of real recordings (eval/eval_tidy.py), the
# transcription is more often right than its neighbours' consensus: a one-off
# note is usually a real variation, so only doubtful ones are dropped, and a
# note is only filled in when practically every repeat has it. (Rewriting
# strums as the chord shape played around them was dropped for the same
# reason: it put in more wrong notes than right ones.)
DROP_SHARE = 0.34  # in at most this share of the repeats and not heard by every pass: a slip, dropped
DROP_SHARE_SURE = -1.0  # ... for a note every transcription pass heard: never
SURE = 0.99
ADD_SHARE = 0.9  # in at least this share: part of the riff, restored


@dataclass
class Report:
    """What tidying changed, for the note at the bottom of the tab."""

    riffs: int = 0  # groups of repeated bars found
    bars: int = 0  # bars that belong to one
    dropped: int = 0
    added: int = 0
    overtones: int = 0
    out_of_reach: int = 0
    hard_jumps: int = 0  # fast position changes left in the tab

    def lines(self) -> list[str]:
        out = []
        if self.bars:
            out.append(
                f"({self.bars} bars are repeats of {self.riffs} riffs and were made to agree: "
                f"{self.dropped} one-off notes left out, {self.added} missing notes filled in.)"
            )
        if self.overtones:
            out.append(f"({self.overtones} overtone ghosts of lower notes were left out.)")
        if self.out_of_reach:
            out.append(f"({self.out_of_reach} notes out of the hand's reach were left out.)")
        if self.hard_jumps:
            out.append(f"(Playability: {self.hard_jumps} quick jumps of 5+ frets remain; check those spots by ear.)")
        return out


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _consensus(sets: list[frozenset], share: float = 0.5) -> frozenset:
    counts = Counter(x for s in sets for x in s)
    return frozenset(x for x, c in counts.items() if c >= share * len(sets))


def find_repeats(sets: list[frozenset], min_items: int = 3) -> list[list[int]]:
    """Group segments (sets of (position, pitch)) that are the same riff."""
    free = [i for i in range(len(sets)) if len(sets[i]) >= min_items]
    groups: list[list[int]] = []
    while len(free) >= MIN_REPEATS:
        sim = {i: [j for j in free if _jaccard(sets[i], sets[j]) >= SAME_RIFF] for i in free}
        seed = max(free, key=lambda i: len(sim[i]))
        members = sim[seed]
        if len(members) < MIN_REPEATS:
            break
        # Re-centre on what the group agrees on, so the group isn't defined
        # by one bar's mistakes.
        for _ in range(2):
            core = _consensus([sets[i] for i in members])
            again = [j for j in free if _jaccard(sets[j], core) >= SAME_RIFF]
            if len(again) < MIN_REPEATS:
                break
            members = again
        groups.append(sorted(members))
        taken = set(members)
        free = [i for i in free if i not in taken]
    return groups


def segments(bm: BeatMap) -> list[tuple[int, int]]:
    """(start tick, length in ticks) of every bar."""
    bt = bm.beat_ticks
    return [(bar.first_beat * bt, bar.n_beats * bt) for bar in bm.bars]


def unify_repeats(notes: list[QNote], bm: BeatMap, report: Report | None = None,
                  key=lambda q: q.pitch) -> list[QNote]:
    """Make repeats of a riff agree with each other (see module docstring).

    `key` says what counts as "the same note" (the pitch; for drums, the
    drum)."""
    segs = segments(bm)
    starts = np.array([s for s, _ in segs])
    per_seg: list[dict] = [{} for _ in segs]  # (relative tick, key) -> note
    loose: list[QNote] = []
    for q in notes:
        i = int(np.searchsorted(starts, q.onset_tick, side="right")) - 1
        if i < 0 or q.onset_tick >= segs[i][0] + segs[i][1]:
            loose.append(q)
            continue
        per_seg[i].setdefault((q.onset_tick - segs[i][0], key(q)), q)
    by_len: dict[int, list[int]] = {}
    for i, (_, length) in enumerate(segs):
        by_len.setdefault(length, []).append(i)
    bt = bm.beat_ticks
    report = report if report is not None else Report()
    for idx in by_len.values():
        for group in find_repeats([frozenset(per_seg[i]) for i in idx]):
            members = [idx[g] for g in group]
            n = len(members)
            counts = Counter(x for m in members for x in per_seg[m])
            report.riffs += 1
            report.bars += n
            for m in members:
                for item in list(per_seg[m]):
                    share = DROP_SHARE if per_seg[m][item].conf < SURE else DROP_SHARE_SURE
                    if counts[item] <= share * n:
                        del per_seg[m][item]
                        report.dropped += 1
            for item, c in counts.items():
                if c < ADD_SHARE * n:
                    continue
                donors = [per_seg[m][item] for m in members if item in per_seg[m]]
                length = int(np.median([d.end_tick - d.onset_tick for d in donors]))
                seconds = float(np.median([d.offset - d.onset for d in donors]))
                for m in members:
                    if item in per_seg[m]:
                        continue
                    tick = segs[m][0] + item[0]
                    onset = float(bm.beat_to_time(tick / bt))
                    per_seg[m][item] = replace(
                        donors[0], onset_tick=tick, end_tick=tick + length, onset=onset, offset=onset + seconds
                    )
                    report.added += 1
    out = loose + [q for seg in per_seg for q in seg.values()]
    out.sort(key=lambda q: (q.onset_tick, q.pitch))
    return out


# ----------------------------------------------------------------------------
# Overtones

# Semitones above a fundamental at which its 3rd, 4th, 5th and 6th harmonics
# sit. (The 2nd, an octave up, is left alone: it is a normal chord tone.)
OVERTONES = (19, 24, 28, 31)
ISOLATED_GAP = 10  # an overtone ghost has no chord tone this close below it


def remove_overtones(notes: list[QNote], report: Report | None = None) -> list[QNote]:
    """Drop notes that are a harmonic of a lower note in the same strum.

    Distortion makes the 3rd to 6th harmonics of the low strings loud enough
    to be transcribed as notes. Such a ghost sits alone, well above the real
    chord; a real chord tone that happens to be a 12th above the bass (the B
    in an open E chord) has neighbours, and is kept. Only notes the model
    heard as distorted guitar are looked at.
    """
    by_tick: dict[int, list[QNote]] = {}
    for q in notes:
        by_tick.setdefault(q.onset_tick, []).append(q)
    out = []
    for group in by_tick.values():
        group.sort(key=lambda q: q.pitch)
        for i, q in enumerate(group):
            ghost = (
                i > 0
                and "distorted" in q.instrument
                and q.pitch - group[i - 1].pitch >= ISOLATED_GAP
                and any((q.pitch - lower.pitch) in OVERTONES for lower in group[:i])
            )
            if ghost:
                if report is not None:
                    report.overtones += 1
            else:
                out.append(q)
    out.sort(key=lambda q: (q.onset_tick, q.pitch))
    return out


# ----------------------------------------------------------------------------
# Layers: a second guitar on top of the riff

LAYER_GAP = 8  # semitones of empty space between the riff and what sits on top
OVERLAY_PRESENT = 0.7  # on top of fewer of the riff's repeats than this: another guitar


def _split_register(pitches: list[int]) -> int | None:
    """The pitch above which a bar's notes form a separate upper group
    (largest gap of at least LAYER_GAP semitones), or None."""
    uniq = sorted(set(pitches))
    best = None
    for lo, hi in zip(uniq, uniq[1:]):
        if hi - lo >= LAYER_GAP and (best is None or hi - lo > best[0]):
            best = (hi - lo, hi)
    if best is None:
        return None
    cut = best[1]
    n_hi = sum(p >= cut for p in pitches)
    return cut if n_hi >= 2 and len(pitches) - n_hi >= 2 else None


def split_layers(notes: list, bm: BeatMap, quantized: list[QNote]) -> tuple[list, list]:
    """Separate a line played on top of a riff by another guitar.

    `quantized[i]` is `notes[i]` on the grid. When the same riff (the lower
    notes of a bar) comes round several times and only some of the repeats
    have a line on top of it, or the line differs from repeat to repeat,
    that line is not part of the riff: one guitarist plays the riff, someone
    else the line. Returns (riff notes, notes of the line on top).

    A pattern one player picks across the strings is left alone: its upper
    notes come back with the lower ones every time.
    """
    segs = segments(bm)
    starts = np.array([s for s, _ in segs])
    in_seg: list[list[int]] = [[] for _ in segs]
    for i, q in enumerate(quantized):
        s = int(np.searchsorted(starts, q.onset_tick, side="right")) - 1
        if 0 <= s and q.onset_tick < segs[s][0] + segs[s][1]:
            in_seg[s].append(i)
    low_sets, high_sets, high_idx = [], [], []
    for s, members in enumerate(in_seg):
        cut = _split_register([quantized[i].pitch for i in members])
        lo = [i for i in members if cut is None or quantized[i].pitch < cut]
        hi = [i for i in members if cut is not None and quantized[i].pitch >= cut]
        item = lambda i: (quantized[i].onset_tick - segs[s][0], quantized[i].pitch)
        # The riff is recognised by its pitches alone: the model loses
        # different chord tones in different repeats.
        low_sets.append(frozenset(quantized[i].pitch for i in lo))
        high_sets.append(frozenset(item(i) for i in hi))
        high_idx.append(hi)
    overlay: set[int] = set()
    by_len: dict[int, list[int]] = {}
    for s, (_, length) in enumerate(segs):
        by_len.setdefault(length, []).append(s)
    for idx in by_len.values():
        for group in find_repeats([low_sets[s] for s in idx], min_items=1):
            members = [idx[g] for g in group]
            with_high = [s for s in members if high_sets[s]]
            if not with_high:
                continue
            same = [
                _jaccard(high_sets[a], high_sets[b])
                for k, a in enumerate(with_high) for b in with_high[k + 1:]
            ]
            steady = len(with_high) >= OVERLAY_PRESENT * len(members) and (not same or np.mean(same) >= SAME_RIFF)
            if not steady:
                for s in with_high:
                    overlay.update(high_idx[s])
    main = [n for i, n in enumerate(notes) if i not in overlay]
    top = [n for i, n in enumerate(notes) if i in overlay]
    return main, top


# ----------------------------------------------------------------------------
# Playability

REACH = 5  # frets from the index finger that a hand covers at a stretch
JUMP_FRETS = 5
QUICK_S = 0.25  # less time than this to get there and less to get back


def _hand(fingering) -> int | None:
    """Fret under the index finger (None: open strings only, hand is free)."""
    fretted = [f for f in fingering.frets if f > 0]
    return min(fretted) if fretted else None


def check_playable(events: list, fingerings: list, report: Report | None = None) -> list:
    """Check chosen string/fret positions for what one hand cannot do.

    A strum that needs the hand five or more frets away from where it is
    just before and just after, with no time to get there and back, is not
    something a player does: the notes of that strum that are out of reach
    are left out (open strings and notes within reach stay). Quick jumps
    that remain, where the music really moves, are counted in the report.
    """
    from .fretboard import Fingering

    report = report if report is not None else Report()
    out = list(fingerings)
    fretted = [i for i, f in enumerate(out) if _hand(f) is not None]
    for k in range(1, len(fretted) - 1):
        a, i, b = fretted[k - 1], fretted[k], fretted[k + 1]
        ha, hi, hb = _hand(out[a]), _hand(out[i]), _hand(out[b])
        if ha is None or hi is None or hb is None:
            continue
        there = events[i].time - events[a].time
        back = events[b].time - events[i].time
        if abs(ha - hb) > 2 or there > QUICK_S or back > QUICK_S:
            continue
        home = min(ha, hb)
        if abs(hi - home) < JUMP_FRETS:
            continue
        f = out[i]
        keep = [j for j, fr in enumerate(f.frets) if fr == 0 or home - 1 <= fr <= home + REACH]
        gone = [f.pitches[j] for j in range(len(f.frets)) if j not in keep]
        report.out_of_reach += len(gone)
        out[i] = Fingering(
            tuple(f.strings[j] for j in keep), tuple(f.frets[j] for j in keep),
            tuple(f.pitches[j] for j in keep), tuple(f.dropped) + tuple(gone),
        )
    last = None
    for i, f in enumerate(out):
        h = _hand(f)
        if h is None:
            continue
        if last is not None and abs(h - last[0]) >= JUMP_FRETS and events[i].time - last[1] < QUICK_S:
            report.hard_jumps += 1
        last = (h, events[i].time)
    return out
