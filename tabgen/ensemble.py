"""Combine several transcriptions of the same part by voting.

Each transcription sees different audio (a stem, a partial mix, the full
mix), so their mistakes differ: a note that at least `min_votes` of them agree
on (same pitch - or same drum type - within 50 ms) is kept, the rest dropped.
On the evaluation sets this beat the best single transcription for every
part (see README).
"""

import bisect

from .transcribe import TNote

TOLERANCE = 0.05

DRUM_TYPE = {}
for _kind, _notes in {
    "kick": [35, 36],
    "snare": [37, 38, 39, 40],
    "hihat": [42, 44, 46, 26],
    "tom": [41, 43, 45, 47, 48, 50],
    "cymbal": [49, 51, 52, 53, 55, 57, 59],
}.items():
    for _n in _notes:
        DRUM_TYPE[_n] = _kind


def _key(n: TNote):
    return DRUM_TYPE.get(n.pitch, n.pitch) if n.is_drum else n.pitch


class _Index:
    """Onset times per pitch (or drum type) for fast 'is there one nearby'."""

    def __init__(self, notes: list[TNote]):
        self.times: dict = {}
        for n in notes:
            self.times.setdefault(_key(n), []).append(n.onset)
        for v in self.times.values():
            v.sort()

    def has(self, n: TNote, tol: float = TOLERANCE) -> bool:
        ts = self.times.get(_key(n))
        if not ts:
            return False
        i = bisect.bisect_left(ts, n.onset - tol)
        return i < len(ts) and ts[i] <= n.onset + tol

    def add(self, n: TNote):
        bisect.insort(self.times.setdefault(_key(n), []), n.onset)


def vote(transcriptions: list[list[TNote]], min_votes: int) -> list[TNote]:
    """Notes found by at least `min_votes` transcriptions.

    Earlier transcriptions win when choosing which copy of a note (its exact
    timing and length) to keep, so list the most reliable one first.
    """
    if min_votes <= 1 and len(transcriptions) == 1:
        return list(transcriptions[0])
    indexes = [_Index(t) for t in transcriptions]
    kept: list[TNote] = []
    kept_index = _Index([])
    for i, notes in enumerate(transcriptions):
        for n in notes:
            if kept_index.has(n, 0.03):
                continue  # already kept from an earlier transcription
            votes = 1 + sum(idx.has(n) for j, idx in enumerate(indexes) if j != i)
            if votes >= min_votes:
                kept.append(n)
                kept_index.add(n)
    kept.sort(key=lambda n: (n.onset, n.pitch))
    return kept


def lowest_only(notes: list[TNote], window: float = 0.03) -> list[TNote]:
    """Keep the lowest of notes starting together. Bass is (almost always)
    one note at a time; the extra notes are guitar or keys bleeding in.
    This raised bass F1 from 0.777 to 0.792 on BabySlakh."""
    out: list[TNote] = []
    for n in sorted(notes, key=lambda n: (n.onset, n.pitch)):
        if out and n.onset - out[-1].onset < window:
            continue
        out.append(n)
    return out


def merge(
    transcriptions: list[list[TNote]],
    min_votes: float,
    weights: list[float] | None = None,
    tol: float = TOLERANCE,
) -> list[TNote]:
    """Like `vote`, but the kept note is the consensus of the copies that
    agree instead of the first copy: its start and end are the medians over
    the transcriptions that found it, so one late or early copy doesn't set
    the timing. `weights` lets some transcriptions count for more.

    Each kept note's `conf` is the share of the vote it received (0..1).
    """
    weights = weights or [1.0] * len(transcriptions)
    by_key: dict = {}
    for ti, notes in enumerate(transcriptions):
        for n in notes:
            by_key.setdefault(_key(n), []).append((n.onset, ti, n))
    total = sum(weights)
    out: list[TNote] = []
    for items in by_key.values():
        items.sort(key=lambda x: x[0])
        i = 0
        while i < len(items):
            # A cluster: copies of one note, each transcription at most once,
            # all within `tol` of the first.
            start = items[i][0]
            seen: dict[int, TNote] = {}
            j = i
            while j < len(items) and items[j][0] - start <= tol and items[j][1] not in seen:
                seen[items[j][1]] = items[j][2]
                j += 1
            score = sum(weights[t] for t in seen)
            if score >= min_votes:
                copies = list(seen.values())
                onsets = sorted(c.onset for c in copies)
                ends = sorted(c.offset for c in copies)
                first = seen[min(seen)]  # earliest-listed transcription names the instrument
                on = onsets[len(onsets) // 2] if len(onsets) % 2 else 0.5 * (onsets[len(onsets) // 2 - 1] + onsets[len(onsets) // 2])
                off = ends[len(ends) // 2]
                out.append(TNote(on, max(off, on + 0.01), first.pitch, first.instrument, first.is_drum, score / total))
            i = j
    out.sort(key=lambda n: (n.onset, n.pitch))
    return out


HIHAT_ALIASES = {54}  # tambourine
CLOSED_HIHAT = 42


def normalize_drums(notes: list[TNote], window: float = 0.03) -> list[TNote]:
    """Fold the model's "tambourine" into the hi-hat.

    MuScriptor often writes a tambourine hit on top of (or instead of) a
    closed hi-hat: on MDB Drums 70% of its tambourine hits sit on a real
    hi-hat hit and 1% on a real tambourine. A tambourine hit with a hi-hat
    hit next to it is dropped; one on its own becomes a closed hi-hat."""
    hats = sorted(n.onset for n in notes if n.is_drum and DRUM_TYPE.get(n.pitch) == "hihat")
    out = []
    for n in notes:
        if n.is_drum and n.pitch in HIHAT_ALIASES:
            i = bisect.bisect_left(hats, n.onset - window)
            if i < len(hats) and hats[i] <= n.onset + window:
                continue
            n = TNote(n.onset, n.offset, CLOSED_HIHAT, n.instrument, True, n.conf)
        out.append(n)
    return out
