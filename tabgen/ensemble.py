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
