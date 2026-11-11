"""Human-made Guitar Pro tabs (dumped by tabdump.sh) as fingering data.

Each guitar track becomes a list of events (notes starting together, as
`fretboard.Event`) with the string each note was played on. Dead notes,
harmonics and tied continuations are left out: their pitch is not what the
string/fret says. Songs are grouped so that the .gp and .gpx copies of one
song never end up on both sides of a train/test split.
"""
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import env

from tabgen import fretboard

TICKS = 960  # TuxGuitar ticks per quarter note
SKIP_FLAGS = set("txhc")  # tied, dead, harmonic, grace


@dataclass
class Track:
    song: str  # group key
    name: str
    tuning: fretboard.Tuning
    events: list  # fretboard.Event
    truth: list  # per event: {pitch: string index (0 = lowest)}
    frets: list  # per event: {pitch: fret}


def _group(path: str) -> str:
    stem = Path(path).stem.lower()
    stem = re.sub(r"\(.*?\)|\[.*?\]", "", stem)
    return re.sub(r"[^a-z]", "", stem)


def is_guitar(t) -> bool:
    if t["percussion"] or not t["strings"]:
        return False
    n, low, p = len(t["strings"]), min(t["strings"]), t["program"]
    if 32 <= p <= 39 or low < 33:
        return False  # bass
    return n in (6, 7, 8) and (24 <= p <= 31 or p < 0 or n >= 6)


def _seconds(bars):
    """tick -> seconds, following the tempo of each bar."""
    starts, secs, rates = [], [], []
    t = 0.0
    for start, length, tempo, *_ in bars:
        rate = 60.0 / (max(tempo, 1) * TICKS)
        starts.append(start)
        secs.append(t)
        rates.append(rate)
        t += length * rate
    import bisect

    def conv(tick):
        i = max(0, bisect.bisect_right(starts, tick) - 1)
        return secs[i] + (tick - starts[i]) * rates[i]

    return conv


def load(paths=None, min_events=20):
    paths = paths or sorted((env.DATA / "tabs").glob("*.jsonl"))
    out = []
    for p in paths:
        for line in open(p, encoding="utf-8"):
            song = json.loads(line)
            conv = _seconds(song["bars"])
            for t in song["tracks"]:
                if not is_guitar(t):
                    continue
                strings = t["strings"]  # string 1 = highest
                n = len(strings)
                tuning = fretboard.Tuning("tab", tuple(sorted(strings)))
                order = sorted(range(n), key=lambda i: strings[i])
                index_of = {i + 1: order.index(i) for i in range(n)}  # TG string number -> 0 = lowest
                by_tick: dict = {}
                for tick, dur, voice, s, fret, flags in t["notes"]:
                    if set(flags) & SKIP_FLAGS or not 1 <= s <= n:
                        continue
                    by_tick.setdefault(tick, []).append((strings[s - 1] + fret, index_of[s], fret, tick + dur))
                events, truth, frets = [], [], []
                for tick in sorted(by_tick):
                    notes = {}
                    for pitch, si, fret, end in by_tick[tick]:
                        notes.setdefault(pitch, (si, fret, end))  # unison on two strings: keep one
                    ps = sorted(notes)
                    events.append(fretboard.Event(conv(tick), ps, [conv(notes[q][2]) for q in ps]))
                    truth.append({q: notes[q][0] for q in ps})
                    frets.append({q: notes[q][1] for q in ps})
                if len(events) >= min_events:
                    out.append(Track(_group(song["path"]), t["name"], tuning, events, truth, frets))
    return out


def split(tracks, test_share=0.2, seed="tabs"):
    """Deterministic split by song."""
    def is_test(key):
        return int(hashlib.sha1((seed + key).encode()).hexdigest(), 16) % 1000 < test_share * 1000
    return [t for t in tracks if not is_test(t.song)], [t for t in tracks if is_test(t.song)]


def score(tracks, weights=None, assign=None, max_events=None):
    """Share of notes put on the string the tab uses; also by notes per event."""
    assign = assign or fretboard.assign_frets
    right = total = 0
    single = [0, 0]
    chord = [0, 0]
    for tr in tracks:
        ev = tr.events[:max_events] if max_events else tr.events
        import copy

        fing, _ = assign(copy.deepcopy(ev), tr.tuning, 24, weights)
        for f, t in zip(fing, tr.truth):
            got = dict(zip(f.pitches, f.strings))
            bucket = single if len(t) == 1 else chord
            for p, s in t.items():
                ok = got.get(p) == s
                right += ok
                total += 1
                bucket[0] += ok
                bucket[1] += 1
    return right / max(1, total), single[0] / max(1, single[1]), chord[0] / max(1, chord[1]), total


if __name__ == "__main__":
    import time

    tracks = load()
    train, test = split(tracks)
    print(len(tracks), "guitar tracks;", len(train), "train /", len(test), "test;",
          sum(len(t.events) for t in tracks), "events")
    t0 = time.time()
    acc, s, c, n = score(test)
    print(f"current fingering on held-out tabs: {acc:.4f} (single notes {s:.4f}, chords {c:.4f}; {n} notes) {time.time() - t0:.0f}s")
