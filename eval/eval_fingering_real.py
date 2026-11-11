#!/usr/bin/env python3
"""String choice on the pipeline's own transcriptions, with the tuning given.

eval_fingering_model.py asks: given the right notes, is each one put on the
tab's string? This asks the same of what the fret chooser really gets: the
transcribed, voted, quantized and tidied notes of every part of every real
recording that has a tab (Official tabs: guitar and bass; your tabs: guitar),
with missing notes, extra notes and several guitars in one part. A note
counts when the aligned tab has it too (same pitch within 100 ms); the part
is fingered in the tab's tuning, so only the string choice is scored.

    python3 eval/eval_fingering_real.py [--refresh]

The parts are cached in eval/work/fingering_real.pkl; --refresh rebuilds
them (after changing transcription, voting, part splitting or tidying).
"""
import argparse
import bisect
import collections
import contextlib
import io
import pickle
from pathlib import Path

import env
import numpy as np

import eval_tuning_band as TB
import official as O
import realeval as R
import realsongs
import tabdata
from tabgen import arrange, fretboard, pipeline

CACHE = env.WORK / "fingering_real.pkl"
SKIP = set("txhc")


def _tracks(row, kind, order=None):
    """[(sounding open strings lowest first, [(tab seconds, pitch, string from lowest)])] per track,
    bars played in `order` (default: as written)."""
    bars = row["bars"]
    order = list(range(len(bars))) if order is None else order
    starts = [b[0] for b in bars]
    out = []
    for tr in row["tracks"]:
        if not kind(tr):
            continue
        cap = tr.get("offset") or 0
        strings = [p + cap for p in tr["strings"]]
        rank = sorted(range(len(strings)), key=lambda i: strings[i])
        by_bar = collections.defaultdict(list)
        for tick, dur, voice, s, fret, flags in tr["notes"]:
            if set(flags) & SKIP or not 1 <= s <= len(strings):
                continue
            by_bar[max(0, bisect.bisect_right(starts, tick) - 1)].append((tick, strings[s - 1] + fret, rank.index(s - 1)))
        notes, t = [], 0.0
        for b in order:
            start, length, tempo = bars[b][:3]
            rate = 60.0 / (max(tempo, 1) * tabdata.TICKS)
            notes += [(t + (tick - start) * rate, pitch, s) for tick, pitch, s in by_bar.get(b, [])]
            t += length * rate
        out.append((tuple(sorted(strings)), sorted(notes)))
    return out


def _generated(audio):
    """[(kind, events before the tuning is applied)] per part, and the voted notes."""
    a = pipeline.analyse(Path(audio), pipeline.Options(parts=("guitar", "bass")))
    pipeline._correct_onset_delay(a.notes, a.beats)
    bm, shuffle = pipeline.song_grid(a.notes, a.beats, a.bar_starts, a.duration)
    parts = []
    if "guitar" in a.notes:
        for _, gnotes in pipeline.guitar_parts(a.notes["guitar"], a.guitar_groups, bm, a.opts):
            parts.append(("guitar", pipeline._events(gnotes, pipeline._part_grid(bm, gnotes, shuffle), arrange.Report())[2]))
    if "bass" in a.notes:
        parts.append(("bass", pipeline._events(a.notes["bass"], pipeline._part_grid(bm, a.notes["bass"], shuffle), arrange.Report())[2]))
    raw = {k: sorted((n.onset, n.pitch) for n in ns) for k, ns in a.notes.items()}
    return parts, raw


def _items(name, parts, tracks, to_audio, shift):
    """One benchmark item per generated part that the tab has a track for."""
    out = []
    for kind, events in parts:
        truth = [(float(to_audio(t)), p + shift, s, ti) for ti, (_, notes) in enumerate(tracks[kind]) for t, p, s in notes]
        mine = [(e.time, p, ei) for ei, e in enumerate(events) for p in e.pitches]
        m = R.match(truth, mine, lambda x: x[1], lambda y: y[1], 0.1)
        if len(m) < 30:
            continue
        # The tab's track this part mostly is; only its tuning's notes count.
        main = collections.Counter(truth[i][3] for i, _ in m).most_common(1)[0][0]
        strings = tracks[kind][main][0]
        want = {(mine[j][2], mine[j][1]): truth[i][2] for i, j in m if tracks[kind][truth[i][3]][0] == strings}
        out.append({"song": name, "kind": kind, "strings": tuple(p + shift for p in strings),
                    "events": [(e.time, list(e.pitches), list(e.ends)) for e in events], "truth": want})
    return out


def build(refresh=False):
    if CACHE.is_file() and not refresh:
        return pickle.loads(CACHE.read_bytes())
    items = []
    pairs = O.pairs(TB.TEST_FOLDER) if TB.TEST_FOLDER.is_dir() else []
    rows = O.dump([s["tab"] for s in pairs])
    jobs = [(s["title"], s["audio"], rows[str(s["tab"])], None, O.MIN_QUALITY, ("guitar", "bass")) for s in pairs]
    for s in realsongs.songs():
        row = realsongs.best_tab(s["tabs"])
        if row is not None and TB.separated(s["audio"]):  # songs never run before would take minutes each
            jobs.append((f"{s['artist']} - {s['title']}", s["audio"], row, realsongs.play_order(row["bars"]),
                         R.MIN_QUALITY, ("guitar",)))
    for name, audio, row, order, min_quality, kinds in jobs:
        print(f"  {name}", flush=True)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                parts, raw = _generated(audio)
        except (Exception, SystemExit) as e:
            print(f"    failed: {e!r}")
            continue
        tracks = {"guitar": _tracks(row, tabdata.is_guitar, order), "bass": _tracks(row, tabdata.is_bass, order)}
        tab = sorted((t, p) for k in kinds for _, notes in tracks[k] for t, p, _ in notes)
        heard = sorted(x for k in kinds for x in raw.get(k, []))
        if len(tab) < 30 or len(heard) < 30:
            continue
        bars, t = [], 0.0
        for b in (order if order is not None else range(len(row["bars"]))):
            bars.append(t)
            t += row["bars"][b][1] * 60.0 / (max(row["bars"][b][2], 1) * tabdata.TICKS)
        to_audio, shift, quality = R.align(tab, bars, heard)
        if quality < min_quality:
            print(f"    alignment failed ({quality:.2f}): left out")
            continue
        items += _items(name, [p for p in parts if p[0] in kinds], tracks, to_audio, shift)
        CACHE.write_bytes(pickle.dumps(items))
    return items


def evaluate(weights=None, items=None, verbose=False):
    """{kind: (share of notes on the tab's string, mean over songs, notes)}."""
    items = build() if items is None else items
    right = collections.Counter()
    total = collections.Counter()
    for it in items:
        tuning = fretboard.Tuning("tab", it["strings"])
        events = [fretboard.Event(t, list(p), list(e)) for t, p, e in it["events"]]
        fretboard.fit_to_range(events, tuning)
        fing, _ = fretboard.assign_frets(events, tuning, 24, weights)
        fing = arrange.check_playable(events, fing)
        for (ei, pitch), string in it["truth"].items():
            got = dict(zip(fing[ei].pitches, fing[ei].strings)).get(pitch)
            right[(it["kind"], it["song"])] += got == string
            total[(it["kind"], it["song"])] += 1
    out = {}
    for kind in ("guitar", "bass"):
        songs = [k for k in total if k[0] == kind]
        if not songs:
            continue
        n = sum(total[k] for k in songs)
        out[kind] = (sum(right[k] for k in songs) / n, float(np.mean([right[k] / total[k] for k in songs])), n)
        if verbose:
            for k in songs:
                print(f"    {k[1][:40]:40s} {right[k] / total[k]:.3f} ({total[k]} notes)")
        print(f"{kind}: {out[kind][0]:.4f} of {n} notes on the tab's string; mean over {len(songs)} songs {out[kind][1]:.4f}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    evaluate(items=build(a.refresh), verbose=a.verbose)
