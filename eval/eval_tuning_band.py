#!/usr/bin/env python3
"""Tuning detection as the pipeline does it, on every real recording that
has a tab (the Official tabs of official.py and the songs of realeval.py).

Unlike eval_tuning_real.py this goes through the pipeline's own steps
(cached transcriptions -> beat grid -> quantized chords -> band tuning), for
the guitar and the bass together.

    python3 eval/eval_tuning_band.py [--refresh] [--all] [-v]

The events are cached in eval/work/tuning_band.pkl; --refresh rebuilds them
(needed after changing transcription, voting or quantizing). Songs that
were never run through TabGenerator are left out unless --all is given
(each takes minutes of GPU time).
"""
import argparse
import contextlib
import io
import json
import pickle
from pathlib import Path

import env
import official as O
import realsongs
import tabdata
from tabgen import arrange, fretboard, pipeline

CACHE = env.WORK / "tuning_band.pkl"
TEST_FOLDER = Path("/home/luna/Desktop/Test tabs and flacs")


def _names(tracks, is_kind, table):
    known = {t.strings: t.name for t in table}
    out = {known.get(tuple(sorted(p + (t.get("offset") or 0) for p in t["strings"]))) for t in tracks if is_kind(t)}
    return sorted(out - {None})


def songs():
    """(name, audio path, guitar tunings of the tab, bass tunings of the tab)."""
    out = []
    if TEST_FOLDER.is_dir():
        pairs = O.pairs(TEST_FOLDER)
        rows = O.dump([s["tab"] for s in pairs])
        for s in pairs:
            tr = rows[str(s["tab"])]["tracks"]
            out.append((s["title"], str(s["audio"]), _names(tr, tabdata.is_guitar, fretboard.GUITAR_TUNINGS),
                        _names(tr, tabdata.is_bass, fretboard.BASS_TUNINGS)))
    by = {json.loads(l)["path"]: json.loads(l) for l in open(env.DATA / "tabs" / "user.jsonl", encoding="utf-8")}
    for s in realsongs.songs():
        tr = [t for p in s["tabs"] if p in by for t in by[p]["tracks"]]
        out.append((f"{s['artist']} - {s['title']}", s["audio"], _names(tr, tabdata.is_guitar, fretboard.GUITAR_TUNINGS),
                    _names(tr, tabdata.is_bass, fretboard.BASS_TUNINGS)))
    return out


def separated(audio) -> bool:
    """Whether the song's stems are in .cache (it was run before)."""
    return (env.ROOT / ".cache" / Path(audio).stem / "stems_bs_roformer_sw" / "bass.wav").is_file()


def build(refresh=False, everything=False):
    """Per song: the guitar and bass chords the tuning is chosen from."""
    done = {}
    if CACHE.is_file() and not refresh:
        done = pickle.loads(CACHE.read_bytes())
    for name, audio, gt, bt in songs():
        if name in done or not (everything or separated(audio)):
            continue
        print(f"  reading {name}", flush=True)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                a = pipeline.analyse(Path(audio), pipeline.Options(parts=("guitar", "bass")))
                pipeline._correct_onset_delay(a.notes, a.beats)
                bm, shuffle = pipeline.song_grid(a.notes, a.beats, a.bar_starts, a.duration)
                ev = {}
                for part in ("guitar", "bass"):
                    if part in a.notes:
                        part_bm = pipeline._part_grid(bm, a.notes[part], shuffle)
                        ev[part] = pipeline._events(a.notes[part], part_bm, arrange.Report())[2]
        except (Exception, SystemExit) as e:
            print(f"    failed: {e!r}")
            continue
        done[name] = ev
        CACHE.write_bytes(pickle.dumps(done))
    return done


def copy(events):
    return [fretboard.Event(e.time, list(e.pitches), list(e.ends)) for e in events]


def evaluate(choose=None, verbose=False):
    """`choose(guitar events or None, bass events or None)` -> (guitar tuning
    name, bass tuning name); default: the pipeline's."""
    choose = choose or (lambda g, b: tuple(t.name if t else None for t in fretboard.choose_band_tuning(g, b)))
    data = build()
    right = {"guitar": 0, "bass": 0}
    total = {"guitar": 0, "bass": 0}
    for name, _, gt, bt in songs():
        if name not in data:
            continue
        ev = data[name]
        g, b = choose(copy(ev["guitar"]) if "guitar" in ev else None, copy(ev["bass"]) if "bass" in ev else None)
        for part, got, truth in (("guitar", g, gt), ("bass", b, bt)):
            if len(truth) != 1 or got is None:
                continue  # no such track, a tuning we don't know, or tracks that disagree
            total[part] += 1
            right[part] += got == truth[0]
            if verbose and got != truth[0]:
                print(f"  {name[:42]:42s} {part:6s} got {got}, tab {truth[0]}")
    for part in right:
        print(f"{part}: right on {right[part]}/{total[part]} songs")
    return right, total


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--all", action="store_true", help="also run songs that were never run before")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    build(a.refresh, a.all)
    evaluate(verbose=a.verbose)
