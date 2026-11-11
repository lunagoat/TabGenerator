#!/usr/bin/env python3
"""Tuning detection on real recordings: the transcribed guitar notes of the
songs scored by realeval.py and official.py, against their tabs' tunings."""
import json
from pathlib import Path

import env
import official as O
import realeval as R
import realsongs
import tabdata
from tabgen import fretboard

KNOWN = {t.strings: t.name for t in fretboard.GUITAR_TUNINGS}


def cases():
    out = []
    folder = Path("/home/luna/Desktop/Test tabs and flacs")
    if folder.is_dir():
        songs = O.pairs(folder)
        rows = O.dump([s["tab"] for s in songs])
        for s in songs:
            f = O.OUT / R.slug(s["title"]) / "result.json"
            if not f.is_file():
                continue
            row = rows[str(s["tab"])]
            tun = {KNOWN.get(tuple(sorted(p + (t.get("offset") or 0) for p in t["strings"]))) for t in row["tracks"] if tabdata.is_guitar(t)}
            out.append((s["title"], json.loads(f.read_text())["notes"]["guitar"], tun - {None}))
    by = {json.loads(l)["path"]: json.loads(l) for l in open(env.DATA / "tabs" / "user.jsonl", encoding="utf-8")}
    for s in realsongs.songs():
        name = f"{s['artist']} - {s['title']}"
        f = R.OUT / R.slug(name) / "result.json"
        if not f.is_file():
            continue
        tun = {KNOWN.get(tuple(sorted(p_ + (t.get("offset") or 0) for p_ in t["strings"]))) for p in s["tabs"] if p in by for t in by[p]["tracks"] if tabdata.is_guitar(t)}
        out.append((name, json.loads(f.read_text())["raw"], tun - {None}))
    return [c for c in out if len(c[2]) == 1]


def evaluate(verbose=True):
    right = 0
    cs = cases()
    for name, notes, tun in cs:
        ev = [fretboard.Event(t, [p], [t + 0.2]) for t, p in sorted(notes)]
        got = fretboard.choose_tuning(ev, False).name
        right += got in tun
        if verbose and got not in tun:
            print(f"  {name[:40]:40s} got {got}, tab {sorted(tun)}")
    print(f"right on {right}/{len(cs)} songs")
    return right / len(cs)


if __name__ == "__main__":
    evaluate()
