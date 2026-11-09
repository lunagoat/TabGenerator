#!/usr/bin/env python3
"""String/fret choice accuracy on GuitarSet: given the true notes, how often
does fretboard.assign_frets put a note on the string the player used?

    python3 eval/eval_fingering.py [--style solo|comp] [--limit N]
"""
import argparse
import json

import env  # noqa: F401
import numpy as np

from tabgen import fretboard

ANNOT = env.DATA / "guitarset" / "annotation"
E_STD = fretboard.GUITAR_TUNINGS[0]


def load_notes(path):
    """[(onset, offset, pitch, string)] with string 0 = low E."""
    jam = json.loads(path.read_text())
    notes = []
    for ann in jam["annotations"]:
        if ann["namespace"] != "note_midi":
            continue
        string = int(ann["annotation_metadata"]["data_source"])
        for d in ann["data"]:
            pitch = int(round(d["value"]))
            # A bent or mistracked note can round outside the string's range.
            fret = pitch - E_STD.strings[string]
            if 0 <= fret <= 24:
                notes.append((d["time"], d["time"] + d["duration"], pitch, string))
    notes.sort()
    return notes


def to_events(notes, window=0.03):
    """Group notes starting together, like the pipeline does after quantizing."""
    events, truth = [], []
    for on, off, pitch, string in notes:
        if events and on - events[-1].time < window and pitch not in events[-1].pitches:
            events[-1].pitches.append(pitch)
            events[-1].ends.append(off)
            truth[-1][pitch] = string
        elif events and on - events[-1].time < window:
            continue  # same pitch twice at once (unison on two strings): skip
        else:
            events.append(fretboard.Event(time=on, pitches=[pitch], ends=[off]))
            truth.append({pitch: string})
    for e in events:
        order = np.argsort(e.pitches)
        e.pitches = [e.pitches[i] for i in order]
        e.ends = [e.ends[i] for i in order]
    return events, truth


def score_file(path, weights=None):
    events, truth = to_events(load_notes(path))
    fingerings, _ = fretboard.assign_frets(events, E_STD, 24, weights)
    right = total = 0
    for f, t in zip(fingerings, truth):
        got = dict(zip(f.pitches, f.strings))
        for pitch, string in t.items():
            total += 1
            right += got.get(pitch) == string
    return right, total


def evaluate(weights=None, style=None, limit=None, verbose=False):
    """Returns (overall accuracy, {group: accuracy}) with groups = player ids
    and playing style (comp/solo)."""
    files = sorted(ANNOT.glob("*.jams"))
    if style:
        files = [f for f in files if f.stem.endswith(style)]
    if limit:
        files = files[:: max(1, len(files) // limit)]
    groups: dict[str, list[int]] = {}
    R = T = 0
    for f in files:
        r, t = score_file(f, weights)
        R += r
        T += t
        for g in (f.stem[:2], f.stem.rsplit("_", 1)[1], f.stem.split("_")[1].split("-")[0].rstrip("0123456789")):
            a = groups.setdefault(g, [0, 0])
            a[0] += r
            a[1] += t
        if verbose:
            print(f"{f.stem:32s} {r / max(1, t):.3f}  ({t} notes)")
    return R / max(1, T), {g: a[0] / max(1, a[1]) for g, a in sorted(groups.items())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--style")
    ap.add_argument("--limit", type=int)
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    acc, groups = evaluate(style=a.style, limit=a.limit, verbose=a.v)
    print("  ".join(f"{g}={v:.3f}" for g, v in groups.items()))
    print(f"string accuracy: {acc:.4f}")
