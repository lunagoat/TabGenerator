#!/usr/bin/env python3
"""Tuning detection on human-made tabs: given only a track's pitches, is the
tab's own tuning picked? Tracks in tunings TabGenerator doesn't know (open
G, ...) are left out.

    python3 eval/eval_tuning.py
"""
import collections
import copy

import env
import tabdata
from tabgen import fretboard

KNOWN = {t.strings: t.name for t in fretboard.GUITAR_TUNINGS}


def tracks():
    out = []
    for tr in tabdata.load():
        name = KNOWN.get(tr.tuning.strings)
        if name:
            out.append((tr, name))
    return out


def evaluate(data, choose, verbose=False):
    right = 0
    confusion = collections.Counter()
    per = collections.defaultdict(lambda: [0, 0])
    for tr, name in data:
        events = [fretboard.Event(e.time, list(e.pitches), list(e.ends)) for e in tr.events]
        got = choose(events).name
        right += got == name
        per[name][0] += got == name
        per[name][1] += 1
        if got != name:
            confusion[(name, got)] += 1
    if verbose:
        print("  per tuning:", {k: f"{a}/{b}" for k, (a, b) in sorted(per.items(), key=lambda x: -x[1][1])})
        print("  most common mistakes:", confusion.most_common(6))
    return right / max(1, len(data))


if __name__ == "__main__":
    data = tracks()
    train = [d for d in data if d[0] in tabdata.split([d[0] for d in data])[0]]
    print(len(data), "tracks;", collections.Counter(n for _, n in data).most_common())
    acc = evaluate(data, lambda ev: fretboard.choose_tuning(ev, False), verbose=True)
    print(f"current choose_tuning: {acc:.3f}")


def chooser(freq: dict, beta: float, slack: int = 0, max_events: int = 1500):
    """Every tuning the part fits on competes; fingering cost per note (with
    the learned shape/string prior) plus beta * -log(how common the tuning is)."""
    import math

    total = sum(freq.values())

    def choose(events):
        low = fretboard.robust_lowest(fretboard.credible_low_pitches(events))
        table = fretboard.GUITAR_TUNINGS
        if low is None:
            return table[0]
        fits = [t for t in table if t.lowest <= low + slack] or [min(table, key=lambda t: t.lowest)]
        ev = events[:max_events]
        n = max(1, sum(len(e.pitches) for e in ev))
        best = None
        for t in fits:
            _, cost = fretboard.assign_frets(copy.deepcopy(ev), t)
            score = cost / n + beta * -math.log((freq.get(t.name, 0) + 1) / (total + len(table)))
            if best is None or score < best[0]:
                best = (score, t)
        return best[1]

    return choose


def search():
    data = tracks()
    trs = [d[0] for d in data]
    train_ids = {id(t) for t in tabdata.split(trs)[0]}
    train = [d for d in data if id(d[0]) in train_ids]
    test = [d for d in data if id(d[0]) not in train_ids]
    freq = collections.Counter(n for _, n in train)
    print("current, held-out: %.3f" % evaluate(test, lambda ev: fretboard.choose_tuning(ev, False)))
    best = None
    for beta in (0.0, 0.05, 0.1, 0.2, 0.4):
        acc = evaluate(train, chooser(freq, beta))
        print(f"  beta {beta}: train {acc:.3f}", flush=True)
        if best is None or acc > best[0]:
            best = (acc, beta)
    print("best beta", best[1])
    print("held-out: %.3f" % evaluate(test, chooser(freq, best[1]), verbose=True))
    return best[1]
