#!/usr/bin/env python3
"""Which tempo level do people write a song at? Learned from Guitar Pro tabs.

A beat tracker can lock onto the beat, twice the beat or half of it; all
three fit the audio. Tabbers pick the one where the tempo is ordinary and
the notes are mostly 8ths and 16ths (not 32nds). For every tab track the
written level is compared with the same notes written at double and at half
the tempo, and a logistic model learns to prefer the written one. The model
is saved into tabgen/data/fingering_prior.json under "tempo_level".

    python3 eval/tempo_level.py
"""
import json

import env
import numpy as np

import tabdata
from tabgen import rhythm

PRIOR = env.ROOT / "tabgen" / "data" / "fingering_prior.json"


def tab_examples():
    """(quarter-note positions of the onsets, bpm) for tracks in simple meter, one tempo."""
    out = []
    for p in sorted((env.DATA / "tabs").glob("*.jsonl")):
        for line in open(p, encoding="utf-8"):
            song = json.loads(line)
            bars = song["bars"]
            tempos = {b[2] for b in bars}
            if len(tempos) != 1 or any(b[4] != 4 for b in bars):
                continue
            bpm = tempos.pop()
            for t in song["tracks"]:
                if t["percussion"] or t["name"].startswith("@$"):
                    continue
                ticks = sorted({n[0] for n in t["notes"] if "t" not in n[5]})
                if len(ticks) >= 64:
                    out.append((np.array(ticks) / tabdata.TICKS, float(bpm), song["path"]))
    return out


def main():
    ex = tab_examples()
    print(len(ex), "tracks")
    X, y = [], []
    for q, bpm, _ in ex:
        f1 = rhythm.level_features(q, bpm)
        for k in (2.0, 0.5):  # written twice as fast / half as fast
            fk = rhythm.level_features(q * k, bpm * k)
            X += [f1 - fk, fk - f1]
            y += [1, 0]
    X, y = np.array(X), np.array(y)
    # Logistic regression by gradient descent (no bias: the comparison is symmetric).
    w = np.zeros(X.shape[1])
    for _ in range(4000):
        p = 1 / (1 + np.exp(-X @ w))
        w -= 0.5 * (X.T @ (p - y) / len(y) + 1e-3 * w)
    acc = np.mean((X @ w > 0) == (y == 1))
    print("pairwise accuracy (written level beats the alternative):", round(float(acc), 4))
    # Held-out check by song: retrain without each fifth.
    songs = sorted({e[2] for e in ex})
    right = total = 0
    for fold in range(5):
        test = set(songs[fold::5])
        Xtr, ytr, Xte, yte = [], [], [], []
        for (q, bpm, path) in ex:
            f1 = rhythm.level_features(q, bpm)
            for k in (2.0, 0.5):
                fk = rhythm.level_features(q * k, bpm * k)
                (Xte if path in test else Xtr).extend([f1 - fk, fk - f1])
                (yte if path in test else ytr).extend([1, 0])
        Xtr, ytr, Xte, yte = map(np.array, (Xtr, ytr, Xte, yte))
        wf = np.zeros(X.shape[1])
        for _ in range(4000):
            p = 1 / (1 + np.exp(-Xtr @ wf))
            wf -= 0.5 * (Xtr.T @ (p - ytr) / len(ytr) + 1e-3 * wf)
        right += np.sum((Xte @ wf > 0) == (yte == 1))
        total += len(yte)
    print("held-out pairwise accuracy:", round(right / total, 4))
    d = json.loads(PRIOR.read_text())
    d["tempo_level"] = [round(float(v), 5) for v in w]
    PRIOR.write_text(json.dumps(d, separators=(",", ":")))
    print("weights", d["tempo_level"])


if __name__ == "__main__":
    main()
