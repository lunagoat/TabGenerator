#!/usr/bin/env python3
"""Check files for the real-song evaluation, to hear where a tab and a
recording disagree. For each song, two stereo files:

  <song> - human tab.flac   left: the guitar stem   right: the aligned human tab, synthesized
  <song> - generated.flac   left: the guitar stem   right: TabGenerator's notes, synthesized

Both cover the part of the recording the tab was aligned to.

    python3 eval/listen.py "Dam That River" "Glycerine" ...
"""
import json
import sys
from pathlib import Path

import env
import numpy as np
import soundfile as sf

import realeval as R
import realsongs

SR = 44100
OUT = env.WORK / "listen"


def synth(notes, length):
    """Plucked-string-ish tones: a few decaying harmonics per note."""
    out = np.zeros(length)
    dur = int(0.45 * SR)
    t = np.arange(dur) / SR
    env_ = np.exp(-t * 7.0) * np.minimum(1, t * 400)
    cache = {}
    for onset, pitch in notes:
        i = int(onset * SR)
        if i < 0 or i >= length:
            continue
        if pitch not in cache:
            f = 440.0 * 2 ** ((pitch - 69) / 12)
            cache[pitch] = sum(np.sin(2 * np.pi * f * k * t) / k ** 1.3 for k in range(1, 6) if f * k < SR / 2) * env_
        tone = cache[pitch]
        j = min(length, i + dur)
        out[i:j] += tone[: j - i]
    return out


def render(song):
    name = f"{song['artist']} - {song['title']}"
    res = json.loads((R.OUT / R.slug(name) / "result.json").read_text())
    tab, bars = realsongs.tab_notes(song["tabs"])
    tab = [n for n in tab if not set(n[4]) & set("txhc")]
    to_audio, shift, q = R.align(tab, bars, res["raw"])
    truth = [(float(to_audio(n[0])), n[1] + shift) for n in tab]
    lo, hi = max(0.0, min(t for t, _ in truth) - 1), max(t for t, _ in truth) + 2
    stem = env.ROOT / ".cache" / Path(song["audio"]).stem / "stems_bs_roformer_sw" / "guitar.wav"
    audio, sr = sf.read(str(stem), dtype="float64", always_2d=True)
    assert sr == SR
    a, b = int(lo * SR), min(len(audio), int(hi * SR))
    left = audio[a:b].mean(axis=1)
    left /= np.abs(left).max() + 1e-9
    OUT.mkdir(parents=True, exist_ok=True)
    for label, notes in (("human tab", truth), ("generated", [(g[0], g[1]) for g in res["tab"]])):
        right = synth([(t - lo, p) for t, p in notes if lo <= t < hi], b - a)
        right /= np.abs(right).max() + 1e-9
        path = OUT / f"{R.slug(name)} - {label}.flac"
        sf.write(str(path), np.stack([0.8 * left, 0.6 * right], axis=1), SR)
        print(path)
    print(f"   aligned section: {lo:.0f}-{hi:.0f} s of the recording (alignment score {q:.2f})")


if __name__ == "__main__":
    wanted = [w.lower() for w in sys.argv[1:]]
    for s in realsongs.songs():
        name = f"{s['artist']} - {s['title']}"
        if any(w in name.lower() for w in wanted) and (R.OUT / R.slug(name) / "result.json").is_file():
            render(s)
