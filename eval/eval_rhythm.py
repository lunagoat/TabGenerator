#!/usr/bin/env python3
"""Does quantization put notes on the right place in the beat?

BabySlakh's MIDI files give each note's true position in its beat. Notes are
given the timing error a transcription has (default: 15 ms jitter), snapped
by rhythm.choose_subdivisions + rhythm.quantize on the true beats, and
compared with the truth. Only sequenced tracks (notes on a 16th/triplet
grid) are used.
"""
import sys

import env  # noqa
import mido
import numpy as np

from tabgen import rhythm
from tabgen.transcribe import TNote


def load_track(track, max_s=None):
    mid = mido.MidiFile(str(track / "all_src.mid"))
    tpb = mid.ticks_per_beat
    tempo_ev = [(0, 500000)]
    notes = []  # (tick, is_drum, pitch)
    sig = (4, 4)
    for tr in mid.tracks:
        tick = 0
        for msg in tr:
            tick += msg.time
            if msg.type == "set_tempo":
                tempo_ev.append((tick, msg.tempo))
            elif msg.type == "time_signature" and tick == 0:
                sig = (msg.numerator, msg.denominator)
            elif msg.type == "note_on" and msg.velocity > 0:
                notes.append((tick, msg.channel == 9, msg.note))
    tempo_ev.sort()
    ticks = np.array([t for t, _ in tempo_ev])
    secs = np.zeros(len(tempo_ev))
    for i in range(1, len(tempo_ev)):
        secs[i] = secs[i - 1] + (ticks[i] - ticks[i - 1]) * tempo_ev[i - 1][1] / tpb / 1e6

    def to_s(tick):
        i = np.searchsorted(ticks, tick, side="right") - 1
        return secs[i] + (tick - ticks[i]) * tempo_ev[i][1] / tpb / 1e6

    return tpb, sig, notes, to_s


def tracks():
    out = []
    for track in sorted((env.DATA / "babyslakh_16k").glob("Track*")):
        tpb, sig, notes, to_s = load_track(track)
        if sig[1] != 4 or not notes:
            continue
        t = np.array([n[0] for n in notes])
        frac = (t % tpb) / tpb
        on_grid = np.minimum(np.abs(frac * 12 - np.round(frac * 12)), np.abs(frac * 8 - np.round(frac * 8))) < 0.02
        if on_grid.mean() < 0.85:
            continue
        keep = on_grid
        last = int(t.max() // tpb) + 2
        beats = np.array([to_s(b * tpb) for b in range(last + 1)])
        out.append((track.name, sig[0], beats, t[keep] / tpb, np.array([to_s(x) for x in t[keep]]),
                    np.array([n[1] for n in notes])[keep], np.array([n[2] for n in notes])[keep]))
    return out


def evaluate(jitter=0.015, seed=0, verbose=False, data=None):
    rng = np.random.default_rng(seed)
    total = right = 0
    trip_true = trip_pred = 0
    for name, meter, beats, pos, secs, is_drum, pitch in data or tracks():
        noisy = secs + rng.normal(0, jitter, len(secs))
        notes = [TNote(float(s), float(s) + 0.2, int(p), "x", bool(d)) for s, p, d in zip(noisy, pitch, is_drum)]
        bm = rhythm.build_beatmap(beats, np.arange(0, len(beats), meter), float(beats[-1]))
        shift = bm.time_to_beat(beats[0])  # beats added in front
        if rhythm.detect_compound(bm, noisy):
            bm.compound = True
        bm.subdivisions = rhythm.choose_subdivisions(bm, noisy)
        q = {}
        for n, p in zip(notes, pos):
            on = rhythm.quantize(bm, [n], bm.subdivisions)[0].onset_tick
            q[id(n)] = (on / bm.beat_ticks - shift, p)
        ok = sum(abs(a - b) < 1 / 48 for a, b in q.values())
        total += len(q)
        right += ok
        frac = pos - np.floor(pos)
        true_trip = np.abs(frac * 4 - np.round(frac * 4)) > 0.02
        if verbose:
            subs = bm.subdivisions[: len(beats)]
            print(f"{name}: {ok / len(q):.3f} of {len(q)}  true triplet notes {true_trip.mean():.3f}  "
                  f"triplet beats chosen {np.mean(np.isin(subs, (3, 6, 12))):.3f} compound={bm.compound}")
    return right / max(1, total)


if __name__ == "__main__":
    data = tracks()
    print(len(data), "tracks")
    for j in (0.005, 0.015, 0.025):
        print(f"jitter {j * 1000:.0f} ms: {evaluate(j, data=data, verbose='-v' in sys.argv and j == 0.015):.4f}")
