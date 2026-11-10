#!/usr/bin/env python3
"""Does tidying (arrange.py) cost note accuracy? Guitar F1 on the band
mixes, on the quantized notes before and after tidying."""
import json
import sys

import env  # noqa
import numpy as np

import datasets
from harness import prf, mean_prf, part_notes, run_passes
from tabgen import arrange, ensemble, pipeline, rhythm
from tabgen.transcribe import TNote

SPECS = ["rofo|guitar|any|0", "rofo|guitar|any|2.5", "rofo|guitar|any|3.75"]


def beatmap(item):
    f = item.work / "beats_v2.json"
    if f.is_file():
        b = json.loads(f.read_text())
        beats, bars = np.array(b["beats"]), np.array(b["bar_starts"], dtype=int)
    else:
        beats, bars = rhythm.track_beats(pipeline.load_mono(item.wav, 22050), 22050, "cuda")
        f.write_text(json.dumps({"beats": beats.tolist(), "bar_starts": bars.tolist()}))
    import soundfile as sf

    return rhythm.build_beatmap(beats, bars, sf.info(str(item.wav)).duration)


def as_notes(q):
    return [TNote(x.onset, x.offset, x.pitch, x.instrument, False) for x in q]


if __name__ == "__main__":
    for ds in sys.argv[1:] or ["gsband_dist", "gsband_clean"]:
        rows = {"voted": [], "quantized": [], "tidied": []}
        rep = arrange.Report()
        for it in getattr(datasets, ds)():
            res = run_passes(it, SPECS)
            notes = ensemble.merge([part_notes(res[s], "guitar") for s in SPECS], 2)
            bm = beatmap(it)
            bm.subdivisions = rhythm.choose_subdivisions(bm, np.array([n.onset for n in notes]))
            q = rhythm.quantize(bm, notes, bm.subdivisions)
            t = arrange.remove_overtones(q, rep)
            t = arrange.regularize_chords(t, rep)
            t = arrange.unify_repeats(t, bm, rep)
            rows["voted"].append(prf(notes, it.truth["guitar"]))
            rows["quantized"].append(prf(as_notes(q), it.truth["guitar"]))
            rows["tidied"].append(prf(as_notes(t), it.truth["guitar"]))
        print(ds, "  ".join(f"{k}: P {p:.3f} R {r:.3f} F1 {f:.3f}" for k, (p, r, f) in ((k, mean_prf(v)) for k, v in rows.items())))
        print("   ", rep)
