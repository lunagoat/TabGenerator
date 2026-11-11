#!/usr/bin/env python3
"""Does tidying (arrange.py) cost note accuracy?

    python3 eval/eval_tidy.py [gsband_dist gsband_clean]   guitar F1 on the band mixes
    python3 eval/eval_tidy.py official FOLDER              guitar and bass F1 against Official tabs
                                                           (after eval/official.py has run on FOLDER)

Both score the voted notes, the quantized notes and the notes after each
tidying step.
"""
import contextlib
import io
import json
import sys
from pathlib import Path

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


def official(folder):
    """Per stage, mean P/R/F1 (onsets within 100 ms) over the aligned songs."""
    import official as O
    import realeval as R

    songs = O.pairs(Path(folder))
    tabs = O.dump([s["tab"] for s in songs])
    rows: dict = {}
    for s in songs:
        cached = O.OUT / R.slug(s["title"]) / "result.json"
        if not cached.is_file():
            continue
        res = json.loads(cached.read_text())
        parts, bar_secs, _ = O.tab_parts(tabs[str(s["tab"])])
        tab = sorted(n for k in ("guitar", "bass") for n in parts.get(k, []))
        heard = sorted(x for k in ("guitar", "bass") for x in res["notes"].get(k, []))
        to_audio, shift, quality = R.align(tab, bar_secs, heard)
        if quality < O.MIN_QUALITY:
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            a = pipeline.analyse(Path(s["audio"]), pipeline.Options())
            pipeline._correct_onset_delay(a.notes, a.beats)
            bm, shuffle = pipeline.song_grid(a.notes, a.beats, a.bar_starts, a.duration)
            groups = [("guitar", g) for _, g in pipeline.guitar_parts(a.notes["guitar"], a.guitar_groups, bm, a.opts)]
        groups.append(("bass", a.notes["bass"]))
        out: dict = {}
        for kind, notes in groups:
            pbm = pipeline._part_grid(bm, notes, shuffle)
            q = rhythm.quantize(pbm, notes, pbm.subdivisions)
            o = arrange.remove_overtones(q)
            stages = {"voted": [(n.onset, n.pitch) for n in notes], "quantized": q, "overtones removed": o,
                      "repeats unified": arrange.unify_repeats(o, pbm)}
            for name, st in stages.items():
                if name != "voted":
                    st = [(float(pbm.beat_to_time(x.onset_tick / pbm.beat_ticks)), x.pitch) for x in st]
                out.setdefault((kind, name), []).extend(st)
        for (kind, name), pred in out.items():
            truth = sorted({(float(to_audio(n[0])), n[1] + shift) for n in parts[kind]})
            rows.setdefault((kind, name), []).append(O.prf(truth, sorted(pred), 0.1)[:3])
    for (kind, name), v in rows.items():
        p, r, f = np.mean(v, axis=0)
        print(f"{kind:6s} {name:18s} P {p:.3f} R {r:.3f} F1 {f:.3f}  ({len(v)} songs)")


if __name__ == "__main__":
    if sys.argv[1:2] == ["official"]:
        official(sys.argv[2])
        sys.exit()
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
            t = arrange.unify_repeats(t, bm, rep)
            rows["voted"].append(prf(notes, it.truth["guitar"]))
            rows["quantized"].append(prf(as_notes(q), it.truth["guitar"]))
            rows["tidied"].append(prf(as_notes(t), it.truth["guitar"]))
        print(ds, "  ".join(f"{k}: P {p:.3f} R {r:.3f} F1 {f:.3f}" for k, (p, r, f) in ((k, mean_prf(v)) for k, v in rows.items())))
        print("   ", rep)
