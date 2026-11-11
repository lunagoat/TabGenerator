#!/usr/bin/env python3
"""Learn fingering from human-made Guitar Pro tabs.

1. Count chord shapes and string choices in the training tabs
   (fretboard.FingeringPrior).
2. Tune how much those counts weigh against the hand-movement costs, on
   training songs whose counts were left out (so the tuning is honest).
3. Score on held-out songs, and on GuitarSet (other players, other style).
4. Write tabgen/data/fingering_prior.json from all tabs, and print the
   weights to put in FingeringWeights.

    python3 eval/train_fingering.py
"""
import itertools
import json
from dataclasses import replace

import env
import tabdata
from tabgen import fretboard

OUT = env.ROOT / "tabgen" / "data" / "fingering_prior.json"


def build_prior(tracks):
    prior = fretboard.FingeringPrior()
    for tr in tracks:
        for e, t, f in zip(tr.events, tr.truth, tr.frets):
            ps = list(e.pitches)
            prior.add(tr.tuning, ps, [t[p] for p in ps], [f[p] for p in ps])
    return prior


def guitarset_score(w):
    import eval_fingering

    acc, groups = eval_fingering.evaluate(weights=w)
    return acc, groups.get("Rock", 0), groups.get("comp", 0)


def main():
    tracks = tabdata.load()
    train, test = tabdata.split(tracks)
    fit, tune = tabdata.split(train, 0.25, seed="tune")
    print(f"{len(train)} training tracks ({len(fit)} to count, {len(tune)} to tune), {len(test)} held out")

    # The weights before learning from tabs, for comparison.
    base = replace(fretboard.FingeringWeights(), shape_prior=0.0, string_prior=0.0, move=1.5, run_move=2.0, open_anchor=0.3)
    print("without tabs:      held-out %.4f  (single %.4f, chords %.4f)" % tabdata.score(test, base)[:3])

    prior = build_prior(fit)
    best = None
    grid = itertools.product((0.5, 1.0, 1.5, 2.5), (0.0, 0.25, 0.5, 1.0), (0.5, 1.5, 3.0))
    for sp, rp, mv in grid:
        w = replace(base, prior=prior, shape_prior=sp, string_prior=rp, move=mv)
        acc = tabdata.score(tune, w)[0]
        if best is None or acc > best[0]:
            best = (acc, sp, rp, mv)
    print("tuned: shape_prior %.2f string_prior %.2f move %.2f (tuning-set accuracy %.4f)" % (best[1:] + (best[0],)))
    _, sp, rp, mv = best
    # Second round on the remaining movement costs.
    w = replace(base, prior=prior, shape_prior=sp, string_prior=rp, move=mv)
    for name, values in (("run_move", (0.0, 1.0, 2.0, 4.0)), ("string_jump", (0.0, 0.36, 1.0)),
                         ("height", (0.02, 0.05, 0.1)), ("open_anchor", (0.0, 0.3, 0.6))):
        scores = [(tabdata.score(tune, replace(w, **{name: v}))[0], v) for v in values]
        w = replace(w, **{name: max(scores)[1]})
    tuned = {k: getattr(w, k) for k in ("shape_prior", "string_prior", "move", "run_move", "string_jump", "height", "open_anchor")}
    print("tuned weights:", tuned)

    full = build_prior(train)
    final = replace(w, prior=full)
    print("learned (train prior): held-out %.4f  (single %.4f, chords %.4f)" % tabdata.score(test, final)[:3])
    print("GuitarSet: without tabs %.4f / Rock %.4f / comp %.4f" % guitarset_score(base))
    print("GuitarSet: learned %.4f / Rock %.4f / comp %.4f" % guitarset_score(final))

    shipped = build_prior(tracks).pruned(2)
    known = {t.strings: t.name for t in fretboard.GUITAR_TUNINGS}
    for tr in tracks:
        name = known.get(tr.tuning.strings)
        if name:
            shipped.tunings[name] = shipped.tunings.get(name, 0) + 1
    OUT.parent.mkdir(exist_ok=True)
    keep = json.loads(OUT.read_text()) if OUT.is_file() else {}  # e.g. the tempo-level model
    OUT.write_text(json.dumps({**keep, **shipped.to_json()}, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
