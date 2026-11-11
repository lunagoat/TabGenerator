#!/usr/bin/env python3
"""How much does the learned string model help the fret chooser?

Given the right notes, share of notes put on the tab's string, for several
weights of the model against the existing costs, on: your held-out tabs,
the Official Ultimate Guitar tabs and GuitarSet.

    python3 eval/eval_fingering_model.py
"""
from dataclasses import replace

import env
import numpy as np

import eval_fingering
import tabdata
from tabgen import fretboard


def main():
    user_test = tabdata.split(tabdata.load())[1]
    official = tabdata.load([env.DATA / "official" / "ug.jsonl"])
    base = fretboard.FingeringWeights()
    print(f"{len(user_test)} held-out tracks of yours, {len(official)} Official guitar tracks")
    rows = []
    for mw, prior_scale in ((0.0, 1.0), (0.5, 1.0), (1.0, 1.0), (2.0, 1.0), (1.0, 0.5), (2.0, 0.5), (2.0, 0.0), (4.0, 0.0)):
        w = replace(base, model_weight=mw, shape_prior=base.shape_prior * prior_scale, string_prior=base.string_prior * prior_scale)
        u = tabdata.score(user_test, w)[0]
        o = tabdata.score(official, w)[0]
        g = eval_fingering.evaluate(weights=w)[0]
        rows.append((mw, prior_scale, u, o, g))
        print(f"model {mw:3.1f}, counted priors x{prior_scale:.1f}: yours {u:.4f} | Official {o:.4f} | GuitarSet {g:.4f}", flush=True)
    best = max(rows, key=lambda r: r[2] + r[3])
    print("best by yours + Official:", best)


if __name__ == "__main__":
    main()
