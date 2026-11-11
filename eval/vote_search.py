#!/usr/bin/env python3
"""Which passes should vote, and how many votes should a note need?

    python3 eval/vote_search.py <dataset> <part> [spec ...]

Prints every single pass, then a greedy forward selection of voters (for
each ensemble size, the best min_votes), using the cached transcriptions.
"""
import sys

import env  # noqa

import datasets
from harness import mean_prf, part_notes, prf, run_passes
from tabgen import ensemble


def load(dataset, part, specs=None):
    items = [it for it in getattr(datasets, dataset)() if part in it.truth]
    if not specs:
        names = None
        for it in items:
            have = {p.name[len("notes_"):-len(".h.json")] for p in it.work.glob("notes_*.h.json")}
            names = have if names is None else names & have
        specs = sorted("|".join(n.rsplit("_", 2)[:1] + n.rsplit("_", 2)[1:]).replace("_", "|", 1) for n in names)
    data = []
    for it in items:
        res = run_passes(it, specs)
        data.append((it, {s: part_notes(res[s], part) for s in specs}))
    return data, specs


def score(data, part, specs, min_votes, post=None, fn=ensemble.merge, **kw):
    rows = []
    for it, res in data:
        notes = fn([res[s] for s in specs], min_votes, **kw) if len(specs) > 1 or min_votes > 1 else list(res[specs[0]])
        if post:
            notes = post(notes)
        rows.append(prf(notes, it.truth[part]))
    return mean_prf(rows)


def greedy(data, part, specs, post=None, max_size=8):
    chosen = []
    best_overall = None
    while len(chosen) < min(max_size, len(specs)):
        best = None
        for s in specs:
            if s in chosen:
                continue
            for mv in range(1, len(chosen) + 2):
                f = score(data, part, chosen + [s], mv, post)[2]
                if best is None or f > best[0]:
                    best = (f, s, mv)
        chosen.append(best[1])
        p, r, f = score(data, part, chosen, best[2], post)
        print(f"  {len(chosen)} voters, min_votes {best[2]}: P {p:.3f} R {r:.3f} F1 {f:.3f}   + {best[1]}", flush=True)
    return chosen


if __name__ == "__main__":
    dataset, part = sys.argv[1], sys.argv[2]
    data, specs = load(dataset, part, sys.argv[3:])
    post = ensemble.lowest_only if part == "bass" else None
    for s in specs:
        p, r, f = score(data, part, [s], 1, post)
        print(f"{s:44s} P {p:.3f} R {r:.3f} F1 {f:.3f}")
    greedy(data, part, specs, post)
