#!/usr/bin/env python3
"""Run transcription passes on a dataset and print per-pass F1 per part.

    python3 eval/run_band.py babyslakh "demucs|guitar|any|0" ...
"""
import sys

import env  # noqa
import datasets
from harness import mean_prf, part_notes, prf, run_pass, run_passes, stems_for

items = getattr(datasets, sys.argv[1])()
# Separate everything first, while the GPU is otherwise empty.
for sep in sorted({s.split("|")[0] for s in sys.argv[2:]} - {"none"}):
    for it in items:
        stems_for(it, sep)
for it in items:
    run_passes(it, sys.argv[2:])
    print(".", end="", file=sys.stderr, flush=True)
for spec in sys.argv[2:]:
    line = f"{spec:44s}"
    for part in ("guitar", "bass", "drums"):
        rows = [prf(part_notes(run_pass(it, spec), part), it.truth[part]) for it in items if part in it.truth]
        if rows:
            p, r, f = mean_prf(rows)
            line += f" | {part} P {p:.3f} R {r:.3f} F1 {f:.3f}"
    print(line, flush=True)
