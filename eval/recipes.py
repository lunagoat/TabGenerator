#!/usr/bin/env python3
"""Score named voting recipes for a part on several datasets."""
import sys

import env  # noqa
import vote_search as V
from tabgen import ensemble

GA = "rofo|guitar|any|"
GG = "rofo|guitar|guitars|"
GOP = "rofo|guitar+other+piano|guitars|"
BAND = "rofo|guitar+other+piano+bass|any|"
MIX = "none|mix|any|"
RECIPES = {
    "guitar": {
        "old (demucs)": (["demucs|guitar+other+piano+bass|any|0", "demucs|guitar+other+piano|guitars|0", "demucs|guitar|any|0", MIX + "0"], 2),
        "old recipe, rofo": ([BAND + "0", GOP + "0", GA + "0", MIX + "0"], 2),
        "GA0": ([GA + "0"], 1),
        "GA0 GA2.5 mv2": ([GA + "0", GA + "2.5"], 2),
        "GA0 GA2.5 mv1": ([GA + "0", GA + "2.5"], 1),
        "GA0 GA2.5 GG0 mv2": ([GA + "0", GA + "2.5", GG + "0"], 2),
        "GA0 GA2.5 GOP0 mv2": ([GA + "0", GA + "2.5", GOP + "0"], 2),
        "GA0 GA2.5 GOP2.5 mv2": ([GA + "0", GA + "2.5", GOP + "2.5"], 2),
        "GA0 GA2.5 GG0 GG2.5 mv2": ([GA + "0", GA + "2.5", GG + "0", GG + "2.5"], 2),
        "GA0 GA2.5 GG0 GOP2.5 mv2": ([GA + "0", GA + "2.5", GG + "0", GOP + "2.5"], 2),
        "GA0 GA2.5 GOP0 GOP2.5 mv2": ([GA + "0", GA + "2.5", GOP + "0", GOP + "2.5"], 2),
        "GA0 GA2.5 GG0 GG2.5 GOP0 GOP2.5 mv2": ([GA + "0", GA + "2.5", GG + "0", GG + "2.5", GOP + "0", GOP + "2.5"], 2),
        "GA0 GA2.5 GG0 GG2.5 GOP0 GOP2.5 mv3": ([GA + "0", GA + "2.5", GG + "0", GG + "2.5", GOP + "0", GOP + "2.5"], 3),
        "GA0 GA2.5 GG0 GG2.5 BAND0 MIX0 mv2": ([GA + "0", GA + "2.5", GG + "0", GG + "2.5", BAND + "0", MIX + "0"], 2),
        "GA0 GA2.5 GG0 GG2.5 BAND0 MIX0 mv3": ([GA + "0", GA + "2.5", GG + "0", GG + "2.5", BAND + "0", MIX + "0"], 3),
    },
    "bass": {
        "old (demucs)": (["demucs|bass|any|0", "demucs|bass|basses|0", "demucs|guitar+other+piano+bass|any|0"], 2),
        "BB0": (["rofo|bass|basses|0"], 1),
        "BA0": (["rofo|bass|any|0"], 1),
        "BB0 BB2.5 mv2": (["rofo|bass|basses|0", "rofo|bass|basses|2.5"], 2),
        "BA0 BA2.5 mv2": (["rofo|bass|any|0", "rofo|bass|any|2.5"], 2),
        "BB0 BB2.5 BA0 mv2": (["rofo|bass|basses|0", "rofo|bass|basses|2.5", "rofo|bass|any|0"], 2),
        "BB0 BA0 BA2.5 mv2": (["rofo|bass|basses|0", "rofo|bass|any|0", "rofo|bass|any|2.5"], 2),
        "BB0 BB2.5 BA0 BA2.5 mv2": (["rofo|bass|basses|0", "rofo|bass|basses|2.5", "rofo|bass|any|0", "rofo|bass|any|2.5"], 2),
        "BB0 BB2.5 BAND0 mv2": (["rofo|bass|basses|0", "rofo|bass|basses|2.5", BAND + "0"], 2),
    },
    "drums": {
        "old (demucs)": (["demucs|drums|drums|0", "demucs|drums|any|0", MIX + "0"], 2),
        "D0": (["rofo|drums|any|0"], 1),
        "D0 D2.5 D3.75 mv2": (["rofo|drums|any|0", "rofo|drums|any|2.5", "rofo|drums|any|3.75"], 2),
        "D0 D2.5 mv1": (["rofo|drums|any|0", "rofo|drums|any|2.5"], 1),
        "D0 D2.5 D3.75 MIX0 mv2": (["rofo|drums|any|0", "rofo|drums|any|2.5", "rofo|drums|any|3.75", MIX + "0"], 2),
        "D0 D2.5 D3.75 MIX0 MIX2.5 mv3": (["rofo|drums|any|0", "rofo|drums|any|2.5", "rofo|drums|any|3.75", MIX + "0", MIX + "2.5"], 3),
    },
}

if __name__ == "__main__":
    part = sys.argv[1]
    sets = sys.argv[2:]
    post = ensemble.lowest_only if part == "bass" else None
    loaded = {}
    for d in sets:
        need = sorted({s for specs, _ in RECIPES[part].values() for s in specs})
        try:
            loaded[d] = V.load(d, part, None)
        except Exception as e:
            print("skip", d, e)
    print(f"{'':40s}" + "".join(f"{d:>26s}" for d in loaded))
    for name, (specs, mv) in RECIPES[part].items():
        line = f"{name:40s}"
        for d, (data, have) in loaded.items():
            if not set(specs) <= set(have):
                line += f"{'-':>26s}"
                continue
            fn = ensemble.vote if name.startswith("old (") else ensemble.merge
            prep = (lambda n: ensemble.normalize_drums(n)) if part == "drums" and not name.startswith("old (") else (lambda n: n)
            dd = [(it, {s: prep(v) for s, v in res.items()}) for it, res in data]
            p, r, f = V.score(dd, part, specs, mv, post, fn=fn)
            line += f"    P {p:.3f} R {r:.3f} F {f:.3f}"
        print(line)
