"""Are the notes the transcription misses really in the audio, or is the
alignment off? Recall at growing time tolerances, and whether missed tab
notes are audible in the guitar stem (support.py)."""
import json
from pathlib import Path

import env
import numpy as np

import realeval as R
import realsongs
import support


class N:
    def __init__(self, t, p):
        self.onset, self.pitch = t, p


def main():
    rows = []
    for s in realsongs.songs():
        name = f"{s['artist']} - {s['title']}"
        f = R.OUT / R.slug(name) / "result.json"
        if not f.is_file():
            continue
        res = json.loads(f.read_text())
        tab, bars = realsongs.tab_notes(s["tabs"])
        tab = [n for n in tab if not set(n[4]) & set("txhc")]
        if len(tab) < 30 or not res["raw"]:
            continue
        to_audio, shift, q = R.align(tab, bars, res["raw"])
        if q < R.MIN_QUALITY:
            continue
        truth = [(float(to_audio(n[0])), n[1] + shift) for n in tab]
        raw = res["raw"]
        rec = {tol: len(R.match(truth, raw, lambda x: x[1], lambda y: y[1], tol)) / len(truth)
               for tol in (0.05, 0.075, 0.1, 0.15, 0.25)}
        hit = {i for i, _ in R.match(truth, raw, lambda x: x[1], lambda y: y[1], 0.15)}
        missed = [N(*truth[i]) for i in range(len(truth)) if i not in hit]
        found = [N(*truth[i]) for i in sorted(hit)]
        stem = env.ROOT / ".cache" / Path(s["audio"]).stem / "stems_bs_roformer_sw" / "guitar.wav"
        sp = support.Spectrum(stem)
        sub = missed[:: max(1, len(missed) // 300)]
        fsub = found[:: max(1, len(found) // 300)]
        rows.append((name, q, rec, sp.share(sub), sp.share(fsub), sp.chance(sub)))
        r = rows[-1]
        print(f"{name[:36]:36s} align {q:.2f} recall@50/75/100/150/250ms " + " ".join(f"{rec[t]:.2f}" for t in sorted(rec))
              + f" | audible: missed {r[3]:.2f} found {r[4]:.2f} chance {r[5]:.2f}", flush=True)
    print("MEAN recall by tolerance:", {k: round(float(np.mean([r[2][k] for r in rows])), 3) for k in rows[0][2]})
    print("MEAN audible: missed %.2f  found %.2f  chance %.2f" % tuple(float(np.mean([r[i] for r in rows])) for i in (3, 4, 5)))


if __name__ == "__main__":
    main()
