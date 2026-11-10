"""Load a cached song's voted notes and beat map without the GPU (for
working on the arranging / tab-writing stage)."""
import json
import sys

import env
import numpy as np

from tabgen import ensemble, pipeline, rhythm
from tabgen.transcribe import TNote


def load(name_part="Hazel"):
    cache = next(c for c in sorted((env.ROOT / ".cache").iterdir()) if name_part in c.name)
    def rd(tag):
        return [TNote(**n) for n in json.loads((cache / f"notes_v2_roformer-{tag}_large_b1_fp16.json").read_text())]
    g = [[n for n in rd(t) if n.instrument in pipeline.PART_LABELS["guitar"]] for t in ("guitar_any", "guitar_any_s2.5", "guitar_any_s3.75")]
    guitar = ensemble.merge(g, 2)
    b = json.loads((cache / "beats_v2.json").read_text())
    import soundfile as sf
    dur = sf.info(str(cache / f"{cache.name}.wav")).duration
    bm = rhythm.build_beatmap(np.array(b["beats"]), np.array(b["bar_starts"]), dur)
    return cache, guitar, g, bm


if __name__ == "__main__":
    cache, guitar, g, bm = load(sys.argv[1] if len(sys.argv) > 1 else "Hazel")
    print(len(guitar), [len(x) for x in g])
    conf = np.array([n.conf for n in guitar]); pitch = np.array([n.pitch for n in guitar])
    for lo, hi in ((0, 52), (52, 60), (60, 100)):
        m = (pitch >= lo) & (pitch < hi)
        print(f"pitch {lo}-{hi}: n={m.sum()} unanimous={np.mean(conf[m] > 0.99):.2f}  median dur={np.median([n.offset-n.onset for n,k in zip(guitar,m) if k]):.2f}")
    bm.subdivisions = rhythm.choose_subdivisions(bm, np.array([n.onset for n in guitar]))
    q = rhythm.quantize(bm, guitar, bm.subdivisions)
    bt = bm.beat_ticks
    for bar_i in (3, 4, 11):
        bar = bm.bars[bar_i]
        print("bar", bar_i + 1)
        lo, hi = bar.first_beat * bt, (bar.first_beat + bar.n_beats) * bt
        ev = {}
        for n, src in zip(q, sorted(guitar, key=lambda n: 0)):
            pass
        for n in q:
            if lo <= n.onset_tick < hi:
                ev.setdefault(n.onset_tick, []).append(n)
        by = {(round(n.onset, 3), n.pitch): n.conf for n in guitar}
        for t in sorted(ev):
            print("  ", (t - lo) / bt, [(n.pitch, round(by.get((round(n.onset, 3), n.pitch), -1), 2), round(n.offset - n.onset, 2)) for n in ev[t]])
