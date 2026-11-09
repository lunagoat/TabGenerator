"""Re-run the tab-writing stage for a song in .cache from its old cached
transcriptions (no GPU): for checking rhythm / tab / Guitar Pro output."""
import json
import sys
from pathlib import Path

import env
import numpy as np

import beat_data
from tabgen import beats as B, ensemble, pipeline
from tabgen.transcribe import TNote

OLD = {
    "band": "guitar+other+piano+bass_any", "guitars_in_gop": "guitar+other+piano_acoustic_guitar+clean_electric_guitar+distorted_electric_guitar",
    "guitar_stem": "guitar_any", "mix": "mix_any", "bass_stem": "bass_any", "bass_in_bass_stem": "bass_electric_bass+acoustic_bass",
    "drums_in_drum_stem": "drums_drums", "drum_stem": "drums_any",
}
OLD_PARTS = {"guitar": ["band", "guitars_in_gop", "guitar_stem", "mix"], "bass": ["bass_stem", "bass_in_bass_stem", "band"],
             "drums": ["drums_in_drum_stem", "drum_stem", "mix"]}


def old_notes(cache: Path):
    res = {}
    for k, v in OLD.items():
        f = cache / f"notes_v1_{v}_large_b1.json"
        if f.is_file():
            res[k] = [TNote(**n) for n in json.loads(f.read_text())]
    out = {}
    for part, passes in OLD_PARTS.items():
        if all(p in res for p in passes):
            per = [[n for n in res[p] if n.instrument in pipeline.PART_LABELS[part]] for p in passes]
            out[part] = ensemble.vote(per, 2)
            if part == "bass":
                out[part] = ensemble.lowest_only(out[part])
    return out


def main(name_part, out_dir):
    cache = next(c for c in sorted((env.ROOT / ".cache").iterdir()) if name_part in c.name)
    key = f"user/{cache.name}"
    b, d = beat_data._logits(key, cache / f"{cache.name}.wav")
    beats = B.decode_beats(b)
    bars = B.decode_bars(beats, d)
    notes = old_notes(cache)
    import soundfile as sf
    duration = sf.info(str(cache / f"{cache.name}.wav")).duration
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    opts = pipeline.Options()
    files = pipeline._write_outputs(cache.name, cache / f"{cache.name}.wav", notes, None, beats, bars, duration, opts, out)
    return files


if __name__ == "__main__":
    for f in main(sys.argv[1], sys.argv[2]):
        print(f)
