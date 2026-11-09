"""Shared evaluation code: test items with ground truth, cached separation
and transcription passes, and note F1 scoring.

A pass is named "<separator>|<stems>|<instruments>|<shift>":
  separator    demucs | rofo | none (none: stems must be "mix")
  stems        "+"-joined stem names, or "mix" for the unseparated song
  instruments  any | guitars | basses | drums   (restricts the model's output)
  shift        seconds of silence put in front, which moves the model's
               5-second chunk boundaries (taken out of the note times again)
"""
import json
import types
from dataclasses import dataclass, field
from pathlib import Path

import env
import numpy as np

from tabgen import pipeline, separate, separate_roformer
from tabgen.ensemble import DRUM_TYPE
from tabgen.transcribe import BASSES, DRUMS, GUITARS, SAMPLE_RATE, TNote, Transcriber

INSTRUMENTS = {"any": None, "guitars": GUITARS, "basses": BASSES, "drums": DRUMS}
PART_LABELS = pipeline.PART_LABELS
TOL = 0.05


@dataclass
class Item:
    dataset: str
    name: str
    wav: Path  # 44.1 kHz stereo
    truth: dict = field(default_factory=dict)  # part -> [(onset, key)]

    @property
    def work(self) -> Path:
        d = env.WORK / self.dataset / self.name
        d.mkdir(parents=True, exist_ok=True)
        return d


def key_of(n: TNote):
    return DRUM_TYPE.get(n.pitch) if n.is_drum else n.pitch


# ---------------------------------------------------------------------------
# Separation and transcription, cached on disk

_transcriber = None


def transcriber():
    global _transcriber
    if _transcriber is None:
        import torch

        _transcriber = Transcriber("large", "cuda", dtype=torch.float16 if HALF else torch.float32)
    return _transcriber


def stems_for(item: Item, sep: str) -> dict:
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        return _stems_for(item, sep)


def _stems_for(item: Item, sep: str) -> dict:
    if sep == "demucs":
        fake = types.SimpleNamespace(OUTPUT_DIR=str(item.work / "demucs"), MODEL_NAME="htdemucs_6s")
        link = item.work / f"{item.name}.wav"
        if not link.exists():
            link.symlink_to(item.wav)
        return separate.separate(fake, link, item.name, "cuda", 2)
    if sep == "rofo":
        return separate_roformer.separate(item.wav, item.work, "cuda")
    raise ValueError(sep)


def pass_audio(item: Item, sep: str, stems: str) -> np.ndarray:
    if stems == "mix":
        return pipeline.load_mono(item.wav, SAMPLE_RATE)
    return pipeline.mix_stems(stems_for(item, sep), stems.split("+"), SAMPLE_RATE)


HALF = True  # fp16 model (what the pipeline uses on a GPU); cached separately from fp32
BATCH = 8


def _path(item: Item, spec: str) -> Path:
    sep, stems, inst, shift = spec.split("|")
    return item.work / f"notes_{sep}_{stems}_{inst}_{shift}{'.h' if HALF else ''}.json"


def run_passes(item: Item, specs: list[str]) -> dict[str, list[TNote]]:
    """Transcribe (or load from cache) several passes of one item; passes
    with the same instrument restriction are decoded together in a batch."""
    out, todo = {}, {}
    for spec in specs:
        path = _path(item, spec)
        if path.is_file():
            out[spec] = [TNote(**n) for n in json.loads(path.read_text())]
        else:
            todo.setdefault(spec.split("|")[2], []).append(spec)
    for inst, group in todo.items():
        for i in range(0, len(group), BATCH):
            batch = group[i: i + BATCH]
            audios, shifts = [], []
            for spec in batch:
                sep, stems, _, shift = spec.split("|")
                audio = pass_audio(item, sep, stems)
                shift = float(shift)
                if shift:
                    audio = np.concatenate([np.zeros(int(round(shift * SAMPLE_RATE)), dtype=np.float32), audio])
                audios.append(audio)
                shifts.append(shift)
            results = transcriber().transcribe_many(audios, INSTRUMENTS[inst])
            for spec, shift, notes in zip(batch, shifts, results):
                if shift:
                    for n in notes:
                        n.onset -= shift
                        n.offset -= shift
                    notes = [n for n in notes if n.onset >= 0]
                _path(item, spec).write_text(json.dumps([n.to_json() for n in notes]))
                out[spec] = notes
    return out


def run_pass(item: Item, spec: str) -> list[TNote]:
    return run_passes(item, [spec])[spec]


def part_notes(notes, part):
    return [n for n in notes if n.instrument in PART_LABELS[part]]


# ---------------------------------------------------------------------------
# Scoring


def match_count(pred: list[tuple], truth: list[tuple], tol: float = TOL) -> int:
    """Number of one-to-one matches between (onset, key) lists: same key,
    onsets within tol. Greedy in time order per key, which is optimal for
    points on a line with a fixed window."""
    by_key: dict = {}
    for t, k in truth:
        by_key.setdefault(k, []).append(t)
    pk: dict = {}
    for t, k in pred:
        pk.setdefault(k, []).append(t)
    tp = 0
    for k, ts in pk.items():
        ref = sorted(by_key.get(k, []))
        ts = sorted(ts)
        i = j = 0
        while i < len(ts) and j < len(ref):
            d = ts[i] - ref[j]
            if abs(d) <= tol:
                tp += 1
                i += 1
                j += 1
            elif d < 0:
                i += 1
            else:
                j += 1
    return tp


def prf(pred_notes: list[TNote], truth: list[tuple], window=None, tol: float = TOL):
    pred = [(n.onset, key_of(n)) for n in pred_notes if key_of(n) is not None]
    if window:
        pred = [p for p in pred if window[0] <= p[0] < window[1]]
    tp = match_count(pred, truth, tol)
    p = tp / max(1, len(pred))
    r = tp / max(1, len(truth))
    f = 2 * p * r / max(1e-9, p + r)
    return p, r, f


def mean_prf(rows):
    a = np.array(rows, dtype=float)
    return tuple(a.mean(axis=0)) if len(a) else (0, 0, 0)
