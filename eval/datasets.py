"""Test sets: MDB Drums (real recordings, drums), BabySlakh (synthetic band
mixes, guitar and bass), GuitarSet (real solo acoustic guitar)."""
import json

import env
import numpy as np

from harness import Item

MDB = env.DATA / "MDBDrums" / "MDB Drums"
MDB_KIND = {"KD": "kick", "SD": "snare", "HH": "hihat", "TT": "tom", "CY": "cymbal"}


def mdb_drum_only():
    return mdb_drums("drum_only")


def mdb_drums(source="full_mix"):
    """source: full_mix (the song) or drum_only (isolated drums = ceiling)."""
    items = []
    for wav in sorted((MDB / "audio" / source).glob("*.wav")):
        song = wav.stem.rsplit("_", 1)[0]
        truth = []
        for line in (MDB / "annotations" / "class" / f"{song}_class.txt").read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1] in MDB_KIND:
                truth.append((float(parts[0]), MDB_KIND[parts[1]]))
        items.append(Item(f"mdb_{source}", song, _stereo44(wav, f"mdb_{source}", song), {"drums": truth}))
    return items


def _stereo44(wav, dataset, name, start=None, dur=None):
    """Copy as 44.1 kHz stereo 16-bit WAV into the work folder (what the
    pipeline's decode step produces)."""
    import subprocess

    out = env.WORK / dataset / name / "song.wav"
    if not out.is_file():
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
        if start is not None:
            cmd += ["-ss", str(start), "-t", str(dur)]
        cmd += ["-i", str(wav), "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", str(out)]
        subprocess.run(cmd, check=True)
    return out


def _midi_notes(path):
    """[(onset_s, offset_s, pitch)] from a MIDI file, following its tempo map."""
    import mido

    mid = mido.MidiFile(str(path))
    notes, open_, t = [], {}, 0.0
    for msg in mid:  # iterating a MidiFile yields delta times in seconds
        t += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            open_.setdefault(msg.note, []).append(t)
        elif msg.type in ("note_off", "note_on") and open_.get(msg.note):
            notes.append((open_[msg.note].pop(0), t, msg.note))
    return notes


def babyslakh(start=30.0, dur=60.0):
    """60 s excerpts of the BabySlakh tracks that have guitar and bass."""
    import yaml

    items = []
    root = env.DATA / "babyslakh_16k"
    for track in sorted(root.glob("Track*")):
        meta = yaml.safe_load((track / "metadata.yaml").read_text())
        truth = {"guitar": [], "bass": [], "drums": []}
        for sid, s in meta["stems"].items():
            midi = track / "MIDI" / f"{sid}.mid"
            # (BabySlakh's metadata says audio_rendered: false for everything.)
            if not (track / "stems" / f"{sid}.wav").is_file() or not midi.is_file():
                continue
            part = {"Guitar": "guitar", "Bass": "bass"}.get(s["inst_class"])
            if s.get("is_drum"):
                continue
            if part:
                for on, off, pitch in _midi_notes(midi):
                    if start <= on < start + dur:
                        truth[part].append((on - start, pitch))
        truth = {k: sorted(set(v)) for k, v in truth.items() if len(v) >= 20}
        if "guitar" not in truth and "bass" not in truth:
            continue
        wav = _stereo44(track / "mix.wav", "babyslakh", track.name, start, dur)
        items.append(Item("babyslakh", track.name, wav, truth))
    return items


def guitarset(limit=36):
    """Real solo acoustic guitar (microphone recordings), evenly sampled."""
    audio = env.DATA / "guitarset" / "audio_mono-mic"
    files = sorted(audio.glob("*_mic.wav"))
    files = files[:: max(1, len(files) // limit)][:limit]
    items = []
    for wav in files:
        name = wav.stem[: -len("_mic")]
        jam = json.loads((env.DATA / "guitarset" / "annotation" / f"{name}.jams").read_text())
        truth = []
        for ann in jam["annotations"]:
            if ann["namespace"] == "note_midi":
                truth += [(d["time"], int(round(d["value"]))) for d in ann["data"]]
        items.append(Item("guitarset", name, _stereo44(wav, "guitarset", name), {"guitar": sorted(truth)}))
    return items
