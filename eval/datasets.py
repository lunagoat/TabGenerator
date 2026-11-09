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
        _fix_bass_octave(items[-1])
    return items


def _fix_bass_octave(item):
    """Some Slakh bass patches sound an octave below their MIDI notes. Move
    the bass truth to the octave the bass-stem transcription agrees with
    (needs that transcription in the cache; otherwise leaves it alone)."""
    from harness import match_count

    ref = item.work / "notes_rofo_bass_any_0.h.json"
    if "bass" not in item.truth or not ref.is_file():
        return
    pred = [(n["onset"], n["pitch"]) for n in json.loads(ref.read_text()) if "bass" in n["instrument"]]
    truth = item.truth["bass"]
    best = max((-12, 0, 12), key=lambda sh: (match_count(pred, [(t, p + sh) for t, p in truth]), sh == 0))
    item.truth["bass"] = [(t, p + best) for t, p in truth]


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


# ---------------------------------------------------------------------------
# A made-up band around real guitar playing: GuitarSet guitar (real player,
# with note annotations) + real drums (MDB Drums, isolated drum tracks, with
# annotations) + a Slakh bass stem. The three don't play the same song, but
# separation and transcription don't know that; it tests "real guitar inside
# a mix", clean and through distortion, which BabySlakh's synthetic guitars
# (often heard as pianos or organs) cannot.

ROCK_DRUMS = ["80sRock", "Britpop", "Grunge", "Punk", "Rock", "SpeedMetal", "Zeppelin", "Hendrix",
              "Beatles", "Disco", "Shadows", "Rockabilly"]


def _fft_filter(x, sr, lo, hi, order=2):
    """Zero-phase band-pass with Butterworth-shaped skirts."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / sr)
    f[0] = 1e-6
    H = 1 / np.sqrt(1 + (lo / f) ** (2 * order)) / np.sqrt(1 + (f / hi) ** (2 * order))
    return np.fft.irfft(X * H, len(x))


def distort(x, sr, gain=25.0):
    """A high-gain amp, roughly: tighten the lows, clip hard, speaker cabinet."""
    x = _fft_filter(x, sr, 120, 12000)
    x = x / (np.abs(x).max() + 1e-9)
    y = np.tanh(gain * x)
    y = _fft_filter(y, sr, 90, 4500, order=2)
    return y


def gsband(kind="dist", limit=24):
    """kind: "dist" (pickup signal through distortion), "clean" (pickup
    signal as is) or "mic" (the acoustic microphone recording)."""
    import soundfile as sf
    import soxr

    gs = env.DATA / "guitarset"
    files = sorted((gs / "annotation").glob("*.jams"))
    files = files[5::13][:limit]  # 13: a mix of players, pieces and comp/solo
    bass_tracks = sorted((env.DATA / "babyslakh_16k").glob("Track*"))
    items = []
    for i, jam_path in enumerate(files):
        name = jam_path.stem
        out = env.WORK / f"gsband_{kind}" / name / "song.wav"
        jam = json.loads(jam_path.read_text())
        truth_g = []
        for ann in jam["annotations"]:
            if ann["namespace"] == "note_midi":
                truth_g += [(d["time"], int(round(d["value"]))) for d in ann["data"]]
        drum_song = "MusicDelta_" + ROCK_DRUMS[i % len(ROCK_DRUMS)]
        drum_wav = MDB / "audio" / "drum_only" / f"{drum_song}_Drum.wav"
        if not drum_wav.is_file():
            drum_wav = next((MDB / "audio" / "drum_only").glob(f"{drum_song}_*.wav"))
        src = gs / ("audio_mono-mic" if kind == "mic" else "audio_mono-pickup_mix") / f"{name}_{'mic' if kind == 'mic' else 'mix'}.wav"
        g, sr = sf.read(str(src), dtype="float64", always_2d=True)
        g = g.mean(axis=1)
        dur = len(g) / sr
        d, dsr = sf.read(str(drum_wav), dtype="float64", always_2d=True)
        drum_len = len(d) / dsr
        truth_d = []
        for line in (MDB / "annotations" / "class" / f"{drum_song}_class.txt").read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1] in MDB_KIND:
                t = float(parts[0])
                k = 0
                while t + k * drum_len < dur:  # the drum track loops under the guitar
                    truth_d.append((t + k * drum_len, MDB_KIND[parts[1]]))
                    k += 1
        import yaml

        for k in range(len(bass_tracks)):
            track = bass_tracks[(i + k) % len(bass_tracks)]
            meta = yaml.safe_load((track / "metadata.yaml").read_text())
            sid = next((s for s, m in meta["stems"].items()
                        if m["inst_class"] == "Bass" and (track / "stems" / f"{s}.wav").is_file()), None)
            if sid:
                break
        # The bass stem is used from 30 s on, looped if it is too short.
        truth_b = [(on - 30.0, p) for on, off, p in _midi_notes(track / "MIDI" / f"{sid}.mid") if 30.0 <= on < 30.0 + dur]
        if not out.is_file():
            assert sr == 44100 and dsr == 44100, (sr, dsr)
            if kind == "dist":
                g = distort(g, sr)
            reps = int(np.ceil(len(g) / len(d)))
            d = np.tile(d, (reps, 1))[: len(g)]
            if d.shape[1] == 1:
                d = np.repeat(d, 2, axis=1)
            b, bsr = sf.read(str(track / "stems" / f"{sid}.wav"), dtype="float64")
            b = soxr.resample(b[30 * bsr:], bsr, sr)
            b = np.resize(b, len(g)) if len(b) >= len(g) else np.pad(b, (0, len(g) - len(b)))
            rms = lambda x: np.sqrt(np.mean(x ** 2)) + 1e-9
            g, b, d = g / rms(g), 0.8 * b / rms(b), d / rms(d)
            mix = d[:, :2] + b[:, None] + np.stack([1.15 * g, 0.85 * g], axis=1)
            mix *= 0.9 / np.abs(mix).max()
            out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out), mix, sr, subtype="PCM_16")
        truth = {"guitar": sorted(truth_g), "drums": sorted(truth_d)}
        if len(truth_b) >= 10:
            truth["bass"] = sorted(truth_b)
        items.append(Item(f"gsband_{kind}", name, out, truth))
        _fix_bass_octave(items[-1])
    return items


def gsband_dist():
    return gsband("dist")


def gsband_clean():
    return gsband("clean")
