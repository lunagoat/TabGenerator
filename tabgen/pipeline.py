"""Audio file in, guitar/bass/drum tabs out."""

import json
import shutil
from glob import escape as glob_escape
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import fretboard, gp_export, rhythm, separate, tab_text
from .transcribe import BASSES, DRUMS, GUITARS, SAMPLE_RATE, TNote

ROOT = Path(__file__).resolve().parents[1]
CACHE_VERSION = 1

# Transcription passes: which audio the model hears, and which instruments
# it may output (None = any; notes of other instruments are then discarded,
# which works better than forcing leaked sounds into the wanted instrument).
PASSES = {
    "mix": {"stems": ["mix"], "instruments": None},
    "band": {"stems": ["guitar", "other", "piano", "bass"], "instruments": None},
    "guitars_in_gop": {"stems": ["guitar", "other", "piano"], "instruments": GUITARS},
    "guitar_stem": {"stems": ["guitar"], "instruments": None},
    "bass_stem": {"stems": ["bass"], "instruments": None},
    "bass_in_bass_stem": {"stems": ["bass"], "instruments": BASSES},
    "drums_in_drum_stem": {"stems": ["drums"], "instruments": DRUMS},
    "drum_stem": {"stems": ["drums"], "instruments": None},
    # Guitar audio split by where it sits in the stereo mix; used to tell
    # guitar parts apart, not for voting.
    "guitars_left": {"stems": ["guitar", "piano", "other"], "instruments": None, "position": "left"},
    "guitars_center": {"stems": ["guitar", "piano", "other"], "instruments": None, "position": "center"},
    "guitars_right": {"stems": ["guitar", "piano", "other"], "instruments": None, "position": "right"},
}
POSITION_PASSES = ["guitars_left", "guitars_center", "guitars_right"]
# Below this much left/right difference the guitars are treated as mono.
MIN_STEREO_SPREAD = 0.03

# Per part: passes to vote over (most reliable first) and the votes a note
# needs. Chosen on BabySlakh (guitar, bass) and MDB Drums (real recordings);
# see README "How accurate is it?".
PARTS = {
    "guitar": {"passes": ["band", "guitars_in_gop", "guitar_stem", "mix"], "min_votes": 2},
    "bass": {"passes": ["bass_stem", "bass_in_bass_stem", "band"], "min_votes": 2, "lowest_only": True},
    "drums": {"passes": ["drums_in_drum_stem", "drum_stem", "mix"], "min_votes": 2},
}
# --fast: the best single pass per part.
FAST_PARTS = {
    "guitar": {"passes": ["band"], "min_votes": 1},
    "bass": {"passes": ["bass_stem"], "min_votes": 1, "lowest_only": True},
    "drums": {"passes": ["drums_in_drum_stem"], "min_votes": 1},
}
PART_LABELS = {"guitar": set(GUITARS), "bass": set(BASSES), "drums": {"drums"}}

GM_PROGRAMS = {
    "acoustic_guitar": 25,
    "clean_electric_guitar": 27,
    "distorted_electric_guitar": 30,
    "electric_bass": 33,
    "acoustic_bass": 32,
}
PART_TITLES = {
    "acoustic_guitar": "acoustic",
    "clean_electric_guitar": "clean",
    "distorted_electric_guitar": "distorted",
}


@dataclass
class Options:
    model_size: str = "large"
    weights: str | None = None  # local model.safetensors (default: ~/muscriptor-<size>)
    beam_size: int = 1
    shifts: int = 2
    device: str | None = None
    guitar_tuning: str | None = None
    bass_tuning: str | None = None
    resplit: bool = False
    retranscribe: bool = False
    out_dir: Path | None = None
    parts: tuple[str, ...] = ("guitar", "bass", "drums")
    split_guitars: bool = True
    fast: bool = False
    open_folder: bool = True


def _device(opts: Options) -> str:
    if opts.device:
        return opts.device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def decode(input_path: Path, out_wav: Path):
    """Any audio/video file ffmpeg understands -> 44.1 kHz stereo WAV."""
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(input_path), "-vn", "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le",
        str(out_wav),
    ]
    if subprocess.run(cmd).returncode != 0 or not out_wav.is_file():
        raise SystemExit(f"ffmpeg could not decode {input_path}")


def load_mono(path: Path, sr: int) -> np.ndarray:
    import soundfile as sf
    import soxr

    data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if file_sr != sr:
        mono = soxr.resample(mono, file_sr, sr, quality="VHQ")
    return mono.astype(np.float32)


def load_stereo_sum(stems: dict[str, Path], names: list[str]) -> tuple[np.ndarray, int]:
    import soundfile as sf

    parts = [sf.read(str(stems[n]), dtype="float32", always_2d=True) for n in names]
    sr = parts[0][1]
    n = min(len(a) for a, _ in parts)
    total = np.sum([a[:n, :2] if a.shape[1] > 1 else np.repeat(a[:n], 2, axis=1) for a, _ in parts], axis=0)
    return total, sr


def mix_stems(stems: dict[str, Path], names: list[str], sr: int) -> np.ndarray:
    parts = [load_mono(stems[n], sr) for n in names]
    n = min(len(p) for p in parts)
    return np.sum([p[:n] for p in parts], axis=0)


# ----------------------------------------------------------------------------


def run(input_path: Path, opts: Options) -> Path:
    input_path = input_path.expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"File not found: {input_path}")
    name = input_path.stem
    cache = ROOT / ".cache" / name
    cache.mkdir(parents=True, exist_ok=True)
    out_dir = opts.out_dir or (ROOT / "Tabs" / name)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _device(opts)
    steps = 5

    print(f"\n[1/{steps}] Decoding {input_path.name}")
    song_wav = cache / f"{name}.wav"
    if not song_wav.is_file():
        decode(input_path, song_wav)
    mix16 = load_mono(song_wav, SAMPLE_RATE)
    duration = len(mix16) / SAMPLE_RATE
    print(f"      {duration / 60:.0f}:{duration % 60:04.1f} long")

    print(f"\n[2/{steps}] Splitting stems with StemSplitter (Demucs, {opts.shifts} shifts)")
    splitter = separate.load_splitter(separate.find_stemsplitter())
    plan = FAST_PARTS if opts.fast else PARTS
    needed = {s for p in opts.parts for ps in plan[p]["passes"] for s in PASSES[ps]["stems"] if s != "mix"}
    stems = {}
    if needed:
        with tempfile.TemporaryDirectory() as tmp:
            # Demucs names its output folder after the input file.
            link = Path(tmp) / f"{name}.wav"
            link.symlink_to(song_wav)
            stems = separate.separate(splitter, link, name, device, opts.shifts, opts.resplit)

    print(f"\n[3/{steps}] Finding the beat")
    beats_file = cache / "beats.json"
    if beats_file.is_file():
        b = json.loads(beats_file.read_text())
        beats, downbeats = np.array(b["beats"]), np.array(b["downbeats"])
    else:
        beats, downbeats = rhythm.track_beats(load_mono(song_wav, 22050), 22050, device)
        beats_file.write_text(json.dumps({"beats": beats.tolist(), "downbeats": downbeats.tolist()}))
    if len(beats) >= 2:
        bpm = 60.0 / float(np.median(np.diff(beats)))
        print(f"      ~{bpm:.0f} BPM, {len(downbeats)} bars")

    print(f"\n[4/{steps}] Transcribing notes (MuScriptor {opts.model_size})")
    notes, guitar_groups = _transcribe_parts(opts, stems, mix16, cache, device)

    print(f"\n[5/{steps}] Writing tabs")
    files = _write_outputs(name, input_path, notes, guitar_groups, beats, downbeats, duration, opts, out_dir)
    for f in files:
        print(f"      {f.name}")
    print(f"\nDone! Tabs are in: {out_dir}")
    return out_dir


def _transcribe_parts(opts, stems, mix16, cache: Path, device: str):
    """Run the transcription passes and vote. Returns the notes per part, and
    the guitar notes grouped by stereo position (None when not split)."""
    from . import stereo
    from .ensemble import lowest_only, vote
    from .transcribe import Transcriber, print_progress

    plan = FAST_PARTS if opts.fast else PARTS
    pass_names = []
    for part in opts.parts:
        for p in plan[part]["passes"]:
            if p not in pass_names:
                pass_names.append(p)

    guitar_stereo = None
    if "guitar" in opts.parts and opts.split_guitars and not opts.fast:
        audio, sr = load_stereo_sum(stems, PASSES[POSITION_PASSES[0]]["stems"])
        if stereo.stereo_spread(audio) >= MIN_STEREO_SPREAD:
            guitar_stereo = (audio, sr)
            pass_names += POSITION_PASSES
        else:
            print("      guitars are mono: no left/center/right split")
    positions_audio = None

    transcriber = None
    results: dict[str, list[TNote]] = {}
    for i, name in enumerate(pass_names):
        spec = PASSES[name]
        inst = "any" if spec["instruments"] is None else "+".join(spec["instruments"])
        where = f"@{spec['position']}" if "position" in spec else ""
        key = f"v{CACHE_VERSION}_{'+'.join(spec['stems'])}{where}_{inst}_{opts.model_size}_b{opts.beam_size}"
        path = cache / f"notes_{key}.json"
        label = f"pass {i + 1}/{len(pass_names)} ({name})"
        if path.is_file() and not opts.retranscribe:
            results[name] = [TNote(**n) for n in json.loads(path.read_text())]
            print(f"      {label}: reusing earlier transcription")
            continue
        if transcriber is None:
            transcriber = Transcriber(opts.model_size, device, opts.weights)
        if "position" in spec:
            if positions_audio is None:
                positions_audio = stereo.split_positions(guitar_stereo[0])
            import soxr

            audio = soxr.resample(positions_audio[spec["position"]], guitar_stereo[1], SAMPLE_RATE, quality="VHQ")
        elif spec["stems"] == ["mix"]:
            audio = mix16
        else:
            audio = mix_stems(stems, spec["stems"], SAMPLE_RATE)
        notes = transcriber.transcribe(audio, spec["instruments"], opts.beam_size, print_progress(label))
        path.write_text(json.dumps([n.to_json() for n in notes]))
        results[name] = notes

    out: dict[str, list[TNote]] = {}
    for part in opts.parts:
        labels = PART_LABELS[part]
        per_pass = [[n for n in results[p] if n.instrument in labels] for p in plan[part]["passes"]]
        if part == "guitar" and guitar_stereo is not None:
            # The left/center/right transcriptions also vote (as one voter):
            # a guitar that is quiet in the mono mixes, e.g. one panned away
            # from the others, is heard there. On panned test mixes this found
            # more guitar notes (F1 0.502 -> 0.523) and cleaner parts.
            per_pass.append([n for p in POSITION_PASSES for n in results[p] if n.instrument in labels])
        out[part] = vote(per_pass, plan[part]["min_votes"])
        if plan[part].get("lowest_only"):
            out[part] = lowest_only(out[part])
        counts = ", ".join(str(len(x)) for x in per_pass)
        print(f"      {part}: {len(out[part])} notes agreed on (per pass: {counts})")

    guitar_groups = None
    if guitar_stereo is not None:
        guitar_groups = split_guitars_by_position(
            out["guitar"], guitar_stereo, [results[p] for p in POSITION_PASSES]
        )
        if len(guitar_groups) > 1:
            sizes = ", ".join(f"{pos} {len(ns)}" for pos, ns in guitar_groups)
            print(f"      guitar parts by stereo position: {sizes}")
    return out, guitar_groups


MIN_PART_SHARE = 0.12  # a stereo position needs this share of the notes to be its own part
MIN_PART_NOTES = 40


def split_guitars_by_position(notes, guitar_stereo, position_notes):
    """Group guitar notes into parts by where they sit in the stereo mix."""
    from . import stereo
    from .ensemble import _Index

    if not notes:
        return [("", notes)]
    audio, sr = guitar_stereo
    pans = stereo.note_pans(audio, sr, notes)
    indexes = [_Index([n for n in ns if n.instrument in PART_LABELS["guitar"]]) for ns in position_notes]
    found = np.array([[idx.has(n) for idx in indexes] for n in notes])
    scores = stereo.position_scores(notes, pans, found)
    pos = scores.argmax(axis=1)
    counts = np.bincount(pos, minlength=3)
    keep = [p for p in range(3) if counts[p] >= max(MIN_PART_NOTES, MIN_PART_SHARE * len(notes))]
    if len(keep) <= 1:
        return [("", notes)]
    # Notes of too-small positions join their next best position.
    masked = np.full_like(scores, -np.inf)
    masked[:, keep] = scores[:, keep]
    pos = masked.argmax(axis=1)
    return [
        (stereo.POSITIONS[p], [n for n, q in zip(notes, pos) if q == p])
        for p in keep
    ]


# ----------------------------------------------------------------------------


def _correct_onset_delay(notes: dict[str, list[TNote]], beats: np.ndarray) -> float:
    """MuScriptor's onsets can sit a few ms late; measure it against the beats."""
    from muscriptor.utils.beats import BeatGrid, estimate_onset_delay, fit_tempo

    onsets = [n.onset for ns in notes.values() for n in ns]
    if len(beats) < 8 or not onsets:
        return 0.0
    bpm, _ = fit_tempo(beats)
    grid = BeatGrid(bpm=bpm, beats_per_bar=None, first_downbeat=float(beats[0]), beats=beats)
    measured = estimate_onset_delay(onsets, grid)
    if measured is None:
        return 0.0
    for ns in notes.values():
        for n in ns:
            n.onset -= measured.seconds
            n.offset -= measured.seconds
    return measured.seconds


def split_guitar_parts(notes: list[TNote]) -> list[tuple[str, list[TNote]]]:
    """One part per guitar sound when the model hears more than one guitar
    (e.g. distorted rhythm + clean lead); otherwise a single part."""
    if not notes:
        return [("", notes)]
    counts: dict[str, int] = {}
    for n in notes:
        counts[n.instrument] = counts.get(n.instrument, 0) + 1
    major = [k for k, c in counts.items() if c >= 0.2 * len(notes)]
    if len(major) <= 1:
        return [("", notes)]
    major.sort(key=lambda k: -counts[k])
    groups: dict[str, list[TNote]] = {k: [] for k in major}
    onsets = {k: np.array(sorted(n.onset for n in notes if n.instrument == k)) for k in major}
    for n in notes:
        if n.instrument in groups:
            groups[n.instrument].append(n)
            continue
        # Stray notes of a minor guitar sound go to whichever major part is
        # busiest around them.
        def density(k):
            o = onsets[k]
            return np.searchsorted(o, n.onset + 2) - np.searchsorted(o, n.onset - 2)

        groups[max(major, key=density)].append(n)
    return [(PART_TITLES.get(k, k), groups[k]) for k in major]


def _events(notes, bm):
    """Quantized notes grouped into chords: (ticks, notes per tick, fretboard events)."""
    q = rhythm.quantize(bm, notes, bm.subdivisions)
    by_tick: dict[int, list] = {}
    for n in q:
        by_tick.setdefault(n.onset_tick, []).append(n)
    ticks = sorted(by_tick)
    events = []
    for t in ticks:
        group = sorted(by_tick[t], key=lambda n: n.pitch)
        events.append(
            fretboard.Event(
                time=min(n.onset for n in group),
                pitches=[n.pitch for n in group],
                ends=[n.offset for n in group],
            )
        )
    return ticks, by_tick, events


def _fretted_part(title, notes, bm, tunings_table_bass: bool, forced_tuning, program_default, band_tuning=None):
    """Quantize, choose tuning, finger, and build text + GP data for one part."""
    ticks, by_tick, events = _events(notes, bm)
    if forced_tuning:
        tuning = fretboard.find_tuning(forced_tuning, tunings_table_bass)
    else:
        tuning = fretboard.choose_tuning(events, tunings_table_bass, band_tuning=band_tuning)
    moved = fretboard.fit_to_range(events, tuning)
    max_fret = 24
    fingerings, _ = fretboard.assign_frets(events, tuning, max_fret)
    n_strings = len(tuning.strings)
    text_events: dict[int, dict[int, str]] = {}
    gp_events = []
    dropped = 0
    for t, f in zip(ticks, fingerings):
        dropped += len(f.dropped)
        if not f.strings:
            continue
        text_events[t] = {n_strings - 1 - s: str(fr) for s, fr in zip(f.strings, f.frets)}
        end = max(n.end_tick for n in by_tick[t])
        gp_events.append(gp_export.GPEvent(t, end, list(zip(f.strings, f.frets))))

    labels = list(reversed(tuning.labels()))
    counts: dict[str, int] = {}
    for n in notes:
        counts[n.instrument] = counts.get(n.instrument, 0) + 1
    program = GM_PROGRAMS.get(max(counts, key=counts.get), program_default) if counts else program_default
    strings_desc = " ".join(fretboard.note_name(p) for p in tuning.strings)
    info = {
        "tuning": f"{tuning.name} ({strings_desc})",
        "notes": len(notes),
        "dropped": dropped,
        "moved": moved,
        "tuning_name": tuning.name,
    }
    gp_part = gp_export.GPPart(title, gp_events, tuning.strings, program, max_fret)
    return labels, text_events, gp_part, info


def _drum_part(notes, bm):
    q = rhythm.quantize(bm, notes, bm.subdivisions)
    present = set()
    by_tick: dict[int, dict[str, str]] = {}
    gp_by_tick: dict[int, set] = {}
    for n in q:
        label, sym = tab_text.drum_line_for(n.pitch)
        present.add(label)
        cell = by_tick.setdefault(n.onset_tick, {})
        # Open hi-hat beats closed; a crash beats a splash on the same line.
        if label not in cell or sym in "XoC":
            cell[label] = sym
        gp_by_tick.setdefault(n.onset_tick, set()).add(n.pitch)
    lines = [l for l, _ in tab_text.DRUM_LINES if l in present or l in tab_text.ALWAYS_SHOWN]
    index = {l: i for i, l in enumerate(lines)}
    events = {t: {index[l]: s for l, s in cell.items()} for t, cell in by_tick.items()}
    step = {t: bm.beat_ticks // max(1, int(bm.subdivisions[min(t // bm.beat_ticks, len(bm.subdivisions) - 1)])) for t in gp_by_tick}
    gp_events = [gp_export.GPEvent(t, t + step[t], sorted(p)) for t, p in sorted(gp_by_tick.items())]
    return lines, events, gp_export.GPPart("Drums", gp_events, None, 0)


def _write_outputs(name, input_path, notes, guitar_groups, beats, downbeats, duration, opts, out_dir: Path) -> list[Path]:
    delay = _correct_onset_delay(notes, beats)
    bm = rhythm.build_beatmap(beats, downbeats, duration)
    all_onsets = np.array([n.onset for ns in notes.values() for n in ns])
    if rhythm.detect_compound(bm, all_onsets):
        bm.compound = True
    bm.subdivisions = rhythm.choose_subdivisions(bm, all_onsets)

    tempos = [bm.bar_tempo(b) for b in bm.bars]
    tempo = float(np.median(tempos)) if tempos else 120.0
    meters = [bm.time_signature(b) for b in bm.bars]
    common_meter = max(set(meters), key=meters.count) if meters else (4, 4)
    base_header = [
        f"{name}",
        "=" * min(len(name), 80),
        f"Source: {input_path.name}",
        (f"Tempo: ~{tempo / 1.5:.0f} BPM (dotted-quarter beat)" if bm.compound else f"Tempo: ~{tempo:.0f} BPM")
        + f"   Time: {common_meter[0]}/{common_meter[1]}"
        + (f"   (onsets shifted {delay * 1000:+.0f} ms to the beat)" if abs(delay) > 0.0005 else ""),
    ]
    note_line = "Transcribed automatically (Demucs + MuScriptor + TabGenerator). Check it by ear!"

    files: list[Path] = []
    gp_parts: list[gp_export.GPPart] = []

    # The number of guitar parts can change between runs; clear old ones.
    for old in out_dir.glob(f"{glob_escape(name)} - Guitar*.txt"):
        old.unlink()

    guitar_tuning = opts.guitar_tuning
    if "guitar" in notes and not guitar_tuning:
        # Every guitar part gets the same tuning, detected from all of them:
        # bands tune together, and a small part alone says little.
        guitar_tuning = fretboard.choose_tuning(_events(notes["guitar"], bm)[2], False).name
    if "guitar" in notes:
        groups = []
        for where, pnotes in guitar_groups or [("", notes["guitar"])]:
            sounds = split_guitar_parts(pnotes) if opts.split_guitars else [("", pnotes)]
            for sound, gnotes in sounds:
                groups.append((", ".join(x for x in (where, sound) if x), gnotes))
        for gi, (flavour, gnotes) in enumerate(groups):
            title = "Guitar" if len(groups) == 1 else f"Guitar {gi + 1} ({flavour})"
            labels, events, gp_part, info = _fretted_part(
                title, gnotes, bm, False, guitar_tuning, 30
            )
            gp_parts.append(gp_part)
            header = base_header + [
                f"Part: {title}",
                f"Tuning: {info['tuning']}",
                note_line,
            ]
            footer = []
            if info["moved"]:
                footer.append(f"({info['moved']} notes below the lowest string were octave ghosts of power chords or were moved up an octave.)")
            if info["dropped"]:
                footer.append(f"({info['dropped']} notes did not fit on the neck and were left out.)")
            path = out_dir / f"{name} - {title}.txt"
            path.write_text(tab_text.render(header, labels, bm, events, footer))
            files.append(path)

    if "bass" in notes:
        labels, events, gp_part, info = _fretted_part(
            "Bass", notes["bass"], bm, True, opts.bass_tuning, 33, guitar_tuning
        )
        gp_parts.append(gp_part)
        header = base_header + ["Part: Bass", f"Tuning: {info['tuning']}", note_line]
        footer = []
        if info["moved"]:
            footer.append(f"({info['moved']} notes below the lowest string were moved up an octave.)")
        if info["dropped"]:
            footer.append(f"({info['dropped']} notes did not fit on the neck and were left out.)")
        path = out_dir / f"{name} - Bass.txt"
        path.write_text(tab_text.render(header, labels, bm, events, footer))
        files.append(path)

    if "drums" in notes:
        labels, events, gp_part = _drum_part(notes["drums"], bm)
        gp_parts.append(gp_part)
        header = base_header + ["Part: Drums", note_line]
        path = out_dir / f"{name} - Drums.txt"
        path.write_text(tab_text.render(header, labels, bm, events, [""] + tab_text.DRUM_LEGEND))
        files.append(path)

    if gp_parts:
        path = out_dir / f"{name}.gp5"
        gp_export.write_gp5(str(path), name, "", bm, gp_parts)
        files.append(path)

    path = out_dir / f"{name}.mid"
    write_midi(path, notes)
    files.append(path)
    return files


def write_midi(path: Path, notes: dict[str, list[TNote]]):
    """Unquantized performance MIDI of everything transcribed."""
    import mido

    tpq, tempo_us = 480, 500000  # 120 BPM: 1 s = 960 ticks
    mid = mido.MidiFile(ticks_per_beat=tpq)
    channel = 0
    for part, ns in notes.items():
        if not ns:
            continue
        is_drum = part == "drums"
        ch = 9 if is_drum else channel
        if not is_drum:
            channel += 1
            if channel == 9:
                channel = 10
        track = mido.MidiTrack()
        track.append(mido.MetaMessage("track_name", name=part.capitalize(), time=0))
        if not is_drum:
            counts: dict[str, int] = {}
            for n in ns:
                counts[n.instrument] = counts.get(n.instrument, 0) + 1
            program = GM_PROGRAMS.get(max(counts, key=counts.get), 30)
            track.append(mido.Message("program_change", program=program, channel=ch, time=0))
        if not mid.tracks:
            track.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))
        msgs = []
        for n in ns:
            on = max(0, int(round(n.onset * 960)))
            off = max(on + 1, int(round((n.offset if not n.is_drum else n.onset + 0.1) * 960)))
            msgs.append((on, 1, n.pitch))
            msgs.append((off, 0, n.pitch))
        msgs.sort(key=lambda m: (m[0], m[1]))
        last = 0
        for t, is_on, pitch in msgs:
            kind = "note_on" if is_on else "note_off"
            track.append(mido.Message(kind, note=pitch, velocity=90 if is_on else 0, channel=ch, time=t - last))
            last = t
        mid.tracks.append(track)
    mid.save(str(path))


def open_folder(path: Path):
    opener = shutil.which("xdg-open")
    if opener:
        subprocess.Popen([opener, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
