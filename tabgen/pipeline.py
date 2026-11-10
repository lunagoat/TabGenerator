"""Audio file in, guitar/bass/drum tabs out."""

import json
import shutil
from glob import escape as glob_escape
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from . import arrange, fretboard, gp_export, keys, rhythm, separate, tab_text
from .transcribe import BASSES, DRUMS, GUITARS, SAMPLE_RATE, TNote

ROOT = Path(__file__).resolve().parents[1]
CACHE_VERSION = 2

# Transcription passes: which audio the model hears, which instruments it
# may output (None = any; notes of other instruments are then discarded) and
# how much silence is put in front. MuScriptor works in 5-second chunks and
# makes most of its mistakes near their edges; a shifted pass has its edges
# elsewhere, so passes that differ only in shift disagree exactly where one
# of them is unreliable.
PASSES = {
    "guitar": {"stems": ["guitar"], "instruments": None},
    "guitar@2.5": {"stems": ["guitar"], "instruments": None, "shift": 2.5},
    "guitar@3.75": {"stems": ["guitar"], "instruments": None, "shift": 3.75},
    "bass_forced": {"stems": ["bass"], "instruments": BASSES},
    "bass_forced@2.5": {"stems": ["bass"], "instruments": BASSES, "shift": 2.5},
    "drums": {"stems": ["drums"], "instruments": None},
    "drums@2.5": {"stems": ["drums"], "instruments": None, "shift": 2.5},
    "drums@3.75": {"stems": ["drums"], "instruments": None, "shift": 3.75},
    "piano": {"stems": ["piano"], "instruments": None},
    "piano@2.5": {"stems": ["piano"], "instruments": None, "shift": 2.5},
    "piano@3.75": {"stems": ["piano"], "instruments": None, "shift": 3.75},
    # Synths have no stem of their own: they are what is left in "other".
    "synth": {"stems": ["other"], "instruments": None},
    "synth@2.5": {"stems": ["other"], "instruments": None, "shift": 2.5},
    "synth@3.75": {"stems": ["other"], "instruments": None, "shift": 3.75},
    # Left to itself the model writes down only part of a sung line; told
    # that it is listening to a voice, it writes the whole melody.
    "vocals": {"stems": ["vocals"], "instruments": ["voice"]},
    "vocals@2.5": {"stems": ["vocals"], "instruments": ["voice"], "shift": 2.5},
    "vocals@3.75": {"stems": ["vocals"], "instruments": ["voice"], "shift": 3.75},
    # Guitar stem split by where it sits in the stereo mix; used to tell
    # guitar parts apart, not for voting.
    "guitars_left": {"stems": ["guitar"], "instruments": None, "position": "left"},
    "guitars_center": {"stems": ["guitar"], "instruments": None, "position": "center"},
    "guitars_right": {"stems": ["guitar"], "instruments": None, "position": "right"},
}
POSITION_PASSES = ["guitars_left", "guitars_center", "guitars_right"]
# Below this much left/right difference the guitars are treated as mono.
MIN_STEREO_SPREAD = 0.03

# Per part: the passes that vote and the votes a note needs. Chosen on real
# guitar playing inside a band mix, BabySlakh and MDB Drums; see the README
# and eval/recipes.py.
PARTS = {
    "guitar": {"passes": ["guitar", "guitar@2.5", "guitar@3.75"], "min_votes": 2},
    "bass": {"passes": ["bass_forced", "bass_forced@2.5"], "min_votes": 2, "lowest_only": True},
    "drums": {"passes": ["drums", "drums@2.5", "drums@3.75"], "min_votes": 2},
    "piano": {"passes": ["piano", "piano@2.5", "piano@3.75"], "min_votes": 2},
    "synth": {"passes": ["synth", "synth@2.5", "synth@3.75"], "min_votes": 2},
    # Sung notes start softly, so two passes rarely put one within 50 ms.
    "vocals": {"passes": ["vocals", "vocals@2.5", "vocals@3.75"], "min_votes": 2, "tolerance": 0.1},
}
# --fast: the best single pass per part.
FAST_PARTS = {
    "guitar": {"passes": ["guitar"], "min_votes": 1},
    "bass": {"passes": ["bass_forced"], "min_votes": 1, "lowest_only": True},
    "drums": {"passes": ["drums"], "min_votes": 1},
    "piano": {"passes": ["piano"], "min_votes": 1},
    "synth": {"passes": ["synth"], "min_votes": 1},
    "vocals": {"passes": ["vocals"], "min_votes": 1},
}
PART_LABELS = {"guitar": set(GUITARS), "bass": set(BASSES), "drums": {"drums"}}
# "keys" on the command line means both keyboard parts.
PART_GROUPS = {"keys": ("piano", "synth")}
KEY_PARTS = ("piano", "synth", "vocals")  # written as a note roll / notation, not as tab
# A stem this much quieter than the song holds nothing worth transcribing,
# and a part with fewer notes than this was not really there.
SILENT_STEM = 0.02
MIN_KEY_NOTES = 20


def _wanted(part: str, note: TNote) -> bool:
    """Whether a note transcribed from a part's stem belongs to the part."""
    if part == "piano":
        # The separator already says "this is the piano": whatever the model
        # calls a note (a low one often "bass"), it is the piano playing.
        return not note.is_drum
    if part == "vocals":
        return not note.is_drum
    if part == "synth":
        # Synth sounds get all sorts of names (guitar, bells, strings...).
        return not note.is_drum and note.instrument not in PART_LABELS["bass"]
    return note.instrument in PART_LABELS[part]

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
    parts: tuple[str, ...] = ("guitar", "bass", "drums", "piano", "synth", "vocals")
    vocals_as: str = "piano"  # like keys_as, for the sung melody
    keys_as: str = "piano"  # "piano": note roll + notation track; "guitar": guitar tablature; "both"
    split_guitars: bool = True
    fast: bool = False
    open_folder: bool = True
    separator: str = "roformer"  # or "demucs" (htdemucs_6s through StemSplitter)
    batch: int = 8  # transcription passes decoded together (GPU memory vs speed)
    full_precision: bool = False  # fp32 on the GPU instead of fp16
    tidy: bool = True  # make repeats agree and keep each part playable by one guitarist


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


def _check_cache_owner(cache: Path, input_path: Path):
    """The cache folder is named after the song file; make sure what is in it
    was made from this very file (two albums can both have a "01 Intro")."""
    st = input_path.stat()
    mark = {"path": str(input_path), "size": st.st_size, "mtime": int(st.st_mtime)}
    owner = cache / "source.json"
    if owner.is_file():
        try:
            if json.loads(owner.read_text()) == mark:
                return
        except ValueError:
            pass
        for f in cache.iterdir():
            if f.is_dir():
                shutil.rmtree(f)
            else:
                f.unlink()
    owner.write_text(json.dumps(mark))


def _audible_parts(parts, plan, stems, mix16) -> tuple[str, ...]:
    """The parts whose stems actually contain sound."""
    import soundfile as sf

    mix_rms = float(np.sqrt(np.mean(mix16.astype(np.float64) ** 2))) + 1e-12
    level: dict[str, float] = {}
    keep = []
    for part in parts:
        names = {s for ps in plan[part]["passes"] for s in PASSES[ps]["stems"] if s != "mix"}
        loud = False
        for s in names:
            if s not in level:
                audio, _ = sf.read(str(stems[s]), dtype="float32", always_2d=True)
                level[s] = float(np.sqrt(np.mean(audio.mean(axis=1).astype(np.float64) ** 2)))
            loud = loud or level[s] >= SILENT_STEM * mix_rms
        if loud or not names:
            keep.append(part)
        else:
            print(f"      no {part} in this song (its stem is silent)")
    return tuple(keep)


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
    _check_cache_owner(cache, input_path)
    song_wav = cache / f"{name}.wav"
    if not song_wav.is_file():
        decode(input_path, song_wav)
    mix16 = load_mono(song_wav, SAMPLE_RATE)
    duration = len(mix16) / SAMPLE_RATE
    print(f"      {duration / 60:.0f}:{duration % 60:04.1f} long")

    plan = FAST_PARTS if opts.fast else PARTS
    needed = {s for p in opts.parts for ps in plan[p]["passes"] for s in PASSES[ps]["stems"] if s != "mix"}
    stems = {}
    if opts.separator == "demucs":
        print(f"\n[2/{steps}] Splitting stems with StemSplitter (Demucs, {opts.shifts} shifts)")
        splitter = separate.load_splitter(separate.find_stemsplitter())
        if needed:
            with tempfile.TemporaryDirectory() as tmp:
                # Demucs names its output folder after the input file.
                link = Path(tmp) / f"{name}.wav"
                link.symlink_to(song_wav)
                stems = separate.separate(splitter, link, name, device, opts.shifts, opts.resplit)
    else:
        print(f"\n[2/{steps}] Splitting stems (BS-RoFormer)")
        if needed:
            from . import separate_roformer
            from .transcribe import print_progress

            stems = separate_roformer.separate(
                song_wav, cache, device, opts.resplit, print_progress("separating", "chunks")
            )

    opts = replace(opts, parts=_audible_parts(opts.parts, plan, stems, mix16))

    print(f"\n[3/{steps}] Finding the beat")
    beats_file = cache / "beats_v2.json"
    if beats_file.is_file():
        b = json.loads(beats_file.read_text())
        beats, bar_starts = np.array(b["beats"]), np.array(b["bar_starts"], dtype=int)
    else:
        beats, bar_starts = rhythm.track_beats(load_mono(song_wav, 22050), 22050, device)
        beats_file.write_text(json.dumps({"beats": beats.tolist(), "bar_starts": bar_starts.tolist()}))
    if len(beats) >= 2:
        bpm = 60.0 / float(np.median(np.diff(beats)))
        print(f"      ~{bpm:.0f} BPM, {len(bar_starts)} bars")

    print(f"\n[4/{steps}] Transcribing notes (MuScriptor {opts.model_size})")
    notes, guitar_groups = _transcribe_parts(opts, stems, mix16, cache, device)

    print(f"\n[5/{steps}] Writing tabs")
    files = _write_outputs(name, input_path, notes, guitar_groups, beats, bar_starts, duration, opts, out_dir)
    for f in files:
        print(f"      {f.name}")
    print(f"\nDone! Tabs are in: {out_dir}")
    return out_dir


def _transcribe_parts(opts, stems, mix16, cache: Path, device: str):
    """Run the transcription passes and vote. Returns the notes per part, and
    the guitar notes grouped by stereo position (None when not split)."""
    from . import stereo
    from .ensemble import lowest_only, merge, normalize_drums
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

    half = device.startswith("cuda") and not opts.full_precision
    results: dict[str, list[TNote]] = {}
    todo: dict[tuple, list[str]] = {}  # instrument restriction -> passes to run
    paths = {}
    for name in pass_names:
        spec = PASSES[name]
        inst = "any" if spec["instruments"] is None else "+".join(spec["instruments"])
        where = f"@{spec['position']}" if "position" in spec else ""
        source = "mix" if spec["stems"] == ["mix"] else f"{opts.separator}-{'+'.join(spec['stems'])}"
        shift = f"_s{spec['shift']}" if spec.get("shift") else ""
        key = (
            f"v{CACHE_VERSION}_{source}{where}_{inst}{shift}_{opts.model_size}"
            f"_b{opts.beam_size}{'_fp16' if half else ''}"
        )
        paths[name] = cache / f"notes_{key}.json"
        if paths[name].is_file() and not opts.retranscribe:
            results[name] = [TNote(**n) for n in json.loads(paths[name].read_text())]
        else:
            todo.setdefault(tuple(spec["instruments"] or ()), []).append(name)
    if len(results):
        print(f"      reusing {len(results)} earlier transcription(s)")

    def audio_for(name):
        nonlocal positions_audio
        spec = PASSES[name]
        if "position" in spec:
            if positions_audio is None:
                positions_audio = stereo.split_positions(guitar_stereo[0])
            import soxr

            audio = soxr.resample(positions_audio[spec["position"]], guitar_stereo[1], SAMPLE_RATE, quality="VHQ")
        elif spec["stems"] == ["mix"]:
            audio = mix16
        else:
            audio = mix_stems(stems, spec["stems"], SAMPLE_RATE)
        shift = spec.get("shift", 0.0)
        if shift:
            # Silence in front moves the model's 5-second chunk boundaries,
            # so this pass makes its boundary mistakes somewhere else.
            audio = np.concatenate([np.zeros(int(round(shift * SAMPLE_RATE)), dtype=np.float32), audio])
        return audio.astype(np.float32)

    if todo:
        import torch

        transcriber = Transcriber(
            opts.model_size, device, opts.weights, dtype=torch.float16 if half else torch.float32
        )
        batch_size = max(1, opts.batch if opts.beam_size == 1 else 1)
        batches = [
            (inst, names[i: i + batch_size])
            for inst, names in todo.items()
            for i in range(0, len(names), batch_size)
        ]
        n_total = sum(len(names) for _, names in batches)
        done = 0
        for inst, names in batches:
            label = f"passes {done + 1}-{done + len(names)}/{n_total}" if len(names) > 1 else f"pass {done + 1}/{n_total}"
            audios = [audio_for(n) for n in names]
            if opts.beam_size > 1:
                outs = [transcriber.transcribe(audios[0], list(inst) or None, opts.beam_size, print_progress(label))]
            else:
                outs = transcriber.transcribe_many(audios, list(inst) or None, print_progress(label))
            for name, notes in zip(names, outs):
                shift = PASSES[name].get("shift", 0.0)
                if shift:
                    for n in notes:
                        n.onset -= shift
                        n.offset -= shift
                    notes = [n for n in notes if n.onset >= 0]
                paths[name].write_text(json.dumps([n.to_json() for n in notes]))
                results[name] = notes
            done += len(names)
        del transcriber
        torch.cuda.empty_cache()

    out: dict[str, list[TNote]] = {}
    for part in opts.parts:
        per_pass = [[n for n in results[p] if _wanted(part, n)] for p in plan[part]["passes"]]
        if part == "drums":
            per_pass = [normalize_drums(ns) for ns in per_pass]
        if len(per_pass) == 1:
            out[part] = list(per_pass[0])
        else:
            out[part] = merge(per_pass, plan[part]["min_votes"], tol=plan[part].get("tolerance", 0.05))
        if plan[part].get("lowest_only"):
            out[part] = lowest_only(out[part])
        counts = ", ".join(str(len(x)) for x in per_pass)
        print(f"      {part}: {len(out[part])} notes agreed on (per pass: {counts})")
        if part in KEY_PARTS and len(out[part]) < MIN_KEY_NOTES:
            print(f"      no {part} to speak of: left out")
            del out[part]

    guitar_groups = None
    if guitar_stereo is not None and "guitar" in out:
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
    major.sort(key=lambda k: -counts[k])
    # One guitarist going from a clean verse to a distorted chorus is still
    # one part: a second sound only gets its own part when it mostly plays
    # at the same time as the main one.
    window = 2.0
    n_win = int(max(n.onset for n in notes) // window) + 1
    active = {
        k: np.bincount([int(n.onset // window) for n in notes if n.instrument == k], minlength=n_win) >= 2
        for k in major
    }
    major = major[:1] + [
        k for k in major[1:]
        if active[k].sum() and (active[k] & active[major[0]]).sum() >= 0.5 * active[k].sum()
    ]
    if len(major) <= 1:
        return [("", notes)]
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


def _separate_layers(groups, bm):
    """Move a line that another guitar plays on top of a riff out of the
    riff's part (see arrange.split_layers): into the part that already plays
    in that register if there is one, else into a part of its own."""
    mains, tops = [], []
    for flavour, gnotes in groups:
        q = rhythm.quantize_each(bm, gnotes, bm.subdivisions)
        main, top = arrange.split_layers(gnotes, bm, q)
        if len(main) < MIN_PART_NOTES:  # nothing left underneath: not a layer
            main, top = gnotes, []
        mains.append((flavour, main))
        tops.append(top)
    out = [(f, list(ns)) for f, ns in mains]
    for gi, top in enumerate(tops):
        if not top:
            continue
        pitch = float(np.median([n.pitch for n in top]))
        near = [
            (abs(float(np.median([n.pitch for n in ns])) - pitch), j)
            for j, (_, ns) in enumerate(mains) if j != gi and ns
        ]
        near = [x for x in near if x[0] <= 7]
        if near:
            out[min(near)[1]][1].extend(top)
        elif len(top) >= MIN_PART_NOTES:
            where = mains[gi][0]
            out.append((", ".join(x for x in (where, "on top") if x), top))
        # A handful of notes on top with nowhere to go are left out.
    return [(f, sorted(ns, key=lambda n: (n.onset, n.pitch))) for f, ns in out if ns]


def _part_grid(bm, notes, shuffle):
    """The beat map with subdivisions chosen for this part's own notes, so a
    busy guitar doesn't force a fine grid onto a simple drum beat."""
    onsets = np.array([n.onset for n in notes])
    return replace(bm, subdivisions=rhythm.choose_subdivisions(bm, onsets, shuffle))


def _events(notes, bm, report: "arrange.Report | None" = None):
    """Quantized notes grouped into chords: (ticks, notes per tick, fretboard events).
    With a report, the notes are tidied first (see arrange.py)."""
    q = rhythm.quantize(bm, notes, bm.subdivisions)
    if report is not None:
        q = arrange.remove_overtones(q, report)
        q = arrange.regularize_chords(q, report)
        q = arrange.unify_repeats(q, bm, report)
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


def _fretted_part(title, notes, bm, tunings_table_bass: bool, forced_tuning, program_default, band_tuning=None,
                  tidy: bool = True):
    """Quantize, choose tuning, finger, and build text + GP data for one part."""
    report = arrange.Report() if tidy else None
    ticks, by_tick, events = _events(notes, bm, report)
    if forced_tuning:
        tuning = fretboard.find_tuning(forced_tuning, tunings_table_bass)
    else:
        tuning = fretboard.choose_tuning(events, tunings_table_bass, band_tuning=band_tuning)
    moved = fretboard.fit_to_range(events, tuning)
    max_fret = 24
    fingerings, _ = fretboard.assign_frets(events, tuning, max_fret)
    out_of_reach = 0
    if tidy:
        fingerings = arrange.check_playable(events, fingerings, report)
        out_of_reach = report.out_of_reach
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
        "dropped": dropped - out_of_reach,
        "moved": moved,
        "tuning_name": tuning.name,
        "tidy": report.lines() if report else [],
    }
    gp_part = gp_export.GPPart(title, gp_events, tuning.strings, program, max_fret, bm=bm)
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
    return lines, events, gp_export.GPPart("Drums", gp_events, None, 0, bm=bm)


MIDDLE_C = 60
MIN_HAND_SHARE = 0.15  # each hand needs this share of the notes for a two-track piano


def _keys_part(title, notes, bm, program, two_hands=True):
    """A keyboard part as keyboard music: ({tick: pitches} for the note roll,
    Guitar Pro tracks to read as notation). A piano that uses both halves of
    the keyboard gets a right-hand and a left-hand track."""
    q = rhythm.quantize(bm, notes, bm.subdivisions)
    by_tick: dict[int, list] = {}
    for n in q:
        by_tick.setdefault(n.onset_tick, []).append(n)
    pitches_at = {t: sorted({n.pitch for n in ns}) for t, ns in by_tick.items()}
    low = sum(1 for n in q if n.pitch < MIDDLE_C)
    hands = [("", lambda p: True)]
    if two_hands and q and MIN_HAND_SHARE <= low / len(q) <= 1 - MIN_HAND_SHARE:
        hands = [(" (right hand)", lambda p: p >= MIDDLE_C), (" (left hand)", lambda p: p < MIDDLE_C)]
    parts = []
    for suffix, mine in hands:
        gp_events = []
        for t in sorted(by_tick):
            group = [n for n in by_tick[t] if mine(n.pitch)]
            placed = keys.place_chord([n.pitch for n in group])
            if placed:
                gp_events.append(gp_export.GPEvent(t, max(n.end_tick for n in group), placed))
        parts.append(gp_export.GPPart(title + suffix, gp_events, keys.KEYBOARD_STRINGS, program, keys.KEYBOARD_FRETS, bm=bm))
    return pitches_at, parts


def _write_outputs(name, input_path, notes, guitar_groups, beats, bar_starts, duration, opts, out_dir: Path) -> list[Path]:
    delay = _correct_onset_delay(notes, beats)
    bm = rhythm.build_beatmap(beats, bar_starts, duration)
    all_onsets = np.array([n.onset for ns in notes.values() for n in ns])
    if rhythm.detect_compound(bm, all_onsets):
        bm.compound = True
    shuffle = rhythm.is_shuffle(bm, all_onsets)
    bm.subdivisions = rhythm.choose_subdivisions(bm, all_onsets, shuffle)

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
    note_line = "Transcribed automatically (TabGenerator: BS-RoFormer + MuScriptor). Check it by ear!"

    files: list[Path] = []
    gp_parts: list[gp_export.GPPart] = []

    # The number of guitar parts can change between runs; clear old ones.
    for old in out_dir.glob(f"{glob_escape(name)} - Guitar*.txt"):
        old.unlink()

    for part in ("Bass", "Drums"):
        stale = out_dir / f"{name} - {part}.txt"
        if part.lower() not in notes and stale.is_file():
            stale.unlink()

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
        if opts.tidy and opts.split_guitars:
            groups = _separate_layers(groups, bm)
        for gi, (flavour, gnotes) in enumerate(groups):
            title = "Guitar" if len(groups) == 1 else f"Guitar {gi + 1} ({flavour or 'main'})"
            part_bm = _part_grid(bm, gnotes, shuffle)
            labels, events, gp_part, info = _fretted_part(
                title, gnotes, part_bm, False, guitar_tuning, 30, tidy=opts.tidy
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
            footer += info["tidy"]
            path = out_dir / f"{name} - {title}.txt"
            path.write_text(tab_text.render(header, labels, part_bm, events, footer))
            files.append(path)

    if "bass" in notes:
        part_bm = _part_grid(bm, notes["bass"], shuffle)
        labels, events, gp_part, info = _fretted_part(
            "Bass", notes["bass"], part_bm, True, opts.bass_tuning, 33, guitar_tuning, tidy=opts.tidy
        )
        gp_parts.append(gp_part)
        header = base_header + ["Part: Bass", f"Tuning: {info['tuning']}", note_line]
        footer = []
        if info["moved"]:
            footer.append(f"({info['moved']} notes below the lowest string were moved up an octave.)")
        if info["dropped"]:
            footer.append(f"({info['dropped']} notes did not fit on the neck and were left out.)")
        footer += info["tidy"]
        path = out_dir / f"{name} - Bass.txt"
        path.write_text(tab_text.render(header, labels, part_bm, events, footer))
        files.append(path)

    if "drums" in notes:
        part_bm = _part_grid(bm, notes["drums"], shuffle)
        labels, events, gp_part = _drum_part(notes["drums"], part_bm)
        gp_parts.append(gp_part)
        header = base_header + ["Part: Drums", note_line]
        path = out_dir / f"{name} - Drums.txt"
        path.write_text(tab_text.render(header, labels, part_bm, events, [""] + tab_text.DRUM_LEGEND))
        files.append(path)

    for part, default_program in (("piano", 0), ("synth", 80), ("vocals", 53)):
        for old in out_dir.glob(f"{glob_escape(name)} - {part.capitalize()}*.txt"):
            old.unlink()
        if part not in notes:
            continue
        title = part.capitalize()
        knotes = notes[part]
        part_bm = _part_grid(bm, knotes, shuffle)
        program = keys.program_for(knotes, default_program)
        sounds = f"Sounds like: {keys.heard_as(knotes)}" if part != "vocals" else "The sung melody"
        write_as = opts.vocals_as if part == "vocals" else opts.keys_as
        if write_as in ("piano", "both"):
            pitches_at, new_parts = _keys_part(title, knotes, part_bm, program, two_hands=part == "piano")
            gp_parts += new_parts
            header = base_header + [f"Part: {title}", sounds, note_line,
                                    'Each line is one pitch (C4 = middle C); "o" marks where a note starts.']
            path = out_dir / f"{name} - {title}.txt"
            path.write_text(tab_text.render_roll(header, part_bm, pitches_at))
            files.append(path)
        if write_as in ("guitar", "both"):
            tuning = fretboard.find_tuning(guitar_tuning or "E Standard", False)
            folded, moved = keys.fold_notes(knotes, tuning.lowest, tuning.strings[-1] + 19)
            labels, events, gp_part, info = _fretted_part(
                f"{title} (guitar tab)", folded, part_bm, False, tuning.name, program, tidy=opts.tidy
            )
            gp_part.program = program
            gp_parts.append(gp_part)
            header = base_header + [
                f"Part: {title}, arranged as guitar tablature", sounds, f"Tuning: {info['tuning']}", note_line,
            ]
            footer = []
            if moved:
                footer.append(f"({moved} notes outside the guitar's range were moved by octaves.)")
            if info["dropped"]:
                footer.append(f"({info['dropped']} notes did not fit under one hand and were left out.)")
            footer += info["tidy"]
            path = out_dir / f"{name} - {title} (guitar tab).txt"
            path.write_text(tab_text.render(header, labels, part_bm, events, footer))
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
            if part in KEY_PARTS:
                program = keys.program_for(ns, {"piano": 0, "vocals": 53}.get(part, 80))
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
