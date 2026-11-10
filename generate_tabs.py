#!/usr/bin/env python3
"""Turn any audio file into guitar, bass and drum tabs.

Uses the base Python installation, without a virtual environment, like
StemSplitter. Packages come from this folder's _packages plus StemSplitter's
_packages (which provides PyTorch and Demucs).
"""
import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGES_DIR = SCRIPT_DIR / "_packages"


def use_base_python():
    """Restart with the base interpreter if launched from a virtual environment."""
    if sys.prefix == sys.base_prefix:
        return
    base_python = getattr(sys, "_base_executable", None)
    if not base_python or not os.path.isfile(base_python):
        raise SystemExit("Unable to locate base Python. Run this script with system Python.")
    env = os.environ.copy()
    for var in ("VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH"):
        env.pop(var, None)
    os.execve(base_python, [base_python, str(Path(__file__).resolve()), *sys.argv[1:]], env)


def setup_paths():
    sys.path.insert(0, str(SCRIPT_DIR))
    sys.path.insert(1, str(PACKAGES_DIR))
    from tabgen.separate import find_stemsplitter

    stem_packages = find_stemsplitter() / "_packages"
    if stem_packages.is_dir():
        sys.path.insert(2, str(stem_packages))
    missing = []
    for mod in ("torch", "einops", "muscriptor", "beat_this", "guitarpro", "soundfile", "soxr", "mido"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"\nMissing Python packages: {', '.join(missing)}")
        print(f'Run the setup script: "{SCRIPT_DIR / "setup.sh"}"')
        print("(PyTorch comes from StemSplitter; run its setup.sh if torch or einops is missing.)")
        raise SystemExit(1)


def main():
    use_base_python()
    parser = argparse.ArgumentParser(
        description="Generate guitar, bass, drum and keyboard tabs from an audio file.",
    )
    parser.add_argument("audio", nargs="*", help="audio file(s): flac, wav, mp3, m4a, ogg, opus, ...")
    parser.add_argument("--parts", default="guitar,bass,drums,keys",
                        help="comma-separated parts: guitar, bass, drums, piano, synth, keys (= piano + synth). "
                             "Default: all; parts that are not in the song are skipped")
    parser.add_argument("--keys-as", default="piano", choices=["piano", "guitar", "both"],
                        help="how to write piano/synth parts: piano = note roll and a notation track (default); "
                             "guitar = guitar tablature, with the notes moved into the guitar's range; both = both")
    parser.add_argument("--guitar-tuning", help='e.g. "Drop D", "Eb Standard" (default: detected)')
    parser.add_argument("--bass-tuning", help='e.g. "E Standard", "5-String B Standard" (default: detected)')
    parser.add_argument("--model", default="large", choices=["small", "medium", "large"],
                        help="MuScriptor size (default: large, the most accurate)")
    parser.add_argument("--weights", help="path to MuScriptor model.safetensors (default: ~/muscriptor-<size>/)")
    parser.add_argument("--beam", type=int, default=1, help="beam search width (default: 1)")
    parser.add_argument("--separator", default="roformer", choices=["roformer", "demucs"],
                        help="stem separation model: BS-RoFormer (default, cleaner stems) or Demucs htdemucs_6s via StemSplitter")
    parser.add_argument("--shifts", type=int, default=2, help="with --separator demucs: random shifts; more = cleaner stems, slower (default: 2)")
    parser.add_argument("--batch", type=int, default=8,
                        help="transcriptions decoded at once (default: 8; lower it if the GPU runs out of memory)")
    parser.add_argument("--full-precision", action="store_true", help="run the model in 32-bit floats on the GPU (slower, same results)")
    parser.add_argument("--device", help="cuda or cpu (default: cuda if available)")
    parser.add_argument("--fast", action="store_true",
                        help="one transcription pass per part instead of voting over several (~3x faster, a bit less accurate)")
    parser.add_argument("--one-guitar", action="store_true", help="never split guitars into separate parts")
    parser.add_argument("--raw", action="store_true",
                        help="write the notes as transcribed: no tidying of repeats, chords and overtones, no playability check")
    parser.add_argument("--resplit", action="store_true", help="redo stem separation even if stems exist")
    parser.add_argument("--retranscribe", action="store_true", help="ignore cached transcriptions")
    parser.add_argument("-o", "--output", help="output folder (default: Tabs/<song name>)")
    parser.add_argument("--no-open", action="store_true", help="don't open the output folder when done")
    args = parser.parse_args()

    setup_paths()
    from tabgen.pipeline import PART_GROUPS, PARTS, Options, open_folder, run

    from tabgen.fretboard import find_tuning

    for flag, name, bass in (("--guitar-tuning", args.guitar_tuning, False), ("--bass-tuning", args.bass_tuning, True)):
        if name:
            try:
                find_tuning(name, bass)
            except ValueError as e:
                parser.error(f"{flag}: {e}")

    asked = [p.strip().lower() for p in args.parts.split(",") if p.strip()]
    unknown = [p for p in asked if p not in PARTS and p not in PART_GROUPS]
    if unknown:
        parser.error(f"unknown part(s): {', '.join(unknown)} (choose from {', '.join([*PARTS, *PART_GROUPS])})")
    parts = tuple(dict.fromkeys(q for p in asked for q in PART_GROUPS.get(p, (p,))))

    files = args.audio
    interactive = not files
    if interactive:
        print("\n" + "=" * 60)
        print("   TAB GENERATOR  (guitar / bass / drums / keys)")
        print("=" * 60)
        print("\nTip: You can drag and drop a file onto this window.")
        entered = input("Paste audio file path: ").strip().strip('"').strip("'")
        files = [entered]

    for f in files:
        opts = Options(
            model_size=args.model,
            weights=args.weights,
            beam_size=args.beam,
            shifts=args.shifts,
            device=args.device,
            guitar_tuning=args.guitar_tuning,
            bass_tuning=args.bass_tuning,
            resplit=args.resplit,
            retranscribe=args.retranscribe,
            out_dir=Path(args.output) if args.output else None,
            parts=parts,
            split_guitars=not args.one_guitar,
            fast=args.fast,
            separator=args.separator,
            batch=max(1, args.batch),
            full_precision=args.full_precision,
            tidy=not args.raw,
            keys_as=args.keys_as,
        )
        out = run(Path(f), opts)
        if not args.no_open and len(files) == 1:
            open_folder(out)
    if interactive:
        input("\nPress Enter to exit...")


if __name__ == "__main__":
    sys.exit(main())
