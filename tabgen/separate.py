"""Stem separation through the StemSplitter project next door (Demucs htdemucs_6s)."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

STEMS = ("drums", "bass", "guitar", "piano", "vocals", "other")


def find_stemsplitter() -> Path:
    env = os.environ.get("STEMSPLITTER_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "StemSplitter"


def load_splitter(root: Path):
    """Import StemSplitter's splitter.py to reuse its model choice and paths."""
    path = root / "splitter.py"
    if not path.is_file():
        raise SystemExit(
            f"StemSplitter not found at {root}. Set STEMSPLITTER_DIR to its folder."
        )
    spec = importlib.util.spec_from_file_location("stemsplitter_splitter", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stem_dir(splitter, name: str) -> Path:
    return Path(splitter.OUTPUT_DIR) / splitter.MODEL_NAME / name


def separate(splitter, wav: Path, name: str, device: str, shifts: int, force: bool = False) -> dict[str, Path]:
    """Split `wav` (named `name`.wav) into stems, reusing earlier results."""
    out = stem_dir(splitter, name)
    stems = {s: out / f"{s}.wav" for s in STEMS}
    if not force and all(p.is_file() for p in stems.values()):
        print(f"      Reusing stems in {out}")
        return stems
    if wav.stem != name:
        raise ValueError("Demucs names its output after the input file")
    cmd = [
        sys.executable, "-m", "demucs.separate",
        "-n", splitter.MODEL_NAME,
        "-o", splitter.OUTPUT_DIR,
        "-d", device,
        # Averaging over random time shifts gives cleaner stems (slower).
        "--shifts", str(shifts),
        str(wav),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0 or not all(p.is_file() for p in stems.values()):
        raise SystemExit("Demucs failed; see its output above.")
    return stems
