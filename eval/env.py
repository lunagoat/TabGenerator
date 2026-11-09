"""Import this first: puts the project and its package folders on sys.path."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT.parent / "StemSplitter" / "_packages", ROOT / "_packages", ROOT):
    sys.path.insert(0, str(p))
DATA = ROOT / "eval" / "data"
WORK = ROOT / "eval" / "work"
