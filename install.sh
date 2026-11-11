#!/usr/bin/env bash
# TabGenerator installer: puts every Python package into ./_packages (no
# virtual environment, nothing installed system-wide), then fetches the
# transcription model.
#
#   ./install.sh               install (GPU build of PyTorch if an NVIDIA GPU is found)
#   ./install.sh --cpu         force the CPU build of PyTorch
#   ./install.sh --no-models   skip downloading the model weights now
#
# Safe to run again: it upgrades what is there.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
pkgs="$here/_packages"
cpu=0
models=1
for arg in "$@"; do
    case "$arg" in
        --cpu) cpu=1 ;;
        --no-models) models=0 ;;
        -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
fail() { printf '\n\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# --- system requirements ----------------------------------------------------
command -v python3 >/dev/null || fail "python3 is not installed."
python3 - <<'PY' || fail "Python 3.10 or newer is needed."
import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)
PY
python3 -m pip --version >/dev/null 2>&1 || fail "pip is missing (Debian/Ubuntu: sudo apt install python3-pip; Arch: sudo pacman -S python-pip)."
command -v ffmpeg >/dev/null || fail "ffmpeg is not installed (Debian/Ubuntu: sudo apt install ffmpeg; Arch: sudo pacman -S ffmpeg; Fedora: sudo dnf install ffmpeg)."
python3 -c "import ctypes.util, sys; sys.exit(0 if ctypes.util.find_library('sndfile') else 1)" 2>/dev/null \
    || echo "note: libsndfile not found on the system; the bundled copy from the soundfile wheel will be used."

index=cpu
if [ "$cpu" = 0 ] && command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
    index=cu130
    say "NVIDIA GPU found: $(nvidia-smi -L | head -1 | sed 's/ (UUID.*//')"
else
    say "No NVIDIA GPU in use: installing the CPU build (works, but slowly; consider --model small)"
fi

pip_install() {
    PIP_DISABLE_PIP_VERSION_CHECK=1 python3 -m pip install --upgrade --no-warn-conflicts --target "$pkgs" "$@"
}

# --- packages ---------------------------------------------------------------
mkdir -p "$pkgs"
say "Installing PyTorch ($index build, about 3 GB with the GPU libraries)"
pip_install --index-url "https://download.pytorch.org/whl/$index" "torch==2.14.0"

say "Installing the libraries TabGenerator uses"
pip_install "numpy>=2" einops "huggingface_hub>=0.30" "safetensors>=0.4" packaging tqdm pyyaml typing_extensions
# These declare dependencies TabGenerator doesn't need (web servers, training
# tools), so they go in without them.
pip_install --no-deps --extra-index-url "https://download.pytorch.org/whl/$index" -r "$here/requirements.txt"

say "Checking the installation"
python3 -I - "$here" <<'PY'
import sys
here = sys.argv[1]
sys.path[:0] = [f"{here}/_packages", here]
import torch, numpy, muscriptor, beat_this, guitarpro, soundfile, soxr, mido, einops  # noqa: F401
print(f"PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
PY

# --- model weights ----------------------------------------------------------
if [ "$models" = 1 ]; then
    say "Fetching the MuScriptor transcription model (large, about 5.6 GB)"
    cat <<'TXT'
The MuScriptor weights are free but gated (CC BY-NC 4.0, non-commercial):
  1. accept the license at https://huggingface.co/MuScriptor/muscriptor-large
  2. log in to Hugging Face below (or set HF_TOKEN), unless you already did.
TXT
    python3 -I - "$here" <<'PY' || echo "Skipped: the model will be downloaded on the first run instead (after logging in)."
import sys
here = sys.argv[1]
sys.path[:0] = [f"{here}/_packages", here]
from huggingface_hub import get_token, login
if not get_token() and sys.stdin.isatty():
    login()
from tabgen.transcribe import resolve_weights
w = resolve_weights("large")
if w == "large":  # no local copy: download through the Hugging Face cache
    import muscriptor.transcription_model as tm
    print("saved to", tm.download_if_necessary(tm._resolve_source("large")))
else:
    print("using", w)
PY
    echo "The stem separator (BS-RoFormer, 700 MB) is downloaded on the first run."
fi

chmod +x "$here/launch.sh"
say "Done. Try:  \"$here/launch.sh\" path/to/song.flac"
