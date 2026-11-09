#!/usr/bin/env bash
# Installs TabGenerator's extra packages into ./_packages (no virtual environment).
# PyTorch and Demucs are shared with StemSplitter; run its setup.sh first if needed.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
base_python="$(python3 -c 'import sys; print(getattr(sys, "_base_executable", sys.executable))')"
# --no-deps: the only heavy dependency (torch) must stay StemSplitter's CUDA build.
# torchaudio 2.11 is the last release; beat-this only uses its pure-PyTorch parts.
exec "$base_python" -m pip install --upgrade --no-deps --target "$script_dir/_packages" \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    -r "$script_dir/requirements.txt"
