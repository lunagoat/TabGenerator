"""Stem separation with BS-RoFormer SW (6 stems: bass, drums, other, vocals,
guitar, piano).

A band-split transformer; its guitar, bass and drum stems are much cleaner
than Demucs htdemucs_6s, which is what the transcription model then hears.
The weights (700 MB) are downloaded from Hugging Face on first use.
"""

import os
from pathlib import Path

import numpy as np
import torch

from .polite import Breather

NAME = "bs_roformer_sw"
HF_REPO = "Blakus/bs_roformer_sw_6stem"
HF_FILE = "BS-Rofo-SW-Fixed.ckpt"
SAMPLE_RATE = 44100
CHUNK = 588800  # samples the model was trained on (13.35 s)
STEM_ORDER = ("bass", "drums", "other", "vocals", "guitar", "piano")

_FREQS_PER_BANDS = (2,) * 24 + (4,) * 12 + (12,) * 8 + (24,) * 8 + (48,) * 8 + (128, 129)
_MODEL_ARGS = dict(
    dim=256, depth=12, stereo=True, num_stems=6,
    time_transformer_depth=1, freq_transformer_depth=1, linear_transformer_depth=0,
    freqs_per_bands=_FREQS_PER_BANDS, dim_head=64, heads=8,
    attn_dropout=0.0, ff_dropout=0.0, flash_attn=True,
    dim_freqs_in=1025, stft_n_fft=2048, stft_hop_length=512, stft_win_length=2048,
    stft_normalized=False, mask_estimator_depth=2, mlp_expansion_factor=4,
)


def weights_path() -> Path:
    env = os.environ.get("BS_ROFORMER_WEIGHTS")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "tabgenerator" / HF_FILE


def ensure_weights() -> Path:
    path = weights_path()
    if path.is_file() and path.stat().st_size > 100_000_000:
        return path
    from huggingface_hub import hf_hub_download

    print(f"      Downloading {HF_FILE} (700 MB, once) ...")
    path.parent.mkdir(parents=True, exist_ok=True)
    got = hf_hub_download(HF_REPO, HF_FILE, local_dir=str(path.parent))
    return Path(got)


def load_model(device: str):
    from .roformer.bs_roformer import BSRoformer

    model = BSRoformer(**_MODEL_ARGS)
    # weights_only: the checkpoint is a pickle from the internet.
    state = torch.load(str(ensure_weights()), map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    return model.eval().to(device)


def _fade_window(size: int, fade: int, first: bool, last: bool) -> torch.Tensor:
    w = torch.ones(size)
    if not first:
        w[:fade] = torch.linspace(0, 1, fade)
    if not last:
        w[-fade:] = torch.linspace(1, 0, fade)
    return w


@torch.inference_mode()
def demix(model, audio: np.ndarray, device: str, overlap: int = 2, progress=None) -> np.ndarray:
    """[T, 2] float audio at 44.1 kHz -> [6, T, 2] stems (STEM_ORDER)."""
    mix = torch.from_numpy(np.ascontiguousarray(audio.T, dtype=np.float32))  # [2, T]
    n = mix.shape[1]
    step = CHUNK // overlap
    border = CHUNK - step
    padded = n > 2 * border and border > 0
    if padded:
        mix = torch.nn.functional.pad(mix[None], (border, border), mode="reflect")[0]
    total = mix.shape[1]
    out = torch.zeros(len(STEM_ORDER), 2, total)
    weight = torch.zeros(total)
    fade = CHUNK // 10
    starts = list(range(0, total, step))
    use_amp = device.startswith("cuda")
    breathe = Breather(device)
    for i, start in enumerate(starts):
        part = mix[:, start: start + CHUNK]
        length = part.shape[1]
        if length < CHUNK:
            if length > CHUNK // 2 + 1:
                part = torch.nn.functional.pad(part[None], (0, CHUNK - length), mode="reflect")[0]
            else:
                part = torch.nn.functional.pad(part, (0, CHUNK - length))
        with torch.autocast("cuda", enabled=use_amp):
            est = model(part[None].to(device))[0].float().cpu()  # [6, 2, CHUNK]
        w = _fade_window(CHUNK, fade, i == 0, start + CHUNK >= total)
        out[:, :, start: start + length] += est[:, :, :length] * w[:length]
        weight[start: start + length] += w[:length]
        breathe()
        if progress:
            progress(i + 1, len(starts))
    out /= weight.clamp_min(1e-8)
    if padded:
        out = out[:, :, border: border + n]
    return out.permute(0, 2, 1).numpy()


def stem_dir(cache: Path) -> Path:
    return cache / f"stems_{NAME}"


def separate(wav: Path, cache: Path, device: str, force: bool = False, progress=None) -> dict[str, Path]:
    """Split `wav` into stems under the song's cache folder, reusing earlier results."""
    import soundfile as sf

    out = stem_dir(cache)
    stems = {s: out / f"{s}.wav" for s in STEM_ORDER}
    if not force and all(p.is_file() for p in stems.values()):
        print(f"      Reusing stems in {out}")
        return stems
    audio, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    if sr != SAMPLE_RATE:
        import soxr

        audio = soxr.resample(audio, sr, SAMPLE_RATE, quality="VHQ")
    model = load_model(device)
    try:
        est = demix(model, audio[:, :2], device, progress=progress)
    finally:
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    out.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(STEM_ORDER):
        tmp = stems[s].with_suffix(".tmp.wav")
        sf.write(str(tmp), est[i], SAMPLE_RATE, subtype="FLOAT")
        tmp.replace(stems[s])
    return stems
