"""Audio -> notes with MuScriptor (Kyutai x Mirelo, 2026)."""

import contextlib
import io
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

SAMPLE_RATE = 16000

GUITARS = ["acoustic_guitar", "clean_electric_guitar", "distorted_electric_guitar"]
BASSES = ["electric_bass", "acoustic_bass"]
DRUMS = ["drums"]


@dataclass
class TNote:
    onset: float
    offset: float
    pitch: int
    instrument: str
    is_drum: bool

    def to_json(self):
        return asdict(self)


def resolve_weights(size: str, explicit: str | None = None) -> str:
    """A local weights file if there is one, else the size keyword (MuScriptor
    then downloads it from HuggingFace, which needs a login)."""
    candidates = [explicit, os.environ.get("MUSCRIPTOR_WEIGHTS"), f"~/muscriptor-{size}/model.safetensors"]
    for c in candidates:
        if not c:
            continue
        p = Path(c).expanduser()
        if p.is_dir():
            p = p / "model.safetensors"
        # A git clone without git-lfs leaves a tiny pointer file behind.
        if p.is_file() and p.stat().st_size > 1_000_000:
            return str(p)
        if c is explicit:
            raise SystemExit(f"No model weights at {p}")
    return size


def _load_model(weights: str, device: torch.device, dtype: torch.dtype):
    """TranscriptionModel.load_model, but reading the checkpoint into RAM
    first: loading it straight onto the GPU next to the freshly built model
    peaks at twice the model size (11 GB for large)."""
    import muscriptor.transcription_model as tm
    from safetensors.torch import load_file

    source = tm._resolve_source(weights)
    path = tm.download_if_necessary(source)
    model = tm._build_model(device, tm._resolve_config(source, path))
    model.eval()
    if dtype != torch.float32:
        model.to(dtype)
    state = tm._remap_single_codebook_keys(load_file(path, device="cpu"))
    model.load_state_dict(state)
    del state
    model.to(device)  # a few submodules are built on the CPU
    if dtype != torch.float32:
        # Conditioners keep fp32 numerics (log-mel of quiet passages underflows in fp16).
        model.condition_provider.float()
    tokenizer = tm.MT3Tokenizer(instrument_vocabulary="MT3_FULL_PLUS", max_shift_steps=1001)
    return tm.TranscriptionModel(model=model, tokenizer=tokenizer, device=device)


class Transcriber:
    def __init__(
        self,
        size: str = "large",
        device: str | None = None,
        weights: str | None = None,
        quiet: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        self.quiet = quiet
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        with self._maybe_quiet():
            self.model = _load_model(resolve_weights(size, weights), torch.device(device), dtype)

    @contextlib.contextmanager
    def _maybe_quiet(self):
        if not self.quiet:
            yield
            return
        # MuScriptor prints timing info for every chunk; keep our output clean.
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            yield

    def transcribe(
        self,
        mono16k: np.ndarray,
        instruments: list[str] | None = None,
        beam_size: int = 1,
        progress=None,
    ) -> list[TNote]:
        """Transcribe 16 kHz mono audio. `instruments` hard-limits what the
        model may output (and conditions it on those instruments)."""
        from muscriptor.events import NoteEndEvent, NoteStartEvent, ProgressEvent
        from muscriptor.tokenizer.notes import Note, trim_overlapping_notes, validate_notes

        wav = torch.from_numpy(np.ascontiguousarray(mono16k, dtype=np.float32))[None]
        open_notes: dict[int, TNote] = {}
        notes: list[TNote] = []
        with torch.inference_mode(), self._maybe_quiet():
            for ev in self.model.transcribe(
                (wav, SAMPLE_RATE), instruments=instruments, beam_size=beam_size
            ):
                if isinstance(ev, ProgressEvent):
                    if progress:
                        progress(ev.completed, ev.total)
                elif isinstance(ev, NoteStartEvent):
                    open_notes[ev.index] = TNote(
                        ev.start_time, ev.start_time, ev.pitch, ev.instrument, ev.instrument == "drums"
                    )
                elif isinstance(ev, NoteEndEvent):
                    n = open_notes.pop(ev.start_event_index)
                    n.offset = ev.end_time
                    notes.append(n)
        # Same clean-up MuScriptor applies before writing MIDI. It works on
        # program numbers, so give each instrument name its own.
        names = sorted({n.instrument for n in notes if not n.is_drum})
        program_of = {name: i for i, name in enumerate(names)}
        mnotes = [
            Note(is_drum=n.is_drum, program=128 if n.is_drum else program_of[n.instrument],
                 onset=n.onset, offset=n.offset, pitch=n.pitch)
            for n in notes
        ]
        mnotes = validate_notes(mnotes, fix=True)
        mnotes = trim_overlapping_notes(mnotes, sort=True)
        return [
            TNote(m.onset, m.offset, m.pitch, "drums" if m.is_drum else names[m.program], m.is_drum)
            for m in mnotes
        ]


def print_progress(label: str):
    out = sys.stdout  # the real terminal, before transcribe() silences MuScriptor

    def cb(done, total):
        width = 30
        filled = int(width * done / max(1, total))
        out.write(f"\r      {label}: [{'#' * filled}{'.' * (width - filled)}] {done}/{total} chunks")
        out.flush()
        if done >= total:
            out.write("\n")

    return cb
