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
SEGMENT_SECONDS = 5.0  # MuScriptor transcribes in chunks of this length
TOKEN_BUDGETS = (640, 2000)  # tokens per chunk: usual case, MuScriptor's own limit
VOCAB_SIZE = 1393

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
    conf: float = 1.0  # probability the model gave this note's pitch token

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
        self.long_chunks = 0  # chunks that needed more than the usual token budget
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
        if beam_size > 1:
            return self._transcribe_beam(mono16k, instruments, beam_size, progress)
        return self.transcribe_many([mono16k], instruments, progress)[0]

    def transcribe_many(
        self,
        audios: list[np.ndarray],
        instruments: list[str] | None = None,
        progress=None,
    ) -> list[list[TNote]]:
        """Transcribe several 16 kHz mono signals at once (e.g. different
        mixes of one song), one result per signal.

        The model writes notes token by token, and a step costs almost the
        same for a batch as for one signal, so this is several times faster
        than transcribing them one after another. The outcome is the same as
        MuScriptor's own greedy decoding with prelude forcing (each signal's
        5-second chunks are decoded in order, and the notes still sounding at
        a chunk boundary are forced into the next chunk's opening), plus a
        confidence per note: the probability the model gave its pitch token.
        """
        from muscriptor.events import ChunkBoundary, OpenNoteTracker, _DrumHit, _EndNote, _StartNote
        from muscriptor.tokenizer.mt3 import instrument_group_from_names

        tm = self.model
        lm, tok, device = tm._model, tm._tokenizer, tm._device
        seg = int(SEGMENT_SECONDS * SAMPLE_RATE)
        n_chunks = max(1, max(int(np.ceil(len(a) / seg)) for a in audios))
        wavs = []
        for a in audios:
            w = np.zeros(n_chunks * seg, dtype=np.float32)
            w[: len(a)] = a
            wavs.append(torch.from_numpy(w)[None].to(device))
        group = instrument_group_from_names(instruments) if instruments else None
        forbidden = None
        if instruments:
            forbidden = torch.tensor(tok.forbidden_token_ids(instruments), device=device, dtype=torch.long)
        K = len(audios)
        eos, ungen = tok.eos_id, lm.ungenerated_token_id
        trackers = [OpenNoteTracker(tok._vocab, tok.frame_rate) for _ in range(K)]
        open_notes: list[dict] = [{} for _ in range(K)]
        results: list[list[TNote]] = [[] for _ in range(K)]

        def apply(k, actions, conf):
            for act in actions:
                if isinstance(act, _StartNote):
                    open_notes[k][(act.program, act.pitch)] = (act.time, conf)
                elif isinstance(act, _EndNote):
                    onset, c = open_notes[k].pop((act.program, act.pitch))
                    results[k].append(
                        TNote(onset, act.time, act.pitch, tm._instrument_for_program(act.program), False, c)
                    )
                elif isinstance(act, _DrumHit):
                    results[k].append(TNote(act.time, act.time + 0.01, act.pitch, "drums", True, conf))

        if progress:
            progress(0, n_chunks)
        with torch.inference_mode(), self._maybe_quiet():
            for ci in range(n_chunks):
                bnd = ChunkBoundary(ci * SEGMENT_SECONDS, (ci + 1) * SEGMENT_SECONDS if ci + 1 < n_chunks else None)
                prompts = []
                for k in range(K):
                    # The boundary settles the tracker, so open_keys() is
                    # exactly what the next chunk must declare as sustained.
                    apply(k, trackers[k].feed(bnd), 1.0)
                    prompts.append(tok.tie_section_token_ids(trackers[k].open_keys()) if ci > 0 else [])
                conds = [tm._build_conditions(w[:, ci * seg: (ci + 1) * seg], group)[0] for w in wavs]
                cfg = lm.condition_provider(lm.condition_provider.tokenize(conds))
                # Nearly every chunk needs far fewer tokens than MuScriptor's
                # limit, and the attention cache is allocated for the limit:
                # start small, redo the rare chunk that doesn't fit.
                seq, prob, last, done = self._decode_chunk(lm, cfg, prompts, forbidden, eos, ungen, TOKEN_BUDGETS[0])
                rows, probs = seq[:, 1: last + 1].tolist(), prob[:, 1: last + 1].tolist()
                if not done:
                    # One signal at a time: the full-size cache for the whole
                    # batch would need several GB at once.
                    for k in range(K):
                        if eos in rows[k]:
                            continue
                        one = {name: (c[k: k + 1], m[k: k + 1]) for name, (c, m) in cfg.items()}
                        seq, prob, last, _ = self._decode_chunk(lm, one, [prompts[k]], forbidden, eos, ungen, TOKEN_BUDGETS[1])
                        rows[k], probs[k] = seq[0, 1: last + 1].tolist(), prob[0, 1: last + 1].tolist()
                        self.long_chunks += 1
                for k in range(K):
                    for t, c in zip(rows[k], probs[k]):
                        if t == eos:
                            break
                        apply(k, trackers[k].feed(t), c)
                if progress:
                    progress(ci + 1, n_chunks)
        out = []
        for k in range(K):
            apply(k, trackers[k].finish(), 1.0)
            out.append(_clean(results[k], len(audios[k]) / SAMPLE_RATE))
        return out

    def _decode_chunk(self, lm, cfg, prompts, forbidden, eos, ungen, max_len):
        """Greedy decoding of one chunk for every signal in the batch.
        Returns (tokens, their probabilities, steps taken, all finished)."""
        from muscriptor.modules.streaming import increment_steps, init_states

        device = lm.emb.weight.device
        K = len(prompts)
        prepend = sum(c.shape[1] for c, _ in cfg.values())
        state = init_states(lm, batch_size=K, sequence_length=prepend + max_len)
        seq = torch.full((K, max_len + 1), ungen, device=device, dtype=torch.long)
        seq[:, 0] = lm.initial_token_id
        for k, pr in enumerate(prompts):
            if pr:
                seq[k, 1: 1 + len(pr)] = torch.tensor(pr[:max_len], device=device)
        prob = torch.ones((K, max_len + 1), device=device)
        start = min(min(len(pr) for pr in prompts), max_len - 1)
        finished = torch.zeros(K, dtype=torch.bool, device=device)
        last = start
        with lm.autocast:
            for offset in range(start, max_len):
                first = offset == start
                inp = seq[:, : offset + 1] if first else seq[:, offset: offset + 1]
                logits = lm(inp, cfg, first_step=first, model_state=state)[:, -1, :].float()
                logits[:, VOCAB_SIZE:] = -torch.inf  # reserved tokens
                if forbidden is not None:
                    logits[:, forbidden] = -torch.inf
                increment_steps(lm.transformer, state, increment=inp.shape[-1] + (prepend if first else 0))
                p, nxt = torch.softmax(logits, dim=-1).max(dim=-1)
                forced = seq[:, offset + 1] != ungen
                nxt = torch.where(forced, seq[:, offset + 1], nxt)
                seq[:, offset + 1] = nxt
                prob[:, offset + 1] = torch.where(forced, torch.ones_like(p), p)
                finished |= nxt == eos
                last = offset + 1
                if bool(finished.all()):
                    break
        del state
        return seq, prob, last, bool(finished.all())

    def _transcribe_beam(self, mono16k, instruments, beam_size, progress) -> list[TNote]:
        """MuScriptor's own (unbatched) decoding, for beam search."""
        from muscriptor.events import NoteEndEvent, NoteStartEvent, ProgressEvent

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
        return _clean(notes, len(mono16k) / SAMPLE_RATE)


def _clean(notes: list[TNote], duration: float) -> list[TNote]:
    """The clean-up MuScriptor applies before writing MIDI: a minimum length,
    and a note ends where the next one of the same pitch and instrument
    starts. Notes past the end of the audio (padding) are dropped."""
    notes = [n for n in notes if n.onset < duration]
    by_key: dict = {}
    for n in notes:
        if not n.is_drum and n.offset - n.onset < 0.01:
            n.offset = n.onset + 0.01
        by_key.setdefault((n.instrument, n.pitch, n.is_drum), []).append(n)
    out = []
    for group in by_key.values():
        group.sort(key=lambda n: n.onset)
        for a, b in zip(group, group[1:]):
            if a.offset > b.onset:
                a.offset = b.onset
        out += [n for n in group if n.onset < n.offset]
    out.sort(key=lambda n: (n.onset, n.is_drum, n.pitch, n.instrument))
    return out


def print_progress(label: str, unit: str = "chunks"):
    out = sys.stdout  # the real terminal, before transcribe() silences MuScriptor

    def cb(done, total):
        width = 30
        filled = int(width * done / max(1, total))
        out.write(f"\r      {label}: [{'#' * filled}{'.' * (width - filled)}] {done}/{total} {unit}")
        out.flush()
        if done >= total:
            out.write("\n")

    return cb
