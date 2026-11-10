"""Without ground truth: is a transcribed note actually in the audio?

A note is "supported" when, just after its onset, the stem has clearly more
energy at the note's fundamental than a semitone and a half either side and
more than it had just before. Random notes (right pitches, wrong times) give
the chance level. Used to compare voting settings on a song with no score.
"""
import numpy as np

from tabgen.stereo import HOP, N_FFT, _stft


class Spectrum:
    def __init__(self, wav_path):
        import soundfile as sf

        audio, self.sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
        self.S = np.abs(_stft(audio.mean(axis=1).astype(np.float64), np.hanning(N_FFT))) ** 2
        self.bin_hz = self.sr / N_FFT

    def _band(self, hz, a, b):
        c = hz / self.bin_hz
        lo, hi = max(1, int(round(c)) - 1), int(round(c)) + 2
        return float(self.S[a:b, lo:hi].sum())

    def supported(self, onset, pitch) -> bool:
        n = self.S.shape[0]
        a = int(np.clip(round(onset * self.sr / HOP), 0, n - 2))
        b = min(n, a + 5)  # ~115 ms
        f0 = 440.0 * 2 ** ((pitch - 69) / 12)
        here = self._band(f0, a, b)
        beside = 0.5 * (self._band(f0 * 2 ** (-1.5 / 12), a, b) + self._band(f0 * 2 ** (1.5 / 12), a, b))
        before = self._band(f0, max(0, a - 6), max(1, a - 1)) * (b - a) / max(1, min(5, a - 1))
        return here > 2.0 * beside and here > 1.25 * before

    def share(self, notes) -> float:
        return float(np.mean([self.supported(n.onset, n.pitch) for n in notes])) if notes else 0.0

    def chance(self, notes, seed=0) -> float:
        rng = np.random.default_rng(seed)
        if not notes:
            return 0.0
        t = rng.uniform(min(n.onset for n in notes), max(n.onset for n in notes), 2000)
        p = rng.choice([n.pitch for n in notes], 2000)
        return float(np.mean([self.supported(a, b) for a, b in zip(t, p)]))
