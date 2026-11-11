"""A learned, context-aware guess of which string each note is played on.

A small bidirectional GRU reads a part as a sequence of notes (each note's
height above the lowest open string, how many notes sound with it and where
it sits in that chord, the time since the previous chord, its length, and
the tuning's layout) and gives, for every note, a probability for each
string. It is trained on tens of thousands of human-made tabs
(eval/train_fingering_model.py). Because it sees the notes before and after,
it can tell an open-position riff from the same notes higher up the neck.

fretboard.assign_frets uses these probabilities as one of its costs; the
hand-movement search still decides, so the result stays playable.
"""

from pathlib import Path

import numpy as np

WEIGHTS = Path(__file__).with_name("data") / "fingering_model.pt"
MAX_STRINGS = 7
MAX_HEIGHT = 96
MAX_FRET = 24
LAYOUTS = {
    (5, 5, 5, 4, 5): 0,  # standard (E, Eb, D, C# ... standard)
    (7, 5, 5, 4, 5): 1,  # drop
    (5, 5, 5, 5, 4, 5): 2,  # 7-string standard
    (7, 5, 5, 5, 4, 5): 3,  # 7-string drop
}
N_LAYOUTS = 5  # 4 = anything else (open tunings, ...)
DT_EDGES = np.array([0.0, 0.03, 0.06, 0.09, 0.12, 0.16, 0.2, 0.25, 0.32, 0.4, 0.5, 0.65, 0.8, 1.0, 1.5, 2.5])
DUR_EDGES = np.array([0.0, 0.05, 0.08, 0.11, 0.15, 0.2, 0.25, 0.32, 0.4, 0.5, 0.65, 0.8, 1.0, 1.4, 2.0, 3.0])


def layout_id(opens) -> int:
    opens = sorted(opens)
    return LAYOUTS.get(tuple(b - a for a, b in zip(opens, opens[1:])), 4)


def bucket(x, edges):
    return np.clip(np.searchsorted(edges, np.asarray(x, dtype=np.float32), side="right") - 1, 0, len(edges) - 1)


def string_mask(height, opens):
    """[n, 7] True where the note fits on that string (fret 0..24)."""
    opens = np.asarray(opens)
    n = int(np.sum(opens > 0)) if opens[-1] == 0 else len(opens)
    rel = (opens[:n] - opens[0])[None, :]
    fret = np.asarray(height)[:, None] - rel
    mask = np.zeros((len(height), MAX_STRINGS), dtype=bool)
    mask[:, :n] = (fret >= 0) & (fret <= MAX_FRET)
    return mask


def build(torch):
    nn = torch.nn

    class Net(nn.Module):
        def __init__(self, hidden=192):
            super().__init__()
            self.height = nn.Embedding(MAX_HEIGHT, 48)
            self.size = nn.Embedding(8, 8)
            self.index = nn.Embedding(8, 8)
            self.layout = nn.Embedding(N_LAYOUTS, 8)
            self.dt = nn.Embedding(len(DT_EDGES), 8)
            self.dur = nn.Embedding(len(DUR_EDGES), 8)
            self.inp = nn.Linear(48 + 8 * 5, 128)
            self.gru = nn.GRU(128, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
            self.out = nn.Sequential(nn.Linear(2 * hidden + 48, 128), nn.GELU(), nn.Linear(128, MAX_STRINGS))

        def forward(self, height, size, index, layout, dt, dur):
            h = self.height(height.clamp(0, MAX_HEIGHT - 1))
            x = torch.cat([h, self.size(size.clamp(0, 7)), self.index(index.clamp(0, 7)), self.layout(layout),
                           self.dt(dt), self.dur(dur)], dim=-1)
            y, _ = self.gru(torch.relu(self.inp(x)))
            return self.out(torch.cat([y, h], dim=-1))

    return Net


_MODEL = None


def load():
    """The trained model on the CPU, or None if there is none (or no torch)."""
    global _MODEL
    if _MODEL is None:
        _MODEL = False
        if WEIGHTS.is_file():
            try:
                import torch

                net = build(torch)()
                state = torch.load(str(WEIGHTS), map_location="cpu", weights_only=True)
                net.load_state_dict({k: v.float() for k, v in state.items()})
                net.eval()
                _MODEL = (torch, net)
            except Exception:
                _MODEL = False
    return _MODEL or None


def features(events, opens):
    """Arrays for the model from fretboard.Events (pitches sorted ascending)."""
    height, size, index, dt, dur = [], [], [], [], []
    prev = None
    lowest = min(opens)
    for e in events:
        ends = e.ends or [e.time] * len(e.pitches)
        for k, (p, end) in enumerate(zip(e.pitches, ends)):
            height.append(p - lowest)
            size.append(len(e.pitches))
            index.append(k)
            dt.append((4.0 if prev is None else e.time - prev) if k == 0 else 0.0)
            dur.append(max(0.0, end - e.time))
        prev = e.time
    return (np.array(height), np.array(size), np.array(index), bucket(dt, DT_EDGES), bucket(dur, DUR_EDGES))


def string_logprobs(events, opens) -> list | None:
    """Per event, {pitch: log-probability per string (lowest first)}; None without a model."""
    m = load()
    if m is None or not events or len(opens) not in (6, 7):
        return None
    torch, net = m
    height, size, index, dt, dur = features(events, opens)
    padded = list(sorted(opens)) + [0] * (MAX_STRINGS - len(opens))
    mask = string_mask(height, padded)
    lay = np.full(len(height), layout_id(opens))
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.long)[None]
    with torch.inference_mode():
        logits = net(t(np.clip(height, 0, MAX_HEIGHT - 1)), t(size), t(index), t(lay), t(dt), t(dur))[0].numpy()
    logits = np.where(mask, logits, -1e9)
    logp = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
    out, i = [], 0
    for e in events:
        d = {}
        for p in e.pitches:
            d[p] = logp[i, : len(opens)]
            i += 1
        out.append(d)
    return out
