#!/usr/bin/env python3
"""Train the context-aware string model (tabgen/fingering_model.py).

    python3 eval/fingering_data.py [--kind bass]     # once: build the note sequences
    python3 eval/train_fingering_model.py [--kind bass] [--steps 15000] [--noise 0.7] [--out FILE]

Trains on windows of 256 notes from every training track, reports the share
of validation notes put on the tab's string, and saves the weights (fp16) to
tabgen/data/fingering_model.pt (bass: fingering_model_bass.pt).

`--noise` is the share of training windows that are made to look like a
transcription instead of a clean tab: some notes missing, some extra ones
(octaves, fifths, neighbours of real notes, at the same time or just after),
note lengths off. The pipeline feeds the model such notes, not clean tabs.
"""
import argparse
import time

import env
import numpy as np
import torch

from tabgen import fingering_model as FM

DATA = env.WORK / "fingering"
FIELDS = ("height", "size", "index", "dt", "dur", "target", "layout")
# What a transcription adds next to a real note, in semitones from it.
GHOST_STEPS = np.array([12, 12, 12, -12, -12, 7, 19, 24, -5, 5, 1, -1, 2, -2, 3, -3, 4, -4])
MAX_DROP = 0.3  # up to this share of a window's notes goes missing
MAX_EXTRA = 0.2  # up to this many extra notes per real note


class Windows:
    def __init__(self, path, window=256, noise=0.0):
        z = np.load(path)
        self.ptr = z["ptr"]
        self.a = {k: z[k] for k in FIELDS}
        self.opens = z["opens"]
        lens = np.diff(self.ptr)
        self.tracks = np.nonzero(lens >= 16)[0]
        self.weights = np.minimum(lens[self.tracks], 4000).astype(float)
        self.weights /= self.weights.sum()
        self.window = window
        self.noise = noise

    def _clean(self, start, m):
        a = self.a
        s = slice(start, start + m)
        return (a["height"][s].astype(np.int64), a["size"][s].astype(np.int64), a["index"][s].astype(np.int64),
                a["dt"][s].astype(np.float32), a["dur"][s].astype(np.float32), a["target"][s].astype(np.int64))

    def _noisy(self, start, m, rng):
        """The window as a transcription might give it; extra notes get target -1."""
        h, _, idx, dt, dur, y = self._clean(start, m)
        first = idx == 0
        first[0] = True
        ev = np.cumsum(first) - 1
        t_ev = np.cumsum(np.minimum(dt[first].astype(np.float64), 4.0))
        t = t_ev[ev]
        keep = rng.random(m) >= rng.uniform(0, MAX_DROP)
        if not keep.any():
            keep[:] = True
        src = rng.integers(0, m, rng.binomial(m, rng.uniform(0, MAX_EXTRA)))
        gh = h[src] + rng.choice(GHOST_STEPS, len(src))
        gt = np.where(rng.random(len(src)) < 0.7, t[src], t[src] + rng.uniform(0.03, 0.4, len(src)))
        ok = (gh >= 0) & (gh < FM.MAX_HEIGHT)
        H = np.concatenate([h[keep], gh[ok]])
        T = np.concatenate([t[keep], gt[ok]])
        D = np.concatenate([dur[keep], dur[src][ok]]) * rng.lognormal(0.0, 0.5, len(H))
        Y = np.concatenate([y[keep], np.full(int(ok.sum()), -1)])
        order = np.lexsort((Y < 0, H, T))  # by time, then pitch, real notes before their copies
        H, T, D, Y = H[order], T[order], D[order], Y[order]
        dup = np.r_[False, (np.diff(T) < 1e-9) & (np.diff(H) == 0)]
        H, T, D, Y = H[~dup], T[~dup], D[~dup], Y[~dup]
        n = len(H)
        new = np.r_[True, np.diff(T) > 1e-9]
        e = np.cumsum(new) - 1
        size = np.bincount(e)[e]
        pos = np.arange(n)
        index = pos - np.maximum.accumulate(np.where(new, pos, 0))
        gaps = np.r_[min(float(dt[0]), 4.0) if first[0] else 4.0, np.diff(T)]
        return H, size, index, np.where(new, gaps, 0.0).astype(np.float32), D.astype(np.float32), Y

    def batch(self, n, rng, noise=None):
        W = self.window
        noise = self.noise if noise is None else noise
        cols = [np.zeros((n, W), dtype=np.int64) for _ in range(3)]
        dt, dur = np.zeros((n, W), dtype=np.float32), np.zeros((n, W), dtype=np.float32)
        y = np.zeros((n, W), dtype=np.int64)
        valid = np.zeros((n, W), dtype=bool)
        mask = np.zeros((n, W, FM.MAX_STRINGS), dtype=bool)
        layout = np.zeros((n, W), dtype=np.int64)
        for b, t in enumerate(rng.choice(self.tracks, size=n, p=self.weights)):
            lo, hi = self.ptr[t], self.ptr[t + 1]
            start = lo if hi - lo <= W else rng.integers(lo, hi - W + 1)
            m = min(W, hi - start)
            row = self._noisy(start, m, rng) if rng.random() < noise else self._clean(start, m)
            m = min(W, len(row[0]))
            for dst, src in zip((*cols, dt, dur, y), row):
                dst[b, :m] = src[:m]
            valid[b, :m] = y[b, :m] >= 0
            mask[b, :m] = FM.string_mask(cols[0][b, :m], self.opens[t])
            layout[b] = self.a["layout"][lo]
        T = lambda x: torch.as_tensor(x, dtype=torch.long)
        return (T(np.clip(cols[0], 0, FM.MAX_HEIGHT - 1)), T(np.clip(cols[1], 0, 7)), T(np.clip(cols[2], 0, 7)),
                T(layout), T(FM.bucket(dt, FM.DT_EDGES)), T(FM.bucket(dur, FM.DUR_EDGES)), T(np.maximum(y, 0)),
                torch.as_tensor(mask & valid[..., None]), torch.as_tensor(valid))


def loss_acc(net, batch, dev):
    h, s, i, l, dt, du, y, mask, valid = [x.to(dev) for x in batch]
    logits = net(h, s, i, l, dt, du).masked_fill(~mask, -1e4)
    ok = valid & mask.gather(-1, y.clamp(0, FM.MAX_STRINGS - 1)[..., None])[..., 0]
    loss = torch.nn.functional.cross_entropy(logits[ok], y[ok])
    acc = (logits.argmax(-1) == y)[ok].float().mean()
    return loss, acc


def evaluate(net, data, dev, rng, n=60, noise=0.0):
    net.eval()
    accs = []
    with torch.no_grad():
        for _ in range(n):
            accs.append(loss_acc(net, data.batch(64, rng, noise), dev)[1].item())
    net.train()
    return float(np.mean(accs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="guitar", choices=["guitar", "bass"])
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--noise", type=float, default=0.7, help="share of training windows made to look like a transcription")
    ap.add_argument("--dropout", type=float, default=0.3,
                    help="the tab collections hold many copies of popular songs, which a small dropout lets the model memorize")
    ap.add_argument("--out", help="where to save the weights (default: the file the pipeline loads)")
    a = ap.parse_args()
    prefix = "" if a.kind == "guitar" else f"{a.kind}_"
    out = a.out or str(FM.WEIGHTS[a.kind])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    t0 = time.time()
    train, val = Windows(DATA / f"{prefix}train.npz", noise=a.noise), Windows(DATA / f"{prefix}val.npz")
    print(f"data: {len(train.tracks):,} train tracks, {len(val.tracks):,} val tracks ({time.time() - t0:.0f}s)")
    net = FM.build(torch, a.dropout)().to(dev)
    print(sum(p.numel() for p in net.parameters()), "parameters")
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    best = 0.0
    for step in range(1, a.steps + 1):
        loss, acc = loss_acc(net, train.batch(a.batch, rng), dev)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 1000 == 0 or step == a.steps:
            v = evaluate(net, val, dev, rng)
            vn = evaluate(net, val, dev, rng, noise=1.0)
            print(f"step {step}: train loss {loss.item():.3f} acc {acc.item():.3f} | val string accuracy "
                  f"{v:.4f} clean, {vn:.4f} noisy ({time.time() - t0:.0f}s)", flush=True)
            # Half-half: it has to read both clean tabs and transcriptions.
            if v + vn > best:
                best = v + vn
                torch.save({k: v_.half().cpu() for k, v_ in net.state_dict().items()}, out)
    print(f"best val string accuracy (clean + noisy) / 2 = {best / 2:.4f}; saved {out}")


if __name__ == "__main__":
    main()
