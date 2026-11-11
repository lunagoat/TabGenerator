#!/usr/bin/env python3
"""Train the context-aware string model (tabgen/fingering_model.py).

    python3 eval/fingering_data.py            # once: build the note sequences
    python3 eval/train_fingering_model.py [--steps 30000]

Trains on windows of 256 notes from every training track, reports the share
of validation notes put on the tab's string, and saves the weights (fp16) to
tabgen/data/fingering_model.pt.
"""
import argparse
import time

import env
import numpy as np
import torch

from tabgen import fingering_model as FM

DATA = env.WORK / "fingering"


class Windows:
    def __init__(self, path, window=256):
        z = np.load(path)
        self.ptr = z["ptr"]
        self.a = {k: z[k] for k in ("height", "size", "index", "dt", "dur", "target", "layout")}
        self.opens = z["opens"]
        self.dtb = FM.bucket(self.a["dt"].astype(np.float32), FM.DT_EDGES)
        self.durb = FM.bucket(self.a["dur"].astype(np.float32), FM.DUR_EDGES)
        lens = np.diff(self.ptr)
        self.tracks = np.nonzero(lens >= 16)[0]
        self.weights = np.minimum(lens[self.tracks], 4000).astype(float)
        self.weights /= self.weights.sum()
        self.window = window
        # Which strings each note could be on, from its track's tuning.
        self.mask = np.zeros((len(self.a["height"]), FM.MAX_STRINGS), dtype=bool)
        for t in range(len(lens)):
            lo, hi = self.ptr[t], self.ptr[t + 1]
            self.mask[lo:hi] = FM.string_mask(self.a["height"][lo:hi], self.opens[t])

    def batch(self, n, rng):
        W = self.window
        idx = np.zeros((n, W), dtype=np.int64)
        valid = np.zeros((n, W), dtype=bool)
        for b, t in enumerate(rng.choice(self.tracks, size=n, p=self.weights)):
            lo, hi = self.ptr[t], self.ptr[t + 1]
            start = lo if hi - lo <= W else rng.integers(lo, hi - W + 1)
            m = min(W, hi - start)
            idx[b, :m] = np.arange(start, start + m)
            valid[b, :m] = True
        return self.tensors(idx, valid)

    def tensors(self, idx, valid):
        a = self.a
        T = lambda x: torch.as_tensor(x, dtype=torch.long)
        return (T(np.clip(a["height"][idx], 0, FM.MAX_HEIGHT - 1)), T(a["size"][idx]), T(a["index"][idx]),
                T(a["layout"][idx]), T(self.dtb[idx]), T(self.durb[idx]), T(a["target"][idx]),
                torch.as_tensor(self.mask[idx] & valid[..., None]), torch.as_tensor(valid))


def loss_acc(net, batch, dev):
    h, s, i, l, dt, du, y, mask, valid = [x.to(dev) for x in batch]
    logits = net(h, s, i, l, dt, du).masked_fill(~mask, -1e4)
    ok = valid & mask.gather(-1, y.clamp(0, FM.MAX_STRINGS - 1)[..., None])[..., 0]
    loss = torch.nn.functional.cross_entropy(logits[ok], y[ok])
    acc = (logits.argmax(-1) == y)[ok].float().mean()
    return loss, acc


def evaluate(net, data, dev, rng, n=60):
    net.eval()
    accs = []
    with torch.no_grad():
        for _ in range(n):
            accs.append(loss_acc(net, data.batch(64, rng), dev)[1].item())
    net.train()
    return float(np.mean(accs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=2e-3)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    t0 = time.time()
    train, val = Windows(DATA / "train.npz"), Windows(DATA / "val.npz")
    print(f"data: {len(train.tracks):,} train tracks, {len(val.tracks):,} val tracks ({time.time() - t0:.0f}s)")
    net = FM.build(torch)().to(dev)
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
            print(f"step {step}: train loss {loss.item():.3f} acc {acc.item():.3f} | val string accuracy {v:.4f} ({time.time() - t0:.0f}s)", flush=True)
            if v > best:
                best = v
                FM.WEIGHTS.parent.mkdir(exist_ok=True)
                torch.save({k: v_.half().cpu() for k, v_ in net.state_dict().items()}, str(FM.WEIGHTS))
    print(f"best val string accuracy {best:.4f}; saved {FM.WEIGHTS}")


if __name__ == "__main__":
    main()
