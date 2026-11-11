#!/usr/bin/env python3
"""Score the pipeline against human-made tabs of real recordings.

For every song with both a tab and audio (realsongs.py):
1. run TabGenerator on the audio (guitar only; cached),
2. align the tab to the recording: dynamic time warping between the tab's
   notes and the transcribed notes (pitch classes), with the tab allowed to
   cover only part of the recording and to be in another tuning,
3. score: of the tab's notes, how many the transcription found (recall),
   how many transcribed notes the tab has (precision; low when the
   recording has guitars the tab leaves out), and for the notes both have,
   whether the generated tab puts them on the same string.

    python3 eval/realeval.py [--only SUBSTRING] [--no-run]
"""
import argparse
import json
import sys
from pathlib import Path

import env
import numpy as np

import realsongs
from tabgen import ensemble, pipeline
from tabgen.transcribe import TNote

FRAME = 0.05
OUT = env.WORK / "real"
TOL = 0.075  # tab notes are placed by the alignment, so a little more slack than 50 ms
MIN_QUALITY = 0.35


def slug(s):
    return "".join(c if c.isalnum() else "_" for c in s)[:80]


# ---------------------------------------------------------------- pipeline

def run_pipeline(song):
    """Generated guitar notes (seconds, pitch, string, fret) and the raw
    voted transcription, cached in OUT/<slug>/result.json."""
    d = OUT / slug(f"{song['artist']} - {song['title']}")
    cached = d / "result.json"
    if cached.is_file():
        return json.loads(cached.read_text())
    collect = []
    opts = pipeline.Options(parts=("guitar",), out_dir=d, collect=collect)
    pipeline.run(Path(song["audio"]), opts)
    out = []
    for title, bm, gp, info in collect:
        for ev in gp.events:
            t = float(bm.beat_to_time(ev.tick / bm.beat_ticks))
            for s, f in ev.notes:
                out.append([t, gp.strings[s] + f, s, f, title])
    cache = env.ROOT / ".cache" / Path(song["audio"]).stem
    raw = []
    for f in sorted(cache.glob("notes_v2_*-guitar_any*_fp16.json")):
        if "@" not in f.name:
            raw.append([TNote(**n) for n in json.loads(f.read_text())])
    raw = [[n for n in p if n.instrument in pipeline.PART_LABELS["guitar"]] for p in raw]
    voted = ensemble.merge(raw, 2) if len(raw) > 1 else (raw[0] if raw else [])
    res = {"tab": out, "raw": [[n.onset, n.pitch] for n in voted], "tuning": collect[0][3]["tuning"] if collect else ""}
    cached.write_text(json.dumps(res))
    return res


# ---------------------------------------------------------------- alignment

def chroma_roll(times, pitches, length, shift=0):
    n = int(length / FRAME) + 2
    roll = np.zeros((n, 13))
    for t, p in zip(times, pitches):
        i = int(t / FRAME)
        if 0 <= i < n:
            roll[i, (p + shift) % 12] += 1.0
            if i + 1 < n:
                roll[i + 1, (p + shift) % 12] += 0.5
    roll[:, 12] = (roll[:, :12].sum(1) == 0) * 1.0  # silence
    return roll / (np.linalg.norm(roll, axis=1, keepdims=True) + 1e-9)


def subsequence_dtw(cost):
    """Path through cost [tab frames, audio frames] from the first to the
    last tab frame, starting and ending anywhere in the audio."""
    n, m = cost.shape
    D = np.empty((n, m))
    back = np.zeros((n, m), dtype=np.int8)  # 0 diag, 1 up (tab advances), 2 left (audio advances)
    D[0] = cost[0]
    for i in range(1, n):
        diag = np.r_[np.inf, D[i - 1, :-1]]
        up = D[i - 1]
        a = np.minimum(diag, up)
        choice = np.where(diag <= up, 0, 1)
        # D[i, j] = c[i, j] + min(a[j], D[i, j-1]), solved with a running minimum
        C = np.cumsum(cost[i])
        Cprev = np.r_[0.0, C[:-1]]
        val = a - Cprev
        run = np.minimum.accumulate(val)
        D[i] = C + run
        from_left = val > run  # the minimum came from further left
        back[i] = np.where(from_left, 2, choice)
    j = int(D[-1].argmin())
    i = n - 1
    path = [(i, j)]
    while i > 0:
        b = back[i, j]
        if b == 0:
            i, j = i - 1, j - 1
        elif b == 1:
            i -= 1
        else:
            j -= 1
        path.append((i, j))
    return path[::-1], float(D[-1, j] / n)


def align(tab_notes, bar_times, raw):
    """(function tab seconds -> audio seconds, transposition, quality)."""
    tab_t = np.array([n[0] for n in tab_notes])
    tab_p = np.array([n[1] for n in tab_notes])
    raw_t = np.array([r[0] for r in raw])
    raw_p = np.array([r[1] for r in raw])
    A = chroma_roll(raw_t, raw_p, raw_t.max() + 1)
    best = None
    for shift in (0, -1, 1, -2, 2):
        T = chroma_roll(tab_t, tab_p, tab_t.max() + 1, shift)
        path, score = subsequence_dtw(1 - T @ A.T)
        if best is None or score < best[1]:
            best = (path, score, shift)
    path, _, shift = best
    ti = np.array([p[0] for p in path]) * FRAME
    aj = np.array([p[1] for p in path]) * FRAME
    # Smooth: one straight piece per bar, anchored where the path crosses the bar lines.
    anchors_t = np.array(list(bar_times) + [tab_t.max() + FRAME])
    anchors_a = np.array([np.median(aj[np.abs(ti - b) <= FRAME]) if np.any(np.abs(ti - b) <= FRAME)
                          else np.interp(b, ti, aj) for b in anchors_t])
    anchors_a = np.maximum.accumulate(anchors_a)

    def to_audio(t):
        return np.interp(t, anchors_t, anchors_a)

    mapped = to_audio(tab_t)
    # Quality: share of tab notes with a transcribed note of the same pitch class nearby.
    order = np.argsort(raw_t)
    rt, rp = raw_t[order], raw_p[order]
    hit = 0
    for t, p in zip(mapped, tab_p + shift):
        lo, hi = np.searchsorted(rt, t - 0.1), np.searchsorted(rt, t + 0.1)
        hit += np.any(rp[lo:hi] % 12 == p % 12)
    return to_audio, shift, hit / max(1, len(tab_t))


# ---------------------------------------------------------------- scoring

def match(a, b, key_a, key_b, tol=TOL):
    """One-to-one matches between note lists (times + keys), greedy by time."""
    pairs = []
    used = set()
    bt = np.array([x[0] for x in b])
    order = np.argsort(bt)
    bt_sorted = bt[order]
    for i, x in sorted(enumerate(a), key=lambda e: e[1][0]):
        lo, hi = np.searchsorted(bt_sorted, x[0] - tol), np.searchsorted(bt_sorted, x[0] + tol)
        best = None
        for k in range(lo, hi):
            j = order[k]
            if j in used or key_b(b[j]) != key_a(x):
                continue
            d = abs(b[j][0] - x[0])
            if best is None or d < best[0]:
                best = (d, j)
        if best:
            used.add(best[1])
            pairs.append((i, best[1]))
    return pairs


def score(song, res):
    tab, bars = realsongs.tab_notes(song["tabs"])
    # Notes whose pitch the tab doesn't really give, or which are not struck.
    tab = [n for n in tab if not set(n[4]) & set("txhc")]
    if len(tab) < 30 or not res["raw"]:
        return None
    to_audio, shift, quality = align(tab, bars, res["raw"])
    truth = [(float(to_audio(n[0])), n[1] + shift, n[2], n[3]) for n in tab]
    lo, hi = min(t[0] for t in truth) - 1, max(t[0] for t in truth) + 1
    raw = [r for r in res["raw"] if lo <= r[0] <= hi]
    gen = [g for g in res["tab"] if lo <= g[0] <= hi]
    out = {"quality": quality, "shift": shift, "tab_notes": len(truth)}
    for name, pred in (("transcription", raw), ("tab", gen)):
        exact = match(truth, pred, lambda x: x[1], lambda y: y[1])
        pc = match(truth, pred, lambda x: x[1] % 12, lambda y: y[1] % 12)
        out[name] = {"recall": len(exact) / len(truth), "precision": len(exact) / max(1, len(pred)),
                     "recall_any_octave": len(pc) / len(truth)}
        if name == "tab":
            same = [(i, j) for i, j in exact if pred[j][2] == truth[i][2] and len(set(gp[2] for gp in [pred[j]])) == 1]
            strings_ok = sum(1 for i, j in exact if pred[j][2] == truth[i][2])
            out["string_accuracy"] = strings_ok / max(1, len(exact))
            out["fret_match"] = sum(1 for i, j in exact if pred[j][3] == truth[i][3]) / max(1, len(exact))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--no-run", action="store_true", help="score only songs already run")
    ap.add_argument("--rewrite", action="store_true",
                    help="songs already run: write their tabs again from the cached transcriptions (after changing the tab-writing stage)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for song in realsongs.songs():
        name = f"{song['artist']} - {song['title']}"
        if a.only and a.only.lower() not in name.lower():
            continue
        if a.no_run and not (OUT / slug(name) / "result.json").is_file():
            continue
        if a.rewrite:
            done = OUT / slug(name) / "result.json"
            if not done.is_file():
                continue
            done.unlink()
        try:
            res = run_pipeline(song)
            r = score(song, res)
        except Exception as e:  # one bad file shouldn't stop the run
            print(f"{name}: FAILED {e!r}", file=sys.stderr)
            continue
        if r is None:
            continue
        r["song"] = name
        rows.append(r)
        flag = "" if r["quality"] >= MIN_QUALITY else "   (alignment failed: left out)"
        print(f"{name[:45]:45s} align {r['quality']:.2f} shift {r['shift']:+d} | transcription recall {r['transcription']['recall']:.2f} "
              f"prec {r['transcription']['precision']:.2f} | tab recall {r['tab']['recall']:.2f} strings {r['string_accuracy']:.2f}{flag}",
              flush=True)
    good = [r for r in rows if r["quality"] >= MIN_QUALITY]
    (OUT / "scores.json").write_text(json.dumps(rows, indent=1))
    if good:
        m = lambda f: float(np.mean([f(r) for r in good]))
        print(f"\n{len(good)} of {len(rows)} songs aligned. Means over those:")
        print(f"  transcription: recall {m(lambda r: r['transcription']['recall']):.3f} (any octave {m(lambda r: r['transcription']['recall_any_octave']):.3f}), "
              f"precision {m(lambda r: r['transcription']['precision']):.3f}")
        print(f"  generated tab: recall {m(lambda r: r['tab']['recall']):.3f} (any octave {m(lambda r: r['tab']['recall_any_octave']):.3f}), "
              f"precision {m(lambda r: r['tab']['precision']):.3f}")
        print(f"  same string as the human tab: {m(lambda r: r['string_accuracy']):.3f}, same fret: {m(lambda r: r['fret_match']):.3f}")


if __name__ == "__main__":
    main()
