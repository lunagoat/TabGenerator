#!/usr/bin/env python3
"""Build training data for the fingering model from all tab collections.

Every guitar track becomes a note sequence (notes starting together are a
chord, sorted low to high) with the string each note was played on. Songs
used for testing (your held-out tabs, the Official tabs) are kept out.

    python3 eval/fingering_data.py [--max-notes 40000000]
Writes eval/work/fingering/{train,val}.npz
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import env
import numpy as np

import tabdata
from tabgen import fingering_model as FM

OUT = env.WORK / "fingering"
SOURCES = ["user.jsonl", "hf.jsonl", "synthtab.jsonl"]
SKIP = set("txhc")


def norm_title(s: str) -> str:
    s = re.sub(r"\(.*?\)|\[.*?\]", "", s.lower())
    return re.sub(r"[^a-z0-9]", "", s)


def test_titles() -> set:
    """Titles of every song used for testing, so no copy of them is trained on."""
    out = set()
    user = [json.loads(l) for l in open(env.DATA / "tabs" / "user.jsonl", encoding="utf-8")]
    test_groups = {t.song for t in tabdata.split(tabdata.load())[1]}
    for r in user:
        if tabdata._group(r["path"]) in test_groups:
            out.add(norm_title(r["title"] or Path(r["path"]).stem))
            out.add(norm_title(Path(r["path"]).stem.split(" - ")[-1]))
    off = env.DATA / "official" / "ug.jsonl"
    if off.is_file():
        for l in open(off, encoding="utf-8"):
            r = json.loads(l)
            m = re.match(r"OFFICIAL (.*) CHORDS", Path(r["path"]).stem)
            out.add(norm_title(m.group(1) if m else r["title"]))
    out.discard("")
    return out


def track_arrays(track, tempo):
    """Feature arrays for one guitar track, or None."""
    strings = track["strings"]
    cap = track.get("offset") or 0
    n = len(strings)
    if n not in (6, 7) or min(strings) < 33 or len(set(strings)) < n - 1:
        return None
    p = track["program"]
    if not (24 <= p <= 31 or p < 0) or track.get("percussion") or track.get("name", "").startswith("@$"):
        return None
    opens = sorted(s + cap for s in strings)  # lowest first
    order = sorted(range(n), key=lambda i: strings[i])
    idx_of = {i + 1: order.index(i) for i in range(n)}  # TG string number (1 = highest) -> 0 = lowest
    by_tick = {}
    for tick, dur, voice, s, fret, flags in track["notes"]:
        if set(flags) & SKIP or not 1 <= s <= n or not 0 <= fret <= 30:
            continue
        pitch = strings[s - 1] + cap + fret
        by_tick.setdefault(tick, {}).setdefault(pitch, (idx_of[s], dur))
    if len(by_tick) < 16:
        return None
    sec = 60.0 / (max(tempo, 30) * tabdata.TICKS)
    rows = []
    prev = None
    for tick in sorted(by_tick):
        chord = sorted(by_tick[tick].items())
        dt = 4.0 if prev is None else (tick - prev) * sec
        prev = tick
        for k, (pitch, (string, dur)) in enumerate(chord):
            rows.append((pitch - opens[0], len(chord), k, dt if k == 0 else 0.0, dur * sec, string))
    a = np.array(rows, dtype=np.float32)
    return {"height": a[:, 0].astype(np.int16), "size": a[:, 1].astype(np.int8), "index": a[:, 2].astype(np.int8),
            "dt": a[:, 3].astype(np.float16), "dur": a[:, 4].astype(np.float16), "target": a[:, 5].astype(np.int8),
            "layout": np.full(len(a), FM.layout_id(opens), dtype=np.int8),
            "opens": np.array(opens + [0] * (7 - n), dtype=np.int16)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-notes", type=int, default=40_000_000)
    a = ap.parse_args()
    banned = test_titles()
    print(len(banned), "test titles kept out")
    shards = {"train": [], "val": []}
    counts = {"train": 0, "val": 0}
    skipped = 0
    for src in SOURCES:
        path = env.DATA / "tabs" / src
        if not path.is_file():
            continue
        n_src = 0
        for line in open(path, encoding="utf-8"):
            row = json.loads(line)
            title = norm_title(row.get("title") or Path(row["path"]).stem)
            if src == "user.jsonl":
                # Your own collection: its test split is defined by tabdata.split.
                title_keys = {title, norm_title(Path(row["path"]).stem.split(" - ")[-1])}
            else:
                title_keys = {title}
            if title_keys & banned:
                skipped += 1
                continue
            key = (row.get("artist", "") + "|" + title)
            split = "val" if int(hashlib.sha1(key.encode()).hexdigest(), 16) % 100 < 2 else "train"
            tempo = row["bars"][0][2] if row["bars"] else 120
            for tr in row["tracks"]:
                arr = track_arrays(tr, tempo)
                if arr is None:
                    continue
                shards[split].append(arr)
                counts[split] += len(arr["height"])
                n_src += len(arr["height"])
            if counts["train"] >= a.max_notes:
                break
        print(f"{src}: {n_src:,} notes", flush=True)
        if counts["train"] >= a.max_notes:
            break
    OUT.mkdir(parents=True, exist_ok=True)
    for split, tracks in shards.items():
        ptr = np.cumsum([0] + [len(t["height"]) for t in tracks])
        out = {k: np.concatenate([t[k] for t in tracks]) for k in ("height", "size", "index", "dt", "dur", "target", "layout")}
        out["opens"] = np.stack([t["opens"] for t in tracks])
        np.savez(OUT / f"{split}.npz", ptr=ptr, **out)
        print(f"{split}: {len(tracks):,} tracks, {ptr[-1]:,} notes")
    print(f"{skipped} songs left out as test songs")


if __name__ == "__main__":
    main()
