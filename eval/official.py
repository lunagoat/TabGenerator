#!/usr/bin/env python3
"""Score every part against Official Ultimate Guitar tabs of the same recordings.

Official tabs are transcribed from the studio recording and written out in
full (no repeat signs), with every instrument: guitars, bass, drums, vocals,
keys. Put the .gp/.gpx files and the audio (flac, dsf, ...) in one folder;
they are paired by title.

    python3 eval/official.py "/home/luna/Desktop/Test tabs and flacs" [--only TITLE] [--rewrite]

For each song: run TabGenerator (all parts; cached), align the tab to the
recording (dynamic time warping on the guitar and bass notes), then score
each part: note recall / precision / F1 (exact pitch, or the right kind of
drum; onsets within 50 and 100 ms) and, for guitar and bass, how often a
note found by both is on the tab's string and fret.
"""
import argparse
import bisect
import json
import re
import subprocess
from pathlib import Path

import env
import numpy as np

import realeval as R
import tabdata
from tabgen import pipeline
from tabgen.ensemble import DRUM_TYPE
from tabgen.transcribe import TNote

OUT = env.WORK / "official"
AUDIO = (".flac", ".dsf", ".wav", ".mp3", ".m4a", ".ogg", ".opus", ".aiff", ".ape", ".wv")
SKIP_FLAGS = set("txhc")
# Official tabs cover the whole recording, so a good alignment scores far
# higher than with the partial tabs of realeval.py.
MIN_QUALITY = 0.5  # tied, dead, harmonic, grace: not a struck note at that pitch


def words(s):
    return set(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()) - {"the", "a", "official", "chords", "tabs", "by", "ultimate", "guitar", "com", "and"}


def pairs(folder: Path):
    tabs = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".gp", ".gpx", ".gp5", ".gp4", ".gp3"))
    audio = [p for p in folder.iterdir() if p.suffix.lower() in AUDIO]
    out = []
    for t in tabs:
        m = re.match(r"OFFICIAL (.*) CHORDS & TABS by (.*?) @", t.stem)
        title, artist = (m.group(1), m.group(2)) if m else (t.stem, "")
        tw = words(title)
        best = max(audio, key=lambda a: (len(tw & words(a.stem)) / max(1, len(tw)), -len(words(a.stem))), default=None)
        if best and tw <= words(best.stem) | {"slight", "return", "song"} and len(tw & words(best.stem)) >= max(1, len(tw) - 1):
            out.append({"title": title.title(), "artist": artist, "tab": t, "audio": best})
        else:
            print(f"no audio for {t.name}")
    return out


def dump(tab_paths):
    """Guitar Pro -> JSON rows, cached next to the official tabs dump."""
    out_file = env.DATA / "official" / "ug.jsonl"
    rows = {}
    if out_file.is_file():
        rows = {json.loads(l)["path"]: json.loads(l) for l in open(out_file, encoding="utf-8")}
    missing = [str(p) for p in tab_paths if str(p) not in rows]
    if missing:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_file.with_suffix(".new.jsonl")
        subprocess.run([str(env.ROOT / "eval" / "tabdump.sh"), str(tmp), *missing], check=True,
                       env={**__import__("os").environ, "TMPDIR": str(env.WORK)})
        for l in open(tmp, encoding="utf-8"):
            r = json.loads(l)
            rows[r["path"]] = r
        tmp.unlink()
        with open(out_file, "w", encoding="utf-8") as f:
            for r in rows.values():
                f.write(json.dumps(r) + "\n")
    return rows


def kind(t):
    name = t["name"].lower()
    if t["name"].startswith("@$"):
        return None
    if t["percussion"]:
        return "drums" if "drum" in name else None
    if tabdata.is_guitar(t):
        return "guitar"
    if tabdata.is_bass(t):
        return "bass"
    if t["program"] == 85 and name.startswith("vocal"):
        return "vocals"
    if 0 <= t["program"] <= 7 or "piano" in name:
        return "piano"
    if 80 <= t["program"] <= 95 or "synth" in name or 16 <= t["program"] <= 23:
        return "synth"
    return None


def tab_parts(row):
    """{part: [(seconds, pitch or drum kind, string from lowest, fret)]} in the tab's own tempo."""
    bars = row["bars"]
    starts = [b[0] for b in bars]
    secs, t = [], 0.0
    for start, length, tempo, *_ in bars:
        secs.append(t)
        t += length * 60.0 / (max(tempo, 1) * tabdata.TICKS)

    def seconds(tick):
        i = max(0, bisect.bisect_right(starts, tick) - 1)
        return secs[i] + (tick - starts[i]) * 60.0 / (max(bars[i][2], 1) * tabdata.TICKS)

    out = {}
    for tr in row["tracks"]:
        k = kind(tr)
        if not k:
            continue
        strings = tr["strings"]
        order = sorted(range(len(strings)), key=lambda i: strings[i])
        notes = out.setdefault(k, [])
        for tick, dur, voice, s, fret, flags in tr["notes"]:
            if set(flags) & SKIP_FLAGS or not 1 <= s <= len(strings):
                continue
            if k == "drums":
                kind_ = DRUM_TYPE.get(fret)
                if kind_:
                    notes.append((seconds(tick), kind_, -1, -1))
            else:
                # offset = capo: frets count from it, so the sounding pitch
                # and the fret on a neck without a capo are both + offset.
                cap = tr.get("offset", 0) or 0
                notes.append((seconds(tick), strings[s - 1] + fret + cap, order.index(s - 1), fret + cap))
    for k in out:
        out[k] = sorted(set(out[k]))
    return out, secs


def run_pipeline(song):
    d = OUT / R.slug(song["title"])
    cached = d / "result.json"
    if cached.is_file():
        return json.loads(cached.read_text())
    collect = []
    pipeline.run(Path(song["audio"]), pipeline.Options(out_dir=d, collect=collect))
    res = {"notes": {}, "fretted": {}, "tuning": {}}
    for title, bm, payload, info in collect:
        if title == "notes":
            for part, ns in payload.items():
                res["notes"][part] = [[n.onset, DRUM_TYPE.get(n.pitch) if n.is_drum else n.pitch] for n in ns
                                      if not n.is_drum or n.pitch in DRUM_TYPE]
            continue
        part = "bass" if title == "Bass" else "guitar"
        res["tuning"].setdefault(part, info["tuning"])
        for ev in payload.events:
            t = float(bm.beat_to_time(ev.tick / bm.beat_ticks))
            for s, f in ev.notes:
                res["fretted"].setdefault(part, []).append([t, payload.strings[s] + f, s, f])
    cached.write_text(json.dumps(res))
    return res


def prf(truth, pred, tol):
    m = R.match(truth, pred, lambda x: x[1], lambda y: y[1], tol)
    p, r = len(m) / max(1, len(pred)), len(m) / max(1, len(truth))
    return p, r, 2 * p * r / max(1e-9, p + r), m


def score(song, row, res):
    parts, bar_secs = tab_parts(row)
    pitched = [n for k in ("guitar", "bass") for n in parts.get(k, [])]
    raw = [x for k in ("guitar", "bass") for x in res["notes"].get(k, [])]
    if len(pitched) < 30 or len(raw) < 30:
        return None
    to_audio, shift, quality = R.align(sorted(pitched), bar_secs, sorted(raw))
    out = {"song": song["title"], "quality": quality, "shift": shift, "parts": {}}
    for part, notes in parts.items():
        truth = [(float(to_audio(n[0])), n[1] if part == "drums" else n[1] + shift, n[2], n[3]) for n in notes]
        pred = res["notes"].get(part, [])
        if len(truth) < 20:
            continue
        r = {"tab_notes": len(truth), "found": len(pred)}
        for tol in (0.05, 0.1):
            p, rc, f, _ = prf(truth, pred, tol)
            r[f"f1@{int(tol * 1000)}"], r[f"recall@{int(tol * 1000)}"], r[f"precision@{int(tol * 1000)}"] = f, rc, p
        if part in res["fretted"]:
            gen = res["fretted"][part]
            _, _, _, m = prf(truth, gen, 0.1)
            r["tab_recall@100"] = len(m) / len(truth)
            r["same_string"] = sum(gen[j][2] == truth[i][2] for i, j in m) / max(1, len(m))
            r["same_fret"] = sum(gen[j][3] == truth[i][3] for i, j in m) / max(1, len(m))
            r["tuning"] = res["tuning"].get(part, "")
        out["parts"][part] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--only")
    ap.add_argument("--rewrite", action="store_true", help="write the tabs again from cached transcriptions")
    a = ap.parse_args()
    songs = pairs(Path(a.folder))
    rows = dump([s["tab"] for s in songs])
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for s in songs:
        if a.only and a.only.lower() not in s["title"].lower():
            continue
        if a.rewrite:
            (OUT / R.slug(s["title"]) / "result.json").unlink(missing_ok=True)
        res = run_pipeline(s)
        r = score(s, rows[str(s["tab"])], res)
        if not r:
            continue
        results.append(r)
        line = f"{s['title'][:26]:26s} align {r['quality']:.2f}"
        for part in ("guitar", "bass", "drums", "vocals", "piano", "synth"):
            p = r["parts"].get(part)
            if p:
                line += f" | {part} F1 {p['f1@100']:.2f}" + (f" str {p['same_string']:.2f}" if "same_string" in p else "")
        print(line, flush=True)
    (OUT / "scores.json").write_text(json.dumps(results, indent=1))
    good = [r for r in results if r["quality"] >= MIN_QUALITY]
    print(f"\n{len(good)} of {len(results)} songs aligned (alignment score >= {MIN_QUALITY}). Means over those:")
    for part in ("guitar", "bass", "drums", "vocals", "piano", "synth"):
        ps = [r["parts"][part] for r in good if part in r["parts"]]
        if not ps:
            continue
        m = lambda k: np.mean([p[k] for p in ps if k in p])
        line = (f"  {part:7s} ({len(ps)} songs): F1 {m('f1@50'):.3f} @50ms, {m('f1@100'):.3f} @100ms "
                f"(recall {m('recall@100'):.3f}, precision {m('precision@100'):.3f})")
        if any("same_string" in p for p in ps):
            line += f"; same string {m('same_string'):.3f}, same fret {m('same_fret'):.3f}"
        print(line)


if __name__ == "__main__":
    main()
