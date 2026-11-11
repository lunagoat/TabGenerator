"""Real songs with a human-made tab: the audio, the tab, and the tab's notes
in playing order (repeats unrolled), in seconds of the tab's own tempo."""
import json
import re
from pathlib import Path

import env

import tabdata
import tabsongs

DOWNLOADS = Path.home() / "Music" / "vibeseek" / "missing_audio"
PROGRESS = Path.home() / ".local" / "share" / "vibeseek" / "csv"


def downloaded() -> dict:
    """(norm artist, norm title) -> local file, from vibeseek's progress file."""
    out = {}
    for f in PROGRESS.glob("missing_audio-*.json"):
        d = json.loads(f.read_text())
        root = Path(d["output"])
        for key, row in d["rows"].items():
            if row.get("status") != "Done" or not row.get("file"):
                continue
            local = root / row["file"].replace("\\", "/").split("/")[-1]
            if local.is_file():
                a, t = key.split("|")[:2]
                out[(tabsongs.norm(a), tabsongs.norm(t))] = local
    return out


def songs() -> list[dict]:
    """Songs with audio: {"artist", "title", "audio", "tabs": [paths]}."""
    listed = json.loads((env.DATA / "tabs" / "songs.json").read_text())
    got = downloaded()
    out = []
    for s in listed:
        audio = s["audio"] or got.get((tabsongs.norm(s["artist"]), tabsongs.norm(s["title"])))
        if audio:
            out.append({**s, "audio": str(audio)})
    return out


def play_order(bars) -> list[int]:
    """Bar indexes in the order they are played: repeats taken, alternate
    endings chosen per pass (Guitar Pro's repeat rules)."""
    order, i, start, passes = [], 0, 0, {}
    guard = 0
    while i < len(bars) and guard < 20 * len(bars):
        guard += 1
        _, _, _, _, _, close, open_, alt = (list(bars[i]) + [0])[:8]
        if open_ and passes.get(("open", i)) is None:
            start = i
            passes[("open", i)] = True
        n = passes.get(start, 0)  # how many times the repeat has gone back
        if alt and not (alt >> n) & 1:
            i += 1
            continue
        order.append(i)
        if close and n < close:
            passes[start] = n + 1
            i = start
            continue
        if close:
            passes[start] = 0
            passes.pop(("open", start), None)
        i += 1
    return order


def tab_notes(paths) -> tuple[list, list]:
    """(notes, bar times) of a song's guitar tracks, played in order.

    notes: (seconds, pitch, string from the lowest = 0, fret, flags, track).
    bar times: seconds at which each played bar starts (tab tempo).
    """
    rows = [json.loads(l) for p in [env.DATA / "tabs" / "user.jsonl"] for l in open(p, encoding="utf-8")]
    by_path = {r["path"]: r for r in rows}
    # One tab file per song is enough; prefer the one with the most guitar notes.
    cands = [by_path[p] for p in paths if p in by_path]
    if not cands:
        return [], []
    song = max(cands, key=lambda r: sum(len(t["notes"]) for t in r["tracks"] if tabdata.is_guitar(t)))
    bars = song["bars"]
    order = play_order(bars)
    bar_times, t = [], 0.0
    for b in order:
        bar_times.append(t)
        start, length, tempo = bars[b][:3]
        t += length * 60.0 / (max(tempo, 1) * tabdata.TICKS)
    notes = []
    for ti, tr in enumerate(song["tracks"]):
        if not tabdata.is_guitar(tr):
            continue
        strings = tr["strings"]
        n = len(strings)
        order_s = sorted(range(n), key=lambda i: strings[i])
        by_bar: dict = {}
        for note in tr["notes"]:
            tick = note[0]
            b = max(k for k, bar in enumerate(bars) if bar[0] <= tick) if tick >= bars[0][0] else 0
            by_bar.setdefault(b, []).append(note)
        for pos, b in enumerate(order):
            start, length, tempo = bars[b][:3]
            rate = 60.0 / (max(tempo, 1) * tabdata.TICKS)
            for tick, dur, voice, s, fret, flags in by_bar.get(b, []):
                if not 1 <= s <= n:
                    continue
                notes.append((bar_times[pos] + (tick - start) * rate, strings[s - 1] + fret,
                              order_s.index(s - 1), fret, flags, ti))
    notes.sort()
    return notes, bar_times


if __name__ == "__main__":
    ss = songs()
    print(len(ss), "songs with audio")
    for s in ss[:5]:
        notes, bars = tab_notes(s["tabs"])
        print(s["artist"], "-", s["title"], "|", Path(s["audio"]).name, "|", len(notes), "tab notes over %.0f s" % (bars[-1] if bars else 0))
