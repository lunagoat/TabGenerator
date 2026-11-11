"""Real songs with a human-made tab: the audio, the tab, and the tab's notes
in playing order (repeats unrolled), in seconds of the tab's own tempo."""
import json
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
    endings chosen per pass (Guitar Pro's repeat rules). A repeat that has
    been played out is never taken again (Guitar Pro has no nested repeats;
    a section marked to repeat inside another one would otherwise loop)."""
    order, i, start = [], 0, 0
    went_back: dict[int, int] = {}  # close bar -> times it sent us back
    finished: set[int] = set()
    while i < len(bars) and len(order) < 8 * len(bars):
        close, open_, alt = (list(bars[i]) + [0])[5:8]
        if open_:
            start = i
        # Which pass of the current repeat this is (for alternate endings).
        n = max((went_back.get(j, 0) for j in range(start, len(bars)) if j not in finished), default=0)
        if alt and not (alt >> n) & 1:
            i += 1
            continue
        order.append(i)
        if close and i not in finished:
            if went_back.get(i, 0) < close:
                went_back[i] = went_back.get(i, 0) + 1
                i = start
                continue
            finished.add(i)
        i += 1
    return order


_ROWS = None


def best_tab(paths):
    """The dumped tab to use for a song: one file is enough; the one with
    the most guitar notes. None if none of `paths` was dumped."""
    global _ROWS
    if _ROWS is None:
        _ROWS = {}
        for l in open(env.DATA / "tabs" / "user.jsonl", encoding="utf-8"):
            r = json.loads(l)
            _ROWS[r["path"]] = r
    cands = [_ROWS[p] for p in paths if p in _ROWS]
    if not cands:
        return None
    return max(cands, key=lambda r: sum(len(t["notes"]) for t in r["tracks"] if tabdata.is_guitar(t)))


def tab_notes(paths) -> tuple[list, list]:
    """(notes, bar times) of a song's guitar tracks, played in order.

    notes: (seconds, pitch, string from the lowest = 0, fret, flags, track).
    bar times: seconds at which each played bar starts (tab tempo).
    """
    song = best_tab(paths)
    if song is None:
        return [], []
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
