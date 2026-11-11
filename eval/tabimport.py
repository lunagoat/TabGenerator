#!/usr/bin/env python3
"""Turn downloaded tab collections into the same JSON rows tabdump.sh makes,
so tabdata.py / train_fingering.py can use them.

  alphaTex (.tex): the Hugging Face "vldsavelyev/guitar_tab" set, written by
      its author's converter (guitartab/alphatex.py): `fret.string` notes,
      `(...)` chords, `:N` sets the duration for the bar, `.N` changes it for
      one beat, note effects in `{...}` (d = dotted beat, tu N = tuplet,
      pm = palm mute, nh/ah/... = harmonic, h = hammer, be = bend ...).
      Only the first voice is written, and ties are not marked.
  JAMS (.jams): SynthTab's annotations: one `note_tab` annotation per
      string (first = highest), times in 960ths of a quarter note. No tuning
      is stored; standard tuning is assumed.

    python3 eval/tabimport.py tex  "<folder with .tex files>"  eval/data/tabs/hf.jsonl
    python3 eval/tabimport.py jams "<folder with .jams files>" eval/data/tabs/synthtab.jsonl

The source files are data, never run: only parsed as text / JSON.
"""
import json
import re
import sys
from pathlib import Path

TICKS = 960
WHOLE = 4 * TICKS


# ---------------------------------------------------------------- alphaTex

NOTE_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
HARMONICS = {"nh", "ah", "th", "ph", "sh"}


def _tuning(tok: str) -> int:
    m = re.fullmatch(r"([A-G]b?)(-?\d+)", tok)
    return int(m.group(2)) * 12 + NOTE_NAMES.index(m.group(1))


def _beats(text: str):
    """Split a track body into tokens: beats, `:N`, `|`, keeping (...) and {...} together."""
    out, cur, depth_p, depth_b = [], "", 0, 0
    for ch in text:
        if ch == "{":
            depth_b += 1
        elif ch == "}":
            depth_b -= 1
        elif ch == "(" and depth_b == 0:
            depth_p += 1
        elif ch == ")" and depth_b == 0:
            depth_p -= 1
        if ch.isspace() and depth_p == 0 and depth_b == 0:
            if cur:
                out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur:
        out.append(cur)
    return out


def _duration(value: int, dotted: bool, tuplet: int) -> int:
    d = WHOLE // max(1, value)
    if dotted:
        d = d * 3 // 2
    if tuplet > 1:
        times = 1 << (tuplet.bit_length() - 1)  # 3 in the time of 2, 5 in 4, 6 in 4 ...
        if times == tuplet:
            times //= 2
        d = d * times // tuplet
    return d


NOTE = re.compile(r"(-?\d+|x|-)\.(\d+)(?:\{([^}]*)\})?")


def parse_tex(path: Path, text: str):
    title = re.search(r'\\title "(.*?)"', text)
    tempo = re.search(r"\\tempo (\d+)", text)
    tempo = int(tempo.group(1)) if tempo else 120
    stem = path.stem
    artist, _, song = stem.partition("_-_")
    row = {"path": str(path), "title": (title.group(1) if title else song.replace("_", " ")),
           "artist": artist.replace("_", " "), "album": "", "tracks": []}
    parts = text.split("\\track")[1:]
    total = 0
    for part in parts:
        name = re.match(r'\s*"(.*?)"', part)
        inst = re.search(r"\\instrument (\d+)", part)
        tun = re.search(r"\\tuning ([^\n]+)", part)
        if not tun:
            continue
        try:
            strings = [_tuning(t) for t in tun.group(1).split()]
        except (AttributeError, ValueError):
            continue
        body = part[tun.end():]
        body = re.sub(r"\\frets \d+", "", body)
        notes, tick, cur = [], 0, 4
        for tok in _beats(body):
            if tok == "|" or tok == ".":
                continue
            if tok.startswith(":"):
                try:
                    cur = int(tok[1:])
                except ValueError:
                    pass
                continue
            if tok.startswith("r"):
                m = re.fullmatch(r"r(?:\.(\d+))?", tok)
                tick += WHOLE // max(1, int(m.group(1)) if m and m.group(1) else cur)
                continue
            beat_dur = cur
            m = re.search(r"\)\.(\d+)$|^[^(]*\.\d+(?:\{[^}]*\})?\.(\d+)$", tok)
            if m:
                beat_dur = int(m.group(1) or m.group(2))
            inner = tok[1:tok.rfind(")")] if tok.startswith("(") else tok
            found = NOTE.findall(inner)
            effects = set()
            for _, _, ef in found:
                effects |= set(ef.split())
            tup = 1
            tm = re.search(r"tu (\d+)", " ".join(ef for *_, ef in found))
            if tm:
                tup = int(tm.group(1))
            dur = _duration(beat_dur, "d" in effects, tup)
            for fret, string, ef in found:
                if fret in ("x", "-"):
                    continue
                e = set(ef.split())
                flags = ("m" if "pm" in e else "") + ("h" if e & HARMONICS else "") + ("o" if "h" in e else "") \
                    + ("b" if "be" in e else "") + ("v" if "v" in e else "") + ("r" if "lr" in e else "") + ("g" if "g" in e else "")
                notes.append([tick, dur, 0, int(string), int(fret), flags])
            tick += dur
        total = max(total, tick)
        if notes:
            row["tracks"].append({"name": name.group(1) if name else "", "program": int(inst.group(1)) if inst else -1,
                                  "percussion": False, "offset": 0, "strings": strings, "notes": notes})
    # The files carry no bar lines in time, only a tempo: one long "bar".
    row["bars"] = [[0, max(total, WHOLE), tempo, 4, 4, 0, 0, 0]]
    return row


# ---------------------------------------------------------------- JAMS

JAMS_FLAGS = [("palm_mute", "m"), ("harmonic", "h"), ("bend", "b"), ("dead_note", "x"), ("tie", "t"),
              ("hammer", "o"), ("slide", "s"), ("let_ring", "r"), ("ghost_note", "g"), ("vibrato", "v"), ("grace", "c")]
STANDARD = {6: [64, 59, 55, 50, 45, 40], 7: [64, 59, 55, 50, 45, 40, 35], 4: [43, 38, 33, 28]}


def parse_jams(path: Path, text: str):
    j = json.loads(text)
    tabs = [a for a in j["annotations"] if a["namespace"] == "note_tab"]
    tempo = next((int(a["data"][0]["value"]) for a in j["annotations"] if a["namespace"] == "tempo" and a["data"]), 120)
    if len(tabs) not in STANDARD:
        return None
    folder = path.parent.name  # "<Artist> - <Song> - gp4__1 - Distortion Guitar__midi"
    bits = folder.split(" - ")
    artist, song = (bits[0], bits[1]) if len(bits) >= 3 else ("", folder)
    program = j.get("sandbox", {}).get("instrument", -1)
    notes, end = [], 0
    for i, a in enumerate(tabs):
        for d in a["data"]:
            v = d["value"]
            t, dur = int(round(d["time"])), int(round(d["duration"]))
            flags = "".join(c for key, c in JAMS_FLAGS if v.get(key))
            notes.append([t, dur, 0, i + 1, int(v["fret"]), flags])
            end = max(end, t + dur)
    notes.sort()
    return {"path": str(path), "title": song, "artist": artist, "album": "",
            "bars": [[0, max(end, WHOLE), tempo, 4, 4, 0, 0, 0]],
            "tracks": [{"name": bits[-1].replace("__midi", "") if bits else "", "program": program,
                        "percussion": False, "offset": 0, "strings": STANDARD[len(tabs)], "notes": notes}]}


def main():
    kind, src, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    files = sorted(src.rglob("*.tex" if kind == "tex" else "*.jams"))
    parse = parse_tex if kind == "tex" else parse_jams
    ok = bad = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for p in files:
            try:
                row = parse(p, p.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                row = None
            if row and row["tracks"]:
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
                ok += 1
            else:
                bad += 1
            if (ok + bad) % 5000 == 0:
                print(f"  {ok + bad}/{len(files)}", flush=True)
    print(f"{ok} songs written to {out}, {bad} skipped")


if __name__ == "__main__":
    main()
