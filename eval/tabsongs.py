"""Which songs the Guitar Pro tabs are, and which of them have audio.

Artist and title come from the file's tags, or from its name ("AIC - Rooster")
when the tags name the transcriber instead. A song has audio when a FLAC in
the library has every title word in its file name and the artist (or a known
abbreviation) somewhere in its path.
"""
import csv
import os
import json
import re
import sys
import unicodedata
from pathlib import Path

import env

LIBRARY = Path(os.environ.get("TABGEN_LIBRARY", "~/Music")).expanduser()  # your FLAC library
ALIASES = {
    "aic": "Alice In Chains", "rhcp": "Red Hot Chili Peppers", "stp": "Stone Temple Pilots",
    "ratm": "Rage Against The Machine", "qotsa": "Queens Of The Stone Age", "fnm": "Faith No More",
    "acdc": "AC/DC", "foos": "Foo Fighters", "sp": "Smashing Pumpkins", "nirvanan": "Nirvana",
    "janes addiciton": "Jane's Addiction", "red hot chilli peppers": "Red Hot Chili Peppers",
    "rhcp ": "Red Hot Chili Peppers", "pj": "Pearl Jam", "totd": "Temple Of The Dog",
    "nais": "", "cc": "Chris Cornell", "dino jr": "Dinosaur Jr", "gnr": "Guns N' Roses", "gunsnroses": "Guns N' Roses",
    "led zeppeling": "Led Zeppelin", "led zepplin": "Led Zeppelin", "hendrix": "Jimi Hendrix", "ozzy": "Ozzy Osbourne",
    "stones": "The Rolling Stones", "rolling stones": "The Rolling Stones", "stpp": "Stone Temple Pilots",
    "mlb": "Mother Love Bone", "nhc": "NHC", "tfatd": "The Fierce And The Dead", "wvh": "Mammoth WVH",
    "lenny k": "Lenny Kravitz", "jerry": "Jerry Cantrell", "gigaton": "Pearl Jam", "elliot smith": "Elliott Smith",
    "cranberries": "The Cranberries", "white stripes": "The White Stripes", "silversun": "Silversun Pickups",
    "dead daises": "The Dead Daisies", "ritchie kotzen": "Richie Kotzen", "greta": "Greta Van Fleet",
    "fairport": "Fairport Convention", "them crooked": "Them Crooked Vultures", "umphrey": "Umphrey's McGee",
    "pearl jam (cover)": "Pearl Jam", "the la_s": "The La's", "bsc": "", "toad": "Toad the Wet Sprocket",
}
# Not one song: riff collections, lessons, ukulele arrangements.
SKIP = re.compile(r"\b(riffs?|songs|top 10|lesson|uke|easy guitar|nevermind|ten|30th celebration|volume)\b", re.I)
KNOWN_ARTISTS = {"pearl jam", "chris cornell", "audioslave", "the beatles", "stone gossard", "gigaton"}
NOT_ARTISTS = ("transcribed", "trabscribed", "notepad", "easy", "uke", "")
# Pearl Jam tabs named "<Song> - <guitarist>"
PJ_PLAYERS = {"mike", "stone", "eddie", "jeff", "matt", "bass", "solo", "the hidden solo"}
TAB_WORDS = re.compile(r"\b(solo|intro|riffs?|acoustic|unplugged|live|mtv|bass|easy|lesson|cover|"
                       r"4_4|version|ver|tab|tabs|full|part \d|v\d)\b", re.I)


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = s.replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def song_of(row) -> tuple[str, str]:
    stem = Path(row["path"]).stem
    artist, title = row["artist"].strip(), row["title"].strip()
    parts = re.split(r"\s+[-0]\s+", stem, maxsplit=1)
    if artist.lower().startswith(NOT_ARTISTS) or not artist or artist.lower() in ("gigaton",):
        second = parts[1].strip().lower() if len(parts) == 2 else ""
        if second and re.search(r"\b(mike|stone|eddie|jeff|matt)", second) or second in PJ_PLAYERS:
            artist, title = "Pearl Jam", parts[0]
        elif second in KNOWN_ARTISTS or second.startswith("pearl jam"):
            artist, title = parts[1], parts[0]
        elif len(parts) == 2:
            artist, title = parts
        else:
            artist, title = "", stem
    if title.lower().startswith(NOT_ARTISTS[:3]) or not title:
        title = parts[-1]
    if title.lower().startswith(artist.lower() + " - "):
        title = title[len(artist) + 3:]
    a = ALIASES.get(artist.strip().lower(), artist.strip())
    t = re.sub(r"\(.*?\)|\[.*?\]", "", title).replace("_", "'")
    t = TAB_WORDS.sub("", t)
    t = re.split(r"\s+-\s+", t)[0]
    t = re.sub(r"\b(mike|stone|eddie)'?s?\b.*$|\d$|\bnew\b|\d+'\d+", "", t, flags=re.I)
    return a.strip(), re.sub(r"\s+", " ", t).strip(" -'")


def library():
    return [p for p in LIBRARY.rglob("*.flac")]


def find_audio(artist, title, files):
    tw = norm(title).split()
    an = norm(artist)
    if not tw or not an:
        return None
    hits = []
    for f in files:
        name = norm(f.stem)
        if all(re.search(rf"\b{re.escape(w)}\b", name) for w in tw) and an in norm(str(f.parent) + " " + f.stem):
            extra = len(name.split()) - len(tw)
            live = bool(re.search(r"\b(live|demo|remix|acoustic|instrumental|karaoke)\b", name))
            hits.append((live and "live" not in tw, extra, f))
    return min(hits)[2] if hits else None


def main():
    rows = [json.loads(l) for l in open(env.DATA / "tabs" / "user.jsonl", encoding="utf-8")]
    files = library()
    songs: dict = {}
    for r in rows:
        if SKIP.search(Path(r["path"]).stem):
            continue
        a, t = song_of(r)
        if not a or not t or SKIP.search(t):
            continue
        songs.setdefault((norm(a), norm(t)), {"artist": a, "title": t, "tabs": []})["tabs"].append(r["path"])
    have = need = 0
    out = []
    for s in songs.values():
        audio = find_audio(s["artist"], s["title"], files)
        s["audio"] = str(audio) if audio else ""
        have += bool(audio)
        need += not audio
        out.append(s)
    (env.DATA / "tabs" / "songs.json").write_text(json.dumps(out, indent=1))
    with open(env.DATA / "tabs" / "missing_audio.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["artist", "title"])
        for s in sorted(out, key=lambda s: (s["artist"], s["title"])):
            if not s["audio"]:
                w.writerow([s["artist"], s["title"]])
    print(f"{len(rows)} tab files = {len(out)} songs; audio in the library for {have}, missing for {need}")
    print("examples with audio:", [(s["artist"], s["title"], Path(s["audio"]).name) for s in out if s["audio"]][:12])


if __name__ == "__main__":
    main()
