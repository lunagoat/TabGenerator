"""Piano and synth parts.

Two ways to write a keyboard part down:

- as keyboard music: a note roll in the text file, and in the Guitar Pro
  file a track meant to be read as standard notation. Guitar Pro 5 files
  only know stringed tracks, so the notes sit on a made-up 7-string
  "instrument" tuned in fifths that covers five and a half octaves;
- as guitar tablature (`--keys-as guitar`): the notes are moved by octaves
  into the guitar's range and then fingered like any guitar part, so chords
  come out as shapes a hand can hold.
"""

from dataclasses import replace

# Open strings of the made-up keyboard "instrument", lowest first: E1 up in
# fifths. With 24 frets it reaches from E1 to Bb6, and every pitch in the
# middle can go on four different strings, so close chords fit.
KEYBOARD_STRINGS = (28, 35, 42, 49, 56, 63, 70)
KEYBOARD_FRETS = 24

GM_PROGRAMS = {
    "acoustic_piano": 0,
    "electric_piano": 4,
    "chromatic_percussion": 11,
    "organ": 16,
    "string_ensemble": 48,
    "synth_strings": 50,
    "synth_lead": 80,
    "synth_pad": 88,
    "voice": 53,
}
FRIENDLY = {
    "acoustic_piano": "piano", "electric_piano": "electric piano", "chromatic_percussion": "bells/mallets",
    "organ": "organ", "synth_lead": "synth lead", "synth_pad": "synth pad", "string_ensemble": "strings",
    "synth_strings": "synth strings", "clean_electric_guitar": "clean guitar", "acoustic_guitar": "acoustic guitar",
    "distorted_electric_guitar": "distorted guitar",
}


def fold(pitch: int, lo: int, hi: int) -> int:
    """Move a pitch by octaves into [lo, hi]."""
    while pitch < lo:
        pitch += 12
    while pitch > hi:
        pitch -= 12
    return pitch


def fold_notes(notes, lo: int, hi: int):
    """(notes with every pitch inside [lo, hi], how many were moved)."""
    out, moved = [], 0
    for n in notes:
        p = fold(n.pitch, lo, hi)
        moved += p != n.pitch
        out.append(replace(n, pitch=p) if p != n.pitch else n)
    return out, moved


def heard_as(notes, top: int = 2) -> str:
    """What the model called these notes, most common first: "piano" or
    "clean guitar, bells/mallets"."""
    counts: dict[str, int] = {}
    for n in notes:
        counts[n.instrument] = counts.get(n.instrument, 0) + 1
    names = sorted(counts, key=lambda k: -counts[k])
    names = [k for k in names if counts[k] >= 0.15 * len(notes)][:top] or names[:1]
    return ", ".join(FRIENDLY.get(k, k.replace("_", " ")) for k in names)


def program_for(notes, default: int) -> int:
    counts: dict[str, int] = {}
    for n in notes:
        counts[n.instrument] = counts.get(n.instrument, 0) + 1
    for k in sorted(counts, key=lambda k: -counts[k]):
        if k in GM_PROGRAMS:
            return GM_PROGRAMS[k]
    return default


def place_chord(pitches: list[int]) -> list[tuple[int, int]]:
    """(string, fret) for each pitch on the keyboard "instrument", one note
    per string, frets as low as they go. With more notes than fit, inner
    notes are left out first (the top and the bass matter most)."""
    lo, hi = KEYBOARD_STRINGS[0], KEYBOARD_STRINGS[-1] + KEYBOARD_FRETS
    pitches = sorted({fold(p, lo, hi) for p in pitches})
    while pitches:
        got = _place(pitches)
        if got is not None:
            return got
        # Drop the inner note closest to its neighbours.
        if len(pitches) <= 2:
            pitches = pitches[1:]
            continue
        gaps = [pitches[i + 1] - pitches[i - 1] for i in range(1, len(pitches) - 1)]
        del pitches[1 + gaps.index(min(gaps))]
    return []


def _place(pitches: list[int]):
    """Strings ascend with pitch; search for the assignment with the lowest frets."""
    n_strings = len(KEYBOARD_STRINGS)
    best = None

    def rec(i, min_string, acc, cost):
        nonlocal best
        if best is not None and cost >= best[0]:
            return
        if i == len(pitches):
            best = (cost, list(acc))
            return
        if n_strings - min_string < len(pitches) - i:
            return
        for s in range(min_string, n_strings):
            fret = pitches[i] - KEYBOARD_STRINGS[s]
            if fret < 0:
                break
            if fret > KEYBOARD_FRETS:
                continue
            acc.append((s, fret))
            rec(i + 1, s + 1, acc, cost + fret)
            acc.pop()

    rec(0, 0, [], 0)
    return best[1] if best else None
