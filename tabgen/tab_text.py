"""Plain-text (ASCII) tab rendering for fretted instruments and drums."""

import math

from .rhythm import BeatMap

MAX_WIDTH = 110


def _beat_slots(bm: BeatMap, beat: int) -> int:
    if 0 <= beat < len(bm.subdivisions):
        return int(bm.subdivisions[beat])
    return 1


def _render_bar(bm: BeatMap, bar, n_lines: int, events: dict[int, dict[int, str]]):
    """Returns (count_line, [line strings]) for one bar, without bar lines."""
    bt = bm.beat_ticks
    count = []
    lines = [[] for _ in range(n_lines)]
    for bi in range(bar.n_beats):
        beat = bar.first_beat + bi
        s = _beat_slots(bm, beat)
        slot_w = max(2, math.ceil(8 / s))
        beat_text = str(bi + 1) + ({3: "(3)", 6: "(6)", 12: "(12)"}.get(s, "") if not bm.compound else "")
        beat_start = len(count)
        for k in range(s):
            tick = beat * bt + k * bt // s
            col = events.get(tick, {})
            w = max(slot_w, max((len(t) for t in col.values()), default=1) + 1)
            for li in range(n_lines):
                txt = col.get(li, "")
                lines[li].append(txt + "-" * (w - len(txt)))
            count.append(" " * w)
        # Beat number over the first slot of the beat.
        flat = "".join(count[beat_start:])
        flat = beat_text + flat[len(beat_text):]
        count[beat_start:] = [flat]
    return "".join(count), ["".join(l) for l in lines]


def _bar_has_events(bm: BeatMap, bar, events) -> bool:
    bt = bm.beat_ticks
    lo = bar.first_beat * bt
    hi = (bar.first_beat + bar.n_beats) * bt
    return any(lo <= t < hi for t in events)


def render(
    header: list[str],
    labels: list[str],
    bm: BeatMap,
    events: dict[int, dict[int, str]],
    footer: list[str] | None = None,
    sparse: bool = False,
) -> str:
    """Render a whole part.

    `labels` name the lines top to bottom; `events` maps tick -> {line: text}.
    `sparse`: each system shows only the lines used in it (for a note roll
    with one line per pitch, where most lines are empty most of the time).
    """
    lab_w = max(len(l) for l in labels)
    labels = [l.rjust(lab_w) for l in labels]
    n = len(labels)

    # Build blocks: either a rendered bar or a run of empty bars.
    first_bar = next((i for i, b in enumerate(bm.bars) if _bar_has_events(bm, b, events)), None)
    if first_bar is None:
        return "\n".join(header + ["", "(no notes were detected for this part)", ""])
    last_bar = max(i for i, b in enumerate(bm.bars) if _bar_has_events(bm, b, events))
    blocks = []  # (bar_number, count_line, lines)
    i = 0
    bars = bm.bars
    while i < len(bars):
        if i > last_bar:
            break
        bar = bars[i]
        if not _bar_has_events(bm, bar, events):
            j = i
            while j < len(bars) and j <= last_bar and not _bar_has_events(bm, bars[j], events):
                j += 1
            run = j - i
            if run >= 2 or i < first_bar:
                text = f" {run} bar{'s' if run > 1 else ''} rest "
                w = max(len(text) + 4, 12)
                mid = n // 2
                lines = []
                for li in range(n):
                    if li == mid:
                        lines.append(text.center(w, "-"))
                    else:
                        lines.append("-" * w)
                blocks.append((i + 1, " " * w, lines, text))
                i = j
                continue
        count, lines = _render_bar(bm, bar, n, events)
        blocks.append((i + 1, count, lines, None))
        i += 1

    out = list(header) + [""]
    system: list = []

    def flush():
        if not system:
            return
        nums = " " * (lab_w + 1)
        counts = " " * (lab_w + 1)
        shown = list(range(n))
        if sparse:
            shown = [li for li in range(n)
                     if any(rest is None and lines[li].strip("-") for _, _, lines, rest in system)]
            shown = shown or [n // 2]
        rows = {li: labels[li] + "|" for li in shown}
        for num, count, lines, rest in system:
            tag = str(num)
            nums += tag + " " * (len(lines[0]) + 1 - len(tag))
            counts += count + " "
            for k, li in enumerate(shown):
                if rest is not None:  # "N bars rest": text on the middle shown line
                    w = len(lines[0])
                    rows[li] += (rest.center(w, "-") if k == len(shown) // 2 else "-" * w) + "|"
                else:
                    rows[li] += lines[li] + "|"
        out.append(nums.rstrip())
        out.append(counts.rstrip())
        out.extend(rows[li] for li in shown)
        out.append("")
        system.clear()

    width = lab_w + 1
    for blk in blocks:
        w = len(blk[2][0]) + 1
        if system and width + w > MAX_WIDTH:
            flush()
            width = lab_w + 1
        system.append(blk)
        width += w
    flush()
    if footer:
        out.extend(footer)
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------
# Drums

# (label, {GM note: symbol}) top to bottom.
DRUM_LINES = [
    ("CC", {49: "X", 57: "X", 52: "C", 55: "s"}),
    ("RD", {51: "x", 59: "x", 53: "b"}),
    ("HH", {42: "x", 46: "o", 26: "o"}),
    ("T1", {50: "o", 48: "o"}),
    ("T2", {47: "o", 45: "o"}),
    ("SN", {38: "o", 40: "o", 37: "x", 39: "c"}),
    ("FT", {43: "o", 41: "o"}),
    ("BD", {36: "o", 35: "o"}),
    ("HF", {44: "x"}),
    ("PC", {}),  # everything else (cowbell, tambourine, ...)
]
ALWAYS_SHOWN = {"HH", "SN", "BD"}

DRUM_LEGEND = [
    "Legend:  CC crash (X crash, C china, s splash)   RD ride (x, b = bell)",
    "         HH hi-hat (x closed, o open)   SN snare (o hit, x side-stick, c clap)",
    "         T1/T2 toms   FT floor tom   BD bass drum   HF hi-hat pedal   PC other percussion",
]


def drum_line_for(pitch: int) -> tuple[str, str]:
    for label, table in DRUM_LINES:
        if pitch in table:
            return label, table[pitch]
    return "PC", "x"


# ----------------------------------------------------------------------------
# Keyboards: a note roll, one line per pitch

_ROLL_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def pitch_label(pitch: int) -> str:
    """Scientific pitch name: 60 -> C4 (middle C)."""
    return f"{_ROLL_NAMES[pitch % 12]}{pitch // 12 - 1}"


def render_roll(header: list[str], bm: BeatMap, pitches_at: dict[int, list[int]], footer: list[str] | None = None) -> str:
    """Notes as a roll: time runs left to right like a tab, each line is one
    pitch (highest on top), "o" marks where a note starts. `pitches_at` maps
    tick -> pitches starting there."""
    used = sorted({p for ps in pitches_at.values() for p in ps}, reverse=True)
    if not used:
        return "\n".join(header + ["", "(no notes were detected for this part)", ""])
    line_of = {p: i for i, p in enumerate(used)}
    events = {t: {line_of[p]: "o" for p in ps} for t, ps in pitches_at.items()}
    return render(header, [pitch_label(p) for p in used], bm, events, footer, sparse=True)
