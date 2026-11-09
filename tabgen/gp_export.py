"""Guitar Pro 5 export (opens in Guitar Pro, TuxGuitar, MuseScore, Songsterr)."""

from dataclasses import dataclass

import guitarpro as gp

from .rhythm import BeatMap

# ticks -> (note value, dotted, is 3:2 triplet)
_DURATIONS = {
    5760: (1, True, False),
    3840: (1, False, False),
    2880: (2, True, False),
    1920: (2, False, False),
    1440: (4, True, False),
    960: (4, False, False),
    720: (8, True, False),
    480: (8, False, False),
    360: (16, True, False),
    240: (16, False, False),
    180: (32, True, False),
    120: (32, False, False),
    60: (64, False, False),
    640: (4, False, True),
    320: (8, False, True),
    160: (16, False, True),
    80: (32, False, True),
}
_STRAIGHT = [960, 720, 480, 360, 240, 180, 120, 60]
_TRIPLET = [640, 320, 160, 80]
_COMPOUND = [1440, 960, 720, 480, 360, 240, 180, 120, 60]


@dataclass
class GPEvent:
    tick: int
    end: int
    # (string index from the lowest string, fret) for fretted parts; MIDI
    # drum notes for drums.
    notes: list


@dataclass
class GPPart:
    name: str
    events: list[GPEvent]
    strings: tuple[int, ...] | None = None  # open-string pitches, lowest first; None = drums
    program: int = 30
    fret_count: int = 24
    bm: BeatMap | None = None  # this part's own rhythmic grid (default: the song's)


def _gp_duration(ticks: int) -> gp.Duration:
    value, dotted, triplet = _DURATIONS[ticks]
    tuplet = gp.Tuplet(3, 2) if triplet else gp.Tuplet(1, 1)
    return gp.Duration(value=value, isDotted=dotted, tuplet=tuplet)


def _spell(bm: BeatMap, start: int, end: int) -> list[int]:
    """Split [start, end) into lengths that each have a notation value."""
    bt = bm.beat_ticks
    out = []
    pos = start
    while pos < end:
        beat = pos // bt
        in_beat = pos - beat * bt
        s = int(bm.subdivisions[beat]) if beat < len(bm.subdivisions) else 1
        triplet = (not bm.compound) and s in (3, 6, 12)
        if in_beat == 0 and not triplet:
            # Whole beats at once, as long as none of them are triplet beats.
            n = 0
            while (
                pos + (n + 1) * bt <= end
                and (beat + n >= len(bm.subdivisions) or bm.compound or int(bm.subdivisions[beat + n]) not in (3, 6, 12))
            ):
                n += 1
            if n:
                for k in (4, 3, 2, 1):
                    if k <= n and k * bt in _DURATIONS:
                        out.append(k * bt)
                        pos += k * bt
                        break
                continue
        room = min(end, (beat + 1) * bt) - pos
        table = _TRIPLET if triplet else (_COMPOUND if bm.compound else _STRAIGHT)
        step = next((d for d in table if d <= room), None)
        if step is None:
            # Shouldn't happen with grid-aligned ticks; round up to finish.
            step = table[-1]
        out.append(step)
        pos += step
    return out


def _make_track(song: gp.Song, number: int, part: GPPart, channel: int) -> gp.Track:
    track = gp.Track(song, number=number, name=part.name[:40], measures=[])
    if part.strings is None:
        track.isPercussionTrack = True
        track.strings = [gp.GuitarString(i + 1, 0) for i in range(6)]
        track.channel = gp.MidiChannel(channel=9, effectChannel=9, instrument=0)
    else:
        track.strings = [gp.GuitarString(i + 1, v) for i, v in enumerate(reversed(part.strings))]
        track.channel = gp.MidiChannel(channel=channel, effectChannel=channel + 1, instrument=part.program)
        track.fretCount = part.fret_count
    track.indicateTuning = True
    for header in song.measureHeaders:
        track.measures.append(gp.Measure(track, header))
    return track


def _fill_track(bm: BeatMap, track: gp.Track, part: GPPart, tempo_marks: dict[int, int]):
    bt = bm.beat_ticks
    n_strings = len(track.strings)
    events = sorted(part.events, key=lambda e: e.tick)
    for bi, (bar, measure) in enumerate(zip(bm.bars, track.measures)):
        voice = measure.voices[0]
        bar_start = bar.first_beat * bt
        bar_end = bar_start + bar.n_beats * bt
        # Segments inside this bar: (start, end, notes, is_continuation)
        segs = []
        for i, ev in enumerate(events):
            nxt = events[i + 1].tick if i + 1 < len(events) else 1 << 60
            end = min(max(ev.end, ev.tick + 1), nxt)
            if end <= bar_start or ev.tick >= bar_end:
                continue
            s = max(ev.tick, bar_start)
            e = min(end, bar_end)
            segs.append((s, e, ev.notes, ev.tick < bar_start))
        pos = bar_start
        first_beat_obj = None
        for s, e, notes, cont in segs:
            if s > pos:
                for d in _spell(bm, pos, s):
                    b = gp.Beat(voice, duration=_gp_duration(d), status=gp.BeatStatus.rest)
                    voice.beats.append(b)
                    first_beat_obj = first_beat_obj or b
            for k, d in enumerate(_spell(bm, s, e)):
                tie = cont or k > 0
                if part.strings is None and tie:
                    b = gp.Beat(voice, duration=_gp_duration(d), status=gp.BeatStatus.rest)
                else:
                    b = gp.Beat(voice, duration=_gp_duration(d), status=gp.BeatStatus.normal)
                    if part.strings is None:
                        for j, pitch in enumerate(notes[:n_strings]):
                            b.notes.append(gp.Note(b, value=pitch, string=j + 1, type=gp.NoteType.normal))
                    else:
                        for string_idx, fret in notes:
                            gp_string = n_strings - string_idx  # GP counts from the top string
                            b.notes.append(
                                gp.Note(
                                    b,
                                    value=fret,
                                    string=gp_string,
                                    type=gp.NoteType.tie if tie else gp.NoteType.normal,
                                )
                            )
                voice.beats.append(b)
                first_beat_obj = first_beat_obj or b
            pos = e
        if pos < bar_end:
            for d in _spell(bm, pos, bar_end):
                b = gp.Beat(voice, duration=_gp_duration(d), status=gp.BeatStatus.rest)
                voice.beats.append(b)
                first_beat_obj = first_beat_obj or b
        if bi in tempo_marks and first_beat_obj is not None:
            first_beat_obj.effect.mixTableChange = gp.MixTableChange(
                tempo=gp.MixTableItem(value=tempo_marks[bi], duration=0, allTracks=True),
                hideTempo=False,
            )


def write_gp5(path: str, title: str, artist: str, bm: BeatMap, parts: list[GPPart]):
    tempos = [int(round(bm.bar_tempo(b))) for b in bm.bars]
    song = gp.Song(title=title, artist=artist, tempo=tempos[0] if tempos else 120)
    song.tab = "Generated by TabGenerator (MuScriptor + Demucs)"
    song.measureHeaders = []
    song.tracks = []
    for i, bar in enumerate(bm.bars):
        num, den = bm.time_signature(bar)
        header = gp.MeasureHeader(number=i + 1)
        header.timeSignature = gp.TimeSignature(numerator=num, denominator=gp.Duration(value=den))
        song.addMeasureHeader(header)
    # Tempo changes wherever the played tempo drifts by 2+ BPM, so playback
    # stays in sync with the recording.
    tempo_marks = {}
    current = song.tempo
    for i, t in enumerate(tempos):
        if i > 0 and abs(t - current) >= 2:
            tempo_marks[i] = t
            current = t
    channel = 0
    for n, part in enumerate(parts):
        if part.strings is not None and channel == 9:
            channel = 10
        track = _make_track(song, n + 1, part, channel)
        _fill_track(part.bm or bm, track, part, tempo_marks if n == 0 else {})
        song.tracks.append(track)
        if part.strings is not None:
            channel += 2
    gp.write(song, path, version=(5, 1, 0))
