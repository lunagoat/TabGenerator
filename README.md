# TabGenerator

Turns any audio file (flac, wav, mp3, m4a, ogg, opus, even video files) into
**guitar, bass and drum tabs**.

Run from a terminal (quotes are required because the path contains spaces):

```sh
"/home/luna/Desktop/vibecode slop/TabGenerator/launch.sh" "path/to/song.flac"
```

Without a file argument it asks for one (drag and drop works). Several files
can be given at once. Output goes to `Tabs/<song name>/`:

| File | What it is |
|---|---|
| `<song> - Guitar N (...).txt` | ASCII guitar tab, one per guitar part: guitars are told apart by where they sit in the stereo mix (left / center / right) and by sound (clean / distorted / acoustic) |
| `<song> - Bass.txt` | ASCII bass tab |
| `<song> - Drums.txt` | ASCII drum tab |
| `<song>.gp5` | Guitar Pro 5 file with all parts, real rhythms and tempo changes that follow the recording. Opens in Guitar Pro, TuxGuitar, MuseScore, Songsterr |
| `<song>.mid` | Raw (unquantized) MIDI of everything transcribed |

The `.gp5` is the best way to check a tab: play it back alongside the song.

## How it works

1. **Decode** with ffmpeg (anything ffmpeg reads).
2. **Stem separation** with StemSplitter (the folder next to this one): Demucs
   `htdemucs_6s`, with random-shift averaging for cleaner stems. Stems that
   StemSplitter already made for a song are reused.
3. **Beat and bar tracking** with beat_this, smoothed per beat so songs
   without a click track still line up.
4. **Note transcription** with MuScriptor-large (Kyutai x Mirelo, July 2026),
   the most accurate open multi-instrument transcription model available,
   run 8 times on different stem mixes (plus 3 stereo-position passes for
   guitar); each part keeps the notes that at least two of its
   transcriptions agree on.
5. **Rhythm**: each beat gets the subdivision (8ths, 16ths, triplets, 32nds)
   that best explains its notes; 6/8 and 12/8 feels are detected.
6. **Tuning detection** (E standard, Eb, Drop D, D, Drop C#, Drop C, C, B,
   7-string, 5-string bass, ...) from the lowest notes played and how
   naturally the part fits each tuning. Distorted power chords make ghost
   notes an octave below their root; those are ignored. The bass follows the
   guitar's tuning unless its notes clearly need a lower one.
7. **Separate guitar parts**: the guitar audio is split by stereo position
   (left / center / right) and each side is transcribed on its own. Those
   transcriptions also vote on the guitar notes (a guitar that is quiet in
   the mono mix is heard on its own side), and each note goes to the part
   whose side it was heard on. A riff double-tracked hard left and right
   stays one part. Each position is split again when it holds two guitar
   sounds (clean and distorted). Mono recordings skip this.
8. **Fingering**: a Viterbi search over string/fret choices that tracks where
   the fretting hand is, penalising stretches, big hand shifts, string skips
   and putting two notes that ring together on one string.

## Options

```
--parts guitar,bass     only some parts
--guitar-tuning "Drop D"   override the detected tuning (also --bass-tuning)
--one-guitar            one guitar tab with everything (no stereo/sound split; saves 3 passes)
--fast                  one transcription per part instead of voting (~3x faster)
--model medium          faster, slightly less accurate (default: large)
--beam 4                beam search (slower; made no measurable difference in testing)
--shifts 5              more Demucs shift averaging (slower, marginally cleaner stems)
--resplit               redo stem separation even if stems exist
--retranscribe          ignore cached transcriptions
-o DIR                  output folder
--no-open               don't open the output folder when done
```

Intermediate results (decoded audio, beats, notes) are cached in `.cache/`, so
re-running a song with different tab options is instant.

## Setup

The script uses base Python and loads packages from `_packages` (no virtual
environment), plus PyTorch and Demucs from `StemSplitter/_packages`. To
install or restore packages, run `./setup.sh` from this folder (and
StemSplitter's `setup.sh` if PyTorch/Demucs are missing).

The MuScriptor weights are gated (free, non-commercial CC BY-NC 4.0 license).
They are read from `~/muscriptor-large/model.safetensors`; to put them
elsewhere use `--weights PATH` or set `MUSCRIPTOR_WEIGHTS`. If there is no
local copy, they are downloaded using your Hugging Face login.

The large model needs about 7 GB of GPU memory. It runs on the CPU too, but
very slowly; use `--model small` there.

## How accurate is it?

All numbers are note F1 scores (1.0 = every note found, nothing extra).
A note counts as correct when its pitch is exact (drums: the right drum)
and it starts within 50 ms of the real one.

| Part | Test set | Full mix only | Best single stem | **TabGenerator** | Ceiling |
|---|---|---|---|---|---|
| Drums | MDB Drums: 23 real recordings | 0.711 | 0.758 | **0.767** | 0.812 (isolated drum tracks) |
| Bass | BabySlakh: 8 songs, 60 s each | 0.606 | 0.746 | **0.792** | |
| Guitar | BabySlakh: 8 songs, 60 s each | 0.361 | 0.424 | **0.483** | |

The gains come from voting: each part is transcribed several times from
different audio (its Demucs stem, the band without drums/vocals, the full
mix, with and without restricting the model to that instrument), and a note
is kept when at least two transcriptions agree on it. Bass additionally keeps
only the lowest note when several start together. `--fast` uses the single
best transcription per part instead (about 3x faster).

The guitar number is pessimistic: BabySlakh songs stack 3 to 5 synthetic
guitars, and its 16 kHz audio handicaps Demucs, so stems do better on real
songs (as the drum test on real recordings shows).

Guitar parts, on BabySlakh songs remixed with 2-3 guitars panned apart:
78% of notes played by only one guitar end up in that guitar's tab, and 95%
of a double-tracked part stays together. Letting the stereo-position
transcriptions vote found more guitar notes (F1 0.502 -> 0.523) and made
each part's tab more accurate (0.427 -> 0.451).

String/fret choice, scored on GuitarSet's real tabs (given the right
notes): the string the real player used is chosen for 73% of notes, vs
54.5% for "always the lowest fret". The rest are playable alternatives in
another position.

Speed on an RTX 5070 Ti: about 13 minutes for a 4-minute stereo song (about
4 with `--fast`, which also skips the guitar split). Re-running
a song reuses the cached stems and transcriptions and takes seconds.

## Limitations

- No playing techniques (bends, slides, hammer-ons, palm mutes, harmonics);
  the tabs show which notes are played, where and when.
- String/fret choice is inferred, not heard. It matches the real player's
  choice for about 3 notes in 4 on GuitarSet; the rest are playable
  alternatives in another position.
- Capo and unusual open tunings are not detected; use `--guitar-tuning`.
- Songs without a steady pulse (rubato, free time) get messy bar lines.
