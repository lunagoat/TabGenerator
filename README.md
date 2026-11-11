# TabGenerator

Turns any audio file (flac, wav, mp3, m4a, ogg, opus, even video files) into
**guitar, bass and drum tabs**, plus piano, synth and vocal-melody parts.

## Install

Linux, Python 3.10 or newer, ffmpeg, and ideally an NVIDIA GPU (about 6 GB
of video memory; it runs on the CPU too, very slowly).

```sh
git clone https://github.com/lunagoat/TabGenerator && cd TabGenerator
./install.sh
```

The installer puts every Python package into `./_packages` (no virtual
environment, nothing system-wide; delete the folder to uninstall) and
downloads the transcription model. That model is free but gated: accept its
license at https://huggingface.co/MuScriptor/muscriptor-large and log in to
Hugging Face when the installer asks. `./install.sh --cpu` forces the CPU
build of PyTorch.

## Use

```sh
./launch.sh "path/to/song.flac"
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

## Piano, synth and vocals

Keyboards are found and written out automatically, next to guitar, bass and
drums: **Piano** comes from the separator's piano stem, **Synth** is
whatever pitched is left in its "other" stem (synth leads, pads, bells,
strings; the header says what it sounded like). A part that is not in the
song is skipped: a silent stem is never transcribed, so a piano piece gets
no empty guitar tab and a rock song no empty piano part.

`--keys-as` chooses how they are written:

| | Text file | Guitar Pro file |
|---|---|---|
| `piano` (default) | a note roll: one line per pitch, time left to right | a track to read as standard notation (a piano gets right-hand and left-hand tracks) |
| `guitar` | guitar tablature: notes moved by octaves into the guitar's range, then fingered and checked for playability like a guitar part | a normal 6-string tab track that keeps the piano/synth sound |
| `both` | both files | both sets of tracks |

Guitar Pro 5 files only have stringed tracks, so the notation tracks sit on
a made-up 7-string instrument tuned in fifths; ignore their tab line and
read the score. The files open in TuxGuitar: `eval/tuxcheck.sh file.gp5`
loads one with TuxGuitar's own reader and lists what it finds.

**Vocals** work the same way: the sung melody is transcribed from the vocal
stem and written as a note roll and a notation track, or with
`--vocals-as guitar` (or `both`) as guitar tablature. It is the melody only:
no lyrics, and backing harmonies come out as extra notes in the same part.

Switching `--keys-as` or `--vocals-as` on a song that was already run takes
seconds.

## How it works

1. **Decode** with ffmpeg (anything ffmpeg reads).
2. **Stem separation** with BS-RoFormer SW (drums, bass, guitar, piano,
   vocals, other). Its stems are much cleaner than Demucs `htdemucs_6s`,
   which matters more than anything else for the notes that come out.
   `--separator demucs` uses Demucs through StemSplitter instead.
3. **Beat and bar tracking**: beat_this's beat/downbeat activations, decoded
   with a tempo-continuity search (`tabgen/beats.py`) instead of plain peak
   picking, so quiet intros and breaks keep a steady grid and bar lines
   don't jump around when a downbeat is missed.
4. **Note transcription** with MuScriptor-large (Kyutai x Mirelo, July 2026).
   Each part's stem is transcribed several times with the audio shifted by a
   couple of seconds: the model works in 5-second chunks and makes most of
   its mistakes near chunk edges, and a shifted pass has its edges elsewhere.
   A note is kept when two passes agree on it, and its timing is the median
   of the passes. All passes are decoded together on the GPU in half
   precision, which is several times faster than one after another.
5. **Rhythm**: bars of six, nine or twelve tracked beats are written in
   6/8, 9/8 or 12/8, and the tempo is halved or doubled when a person would
   clearly write it that way (learned from the tabs). Each part gets, per beat, the subdivision (8ths, 16ths,
   triplets, 32nds) that best explains its own notes; a lone triplet beat in
   a straight passage needs clear evidence. 6/8 and 12/8 feels are detected.
6. **Tidying** (`tabgen/arrange.py`): what turns a list of heard notes
   into something a player would write down.
   - A line another guitar plays on top of a riff is moved out of the riff's
     part (the riff comes round with and without it).
   - Bars that are clearly the same riff are compared: a note only one
     repeat has is dropped, a note nearly all repeats have is filled in
     where it went missing.
   - A strum that is nearly a chord played all around it is written as that
     chord, so the same chord is the same shape every time.
   - High notes that are overtones of a distorted low string are removed.
   - After fingering, a playability check leaves out notes that would need
     the hand five or more frets away for a single strum, and counts the
     quick jumps that remain at the bottom of the tab.

   Each tab ends with a note of what was changed. `--raw` turns all of it
   off and writes the notes exactly as transcribed.
7. **Tuning detection** (E standard, Eb, Drop D, D, Drop C#, Drop C, C, B,
   7-string, 5-string bass, ...) from the lowest notes played and how
   naturally the part fits each tuning. Distorted power chords make ghost
   notes an octave below their root; those are ignored. The bass follows the
   guitar's tuning unless its notes clearly need a lower one.
8. **Separate guitar parts**: the guitar stem is split by stereo position
   (left / center / right) and each side is transcribed on its own; each
   note goes to the part whose side it was heard on. A riff double-tracked
   hard left and right stays one part. A position is split again when two
   guitar sounds (clean and distorted) play there at the same time; one
   guitarist switching sounds between sections stays one part. Mono
   recordings skip this.
9. **Fingering**: a Viterbi search over string/fret choices that tracks where
   the fretting hand is, penalising stretches, big hand shifts, string skips
   and putting two notes that ring together on one string. A riff that
   uses open strings is kept in open position.

## Options

```
--parts guitar,bass     only some parts (guitar, bass, drums, piano, synth, vocals; keys = piano + synth)
--vocals-as guitar      the sung melody as guitar tablature (or: both)
--keys-as guitar        piano/synth as guitar tablature (or: both)
--guitar-tuning "Drop D"   override the detected tuning (also --bass-tuning); also open tunings
                        ("Open G", "Open D", "DADGAD", "Open E", "Open G, low G") and a capo
                        ("E Standard, capo 2"), which detection rarely picks on its own
--one-guitar            one guitar tab with everything (no stereo/sound split; saves 3 passes)
--fast                  one transcription per part instead of voting
--separator demucs      Demucs htdemucs_6s (via StemSplitter) instead of BS-RoFormer
--batch 4               fewer transcriptions decoded at once (if the GPU runs out of memory)
--model medium          faster, less accurate (default: large)
--beam 4                beam search (much slower; made no measurable difference in testing)
--raw                   notes as transcribed: no tidying, no playability check
--resplit               redo stem separation even if stems exist
--retranscribe          ignore cached transcriptions
-o DIR                  output folder
--no-open               don't open the output folder when done
```

Intermediate results (decoded audio, stems, beats, notes) are cached in
`.cache/`, so re-running a song with different tab options is instant.

## Setup details

`./install.sh` installs into `./_packages`. If a StemSplitter folder sits
next to this one, its packages are used as a fallback, and
`--separator demucs` uses its Demucs model; nothing else needs it.

MuScriptor's weights are read from `~/muscriptor-large/model.safetensors` if
that exists (`--weights PATH` or `MUSCRIPTOR_WEIGHTS` point elsewhere);
otherwise from the Hugging Face cache. The BS-RoFormer separator weights
(700 MB) are downloaded to `~/.cache/tabgenerator/` on the first run (or set
`BS_ROFORMER_WEIGHTS`).

The large model needs about 6 GB of GPU memory at the default `--batch 8`.
On the CPU use `--model small`.

The evaluation scripts in `eval/` need test data downloaded separately into
`eval/data/` and, for Guitar Pro reading, TuxGuitar from Flathub.

## How accurate is it?

Note F1 scores (1.0 = every note found, nothing extra). A note counts as
correct when its pitch is exact (drums: the right kind of drum) and it
starts within 50 ms of the real one. "Before" is the previous version of
this tool (Demucs stems, voting over different stem mixes).

| Part | Test set | Before | **Now** |
|---|---|---|---|
| Guitar | real playing through distortion, in a band mix (24 excerpts) | 0.638 | **0.778** |
| Guitar | the same playing clean, in a band mix | 0.677 | **0.821** |
| Bass | the same band mixes (distorted / clean guitar) | 0.866 / 0.867 | **0.961 / 0.926** |
| Bass | BabySlakh: 19 synthetic songs, 60 s each | 0.714 | **0.914** |
| Drums | MDB Drums: 23 real recordings | 0.824 | **0.860** |
| Drums | the band mixes | | **0.934** |

The band mixes are made for testing: real guitar playing from GuitarSet
(its pickup signal, clean or through a simulated high-gain amp), real drums
from MDB Drums and a bass stem from Slakh, each with its true notes. They
don't play the same song, but separation and transcription don't know that.
MDB Drums is mostly jazz with brushes and ghost notes; on its rock-style
songs the drum score is 0.88 to 1.0. On BabySlakh the bass truth was moved to
the octave that actually sounds (Slakh's bass patches play an octave below
their MIDI notes).

Where the gain comes from, on the distorted-guitar mixes: cleaner stems and
listening to the guitar stem alone instead of the whole band 0.638 -> 0.717,
voting over shifted passes -> 0.778. For drums on MDB: stems 0.824 -> 0.847,
shifted passes -> 0.860.

Tidying trades a little of that note score for tabs that can be read and
played: on the band mixes (30-second jazz and funk excerpts with almost no
repeated bars, so they show its cost and not its benefit) guitar F1 goes
from 0.779 to 0.773 (distorted) and 0.821 to 0.807 (clean).

String/fret choice is learned from human-made tabs, in two parts: a small
neural network (`tabgen/fingering_model.py`, trained on about 100,000 guitar
tracks from Guitar Pro collections) reads each part forwards and backwards
and says how likely each string is for every note; counted chord shapes and
hand-movement costs then pick a playable path. Given the right notes, the
string the tabber chose is picked for 87.9% of notes on 27 Official
Ultimate Guitar guitar tracks (84.8% without the network) and 83.3% on
GuitarSet's real players (77.0%). On the full pipeline's own transcriptions
of the Official songs it is 76.3% (74.6%). `eval/fingering_data.py` and
`eval/train_fingering_model.py` retrain it.

Speed on an RTX 5070 Ti: about 5.5 minutes for a 5-minute stereo song
(11 transcriptions), down from roughly 13 minutes for a 4-minute song.
Re-running a song reuses the cached stems and transcriptions and takes
seconds.

Everything in `eval/` reproduces these numbers (`eval/recipes.py`,
`eval/eval_fingering.py`, `eval/eval_beats.py`, `eval/eval_rhythm.py`); the
test data is downloaded separately into `eval/data/`.

## Limitations

- No playing techniques (bends, slides, hammer-ons, palm mutes, harmonics);
  the tabs show which notes are played, where and when.
- String/fret choice is inferred, not heard. It matches the real player's
  choice for about 3 notes in 4 on GuitarSet; the rest are playable
  alternatives in another position.
- Capo and unusual open tunings are not detected; use `--guitar-tuning`.
- Songs without a steady pulse (rubato, free time) get messy bar lines.
