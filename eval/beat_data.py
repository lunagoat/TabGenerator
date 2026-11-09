"""Beat test data: beat_this frame logits (cached) + annotated beats/downbeats."""
import json

import env
import numpy as np

CACHE = env.WORK / "beats"
FPS = 50


def _logits(key: str, wav):
    path = CACHE / f"{key}.npz"
    if path.is_file():
        z = np.load(path)
        return z["beat"], z["downbeat"]
    from beat_this.inference import Audio2Frames
    from tabgen.pipeline import load_mono

    global _model
    try:
        _model
    except NameError:
        _model = Audio2Frames(checkpoint_path="final0", device="cuda")
    b, d = _model(load_mono(wav, 22050), 22050)
    b, d = b.cpu().numpy(), d.cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, beat=b, downbeat=d)
    return b, d


def mdb():
    root = env.DATA / "MDBDrums" / "MDB Drums"
    out = []
    for wav in sorted((root / "audio" / "full_mix").glob("*.wav")):
        rows = [l.split() for l in (root / "annotations" / "beats" / f"{wav.stem}.beats").read_text().splitlines() if l.strip()]
        beats = np.array([float(r[0]) for r in rows])
        downs = np.array([float(r[0]) for r in rows if int(float(r[1])) == 1])
        out.append((f"mdb/{wav.stem}", *_logits(f"mdb/{wav.stem}", wav), beats, downs))
    return out


def babyslakh():
    """Beats from the MIDI tempo map and time signatures."""
    import mido
    import soundfile as sf

    out = []
    for track in sorted((env.DATA / "babyslakh_16k").glob("Track*")):
        mid = mido.MidiFile(str(track / "all_src.mid"))
        dur = sf.info(str(track / "mix.wav")).duration
        tpb = mid.ticks_per_beat
        tempo_ev, sig_ev = [(0, 500000)], [(0, 4, 4)]
        for tr in mid.tracks:
            tick = 0
            for msg in tr:
                tick += msg.time
                if msg.type == "set_tempo":
                    tempo_ev.append((tick, msg.tempo))
                elif msg.type == "time_signature":
                    sig_ev.append((tick, msg.numerator, msg.denominator))
        tempo_ev.sort()
        sig_ev.sort()

        def to_s(tick):
            t, last, tempo = 0.0, 0, 500000
            for tk, tp in tempo_ev:
                if tk >= tick:
                    break
                t += (tk - last) * tempo / tpb / 1e6
                last, tempo = tk, tp
            return t + (tick - last) * tempo / tpb / 1e6

        beats, downs = [], []
        tick, si = 0, 0
        while True:
            # the signature in force at this bar
            while si + 1 < len(sig_ev) and sig_ev[si + 1][0] <= tick:
                si += 1
            _, num, den = sig_ev[si]
            step = tpb * 4 // den
            compound = den == 8 and num % 3 == 0 and num >= 6
            if to_s(tick) > dur:
                break
            downs.append(to_s(tick))
            for k in range(num):
                if not compound or k % 3 == 0:
                    beats.append(to_s(tick + k * step))
            tick += num * step
        beats = np.array([b for b in beats if b <= dur])
        out.append((f"babyslakh/{track.name}", *_logits(f"babyslakh/{track.name}", track / "mix.wav"), beats, np.array(downs)))
    return out


def guitarset(limit=45):
    audio = env.DATA / "guitarset" / "audio_mono-mic"
    files = sorted(audio.glob("*_mic.wav"))
    files = files[:: max(1, len(files) // limit)][:limit]
    out = []
    for wav in files:
        name = wav.stem[: -len("_mic")]
        jam = json.loads((env.DATA / "guitarset" / "annotation" / f"{name}.jams").read_text())
        ann = next(a for a in jam["annotations"] if a["namespace"] == "beat_position")
        beats = np.array([d["time"] for d in ann["data"]])
        downs = np.array([d["time"] for d in ann["data"] if d["value"]["position"] == 1])
        out.append((f"guitarset/{name}", *_logits(f"guitarset/{name}", wav), beats, downs))
    return out


def user_songs():
    """The songs in .cache (no annotations)."""
    out = []
    for c in sorted((env.ROOT / ".cache").iterdir()):
        wav = c / f"{c.name}.wav"
        if wav.is_file():
            out.append((f"user/{c.name}", *_logits(f"user/{c.name}", wav), None, None))
    return out


def fmeasure(est, ref, tol=0.07):
    est, ref = np.sort(est), np.sort(ref)
    i = j = tp = 0
    while i < len(est) and j < len(ref):
        d = est[i] - ref[j]
        if abs(d) <= tol:
            tp += 1
            i += 1
            j += 1
        elif d < 0:
            i += 1
        else:
            j += 1
    if not len(est) or not len(ref):
        return 0.0
    p, r = tp / len(est), tp / len(ref)
    return 2 * p * r / max(1e-9, p + r)


if __name__ == "__main__":
    for fn in (mdb, babyslakh, guitarset, user_songs):
        rows = fn()
        print(fn.__name__, len(rows))
