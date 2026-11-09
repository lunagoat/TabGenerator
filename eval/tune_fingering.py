#!/usr/bin/env python3
"""Diagnostics and player-held-out tuning of the fingering chooser.

Run with system python3 (no additional dependencies):
    python3 eval/tune_fingering.py --diagnostic --benchmark
    python3 eval/tune_fingering.py --grid --cv

The grid is fixed before scoring. For each of six folds, choose weights using
only the other five players, requiring their aggregate comp and Rock scores
not to regress. Pool the held-out note counts, rather than averaging player
percentages. The all-data optimum is reported separately from that estimate.
Raw per-file trial counts are cached in --output so a run can be resumed.
"""
import argparse
from collections import Counter
from dataclasses import asdict, replace
from itertools import product
import json
from pathlib import Path
import time

import env  # noqa: F401
import numpy as np

from eval_fingering import ANNOT, E_STD, load_notes, to_events
from tabgen import fretboard as fb


def baseline_weights():
    """The original chooser, retained independently of future default changes."""
    return fb.FingeringWeights(
        span=0., span_hard=5, stretch=1., height=.02, open_bonus=0.,
        open_high_mix=0., string_gap=1.2, move=1.5, shift=0., ease_time=.35,
        string_jump=.36, ringing_clash=3., top_k=16, run_move=0., rest_ease=0.,
    )


def dataset():
    return [(p.stem, *to_events(load_notes(p))) for p in sorted(ANNOT.glob('*.jams'))]


def evaluate(data, weights, diagnostic=False):
    rows = []
    confusion = Counter()
    frets = {'comp': Counter(), 'solo': Counter()}
    for name, events, truth in data:
        fingerings, _ = fb.assign_frets(events, E_STD, weights=weights)
        right = total = 0
        for fingering, target in zip(fingerings, truth):
            got = dict(zip(fingering.pitches, fingering.strings))
            for pitch, string in target.items():
                chosen = got.get(pitch, -1)
                right += chosen == string
                total += 1
                if diagnostic:
                    style = name.rsplit('_', 1)[1]
                    confusion[style, string, chosen] += 1
                    fret = pitch - E_STD.strings[string]
                    predicted = pitch - E_STD.strings[chosen] if chosen >= 0 else -1
                    frets[style][fret, predicted] += 1
        rows.append((name, right, total))
    if diagnostic:
        for style in frets:
            print(style, 'true string (rows), predicted string (columns):')
            print(np.array([[confusion[style, s, g] for g in range(6)] for s in range(6)]))
            wrong = [(pair, n) for pair, n in frets[style].most_common() if pair[0] != pair[1]]
            print('Most common wrong fret pairs:', wrong[:20])
        weakest = sorted((r / t, name) for name, r, t in rows if name.endswith('solo') and t)
        print('Weakest solo files:', weakest[:15])
    return rows


def summary(rows):
    groups = {}
    for name, right, total in rows:
        genre = name.split('_')[1].split('-')[0].rstrip('0123456789')
        for key in ('overall', name[:2], name.rsplit('_', 1)[1], genre):
            counts = groups.setdefault(key, [0, 0])
            counts[0] += right
            counts[1] += total
    return {key: right / total for key, (right, total) in groups.items()}


def select(records, exclude=None):
    def training(rec):
        return summary(row for row in rec['rows'] if row[0][:2] != exclude)

    baseline = training(records[0])
    eligible = [rec for rec in records
                if rec['weights']['height'] >= .02
                and training(rec)['comp'] >= baseline['comp']
                and training(rec)['Rock'] >= baseline['Rock']]
    return max(eligible, key=lambda rec: training(rec)['overall'])


def cross_validate(records):
    held_out, folds = [], {}
    for player in ('00', '01', '02', '03', '04', '05'):
        best = select(records, exclude=player)
        test = [row for row in best['rows'] if row[0][:2] == player]
        held_out.extend(test)
        folds[player] = {'scores': summary(test), 'weights': best['weights']}
        print('Fold', player, json.dumps(folds[player]), flush=True)
    scores = summary(held_out)
    print('LOPO', json.dumps(scores), flush=True)
    best = select(records)
    print('All-data optimum', json.dumps({'scores': summary(best['rows']), 'weights': best['weights']}))
    return {'scores': scores, 'folds': folds}


def benchmark(data, weights):
    result = {}
    for label, key in (('most_events', lambda x: len(x[1])),
                       ('most_notes', lambda x: sum(len(e.pitches) for e in x[1]))):
        name, events, _ = max(data, key=key)
        timings = []
        for _ in range(7):
            start = time.perf_counter()
            fb.assign_frets(events, E_STD, weights=weights)
            timings.append(time.perf_counter() - start)
        result[label] = {'file': name, 'events': len(events),
                         'notes': sum(len(e.pitches) for e in events),
                         'median_seconds': float(np.median(timings))}
    _, source, _ = max(data, key=lambda x: len(x[1]))
    period = source[-1].time + 2.
    events = []
    for i in range(3000):
        event = source[i % len(source)]
        offset = (i // len(source)) * period
        events.append(fb.Event(event.time + offset, list(event.pitches),
                               [end + offset for end in event.ends]))
    start = time.perf_counter()
    fb.assign_frets(events, E_STD, weights=weights)
    result['3000_events_seconds'] = time.perf_counter() - start
    print('Runtime', json.dumps(result), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--diagnostic', action='store_true')
    ap.add_argument('--benchmark', action='store_true')
    ap.add_argument('--grid', action='store_true')
    ap.add_argument('--cv', action='store_true')
    ap.add_argument('--ablate', action='store_true')
    ap.add_argument('--output', type=Path, default=Path('/tmp/tabgen-fingering-grid.json'))
    args = ap.parse_args()
    data = dataset()
    if not data:
        ap.error(f'No GuitarSet annotations in {ANNOT}')
    base = baseline_weights()
    rows = evaluate(data, base, args.diagnostic)
    print('Baseline', json.dumps(summary(rows)), flush=True)
    final_rows = evaluate(data, fb.FingeringWeights())
    print('Defaults', json.dumps(summary(final_rows)), flush=True)
    if args.benchmark:
        benchmark(data, fb.FingeringWeights())
    if args.ablate:
        for kw in ({'run_move': 0.}, {'rest_ease': 0.}, {'span': 0.}, {'top_k': 32}):
            scores = summary(evaluate(data, replace(fb.FingeringWeights(), **kw)))
            print('Ablation', kw, json.dumps(scores), flush=True)
    if args.grid:
        records = [{'weights': asdict(base), 'rows': rows}]
        if args.output.exists():
            cached = json.loads(args.output.read_text())
            if not cached or cached[0]['weights'] != asdict(base):
                ap.error('Cached grid has a different baseline; use another --output path')
            records = cached
        seen = {tuple(sorted(rec['weights'].items())) for rec in records}
        # 108 configurations; no score-dependent proposals or player features.
        configurations = product((0., 1., 2., 3., 4., 6.), (0., .5, 1.), (0., .3), (.15, .36, .6))
        for i, (run, rest, span, jump) in enumerate(configurations):
            w = replace(base, run_move=run, rest_ease=rest, span=span, string_jump=jump)
            key = tuple(sorted(asdict(w).items()))
            if key in seen:
                continue
            rows = evaluate(data, w)
            records.append({'weights': asdict(w), 'rows': rows})
            seen.add(key)
            args.output.write_text(json.dumps(records))
            print(i, json.dumps(asdict(w)), json.dumps(summary(rows)), flush=True)
    if args.cv:
        if not args.output.exists():
            ap.error('Run --grid first or provide its --output cache')
        cross_validate(json.loads(args.output.read_text()))


if __name__ == '__main__':
    main()
