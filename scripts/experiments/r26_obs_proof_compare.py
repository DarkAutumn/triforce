"""Round 26 schema 6 proof: the converted models (1.5.1.0) against the originals (1.5.0.0), episode by episode.

    python scripts/experiments/r26_obs_proof_compare.py REF_DIR NEW_DIR [--markdown]

REF_DIR and NEW_DIR hold one ``<family>-<act>.json`` per evaluation (``triforce eval-batch --families all --acts
argmax,temperature:0.5+relax:16 --episodes-per-state 20 --seed 0 --output-dir DIR``, the old code with the original
runs and the new code with ``triforce convert-model --promoted`` copies). Each episode is matched by (start state,
seed) and compared whole: outcome, steps, frames, hearts and every other field of its record must be equal. The
evaluation's summary fields are compared too, but for the ones that name the model or the code (its path, sha1, hash,
version, git description) and for schema 6's new config key `obs.hidden_positions`, which the new code's config
carries and the old one's lacks: it must be `ram` (the old behaviour), else it counts as a difference. Prints one row
per family and act; exit 1 on any difference or a missing file.
"""

import argparse
import json
import sys
from pathlib import Path

# Summary fields that name the checkpoint or the code, not what the model did.
PROVENANCE = {'checkpoint', 'checkpoint_sha1', 'model_hash', 'version', 'code'}
NAMED = ('outcome', 'steps', 'frames', 'health', 'hearts')


def episodes(evaluation: dict) -> dict:
    out = {}
    for record in evaluation['records']:
        key = (record['start_state'], record['seed'])
        if key in out:
            raise ValueError(f'two records for {key}')
        out[key] = record
    return out


def first_difference(a, b, path=''):
    """The first path at which ``a`` and ``b`` differ (None if equal)."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f'{path}.{k} (only in {"new" if k not in a else "ref"})'
            found = first_difference(a[k], b[k], f'{path}.{k}')
            if found:
                return found
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f'{path} (length {len(a)} != {len(b)})'
        for i, (x, y) in enumerate(zip(a, b)):
            found = first_difference(x, y, f'{path}[{i}]')
            if found:
                return found
        return None
    return None if a == b and type(a) is type(b) else f'{path}: {a!r} != {b!r}'


def compare(ref: dict, new: dict) -> dict:
    """Counts of a family-act pair: episodes compared, identical, differing per named field, the summary's verdict."""
    ref_eps, new_eps = episodes(ref), episodes(new)
    row = {'episodes': len(ref_eps), 'identical': 0, 'missing': len(set(ref_eps) ^ set(new_eps)),
           'differ': {name: 0 for name in NAMED}, 'other': 0, 'first': None}
    for key in sorted(set(ref_eps) & set(new_eps)):
        a, b = ref_eps[key], new_eps[key]
        diff = first_difference(a, b)
        if diff is None:
            row['identical'] += 1
            continue
        row['first'] = row['first'] or f'{key[0]} seed {key[1]}: {diff}'
        named = [name for name in NAMED if a.get(name) != b.get(name)]
        for name in named:
            row['differ'][name] += 1
        row['other'] += not named
    summary_a = {k: v for k, v in ref.items() if k not in PROVENANCE | {'records'}}
    summary_b = {k: v for k, v in new.items() if k not in PROVENANCE | {'records'}}
    obs_a, obs_b = summary_a.get('config', {}).get('obs', {}), summary_b.get('config', {}).get('obs', {})
    if 'hidden_positions' not in obs_a and obs_b.get('hidden_positions') == 'ram':
        summary_b['config'] = {**summary_b['config'], 'obs': {k: v for k, v in obs_b.items() if k != 'hidden_positions'}}
    row['summary'] = first_difference(summary_a, summary_b)
    return row


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('ref', type=Path)
    parser.add_argument('new', type=Path)
    parser.add_argument('--markdown', action='store_true', help='a markdown table')
    args = parser.parse_args(argv)
    names = sorted(p.name for p in args.ref.glob('*.json'))
    if not names:
        parser.error(f'no evaluations in {args.ref}')
    rows, ok = [], True
    for name in names:
        ref = json.loads((args.ref / name).read_text())
        label = (ref['specialist'], ref['act'])
        new_path = args.new / name
        if not new_path.exists():
            rows.append((label, None))
            ok = False
            continue
        row = compare(ref, json.loads(new_path.read_text()))
        ok &= row['identical'] == row['episodes'] and not row['missing'] and row['summary'] is None
        rows.append((label, row))
    if args.markdown:
        print('| Family | Act | Episodes | Identical | Outcome / steps / frames / health / hearts differ | Summary |')
        print('|---|---|---:|---:|---|---|')
    for (family, act), row in rows:
        if row is None:
            print(f'| {family} | `{act}` | missing | | | |' if args.markdown else f'{family:20} {act:28} MISSING')
            continue
        differs = '/'.join(str(row['differ'][n]) for n in NAMED)
        summary = 'equal' if row['summary'] is None else f'differs: {row["summary"]}'
        if args.markdown:
            print(f'| {family} | `{act}` | {row["episodes"]} | {row["identical"]} | {differs} | {summary} |')
        else:
            print(f'{family:20} {act:28} {row["identical"]:4}/{row["episodes"]:<4} differ {differs} other '
                  f'{row["other"]} missing {row["missing"]} summary {summary}')
        if row['first']:
            print(f'    first difference: {row["first"]}')
    total = sum(r['episodes'] for _, r in rows if r)
    same = sum(r['identical'] for _, r in rows if r)
    print(f'\n{len(rows)} evaluations, {same} of {total} episodes identical: {"IDENTICAL" if ok else "DIFFERENT"}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
