"""Round 36 (docs/experiments/round36.md "What the observation lacks", §2): gradient-boosted trees on the look-ahead's
verdicts, for attribution: which RAM bytes carry the verdict.

    python scripts/experiments/r36_hit_trees.py DATA.npz OUT.json --inputs ram|obs|obs+ram [--rounds 600]
        [--drop GROUP ...] [--device cuda]

One row per judged candidate on future 0 (the labels of ``r36_hit_oracle.py``), its features: the candidate's kind,
direction and duration; ``ram``: the kept RAM bytes (``r36_hit_common.ram_columns``) as numbers plus, per object slot
1-12, its X and Y offset from Link and their sum of magnitudes (signed: trees split on thresholds, not on
differences of two columns); ``obs``: the observation's vector blocks flattened (``entity_features``,
``entity_types``, ``link``, ``room_state``, ``history``, ``objective``, ``target``, ``room_id``; the tile grid left
out). Held out as round 35. Written: the held-out numbers (``r36_hit_common.summary``), the gain of every feature
and its sum by RAM group (``RAM_GROUPS``; the offsets count as ``relative``, the action as ``action``).
XGBoost on the GPU: no job queue.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r36_hit_common import group_of, load, ram_columns, ram_names, summary  # noqa: E402

OBS_KEYS = ['entity_features', 'entity_types', 'link', 'room_state', 'history', 'objective', 'target', 'room_id']


def features(data, rows, joints, inputs: str, columns: np.ndarray):
    names, groups, blocks = [], [], []
    k, dr, du = joints // 32, (joints // 4) % 8, joints % 4
    blocks.append(np.stack([k, dr, du], axis=1).astype(np.float32))
    names += ['action_kind', 'action_dir', 'action_dur']
    groups += ['action'] * 3
    if 'ram' in inputs:
        ram = data.d['ram'][rows]
        blocks.append(ram[:, columns].astype(np.float32))
        table = ram_names()
        names += [f'${a:03X} {table.get(int(a), "?")}' for a in columns]
        groups += [group_of(int(a)) for a in columns]
        lx, ly = ram[:, 0x70].astype(np.float32), ram[:, 0x84].astype(np.float32)
        live = ram[:, 0x350:0x35C] != 0
        dx = np.where(live, ram[:, 0x71:0x7D] - lx[:, None], np.nan)
        dy = np.where(live, ram[:, 0x85:0x91] - ly[:, None], np.nan)
        blocks += [dx, dy, np.abs(dx) + np.abs(dy)]
        for what in ('dx', 'dy', 'l1'):
            names += [f'{what}[{s}]' for s in range(1, 13)]
            groups += ['relative'] * 12
    if 'obs' in inputs:
        from triforce import _native  # pylint: disable=import-outside-toplevel
        feature_names = {'entity_features': _native.ENTITY_FEATURES, 'link': _native.LINK_FEATURES,
                         'room_state': _native.ROOM_STATE_FEATURES, 'history': _native.HISTORY_FEATURES,
                         'objective': _native.OBJECTIVE_FEATURES, 'target': _native.TARGET_FEATURES}
        for key in OBS_KEYS:
            v = data.obs[key][rows].reshape(len(rows), -1).astype(np.float32)
            blocks.append(v)
            if key == 'entity_features':
                fn = feature_names[key]
                names += [f'obs.entity[{i // len(fn)}].{fn[i % len(fn)]}' for i in range(v.shape[1])]
            elif key in feature_names and len(feature_names[key]) == v.shape[1]:
                names += [f'obs.{key}.{n}' for n in feature_names[key]]
            else:
                names += [f'obs.{key}[{i}]' for i in range(v.shape[1])]
            groups += [f'obs.{key}'] * v.shape[1]
    return np.concatenate(blocks, axis=1), names, groups


def main(argv) -> int:
    import xgboost as xgb  # pylint: disable=import-outside-toplevel
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('data', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--inputs', choices=['ram', 'obs', 'obs+ram'], default='ram')
    parser.add_argument('--rounds', type=int, default=600)
    parser.add_argument('--depth', type=int, default=8)
    parser.add_argument('--drop', nargs='*', default=[])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args(argv)
    data = load(args.data)
    held = data.held_rows[data.c_row]
    tr, te = np.flatnonzero(~held), np.flatnonzero(held)
    columns = ram_columns(data.d['ram'], data.c_row[tr]) if 'ram' in args.inputs else np.zeros(0, dtype=np.int64)
    x_tr, names, groups = features(data, data.c_row[tr], data.c_joint[tr], args.inputs, columns)
    x_te, _, _ = features(data, data.c_row[te], data.c_joint[te], args.inputs, columns)
    if args.drop:
        keep = ~np.isin(np.array(groups), args.drop)
        x_tr, x_te = x_tr[:, keep], x_te[:, keep]
        names = [n for n, k in zip(names, keep) if k]
        groups = [g for g, k in zip(groups, keep) if k]
    y_tr = data.c_lost[tr]
    print(f'{args.inputs}: {x_tr.shape} training, {x_te.shape} held out', flush=True)
    model = xgb.XGBClassifier(n_estimators=args.rounds, max_depth=args.depth, learning_rate=0.08, subsample=0.8,
                              colsample_bytree=0.6, tree_method='hist', device=args.device, eval_metric='logloss',
                              max_bin=256, n_jobs=args.threads)
    model.fit(x_tr, y_tr)
    scores = model.predict_proba(x_te)[:, 1]
    result = summary(data, scores, held)
    gain = model.get_booster().get_score(importance_type='total_gain')
    per = np.zeros(len(names))
    for f, g in gain.items():
        per[int(f[1:])] = g
    total = per.sum()
    by_group: dict[str, float] = {}
    for g, v in zip(groups, per):
        by_group[g] = by_group.get(g, 0.0) + v / total
    top = np.argsort(-per)[:40]
    out = {'inputs': args.inputs, 'drop': args.drop, 'summary': result,
           'by_group': dict(sorted(by_group.items(), key=lambda kv: -kv[1])),
           'top': [(names[i], groups[i], per[i] / total) for i in top]}
    print(json.dumps(out, indent=1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
