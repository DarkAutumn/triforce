"""Round 29: do two data changes of the vocabulary step change what `d2first`'s rooms show a model?
(docs/experiments/round29.md §4, beside the identity chain.)

    python scripts/experiments/r29_data_ab.py root OUT [--control]
    python scripts/experiments/r29_data_ab.py run FILE.json                       # in the checkout
    cd OUT && TRIFORCE_STATES_DIR=<checkout>/triforce/states PYTHONPATH=OUT \
        python <checkout>/scripts/experiments/r29_data_ab.py run FILE.json        # in a root
    python scripts/experiments/r29_data_ab.py compare A.json B.json

The step changed two things in data that `d2first` rooms read: game.yaml's `path_costs` row in 0:0x17 (step 41
crosses that room) and `candle: true` in ow-zora's yaml.  `root OUT` makes a throwaway data root that is the checkout
without those two (the files linked, game.yaml and ow-zora.yaml rewritten); `--control` instead moves 0x17's cost
onto the rows Link walks there, a change that must show.

`run` plays, in whatever data root it is started in, every installed `d2first` start state of ow-boulder and ow-zora
(chains 0-2) through the single-room env from seeds 0-3: up to 150 decisions, each drawn by a seeded generator from
the actions the mask offers, and hashes every observation (the action mask is one of its arrays), every action,
reward and ending of the rollout.  Two roots that give the same hashes showed the same inputs and offered the same
actions at every decision of the same play; a difference in a mask or an input makes the plays part and the hashes
differ.  It needs no model and no place in the job line."""

import argparse
import collections
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

FAMILIES = ('ow-boulder', 'ow-zora')
COST = "  path_costs: [{cols: [14, 31], rows: [9, 9], cost: 8}]\n"
CONTROL = "  path_costs: [{cols: [14, 31], rows: [10, 13], cost: 8}]\n"
CANDLE = ("items: {bombs: tenth, candle: true}", "items: {bombs: tenth}")
ACTIONS = [f'MOVE {d} {n}' for d in 'NSWE' for n in (4, 8, 16)] + [f'WAIT {n}' for n in (4, 8)] + \
          [f'{k} {d}' for k in ('SWORD', 'BEAMS', 'BOMB', 'BOOMERANG', 'ARROW', 'CANDLE', 'CLIP') for d in 'NSWE'] + \
          ['RECORDER']


def root(args):
    here = Path(subprocess.run(['git', 'rev-parse', '--show-toplevel'], capture_output=True, text=True, check=True,
                               cwd=Path(__file__).parent).stdout.strip())
    out = args.path.resolve()
    (out / 'triforce').mkdir(parents=True)
    (out / 'specialists').mkdir()
    for f in (here / 'triforce').iterdir():
        if f.name != 'game.yaml':
            (out / 'triforce' / f.name).symlink_to(f)
    for name in ('roms', 'models', 'trajectories', 'scripts', 'zelda-asm'):
        if (here / name).exists():
            (out / name).symlink_to(here / name)
    game = (here / 'triforce' / 'game.yaml').read_text()
    if game.count(COST) != 1:
        raise SystemExit("game.yaml: 0:0x17's path_costs row is not as this script knows it")
    (out / 'triforce' / 'game.yaml').write_text(game.replace(COST, CONTROL if args.control else ''))
    for f in (here / 'specialists').glob('*.yaml'):
        text = f.read_text()
        if f.name == 'ow-zora.yaml' and not args.control:
            if CANDLE[0] not in text:
                raise SystemExit("ow-zora.yaml: its items line is not as this script knows it")
            text = text.replace(*CANDLE)
        (out / 'specialists' / f.name).write_text(text)
    print(f'{out}: ' + ("0x17's cost on the rows Link walks (the control)" if args.control else
                        "without 0x17's path cost and ow-zora's candle"))


def run(args):
    import numpy as np                                          # pylint: disable=import-outside-toplevel
    from triforce import _native                                # pylint: disable=import-outside-toplevel
    from triforce._native import Env                            # pylint: disable=import-outside-toplevel
    from triforce.demos import action_indices, legal            # pylint: disable=import-outside-toplevel
    from triforce.entry_states import state_stem                # pylint: disable=import-outside-toplevel
    states = Path(os.environ.get('TRIFORCE_STATES_DIR') or Path(_native.__file__).parent / 'states')
    stems = collections.defaultdict(list)
    for s in _native.route_steps('d2first'):
        stem = state_stem(s['level'], s['room'], s['in_cave'], s['entry'], s['visit'])
        if s['specialist'] in FAMILIES and stem not in stems[s['specialist']]:
            stems[s['specialist']].append(stem)
    rows = []
    for family in FAMILIES:
        env = Env(family, explicit_starts=True, strict_actions=True, seed=0)
        for stem in stems[family]:
            for chain in range(3):
                state = f'{stem}_{chain}'
                if not (states / f'{state}.state').exists():
                    continue
                for seed in range(4):
                    rng = random.Random(seed * 7919 + 13)
                    obs, _ = env.reset(seed=seed, options={'start_state': state})
                    digest, n = hashlib.sha1(), 0
                    for n in range(1, 151):
                        for key in sorted(obs):
                            digest.update(key.encode())
                            digest.update(np.ascontiguousarray(obs[key]).tobytes())
                        mask = obs['action_mask']
                        action = rng.choice([a for a in ACTIONS if legal(mask, action_indices(a))])
                        obs, reward, terminated, truncated, info = env.step(action)
                        digest.update(f'{action} {round(float(reward), 9)!r} {info.get("ending")}'.encode())
                        if terminated or truncated:
                            break
                    rows.append((family, state, seed, n, digest.hexdigest()))
    Path(args.path).write_text(json.dumps(rows))
    print(f'{len(rows)} rollouts, {sum(r[3] for r in rows)} decisions, states: '
          + ', '.join(f'{f} {len(s)}' for f, s in stems.items()))


def compare(args):
    a, b = (json.loads(Path(p).read_text()) for p in (args.path, args.other))
    if [r[:3] for r in a] != [r[:3] for r in b]:
        print('the two files do not hold the same rollouts')
        return 2
    differing = [tuple(x[:3]) for x, y in zip(a, b) if x != y]
    per = collections.Counter(d[0] for d in differing)
    print(f'{len(a)} rollouts, {sum(r[3] for r in a)} decisions: {len(differing)} differ'
          + (f' ({dict(per)}; states {sorted({d[1] for d in differing})})' if differing else ''))
    return 1 if differing else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('what', choices=['root', 'run', 'compare'])
    parser.add_argument('path', type=Path)
    parser.add_argument('other', nargs='?', type=Path)
    parser.add_argument('--control', action='store_true')
    args = parser.parse_args()
    return {'root': root, 'run': run, 'compare': compare}[args.what](args) or 0


if __name__ == '__main__':
    sys.exit(main())
