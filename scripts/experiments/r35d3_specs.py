"""Round 35's Dungeon 3 configs (docs/experiments/round35.md "Dungeon 3"): the evaluation spec of a family's room on
harvested entries, the training yaml of a d3first continuation, its demonstrations' yaml, and the harvested dir it
draws from.

    python scripts/experiments/r35d3_specs.py eval FAMILY ROOM OUT_DIR N ENTRY_DIR [--seed 0]
        N entries drawn from ENTRY_DIR/*.state into OUT_DIR/states (symlinks beside every seed state: `evaluate`
        resolves start states by stem from TRIFORCE_STATES_DIR) and OUT_DIR/spec.yaml: the family's config on ROOM
        (e.g. 3:0x4d) alone, those entries as its start states, no RAM randomization, no curriculum.
    python scripts/experiments/r35d3_specs.py harvested OUT_DIR SRC[xK] [SRC[xK] ...] [--keep-other FAMILY]
        OUT_DIR filled with symlinks of every SRC/*.state, K copies each (copy i renamed h<i*100000+seed>, still
        valid harvested names); --keep-other adds the family's default harvested states of other levels than 3
        (triforce/states/harvested/FAMILY/d3first_[!3]_*: Dungeons 1, 4, 5 keep their practice).
    python scripts/experiments/r35d3_specs.py train FAMILY OUT.yaml HARVESTED_DIR FRACTION [--potential P] [--demos]
        The family's config with state_routes [d3first], harvested_states HARVESTED_DIR at FRACTION and, with
        --potential, reward.heart_value.potential P (the price of a hit in the run's odds); --demos: the same with no
        RAM randomization and no curriculum (the planner's demonstrations keep the entries' hearts and bombs).

Configs are written as JSON (a YAML subset)."""

import argparse
import json
import random
import re
from pathlib import Path

from triforce.specialist import load_specialist

STATES = Path(__file__).resolve().parents[2] / 'triforce' / 'states'
NO_RAM = {'hp': None, 'full_hp': 0.0, 'chain_hp': None, 'bombs': None, 'bombs_share': 0.0, 'keys': None,
          'help_count': None, 'room_kills': {}, 'room_kills_share': 0.0, 'drop_bombs': None}


def config(family):
    d = load_specialist(family).to_dict()
    d['state_routes'] = ['d3first']
    return d


def cmd_eval(a):
    out = Path(a.out)
    states = out / 'states'
    states.mkdir(parents=True, exist_ok=True)
    for s in STATES.glob('*.state'):
        link = states / s.name
        if not link.exists():
            link.symlink_to(s)
    pool = sorted(Path(a.entries).glob('*.state'))
    chosen = sorted(random.Random(a.seed).sample(pool, min(a.n, len(pool))))
    for s in chosen:
        link = states / s.name
        if not link.exists():
            link.symlink_to(s.resolve())
    d = config(a.family)
    level, room = a.room.split(':')
    d['rooms'] = [f'({level}, {room})']
    d['start_states'] = [s.stem for s in chosen]
    d['ram_randomization'] = dict(NO_RAM)
    d['curriculum']['enabled'] = False
    d['harvested_fraction'] = 0.0
    d['harvested_states'] = None
    d['notes'] = f'r35d3 evaluation of {a.room} on {len(chosen)} entries of {a.entries}'
    (out / 'spec.yaml').write_text(json.dumps(d, indent=1))
    print(out / 'spec.yaml', len(chosen), 'entries')


def cmd_harvested(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for spec in a.srcs:
        m = re.fullmatch(r'(.+?)(?:x(\d+))?', spec)
        src, k = m.group(1), int(m.group(2) or 1)
        for s in sorted(Path(src).glob('*.state')):
            for i in range(k):
                name = s.name if i == 0 else re.sub(r'_h(\d+)_', lambda m: f'_h{i * 100000 + int(m.group(1))}_', s.name)
                link = out / name
                if not link.exists():
                    link.symlink_to(s.resolve())
                    n += 1
    if a.keep_other:
        for s in sorted((STATES / 'harvested' / a.keep_other).glob('d3first_*.state')):
            if not s.name.startswith('d3first_3_') and not (out / s.name).exists():
                (out / s.name).symlink_to(s.resolve())
                n += 1
    print(out, n, 'states')


def cmd_train(a):
    d = config(a.family)
    d['harvested_states'] = str(Path(a.harvested).resolve())
    d['harvested_fraction'] = a.fraction
    if a.potential is not None:
        d['reward']['heart_value']['potential'] = a.potential
    if a.demos:
        d['ram_randomization'] = dict(NO_RAM)
        d['curriculum']['enabled'] = False
    Path(a.out).write_text(json.dumps(d, indent=1))
    print(a.out)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('eval')
    e.add_argument('family')
    e.add_argument('room')
    e.add_argument('out')
    e.add_argument('n', type=int)
    e.add_argument('entries')
    e.add_argument('--seed', type=int, default=0)
    h = sub.add_parser('harvested')
    h.add_argument('out')
    h.add_argument('srcs', nargs='+')
    h.add_argument('--keep-other')
    t = sub.add_parser('train')
    t.add_argument('family')
    t.add_argument('out')
    t.add_argument('harvested')
    t.add_argument('fraction', type=float)
    t.add_argument('--potential', type=float)
    t.add_argument('--demos', action='store_true')
    a = p.parse_args()
    {'eval': cmd_eval, 'harvested': cmd_harvested, 'train': cmd_train}[a.cmd](a)


if __name__ == '__main__':
    main()
