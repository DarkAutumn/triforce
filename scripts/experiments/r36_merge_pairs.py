"""Round 36, "The wood-sword pairs": the bench manifest of the default runs' rooms.

    python scripts/experiments/r36_merge_pairs.py --out target/merge/pairs-late.json darknut goriya ...

The yardstick's entries of a paired family (``triforce/lab/bench.json``) are its harvested ``d3first_*`` states,
nearly all before the magic sword: the route part's rooms.  This writes a manifest in the bench's format for the
other half, the rooms the default run plays: the route's own seed start states (``d3first_*``) of the steps from
``own_models_until`` on that the family plays, or, for a family the route plays no more after the sword, the
family's training start states.  Both keys (``family`` and ``family@d3first``) get the same states, as in the
bench, so ``bench run --team`` plays both teachers on them; same episodes and seeds as the bench."""

import argparse
import json
import sys
from pathlib import Path

from triforce import _native
from triforce.lab.bench import ROUTE, load_manifest
from triforce.specialist import load_specialist


def late_states(family: str, states_dir: Path) -> tuple[list[str], str]:
    until = _native.own_models_until(ROUTE)
    wanted = {(s['level'], s['room'], bool(s['in_cave']), s['entry'], s['visit'])
              for s in _native.route_steps(ROUTE) if s['specialist'] == family and s['index'] >= until}
    names = []
    for path in sorted(states_dir.glob(f'{ROUTE}_*.state')):
        parsed = _native.state_name(path.stem)
        if parsed is not None and tuple(parsed) in wanted:
            names.append(path.stem)
    if names:
        return names, 'route-late'
    config = load_specialist(family)
    return sorted(Path(p).stem for p in config.resolve_start_states(str(states_dir))), 'seed'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('families', nargs='+')
    parser.add_argument('--episodes', type=int, default=10, help="episodes a state (the bench: 5)")
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    states_dir = Path(_native.paths()['states_dir'])
    base = load_manifest()
    entries = []
    for family in args.families:
        states, source = late_states(family, states_dir)
        for key in (family, f'{family}@{ROUTE}'):
            entries.append({'key': key, 'family': family, 'source': 'seed', 'origin': source, 'states': states})
        print(f"{family}: {len(states)} states ({source}): {states}")
    manifest = {k: v for k, v in base.items() if k not in ('entries', 'missing_coverage', 'full')}
    manifest['entries'] = entries
    manifest['episodes'] = args.episodes
    Path(args.out).write_text(json.dumps(manifest, indent=1), encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())
