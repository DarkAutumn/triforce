"""Stage 2 (round 40, Stage52): the team spec that plays one network in every slot of the generation, for
``--team`` (``triforce eval-batch``, ``eval-chain``) and ``TRIFORCE_TEAM=... python -m triforce.lab.bench run --team``.

    python scripts/experiments/r52_one_team.py gen1-one/im256               # current+@overworld=gen1-one/im256+...
    python scripts/experiments/r52_one_team.py gen1-one/im256 --base 2.0.0.15

One ``+@GROUP=RUN`` a teacher of ``triforce/groups.yaml`` (``store.group_slots``: every member family's
``family@route`` and ``family@route:late``), so every family of the route plays RUN on every step; the bare slots
(``d2first``) stay the base's. The groups cover every family the route's steps name (tests/test_lab_groups.py)."""

import argparse
import sys

from triforce.lab import groups


def one_team(run: str, base: str = 'current') -> str:
    """``BASE+@GROUP=RUN`` for every group, in the file's order."""
    return '+'.join([base, *(f'@{name}={run}' for name in groups.load())])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n', maxsplit=1)[0])
    parser.add_argument('run', help="the store run every slot plays (NAME/RUN)")
    parser.add_argument('--base', default='current', help="the team whose other slots stay (default current)")
    args = parser.parse_args(argv)
    print(one_team(args.run, args.base))
    return 0


if __name__ == '__main__':
    sys.exit(main())
