"""Hearts Link arrives with in each room during chain runs, for training at chain-like health.

    python -m triforce.chain_hp models/*chain*.jsonl [--out triforce/chain_hp.json]

Chain runs (``triforce eval-chain`` / ``scripts/train_round.py``) record ``health_in`` for every room visit.  This
collects the arrivals per room (``"<level>:0x<room>[:cave]"``, ``env::config::room_key``) into a JSON profile; a
run's first room is its start state (the game start or ``--from``), not an arrival, so it is left out.  A
specialist with ``ram_randomization.chain_hp: p`` draws its reset health from its start room's profile in a share
``p`` of resets (the native ``env::start_states::ram_randomization_values``, which loads ``triforce/chain_hp.json``;
a config with ``chain_hp`` and no room profiled with ``MIN_CHAIN_SAMPLES`` arrivals is refused); the other resets
keep the ``hp`` / ``full_hp`` draw, so training still sees full health (sword beams) and the whole range.  No
profile ships: build one from current-route chains started at the game start before turning ``chain_hp`` on.
"""

import argparse
import json
import os
from typing import Dict, Iterable, List

from .events import write_json_atomic

PROFILE_PATH = os.path.join(os.path.dirname(__file__), 'chain_hp.json')


def build_profile(paths: Iterable[str]) -> Dict[str, List[float]]:
    """room key -> sorted list of the hearts Link entered it with, over every run in the chain jsonl files (each
    run's first room, its start state, excluded)."""
    profile: Dict[str, List[float]] = {}
    for path in paths:
        with open(path, encoding='utf-8') as file:
            for line in file:
                if not line.strip():
                    continue
                for room in json.loads(line)['rooms'][1:]:
                    key = f"{room['level']}:{room['room']}" + (':cave' if room['in_cave'] else '')
                    profile.setdefault(key, []).append(float(room['health_in']))
    return {key: sorted(values) for key, values in sorted(profile.items())}


def main(argv=None):
    """Command line."""
    parser = argparse.ArgumentParser(prog='python -m triforce.chain_hp', description=__doc__.split('\n\n')[0])
    parser.add_argument('chains', nargs='+', help="chain run jsonl files")
    parser.add_argument('--out', default=PROFILE_PATH)
    args = parser.parse_args(argv)
    profile = build_profile(args.chains)
    # Atomically: every training, eval and chain process parses this file at startup.
    write_json_atomic(args.out, profile, indent=1)
    for key, values in profile.items():
        print(f"{key:12} {len(values):5} visits  mean {sum(values) / len(values):.2f} hearts")
    return 0


if __name__ == '__main__':
    main()
