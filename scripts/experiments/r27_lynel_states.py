"""Round 27 (ow-lynel, docs/experiments/round27.md §15): 0:0x0a's second visit (out of the white sword's cave,
leaving S) at what the chain brings, as ow-lynel's harvested starts.

    python scripts/experiments/r27_lynel_states.py CHAIN.jsonl... --out DIR [--count N] [--seed S]

``eval-chain --harvest`` saves handoffs only, and the cave and the room after it are the same family's: the second
visit has no handoff. This writes ``--count`` states ``DIR/0_0ac_h<i>_0.state`` (a harvested name: stem, seed,
room index): the installed pre-entry state ``0_0ac_<i mod 3>`` (restored, not run: it stays pre-entry) with the
hearts and bombs of one second visit drawn from the chains (``health_in``, ``bombs_in`` of the 0x0a record after
the cave's). A harvested start keeps them (no ``ram_randomization``); the rest of the state is the route's.
"""

import argparse
import json
import os
import random
import sys

from triforce._native import Game, Snapshot
from triforce.paths import STATES_DIR

STEM = '0_0ac'


def second_visits(paths: list[str]) -> list[tuple[float, int]]:
    """(hearts in, bombs in) of every 0:0x0a entry right after its cave, over the chains' runs."""
    out = []
    for path in paths:
        with open(path, encoding='utf-8') as file:
            for line in file:
                rooms = json.loads(line)['rooms']
                for before, room in zip(rooms, rooms[1:]):
                    if (before['level'], before['room'], before['in_cave']) == (0, '0x0a', True) \
                            and (room['level'], room['room'], room['in_cave']) == (0, '0x0a', False):
                        out.append((float(room['health_in']), int(room['bombs_in'])))
    return out


def hearts_values(half_hearts: int, max_health: int) -> tuple[int, int]:
    """hearts_and_containers and partial_hearts for ``half_hearts`` (``env::start_states::hearts_encoding``)."""
    containers = (max_health - 1) & 0x0F
    if half_hearts >= 2 * max_health:
        return (containers << 4) | containers, 0xFF
    return (containers << 4) | (half_hearts // 2), 0x7F if half_hearts % 2 else 0


def write_states(chains: list[str], out: str, count: int, seed: int) -> int:
    """Writes the states; returns how many."""
    visits = second_visits(chains)
    if not visits:
        raise ValueError("no second visit of 0:0x0a in the chains")
    rng = random.Random(seed)
    game = Game(seed=0, states_dir=str(STATES_DIR))
    os.makedirs(out, exist_ok=True)
    for index in range(count):
        with open(os.path.join(str(STATES_DIR), f'{STEM}_{index % 3}.state'), 'rb') as file:
            game.restore(Snapshot.from_state_file(file.read()))
        hearts, bombs = rng.choice(visits)
        max_health = (game.read(0x66F) >> 4) + 1
        both, partial = hearts_values(max(1, round(2 * hearts)), max_health)
        game.set_value('hearts_and_containers', both)
        game.set_value('partial_hearts', partial)
        game.set_value('bombs', bombs)
        game.save_state_file(os.path.join(out, f'{STEM}_h{index}_0.state'))
    hearts = sorted(h for h, _ in visits)
    print(f"{out}: {count} states from {len(visits)} second visits (hearts in: mean "
          f"{sum(hearts) / len(hearts):.2f}, median {hearts[len(hearts) // 2]:.1f})")
    return count


def main(argv) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('chains', nargs='+', help="chain run jsonl files that play 0:0x0a and its cave")
    parser.add_argument('--out', required=True)
    parser.add_argument('--count', type=int, default=300)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)
    write_states(args.chains, args.out, args.count, args.seed)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
