"""Olive Garden entries (5:0x05 from E, d3first step 103) at the hearts and bombs a chain brings there.

No chain handoff exists into the Garden (darknut plays 0x06 before it, so `eval-chain --harvest` saves none): each
entry is the route's seed state `d3first_5_05e_<seed % 3>` loaded with the game's own randomness for its seed, Link's
hearts one chain arrival's (`health_in` of the chain's 0x05 step-103 visits, evenly through them in sorted order)
and the bombs that arrival held (`bombs_in + route_bombs_in`: d3first grants the bag at Dungeon 5's entrance and the
Garden's reserve is 0, so every chain enters with 8).  The room is fresh on entry: the Darknuts spawn from the
seed's randomness, as in the chain.

    python scripts/experiments/r34d5_garden_states.py CHAIN.jsonl OUT_DIR N [--seed0 S]"""
import argparse
import json
import sys
from pathlib import Path

from triforce._native import Game
from triforce.paths import STATES_DIR

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r27_walks import hearts_bytes  # noqa: E402  pylint: disable=wrong-import-position

BOMBS = 0x658


def arrivals(chain: Path) -> list:
    """(hearts, bombs) of every 0x05 step-103 visit in the chain's runs, sorted."""
    out = []
    for line in chain.read_text().splitlines():
        for room in json.loads(line)['rooms']:
            if room['level'] == 5 and room['room'] == '0x05' and room.get('step') == 103:
                out.append((room['health_in'], room['bombs_in'] + room['route_bombs_in']))
    return sorted(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('chain', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('n', type=int)
    parser.add_argument('--seed0', type=int, default=8000)
    args = parser.parse_args()
    found = arrivals(args.chain)
    args.out.mkdir(parents=True, exist_ok=True)
    game = Game(seed=0, states_dir=str(STATES_DIR))
    for i in range(args.n):
        seed = args.seed0 + i
        view = game.load(f'd3first_5_05e_{seed % 3}', randomize_rng=True, seed=seed)
        h, bombs = found[i * len(found) // args.n]
        h = max(0.5, round(h * 2) / 2)
        for key, value in hearts_bytes(h, view.max_health).items():
            game.set_value(key, value)
        game.write(BOMBS, min(8, bombs))
        game.save_state_file(str(args.out / f'd3first_5_05e_h{seed}_0.state'))
    print(f"{args.n} Garden entries in {args.out} from {len(found)} arrivals (hearts {found[0][0]}-{found[-1][0]})")


if __name__ == '__main__':
    main()
