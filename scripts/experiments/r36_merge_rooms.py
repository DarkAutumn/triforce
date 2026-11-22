"""Round 36, "The fights": the room-to-head table of a ``roomheads`` student (``triforce/nets/roomheads.py``), from the
recorded data.

    python scripts/experiments/r36_merge_rooms.py --cache target/merge/cache --out target/merge/boss-heads.json

A head a boss family (``boss-*``, ganon, patra, lanmola, moldorm; both keys of a family share it): the rooms
(``obs['room_id']``) where at least ``--purity`` of the rows are that family's and that hold at least ``--share`` of
its rows.  Every other room is the shared head's.  Prints the table with each room's rows."""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

BOSSES = ('ganon', 'patra', 'lanmola', 'moldorm')


def family(key: str) -> str:
    return key.split('@')[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n', maxsplit=1)[0])
    parser.add_argument('--cache', default='target/merge/cache')
    parser.add_argument('--purity', type=float, default=0.95)
    parser.add_argument('--share', type=float, default=0.02)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    cache = Path(args.cache)
    keys = json.loads((cache / 'cache.json').read_text())['keys']
    key = np.load(cache / 'key.npy')
    room = np.asarray(np.load(cache / 'obs_room_id.npy', mmap_mode='r')).reshape(-1)
    fam = np.array([family(k) for k in keys])[key]
    per_room = collections.Counter(room.tolist())
    table = {}
    for name in sorted({family(k) for k in keys}):
        if not (name.startswith('boss-') or name in BOSSES):
            continue
        mine = room[fam == name]
        counts = collections.Counter(mine.tolist())
        rooms = sorted(r for r, n in counts.items()
                       if n >= args.share * len(mine) and n >= args.purity * per_room[r])
        table[name] = rooms
        print(f"{name}: " + ", ".join(f"{r} ({counts[r]} of {per_room[r]})" for r in rooms), flush=True)
    Path(args.out).write_text(json.dumps(table, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
