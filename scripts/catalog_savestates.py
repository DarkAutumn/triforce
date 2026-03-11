#!/usr/bin/env python3
"""Generate PNG screenshots and YAML metadata for every savestate.

For each .state file in custom_integrations/Zelda-NES/, produces:
  - {name}.png         initial frame
  - {name}_spawned.png frame after 50 noop steps (enemies spawned)
  - {name}.yaml        room metadata (exits, enemies, items, treasure)
"""

import os
import sys
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests'))

from tests.utilities import ZeldaActionReplay  # noqa: E402
from triforce.zelda_enums import Direction      # noqa: E402

STATE_DIR = os.path.join(os.path.dirname(__file__), '..', 'triforce', 'custom_integrations', 'Zelda-NES')
SPAWN_STEPS = 50

DIRECTION_ORDER = [Direction.N, Direction.E, Direction.S, Direction.W]


def get_exits(state):
    """Return a dict of exit direction -> destination full location."""
    exits = {}
    for d in DIRECTION_ORDER:
        tiles = state.room.exits.get(d, [])
        if tiles:
            dest = state.full_location.get_location_in_direction(d)
            exits[d.name] = {"level": dest.level, "location": f"0x{dest.value:02x}"}
    return exits


def get_enemy_counts(state):
    """Return a dict of enemy type name -> count."""
    names = []
    for e in state.enemies:
        eid = e.id
        names.append(eid.name if hasattr(eid, 'name') else f"Unknown_{int(eid)}")
    counts = Counter(names)
    return dict(sorted(counts.items()))


def get_items(state):
    """Return item info."""
    items = []
    for item in state.items:
        items.append({"id": item.id, "tile": [item.tile.x, item.tile.y]})
    return items


def get_treasure(state):
    """Return treasure info if present."""
    if state.treasure is None:
        return None
    t = state.treasure
    return {"tile": [t.tile.x, t.tile.y]}


def save_frame(env, path):
    """Save the current rendered frame as a PNG."""
    frame = env.render()
    if frame is not None:
        Image.fromarray(frame).save(path)


def process_state(state_path):
    """Process a single savestate file."""
    name = Path(state_path).stem
    out_dir = os.path.dirname(state_path)

    try:
        replay = ZeldaActionReplay(name, render_mode='rgb_array')
    except Exception as e:
        print(f"  SKIP {name}: {e}")
        return

    try:
        state = replay._prev

        # Initial frame
        save_frame(replay.env, os.path.join(out_dir, f"{name}.png"))

        initial_enemies = get_enemy_counts(state)

        # Step forward with noop to spawn enemies
        raw = replay.env
        while hasattr(raw, 'env'):
            raw = raw.env
        noop = [0] * raw.action_space.shape[0]
        for _ in range(SPAWN_STEPS):
            try:
                raw.step(noop)
            except Exception:
                break

        # Spawned frame
        save_frame(replay.env, os.path.join(out_dir, f"{name}_spawned.png"))

        # Get spawned state via the wrapper chain's data lookup
        from triforce.zelda_game import ZeldaGame
        info = raw.data.lookup_all()
        info['total_frames'] = SPAWN_STEPS
        spawned_state = ZeldaGame(raw, info, SPAWN_STEPS)
        spawned_enemies = get_enemy_counts(spawned_state)

        # Merge enemy counts (take the max of each type seen)
        all_enemies = {}
        for k in set(list(initial_enemies.keys()) + list(spawned_enemies.keys())):
            all_enemies[k] = max(initial_enemies.get(k, 0), spawned_enemies.get(k, 0))
        all_enemies = dict(sorted(all_enemies.items()))

        # Build YAML
        meta = {
            "level": state.level,
            "location": f"0x{state.location:02x}",
            "in_cave": state.in_cave,
            "exits": get_exits(state),
            "enemies": all_enemies if all_enemies else None,
            "items": get_items(state) or None,
            "treasure": get_treasure(state),
            "link_tile": [state.link.tile.x, state.link.tile.y],
            "link_health": state.link.health,
            "keys": state.link.keys,
        }

        yaml_path = os.path.join(out_dir, f"{name}.yaml")
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(meta, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        enemy_str = ", ".join(f"{v}x {k}" for k, v in all_enemies.items()) if all_enemies else "none"
        print(f"  OK {name}: level={state.level} loc=0x{state.location:02x} enemies=[{enemy_str}]")

    except Exception as e:
        print(f"  ERROR {name}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        replay.env.close()


def main():
    state_files = sorted(Path(STATE_DIR).glob("*.state"))
    print(f"Processing {len(state_files)} savestates from {STATE_DIR}")

    for sf in state_files:
        process_state(str(sf))

    print("Done.")


if __name__ == "__main__":
    main()
