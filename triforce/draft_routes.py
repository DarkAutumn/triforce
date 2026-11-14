"""Draft routes (``triforce/draft_routes.yaml``): ways through the game that are planned, not played.

A draft is drawn by ``zelda-map.py --route NAME`` so it can be looked at and corrected before any of it is built.
Nothing else reads it: the objective, the chain runner and training follow ``routes.yaml`` alone, and a draft's rooms
need no ``specialist:`` tag, no start states and no model.  When a stretch of a draft is built for real its steps
move into ``routes.yaml``.

A draft is ``{extends: <routes.yaml route>, steps: [...]}``: that route's steps, then these, numbered on.  A step
has ``level``, ``room`` (quoted hex), ``in_cave``, ``entry`` and ``exit`` as in routes.yaml, and:

- ``exit: recorder`` / ``entry: recorder``: the whirlwind carries Link to a dungeon's entrance room
  (asm:Whirlwind_PrevRoomIds; zelda-asm/items.md "On the overworld, the whirlwind");
- ``exit: end``: the route ends in this room;
- ``wrap: true``: the room is crossed by a screen wrap (``CLIP E`` / ``CLIP W``).  It is required on a step that
  leaves through one of game.yaml's wrap exits (a W exit with ``wrap: true``: 0x51 into 0x50, 0x60 into 0x5f), which
  nothing but the wrap gets out of;
- ``takes: [heart-container]``: what Link takes in the room's overworld cave (``TAKES``; a draft has no cave steps,
  and game.yaml's ``treasure`` covers only what lies in the room itself), so the map can count hearts and items;
- ``note``: what Link does here, shown on the map; ``leg``: a name for the part of the route this step starts.

``problems`` checks a draft against game.yaml: every room exists, every exit is one of the room's and leads into the
next step's room from the side that step says.
"""

from pathlib import Path
from typing import Optional

import yaml

from triforce._native import game_rooms, route_steps

DRAFTS_PATH = Path(__file__).with_name('draft_routes.yaml')
ENTRIES = ('start', 'N', 'S', 'E', 'W', 'cave', 'recorder')
EXITS = ('N', 'S', 'E', 'W', 'cave', 'triforce', 'recorder', 'end')
STEP_KEYS = {'level', 'room', 'in_cave', 'entry', 'exit', 'wrap', 'note', 'leg', 'takes'}
TAKES = ('heart-container', 'magic-sword', 'wood-arrow', 'food')      # what an overworld cave on a draft gives
OPPOSITE = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}
WHIRLWIND_ROOMS = (0x37, 0x3C, 0x74, 0x45, 0x0B, 0x22, 0x42, 0x6D)     # where it sets Link down, levels 1-8


class DraftError(ValueError):
    """A draft that can't be read (a bad file; route problems are ``problems``)."""


def read(path: Path = DRAFTS_PATH) -> dict:
    """The file's drafts by name ({} when there is no file)."""
    if not Path(path).is_file():
        return {}
    data = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(data, dict):
        raise DraftError(f'{path}: not a mapping of draft routes')
    return data


def names(path: Path = DRAFTS_PATH) -> list[str]:
    return list(read(path))


def steps(name: str, path: Path = DRAFTS_PATH) -> list[dict]:
    """The draft's steps as ``route_steps`` dicts: the route it extends, then its own, each also with ``note``,
    ``leg``, ``wrap``, ``takes`` and ``draft`` (False for the extended route's steps)."""
    drafts = read(path)
    if name not in drafts:
        raise DraftError(f'{path}: no draft {name!r} (it has {list(drafts)})')
    draft = drafts[name]
    if not isinstance(draft, dict) or set(draft) - {'extends', 'steps'} or not isinstance(draft.get('steps'), list):
        raise DraftError(f'{path}: {name}: needs `steps:` (a list) and may have `extends:`, nothing else')
    plan = [{**step, 'note': None, 'leg': None, 'wrap': False, 'takes': [], 'draft': False}
            for step in (route_steps(draft['extends']) if draft.get('extends') else [])]
    for number, step in enumerate(draft['steps']):
        what = f'{path}: {name}: step {number} {step!r}'
        if not isinstance(step, dict) or set(step) - STEP_KEYS:
            raise DraftError(f'{what}: the keys are {sorted(STEP_KEYS)}')
        takes = step.get('takes', [])
        if step.get('entry') not in ENTRIES or step.get('exit') not in EXITS:
            raise DraftError(f'{what}: entry is one of {ENTRIES}, exit one of {EXITS}')
        if not isinstance(takes, list) or set(takes) - set(TAKES):
            raise DraftError(f'{what}: takes is a list of {TAKES}')
        try:
            room = int(str(step['room']), 16)
        except (KeyError, ValueError) as error:
            raise DraftError(f"{what}: room is a quoted hex id like '0x7f'") from error
        plan.append({'index': len(plan), 'level': int(step['level']), 'room': room,
                     'in_cave': bool(step.get('in_cave', False)), 'entry': step['entry'], 'exit': step['exit'],
                     'specialist': None, 'waypoint': None, 'retreat': None, 'treasure': None, 'bomb_reserve': None,
                     'way': None, 'stretch': None, 'note': step.get('note'), 'leg': step.get('leg'),
                     'wrap': bool(step.get('wrap', False)), 'takes': list(takes), 'draft': True})
    return plan


def key(step: dict) -> tuple[int, int, bool]:
    return step['level'], step['room'], step['in_cave']


def describe(step: dict) -> str:
    return f"#{step['index']} {step['level']}:0x{step['room']:02x}{' (cave)' if step['in_cave'] else ''}"


def level_entrances(rooms: dict) -> dict[int, int]:
    """Level -> the overworld room whose cave is its entrance."""
    return {info['cave'][0]: room for (level, room, in_cave), info in rooms.items()
            if level == 0 and not in_cave and info['cave'] and info['cave'][0] != 0}


def problems(plan: list[dict], rooms: Optional[dict] = None) -> list[str]:
    """What is wrong with a draft's own steps (``steps``) against game.yaml, one line each; [] when it holds."""
    rooms = game_rooms() if rooms is None else rooms
    entrances = level_entrances(rooms)
    found = []
    for step, after in zip(plan, plan[1:] + [None]):
        if not step['draft']:
            if after is None or not after['draft']:
                continue
        at, info = describe(step), rooms.get(key(step))
        if info is None:
            found.append(f'{at}: game.yaml has no such room')
            continue
        exit_, exits = step['exit'], info['exits']
        if exit_ == 'end':
            if after is not None:
                found.append(f'{at}: `exit: end`, but steps follow')
            continue
        if after is None:
            if step['draft'] and exit_ != 'triforce':
                found.append(f'{at}: the last step leaves by {exit_} (a route ends on `triforce` or `end`)')
            continue
        target, entry = None, None
        if exit_ == 'triforce':
            if info['treasure'] != 'triforce':
                found.append(f"{at}: exit triforce, but the room's treasure is {info['treasure']}")
            target, entry = (0, entrances.get(step['level']), False), 'cave'
        elif exit_ == 'recorder':
            if step['level'] != 0 or step['in_cave']:
                found.append(f'{at}: the recorder calls the whirlwind only on the overworld')
            if key(after) not in [(0, room, False) for room in WHIRLWIND_ROOMS]:
                found.append(f'{at}: the whirlwind sets Link down at a dungeon entrance, not in {describe(after)}')
            entry = 'recorder'
        elif exit_ in exits:
            if exits[exit_]['wrap'] and not step['wrap']:
                found.append(f'{at}: {exit_} is a wrap exit (game.yaml): the step needs `wrap: true`')
                continue
            target = tuple(exits[exit_]['to'])
            entry = 'S' if exit_ == 'cave' else 'cave' if step['in_cave'] else OPPOSITE[exit_]
        else:
            found.append(f'{at}: {exit_} is not an exit of the room (game.yaml has {list(exits)})')
            continue
        if target is not None and key(after) != target:
            found.append(f"{at}: {exit_} leads to {target[0]}:0x{target[1]:02x}{' (cave)' if target[2] else ''}, "
                         f'the next step is {describe(after)}')
        elif after['draft'] and after['entry'] != entry:
            found.append(f"{describe(after)}: entered from {entry} (the step before leaves by {exit_}), "
                         f"not {after['entry']}")
    return found
