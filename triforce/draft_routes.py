"""Draft routes (``triforce/draft_routes.yaml``): ways through the game that are planned, not played.

A draft is drawn by ``zelda-map.py --route NAME`` so it can be looked at and corrected before any of it is built.
Nothing else reads it: the objective, the chain runner and training follow ``routes.yaml`` alone, and a draft's rooms
need no ``specialist:`` tag, no start states and no model.  When a part of a draft is built for real its steps move
into ``routes.yaml``.

A draft is ``{extends: <routes.yaml route>, steps: [...]}``: that route's steps, then these, numbered on.  A step
has ``level``, ``room`` (quoted hex), ``in_cave``, ``entry`` and ``exit`` as in routes.yaml, and:

- ``exit: recorder`` / ``entry: recorder``: the whirlwind carries Link to a dungeon's entrance room
  (asm:Whirlwind_PrevRoomIds; zelda-asm/items.md "On the overworld, the whirlwind");
- ``exit: maze``: out of a maze room by its sequence (game.yaml ``maze:``: 0x61's N W S W into 0x60, 0x1b's N N N
  N into 0x0b), into the room beside it;
- ``exit: end``: the route ends in this room;
- ``wrap: true``: the room is crossed by a screen wrap (``CLIP E`` / ``CLIP W``).  It is required on a step that
  leaves through one of game.yaml's wrap exits (a W exit with ``wrap: true``: 0x51 into 0x50, 0x60 into 0x5f), which
  nothing but the wrap gets out of;
- ``takes: [heart-container]``: what Link takes in the room's overworld cave (``TAKES``; a draft has no cave steps,
  and game.yaml's ``treasure`` covers only what lies in the room itself), so the map can count hearts and items;
- ``note``: what Link does here, shown on the map; ``leg``: a name for the part of the route this step starts.

A list element may also be a ``stretch:`` block as in routes.yaml (round 22): ``{stretch: NAME, parts: {level, room,
entry}, rejoins: {level, room, entry}, clips: [steps], plain: [steps]}``, the way with the run-time clips switch on
and the way with it off; each way's steps get ``way`` and ``stretch``, the clip way's numbered first.

``problems`` checks a draft against game.yaml along each way: every room exists, every exit is one of the room's and
leads into the next step's room from the side that step says, and a stretch parts and rejoins where it says.
"""

from pathlib import Path
from typing import Optional

import yaml

from triforce._native import game_rooms, route_steps

DRAFTS_PATH = Path(__file__).with_name('draft_routes.yaml')
ENTRIES = ('start', 'N', 'S', 'E', 'W', 'cave', 'recorder')
EXITS = ('N', 'S', 'E', 'W', 'cave', 'triforce', 'recorder', 'maze', 'end')
STEP_KEYS = {'level', 'room', 'in_cave', 'entry', 'exit', 'wrap', 'note', 'leg', 'takes'}
STRETCH_KEYS = {'stretch', 'parts', 'rejoins', 'clips', 'plain', 'note'}
WAYS = (('clips', True), ('plain', False))           # a stretch's ways and the clips switch that plays each
TAKES = ('heart-container', 'magic-sword', 'wood-arrow', 'food')      # what an overworld cave on a draft gives
OPPOSITE = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}
SIDE_OF = {-16: 'N', 16: 'S', -1: 'W', 1: 'E'}       # the room id a side leads to - the room's
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


def room_of(value, what: str) -> int:
    try:
        return int(str(value), 16)
    except ValueError as error:
        raise DraftError(f"{what}: room is a quoted hex id like '0x7f'") from error


def parse_step(step, what: str, index: int, way: Optional[str] = None, stretch: Optional[str] = None) -> dict:
    if not isinstance(step, dict) or set(step) - STEP_KEYS:
        raise DraftError(f'{what}: the keys are {sorted(STEP_KEYS)}')
    takes = step.get('takes', [])
    if step.get('entry') not in ENTRIES or step.get('exit') not in EXITS:
        raise DraftError(f'{what}: entry is one of {ENTRIES}, exit one of {EXITS}')
    if not isinstance(takes, list) or set(takes) - set(TAKES):
        raise DraftError(f'{what}: takes is a list of {TAKES}')
    if 'room' not in step or 'level' not in step:
        raise DraftError(f'{what}: needs level and room')
    return {'index': index, 'level': int(step['level']), 'room': room_of(step['room'], what),
            'in_cave': bool(step.get('in_cave', False)), 'entry': step['entry'], 'exit': step['exit'],
            'specialist': None, 'waypoint': None, 'retreat': None, 'treasure': None, 'bomb_reserve': None,
            'way': way, 'stretch': stretch, 'note': step.get('note'), 'leg': step.get('leg'),
            'wrap': bool(step.get('wrap', False)), 'takes': list(takes), 'draft': True}


def room_entry(value, what: str) -> tuple[int, int, str]:
    """A stretch's ``parts`` / ``rejoins``: (level, room, entry)."""
    if not isinstance(value, dict) or set(value) != {'level', 'room', 'entry'}:
        raise DraftError(f'{what}: needs {{level, room, entry}}')
    return int(value['level']), room_of(value['room'], what), value['entry']


def steps(name: str, path: Path = DRAFTS_PATH) -> list[dict]:
    """The draft's steps as ``route_steps`` dicts: the route it extends, then its own (both ways of a stretch, the
    clip way's first), each also with ``note``, ``leg``, ``wrap``, ``takes`` and ``draft`` (False for the extended
    route's steps); a stretch's steps carry ``way`` and ``stretch`` (and the first of each way ``parts``, the
    stretch's (level, room, entry), the last ``rejoins``)."""
    drafts = read(path)
    if name not in drafts:
        raise DraftError(f'{path}: no draft {name!r} (it has {list(drafts)})')
    draft = drafts[name]
    if not isinstance(draft, dict) or set(draft) - {'extends', 'steps'} or not isinstance(draft.get('steps'), list):
        raise DraftError(f'{path}: {name}: needs `steps:` (a list) and may have `extends:`, nothing else')
    plan = [{**step, 'note': None, 'leg': None, 'wrap': False, 'takes': [], 'draft': False}
            for step in (route_steps(draft['extends']) if draft.get('extends') else [])]
    for number, element in enumerate(draft['steps']):
        what = f'{path}: {name}: element {number}'
        if not (isinstance(element, dict) and 'stretch' in element):
            plan.append(parse_step(element, f'{what} {element!r}', len(plan)))
            continue
        stretch = element['stretch']
        if set(element) - STRETCH_KEYS:
            raise DraftError(f'{what}: a stretch has the keys {sorted(STRETCH_KEYS)}')
        parts = room_entry(element.get('parts'), f'{what}: stretch {stretch}: parts')
        rejoins = room_entry(element.get('rejoins'), f'{what}: stretch {stretch}: rejoins')
        for way, _ in WAYS:
            ways = element.get(way)
            if not isinstance(ways, list) or not ways:
                raise DraftError(f'{what}: stretch {stretch}: needs a non-empty `{way}:` list of steps')
            for position, step in enumerate(ways):
                parsed = parse_step(step, f'{what}: stretch {stretch}: {way} {step!r}', len(plan), way, stretch)
                if position == 0:
                    parsed['parts'] = parts
                if position == len(ways) - 1:
                    parsed['rejoins'] = rejoins
                plan.append(parsed)
    return plan


def key(step: dict) -> tuple[int, int, bool]:
    return step['level'], step['room'], step['in_cave']


def describe(step: dict) -> str:
    return f"#{step['index']} {step['level']}:0x{step['room']:02x}{' (cave)' if step['in_cave'] else ''}"


def level_entrances(rooms: dict) -> dict[int, int]:
    """Level -> the overworld room whose cave is its entrance."""
    return {info['cave'][0]: room for (level, room, in_cave), info in rooms.items()
            if level == 0 and not in_cave and info['cave'] and info['cave'][0] != 0}


def on_way(step: dict, clips: bool) -> bool:
    return step.get('way') is None or step['way'] == ('clips' if clips else 'plain')


def problems(plan: list[dict], rooms: Optional[dict] = None) -> list[str]:
    """What is wrong with a draft's own steps (``steps``) against game.yaml, one line each; [] when it holds.  Each
    way is followed on its own (the clips switch on: the clip way of every stretch; off: the plain way)."""
    rooms = game_rooms() if rooms is None else rooms
    found = []
    for clips in (True, False):
        for line in way_problems([step for step in plan if on_way(step, clips)], rooms):
            if line not in found:
                found.append(line)
    return found


def way_problems(plan: list[dict], rooms: dict) -> list[str]:
    entrances = level_entrances(rooms)
    found = []
    for step, after in zip(plan, plan[1:] + [None]):
        if not step['draft'] and (after is None or not after['draft']):
            continue
        at, info = describe(step), rooms.get(key(step))
        if 'parts' in step and (step['level'], step['room'], step['entry']) != step['parts']:
            found.append(f'{at}: the first step of the {step["way"]} way of {step["stretch"]} is not where it parts')
        if 'rejoins' in step and (after is None or (after['level'], after['room'], after['entry']) != step['rejoins']):
            found.append(f'{at}: the {step["way"]} way of {step["stretch"]} does not lead into where it rejoins')
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
            if exit_ == 'maze':          # the sequence's last move leaves by the side the destination lies on
                entry = OPPOSITE.get(SIDE_OF.get(target[1] - step['room'], ''), '?')
            else:
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
