"""Draft routes (``triforce/draft_routes``) against game.yaml, and the dungeon probe (``triforce/probe``) against the
running game."""

import textwrap

import pytest

from .conftest import require_rom

SIDES = ('N', 'S', 'W', 'E')


def write_draft(tmp_path, steps: str):
    path = tmp_path / 'drafts.yaml'
    path.write_text('draft:\n  steps:\n' + textwrap.indent(textwrap.dedent(steps), '  '))
    return path


def problems_of(tmp_path, steps: str) -> list[str]:
    from triforce import draft_routes
    return draft_routes.problems(draft_routes.steps('draft', write_draft(tmp_path, steps)))


def test_the_shipped_drafts_hold_against_game_yaml():
    """Every draft step's exit is one of its room's and leads into the next step's room from the side it says."""
    from triforce import draft_routes
    assert draft_routes.names(), 'no draft routes'
    for name in draft_routes.names():
        assert draft_routes.problems(draft_routes.steps(name)) == [], name


def test_a_walk_a_wrap_exit_and_the_whirlwind_are_followed(tmp_path):
    """0x52 W into 0x51, game.yaml's wrap exit out of 0x51's solid W edge into 0x50 from E, the whirlwind to Dungeon
    1's entrance, down into the level's start room; and the wrap round the world, 0x60 into 0x5f."""
    assert problems_of(tmp_path, """\
        - {level: 0, room: '0x60', entry: N, exit: W, wrap: true}
        - {level: 0, room: '0x5f', entry: E, exit: end}
        """) == []
    assert problems_of(tmp_path, """\
        - {level: 0, room: '0x52', entry: N, exit: W}
        - {level: 0, room: '0x51', entry: E, exit: W, wrap: true}
        - {level: 0, room: '0x50', entry: E, exit: recorder}
        - {level: 0, room: '0x37', entry: recorder, exit: cave}
        - {level: 1, room: '0x73', entry: S, exit: end}
        """) == []


@pytest.mark.parametrize('steps, problem', [
    # 0x51's W exit is a wrap exit: only a wrap gets out that way
    ("- {level: 0, room: '0x51', entry: E, exit: W}\n- {level: 0, room: '0x50', entry: E, exit: end}",
     'W is a wrap exit'),
    # 0x50 has no W exit at all
    ("- {level: 0, room: '0x50', entry: E, exit: W, wrap: true}\n- {level: 0, room: '0x4f', entry: E, exit: end}",
     'W is not an exit of the room'),
    # a wrap exit leads into the room game.yaml names, nowhere else
    ("- {level: 0, room: '0x51', entry: E, exit: W, wrap: true}\n- {level: 0, room: '0x40', entry: S, exit: end}",
     'W leads to 0:0x50'),
    # 0x52 W is 0x51
    ("- {level: 0, room: '0x52', entry: N, exit: W}\n- {level: 0, room: '0x53', entry: E, exit: end}",
     'W leads to 0:0x51'),
    # walking W, Link comes in from E
    ("- {level: 0, room: '0x52', entry: N, exit: W}\n- {level: 0, room: '0x51', entry: N, exit: end}",
     'entered from E'),
    # the whirlwind only goes to a dungeon's entrance
    ("- {level: 0, room: '0x52', entry: N, exit: recorder}\n- {level: 0, room: '0x51', entry: recorder, exit: end}",
     'sets Link down at a dungeon entrance'),
    # a tunnel's W ladder comes up in the room of its attr A, not the other end
    ("- {level: 5, room: '0x64', entry: E, exit: cave}\n- {level: 5, room: '0x07', in_cave: true, entry: S, exit: W}\n"
     "- {level: 5, room: '0x06', entry: cave, exit: end}", 'W leads to 5:0x64'),
    # only a triforce room ends a level
    ("- {level: 5, room: '0x24', entry: E, exit: triforce}\n- {level: 0, room: '0x0b', entry: cave, exit: end}",
     "the room's treasure is heart-container"),
    ("- {level: 5, room: '0xff', entry: E, exit: end}", 'no such room'),
])
def test_a_step_that_does_not_follow_is_named(tmp_path, steps, problem):
    found = problems_of(tmp_path, steps + '\n')
    assert len(found) == 1 and problem in found[0], found


def test_a_tunnel_joins_its_two_rooms(tmp_path):
    """Level 5's tunnel 0x07: down 0x64's stairs, up the right ladder into 0x06, and back the other way."""
    assert problems_of(tmp_path, """\
        - {level: 5, room: '0x64', entry: E, exit: cave}
        - {level: 5, room: '0x07', in_cave: true, entry: S, exit: E}
        - {level: 5, room: '0x06', entry: cave, exit: cave}
        - {level: 5, room: '0x07', in_cave: true, entry: S, exit: W}
        - {level: 5, room: '0x64', entry: cave, exit: end}
        """) == []


@pytest.mark.parametrize('step, why', [
    ("{level: 0, room: '0x52', entry: N, exit: W, takes: [hart-container]}", 'takes is a list of'),
    ("{level: 0, room: '0x52', entry: N, exit: W, notes: 'a typo'}", 'the keys are'),
    ("{level: 0, room: '0x52', entry: N, exit: whirlwind}", 'exit one of'),
    ("{level: 0, room: 82, entry: N, exit: W}", None),          # an unquoted room reads as 0x82: no such room
])
def test_a_typo_in_a_draft_step_is_refused(tmp_path, step, why):
    from triforce import draft_routes
    path = write_draft(tmp_path, f'- {step}\n')
    if why is None:
        assert 'no such room' in draft_routes.problems(draft_routes.steps('draft', path))[0]
    else:
        with pytest.raises(draft_routes.DraftError, match=why):
            draft_routes.steps('draft', path)


@pytest.fixture(scope='module')
def probe():
    require_rom()
    from triforce.paths import STATES_DIR
    from triforce.probe import Probe
    return Probe(STATES_DIR)


@pytest.mark.parametrize('level', range(1, 10))
def test_game_yaml_doors_are_the_games(probe, level):
    """Every side of every game.yaml room of a level: the door kind game.yaml gives (open, key door, bomb wall,
    shutter, or no exit) is the level block's door type in the running game.  The map's door marks and the draft
    routes rest on this."""
    from triforce._native import game_rooms
    from triforce.probe import BOMB, OPEN, SHUTTER, WALL
    rooms = game_rooms()
    assert probe.enter_level(level) in probe.level_rooms[level]
    kinds = {OPEN: 'open', WALL: None, BOMB: 'bomb', 5: 'key', 6: 'key', SHUTTER: 'shutter'}
    wrong = []
    for room in probe.level_rooms[level]:
        exits = rooms[(level, room, False)]['exits']
        for side in SIDES:
            exit_ = exits.get(side)
            yaml_kind = (None if exit_ is None else 'bomb' if exit_['bomb_wall'] else 'key' if exit_['locked']
                         else 'shutter' if exit_['shutter'] else 'open')
            game_kind = kinds[probe.door_type(room, side)]
            # A doorway out of the level (the start room's S side) is a door in the game and no exit in game.yaml.
            if yaml_kind != game_kind and not (exit_ is None and room == probe.enter_level(level) and side == 'S'):
                wrong.append((hex(room), side, yaml_kind, game_kind))
    assert wrong == []


@pytest.mark.parametrize('level, room', [(5, 0x66), (5, 0x15), (9, 0x07), (9, 0x42), (4, 0x11)])
def test_the_probe_stands_in_the_room_it_was_sent_to(probe, level, room):
    """Rooms next to the start room and far from it, with open doors and with none (5:0x15 and 9:0x07 have only bomb
    walls, 4:0x11 four of them, 9:0x42 only shutters): Link plays there, and the picture is a playfield."""
    from triforce.probe import LEVEL, MODE, PLAY, ROOM_ID
    side = probe.enter_room(level, room)
    game = probe.game
    assert (game.read(MODE), game.read(LEVEL), game.read(ROOM_ID)) == (PLAY, level, room)
    assert probe.door_type(room, side) != 1, 'came in through a wall'
    assert probe.playfield(level, room).shape == (176, 256, 3)
