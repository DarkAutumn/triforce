"""Putting Link into any dungeon room without a savestate, to look at it (``zelda-map.py``'s pictures of rooms no
route state reaches yet, and checks of a level's data against the running game).

The method is the one Dungeons 3 and 4 were scouted with (docs/routes/d3.md "Getting in"), made general:

- **Into a level:** from the game start (``start.state``, Link in overworld 0x77) the room id is set to the level's
  entrance room and Link walks into 0x77's cave mouth.  The game takes the cave from the room id's attributes
  (asm:Warp_HandleOW), so it loads that level and Link stands in its start room.
- **Into a room:** Link leaves the room on screen through one of its open doorways with the room id set to the
  target's neighbour on that side, whose door type there is opened in the level block's RAM copy first (the game
  reads the door type of the current room id, asm:Room_CalcNext); it scrolls into the target and draws it from its
  own data.  A target is entered through one of its own open doors when the room on screen has the opposite side
  open; else a hop through another room of the level turns Link the right way.  A room with no open door is
  entered through a shutter (which shuts behind him), else a key door, else a bomb wall: the game draws the side
  Link came in by as open, so that one doorway of such a picture is not as the room starts.

Nothing here is how the game is played: it writes the room id, the level block and Link's position.
"""

from pathlib import Path
from typing import Callable, Optional

from triforce._native import Game, game_rooms

MODE, LEVEL, ROOM_ID = 0x12, 0x10, 0xEB                  # GameMode, CurLevel, RoomId
ATTRS_A, ATTRS_B, ATTRS_E = 0x687E, 0x68FE, 0x6A7E       # LevelBlockAttrsA/B/E: 128 rooms each
HEARTS, HEART_PARTIAL = 0x66F, 0x670                     # HeartValues, HeartPartial
PLAY = 5
HUD_ROWS, TILE_H = 64, 176                               # the frame's status bar, the playfield under it
SIDES = ('N', 'S', 'W', 'E')
OPPOSITE = {'N': 'S', 'S': 'N', 'W': 'E', 'E': 'W'}
DELTA = {'N': -0x10, 'S': 0x10, 'W': -1, 'E': 1}
BUTTON = {'N': 'UP', 'S': 'DOWN', 'W': 'LEFT', 'E': 'RIGHT'}
DOOR_SPOT = {'N': (0x78, 0x5D), 'S': (0x78, 0xBD), 'W': (0x20, 0x8D), 'E': (0xD0, 0x8D)}   # the floor before a door
DOOR_SHIFT = {'N': (ATTRS_A, 5), 'S': (ATTRS_A, 2), 'W': (ATTRS_B, 5), 'E': (ATTRS_B, 2)}  # the type's 3 bits
OPEN, WALL, BOMB, SHUTTER = 0, 1, 4, 7                   # door types; 5 and 6 are key doors
ENTRY_ORDER = (OPEN, SHUTTER, 5, 6, BOMB)                # which of a room's own doors to come in by, best first
DARK = 0x80                                              # attr E: the room is dark
CAVE_MOUTH = (0x40, 0x5D)                                # under 0x77's cave mouth: Up from here enters it
ENTER_FRAMES, WALK_FRAMES, SETTLE_FRAMES = 600, 400, 90
# What the route holds on entering Dungeon 5 (round 24; `write_entry_state`'s development state until the leg from
# Dungeon 4 delivers a played one): the white sword, the wooden boomerang, the bow without arrows, the raft, the
# ladder, Dungeons 1-4's triforce pieces, 10 full heart containers and 6 keys; bombs are the caller's (0-8).
D5_INVENTORY = {'sword': 2, 'regular_boomerang': 1, 'bow': 1, 'arrows': 0, 'raft': 1, 'step_ladder': 1,
                'triforce': 0x0F, 'hearts_and_containers': 0x99, 'partial_hearts': 0xFF, 'keys': 6, 'rupees': 0}


class ProbeError(RuntimeError):
    """The probe didn't get where it was sent."""


class Probe:
    """One emulator, sent into dungeon rooms.  ``enter_room`` leaves Link playing in the room; ``playfield`` is its
    picture."""

    def __init__(self, states_dir):
        self.states_dir = Path(states_dir)
        self.game = Game(render=True, states_dir=str(self.states_dir))
        self.pre_entry = None             # the last room entry's pre-entry snapshot (`_walk_into`)
        rooms = game_rooms()
        self.entrances = {info['cave'][0]: room for (level, room, in_cave), info in rooms.items()
                          if level == 0 and not in_cave and info['cave'] and info['cave'][0]}
        self.level_rooms = {level: sorted(room for (lv, room, in_cave) in rooms if lv == level and not in_cave)
                            for level in range(1, 10)}
        self._levels = {}               # level -> (snapshot in its start room, start room)

    # ---- the level
    def enter_level(self, level: int) -> int:
        """Link in the level's start room, as he comes down its stairs.  Returns the start room."""
        if level not in self._levels:
            if level not in self.entrances:
                raise ProbeError(f'game.yaml has no overworld entrance to level {level}')
            game = self.game
            game.load(str(self.states_dir / 'start.state'), randomize_rng=False)
            game.apply_ram({'location': self.entrances[level], 'link_x': CAVE_MOUTH[0], 'link_y': CAVE_MOUTH[1]})
            for _ in range(ENTER_FRAMES):
                game.press('UP', 1)
                if game.read(MODE) == PLAY and game.read(LEVEL) == level:
                    break
            else:
                raise ProbeError(f'level {level} did not load')
            self._levels[level] = (game.snapshot(), game.read(ROOM_ID))
        snapshot, start = self._levels[level]
        self.game.restore(snapshot)
        return start

    def door_type(self, room: int, side: str) -> int:
        """A room's door type on a side, from the level block in RAM (0 open, 1 wall, 4 bomb, 5/6 key, 7 shutter)."""
        address, shift = DOOR_SHIFT[side]
        return (self.game.read(address + room) >> shift) & 7

    def doors(self, room: int) -> dict[str, int]:
        return {side: self.door_type(room, side) for side in SIDES}

    def _source(self, room: int, entry: str) -> Optional[int]:
        """The room id that leads into ``room`` through its ``entry`` side, None when the id space has none."""
        source = room + DELTA[entry]
        return source if 0 <= source < 0x80 else None

    def _hops(self, start: int, level: int, target: int, entry: Optional[str]) -> list[tuple[int, str]]:
        """[(room, the side it is entered by)] ending in the target: each room is left through a doorway that is
        open on screen (its own open doors and the one Link came in by)."""
        entries = [entry] if entry else sorted(
            (side for side in SIDES if self.door_type(target, side) != WALL),
            key=lambda side: ENTRY_ORDER.index(self.door_type(target, side)))
        entries = [side for side in entries if self._source(target, side) is not None]
        open_sides = lambda room, came: {s for s in SIDES if self.door_type(room, s) == OPEN} | ({came} - {None})
        start_open = open_sides(start, None)
        for side in entries:                                    # straight from the start room
            if OPPOSITE[side] in start_open:
                return [(target, side)]
        for side in entries:                                    # through one room that turns Link
            for hub in self.level_rooms[level]:
                for came in SIDES:
                    if (hub not in (start, target) and self.door_type(hub, came) == OPEN
                            and self._source(hub, came) is not None and OPPOSITE[came] in start_open
                            and OPPOSITE[side] in open_sides(hub, came)):
                        return [(hub, came), (target, side)]
        raise ProbeError(f'no way into {level}:0x{target:02x}' + (f' from {entry}' if entry else ''))

    def _walk_into(self, level: int, room: int, entry: str):
        """From the room on screen out of its doorway opposite ``entry`` and into ``room``; ``pre_entry`` is the
        snapshot of the frame before the room's entry (an entry state's moment: its monsters not made yet)."""
        from triforce.entry_states import is_pre_entry      # pylint: disable=import-outside-toplevel
        game, leave = self.game, OPPOSITE[entry]
        source = self._source(room, entry)
        address, shift = DOOR_SHIFT[leave]
        game.write(address + source, game.read(address + source) & ~(7 << shift) & 0xFF)
        game.apply_ram({'location': source, 'link_x': DOOR_SPOT[leave][0], 'link_y': DOOR_SPOT[leave][1]})
        left, self.pre_entry = False, None
        for _ in range(WALK_FRAMES):
            if left and is_pre_entry(game.read):
                self.pre_entry = game.snapshot()
            game.press(BUTTON[leave], 1)
            mode = game.read(MODE)
            left = left or mode != PLAY
            if left and mode == PLAY:
                break
        if not (left and game.read(MODE) == PLAY and game.read(LEVEL) == level and game.read(ROOM_ID) == room):
            raise ProbeError(f'did not get into {level}:0x{room:02x} from {entry} (mode {game.read(MODE)}, room '
                             f'0x{game.read(ROOM_ID):02x})')

    def enter_room(self, level: int, room: int, entry: Optional[str] = None, lit: bool = True,
                   setup: Optional[Callable[[Game], None]] = None) -> str:
        """Link playing in ``room`` of ``level``, on its first frame, come in by ``entry`` (default: the best of its
        own doors, ``ENTRY_ORDER``).  ``lit`` clears the room's dark flag first; ``setup(game)`` runs in the level's
        start room before Link walks (items, room flags: what a route would have done by then).  Returns the side he
        came in by.  The start room entered without an ``entry`` is the level entry itself (he stands at its S door)."""
        start = self.enter_level(level)
        if setup is not None:
            setup(self.game)
        if room == start and entry is None:
            return 'S'
        hops = self._hops(start, level, room, entry)
        if lit:
            self.game.write(ATTRS_E + room, self.game.read(ATTRS_E + room) & ~DARK & 0xFF)
        for hop, side in hops:
            self._walk_into(level, hop, side)
        return hops[-1][1]

    def settle(self, frames: int = SETTLE_FRAMES):
        """Lets the room's monsters come out, Link standing still and kept alive."""
        game = self.game
        hearts, partial = game.read(HEARTS), game.read(HEART_PARTIAL)
        for _ in range(frames):
            game.press('NONE', 1)
            game.write(HEARTS, hearts)
            game.write(HEART_PARTIAL, partial)
            if game.read(MODE) != PLAY:
                break

    def playfield(self, level: int, room: int):
        """The room's playfield (176 x 256 x 3, uint8) with its monsters out, lit."""
        self.enter_room(level, room)
        self.settle()
        return self.game.screen()[HUD_ROWS:HUD_ROWS + TILE_H]


def write_entry_state(path, level: int, inventory: dict, states_dir) -> None:
    """Writes a pre-entry state of ``level``'s start room (the frame before Link, come down its stairs, plays in it),
    with ``inventory`` (``Game.set_value`` names) written first: a development state for a dungeon no played state
    reaches yet (round 24: Dungeon 5 with ``D5_INVENTORY``), for tests and scouting only.  Never install one into
    triforce/states: its inventory is written, not played (the audit flags its name and hearts)."""
    from triforce.entry_states import is_pre_entry      # pylint: disable=import-outside-toplevel
    probe = Probe(states_dir)
    game = probe.game
    game.load(str(Path(states_dir) / 'start.state'), randomize_rng=False)
    for name, value in inventory.items():
        game.set_value(name, value)
    game.apply_ram({'location': probe.entrances[level], 'link_x': CAVE_MOUTH[0], 'link_y': CAVE_MOUTH[1]})
    pre_entry = None
    for _ in range(ENTER_FRAMES):
        if is_pre_entry(game.read):
            pre_entry = game.snapshot()
        game.press('UP', 1)
        if game.read(MODE) == PLAY and game.read(LEVEL) == level:
            break
    else:
        raise ProbeError(f'level {level} did not load')
    if pre_entry is None:
        raise ProbeError(f'no pre-entry frame on the way into level {level}')
    game.restore(pre_entry)
    game.save_state_file(str(path))
