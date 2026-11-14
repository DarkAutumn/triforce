"""Dungeon 5 (round 24, docs/routes/d5.md): the route `d5`, its objective pieces and the owner's rules, played in the
running game.  Link is put into each room by ``triforce.probe`` with what the route holds there (``D5_INVENTORY``:
no played start state reaches Dungeon 5 yet), and an env of the room's family follows the route step."""

import numpy as np
import pytest

from .conftest import require_rom

LEVEL = 5
OBJ_ID = 0x34F                  # ObjType, slots 0 (Link) to 11 (zelda::tables::OBJ_ID)
HELP_COUNT = 0x50               # the damage-free kill streak: at 9 the next kill arms a forced bomb drop
SIDE_BIT = {'E': 1, 'W': 2, 'S': 4, 'N': 8}     # a room flag's door bits (zelda::enums::Direction::value)
OPPOSITE = {'N': 'S', 'S': 'N', 'W': 'E', 'E': 'W'}
NEIGHBOUR = {'N': -0x10, 'S': 0x10, 'W': -1, 'E': 1}
FACING = {'E': 1, 'W': 2, 'S': 4, 'N': 8}       # LinkDirection


@pytest.fixture(scope='module')
def probe():
    require_rom()
    from triforce.paths import STATES_DIR
    from triforce.probe import Probe
    return Probe(STATES_DIR)


def steps():
    from triforce._native import route_steps
    return route_steps('d5')


def step_of(room: int, entry: str, in_cave: bool = False, visit: int = 0) -> int:
    """The d5 step index of the Dungeon 5 room entered from ``entry`` (its ``visit``-th such step: the tunnel cellar
    0x07 is entered from its stairs twice)."""
    found = [s['index'] for s in steps() if (s['level'], s['room'], s['in_cave'], s['entry']) == (LEVEL, room, in_cave,
                                                                                                    entry)]
    assert len(found) == (2 if (room, in_cave) == (0x07, True) else 1), (hex(room), entry, found)
    return found[visit]


def room_flags_address(game) -> int:
    """The level's 128 room flag bytes (entry_states.WORLD_FLAGS_ADDR points at them)."""
    from triforce.entry_states import WORLD_FLAGS_ADDR
    return game.read(WORLD_FLAGS_ADDR) | (game.read(WORLD_FLAGS_ADDR + 1) << 8)


def open_wall(game, room: int, side: str):
    """Marks the wall between ``room`` and its neighbour on ``side`` open in both rooms' flags (a bombed wall)."""
    base = room_flags_address(game)
    neighbour = room + NEIGHBOUR[side]
    game.write(base + room, game.read(base + room) | SIDE_BIT[side])
    game.write(base + neighbour, game.read(base + neighbour) | SIDE_BIT[OPPOSITE[side]])


def enter(probe, room: int, entry: str, *, recorder: bool = False, bombs: int = 2, walls=(), settle: int = 60):
    """A snapshot of Link playing in Dungeon 5's ``room``, come in from ``entry``, holding the route's items there:
    ``walls`` [(room, side)] are bombed open first (what the route did on the way)."""
    from triforce.probe import D5_INVENTORY

    def setup(game):
        for name, value in dict(D5_INVENTORY, bombs=bombs, whistle=int(recorder)).items():
            game.set_value(name, value)
        for wall in walls:
            open_wall(game, *wall)

    probe.enter_room(LEVEL, room, entry, lit=False, setup=setup)
    probe.settle(settle)
    return probe.game.snapshot()


class Room:
    """An env of ``family`` playing route step ``step`` of d5 from ``snapshot``."""

    def __init__(self, family: str, step: int, snapshot, clips=None):
        from triforce._native import Env
        self.env = Env(family, explicit_starts=True, strict_actions=False)
        self.env.set_route('d5', step)
        if clips is not None:
            self.env.set_clips_allowed(clips)
        self.obs, self.info = self.env.restore(snapshot, f'5_{step}')
        self.rewards = {}
        self.done = False

    @property
    def state(self):
        return self.env.state

    def objective(self) -> str:
        from triforce._native import OBJECTIVE_FEATURES
        values = np.asarray(self.obs['objective'])
        kinds = [(v, name[len('objective_'):]) for v, name in zip(values, OBJECTIVE_FEATURES)
                 if name.startswith('objective_')]
        return max(kinds)[1]

    def next_sides(self) -> set:
        from triforce._native import OBJECTIVE_FEATURES
        values = np.asarray(self.obs['objective'])
        return {name[-1] for v, name in zip(values, OBJECTIVE_FEATURES) if name.startswith('next_room_') and v > 0}

    def allowed(self, kind: str) -> bool:
        from triforce._native import ACTION_KINDS
        return bool(np.asarray(self.env.mask)[ACTION_KINDS.index(kind)])

    def act(self, action: str):
        self.obs, _, terminated, truncated, self.info = self.env.step(action)
        for name, value in self.info.get('rewards', {}).items():
            self.rewards[name] = self.rewards.get(name, 0.0) + value
        self.done = terminated or truncated
        return self.info

    def until_room_changes(self, action: str, limit: int = 40) -> dict:
        start = tuple(self.info['room'])
        for _ in range(limit):
            self.act(action)
            if self.done or tuple(self.info['room']) != start:
                break
        return self.info

    def ending(self):
        return self.info.get('ending')


def place(game, x: int, y: int, facing: str):
    game.apply_ram({'link_x': x, 'link_y': y, 'link_direction': FACING[facing], 'doorway_dir': 0,
                    'link_grid_offset': 0})


def kill_monsters(game, frames: int = 64):
    """Empties the monster slots (the room counts them dead) and lets the game notice."""
    for slot in range(1, 11):
        game.write(OBJ_ID + slot, 0)
    for _ in range(frames):
        game.press('NONE', 1)


# ---- the route


def test_d5_extends_d2first_and_ends_on_the_d5_triforce():
    """`d5` is d2first, then the draft's 23 Dungeon 5 steps (routes.yaml); its families and treasure rules are the
    owner's notes (docs/routes/d5.md)."""
    from triforce._native import route_steps
    base, d5 = route_steps('d2first'), steps()
    assert [s for s in d5[:len(base)]] == base
    own = [(s['room'], s['in_cave'], s['entry'], s['exit']) for s in d5[len(base):]]
    assert own == [
        (0x76, False, 'S', 'N'), (0x66, False, 'S', 'W'), (0x65, False, 'E', 'W'), (0x64, False, 'E', 'cave'),
        (0x07, True, 'S', 'E'), (0x06, False, 'cave', 'W'), (0x05, False, 'E', 'cave'), (0x04, True, 'S', 'N'),
        (0x05, False, 'cave', 'E'), (0x06, False, 'W', 'cave'), (0x07, True, 'S', 'W'), (0x64, False, 'cave', 'E'),
        (0x65, False, 'W', 'E'), (0x66, False, 'W', 'N'), (0x56, False, 'S', 'E'), (0x57, False, 'W', 'N'),
        (0x47, False, 'S', 'N'), (0x37, False, 'S', 'N'), (0x27, False, 'S', 'W'), (0x26, False, 'E', 'W'),
        (0x25, False, 'E', 'W'), (0x24, False, 'E', 'N'), (0x14, False, 'S', 'triforce')]
    rules = {(s['room'], s['entry']): s['treasure'] for s in d5[len(base):] if s['treasure']}
    assert rules == {(0x66, 'S'): 'skip', (0x65, 'E'): 'refuse', (0x56, 'S'): 'skip', (0x57, 'W'): 'skip',
                     (0x47, 'S'): 'skip', (0x37, 'S'): 'skip', (0x26, 'E'): 'skip'}
    assert [s['waypoint'] for s in d5[len(base):] if s['waypoint']] == ['recorder']
    families = {s['room']: s['specialist'] for s in d5[len(base):]}
    assert families == {0x76: 'gibdo', 0x66: 'gibdo', 0x65: 'gibdo', 0x64: 'darknut', 0x07: 'equipment-room',
                        0x06: 'darknut', 0x05: 'darknut', 0x04: 'equipment-room', 0x56: 'dodongo-crossing',
                        0x57: 'zol', 0x47: 'gibdo', 0x37: 'darknut', 0x27: 'pols-voice', 0x26: 'gibdo',
                        0x25: 'pols-voice', 0x24: 'boss-digdogger', 0x14: 'gibdo'}


# ---- 0x66 and 0x65: the walls, the key, the bombs


def test_0x66_on_entry_is_crossed_to_its_w_wall_and_bombed_once(probe):
    """Westbound (step 0x66 S -> W, `treasure: skip`): no fight and no key (the owner: "don't kill 0x66 on entry"),
    the W bomb wall is the exit; BOMB is offered only at the spot that opens it (gibdo `required`), and one bomb
    opens it."""
    snapshot = enter(probe, 0x66, 'S', bombs=2)
    room = Room('gibdo', step_of(0x66, 'S'), snapshot)
    assert len(room.state.enemies) == 3
    assert (room.objective(), room.next_sides(), room.allowed('BOMB')) == ('MOVE', {'W'}, False)
    game = probe.game
    game.restore(snapshot)
    place(game, 0x30, 0x8D, 'W')
    room = Room('gibdo', step_of(0x66, 'S'), game.snapshot())
    assert room.allowed('BOMB'), 'at the W wall'
    room.act('BOMB W')
    assert room.state.bombs == 1
    room.until_room_changes('MOVE W 16')
    assert (room.ending(), tuple(room.info['room'])) == ('exit-correct', (LEVEL, 0x65, False))
    assert room.rewards['reward-opened-bomb-wall'] > 0 and room.state.bombs == 1


def test_0x66_on_the_way_back_is_cleared_for_its_shutter_and_its_key(probe):
    """Eastbound (step 0x66 W -> N): the key the slot-1 Gibdo carries first (the Treasure target rides on it), then
    the fight the N shutter needs, then N over the water by the ladder."""
    snapshot = enter(probe, 0x66, 'W', walls=[(0x66, 'W'), (0x65, 'W')])
    step = step_of(0x66, 'W')
    assert Room('gibdo', step, snapshot).objective() == 'TREASURE'
    room = _touch_carrier(probe, snapshot, step)
    assert room.rewards.get('reward-gained-keys', 0) > 0
    assert (room.objective(), room.next_sides()) == ('FIGHT', set()), 'the N shutter opens when they are dead'
    game = probe.game
    game.restore(room.env.snapshot())
    kill_monsters(game)
    room = Room('gibdo', step, game.snapshot())
    assert (room.objective(), room.next_sides()) == ('MOVE', {'N'})
    place(game, 0x78, 0x8D, 'N')
    room = Room('gibdo', step, game.snapshot())
    room.until_room_changes('MOVE N 16')
    assert (room.ending(), tuple(room.info['room'])) == ('exit-correct', (LEVEL, 0x56, False)), 'the ladder'


def _touch_carrier(probe, snapshot, step: int) -> 'Room':
    """Link walks into the slot-1 Gibdo, which carries the room's item (taken by touch, D5Paths)."""
    game = probe.game
    game.restore(snapshot)
    carrier = next(e for e in Room('gibdo', step, snapshot).state.enemies if e['index'] == 1)
    assert (game.read(0x83), game.read(0x97)) == (carrier['x'], carrier['y']), 'the item rides on its carrier'
    place(game, carrier['x'], carrier['y'] + 2, 'N')
    room = Room('gibdo', step, game.snapshot())
    for _ in range(3):
        room.act('MOVE N 4')
    return room


def test_0x65_westbound_refuses_its_bombs(probe):
    """Westbound (step 0x65 E -> W, `treasure: refuse`; the owner: never take them westbound): no objective goes for
    them, and taking them costs `penalty-refused-treasure` instead of paying `reward-gained-bombs`."""
    snapshot = enter(probe, 0x65, 'E', bombs=1, walls=[(0x66, 'W')])
    step = step_of(0x65, 'E')
    assert (Room('gibdo', step, snapshot).objective(), Room('gibdo', step, snapshot).next_sides()) == ('MOVE', {'W'})
    room = _touch_carrier(probe, snapshot, step)
    assert room.state.bombs == 5, 'the four bombs were taken'
    assert room.rewards.get('penalty-refused-treasure', 0) < 0 and 'reward-gained-bombs' not in room.rewards


def test_0x65_eastbound_takes_its_bombs(probe):
    """Eastbound (step 0x65 W -> E): the bombs are the objective (the Treasure target rides on the Gibdo) and pay."""
    snapshot = enter(probe, 0x65, 'W', bombs=0, walls=[(0x66, 'W'), (0x65, 'W')])
    step = step_of(0x65, 'W')
    assert Room('gibdo', step, snapshot).objective() == 'TREASURE'
    room = _touch_carrier(probe, snapshot, step)
    assert room.rewards.get('reward-gained-bombs', 0) > 0 and 'penalty-refused-treasure' not in room.rewards
    assert (room.objective(), room.next_sides()) == ('MOVE', {'E'})


# ---- the Darknut stretch: 0x64, the tunnel cellar 0x07, 0x06, the Olive Garden 0x05 and the recorder cellar 0x04

WALLS = [(0x66, 'W'), (0x65, 'W')]             # bombed westbound: open on the rooms behind them


def cellar(probe, room: int, entry: str, stairs, *, recorder: bool = False, push=None) -> object:
    """A snapshot of Link come down ``room``'s stairs (at ``stairs``, Link's x and y) into its cellar, controllable at
    the top of its left ladder; ``push`` (x, y, direction) moves the room's block first (its monsters killed)."""
    game = probe.game
    game.restore(enter(probe, room, entry, recorder=recorder, walls=WALLS))
    if push is not None:
        kill_monsters(game)
        x, y, direction = push
        place(game, x, y, direction)
        for _ in range(40):
            game.press({'N': 'UP', 'S': 'DOWN'}[direction], 1)
        for _ in range(64):
            game.press('NONE', 1)
    place(game, *stairs, 'N')
    for _ in range(600):
        game.press('NONE', 1)
        if game.read(0x12) == 9 and (game.read(0x70), game.read(0x84)) == (0x30, 0x5D):
            break
    for _ in range(100):                            # the cellar laid out, its item and Keese out
        game.press('NONE', 1)
    assert (game.read(0x12), game.read(0x70)) == (9, 0x30), 'at the top of the cellar\'s left ladder'
    return game.snapshot()


def test_the_tunnel_cellar_is_left_by_the_steps_ladder(probe):
    """5:0x07 has two ladders (game.yaml exits W, X < $80, to 0x64 and E to 0x06, asm:Warp_CheckSubroomExit); each
    of its two steps counts only its own: without the recorder (westbound) the left ladder Link arrives on is a wrong
    exit and the right one correct; with it (on the way back) the left one is correct."""
    down = cellar(probe, 0x64, 'E', (0x80, 0x8D))
    there, back = step_of(0x07, 'S', in_cave=True), step_of(0x07, 'S', in_cave=True, visit=1)
    assert (steps()[there]['exit'], steps()[back]['exit']) == ('E', 'W')
    room = Room('equipment-room', there, down)
    assert room.objective() == 'MOVE'
    room.until_room_changes('MOVE N 16')
    assert (room.ending(), tuple(room.info['room'])) == ('exit-wrong', (LEVEL, 0x64, False))
    game = probe.game
    game.restore(down)
    game.set_value('whistle', 1)
    room = Room('equipment-room', back, game.snapshot())
    room.until_room_changes('MOVE N 16')
    assert (room.ending(), tuple(room.info['room'])) == ('exit-correct', (LEVEL, 0x64, False))
    # Westbound the right ladder (X $C0): down the left one, along the floor, up the right one.
    room = Room('equipment-room', there, down)
    for action in ('MOVE S 16', 'MOVE E 16'):
        for _ in range(20):
            before = (room.state.link['x'], room.state.link['y'])
            room.act(action)
            if (room.state.link['x'], room.state.link['y']) == before or room.state.link['x'] >= 0xB0:
                break
    assert room.state.link['y'] == 0xBD and not room.done, 'on the floor'
    game.restore(room.env.snapshot())
    place(game, 0xC0, 0xBD, 'N')
    room = Room('equipment-room', there, game.snapshot())
    room.until_room_changes('MOVE N 16')
    assert (room.ending(), tuple(room.info['room'])) == ('exit-correct', (LEVEL, 0x06, False))


def test_0x64_the_clips_switch_picks_the_block_clip_or_the_fight(probe):
    """One step (0x64 E -> its stairs); the run-time clips switch decides: on, no fight, CLIP N from ($A0,$9D) into
    the block diamond with the five Blue Darknuts alive, the stairs; off, the fight, then the W block pushed N, the
    stairs (the owner: "just clip the blocks to the stairs ... the no-clip version")."""
    snapshot = enter(probe, 0x64, 'E', walls=WALLS)
    step = step_of(0x64, 'E')
    assert len(Room('darknut', step, snapshot).state.enemies) == 5
    assert Room('darknut', step, snapshot, clips=True).objective() == 'MOVE'
    assert Room('darknut', step, snapshot, clips=False).objective() == 'FIGHT'
    game = probe.game
    game.restore(snapshot)
    place(game, 0xA0, 0x9D, 'N')
    room = Room('darknut', step, game.snapshot(), clips=True)
    assert room.allowed('CLIP') and len(room.state.enemies) == 5
    room.act('CLIP N')
    room.until_room_changes('MOVE W 16')
    assert (room.ending(), tuple(room.info['room'])) == ('exit-correct', (LEVEL, 0x07, True))
    assert 'penalty-lost-health' not in room.rewards
    off = Room('darknut', step, game.snapshot(), clips=False)
    assert not off.allowed('CLIP'), 'the switch off masks CLIP'
    # The plain way: the room cleared, the push spots, the push N, the stairs.
    game.restore(snapshot)
    kill_monsters(game)
    place(game, 0x60, 0x95, 'N')
    room = Room('darknut', step, game.snapshot(), clips=False)
    assert room.objective() == 'MOVE'
    for action in ['MOVE N 16'] * 4 + ['WAIT 16'] * 4:
        room.act(action)
    assert room.rewards.get('reward-pushed-block', 0) > 0
    room.act('MOVE N 8')
    room.until_room_changes('MOVE E 16')
    assert (room.ending(), tuple(room.info['room'])) == ('exit-correct', (LEVEL, 0x07, True))


def test_0x64_on_the_way_back_runs_for_the_e_wall_among_the_darknuts(probe):
    """Out of the tunnel cellar (step 0x64 cave -> E): no fight, the opened E wall."""
    game = probe.game
    game.restore(enter(probe, 0x64, 'E', recorder=True, walls=WALLS))
    place(game, 0x80, 0x8D, 'N')                     # into the diamond (the clip way's arrival) and down
    for _ in range(300):
        game.press('NONE', 1)
        if game.read(0x12) == 9:
            break
    game.set_value('whistle', 1)
    for _ in range(200):
        game.press('UP', 1)
        if game.read(0x12) == 5 and game.read(0xEB) == 0x64:
            break
    room = Room('darknut', step_of(0x64, 'cave'), game.snapshot())
    assert room.state.enemies, 'the Darknuts the clip way left alive'
    assert (room.objective(), room.next_sides()) == ('MOVE', {'E'})


def test_the_olive_garden_is_cleared_its_block_pushed_and_its_stairs_taken(probe):
    """5:0x05 from E (step 0x05 E -> its stairs): the six Blue Darknuts (forced: the block moves only in a clear
    room, game.yaml `push_block: [N, S, E, W]`), the push, the stairs that appear top right (26, 4), down to the
    recorder cellar 0x04: the recorder (`waypoint: recorder`), then up its ladder, out to 0x05, which is left E."""
    snapshot = enter(probe, 0x05, 'E', walls=WALLS)
    step = step_of(0x05, 'E')
    room = Room('darknut', step, snapshot)
    assert (len(room.state.enemies), room.objective()) == (6, 'FIGHT')
    down = cellar(probe, 0x05, 'E', (0xD0, 0x5D), push=(0x70, 0x95, 'N'))
    game = probe.game
    assert (game.read(0xEB), game.read(0x12)) == (0x04, 9)
    room = Room('equipment-room', step_of(0x04, 'S', in_cave=True), down)
    assert room.objective() == 'TREASURE'
    room.until_room_changes('MOVE N 16')
    assert room.ending() == 'exit-wrong', 'not without the recorder'
    game.restore(down)
    place(game, 0x80, 0x8D, 'E')
    room = Room('equipment-room', step_of(0x04, 'S', in_cave=True), game.snapshot())
    room.act('WAIT 4')
    assert game.read(0x65C) == 0, 'the probe game is not the env\'s'
    assert 'reward-gained-whistle' in room.rewards and room.objective() == 'MOVE'
    game.restore(room.env.snapshot())
    place(game, 0x30, 0x5D, 'N')
    room = Room('equipment-room', step_of(0x04, 'S', in_cave=True), game.snapshot())
    room.until_room_changes('MOVE N 16')
    assert (room.ending(), tuple(room.info['room'])) == ('exit-correct', (LEVEL, 0x05, False))
    game.restore(room.env.snapshot())
    room = Room('darknut', step_of(0x05, 'cave'), game.snapshot())
    assert (room.objective(), room.next_sides()) == ('MOVE', {'E'})


def test_0x06_out_of_the_tunnel_w_and_back_by_its_block(probe):
    """5:0x06 (no monsters): out of the tunnel cellar, W through the key door; back from W, the W block pushed N
    (pushed E it would wall the stairs in) and down its stairs into the tunnel."""
    game = probe.game
    snapshot = enter(probe, 0x06, 'W', recorder=True, walls=WALLS)
    room = Room('darknut', step_of(0x06, 'W'), snapshot)
    assert room.objective() == 'MOVE'
    game.restore(snapshot)
    place(game, 0x60, 0x95, 'N')
    room = Room('darknut', step_of(0x06, 'W'), game.snapshot())
    for action in ['MOVE N 16'] * 4 + ['WAIT 16'] * 4:
        room.act(action)
    assert room.rewards.get('reward-pushed-block', 0) > 0
    room.act('MOVE N 8')
    room.until_room_changes('MOVE E 16')
    assert (room.ending(), tuple(room.info['room'])) == ('exit-correct', (LEVEL, 0x07, True))
    game.restore(enter(probe, 0x06, 'W', walls=WALLS))
    place(game, 0x60, 0x9D, 'S')                            # where the tunnel puts Link (attribute C of 0x07)
    room = Room('darknut', step_of(0x06, 'cave'), game.snapshot())
    assert (room.objective(), room.next_sides()) == ('MOVE', {'W'})


# ---- the crossings after 0x66


@pytest.mark.parametrize('room_id, entry, family, exit_', [
    (0x56, 'S', 'dodongo-crossing', 'E'), (0x57, 'W', 'zol', 'N'), (0x47, 'S', 'gibdo', 'N'),
    (0x37, 'S', 'darknut', 'N'), (0x26, 'E', 'gibdo', 'W'), (0x25, 'E', 'pols-voice', 'W')])
def test_the_crossed_rooms_fight_for_nothing(probe, room_id, entry, family, exit_):
    """0x56 (bombs), 0x57 (5 rupees), 0x47 (a key), 0x37 (the compass), 0x26 (a key) and 0x25: crossed (the owner's
    notes); the item their kills would show is skipped (`treasure: skip`), so neither the monsters nor the item
    are an objective, before or after they die."""
    game = probe.game
    snapshot = enter(probe, room_id, entry, walls=WALLS)
    step = step_of(room_id, entry)
    room = Room(family, step, snapshot)
    assert room.state.enemies and (room.objective(), room.next_sides()) == ('MOVE', {exit_})
    game.restore(snapshot)
    kill_monsters(game)
    room = Room(family, step, game.snapshot())
    assert (room.objective(), room.next_sides()) == ('MOVE', {exit_})


def test_0x27_takes_its_key_without_a_fight_then_leaves_w(probe):
    """5:0x27 (dark; two Pols Voices, two Gibdos, two Keese): the key above the S door is in the open: take it (no
    fight), then the W key door."""
    snapshot = enter(probe, 0x27, 'S', walls=WALLS)
    room = Room('pols-voice', step_of(0x27, 'S'), snapshot)
    assert room.objective() == 'TREASURE'
    game = probe.game
    game.restore(snapshot)
    place(game, game.read(0x83), game.read(0x97), 'N')
    room = Room('pols-voice', step_of(0x27, 'S'), game.snapshot())
    room.act('WAIT 4')
    assert room.rewards.get('reward-gained-keys', 0) > 0
    assert (room.objective(), room.next_sides()) == ('MOVE', {'W'})


def test_a_sword_hit_lets_link_walk_through_a_gibdo_unpunished(probe):
    """The owner's sword-through (0x47, 0x37, 0x25): a hit pays `reward-hit` (no `penalty-attack-miss`) and Link
    walks into the flashing Gibdo without losing health or paying a penalty; the same walk into one not hit costs
    two hearts (`penalty-lost-health`)."""
    snapshot = enter(probe, 0x47, 'S', walls=WALLS)
    step = step_of(0x47, 'S')
    slot = Room('gibdo', step, snapshot).state.enemies[0]['index']
    game = probe.game
    for hit in (True, False):
        game.restore(snapshot)
        game.set_value('hearts_and_containers', 0x96)      # partial health: the sword, no beam
        game.write(0x70 + slot, 0x78)
        game.write(0x84 + slot, 0x95)
        place(game, 0x78, 0xAD, 'N')
        room = Room('gibdo', step, game.snapshot())
        if hit:
            room.act('SWORD N')
            assert room.rewards.get('reward-hit', 0) > 0 and 'penalty-attack-miss' not in room.rewards
        game.restore(room.env.snapshot())
        gibdo = next(e for e in room.state.enemies if e['index'] == slot)
        place(game, gibdo['x'], gibdo['y'], 'N')
        through = Room('gibdo', step, game.snapshot())
        health = through.state.health
        for _ in range(2):
            through.act('MOVE N 4')
        lost = health - through.state.health
        assert (lost, 'penalty-lost-health' in through.rewards) == ((0, False) if hit else (2, True)), hit


# ---- bombs: only in the Olive Garden (down to 0), on Digdogger and at the two walls; the 10th kill's bomb anywhere


@pytest.mark.parametrize('room_id, entry, family, middle, allowed', [
    (0x66, 'S', 'gibdo', (0x80, 0x8D), False), (0x65, 'E', 'gibdo', (0x80, 0x8D), False),
    (0x64, 'E', 'darknut', (0xC0, 0x8D), False), (0x05, 'E', 'darknut', (0x80, 0x8D), True),
    (0x65, 'W', 'gibdo', (0x80, 0x8D), False), (0x66, 'W', 'gibdo', (0x80, 0x8D), False),
    (0x56, 'S', 'dodongo-crossing', (0x78, 0xAD), False), (0x57, 'W', 'zol', (0x40, 0x8D), False),
    (0x47, 'S', 'gibdo', (0x78, 0xBD), False), (0x37, 'S', 'darknut', (0x78, 0xBD), False),
    (0x27, 'S', 'pols-voice', (0x78, 0xAD), False), (0x26, 'E', 'gibdo', (0xC0, 0x8D), False),
    (0x25, 'E', 'pols-voice', (0xC0, 0x8D), False), (0x24, 'E', 'boss-digdogger', (0xC0, 0x8D), True)])
def test_bombs_are_a_weapon_only_in_the_olive_garden_and_on_digdogger(probe, room_id, entry, family, middle, allowed):
    """With 4 bombs, away from any wall: BOMB is offered only in 0x05 (reserve 0, darknut `free`) and 0x24 (reserve 0,
    boss-digdogger `free`); bomb_reserve 8 masks it for the `free` darknut in 0x64 and 0x37, the `required` gibdo
    rooms only offer it at their wall, the rest are `tenth`.  The 10th damage-free kill's bomb is offered in every
    one of them (it arms a forced drop of four)."""
    game = probe.game
    walls = WALLS if (room_id, entry) not in ((0x66, 'S'), (0x65, 'E')) else WALLS[:1] if room_id == 0x65 else []
    snapshot = enter(probe, room_id, entry, bombs=4, recorder=room_id == 0x24, walls=walls)
    step = step_of(room_id, entry)
    game.restore(snapshot)
    place(game, *middle, 'N')
    assert Room(family, step, game.snapshot()).allowed('BOMB') == allowed
    game.write(HELP_COUNT, 9)
    assert Room(family, step, game.snapshot()).allowed('BOMB'), 'the 10th kill'


def test_the_bomb_reserve_along_dungeon_5():
    """routes.yaml `bomb_reserve`: 8 from the entrance (Link holds at most 8: a `free` family never bombs), 0 in the
    Olive Garden, 8 again from its way back, 0 for Digdogger."""
    held, along = 0, []
    for step in steps():
        held = step['bomb_reserve'] if step['bomb_reserve'] is not None else held
        if step['level'] == LEVEL:
            along.append((step['room'], step['entry'], held))
    zero = [(room, entry) for room, entry, reserve in along if reserve == 0]
    assert zero == [(0x05, 'E'), (0x04, 'S'), (0x24, 'E'), (0x14, 'S')]
    assert all(reserve == 8 for room, entry, reserve in along if (room, entry) not in zero)


# ---- Digdogger


def test_digdogger_is_out_of_reach_until_the_recorder(probe):
    """5:0x24: the big Digdogger ($38) is no killable entity and the objective's fight target; the recorder played in
    the room (here by the buttons: the action kind is round 24's `r24-kinds`) splits it at the end of its freeze into
    the small one ($18), killable: the entity's type and killable bit tell the model the room heard the recorder."""
    from triforce._native import ENTITY_FEATURES
    killable = ENTITY_FEATURES.index('killable')
    snapshot = enter(probe, 0x24, 'E', recorder=True, settle=90)
    step = step_of(0x24, 'E')
    room = Room('boss-digdogger', step, snapshot)
    types = list(np.asarray(room.obs['entity_types']))
    assert 0x38 in types and room.obs['entity_features'][types.index(0x38)][killable] == 0
    assert room.objective() == 'FIGHT'
    game = probe.game
    game.restore(snapshot)
    place(game, 0x80, 0x8D, 'W')
    game.set_value('selected_item', 5)          # the recorder's sub-screen slot
    game.press('B', 1)
    for frame in range(400):
        game.press('NONE', 1)
        if any(game.read(OBJ_ID + slot) == 0x18 for slot in range(1, 12)):
            break
    assert 140 <= frame <= 160, 'the split ends the recorder freeze (B + 152)'
    room = Room('boss-digdogger', step, game.snapshot())
    types = list(np.asarray(room.obs['entity_types']))
    assert 0x38 not in types and room.obs['entity_features'][types.index(0x18)][killable] == 1
    assert room.objective() == 'FIGHT'


def test_the_boss_family_plays_the_recorder(probe):
    """boss-digdogger (`items: {recorder: true}`) is offered RECORDER while the big Digdogger hasn't heard it; the
    env plays it (the action kind, through the sub-screen): the play pays `reward-recorder-split` once, is no attack
    miss, the big one splits into the small, killable one, RECORDER is masked from then on, and a sword hit on the
    small one is a boss hit (`reward-boss-hit`)."""
    from triforce._native import ENTITY_FEATURES
    killable = ENTITY_FEATURES.index('killable')
    snapshot = enter(probe, 0x24, 'E', recorder=True, settle=90)
    step = step_of(0x24, 'E')
    room = Room('boss-digdogger', step, snapshot)
    assert not room.allowed('RECORDER'), 'in the doorway no B item is offered'
    room.act('MOVE W 16')
    assert room.allowed('RECORDER')
    room.act('RECORDER')
    assert room.rewards.get('reward-recorder-split') == pytest.approx(0.75), room.rewards
    assert 'penalty-attack-miss' not in room.rewards
    for _ in range(8):
        if 0x18 in list(np.asarray(room.obs['entity_types'])):
            break
        room.act('WAIT 8')
    types = list(np.asarray(room.obs['entity_types']))
    assert 0x38 not in types and room.obs['entity_features'][types.index(0x18)][killable] == 1
    assert not room.allowed('RECORDER') and room.objective() == 'FIGHT' and not room.done
    room.act('WAIT 8')
    assert room.rewards['reward-recorder-split'] == pytest.approx(0.75), 'paid once'

    # The small one in front of Link's sword: a boss hit.
    game = probe.game
    game.restore(room.env.snapshot())
    hit = False
    for _ in range(12):
        slot = next(s for s in range(1, 12) if game.read(OBJ_ID + s) == 0x18)
        x, y = game.read(0x70 + slot), game.read(0x84 + slot)
        place(game, max(x - 14, 0x20), y, 'E')
        attack = Room('boss-digdogger', step, game.snapshot())
        attack.act('SWORD E')
        if 'reward-boss-hit' in attack.rewards:
            assert 'reward-hit' not in attack.rewards
            hit = True
            break
        game.press('NONE', 3)
    assert hit, 'no sword hit on the small Digdogger'
