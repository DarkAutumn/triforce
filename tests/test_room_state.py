"""Room-state decoding (docs/ram-catalog.md §R) on real savestates, including puzzle transitions."""

from triforce.actions import Action
from triforce.room_state import DoorType, SecretTrigger, decode_room_state, room_flags
from triforce.zelda_enums import Direction, SwordKind

N, S, W, E = Direction.N, Direction.S, Direction.W, Direction.E
FULL_HEALTH = {'hearts_and_containers': 0xFF, 'partial_hearts': 0xFF}


def _doors(rs):
    return {d.name: (door.type, door.is_open) for d, door in rs.doors.items()}


def test_goriya_room_doors_trigger_and_hidden_item(session):
    rs = decode_room_state(session.load('1_44w', randomize_rng=False))
    assert (rs.level, rs.room, rs.in_cave) == (1, 0x44, False)
    # ram-catalog R.9: room $44 = wall, bomb, key, open; entered through the (now unlocked) W key door.
    assert _doors(rs) == {'N': (DoorType.WALL, False), 'S': (DoorType.BOMB, False),
                          'W': (DoorType.KEY, True), 'E': (DoorType.OPEN, True)}
    assert rs.door(S).kind == 'bomb' and rs.open_doors == {W, E}
    assert rs.secret_trigger == SecretTrigger.ALL_DEAD_ITEM
    assert rs.has_item and rs.item_id == 0x1D and not rs.item_visible and rs.item_hidden_until_clear
    assert rs.item_position == (0x80, 0x70)
    assert rs.enemies_remaining == 3 and rs.enemies_unspawned == 0 and not rs.all_dead
    assert rs.visited and not rs.item_taken and rs.kill_state == 0


def test_key_rooms_hidden_and_carried(session):
    rs = decode_room_state(session.load('1_72e', randomize_rng=False))
    assert rs.item_id == 0x19 and rs.item_position == (0xA0, 0xC0) and rs.item_hidden_until_clear
    assert rs.enemies_remaining == 3 and rs.enemies_unspawned == 3       # still in their spawn clouds

    state = session.load('1_74w', randomize_rng=False)
    rs = decode_room_state(state)
    assert rs.item_id == 0x19 and rs.item_visible and rs.item_carried and not rs.item_hidden_until_clear
    assert rs.item_position == (int(state.ram[0x71]), int(state.ram[0x85]))     # drawn on the Stalfos in slot 1


def test_cleared_boss_room_and_persisted_flags(session):
    state = session.load('1_35e', randomize_rng=False)
    rs = decode_room_state(state)
    assert rs.is_boss_room and not rs.is_triforce_room
    assert rs.door(S) == rs.doors[S] and rs.doors[S].type == DoorType.KEY and rs.doors[S].is_open
    assert rs.doors[E].type == DoorType.SHUTTER and rs.doors[E].is_open        # CurOpenedDoors only
    assert rs.item_taken and not rs.has_item and rs.kill_state == 3 and rs.all_dead
    # Observed flag bytes listed in ram-catalog R.1.
    assert [room_flags(state, r) for r in (0x74, 0x53, 0x35, 0x52, 0x43)] == [0xB0, 0xF0, 0xF4, 0x68, 0x21]
    assert room_flags(state, state.location) == rs.flags


def test_push_block_progress(session):
    session.load('1_22e', randomize_rng=False, ram_overrides={'link_x': 96, 'link_y': 125})
    rs = decode_room_state(session.state)
    assert rs.has_push_block and rs.push_block_state == 'pending' and rs.block_push_complete == 0
    assert rs.doors[E].type == DoorType.KEY and rs.doors[E].is_open
    session.act(Action.parse('MOVE S 32'))
    assert decode_room_state(session.state).push_block_state in ('moving', 'done')
    session.act(Action.parse('WAIT 32'))
    rs = decode_room_state(session.state)
    assert rs.push_block_state == 'done' and rs.block_push_complete >= 1


def test_bombing_the_south_wall_opens_it(session):
    overrides = dict(FULL_HEALTH, link_x=120, link_y=189, link_direction=4)
    session.load('1_44w', randomize_rng=False, ram_overrides=overrides)
    session.act(Action.parse('BOMB S'))
    opened = False
    for _ in range(10):
        session.act(Action.parse('WAIT 16'))
        door = decode_room_state(session.state).doors[S]
        assert door.type == DoorType.BOMB
        if door.is_open:
            opened = True
            break
    assert opened


def test_sword_cave_ware(session):
    state = session.load('0_77cave', randomize_rng=False)
    rs = decode_room_state(state)
    assert rs.in_cave and rs.is_overworld and rs.doors == {}
    assert rs.has_item and rs.item_id == 0x01 and rs.item_visible and rs.item_position == (0x78, 0x98)
    assert [w.item_id for w in rs.cave_wares] == [0x01] and rs.cave_wares[0].price == 0
    assert rs.enemies_remaining == 0                 # the old man's fires are not foes

    for _ in range(30):
        cmd = 'MOVE E 4' if session.state.link.position.x < 0x78 else 'MOVE N 8'
        session.act(Action.parse(cmd))
        rs = decode_room_state(session.state)
        if not rs.has_item:
            break
    assert not rs.has_item and rs.item_taken and session.state.link.sword == SwordKind.WOOD


def test_overworld_rooms(session):
    rs = decode_room_state(session.load('start', randomize_rng=False))
    assert (rs.level, rs.room) == (0, 0x77) and rs.doors == {} and not rs.has_item and not rs.item_taken
    assert not rs.has_push_block and rs.secret_trigger == SecretTrigger.NONE

    rs = decode_room_state(session.load('0_5fs', randomize_rng=False))
    assert rs.item_id == 0x1A and rs.item_visible and rs.item_position == (0xC0, 0x90)


def test_map_and_compass_use_the_level_bit(session):
    state = session.load('2_7dn', randomize_rng=False, ram_overrides={'map': 0x01, 'compass': 0x02})
    rs = decode_room_state(state)
    assert rs.level == 2
    assert not rs.has_map and rs.has_compass               # D1's map bit does not count in D2
    assert not state.link.map and state.link.compass

    rs = decode_room_state(session.load('1_42e', randomize_rng=False))
    assert rs.has_map and not rs.has_compass
