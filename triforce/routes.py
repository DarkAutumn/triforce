# Defines the route that models should take through the game

from .zelda_game import get_num_triforce_pieces

def _build_walk(level, rooms):
    return [(level, room) for room in rooms]

DUNGEON_ENTRANCES = {1 : 0x73, 2:0x7d}

DANGEROUS_ROOMS = [
    (0, 0x38),   # Octorok with Zora room
    (1, 0x45),   # Wallmaster room
]

ROOMS_WITH_REVEALED_TREASURE = [
    (1, 0x72),   # keese room with key
    (1, 0x53),   # stalfos with key
    (1, 0x23),   # gojira with key
    (1, 0x44),   # gojira with boomerang
    (1, 0x35),   # boss room,
    (2, 0x7e),   # rope room with key
    (2, 0x4e),   # rope room with key
    (2, 0x4f),   # gojira with magic boomerang
    (2, 0x3e),   # moldorm room with key,
    (2, 0x3f),   # keese with bombs
    (2, 0x2e),   # rope room with lockout
    (2, 0x1e),   # gojira with bombs
    (2, 0x0e),   # boss room,
]

ROOMS_WITH_TREASURE = [
    (1, 0x74),   # stalfos room with key
    (1, 0x45),   # wallmaster with key
    (1, 0x36),   # triforce room
    (1, 0x33),   # stalfos with key
]

ROOMS_WITH_PUSHBLOCKS = [
    (1, 0x22),   # bow room
]

CAVES_WITH_TREASURE = {
    (0, 0x77) : "sword",
    (1, 0x7f) : "bow",
}

DUNGEON1_WALK = [
    (1, 0x73),   # entrance to dungeon 1
    (1, 0x72),   # keese room with key
    (1, 0x73),   # entrance to dungeon 1
    (1, 0x74),   # stalfos room with key
    (1, 0x73),   # entrance to dungeon 1
    (1, 0x63),   # stalfos
    (1, 0x53),   # stalfos with key
    (1, 0x52),   # keese
    (1, 0x42),   # gel
    (1, 0x43),   # gel with map
#    (1, 0x33),   # stalfos with key
#    (1, 0x23),   # gojira with key
#    (1, 0x22),   # trap room
#    (1, 0x7f),   # bow room
#    (1, 0x22),   # trap room
#    (1, 0x23),   # gojira with key
#    (1, 0x33),   # stalfos with key
#    (1, 0x43),   # gel with map
    (1, 0x44),   # gojira with boomerang
    (1, 0x45),   # wallmaster with key
    (1, 0x35),   # boss room
    (1, 0x36),   # triforce room
    (0, 0x37),   # overworld
]

DUNGEON2_WALK = [
    (2, 0x7d),   # entrance to dungeon 2
    (2, 0x7e),   # rope room with key
    (2, 0x6e),   # rope room
    (2, 0x5e),   # goriya room
    (2, 0x4e),   # rope room with key
    (2, 0x3e),   # moldorm room with key,
    (2, 0x2e),   # rope room with lockout
    (2, 0x1e),   # gojira with bombs
    (2, 0x0e),   # boss room
    (2, 0x0d),   # triforce room
    (0, 0x3c),   # overworld
]

OVERWORLD_SWORD_WALK = [(0, 0x77), (0, 0x67)]
OVERWORLD1_WALK = _build_walk(0, [0x77, 0x67, 0x68, 0x58, 0x48, 0x38, 0x37]) + \
                    [(1, DUNGEON_ENTRANCES[1]), (1, DUNGEON_ENTRANCES[1] - 1)]

OVERWORLD2_WALK = _build_walk(0, [0x37, 0x38, 0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x3d, 0x4d, 0x4c, 0x3c]) \
                    + [(2, DUNGEON_ENTRANCES[2]), (2, DUNGEON_ENTRANCES[2] + 1)]

OVERWORLD2A_WALK = _build_walk(0, [0x37, 0x38, 0x48, 0x49, 0x59, 0x5a, 0x5b, 0x5c, 0x5d, 0x4d, 0x4c, 0x3c]) \
                    + [(2, DUNGEON_ENTRANCES[2]), (2, DUNGEON_ENTRANCES[2] + 1)]

def get_walk(info):
    """Returns the series of rooms that the agent should walk through based on the current game state."""
    triforce = get_num_triforce_pieces(info)
    match triforce:
        case 0:
            if info['level'] == 0:
                return OVERWORLD1_WALK

            if info['level'] == 1:
                return DUNGEON1_WALK

            raise NotImplementedError("No walk defined for this scenario")

        case 1:
            loc = (info['level'], info['location'])
            if loc in OVERWORLD2_WALK:
                return OVERWORLD2_WALK

            if loc in OVERWORLD2A_WALK:
                return OVERWORLD2A_WALK

            raise NotImplementedError("No walk defined for this scenario")

        case _:
            raise NotImplementedError("No walk defined for this scenario")

def get_next_room(info, walk=None):
    """Returns the next room that the agent should walk to based on the current game state."""
    if walk is None:
        walk = get_walk(info)

    loc = (info['level'], info['location'])
    idx = walk.index(loc)
    return walk[idx + 1] if idx + 1 < len(walk) else None

def get_target_room(info, walk=None):
    """Returns the target room that the agent should walk to based on the current game state."""
    if walk is None:
        walk = get_walk(info)

    return walk[-1]
