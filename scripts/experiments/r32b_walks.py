"""Round 32's walks of `d3first`'s second half, from Level 5 to the ending (docs/routes/d3first-walks-b.md): the
helpers the walk scripts share, on top of rounds 26-29's (`r26_walks.Walk`, `r27_walks.Tally`, `stage_dungeon`).

Staging (nothing here is how the game is played: it writes the inventory, the whirlwind index and room flags):

- **Out of a dungeon:** `triforce.probe` puts Link into the level's triforce room with what `d3first` holds there
  (`D3`), he walks onto the piece and **the game itself** ends the level and sets him down at its entrance
  (`out_of`): his first playable frame, facing down, full hearts.
- **On the overworld:** from such an exit, or from a room's first frame, Link walks with the monster slots emptied
  every frame (`Walk.staging`) to the room under study, whose scroll is saved as its pre-entry state; every trial
  loads one with the game's own randomness for its seed (`r27_walks.load`).

`d3first`'s inventory by milestone (the owner's list, docs/routes/d3first-walks-b.md §0): the wood sword until the
grave; the boomerang and bow from Dungeon 1, the raft (3), ladder (4), blue candle (0x0c, before Level 5), the
recorder (5), arrows and meat after Level 2, the red candle (7), the magic sword (the grave, 12 containers).  Keys,
bombs and rupees are the caller's where a walk depends on them (the ledgers of §7 say what a run brings).

Addresses are the disassembly's (zelda-asm/src/Variables.inc)."""

from r26_walks import (BOMBS, KEYS, OBJ_DIR, PLAY, RUPEES, TELEPORT_INDEX, TRIFORCE, Walk, stage_dungeon)
from r27_walks import StagingWalk, hearts_bytes

INV_SWORD, INV_CANDLE, INV_CONTAINERS = 0x657, 0x65B, 0x66F

# Triforce bits: level n is bit n - 1 (asm:InvTriforce).
L1, L2, L3, L4, L5, L6, L7, L8 = (1 << i for i in range(8))


def _inv(containers: int, sword: int, pieces: int, **extra) -> dict:
    base = {'sword': sword, 'regular_boomerang': 1, 'bow': 1, 'arrows': 0, 'raft': 1, 'step_ladder': 1,
            'whistle': 1, 'food': 0, 'candle': 1, 'triforce': pieces, 'keys': 2, 'bombs': 4, 'rupees': 20}
    base.update(hearts_bytes(containers, containers))
    base.update(extra)
    return base


# What `d3first` holds on reaching each dungeon's triforce piece (the piece's own bit not yet set), the whirlwind
# index the route's earlier plays leave (§1), and the triforce room and the button that walks onto the piece.
D3 = {
    5: dict(inv=_inv(8, 1, L1 | L3 | L4), index=0, room=0x14, entry='S', button='UP'),
    2: dict(inv=_inv(11, 1, L1 | L3 | L4 | L5, rupees=140), index=2, room=0x0D, entry='E',
            button=[('x', 0xD0), ('y', 0xBD), ('x', 0x78)]),
    7: dict(inv=_inv(12, 1, L1 | L2 | L3 | L4 | L5, candle=2, arrows=1), index=3, room=0x2B, entry='W', button=None),
    6: dict(inv=_inv(13, 3, L1 | L2 | L3 | L4 | L5 | L7, candle=2, arrows=1), index=3, room=0x0C, entry='S',
            button='UP'),
    8: dict(inv=_inv(14, 3, 0x7F, candle=2, arrows=1), index=1, room=0x2C, entry='S', button='UP'),
}


def out_of(level: int, inventory: dict = None, index: int = None) -> Walk:
    """Link on the overworld at `level`'s entrance, put out by the game after its piece: his first playable frame.
    `inventory` / `index` replace the milestone's (`D3`)."""
    spec = D3[level]
    inv = dict(spec['inv'] if inventory is None else inventory)
    idx = spec['index'] if index is None else index
    probe_walk, _, _ = stage_dungeon(level, spec['room'], spec['entry'], inventory=inv, ram={TELEPORT_INDEX: idx})
    w = StagingWalk(game=probe_walk.game)
    if spec['button'] is None:                                   # 7:0x2b: round the block ring onto the piece
        if w.path(('x', 0x20), ('y', 0xBD), ('x', 0x78)) < 0:
            raise RuntimeError('7:0x2b: the walk to the piece failed at ' + w.where())
        w.until(lambda w: w.mode != PLAY, 'UP', 300)
    elif isinstance(spec['button'], list):                       # 2:0x0d: under the block ring, then up its gap
        if w.path(*spec['button']) < 0:
            raise RuntimeError(f'{level}:0x{spec["room"]:02x}: the walk to the piece failed at ' + w.where())
        w.until(lambda w: w.mode != PLAY, 'UP', 300)
    else:
        w.until(lambda w: w.mode != PLAY, spec['button'], 400)
    if w.until(lambda w: w.mode == PLAY and w.level == 0, 'NONE', 3000) < 0:
        raise RuntimeError(f'the D{level} piece did not end the level: ' + w.where())
    return w


def ledger_row(w: Walk) -> dict:
    return {'where': w.where(), 'hearts': w.hearts(), 'max': w.max_hearts(), 'sword': w.r(INV_SWORD),
            'candle': w.r(INV_CANDLE), 'triforce': f'{w.r(TRIFORCE):#04x}', 'index': w.r(TELEPORT_INDEX),
            'keys': w.r(KEYS), 'bombs': w.r(BOMBS), 'rupees': w.r(RUPEES), 'facing': w.r(OBJ_DIR)}
