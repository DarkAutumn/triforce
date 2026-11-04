"""Confirmed shield collisions from consecutive *executed* NES frames, not the delayed audio state.

Z_01 CheckLinkCollision (5598–5715), Z_04 CheckShotLinkCollision / UpdateFireball / UpdateMonsterArrow,
Z_07 DrawBoomerangAndCheckCollision: clinks also come from monster immunity, and a returning boomerang
cannot hurt Link. Require the projectile's collision signature, the shield eligibility/facing/proximity
checks and this frame's parry request ($604 bit 0). $605 is the sound already playing, a frame too late.
"""

from dataclasses import dataclass

from .zelda_enums import MapLocation

# $56 and $5A cannot be shielded, even with the magic shield.
_BASIC_SHOTS = frozenset({0x53, 0x54, 0x5B, 0x5C})
_MAGIC_SHOTS = frozenset({0x55, 0x57, 0x58, 0x59})
_BLOCKABLE_SHOTS = _BASIC_SHOTS | _MAGIC_SHOTS


@dataclass(frozen=True)
class ProjectileBlock:
    """One actual collision; frame is absolute emulator time, slot is the hostile projectile slot."""
    frame: int
    slot: int
    projectile_id: int
    full_location: MapLocation


class ShieldCollisionWatch:
    """Only snapshot outgoing blockable shots; never retain emulator RAM views across frames."""
    def __init__(self, read):
        self.shots = []
        for slot in range(1, 13):  # monster slots; player's weapons start at $0D
            kind = read(0x34F + slot)
            if kind in _BLOCKABLE_SHOTS and read(0xAC + slot) & 0xF0 == 0x10:
                direction = read(0x98 + slot)
                x, y = read(0x70 + slot), read(0x84 + slot)
                can_collide = True
                if kind == 0x55 and not read(0x28 + slot):
                    # Fireball1 also disappears when BoundByRoom rejects it *before* movement/collision.
                    # Do not mistake that disappearance plus an unrelated immune/Darknut clink for a block.
                    horizontal = 2 if x < read(0x346) else 1 if x >= read(0x347) else 0
                    vertical = 8 if y < read(0x348) else 4 if y >= read(0x349) else 0
                    can_collide = not direction & (horizontal | vertical)
                self.shots.append((slot, kind, direction, read(0x4BF + slot), x, y, can_collide))
        if self.shots:
            self.world = (read(0x10), read(0xEB), read(0x12))
            self.health = (read(0x66F), read(0x670))

    def collisions(self, read, frame: int) -> tuple[ProjectileBlock, ...]:
        """Match a state-changing collision, not disappearance, natural return, or sustained clink audio."""
        if not self.shots or not read(0x604) & 1:
            return ()
        level, room, mode = self.world
        if mode not in (5, 11) or self.world != (read(0x10), read(0xEB), read(0x12)):
            return ()
        # These are the RAM conditions used when the game actually checks the collision, after Link updates.
        # Health/invincibility also reject harmful collisions that use the same boomerang spark signature.
        if read(0x66C) or read(0x4F0) or read(0x3D) or read(0x512) \
                or read(0xAC) & 0xF0 or (read(0x66F), read(0x670)) < self.health:
            return ()
        facing = read(0x98)
        lx, ly = int(read(0x70)), int(read(0x84))
        events = []
        for slot, kind, direction, attributes, old_x, old_y, can_collide in self.shots:
            if not can_collide or read(0x3D + slot):
                continue
            combined = facing | direction
            if combined & 0x0C != 0x0C and combined & 0x03 != 0x03:
                continue
            if kind in _MAGIC_SHOTS and not read(0x676):
                continue

            current_kind, phase = read(0x34F + slot), read(0xAC + slot)
            if kind == 0x5C:
                # Shield and harmful boomerang hits spark $20 with counter 3; natural returns are $30/$40/$50.
                # A tile/boundary spark leaves the outgoing shot stationary and skips the Link collision check.
                signature = current_kind == kind and phase == 0x20 and read(0x3D0 + slot) == 3 \
                    and (read(0x70 + slot), read(0x84 + slot)) != (old_x, old_y)
            elif kind == 0x55:
                # Fireball1 is destroyed after *any* Link collision. Its collision routine first stamps the
                # shield direction and resets bounce distance; DestroyObject leaves both and the position intact.
                signature = current_kind == 0 and phase == 0 \
                    and read(0x380 + slot) == facing and read(0x394 + slot) == 0
            else:
                # Rocks/arrows/sword and magic shots enter bounce $30 only on a Link collision; harmful ones
                # are then destroyed instead. No wall-hit spark or disappearance is enough.
                signature = current_kind == kind and phase == 0x30 \
                    and read(0x380 + slot) == facing and read(0x394 + slot) == 0
            if not signature:
                continue

            # GetObjectMiddle uses a half-width X offset when ObjAttr bit $40 is set. The collision threshold
            # is strictly <9 on each axis, not a Euclidean "near projectile" radius. Match 8-bit subtraction.
            dx = (int(read(0x70 + slot)) + (4 if attributes & 0x40 else 8) - lx - 8) & 0xFF
            dy = (int(read(0x84 + slot)) - ly) & 0xFF
            if min(dx, 256 - dx) >= 9 or min(dy, 256 - dy) >= 9:
                continue
            events.append(ProjectileBlock(frame, slot, kind, MapLocation(level, room, mode == 11)))
        return tuple(events)
