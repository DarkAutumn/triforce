"""The class that represents an enemy in the game state."""

from dataclasses import dataclass
from typing import Tuple
from .zelda_enums import Direction, ZeldaEnemyKind
from .zelda_objects import ZeldaObject

ENEMY_INVULNERABLE = 0x100

@dataclass
class Enemy(ZeldaObject):
    """Structured data for an enemy."""
    direction : Direction
    health : int
    stun_timer : int
    spawn_state : int
    status : int

    def __eq__(self, value):
        return self.id == value or super().__eq__(value)

    @property
    def dimensions(self) -> Tuple[int, int]:
        """The dimensions of the object."""
        if self.id == ZeldaEnemyKind.Aquamentus:
            return 3, 3

        return 2, 2
    @property
    def is_dying(self) -> bool:
        """Returns True if the enemy has been dealt enough damage to die, but hasn't yet been removed
        from the game state.

        Death sparkle sequence: metastate $10(1f) -> $11(6f) -> $12(6f) -> $13(6f) -> $14.
        At $14, the object becomes a dropped item (type=$60) in the same frame, so $14 is
        never directly observed.  Observable range: $10-$13 (16-19).
        """
        return 16 <= self.spawn_state <= 19

    @property
    def is_active(self) -> bool:
        """Returns True if the enemy is 'active', meaning it is targetable by link.  This used for enemies like the
        lever or zora, which peroidically go underground/underwater, or the wallmaster which disappears behind the
        wall.  An enemy not active cannot take or deal damage to link by touching him."""
        status = self.status & 0xff

        # Leevers are only targetable when fully surfaced (state 3).
        if self.id in (ZeldaEnemyKind.RedLeever, ZeldaEnemyKind.BlueLeever):
            return status == 3

        # Zora is targetable in states 2, 3, and 4 (partially/fully surfaced).
        # Assembly: UpdateBurrower checks collisions for Zora at states 2, 3, 4.
        if self.id == ZeldaEnemyKind.Zora:
            return 2 <= status <= 4

        # WallMaster state 1 means on-screen and moving.
        if self.id == ZeldaEnemyKind.Wallmaster:
            return status == 1

        # Default: spawn_state of 0 means the object is fully active.
        return not self.spawn_state

    @property
    def is_stunned(self) -> bool:
        """Returns True if the enemy is stunned."""
        return self.stun_timer > 0
