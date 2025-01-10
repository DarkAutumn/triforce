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

    def mark_invulnerable(self):
        """Marks the enemy as invulnerable."""
        self.status |= ENEMY_INVULNERABLE

    @property
    def dimensions(self) -> Tuple[int, int]:
        """The dimensions of the object."""
        if self.id == ZeldaEnemyKind.AquaMentus:
            return 3, 3

        return 2, 2
    @property
    def is_dying(self) -> bool:
        """Returns True if the enemy has been dealt enough damage to die, but hasn't yet been removed
        from the game state."""

        # This is through trial and error.  The dying state of enemies seems to start and 16 and increase
        # by one frame until they are removed.  I'm not sure if it ends with 19 or not.
        return 16 <= self.spawn_state <= 19

    @property
    def is_active(self) -> bool:
        """Returns True if the enemy is 'active', meaning it is targetable by link.  This used for enemies like the
        lever or zora, which peroidically go underground/underwater, or the wallmaster which disappears behind the
        wall.  An enemy not active cannot take or deal damage to link by touching him."""
        status = self.status & 0xff

        # status == 3 means the lever/zora is up
        if self.id in (ZeldaEnemyKind.RedLever, ZeldaEnemyKind.BlueLever, ZeldaEnemyKind.Zora):
            return status & 0xff == 3

        # status == 1 means the wallmaster is active
        if self.id == ZeldaEnemyKind.WallMaster:
            return status == 1

        # spawn_state of 0 means the object is active
        return not self.spawn_state

    @property
    def is_stunned(self) -> bool:
        """Returns True if the enemy is stunned."""
        return self.stun_timer > 0

    @property
    def is_invulnerable(self) -> bool:
        """Returns True if the enemy is invulnerable and cannot take damage."""
        return not self.is_active or self.status & ENEMY_INVULNERABLE == ENEMY_INVULNERABLE
