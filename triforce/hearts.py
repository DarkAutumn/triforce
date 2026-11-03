"""Heart drop tracking: which dropped hearts Link picks up, which he leaves behind, and what they healed.

Dropped items live in monster slots 1-$B (``ZeldaGame.items``); a heart is item id ``ZeldaItemKind.Heart`` (0x22).
Fairies, triforce pieces and heart containers heal too but are not counted here.  An item's lifetime
(``Item.timer``, RAM $3A8+slot) is set to $FF on drop and counts down once every other frame; the item is destroyed
when it reaches 0 (Z_04 UpdateItem).  So when a tracked heart is gone from the floor after a step:

- Link left the room or died (``state.game_over``): ``left``;
- its last seen timer was larger than the step's frames could count down: Link (or his sword, arrow or boomerang)
  took it: ``taken``;
- otherwise it may have timed out: ``expired``, unless Link's health rose this step, then ``taken``.

A slot whose heart timer jumps up held an old heart (gone) and now holds a new drop.  A heart heals
``min(1, max_health - health)`` (Z_01 TakeHearts adds a whole heart, or fills the partial one when Link is less than
a heart from full); several hearts taken in one step heal in sequence.  The rest of each taken heart is ``wasted``
(a heart picked up at full health wastes 1), so the "total" heart change of an episode is the real health change
plus ``wasted``.  Hearts on the floor at ``reset`` count as dropped; ``finish`` counts the hearts still on the floor
as ``left``.
"""

from typing import Optional

from .zelda_enums import ZeldaItemKind

# Lifetime ticks a step may count down beyond frames/2 before a vanished heart counts as possibly expired.
TIMER_MARGIN = 2


class HeartTracker:
    """Follows dropped hearts through one room visit (an env episode, or one runner room record)."""
    # pylint: disable=too-many-instance-attributes

    def __init__(self):
        self.dropped = 0
        self.taken = 0
        self.healed = 0.0
        self.left = 0
        self.expired = 0
        self._floor: dict[int, int] = {}     # slot -> last seen lifetime timer
        self._location = None
        self._health = 0.0
        self._open = False                   # between reset() and finish()

    def reset(self, state):
        """Starts tracking in ``state``'s room; hearts already on the floor count as dropped."""
        self.dropped = self.taken = self.left = self.expired = 0
        self.healed = 0.0
        self._location = state.full_location
        self._health = state.link.health
        self._floor = {} if state.game_over else _hearts(state)
        self.dropped = len(self._floor)
        self._open = True

    def update(self, change):
        """Follows one ``GameSession.act`` step (``change.state`` must be the current, active state).  Leaving the
        room or dying leaves the hearts on the floor; tracking goes on in the new room."""
        if not self._open:
            return
        state = change.state
        if state.game_over or state.full_location != self._location:
            self.left += len(self._floor)
            self._floor = {}
            self._location = state.full_location
            self._health = state.link.health
            return

        current = _hearts(state)
        reach = change.frames // 2 + TIMER_MARGIN
        taken, maybe = [], []
        for slot, timer in self._floor.items():
            now = current.get(slot)
            if now is None or now > timer:
                (taken if timer > reach else maybe).append(slot)

        health = self._health
        max_health = state.link.max_health
        for _ in taken:
            health = self._take(health, max_health)
        for _ in maybe:
            if state.link.health > health + 1e-6:
                health = self._take(health, max_health)
            else:
                self.expired += 1

        self.dropped += sum(1 for slot, timer in current.items()
                            if slot not in self._floor or timer > self._floor[slot])
        self._floor = current
        self._health = state.link.health

    def finish(self, state=None):
        """Ends the visit (until the next ``reset``): hearts still on the floor count as ``left``.  With ``state``
        (the current state, still in the tracked room) hearts there that ``update`` never saw count as dropped
        first."""
        if not self._open:
            return
        if state is not None and not state.game_over and state.full_location == self._location:
            for slot, timer in _hearts(state).items():
                if slot not in self._floor or timer > self._floor[slot]:
                    self.left += slot in self._floor     # the slot's old heart, replaced by a new drop
                    self.dropped += 1
                    self._floor[slot] = timer
        self.left += len(self._floor)
        self._floor = {}
        self._open = False

    def summary(self) -> dict:
        """``dropped``, ``taken``, ``healed`` (real hearts gained), ``wasted`` (taken - healed), ``left``,
        ``expired``."""
        return {'dropped': self.dropped, 'taken': self.taken, 'healed': self.healed,
                'wasted': self.taken - self.healed, 'left': self.left, 'expired': self.expired}

    def _take(self, health: float, max_health: float) -> float:
        heal = max(0.0, min(1.0, max_health - health))
        self.taken += 1
        self.healed += heal
        return health + heal


def _hearts(state) -> dict[int, int]:
    return {item.index: int(item.timer) for item in state.items if item.id == ZeldaItemKind.Heart}


def heart_change_total(health_change: float, hearts: Optional[dict]) -> float:
    """Real health change plus the hearts wasted by picking them up at (or near) full health."""
    return health_change + (hearts or {}).get('wasted', 0.0)
