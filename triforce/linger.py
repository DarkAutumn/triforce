"""Behavior-neutral progress timing for env episodes and runner room visits.

A frozen wavefront measures new best approaches: moving enemies cannot manufacture progress for a stationary
Link, and walking the same loop cannot repeatedly reset the clock. Refresh the distance baseline after a hit,
kill, pickup/healing, room exit or objective phase change; a phase change by itself is not progress.
"""

from .critics import BOSS_KINDS
from .zelda_enums import Direction

_CARDINALS = (Direction.N, Direction.S, Direction.E, Direction.W)


class LingerTracker:
    """Counts actions and actual elapsed frames, never rewards or terminates gameplay."""

    def reset(self, state):
        self.steps = self.frames = 0
        self.max_steps = self.max_frames = 0
        self.tail_steps = self.tail_frames = 0
        self.boss_kill_steps = self.boss_kill_frames = None
        self.heart_container_steps = self.heart_container_frames = None
        self.start_room = (state.level, state.location, state.in_cave)
        self._baseline(state)

    def _baseline(self, state):
        objective = state.objectives
        self._phase = (state.full_location, objective.kind, frozenset(objective.next_rooms))
        self._has_targets = bool(objective.pbrs_targets)
        self._wavefront = None
        self._best = None
        if not state.game_over and self._has_targets:
            self._wavefront = getattr(state, 'pbrs_wavefront', None)
            if self._wavefront is None:
                locked = frozenset(d for d in _CARDINALS if state.is_door_locked(d)) \
                    if state.level > 0 and state.link.keys > 0 else None
                self._wavefront = state.room.calculate_wavefront_for_link(objective.pbrs_targets,
                                                                         locked_doors=locked or None)
            self._best = self._wavefront.get(state.link.tile)

    def update(self, change, hearts_taken=0):
        prev, curr = change.previous, change.state
        self.steps += 1
        self.frames += change.frames
        changed_room = prev.full_location != curr.full_location
        pickup = bool(change.items_gained or hearts_taken or change.gained_triforce
                      or curr.link.keys > prev.link.keys or curr.link.max_health > prev.link.max_health)
        # Health gained includes pond/fairy refills; taking damage never resets the progress clock.
        tangible = changed_room or pickup or bool(change.enemies_hit) or change.health_gained > 0 \
            or curr.room_kills > prev.room_kills
        if not changed_room:
            if self.boss_kill_steps is None and any(e.id in BOSS_KINDS and not e.is_dying for e in prev.enemies) \
                    and not any(e.id in BOSS_KINDS and not e.is_dying for e in curr.enemies):
                self.boss_kill_steps, self.boss_kill_frames = self.steps, self.frames
            if self.heart_container_steps is None and curr.link.max_health > prev.link.max_health:
                self.heart_container_steps, self.heart_container_frames = self.steps, self.frames

        phase = (curr.full_location, curr.objectives.kind, frozenset(curr.objectives.next_rooms))
        approaching = False
        if not tangible and phase == self._phase and self._wavefront is not None \
                and curr.link.tile != prev.link.tile:
            distance = self._wavefront.get(curr.link.tile)
            if distance is not None and (self._best is None or distance < self._best):
                self._best = distance
                approaching = True
        if tangible or approaching:
            self.tail_steps = self.tail_frames = 0
        else:
            self.tail_steps += 1
            self.tail_frames += change.frames
            self.max_steps = max(self.max_steps, self.tail_steps)
            self.max_frames = max(self.max_frames, self.tail_frames)
        if tangible or phase != self._phase or (not self._has_targets and curr.objectives.pbrs_targets):
            self._baseline(curr)

    def summary(self):
        return {name: getattr(self, name) for name in (
            'max_steps', 'max_frames', 'tail_steps', 'tail_frames', 'steps', 'frames',
            'boss_kill_steps', 'boss_kill_frames', 'heart_container_steps', 'heart_container_frames')}
