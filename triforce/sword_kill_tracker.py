"""Temporary experiment: tracks max sword (non-beam) kill distance.

On each step, saves the distance from Link to every active enemy. On the next step, if an
enemy died, records the *previous* step's distance to hit-distance.txt — but only when the
distance exceeds the running maximum. Any frame where beam animation is playing causes the
tracker to skip entirely, ensuring we never attribute a beam kill to the sword.
"""

from typing import Dict, Tuple

from .zelda_enums import AnimationState, ZeldaAnimationKind, ZeldaEnemyKind


class SwordKillDistanceTracker:
    """Writes "{EnemyType} {distance}" lines to a file for sword-only kills."""

    def __init__(self, output_path="hit-distance.txt"):
        self._output_path = output_path
        self._prev_distances: Dict[int, Tuple[str, float]] = {}
        self._max_written_distance: float = -1.0

    def reset(self):
        """Called on environment reset."""
        self._prev_distances.clear()

    def step(self, state_change):
        """Called each step with the StateChange object."""
        prev = state_change.previous
        curr = state_change.state

        # Room changed — enemies are from a different room, discard saved distances.
        if state_change.changed_location:
            self._prev_distances.clear()
            return

        # If beam animation is playing on either frame, clear and skip.
        prev_beams = prev.link.get_animation_state(ZeldaAnimationKind.BEAMS)
        curr_beams = curr.link.get_animation_state(ZeldaAnimationKind.BEAMS)
        if prev_beams != AnimationState.INACTIVE or curr_beams != AnimationState.INACTIVE:
            self._prev_distances.clear()
            return

        # Detect kills: enemy was alive in prev, now dying or gone in curr.
        if self._prev_distances:
            for enemy in prev.enemies:
                if enemy.is_dying:
                    continue

                curr_enemy = curr.get_enemy_by_index(enemy.index)
                killed = curr_enemy is None or curr_enemy.is_dying

                if killed and enemy.index in self._prev_distances:
                    enemy_type, distance = self._prev_distances[enemy.index]
                    if distance > self._max_written_distance:
                        self._write(enemy_type, distance)
                        self._max_written_distance = distance

        # Save current distances for next step.
        self._prev_distances.clear()
        for enemy in curr.enemies:
            if not enemy.is_dying and enemy.is_active:
                type_name = enemy.id.name if isinstance(enemy.id, ZeldaEnemyKind) else str(enemy.id)
                self._prev_distances[enemy.index] = (type_name, float(enemy.distance))

    def _write(self, enemy_type: str, distance: float):
        """Append a single line to the output file."""
        with open(self._output_path, "a", encoding="utf-8") as f:
            f.write(f"{enemy_type} {distance:.1f}\n")
