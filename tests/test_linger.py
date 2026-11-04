"""Progress timing invariants, with synthetic distances and no emulator."""

from types import SimpleNamespace as NS

from triforce.linger import LingerTracker
from triforce.objectives import ObjectiveKind
from triforce.zelda_enums import ZeldaEnemyKind


def state(tile=5, target=0, kind=ObjectiveKind.MOVE, room=0x35, health=3, kills=0, boss=False,
          max_health=3, game_over=False):
    objective = NS(kind=kind, next_rooms={0x36}, pbrs_targets={target})
    enemy = NS(id=ZeldaEnemyKind.Aquamentus, is_dying=False)
    return NS(level=1, location=room, in_cave=False, full_location=(1, room, False), game_over=game_over,
              objectives=objective, pbrs_wavefront={t: abs(t - target) for t in range(20)},
              link=NS(tile=tile, keys=0, health=health, max_health=max_health), room_kills=kills,
              enemies=[enemy] if boss else [])


def update(tracker, prev, curr, frames=8, hit=False, pickup=False, taken=0):
    tracker.update(NS(previous=prev, state=curr, frames=frames, enemies_hit={1: 1} if hit else {},
                      items_gained=[1] if pickup else [], gained_triforce=False,
                      health_gained=max(0, curr.link.health - prev.link.health)), taken)


def test_round_trip_does_not_manufacture_new_best_progress():
    tracker = LingerTracker()
    initial = state()
    tracker.reset(initial)
    closer = state(tile=4)
    update(tracker, initial, closer)
    update(tracker, closer, initial, frames=32)
    update(tracker, initial, closer, frames=4)
    update(tracker, closer, initial, frames=16)
    assert (tracker.max_steps, tracker.max_frames, tracker.tail_steps, tracker.tail_frames) == (3, 52, 3, 52)


def test_moving_enemy_cannot_reset_stationary_or_oscillating_link():
    tracker = LingerTracker()
    initial = state(kind=ObjectiveKind.FIGHT)
    tracker.reset(initial)
    moved_enemy = state(target=4, kind=ObjectiveKind.FIGHT)
    update(tracker, initial, moved_enemy, frames=16)
    away = state(tile=6, target=5, kind=ObjectiveKind.FIGHT)
    update(tracker, moved_enemy, away)
    update(tracker, away, moved_enemy)
    assert (tracker.tail_steps, tracker.tail_frames) == (3, 32)


def test_damage_is_not_progress_but_hits_pickups_and_healing_are():
    tracker = LingerTracker()
    prev = state()
    tracker.reset(prev)
    hurt = state(health=2)
    update(tracker, prev, hurt, frames=16)
    update(tracker, hurt, hurt, frames=32)
    update(tracker, hurt, hurt, hit=True)
    assert (tracker.max_steps, tracker.max_frames, tracker.tail_steps) == (2, 48, 0)
    update(tracker, hurt, hurt)
    update(tracker, hurt, hurt, taken=1)  # collected by weapon, even at full hp
    assert tracker.tail_steps == 0
    update(tracker, hurt, hurt)
    healed = state(health=3)
    update(tracker, hurt, healed)
    assert tracker.tail_steps == 0
    update(tracker, healed, healed)
    update(tracker, healed, healed, pickup=True)
    assert tracker.tail_steps == 0


def test_objective_phase_changes_rebaseline_without_counting_as_progress():
    tracker = LingerTracker()
    prev = state(kind=ObjectiveKind.FIGHT)
    tracker.reset(prev)
    next_phase = state(kind=ObjectiveKind.TREASURE, target=10)
    update(tracker, prev, next_phase)
    assert tracker.tail_steps == 1
    closer = state(tile=6, kind=ObjectiveKind.TREASURE, target=10)
    update(tracker, next_phase, closer)
    assert tracker.tail_steps == 0


def test_exit_closes_streak_and_new_visit_reset_clears_milestones():
    tracker = LingerTracker()
    prev = state(boss=True)
    tracker.reset(prev)
    update(tracker, prev, prev, frames=16)
    dead = state(kills=1)
    update(tracker, prev, dead, frames=4, hit=True)
    heart = state(kills=1, max_health=4)
    update(tracker, dead, heart, frames=32)
    update(tracker, heart, heart, frames=8)
    next_room = state(room=0x36, max_health=4)
    update(tracker, heart, next_room, frames=100)
    assert tracker.summary() == dict(max_steps=1, max_frames=16, tail_steps=0, tail_frames=0, steps=5,
                                    frames=160, boss_kill_steps=2, boss_kill_frames=20,
                                    heart_container_steps=3, heart_container_frames=52)
    tracker.reset(next_room)
    assert tracker.max_steps == tracker.frames == 0 and tracker.boss_kill_steps is None
