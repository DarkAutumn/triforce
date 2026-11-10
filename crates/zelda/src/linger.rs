//! Behavior-neutral progress timing for episodes and room visits (`triforce/linger.py`).
//!
//! A frozen wavefront measures new best approaches: moving enemies cannot manufacture progress for a stationary
//! Link, and walking the same loop cannot repeatedly reset the clock.  The distance baseline refreshes after a
//! hit, kill, pickup/healing, room exit or objective phase change (its kind, next rooms or, but in a fight, its
//! targets: a shutter or bomb wall that opens, a pushed block), and while the frozen wavefront can't reach Link; a
//! refresh by itself is not progress.

use crate::change::Change;
use crate::enums::{Direction, MapLocation, Tile, enemy};
use crate::game::GameView;
use crate::objectives::{Objective, ObjectiveKind};
use crate::room::{Target, Wavefront};
use std::collections::BTreeSet;
use std::sync::Arc;

/// What the tracker reads from one state and its objective.
#[derive(Clone, Debug)]
pub struct LingerState {
    pub full_location: MapLocation,
    pub game_over: bool,
    pub tile: Tile,
    pub keys: i32,
    pub max_health: i32,
    pub room_kills: i32,
    /// A boss is alive (and not dying).
    pub boss_alive: bool,
    pub objective_kind: ObjectiveKind,
    pub next_rooms: BTreeSet<MapLocation>,
    pub has_targets: bool,
    /// The objective's PBRS targets but in a fight (whose targets are the moving monsters): part of the phase.
    pub targets: BTreeSet<Target>,
}

impl LingerState {
    pub fn of(state: &GameView, objective: &Objective) -> LingerState {
        LingerState {
            full_location: state.full_location(),
            game_over: state.game_over(),
            tile: state.link.tile(),
            keys: state.keys(),
            max_health: state.max_health(),
            room_kills: state.room_kills(),
            boss_alive: state
                .enemies
                .iter()
                .any(|e| enemy::is_boss(e.id) && !e.is_dying()),
            objective_kind: objective.kind,
            next_rooms: objective.next_rooms.clone(),
            has_targets: !objective.pbrs_targets.is_empty(),
            targets: if objective.kind == ObjectiveKind::Fight {
                BTreeSet::new()
            } else {
                objective.pbrs_targets.clone()
            },
        }
    }

    fn phase(&self) -> Phase {
        (
            self.full_location,
            self.objective_kind,
            self.next_rooms.clone(),
            self.targets.clone(),
        )
    }
}

/// What a frozen wavefront belongs to: room, objective kind, next rooms, targets (none in a fight).
type Phase = (MapLocation, ObjectiveKind, BTreeSet<MapLocation>, BTreeSet<Target>);

/// What the tracker reads from one step.
#[derive(Clone, Copy, Debug, Default)]
pub struct LingerStep {
    pub frames: u32,
    pub items_gained: bool,
    pub gained_triforce: bool,
    pub enemies_hit: bool,
    pub health_gained: f64,
}

impl LingerStep {
    /// Observed effects only: a hit counts on the step it lands.
    pub fn of(change: &Change) -> LingerStep {
        LingerStep {
            frames: change.frames,
            items_gained: !change.observed.items_gained.is_empty(),
            gained_triforce: change.gained_triforce(),
            enemies_hit: !change.observed.enemies_hit.is_empty(),
            health_gained: change.health_gained,
        }
    }
}

/// The wavefront linger measures on: the env's PBRS wavefront when it has one, else the objective's
/// (with locked doors open when Link has keys in a dungeon).
pub fn linger_wavefront(
    state: &GameView,
    objective: &Objective,
    pbrs: Option<Arc<Wavefront>>,
) -> Arc<Wavefront> {
    if let Some(w) = pbrs {
        return w;
    }
    let locked: Vec<Direction> = if state.level() > 0 && state.keys() > 0 {
        [Direction::N, Direction::S, Direction::E, Direction::W]
            .into_iter()
            .filter(|&d| state.is_door_locked(d))
            .collect()
    } else {
        Vec::new()
    };
    let targets: Vec<Target> = objective.pbrs_targets.iter().copied().collect();
    Arc::new(state.room().wavefront(&targets, &[], &locked))
}

/// `LingerTracker.summary()`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LingerSummary {
    pub max_steps: u32,
    pub max_frames: u32,
    pub tail_steps: u32,
    pub tail_frames: u32,
    pub steps: u32,
    pub frames: u32,
    pub boss_kill_steps: Option<u32>,
    pub boss_kill_frames: Option<u32>,
    pub heart_container_steps: Option<u32>,
    pub heart_container_frames: Option<u32>,
}

/// Counts actions and elapsed frames without progress.
#[derive(Clone, Debug, Default)]
pub struct LingerTracker {
    pub s: LingerSummary,
    /// (level, room, in cave) the visit started in.
    pub start_room: (u8, u8, bool),
    phase: Option<Phase>,
    has_targets: bool,
    wavefront: Option<Arc<Wavefront>>,
    best: Option<i32>,
}

impl LingerTracker {
    /// Starts a visit.  `wavefront` is called when a baseline needs one (see [`linger_wavefront`]).
    pub fn reset(&mut self, state: &LingerState, wavefront: impl FnOnce() -> Arc<Wavefront>) {
        self.s = LingerSummary::default();
        let loc = state.full_location;
        self.start_room = (loc.level, loc.value, loc.in_cave);
        self.baseline(state, wavefront);
    }

    fn baseline(&mut self, state: &LingerState, wavefront: impl FnOnce() -> Arc<Wavefront>) {
        self.phase = Some(state.phase());
        self.has_targets = state.has_targets;
        self.wavefront = None;
        self.best = None;
        if !state.game_over && self.has_targets {
            let w = wavefront();
            self.best = w.get(state.tile);
            self.wavefront = Some(w);
        }
    }

    /// Follows one step from `prev` to `curr`; `hearts_taken` counts hearts picked up this step.
    pub fn update(
        &mut self,
        prev: &LingerState,
        curr: &LingerState,
        step: &LingerStep,
        hearts_taken: u32,
        wavefront: impl FnOnce() -> Arc<Wavefront>,
    ) {
        let s = &mut self.s;
        s.steps += 1;
        s.frames += step.frames;
        let changed_room = prev.full_location != curr.full_location;
        let pickup = step.items_gained
            || hearts_taken > 0
            || step.gained_triforce
            || curr.keys > prev.keys
            || curr.max_health > prev.max_health;
        let tangible = changed_room
            || pickup
            || step.enemies_hit
            || step.health_gained > 0.0
            || curr.room_kills > prev.room_kills;
        if !changed_room {
            if s.boss_kill_steps.is_none() && prev.boss_alive && !curr.boss_alive {
                s.boss_kill_steps = Some(s.steps);
                s.boss_kill_frames = Some(s.frames);
            }
            if s.heart_container_steps.is_none() && curr.max_health > prev.max_health {
                s.heart_container_steps = Some(s.steps);
                s.heart_container_frames = Some(s.frames);
            }
        }
        let phase = curr.phase();
        let same_phase = self.phase.as_ref() == Some(&phase);
        let mut approaching = false;
        if !tangible
            && same_phase
            && let Some(w) = &self.wavefront
            && curr.tile != prev.tile
            && let Some(distance) = w.get(curr.tile)
            && self.best.is_none_or(|b| distance < b)
        {
            self.best = Some(distance);
            approaching = true;
        }
        if tangible || approaching {
            s.tail_steps = 0;
            s.tail_frames = 0;
        } else {
            s.tail_steps += 1;
            s.tail_frames += step.frames;
            s.max_steps = s.max_steps.max(s.tail_steps);
            s.max_frames = s.max_frames.max(s.tail_frames);
        }
        // A frozen wavefront that can't reach Link (built while the exit was still shut) measures nothing.
        let blind = curr.has_targets
            && self.best.is_none()
            && self.wavefront.as_ref().is_none_or(|w| w.get(curr.tile).is_none());
        if tangible || !same_phase || (!self.has_targets && curr.has_targets) || blind {
            self.baseline(curr, wavefront);
        }
    }

    pub fn tail_steps(&self) -> u32 {
        self.s.tail_steps
    }

    pub fn summary(&self) -> LingerSummary {
        self.s
    }
}

#[cfg(test)]
mod tests {
    //! Port of tests/test_linger.py (synthetic distances, no emulator).
    use super::*;

    struct S {
        st: LingerState,
        target: i32,
    }

    fn state(
        tile: i32,
        target: i32,
        kind: ObjectiveKind,
        room: u8,
        kills: i32,
        boss: bool,
        max_health: i32,
    ) -> S {
        S {
            st: LingerState {
                full_location: MapLocation::new(1, room, false),
                game_over: false,
                tile: (tile, 0),
                keys: 0,
                max_health,
                room_kills: kills,
                boss_alive: boss,
                objective_kind: kind,
                next_rooms: [MapLocation::new(1, 0x36, false)].into_iter().collect(),
                has_targets: true,
                targets: BTreeSet::new(),
            },
            target,
        }
    }

    fn base() -> S {
        state(5, 0, ObjectiveKind::Move, 0x35, 0, false, 3)
    }

    fn wf(s: &S) -> impl FnOnce() -> Arc<Wavefront> + '_ {
        move || {
            Arc::new(Wavefront::from_distances(
                (0..20).map(|t| ((t, 0), (t - s.target).abs())),
            ))
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn update(
        t: &mut LingerTracker,
        prev: &S,
        curr: &S,
        frames: u32,
        hit: bool,
        pickup: bool,
        taken: u32,
        gained: f64,
    ) {
        let step = LingerStep {
            frames,
            items_gained: pickup,
            gained_triforce: false,
            enemies_hit: hit,
            health_gained: gained,
        };
        t.update(&prev.st, &curr.st, &step, taken, wf(curr));
    }

    #[test]
    fn round_trip_does_not_manufacture_new_best_progress() {
        let mut t = LingerTracker::default();
        let initial = base();
        t.reset(&initial.st, wf(&initial));
        let closer = state(4, 0, ObjectiveKind::Move, 0x35, 0, false, 3);
        update(&mut t, &initial, &closer, 8, false, false, 0, 0.0);
        update(&mut t, &closer, &initial, 32, false, false, 0, 0.0);
        update(&mut t, &initial, &closer, 4, false, false, 0, 0.0);
        update(&mut t, &closer, &initial, 16, false, false, 0, 0.0);
        let s = t.summary();
        assert_eq!(
            (s.max_steps, s.max_frames, s.tail_steps, s.tail_frames),
            (3, 52, 3, 52)
        );
    }

    /// F154: a baseline taken while the exit was shut (a wavefront that can't reach Link) refreshes once the
    /// wavefront reaches him; then each step closer is progress.
    #[test]
    fn a_blind_baseline_refreshes_when_the_exit_opens() {
        let mut t = LingerTracker::default();
        let shut = base();
        t.reset(&shut.st, || Arc::new(Wavefront::from_distances([((0, 1), 0)])));
        let open = |tile| state(tile, 0, ObjectiveKind::Move, 0x35, 0, false, 3);
        // The door opens: same phase, the new wavefront reaches Link (a refresh, not progress).
        let (a, b, c) = (open(5), open(4), open(3));
        update(&mut t, &shut, &a, 4, false, false, 0, 0.0);
        assert_eq!(t.tail_steps(), 1);
        update(&mut t, &a, &b, 4, false, false, 0, 0.0);
        assert_eq!(t.tail_steps(), 0, "one tile closer to the open exit");
        update(&mut t, &b, &c, 4, false, false, 0, 0.0);
        assert_eq!(t.tail_steps(), 0);
    }

    #[test]
    fn moving_enemy_cannot_reset_stationary_or_oscillating_link() {
        let mut t = LingerTracker::default();
        let initial = state(5, 0, ObjectiveKind::Fight, 0x35, 0, false, 3);
        t.reset(&initial.st, wf(&initial));
        let moved = state(5, 4, ObjectiveKind::Fight, 0x35, 0, false, 3);
        update(&mut t, &initial, &moved, 16, false, false, 0, 0.0);
        let away = state(6, 5, ObjectiveKind::Fight, 0x35, 0, false, 3);
        update(&mut t, &moved, &away, 8, false, false, 0, 0.0);
        update(&mut t, &away, &moved, 8, false, false, 0, 0.0);
        assert_eq!((t.s.tail_steps, t.s.tail_frames), (3, 32));
    }

    #[test]
    fn damage_is_not_progress_but_hits_pickups_and_healing_are() {
        let mut t = LingerTracker::default();
        let prev = base();
        t.reset(&prev.st, wf(&prev));
        let hurt = base();
        update(&mut t, &prev, &hurt, 16, false, false, 0, 0.0);
        update(&mut t, &hurt, &hurt, 32, false, false, 0, 0.0);
        update(&mut t, &hurt, &hurt, 8, true, false, 0, 0.0);
        assert_eq!((t.s.max_steps, t.s.max_frames, t.s.tail_steps), (2, 48, 0));
        update(&mut t, &hurt, &hurt, 8, false, false, 0, 0.0);
        update(&mut t, &hurt, &hurt, 8, false, false, 1, 0.0);
        assert_eq!(t.s.tail_steps, 0);
        update(&mut t, &hurt, &hurt, 8, false, false, 0, 0.0);
        let healed = base();
        update(&mut t, &hurt, &healed, 8, false, false, 0, 1.0);
        assert_eq!(t.s.tail_steps, 0);
        update(&mut t, &healed, &healed, 8, false, false, 0, 0.0);
        update(&mut t, &healed, &healed, 8, false, true, 0, 0.0);
        assert_eq!(t.s.tail_steps, 0);
    }

    #[test]
    fn objective_phase_changes_rebaseline_without_counting_as_progress() {
        let mut t = LingerTracker::default();
        let prev = state(5, 0, ObjectiveKind::Fight, 0x35, 0, false, 3);
        t.reset(&prev.st, wf(&prev));
        let next = state(5, 10, ObjectiveKind::Treasure, 0x35, 0, false, 3);
        update(&mut t, &prev, &next, 8, false, false, 0, 0.0);
        assert_eq!(t.s.tail_steps, 1);
        let closer = state(6, 10, ObjectiveKind::Treasure, 0x35, 0, false, 3);
        update(&mut t, &next, &closer, 8, false, false, 0, 0.0);
        assert_eq!(t.s.tail_steps, 0);
    }

    #[test]
    fn exit_closes_streak_and_new_visit_reset_clears_milestones() {
        let mut t = LingerTracker::default();
        let prev = state(5, 0, ObjectiveKind::Move, 0x35, 0, true, 3);
        t.reset(&prev.st, wf(&prev));
        update(&mut t, &prev, &prev, 16, false, false, 0, 0.0);
        let dead = state(5, 0, ObjectiveKind::Move, 0x35, 1, false, 3);
        update(&mut t, &prev, &dead, 4, true, false, 0, 0.0);
        let heart = state(5, 0, ObjectiveKind::Move, 0x35, 1, false, 4);
        update(&mut t, &dead, &heart, 32, false, false, 0, 0.0);
        update(&mut t, &heart, &heart, 8, false, false, 0, 0.0);
        let next_room = state(5, 0, ObjectiveKind::Move, 0x36, 0, false, 4);
        update(&mut t, &heart, &next_room, 100, false, false, 0, 0.0);
        assert_eq!(
            t.summary(),
            LingerSummary {
                max_steps: 1,
                max_frames: 16,
                tail_steps: 0,
                tail_frames: 0,
                steps: 5,
                frames: 160,
                boss_kill_steps: Some(2),
                boss_kill_frames: Some(20),
                heart_container_steps: Some(3),
                heart_container_frames: Some(52),
            }
        );
        t.reset(&next_room.st, wf(&next_room));
        assert!(t.s.max_steps == 0 && t.s.frames == 0 && t.s.boss_kill_steps.is_none());
    }
}
