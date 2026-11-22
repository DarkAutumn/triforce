//! The aimless-walking score (round 38, the owner: "walking back and forth" with nothing to fight and nowhere to go
//! but the goal; docs/architecture.md "Aimless walking").
//!
//! A decision is **calm** when, as it is taken, no object in the room can hurt Link ([`threatened`]: a living or
//! spawning monster, a monster's shot, a hazard but the cave's standing fires and the boulders' invisible generator),
//! the objective is a place that does not go away ([`fixed_goal`]: an exit side, a cave or stairs, a room treasure,
//! the triforce, a shop's ware, the sword in its cave; never a fight's moving targets), no dropped item pulls him
//! (the env's `pulled`: a pickup the family's shaping makes a target, or one he wants: a heart or fairy below full
//! health, bombs below his maximum; such items vanish), and the shortest walk to the goal (the objective's own
//! wavefront, no dropped items) reaches him and is not 0 (on the goal the game makes him wait: a block's push, the
//! whirlwind, the stairs). A **stretch** is a run of calm decisions in one room and one objective phase; its
//! wavefront is frozen at its first decision. Per stretch: **lasted** = the frames Link had control in its decisions
//! (`Change::control_frames` but the item sub-screen's and the recorder's; the scroll, a pickup's fanfare, the stairs
//! and the cave's text after the decision are frozen and never calm); **needed** = the tiles it gained toward the
//! goal (start distance minus end distance, 0 when it lost ground; the end distance is 0 when it left by the
//! objective's exit) at Link's walking speed ([`FRAMES_PER_TILE`]); **aimless** = lasted − needed −
//! [`STRETCH_ALLOWANCE`], never below 0. A stretch ends when a decision is no longer calm (a monster appears: scored
//! to that point by the distance gained), the phase changes, Link reaches the goal or leaves the room. Beside it:
//! direction **reversals** (a MOVE opposite to the stretch's previous MOVE) and the calm decisions that ended
//! **farther** from the goal than they began.

use crate::change::MazeLoop;
use crate::enums::{Direction, MapLocation, Tile, object};
use crate::game::GameView;
use crate::objectives::{Objective, ObjectiveKind};
use crate::room::{Target, Wavefront};
use std::collections::BTreeSet;
use std::sync::Arc;

/// Link's walking speed in path-search tiles (8 pixels), as the env's MOVEs play it: 64 pixels in 48 frames of three
/// 16-frame MOVEs north in the sword's cave, 21-23 pixels a 16-frame MOVE on the overworld (measured, round 38; the
/// wavefront's 16 tiles from the cave's mouth to 0x77's west edge took 96 control frames).
pub const FRAMES_PER_TILE: f64 = 6.0;

/// Frames a stretch may last beyond the walk it needed without counting: lining up on the 8-pixel grid to turn
/// (Link turns only on a grid point, so a walk with a turn may need up to a tile's frames more on each axis than the
/// wavefront's tile count) and the last MOVE's frames past the tile that ends the stretch. The scripted shortest
/// walks of `tests/test_aimless.py` (following the `target` observation's first step) score 0 with it.
pub const STRETCH_ALLOWANCE: u64 = 16;

/// Whether anything in the room can hurt Link: a monster (living or spawning, not dying), a monster's shot, or a
/// hazard (traps, boulders, Bubbles, Gleeok's flying heads) but the cave's standing fires (they hurt only on touch,
/// never move, and the walk to the sword goes round them) and the boulders' invisible generator.
pub fn threatened(state: &GameView) -> bool {
    state.enemies.iter().any(|e| !e.is_dying())
        || !state.projectiles.is_empty()
        || state
            .hazards
            .iter()
            .any(|h| h.id != object::STANDING_FIRE && h.id != object::BOULDER_SET)
}

/// Whether `objective` is a place that does not go away: an exit side, a cave or stairs, a room treasure (a heart
/// container, the triforce, a key, a shop's ware, the sword in its cave), with targets. A fight (its targets move)
/// is none.
pub fn fixed_goal(objective: &Objective) -> bool {
    matches!(objective.kind, ObjectiveKind::Move | ObjectiveKind::Cave | ObjectiveKind::Treasure)
        && !objective.pbrs_targets.is_empty()
}

/// What a stretch's wavefront belongs to: the room, the objective's kind, targets and next rooms, and Link's keys
/// (locked doors are open to the path search while he holds one).
type Phase = (MapLocation, ObjectiveKind, BTreeSet<Target>, BTreeSet<MapLocation>, i32);

/// What the tracker reads from one state and its objective.
#[derive(Clone, Debug)]
pub struct AimlessState {
    pub full_location: MapLocation,
    pub tile: Tile,
    /// Calm but for the wavefront: nothing can hurt Link ([`threatened`]), the objective is a [`fixed_goal`], no
    /// dropped item pulls him (the env's `pulled`), and the state is not game over.
    pub quiet: bool,
    pub next_rooms: BTreeSet<MapLocation>,
    phase: Phase,
}

impl AimlessState {
    /// `pulled`: a dropped item in the room pulls Link (the env decides: its family's shaping targets and the drops
    /// he wants).
    pub fn of(state: &GameView, objective: &Objective, pulled: bool) -> AimlessState {
        let full_location = state.full_location();
        AimlessState {
            full_location,
            tile: state.link.tile(),
            quiet: !pulled && !state.game_over() && fixed_goal(objective) && !threatened(state),
            next_rooms: objective.next_rooms.clone(),
            phase: (
                full_location,
                objective.kind,
                objective.pbrs_targets.clone(),
                objective.next_rooms.clone(),
                state.keys(),
            ),
        }
    }
}

/// What the tracker reads from one step.
#[derive(Clone, Copy, Debug, Default)]
pub struct AimlessStep {
    /// Frames Link had control in the step: `control_frames` but the item sub-screen's and the recorder's.
    pub frames: u32,
    /// The direction of a MOVE (None for any other action).
    pub moved: Option<Direction>,
    /// The step looped through a maze room (`Change::maze_loop`).
    pub maze_loop: Option<MazeLoop>,
}

impl AimlessStep {
    pub fn of(change: &crate::change::Change) -> AimlessStep {
        AimlessStep {
            frames: change.control_frames.saturating_sub(change.menu_frames),
            moved: (change.action.kind == crate::enums::ActionKind::Move).then_some(change.action.direction),
            maze_loop: change.maze_loop(),
        }
    }
}

/// A visit's totals (`aimless_frames` and its companions in visits and chain room records).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AimlessSummary {
    /// Frames of calm decisions (Link in control).
    pub calm_frames: u64,
    /// Frames the shortest walk needed for the ground the stretches gained.
    pub needed_frames: u64,
    /// Per stretch lasted − needed − [`STRETCH_ALLOWANCE`], at least 0, summed.
    pub aimless_frames: u64,
    pub stretches: u32,
    /// Calm decisions, and those that ended farther from the goal than they began.
    pub calm_steps: u32,
    pub away_steps: u32,
    /// MOVEs opposite to the stretch's previous MOVE.
    pub reversals: u32,
}

/// What one decision was: calm, the frames of progress it made toward the goal (tiles gained at walking speed;
/// negative when it lost ground; 0 when not calm), and the frames it added to its stretch's waste (the rise of
/// lasted − needed so far, never below 0: over a stretch they sum to its score before the allowance, but for the
/// few frames a MOVE that crosses a tile faster than walking speed takes back). The lab's per-decision rollout
/// targets and `penalty-aimless`'s measure.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AimlessDecision {
    pub calm: bool,
    pub progress_frames: f32,
    pub waste_frames: f32,
}

impl AimlessSummary {
    /// The JSON fields of a visit or room record, by name.
    pub fn fields(&self) -> [(&'static str, u64); 7] {
        [
            ("aimless_frames", self.aimless_frames),
            ("calm_frames", self.calm_frames),
            ("calm_needed_frames", self.needed_frames),
            ("calm_stretches", u64::from(self.stretches)),
            ("calm_decisions", u64::from(self.calm_steps)),
            ("calm_away", u64::from(self.away_steps)),
            ("calm_reversals", u64::from(self.reversals)),
        ]
    }
}

#[derive(Clone, Debug)]
struct Stretch {
    phase: Phase,
    wavefront: Arc<Wavefront>,
    start: i32,
    last: i32,
    frames: u64,
    last_move: Option<Direction>,
}

impl Stretch {
    /// Lasted − needed so far, never below 0.
    fn waste(&self) -> f64 {
        (self.frames as f64 - f64::from((self.start - self.last).max(0)) * FRAMES_PER_TILE).max(0.0)
    }
}

/// Follows a room visit's calm stretches.
#[derive(Clone, Debug, Default)]
pub struct AimlessTracker {
    pub s: AimlessSummary,
    open: Option<Stretch>,
    /// The last wavefront built and its phase: a stretch that opens again in the same phase reuses it while it
    /// reaches Link (a shut door that opened since is built again).
    cache: Option<(Phase, Arc<Wavefront>)>,
}

impl AimlessTracker {
    /// Starts a visit at `state`. `wavefront` builds the objective's wavefront at the state a stretch opens in.
    pub fn reset(&mut self, state: &AimlessState, wavefront: impl FnOnce() -> Arc<Wavefront>) {
        self.s = AimlessSummary::default();
        self.open = None;
        self.cache = None;
        self.try_open(state, wavefront);
    }

    /// Follows one step from `prev` (the state the decision was taken in) to `curr`; `closing`: the visit ends with
    /// this step (a room change does by itself). `wavefront` builds the objective's wavefront at `curr` when a stretch
    /// opens there.
    pub fn update(
        &mut self,
        prev: &AimlessState,
        curr: &AimlessState,
        step: &AimlessStep,
        closing: bool,
        wavefront: impl FnOnce() -> Arc<Wavefront>,
    ) -> AimlessDecision {
        let left = prev.full_location != curr.full_location || step.maze_loop.is_some();
        let mut decision = AimlessDecision::default();
        if let Some(stretch) = self.open.as_mut() {
            let distance = if left {
                // Leaving by the objective's exit reaches the goal; any other way gains nothing on the last step.
                let correct = match step.maze_loop {
                    Some(l) => l.correct(&prev.next_rooms),
                    None => prev.next_rooms.contains(&curr.full_location),
                };
                Some(if correct { 0 } else { stretch.last })
            } else {
                stretch.wavefront.get(curr.tile)
            };
            let d = distance.unwrap_or(stretch.last);
            let s = &mut self.s;
            let waste = stretch.waste();
            stretch.frames += u64::from(step.frames);
            s.calm_frames += u64::from(step.frames);
            s.calm_steps += 1;
            s.away_steps += u32::from(d > stretch.last);
            if let Some(dir) = step.moved {
                s.reversals += u32::from(stretch.last_move == Some(dir.opposite()));
                stretch.last_move = Some(dir);
            }
            let progress_frames = ((stretch.last - d) as f64 * FRAMES_PER_TILE) as f32;
            stretch.last = d;
            decision = AimlessDecision {
                calm: true,
                progress_frames,
                waste_frames: (stretch.waste() - waste).max(0.0) as f32,
            };
            if left || closing || distance.is_none() || d == 0 || !curr.quiet || curr.phase != stretch.phase {
                self.close();
            }
        }
        if self.open.is_none() && !left && !closing {
            self.try_open(curr, wavefront);
        }
        decision
    }

    /// Ends the open stretch (the env calls it when a visit ends without a step, e.g. a run cut off).
    pub fn close(&mut self) {
        let Some(stretch) = self.open.take() else {
            return;
        };
        let needed = (f64::from((stretch.start - stretch.last).max(0)) * FRAMES_PER_TILE).round() as u64;
        self.s.needed_frames += needed;
        self.s.aimless_frames += stretch.frames.saturating_sub(needed + STRETCH_ALLOWANCE);
        self.s.stretches += 1;
    }

    /// Whether a stretch is open (the next decision is calm).
    pub fn calm(&self) -> bool {
        self.open.is_some()
    }

    pub fn summary(&self) -> AimlessSummary {
        self.s
    }

    fn try_open(&mut self, state: &AimlessState, wavefront: impl FnOnce() -> Arc<Wavefront>) {
        if !state.quiet {
            return;
        }
        let cached = self
            .cache
            .as_ref()
            .filter(|(phase, w)| *phase == state.phase && w.get(state.tile).is_some())
            .map(|(_, w)| w.clone());
        let wavefront = cached.unwrap_or_else(|| {
            let w = wavefront();
            self.cache = Some((state.phase.clone(), w.clone()));
            w
        });
        if let Some(d) = wavefront.get(state.tile)
            && d > 0
        {
            self.open = Some(Stretch {
                phase: state.phase.clone(),
                wavefront,
                start: d,
                last: d,
                frames: 0,
                last_move: None,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    //! Synthetic distances, no emulator (the emulator's: `crates/env/tests/aimless.rs`).
    use super::*;

    fn state(tile: i32, quiet: bool, room: u8) -> AimlessState {
        let full_location = MapLocation::new(0, room, true);
        let next: BTreeSet<MapLocation> = [MapLocation::new(0, 0x77, false)].into_iter().collect();
        AimlessState {
            full_location,
            tile: (tile, 0),
            quiet,
            next_rooms: next.clone(),
            phase: (full_location, ObjectiveKind::Move, BTreeSet::new(), next, 0),
        }
    }

    /// Distance = the tile's column: the goal at column 0.
    fn wf() -> Arc<Wavefront> {
        Arc::new(Wavefront::from_distances((0..32).map(|t| ((t, 0), t))))
    }

    fn walk(t: &mut AimlessTracker, path: &[(i32, Direction)], frames: u32) -> Vec<AimlessDecision> {
        let mut out = Vec::new();
        for pair in path.windows(2) {
            let (prev, curr) = (state(pair[0].0, true, 0x77), state(pair[1].0, true, 0x77));
            let step = AimlessStep { frames, moved: Some(pair[1].1), maze_loop: None };
            out.push(t.update(&prev, &curr, &step, false, wf));
        }
        out
    }

    #[test]
    fn a_straight_walk_scores_nothing_and_a_detour_scores_what_it_wasted() {
        let mut t = AimlessTracker::default();
        t.reset(&state(10, true, 0x77), wf);
        let path: Vec<(i32, Direction)> = (0..=10).rev().map(|c| (c, Direction::W)).collect();
        let d = walk(&mut t, &path, 16);
        let s = t.summary();
        // 10 tiles at 16 frames a tile: 160 lasted against 60 needed.
        assert_eq!((s.calm_steps, s.stretches, s.reversals, s.away_steps), (10, 1, 0, 0));
        assert_eq!((s.calm_frames, s.needed_frames), (160, 60));
        assert_eq!(s.aimless_frames, 160 - 60 - STRETCH_ALLOWANCE);
        assert!(d.iter().all(|d| d.calm && (f64::from(d.progress_frames) - FRAMES_PER_TILE).abs() < 1e-5));

        // At walking speed (18 frames for 3 tiles): nothing.
        let mut t = AimlessTracker::default();
        t.reset(&state(9, true, 0x77), wf);
        let path: Vec<(i32, Direction)> = [9, 6, 3, 0].iter().map(|&c| (c, Direction::W)).collect();
        let d = walk(&mut t, &path, 18);
        assert_eq!((t.s.calm_frames, t.s.needed_frames, t.s.aimless_frames), (54, 54, 0));
        assert!(d.iter().all(|d| d.waste_frames == 0.0));

        // Back and forth: 6 tiles away and back costs what it took, the reversals and the away decisions counted.
        let mut t = AimlessTracker::default();
        t.reset(&state(6, true, 0x77), wf);
        let path: Vec<(i32, Direction)> = [(6, Direction::W), (9, Direction::E), (12, Direction::E),
            (9, Direction::W), (6, Direction::W), (3, Direction::W), (0, Direction::W)]
            .into_iter()
            .collect();
        let d = walk(&mut t, &path, 18);
        let s = t.summary();
        assert_eq!((s.calm_frames, s.needed_frames, s.reversals, s.away_steps), (108, 36, 1, 2));
        assert_eq!(s.aimless_frames, 108 - 36 - STRETCH_ALLOWANCE);
        // The waste rises while Link is behind where he began, and its rises sum to lasted − needed.
        assert_eq!(d.iter().map(|d| d.waste_frames).sum::<f32>(), 72.0);
    }

    #[test]
    fn a_threat_ends_the_stretch_at_the_ground_gained_and_nothing_counts_while_it_lasts() {
        let mut t = AimlessTracker::default();
        t.reset(&state(12, true, 0x77), wf);
        // Two tiles gained in 64 frames, then a monster appears.
        let d = t.update(&state(12, true, 0x77), &state(11, true, 0x77), &AimlessStep { frames: 32, ..Default::default() }, false, wf);
        assert!(d.calm);
        t.update(&state(11, true, 0x77), &state(10, false, 0x77), &AimlessStep { frames: 32, ..Default::default() }, false, wf);
        assert!(!t.calm());
        let needed = (2.0 * FRAMES_PER_TILE).round() as u64;
        assert_eq!((t.s.stretches, t.s.calm_frames, t.s.needed_frames), (1, 64, needed));
        assert_eq!(t.s.aimless_frames, 64 - needed - STRETCH_ALLOWANCE);
        // While it lives, nothing is calm, whatever Link does.
        for _ in 0..10 {
            let d = t.update(&state(10, false, 0x77), &state(14, false, 0x77), &AimlessStep { frames: 100, ..Default::default() }, false, wf);
            assert_eq!(d, AimlessDecision::default());
        }
        assert_eq!(t.s.calm_frames, 64);
    }

    #[test]
    fn leaving_by_the_objective_reaches_the_goal_and_on_the_goal_nothing_is_calm() {
        let mut t = AimlessTracker::default();
        t.reset(&state(3, true, 0x77), wf);
        // A MOVE out of the room through the objective's exit (the next room): the whole distance gained.
        let mut out = state(3, true, 0x77);
        out.full_location = MapLocation::new(0, 0x77, false);
        out.phase.0 = out.full_location;
        t.update(&state(3, true, 0x77), &out, &AimlessStep { frames: 18, ..Default::default() }, false, wf);
        assert_eq!((t.s.stretches, t.s.needed_frames, t.s.aimless_frames), (1, 18, 0));
        assert!(!t.calm(), "a room change closes the visit's stretch and opens none");
        // Standing on the goal (distance 0: a block's push, the whirlwind's spot) is no calm decision.
        let mut t = AimlessTracker::default();
        t.reset(&state(0, true, 0x77), wf);
        assert!(!t.calm());
    }
}
