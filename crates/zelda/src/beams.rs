//! Sword shots at range (round 38, the owner: "It should get more reward for long range kills (out of melee range),
//! and possibly a small penalty for ignoring a lined up shot [...] reward timing beams that WOULD intersect even if
//! the enemy turns the other way in reality before it hits"): which monsters a sword shot fired now would meet, each
//! monster moving straight on (the projection of the contact columns, `crate::contact`), and what a visit's shots and
//! kills were ([`BeamStats`]). Arithmetic on the state alone: no emulator step, no look-ahead.
//!
//! The shot (asm:SwordShot_Make, asm:SwordShot_SetUpWithState): made on the swing's frame 12 after the press, $10
//! pixels ahead of Link in his facing, 3 pixels a frame (q-speed $C0); it flies over every tile and stops only at the
//! room's bounds (asm:Room_BoundDir, the weapon slots' wider ones) or on a monster: its hit point (X+6, Y+8) facing
//! up or down, (X+8, Y+6) facing left or right, within $0C pixels of the monster's middle on both axes
//! (asm:SwordShot_CheckMonsterCollision, asm:Collision_CheckThinWeaponVsMonster_Thresholds).

use crate::attack;
use crate::change::Change;
use crate::contact;
use crate::enums::{ActionKind, AnimationKind, AnimationState, Direction, enemy};
use crate::game::{Enemy, GameView};
use crate::room_state::decode_room_state;
use crate::tiles::{Grid, build_grid};
use std::ops::AddAssign;

/// Frames from the press to the frame the shot is made (asm:Sword_UpdateSwordOrRod: the swing's state 3).
pub const SHOT_DELAY: u32 = 12;
/// Pixels ahead of Link the shot starts (asm:SwordShot_SetUpWithState).
pub const SHOT_AHEAD: f32 = 16.0;
/// Pixels a frame (q-speed $C0).
pub const SHOT_SPEED: f32 = 3.0;
/// The hit test passes below this distance on both axes (threshold $0C).
pub const SHOT_HIT: f32 = 12.0;
/// Frames of flight followed: more than a shot needs to cross the screen.
pub const SHOT_HORIZON: u32 = 90;
/// Out of melee range: a monster whose corner is farther than this from Link's (the `distance` of
/// `crate::game::Enemy`). The swing's thin box reaches a monster's corner about 29 pixels out
/// (asm:Sword_LinkToWeaponOffsetsX/Y, $10 along and $0C across, asm:Collision_CheckStabVsMonster).
pub const MELEE_REACH: f32 = 32.0;
/// asm:ObjInvincibilityMask ($4B2 + slot): bit 1 is the sword's damage type (sword, rod swing, sword shot).
const INVINCIBILITY_MASK: u16 = 0x4B2;
const SWORD_DAMAGE: u8 = 0x01;

/// The weapon slots' room bounds for a shot's corner: (left, right, top, bottom), moving on while left <= x < right
/// and top <= y < bottom (asm:Room_BoundDirHorizontally/Vertically: the monsters' bounds widened by $0B, $0C, $0F,
/// $12).
fn shot_bounds(overworld: bool) -> (f32, f32, f32, f32) {
    if overworld {
        ((0x11 - 0x0B) as f32, (0xE0 + 0x0C) as f32, (0x4E - 0x0F) as f32, (0xCD + 0x12) as f32)
    } else {
        ((0x21 - 0x0B) as f32, (0xD0 + 0x0C) as f32, (0x5E - 0x0F) as f32, (0xBD + 0x12) as f32)
    }
}

fn unit(d: Direction) -> (f32, f32) {
    match d {
        Direction::N => (0.0, -1.0),
        Direction::S => (0.0, 1.0),
        Direction::W => (-1.0, 0.0),
        Direction::E => (1.0, 0.0),
        _ => (0.0, 0.0),
    }
}

fn opposite(d: Direction) -> Direction {
    match d {
        Direction::N => Direction::S,
        Direction::S => Direction::N,
        Direction::W => Direction::E,
        Direction::E => Direction::W,
        other => other,
    }
}

/// A monster a sword shot in `d` can hurt now: active (a surfaced Leever or Zora, a Wallmaster on screen, else
/// spawned), not dying, not immune to the sword (ObjInvincibilityMask), not a Darknut facing the shot
/// (asm:Damage_HandleWeaponHit@CheckDarknut), not the big Digdogger.
fn shootable(state: &GameView, e: &Enemy, d: Direction) -> bool {
    e.is_active()
        && !e.is_dying()
        && !e.is_big_digdogger()
        && state.peek(INVINCIBILITY_MASK + u16::from(e.index)) & SWORD_DAMAGE == 0
        && !(matches!(e.id, enemy::BLUE_DARKNUT | enemy::RED_DARKNUT) && e.direction == opposite(d))
}

/// The first monster out of melee range ([`MELEE_REACH`]) a sword shot pressed in `d` from `state` would meet, each
/// monster carried straight on at its present velocity (ObjDir, ObjQSpeedFrac: `contact::velocity`), a walker
/// stopped where a wall stops it (`contact::travel`), whatever it does after: (slot, frames from the press to the
/// hit). None when no such monster is on the shot's course or the sword can't swing in `d` (`attack::press`).
pub fn shot_meets(state: &GameView, grid: &Grid, d: Direction) -> Option<(u8, u32)> {
    let (turn, link) = attack::press(state, ActionKind::Beams, d)?;
    let (ux, uy) = unit(d);
    let (sx, sy) = (f32::from(link.x) + SHOT_AHEAD * ux, f32::from(link.y) + SHOT_AHEAD * uy);
    // The shot's hit point from its corner, by Link's facing.
    let (hx, hy) = if uy != 0.0 { (6.0, 8.0) } else { (8.0, 6.0) };
    let (left, right, top, bottom) = shot_bounds(state.level() == 0);
    let ram = state.ram();
    let mut best: Option<(u8, u32)> = None;
    for e in state.enemies.iter().filter(|e| e.distance > MELEE_REACH && shootable(state, e, d)) {
        let slot = usize::from(e.index);
        let (vx, vy) = contact::velocity(ram, slot);
        let speed = vx.hypot(vy);
        let (mux, muy) = if speed > 0.0 { (vx / speed, vy / speed) } else { (0.0, 0.0) };
        let (x, y) = (e.position.x as f32, e.position.y as f32);
        let reach = if speed > 0.0 && contact::walks(ram, slot) {
            match contact::travel(x, y, mux, muy, grid) {
                t if t >= contact::TRAVEL_LIMIT as f32 => contact::UNSTOPPED,
                t => t,
            }
        } else {
            contact::UNSTOPPED
        };
        let (_, attr) = contact::slot_type(ram, slot);
        let (mx, my) = (x + contact::middle_x(attr), y + 8.0);
        for k in 0..SHOT_HORIZON {
            let (cx, cy) = (sx + SHOT_SPEED * k as f32 * ux, sy + SHOT_SPEED * k as f32 * uy);
            if !(left <= cx && cx < right && top <= cy && cy < bottom) {
                break;
            }
            let frames = turn + SHOT_DELAY + k;
            let moved = (speed * frames as f32).min(reach);
            let (ex, ey) = (mx + mux * moved, my + muy * moved);
            if (cx + hx - ex).abs() < SHOT_HIT && (cy + hy - ey).abs() < SHOT_HIT {
                if best.is_none_or(|(_, f)| frames < f) {
                    best = Some((e.index, frames));
                }
                break;
            }
        }
    }
    best
}

/// The directions a sword shot fired now would meet a monster out of melee range ([`shot_meets`]), when Link can fire
/// one (full hearts, the shot's slot free, the sword ready: `GameView::are_beams_available`, `weapon_ready`). Empty
/// otherwise, and without building the tile grid when no monster stands out of melee range.
pub fn lined_up(state: &GameView) -> Vec<Direction> {
    if !state.are_beams_available()
        || !state.weapon_ready(ActionKind::Beams)
        || !state.enemies.iter().any(|e| e.distance > MELEE_REACH && e.is_active() && !e.is_dying())
    {
        return Vec::new();
    }
    let grid = build_grid(state, &decode_room_state(state.ram()));
    Direction::CARDINALS.into_iter().filter(|&d| shot_meets(state, &grid, d).is_some()).collect()
}

/// The step fired a sword shot: a BEAMS action from a state that could, and the shot's slot holds a flying sword shot
/// after it.
pub fn fired(change: &Change) -> bool {
    change.action.kind == ActionKind::Beams
        && change.previous.are_beams_available()
        && change.state.animation_state(AnimationKind::Beams) != AnimationState::Inactive
}

/// The step's sword shot damaged a monster. The shot flew during the step (in flight before it, or made by it:
/// [`fired`]) and stopped in it: slot $0E left state $10. A shot that overlaps a monster stops there, damaged or
/// not (asm:SwordShot_CheckMonsterCollision calls asm:SwordShot_HandleBlocked: $10 becomes $11, the spread), and
/// so does one at the room's bounds (asm:SwordShot_UpdateOrMagicShot); what tells the two apart is a monster that
/// lost hit points on the step (`Change::observed`, asm:Damage_DealToMonster) where the sword itself could not
/// reach it: out of melee range ([`MELEE_REACH`]), or on a step that did not swing it. A shot stops on its first
/// overlap (a spreading shot hits nothing), so it damages one monster at most.
///
/// `Change::observed` knows the hit points lost, not the weapon: a step on which another of Link's weapons was live
/// before or after it (a bomb placed or going off, a fire, an arrow, the boomerang, the rod's shot:
/// [`OTHER_WEAPONS`]) counts no hit, as `kills_other` takes the kills no shot or swing explains. Still ambiguous: a
/// weapon live only inside the step (an arrow loosed and spent within one long step) is not seen, and a shot that
/// did hit on a step with another weapon live is not counted (an undercount).
pub fn shot_hit(change: &Change) -> bool {
    let (prev, curr) = (&change.previous, &change.state);
    let flew = prev.animation_state(AnimationKind::Beams) == AnimationState::Active || fired(change);
    if !flew || curr.animation_state(AnimationKind::Beams) == AnimationState::Active {
        return false;
    }
    if OTHER_WEAPONS
        .iter()
        .any(|&w| [prev, curr].iter().any(|s| s.animation_state(w) != AnimationState::Inactive))
    {
        return false;
    }
    let swung = matches!(change.action.kind, ActionKind::Sword | ActionKind::Beams);
    change.observed.enemies_hit.iter().any(|&(slot, damage)| {
        damage > 0 && prev.enemy_by_index(slot).is_some_and(|e| e.distance > MELEE_REACH || !swung)
    })
}

/// Link's weapons other than the sword and its shot ([`shot_hit`]).
pub const OTHER_WEAPONS: [AnimationKind; 7] = [
    AnimationKind::Magic,
    AnimationKind::Bomb1,
    AnimationKind::Bomb2,
    AnimationKind::Flame1,
    AnimationKind::Flame2,
    AnimationKind::Arrow,
    AnimationKind::Boomerang,
];

/// A visit's sword shots and kills (chain room records and evaluation visits: JSON `beam_<field>`). Kills are the
/// step's observed ones (`Change::observed`: on the step a hit lands), each by where its monster stood: `far`, out of
/// melee range ([`MELEE_REACH`]) while a sword shot flew; `near`, in melee range on a step that swung the sword; the
/// rest `other` (bombs, arrows, fire, the rod, the boomerang on a monster at 0 HP). A step that enters a room counts
/// nothing (the visit's last: the scroll).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BeamStats {
    /// Sword shots fired ([`fired`]), and those the decision's projection had meeting a monster ([`lined_up`] in
    /// the direction fired).
    pub shots: u32,
    pub shots_lined: u32,
    /// Decisions begun lined up for a shot ([`lined_up`] not empty), and those of them that moved or waited instead
    /// (what `penalty-beam-unfired` charges).
    pub lined: u32,
    pub lined_unfired: u32,
    pub kills_far: u32,
    pub kills_near: u32,
    pub kills_other: u32,
    /// Sum of the distances (pixels, corner to corner) of the far and near kills.
    pub kill_distance: u32,
    /// Hearts lost, and steps that cost Link his beams (full hearts before, not after, holding a sword).
    pub hearts_lost: f64,
    pub beams_lost: u32,
    /// Shots that damaged a monster ([`shot_hit`]; round 40).
    pub shots_hit: u32,
    /// Frames counted (the steps' frames with the item sub-screen's, as the visit's `frames`), those of the steps
    /// begun holding the beams (`GameView::holds_beams`: a sword and the hearts asm:SwordShot_Make needs), and the
    /// frame of the visit (counted so) at the end of the first step that began holding them and ended without
    /// (None: kept through the visit, or never held; round 40).
    pub frames: u64,
    pub held_frames: u64,
    pub lost_frame: Option<u64>,
}

impl BeamStats {
    /// Counts one step.
    pub fn update(&mut self, change: &Change) {
        let (prev, curr) = (&change.previous, &change.state);
        if change.entered_room() {
            return;
        }
        self.frames += u64::from(change.frames + change.menu_frames);
        if prev.holds_beams() {
            self.held_frames += u64::from(change.frames + change.menu_frames);
            if !curr.holds_beams() && self.lost_frame.is_none() {
                self.lost_frame = Some(self.frames);
            }
        }
        self.shots_hit += u32::from(shot_hit(change));
        let lined = lined_up(prev);
        let fired = fired(change);
        self.shots += u32::from(fired);
        if !lined.is_empty() {
            self.lined += 1;
            if !fired && matches!(change.action.kind, ActionKind::Move | ActionKind::Wait) {
                self.lined_unfired += 1;
            } else if fired && lined.contains(&change.action.direction) {
                self.shots_lined += 1;
            }
        }
        let shot_flies = [prev, curr].iter().any(|s| s.animation_state(AnimationKind::Beams) == AnimationState::Active);
        let swung = matches!(change.action.kind, ActionKind::Sword | ActionKind::Beams);
        for &slot in &change.observed.enemies_killed {
            let Some(e) = prev.enemy_by_index(slot) else {
                self.kills_other += 1;
                continue;
            };
            if e.distance > MELEE_REACH && shot_flies {
                self.kills_far += 1;
            } else if e.distance <= MELEE_REACH && swung {
                self.kills_near += 1;
            } else {
                self.kills_other += 1;
                continue;
            }
            self.kill_distance += e.distance.round() as u32;
        }
        self.hearts_lost += change.health_lost;
        self.beams_lost += u32::from(
            change.health_lost > 0.0 && prev.is_health_full() && !curr.is_health_full() && curr.sword() != crate::enums::SwordKind::None,
        );
    }

    /// `{shots, shots_lined, lined, lined_unfired, kills_far, kills_near, kills_other, kill_distance, hearts_lost,
    /// beams_lost, shots_hit, held_frames, lost_frame}` (numbers as floats; `lost_frame` None when the beams were
    /// not lost in the visit). `frames` is the visit's own `frames` and is not repeated.
    pub fn fields(&self) -> [(&'static str, Option<f64>); 13] {
        [
            ("shots", Some(f64::from(self.shots))),
            ("shots_lined", Some(f64::from(self.shots_lined))),
            ("lined", Some(f64::from(self.lined))),
            ("lined_unfired", Some(f64::from(self.lined_unfired))),
            ("kills_far", Some(f64::from(self.kills_far))),
            ("kills_near", Some(f64::from(self.kills_near))),
            ("kills_other", Some(f64::from(self.kills_other))),
            ("kill_distance", Some(f64::from(self.kill_distance))),
            ("hearts_lost", Some(self.hearts_lost)),
            ("beams_lost", Some(f64::from(self.beams_lost))),
            ("shots_hit", Some(f64::from(self.shots_hit))),
            ("held_frames", Some(self.held_frames as f64)),
            ("lost_frame", self.lost_frame.map(|f| f as f64)),
        ]
    }
}

impl AddAssign for BeamStats {
    /// `o` counted after `self` (a visit's steps in order): its `lost_frame` moves by `self.frames`.
    fn add_assign(&mut self, o: BeamStats) {
        if self.lost_frame.is_none() {
            self.lost_frame = o.lost_frame.map(|f| self.frames + f);
        }
        self.frames += o.frames;
        self.held_frames += o.held_frames;
        self.shots_hit += o.shots_hit;
        self.shots += o.shots;
        self.shots_lined += o.shots_lined;
        self.lined += o.lined;
        self.lined_unfired += o.lined_unfired;
        self.kills_far += o.kills_far;
        self.kills_near += o.kills_near;
        self.kills_other += o.kills_other;
        self.kill_distance += o.kill_distance;
        self.hearts_lost += o.hearts_lost;
        self.beams_lost += o.beams_lost;
    }
}
