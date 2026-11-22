//! The game's monster-touch test in Link-relative terms, and its straight-line extrapolation (observation schema 7,
//! `env::observation` `entity_rel` / `entity_contact`; docs/experiments/round36.md "What the observation lacks" §4,
//! §8). Arithmetic on the current RAM and tile grid only: no emulator step, no look-ahead.
//!
//! The game decides a touch in asm:Link_CheckMonsterTouch_NoReset: Link's middle and the object's middle
//! (asm:Obj_GetMiddle) both within 9 pixels on both axes. Floats are f32 in the order
//! `scripts/experiments/r36_hit_bench.py` computes them in numpy, so the two agree bit for bit
//! (`tests/test_obs_v7.py`).

use crate::tiles::{Grid, TileClass};

/// asm:ObjType ($34F + slot).
const OBJ_TYPE: usize = 0x34F;
/// asm:ObjDir ($98 + slot): 1 right, 2 left, 4 down, 8 up.
const OBJ_DIR: usize = 0x98;
/// asm:ObjState ($AC + slot).
const OBJ_STATE: usize = 0xAC;
/// asm:ObjQSpeedFrac ($3BC + slot): pixels a frame × 64.
const OBJ_QSPEED: usize = 0x3BC;
/// asm:ObjMetastate ($405 + slot): 0 normal, 1-4 spawning, $10-$14 dying.
const OBJ_METASTATE: usize = 0x405;
/// asm:ObjStunTimer ($3D + slot).
const OBJ_STUN: usize = 0x3D;
/// asm:ObjAttr ($4BF + slot): bit $40 is a half-width (8 px) object for asm:Obj_GetMiddle.
const OBJ_ATTR: usize = 0x4BF;
/// The first shot type (asm:ObjType $53 and up): a shot harms only while flying out.
const FIRST_SHOT: u8 = 0x53;

/// The touch test passes below this distance between middles on both axes (asm:Link_CheckMonsterTouch_NoReset).
pub const TOUCH: f32 = 9.0;
/// Frames the contact columns look ahead.
pub const HORIZON: usize = 32;
/// Link's walking speed, pixels a frame (MOVE 4 frames moves him 6).
pub const LINK_SPEED: f32 = 1.5;
/// The furthest a wall stop is searched, pixels.
pub const TRAVEL_LIMIT: u32 = 48;
/// A motion that no wall stops.
pub const UNSTOPPED: f32 = 1000.0;
/// Link standing, then walking N, S, W, E: his velocity sign per axis.
pub const MOTIONS: [(f32, f32); 5] = [(0.0, 0.0), (0.0, -1.0), (0.0, 1.0), (-1.0, 0.0), (1.0, 0.0)];

/// Keese, Gleeok's heads and body, Peahat: they fly over every tile.
const FLIERS: [u8; 9] = [0x1B, 0x1C, 0x1D, 0x46, 0x2F, 0x42, 0x43, 0x44, 0x45];

/// The x offset of an object's middle from its corner (asm:Obj_GetMiddle: 4 for an 8-pixel-wide object, ObjAttr
/// bit $40; else 8). Y is always + 8, and Link's middle is his corner + 8 on both axes.
pub fn middle_x(attr: u8) -> f32 {
    if attr & 0x40 != 0 { 4.0 } else { 8.0 }
}

/// Whether the object in RAM `slot` can harm Link on touch this frame (asm:Link_CheckMonsterTouch_NoReset): a live
/// object, not spawning or dying (ObjMetastate 0), not stunned (ObjStunTimer 0), and a shot only while flying out
/// (ObjState $10-$1F).
pub fn harms_on_touch(ram: &[u8], slot: usize) -> bool {
    let t = ram[OBJ_TYPE + slot];
    t != 0
        && ram[OBJ_METASTATE + slot] == 0
        && ram[OBJ_STUN + slot] == 0
        && (t < FIRST_SHOT || ram[OBJ_STATE + slot] & 0xF0 == 0x10)
}

/// The object's velocity in pixels a frame from its direction and q-speed (ObjDir, ObjQSpeedFrac / 64): what it
/// moves now, not what it moved (a walker's measured motion is stale across a turn).
pub fn velocity(ram: &[u8], slot: usize) -> (f32, f32) {
    let d = ram[OBJ_DIR + slot];
    let s = ram[OBJ_QSPEED + slot] as f32 / 64.0;
    let axis = |plus: u8, minus: u8| (if d & plus != 0 { s } else { 0.0 }) - (if d & minus != 0 { s } else { 0.0 });
    (axis(1, 2), axis(4, 8))
}

/// Whether a wall stops the object in RAM `slot` (a walking monster: not a shot, not a flier).
pub fn walks(ram: &[u8], slot: usize) -> bool {
    let t = ram[OBJ_TYPE + slot];
    t != 0 && t < FIRST_SHOT && !FLIERS.contains(&t)
}

/// The object's type in RAM `slot` (0: empty) and its ObjAttr.
pub fn slot_type(ram: &[u8], slot: usize) -> (u8, u8) {
    (ram[OBJ_TYPE + slot], ram[OBJ_ATTR + slot])
}

/// A tile class a walker crosses: walkable, the four door corridors, stairs.
fn crossable(class: u8) -> bool {
    [TileClass::Walkable, TileClass::DoorN, TileClass::DoorS, TileClass::DoorW, TileClass::DoorE, TileClass::Stairs]
        .iter()
        .any(|&c| c as u8 == class)
}

/// Pixels a 16 x 16 object with its corner at screen `(x, y)` moves along the unit step `(ux, uy)` before its lower
/// half (rows 8-15) meets a tile it cannot cross or the play area's edge, up to [`TRAVEL_LIMIT`].
pub fn travel(x: f32, y: f32, ux: f32, uy: f32, grid: &Grid) -> f32 {
    for s in 1..=TRAVEL_LIMIT {
        let step = s as f32;
        let nx = x + ux * step;
        let ny = (y + uy * step) - 64.0;
        let blocked = [nx, nx + 15.0].iter().any(|&cx| {
            [ny + 8.0, ny + 15.0].iter().any(|&cy| {
                let (c, r) = ((cx / 8.0).floor(), (cy / 8.0).floor());
                !(c >= 0.0 && c < 32.0 && r >= 0.0 && r < 22.0 && crossable(grid[r as usize][c as usize]))
            })
        });
        if blocked {
            return (s - 1) as f32;
        }
    }
    TRAVEL_LIMIT as f32
}

/// How soon (`(33 - t) / 33`, 0 never within [`HORIZON`]) and how close (the smallest gap before the touch test
/// passes, clipped to 0-64, / 32) an object and Link come, `offset(t)` their middles' offset after `t` frames.
pub fn approach(offset: impl Fn(f32) -> (f32, f32)) -> (f32, f32) {
    let mut first = HORIZON + 1;
    let mut closest = f32::INFINITY;
    for t in 0..=HORIZON {
        let (rx, ry) = offset(t as f32);
        let gap = rx.abs().max(ry.abs()) - TOUCH;
        if gap < 0.0 && first > HORIZON {
            first = t;
        }
        closest = closest.min(gap);
    }
    let soon = ((HORIZON + 1 - first) as f64 / (HORIZON + 1) as f64) as f32;
    (soon, closest.clamp(0.0, 64.0) / 32.0)
}
