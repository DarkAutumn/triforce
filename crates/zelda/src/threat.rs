//! Range-aware danger in Link's terms (the fresh generation's observation, `env::observation` `entity_threat`;
//! docs/experiments/round40.md "Observation v7's danger block"): when the game's touch test would pass within a
//! horizon of frames, and the owner's **shooter's threat lane**. Arithmetic on the current RAM and tile grid only: no
//! emulator step, nothing looked ahead.
//!
//! **Range-aware** (the owner, Nov 23): a monster or shot that cannot reach Link within the horizon is "not soon", not
//! a far-away hit, so the columns speak in frames to contact and stop at the horizon ([`first_touch`]).
//!
//! **The threat lane** (the owner: "maybe 3 tiles (ish, you measure) in front of the enemies with projectiles [...]
//! standing 1 tile away when their projectile is ready to fire is the right time"): per shooter that fires along its
//! facing ([`shooter`]), whether Link's middle is in the lane its next shot would fly, the frames until that shot
//! could touch him at the shot's speed ([`lane`]), and only while he could not leave the lane first (cut off where
//! he still can): off while the shooter's shot flies (the game lets it shoot again only after).

use crate::contact::{self, TOUCH};
use crate::tiles::Grid;

/// asm:ObjType ($34F + slot).
const OBJ_TYPE: usize = 0x34F;
/// asm:ObjX / asm:ObjY ($70 / $84 + slot).
const OBJ_X: usize = 0x70;
const OBJ_Y: usize = 0x84;
/// asm:ObjDir ($98 + slot): 1 right, 2 left, 4 down, 8 up.
const OBJ_DIR: usize = 0x98;
/// asm:ObjState ($AC + slot): a Goriya's bit 7 is set while it waits for its boomerang.
const OBJ_STATE: usize = 0xAC;
/// asm:ObjTimer ($28 + slot): a Goriya's frames to its next throw (set to $30, $50 or $70 at the catch).
const OBJ_TIMER: usize = 0x28;
/// asm:ObjMetastate ($405 + slot), asm:ObjStunTimer ($3D + slot).
const OBJ_METASTATE: usize = 0x405;
const OBJ_STUN: usize = 0x3D;
/// asm:ObjShootTimer ($451 + slot): a walking shooter stands still from $30 and its shot leaves at $10
/// (asm:Shooter_RunShootTimer).
const OBJ_SHOOT_TIMER: usize = 0x451;
/// asm:ObjAttr ($4BF + slot).
const OBJ_ATTR: usize = 0x4BF;
/// asm:ObjInvincibilityTimer ($4F0 + slot); Link's is slot 0: no touch harms him while it runs.
pub const OBJ_INVINCIBILITY_TIMER: usize = 0x4F0;
/// The shoot timer's value at which the shot leaves.
const SHOT_LEAVES: u8 = 0x10;

/// The horizon the observation uses, in frames: hits are decided in an action's first frames (round 36: half by
/// frame 2, three quarters by frame 4; the bench's choice, round40.md).
pub const HORIZON: usize = 16;
/// Frames past the time Link needs to leave a lane during which a shot arriving still counts (the bench's choice).
pub const LANE_SLACK: f32 = 0.0;
/// A lane's wait when the shooter walks: its next shot needs a stop of 32 frames first (asm:Shooter_RunShootTimer).
pub const WALKING_WAIT: f32 = 33.0;
/// The `near_*` columns' scale: a gap this wide (pixels before the touch test passes) or wider reads 1.
pub const NEAR_SCALE: f32 = 16.0;

/// What a shooter fires along its facing ([`shooter`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shooter {
    /// The shot's speed, pixels a frame (q-speed / 64).
    pub speed: f32,
    /// How far the shot flies, pixels (the boomerang turns back after $51).
    pub reach: f32,
    /// A tile it cannot cross stops the shot (the rock, asm:MonsterShot_Update).
    pub walls: bool,
    /// The Goriya's throw (ObjTimer, ObjState bit 7), not the walking shooters' shoot timer.
    pub goriya: bool,
}

/// The shooters whose shot flies straight along their facing (zelda-asm enemies.md): Lynels `$01/$02` (sword shot
/// `$57`, 3 px a frame, over tiles), Moblins `$03/$04` (arrow `$5B`, 2, over tiles), Goriyas `$05/$06` (boomerang
/// `$5C`, 2.5 for `$51` pixels), Octoroks `$07-$0A` (rock `$53`, 3, stopped by walls). Aimed shots (Zora, Aquamentus,
/// Gleeok, Gohma fireballs) and the Wizzrobes' magic are not lanes: their shots are seen by the contact columns once
/// they exist.
pub fn shooter(obj_type: u8) -> Option<Shooter> {
    let s = |speed, reach, walls, goriya| Some(Shooter { speed, reach, walls, goriya });
    match obj_type {
        0x01 | 0x02 => s(3.0, 256.0, false, false),
        0x03 | 0x04 => s(2.0, 256.0, false, false),
        0x05 | 0x06 => s(2.5, 81.0, false, true),
        0x07..=0x0A => s(3.0, 256.0, true, false),
        _ => None,
    }
}

/// The first frame `t` in `from..=horizon` at which the touch test passes for middles offset by `offset(t)`, and the
/// smallest gap before it passes (max(|dx|, |dy|) − 9, at least 0) over frames `0..=horizon`.
pub fn first_touch(offset: impl Fn(f32) -> (f32, f32), from: usize, horizon: usize) -> (Option<usize>, f32) {
    let mut first = None;
    let mut closest = f32::INFINITY;
    for t in 0..=horizon {
        let (rx, ry) = offset(t as f32);
        let gap = rx.abs().max(ry.abs()) - TOUCH;
        if gap < 0.0 && first.is_none() && t >= from {
            first = Some(t);
        }
        closest = closest.min(gap);
    }
    (first, closest.max(0.0))
}

/// `(horizon + 1 - t) / (horizon + 1)` for a touch at frame `t`, 0 for none: 1 touching now, falling to 0 at the
/// horizon ("not soon").
pub fn soon(t: Option<usize>, horizon: usize) -> f32 {
    t.map_or(0.0, |t| ((horizon + 1 - t.min(horizon + 1)) as f64 / (horizon + 1) as f64) as f32)
}

/// A shooter's lane against Link ([`lane`]).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Lane {
    /// The shooter can fire: live, not stunned, its last shot not flying (a Goriya not waiting for its boomerang; a
    /// walking shooter not between its shot's release and its next walk).
    pub ready: bool,
    /// Frames until its next shot can leave (a walking shooter [`WALKING_WAIT`]).
    pub wait: f32,
    /// Link's middle is in the lane the shot would fly: ahead along the facing, within 9 pixels across, within
    /// the shot's reach (and before a wall for the rock).
    pub inside: bool,
    /// Frames until that shot touches Link where he stands (`None` outside the lane).
    pub arrival: Option<f32>,
    /// Frames Link needs to leave the lane sideways (the nearer side a wall leaves open; infinite when neither).
    pub leave: f32,
}

/// The lane of the shooter in RAM `slot` against Link (corner `$70`, `$84`), the walls from `grid`; `None` for an
/// object that is not a [`shooter`].
pub fn lane(ram: &[u8], slot: usize, grid: &Grid) -> Option<Lane> {
    let obj_type = ram[OBJ_TYPE + slot];
    let s = shooter(obj_type)?;
    let live = ram[OBJ_METASTATE + slot] == 0 && ram[OBJ_STUN + slot] == 0;
    let (ready, wait) = if s.goriya {
        (live && ram[OBJ_STATE + slot] & 0x80 == 0, ram[OBJ_TIMER + slot] as f32)
    } else {
        let timer = ram[OBJ_SHOOT_TIMER + slot];
        if timer > SHOT_LEAVES {
            (live, (timer - SHOT_LEAVES) as f32)
        } else {
            (live && timer == 0, WALKING_WAIT)
        }
    };
    let dir = ram[OBJ_DIR + slot];
    let (fx, fy): (f32, f32) = match dir & 0x0F {
        1 => (1.0, 0.0),
        2 => (-1.0, 0.0),
        4 => (0.0, 1.0),
        8 => (0.0, -1.0),
        _ => (0.0, 0.0),
    };
    let (x, y) = (ram[OBJ_X + slot] as f32, ram[OBJ_Y + slot] as f32);
    let (lx, ly) = (ram[OBJ_X] as f32, ram[OBJ_Y] as f32);
    let (mx, my) = (x + contact::middle_x(ram[OBJ_ATTR + slot]), y + 8.0);
    let (rx, ry) = (lx + 8.0 - mx, ly + 8.0 - my);
    let along = rx * fx + ry * fy;
    let across = rx * fy.abs() + ry * fx.abs();
    // The rock stops at the first tile it cannot cross (the walker's test along its path, to the screen's width).
    let reach = if s.walls { s.reach.min(contact::travel_up_to(x, y, fx, fy, grid, 256)) } else { s.reach };
    let facing = fx != 0.0 || fy != 0.0;
    let inside = ready && facing && along > 0.0 && across.abs() < TOUCH && along - (TOUCH - 1.0) <= reach;
    let arrival = inside.then(|| wait + 1.0 + (along - (TOUCH - 1.0)).max(0.0) / s.speed);
    // Leaving sideways: his middle 9 pixels off the lane's line, along the perpendicular axis, if walls let him.
    let (px, py) = (fy.abs(), fx.abs());
    let mut leave = f32::INFINITY;
    if inside {
        for sign in [1.0f32, -1.0] {
            let need = TOUCH - sign * across;
            if contact::travel(lx, ly, sign * px, sign * py, grid) >= need {
                leave = leave.min((need / contact::LINK_SPEED).ceil());
            }
        }
    }
    Some(Lane { ready, wait, inside, arrival, leave })
}
