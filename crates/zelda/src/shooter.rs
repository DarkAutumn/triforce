//! When a shooter fires next (the fresh generation's observation, `env::observation` `entity_shooter`, schema 10;
//! round40.md "The Goriya teacher" and "Every shooter"; the owner's three columns, Nov 26: "Knowing when a Goriya is
//! loaded", and "If the training goes well, apply that (b) not only to goriya but also all enemies that shoot").
//! Arithmetic on the current RAM only: no emulator step, no random byte read. [`throw`] covers:
//!
//! - the Goriyas `$05/$06` ([`goriya`]: the boomerang);
//! - the walking shooters ([`walking`]: Lynels `$01/$02` sword shots, Moblins `$03/$04` arrows, Octoroks `$07-$0A`
//!   rocks): a stand of 32 frames on the shoot timer before each shot;
//! - the Zora `$11` ([`zora`]: one fireball each surfacing), Aquamentus `$3D` ([`aquamentus`]: a volley on its
//!   ObjTimer), Gohma `$33/$34` ([`gohma`]: a fireball every 65 frames) and the red Wizzrobe `$24` ([`red_wizzrobe`]:
//!   one magic shot each appearance): shots on a timer, aimed or along the facing, the same three columns.
//!
//! Not covered: the blue Wizzrobe `$23` (its shot frame is the frame counter's, but whether it fires there depends on
//! its own walk over the room's tiles, a random teleport and Link in its 16-pixel row band (`Y AND $F0` equal) or
//! exactly on its column (Link's X its X rounded down to 16): round40.md "Every shooter"), and the other fireball
//! shooters (Manhandla, Gleeok's heads, Ganon, the statues).
//!
//! The columns read the same for every type: `loaded` (it holds its next shot and can act), `throw_time` (1 − the
//! frames until its timer lets it shoot / [`TIMER_MAX`], at least 0; 0 when not loaded) and `will_throw` (the chance
//! it shoots within the next [`HORIZON`] frames, the room as it stands).
//!
//! **The Goriya's rule** (asm:Goriya_Update, asm:Walker_SetInputDirAndTryThrowBoomerang, asm:Shooter_ShootIfWanted,
//! asm:Shooter_ShootLimited): every frame a Goriya that is not waiting for its boomerang (ObjState bit 7) moves
//! (asm:Walker_Move: not while stunned or under the clock; a knockback moves it instead and ends its update), and at
//! each grid point sets "wants to shoot" (`$412`) when the longer axis to the chase target is under `$51` pixels and
//! clears it otherwise. Then it throws when its ObjTimer (`$28`) is 0, it wants to shoot, there is no clock and no
//! stun, a monster slot `$01-$0B` is free and fewer than 4 counted shots fly (`$34C`); a red one (`$06`) besides only
//! on a frame whose `Random,X` is `$23` or `$77` (2 of 256). The NMI counts ObjTimer down before the update reads it
//! (asm:Nmi_Isr), so a timer of 1 now throws on the next frame. The catch sets the timer to `$30`, `$50` or `$70`.
//!
//! A red Goriya's random byte is the game's register (`$18-$24`), which the game shifts by a fixed rule; the column
//! gives its throw as the chance a fair byte would, as the owner's table asks ("its chance of throwing within the
//! next decision"), not the byte the register will hold.
//!
//! **The walking shooters' rule** (asm:Shooter_RunShootTimer, asm:Shooter_ShootAndSetTimer): the shot is a stand on
//! ObjShootTimer (`$451`). With the timer at 0 and "wants to shoot" set it loads `$30` and the shooter stands; the
//! update counts it down by one a frame (not the NMI) and the shot leaves on the frame it reaches `$10`, 32 frames
//! after the stand began, then it stands 15 frames more (`$0F` to 1; it walks at 0). "Wants to shoot" is set only
//! while it walks (a wanderer's turn at a grid point, asm:Wanderer_MoveAndTurn: toward the chase target in line when
//! its turn rate is at least `Random+1,X`, or perpendicular when its turn timer ran out; a Lynel's Goriya rule at a
//! grid point), so it holds through the stand. A red type (`$02`, `$04`, `$07`, `$08`) starts a stand only on a frame
//! whose `Random,X` is `$F8` or more (8 of 256): a stand starts 33 frames or more before its shot, so no random byte
//! falls inside the horizon and `will_throw` is 1 or 0 for both colours. The shot is lost when, on its frame, the clock
//! or a stun holds, no slot is free or 4 counted shots fly; a hit (the invincibility timer, counted down on even frames
//! before the update, asm:Obj_DecInvincibilityTimer) puts the timer back to 0.

use crate::chase;

/// asm:ObjType ($34F + slot).
const OBJ_TYPE: usize = 0x34F;
/// asm:ObjX / asm:ObjY ($70 / $84 + slot), asm:ObjDir ($98 + slot): 1 right, 2 left, 4 down, 8 up.
const OBJ_X: usize = 0x70;
const OBJ_Y: usize = 0x84;
const OBJ_DIR: usize = 0x98;
/// asm:ObjGridOffset ($394 + slot): signed pixels from the last grid point.
const OBJ_GRID_OFFSET: usize = 0x394;
/// asm:ObjState ($AC + slot): a Goriya's bit 7 while its boomerang is out.
const OBJ_STATE: usize = 0xAC;
/// asm:ObjTimer ($28 + slot): frames to a Goriya's next throw (counted down in the NMI, before the update).
const OBJ_TIMER: usize = 0x28;
/// asm:ObjMetastate ($405 + slot): 0 normal, else spawning or dying (no update runs).
const OBJ_METASTATE: usize = 0x405;
/// asm:ObjStunTimer ($3D + slot): a long timer, 10-frame units.
const OBJ_STUN: usize = 0x3D;
/// asm:ObjShoveDir ($C0 + slot): bit 7 on a knockback's first frame; asm:ObjShoveDistance ($D3 + slot): pixels left.
const OBJ_SHOVE_DIR: usize = 0xC0;
const OBJ_SHOVE_DISTANCE: usize = 0xD3;
/// asm:Shooter_ObjWantsToShoot ($412 + slot).
const OBJ_WANTS_TO_SHOOT: usize = 0x412;
/// asm:InvClock ($66C): nothing moves or throws while it is held.
const INV_CLOCK: usize = 0x66C;
/// asm:MonsterShotCount ($34C): counted shots in flight; asm:Shooter_ShootLimited allows at most 4.
const MONSTER_SHOT_COUNT: usize = 0x34C;
const SHOT_LIMIT: u8 = 4;
/// LongTimerCycle ($26): the long timers tick on the frame that starts with it at 0 (asm:Nmi_Isr@UpdateTimers).
const LONG_TIMER_CYCLE: usize = 0x26;
/// The longest throw timer, set at the catch (`$70`): `throw_time` reads 0 there.
pub const TIMER_MAX: f32 = 112.0;
/// A red Goriya's chance a frame on which the rest of the rule lets it throw: `Random,X` is `$23` or `$77`.
pub const RED_CHANCE: f32 = 2.0 / 256.0;
/// Frames `will_throw` looks ahead: the threat block's horizon, about one decision.
pub const HORIZON: u32 = crate::threat::HORIZON as u32;
/// asm:ObjShootTimer ($451 + slot): a walking shooter's stand, counted down in its update.
const OBJ_SHOOT_TIMER: usize = 0x451;
/// The shoot timer's value on the frame the shot leaves, and its load at the start of the stand.
const SHOT_LEAVES: u8 = 0x10;
const STAND: u8 = 0x30;
/// asm:ObjInvincibilityTimer ($4F0 + slot): counted down on even frames, before the slot's update.
const OBJ_INVINCIBILITY: usize = 0x4F0;
/// asm:FrameCounter ($15): incremented in the NMI before the frame's updates.
const FRAME_COUNTER: usize = 0x15;
/// asm:Gohma_ObjShootTimer ($380 + slot): frames to Gohma's next fireball, counted down in its update.
const GOHMA_SHOOT_TIMER: usize = 0x380;
/// asm:ObjUninitialized ($492 + slot): nonzero until the object's init routine runs.
const OBJ_UNINITIALIZED: usize = 0x492;
/// Object types.
const ZORA: u8 = 0x11;
const AQUAMENTUS: u8 = 0x3D;
const RED_WIZZROBE: u8 = 0x24;
/// A Zora's states (ObjState, asm:Burrower_Update): 1 mound, 2 surfacing (`$0F` frames), 3 surfaced (starts at
/// `$FF`; the fireball when its update reads `$FD`).
const ZORA_SURFACING_FRAMES: u32 = 0x0F;
const ZORA_SHOOTS_AT: u8 = 0xFD;
/// A red Wizzrobe's ObjState on the frame it shoots (asm:Wizzrobe_Red_State2Solid); `$FF` the frame it took its
/// place (asm:Wizzrobe_Red_State3Appear).
const RED_WIZZROBE_SHOOTS_AT: u8 = 0xB0;

/// A shooter's next shot ([`throw`]).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Throw {
    /// It holds its next shot and can act (each type's reading in its function): a Goriya live, not waiting for its
    /// boomerang, not stunned, no clock.
    pub loaded: bool,
    /// Its shot timer as it runs down: 1 − the frames until the timer lets it shoot / [`TIMER_MAX`] (a Goriya's
    /// ObjTimer), at least 0; 0 when not `loaded`.
    pub throw_time: f32,
    /// The chance it shoots within the next [`HORIZON`] frames, the room as it stands (the chase target where it is,
    /// no other monster moving): 1 or 0 where the game's rule holds no random byte in the horizon, a red Goriya's
    /// 1 − (1 − [`RED_CHANCE`])^n over the n frames the rest of the rule lets it.
    pub will_throw: f32,
    /// Every condition of the shot but a red Goriya's random byte holds on the next frame: it shoots then.
    pub next: bool,
}

/// The shot of the object in RAM `slot` by its type (`None` for an object no rule covers: the module docstring).
pub fn throw(ram: &[u8], slot: usize) -> Option<Throw> {
    match ram[OBJ_TYPE + slot] {
        0x05 | 0x06 => goriya(ram, slot),
        0x01..=0x04 | 0x07..=0x0A => walking(ram, slot),
        ZORA => zora(ram, slot),
        AQUAMENTUS => aquamentus(ram, slot),
        0x33 | 0x34 => gohma(ram, slot),
        RED_WIZZROBE => red_wizzrobe(ram, slot),
        _ => None,
    }
}

/// A monster slot `$01-$0B` is free (asm:Obj_FindEmptyMonsterSlot) and, for a counted shot (type `$53` or more but
/// a fireball, asm:Shooter_ShootLimited), fewer than 4 counted shots fly; as they stand.
fn room_for_shot(ram: &[u8], counted: bool) -> bool {
    (1..=0x0B).any(|s| ram[OBJ_TYPE + s] == 0) && (!counted || ram[MONSTER_SHOT_COUNT] < SHOT_LIMIT)
}

/// `throw_time` for a shot the timer allows `frames` from now.
fn time_left(frames: u32) -> f32 {
    (1.0 - frames as f32 / TIMER_MAX).max(0.0)
}

/// A shot that comes on frame `at` (`None`: none to come) for sure if the room has space then ([`room_for_shot`] as it
/// stands); `throw_time` counts to it while `loaded`.
fn timed(loaded: bool, at: Option<u32>, room: bool) -> Throw {
    let throw_time = match (loaded, at) {
        (true, Some(at)) => time_left(at),
        _ => 0.0,
    };
    let within = at.is_some_and(|at| (1..=HORIZON).contains(&at)) && room;
    Throw { loaded, throw_time, will_throw: if within { 1.0 } else { 0.0 }, next: room && at == Some(1) }
}

/// Frames until a long timer holding `value` reads 0 at the update (0: it does now).
fn long_timer_frames(ram: &[u8], value: u8) -> u32 {
    if value == 0 {
        return 0;
    }
    (ram[LONG_TIMER_CYCLE] as u32).min(9) + 1 + 10 * (value as u32 - 1)
}

/// Where a kept knockback ends, its pixels walked as asm:Shove_MovePixels moves them (4 a frame; the grid offset moved
/// with each and set to 0 at a multiple of 16; walls and room bounds, which end it sooner, not walked): the corner
/// and whether the grid offset is 0 there.
fn shove_end_point(ram: &[u8], slot: usize) -> (u8, u8, bool) {
    let dir = ram[OBJ_SHOVE_DIR + slot] & 0x0F;
    let step: u8 = if dir & 0x05 != 0 { 1 } else { 0xFF };
    let (mut x, mut y) = (ram[OBJ_X + slot], ram[OBJ_Y + slot]);
    let mut offset = ram[OBJ_GRID_OFFSET + slot];
    for _ in 0..(ram[OBJ_SHOVE_DISTANCE + slot] as u32).div_ceil(4) * 4 {
        offset = offset.wrapping_add(step);
        if offset & 0x0F == 0 {
            offset = 0;
        }
        if dir & 0x03 != 0 {
            x = x.wrapping_add(step);
        } else {
            y = y.wrapping_add(step);
        }
    }
    (x, y, offset == 0)
}

/// The throw of the Goriya in RAM `slot` (`None` for any other object). The frames walked, `t` = 1 the next frame:
/// - the first frame its update can reach the throw: after the stun ([`long_timer_frames`]) and after a knockback
///   the game keeps (its first frame, then 4 pixels a frame, then the frame that ends it; a wall ending it sooner is
///   not walked). A knockback across the Goriya's facing axis off a grid point is dropped on its first frame
///   (asm:Shove_Update), and that frame's update goes on to the throw;
/// - "wants to shoot": the flag as it stands until the frame it reaches its next grid point
///   ([`chase::next_grid_point`], delayed by the stun and a dropped knockback's frame), from then the Goriya rule
///   there ([`chase::goriya_facing`]) against the chase target as it stands; a kept knockback keeps the flag unless it
///   ends on a grid point, where the frame that ends it decides again ([`shove_end_point`]);
/// - ObjTimer `T` reads 0 from frame `T` on;
/// - a free monster slot and the shot count as they stand (a shot that ends within the frames, freeing the room for a
///   throw, is not foreseen);
/// - in its spawn cloud (ObjMetastate 1-3) the cloud's frames instead: it decides on its spawn square the frame after
///   the cloud ends and may throw from the sixth (asm:Obj_AnimateMetastate leaves the timer at 6).
///
/// The test that skips the wait for the boomerang when the next slot holds an Armos (asm:Goriya_Update) is not
/// modelled: no room holds both.
pub fn goriya(ram: &[u8], slot: usize) -> Option<Throw> {
    let obj_type = ram[OBJ_TYPE + slot];
    if !matches!(obj_type, 0x05 | 0x06) {
        return None;
    }
    let waiting = ram[OBJ_STATE + slot] & 0x80 != 0;
    let metastate = ram[OBJ_METASTATE + slot];
    let spawning = (1..=3).contains(&metastate);
    let clock = ram[INV_CLOCK] != 0;
    let stun = ram[OBJ_STUN + slot];
    let timer = ram[OBJ_TIMER + slot];
    let loaded = metastate == 0 && !waiting && stun == 0 && !clock;
    let throw_time = if loaded { (1.0 - timer as f32 / TIMER_MAX).max(0.0) } else { 0.0 };
    if !(metastate == 0 || spawning) || waiting || clock {
        return Some(Throw { loaded, throw_time, will_throw: 0.0, next: false });
    }
    let room = (1..=0x0B).any(|s| ram[OBJ_TYPE + s] == 0) && ram[MONSTER_SHOT_COUNT] < SHOT_LIMIT;
    let (tx, ty) = (ram[chase::CHASE_X], ram[chase::CHASE_Y]);
    // From frame `free` its update can reach the throw, from `fired` its ObjTimer reads 0; "wants to shoot" is
    // `wants_now` until frame `decided.0`, `decided.1` from then.
    let (free, fired, wants_now, decided) = if spawning {
        // The spawn cloud (asm:Obj_AnimateMetastate): each of its three steps ends on the frame ObjTimer reads 0 and
        // loads 6. The frame the last one ends it is live with timer 6, its update first runs the frame after, on its
        // spawn square (a grid point: the Goriya rule decides there), and its timer reads 0 six frames after the end.
        let live = timer as u32 + 6 * (3 - metastate as u32);
        let (x, y) = (ram[OBJ_X + slot], ram[OBJ_Y + slot]);
        (live + 1, live + 6, false, Some((live + 1, chase::goriya_facing(tx, ty, x, y).is_some())))
    } else {
        let stun_end = long_timer_frames(ram, stun).max(1);
        let shove = ram[OBJ_SHOVE_DIR + slot];
        let across = if ram[OBJ_DIR + slot] < 3 { shove & 0x0C != 0 } else { shove & 0x03 != 0 };
        let dropped = shove & 0x80 != 0 && across && ram[OBJ_GRID_OFFSET + slot] != 0;
        let shoved = shove != 0 && !dropped;
        let shove_end = if shoved {
            u32::from(shove & 0x80 != 0) + (ram[OBJ_SHOVE_DISTANCE + slot] as u32).div_ceil(4) + 1
        } else {
            1
        };
        // The end of a kept knockback on a grid point decides again, else its next grid point counted from the first
        // frame it walks again (a dropped knockback's frame moves nothing).
        let decided = if shoved {
            let (x, y, on_grid) = shove_end_point(ram, slot);
            on_grid.then(|| (shove_end, chase::goriya_facing(tx, ty, x, y).is_some()))
        } else {
            chase::next_grid_point(ram, slot).map(|p| {
                (p.frames + stun_end - 1 + u32::from(dropped), chase::goriya_facing(tx, ty, p.x, p.y).is_some())
            })
        };
        (stun_end.max(shove_end), timer as u32, ram[OBJ_WANTS_TO_SHOOT + slot] != 0, decided)
    };
    let qualifies = |t: u32| {
        let wants = match decided {
            Some((at, wants)) if t >= at => wants,
            _ => wants_now,
        };
        room && t >= free && t >= fired && wants
    };
    let p = if obj_type == 0x05 { 1.0 } else { RED_CHANCE };
    let miss: f32 = (1..=HORIZON).filter(|&t| qualifies(t)).map(|_| 1.0 - p).product();
    Some(Throw { loaded, throw_time, will_throw: 1.0 - miss, next: qualifies(1) })
}

/// The walking shooters' shot (Lynels `$01/$02`, Moblins `$03/$04`, Octoroks `$07-$0A`; the module docstring), from
/// ObjShootTimer `T`:
/// - `loaded`: live (ObjMetastate 0), not stunned, no clock;
/// - the frames until the timer lets it shoot: `T − $10` in its stand (`T` above `$10`), 33 at 0 (a stand starting
///   next frame), `T + 33` on its shot's frame and the 15 frames standing after it (`T` from `$10` to 1);
/// - `will_throw`: 1 when its stand releases the shot within the horizon (`T` from `$11` to `$10` + [`HORIZON`]) and on
///   that frame "wants to shoot" still holds (as it stands: no walk in a stand changes it), no clock, the stun reads 0,
///   the room has space as it stands, and its invincibility timer reads 0 at the next frame's update (a hit after this
///   frame's update cancels the stand on the next).
pub fn walking(ram: &[u8], slot: usize) -> Option<Throw> {
    if !matches!(ram[OBJ_TYPE + slot], 0x01..=0x04 | 0x07..=0x0A) {
        return None;
    }
    let live = ram[OBJ_METASTATE + slot] == 0;
    let clock = ram[INV_CLOCK] != 0;
    let stun = ram[OBJ_STUN + slot];
    let timer = ram[OBJ_SHOOT_TIMER + slot];
    let loaded = live && stun == 0 && !clock;
    let wait = match timer {
        0 => 1 + (STAND - SHOT_LEAVES) as u32,
        t if t > SHOT_LEAVES => (t - SHOT_LEAVES) as u32,
        t => t as u32 + 1 + (STAND - SHOT_LEAVES) as u32,
    };
    let inv = ram[OBJ_INVINCIBILITY + slot];
    let next_even = ram[FRAME_COUNTER].wrapping_add(1) & 1 == 0;
    let hit = inv > u8::from(next_even);
    let at = (timer > SHOT_LEAVES).then(|| (timer - SHOT_LEAVES) as u32).filter(|&at| {
        live && !clock && !hit && ram[OBJ_WANTS_TO_SHOOT + slot] != 0 && long_timer_frames(ram, stun) <= at
    });
    let mut t = timed(loaded, at, room_for_shot(ram, true));
    t.throw_time = if loaded { time_left(wait) } else { 0.0 };
    Some(t)
}

/// The Zora's fireball (asm:Zora_Update, asm:Burrower_Update, asm:Fireball_Shoot): one each surfacing, on the frame its
/// update reads ObjTimer `$FD` in state 3, two frames into it (the NMI counts the timer down first; a state ends on the
/// frame its update reads 0). From state 1 (the mound) it comes `max(T, 1) + $0F + 2` frames on, from state 2
/// `max(T, 1) + 2`, in state 3 `T − $FD`. `loaded` while it is to come and there is no clock (its update returns at
/// once then); no hit or stun holds it; a fireball needs only a free slot (not counted).
pub fn zora(ram: &[u8], slot: usize) -> Option<Throw> {
    if ram[OBJ_TYPE + slot] != ZORA {
        return None;
    }
    let timer = ram[OBJ_TIMER + slot];
    let zero = (timer as u32).max(1);
    let at = match ram[OBJ_STATE + slot] {
        1 => Some(zero + ZORA_SURFACING_FRAMES + 2),
        2 => Some(zero + 2),
        3 if timer > ZORA_SHOOTS_AT => Some((timer - ZORA_SHOOTS_AT) as u32),
        _ => None,
    }
    .filter(|_| ram[OBJ_METASTATE + slot] == 0 && ram[INV_CLOCK] == 0);
    Some(timed(at.is_some(), at, room_for_shot(ram, false)))
}

/// Aquamentus's volley (asm:Aquamentus_Shoot): three fireballs on the frame its update reads ObjTimer 0 (`max(T, 1)`
/// frames on), unless the clock holds; each takes a free slot (not counted), so one free slot is enough for a shot.
/// `throw_time` reads ObjTimer as a Goriya's (`1 − T / 112`, reloaded to `$70-$7F` or `$F0-$FF`). In its spawn cloud
/// (ObjMetastate 1-3) the cloud's frames, as a Goriya's: the volley comes six frames after the cloud ends.
pub fn aquamentus(ram: &[u8], slot: usize) -> Option<Throw> {
    if ram[OBJ_TYPE + slot] != AQUAMENTUS {
        return None;
    }
    let metastate = ram[OBJ_METASTATE + slot];
    let timer = ram[OBJ_TIMER + slot];
    let clock = ram[INV_CLOCK] != 0;
    let loaded = metastate == 0 && !clock;
    let at = match metastate {
        0 => Some((timer as u32).max(1)),
        1..=3 => Some(timer as u32 + 6 * (3 - metastate as u32) + 6),
        _ => None,
    }
    .filter(|_| !clock);
    let mut t = timed(loaded, at, room_for_shot(ram, false));
    t.throw_time = if loaded { (1.0 - timer as f32 / TIMER_MAX).max(0.0) } else { 0.0 };
    Some(t)
}

/// Gohma's fireball (asm:Gohma_Update@ShootAnimateCollide): its own shoot timer (`$380`) counted down in its update,
/// the fireball on the frame it reaches 0 (`S` frames on; 256 from 0) and reloaded with `$41`. No clock or stun test:
/// `loaded` while live; a fireball needs only a free slot. Before its init has run (ObjUninitialized, metastate 1 on
/// entering its room) the init sets the timer to 1 (asm:Gohma_Init) and the first fireball comes the frame after: 2
/// frames on, read as loaded.
pub fn gohma(ram: &[u8], slot: usize) -> Option<Throw> {
    if !matches!(ram[OBJ_TYPE + slot], 0x33 | 0x34) {
        return None;
    }
    if ram[OBJ_UNINITIALIZED + slot] != 0 {
        return Some(timed(true, Some(2), room_for_shot(ram, false)));
    }
    let live = ram[OBJ_METASTATE + slot] == 0;
    let timer = ram[GOHMA_SHOOT_TIMER + slot];
    let at = live.then_some(if timer == 0 { 256 } else { timer as u32 });
    Some(timed(live, at, room_for_shot(ram, false)))
}

/// The red Wizzrobe's magic shot (asm:Wizzrobe_Red_Update): its update counts ObjState down by one and shoots on the
/// frame it reads `$B0`, the 16th solid frame, once a cycle; `loaded` from the frame it takes its place (`$FF`) to the
/// shot (ObjState above `$B0`), no clock (its update does nothing then); the shot is counted (`$59`).
pub fn red_wizzrobe(ram: &[u8], slot: usize) -> Option<Throw> {
    if ram[OBJ_TYPE + slot] != RED_WIZZROBE {
        return None;
    }
    let state = ram[OBJ_STATE + slot];
    let loaded = ram[OBJ_METASTATE + slot] == 0 && ram[INV_CLOCK] == 0 && state > RED_WIZZROBE_SHOOTS_AT;
    let at = loaded.then(|| (state - RED_WIZZROBE_SHOOTS_AT) as u32);
    Some(timed(loaded, at, room_for_shot(ram, true)))
}

#[cfg(test)]
mod tests {
    use super::*;

    const OBJ_INPUT_DIR: usize = 0x3F8;
    const OBJ_QSPEED: usize = 0x3BC;

    /// A loaded Goriya of `obj_type` in slot 1, 8 pixels past a grid point walking right at q-speed $20, wanting to
    /// shoot, Link (the chase target) 32 pixels right of it on its row; slot 2 free, no shot flying.
    fn room(obj_type: u8) -> Vec<u8> {
        let mut r = vec![0u8; 0x800];
        r[OBJ_TYPE + 1] = obj_type;
        (r[OBJ_X + 1], r[OBJ_Y + 1]) = (0x40, 0x8D);
        r[OBJ_INPUT_DIR + 1] = 1;
        r[OBJ_QSPEED + 1] = 0x20;
        r[OBJ_GRID_OFFSET + 1] = 8;
        r[OBJ_WANTS_TO_SHOOT + 1] = 1;
        (r[OBJ_X], r[OBJ_Y]) = (0x60, 0x8D);
        (r[chase::CHASE_X], r[chase::CHASE_Y]) = (0x60, 0x8D);
        r
    }

    #[test]
    fn a_blue_goriya_throws_when_its_timer_runs_out() {
        let mut r = room(0x05);
        r[OBJ_TIMER + 1] = 0;
        let t = goriya(&r, 1).unwrap();
        assert_eq!((t.loaded, t.throw_time, t.will_throw, t.next), (true, 1.0, 1.0, true));
        // The NMI counts the timer down before the update reads it: 1 throws on the next frame, 2 the one after.
        r[OBJ_TIMER + 1] = 1;
        assert!(goriya(&r, 1).unwrap().next);
        r[OBJ_TIMER + 1] = 2;
        let t = goriya(&r, 1).unwrap();
        assert!(!t.next && t.will_throw == 1.0);
        // Beyond the horizon: loaded, not throwing yet.
        r[OBJ_TIMER + 1] = 0x30;
        let t = goriya(&r, 1).unwrap();
        assert_eq!((t.loaded, t.will_throw, t.next), (true, 0.0, false));
        assert!((t.throw_time - (1.0 - 48.0 / 112.0)).abs() < 1e-6);
    }

    #[test]
    fn without_the_wish_it_throws_only_after_a_grid_point_with_the_target_in_range() {
        let mut r = room(0x05);
        r[OBJ_WANTS_TO_SHOOT + 1] = 0;
        r[OBJ_GRID_OFFSET + 1] = 4;
        // 12 pixels to its grid point at half a pixel a frame: past the horizon, so no throw yet.
        assert_eq!(goriya(&r, 1).unwrap().will_throw, 0.0);
        // 8 pixels (16 frames): it decides there on the horizon's last frame, Link 32 px off on its row: a throw.
        r[OBJ_GRID_OFFSET + 1] = 8;
        assert_eq!(goriya(&r, 1).unwrap().will_throw, 1.0);
        // 2 pixels away: it decides there on frame 4 with Link 30 px off on its row, and throws.
        r[OBJ_GRID_OFFSET + 1] = 14;
        let t = goriya(&r, 1).unwrap();
        assert_eq!((t.will_throw, t.next), (1.0, false));
        // With Link out of range ($51 px or more on the longer axis) the grid point clears the wish.
        r[OBJ_WANTS_TO_SHOOT + 1] = 1;
        (r[chase::CHASE_X], r[OBJ_X]) = (0xA0, 0xA0);
        let t = goriya(&r, 1).unwrap();
        assert!(t.next && t.will_throw == 1.0, "the wish holds until the grid point");
        r[OBJ_GRID_OFFSET + 1] = 15;
        r[0x3A8 + 1] = 0xF0;                                // the next frame's move carries onto the grid point
        assert!(!goriya(&r, 1).unwrap().next);
    }

    #[test]
    fn a_red_goriya_throws_by_chance_and_none_throws_waiting_stunned_or_crowded() {
        let mut r = room(0x06);
        let t = goriya(&r, 1).unwrap();
        let want = 1.0 - (1.0 - RED_CHANCE).powi(HORIZON as i32);
        assert!(t.next && (t.will_throw - want).abs() < 1e-6, "{}", t.will_throw);
        r[OBJ_TIMER + 1] = 10;
        let late = 1.0 - (1.0 - RED_CHANCE).powi(HORIZON as i32 - 9);
        assert!((goriya(&r, 1).unwrap().will_throw - late).abs() < 1e-6);
        r[OBJ_TIMER + 1] = 0;
        r[OBJ_STATE + 1] = 0x80;
        assert_eq!(goriya(&r, 1).unwrap(), Throw::default());
        r[OBJ_STATE + 1] = 0;
        r[OBJ_STUN + 1] = 0x10;
        let t = goriya(&r, 1).unwrap();
        assert_eq!((t.loaded, t.will_throw, t.next), (false, 0.0, false));
        r[OBJ_STUN + 1] = 1;
        r[LONG_TIMER_CYCLE] = 3;                            // the stun reads 0 on the 4th frame
        let t = goriya(&r, 1).unwrap();
        let after = 1.0 - (1.0 - RED_CHANCE).powi(HORIZON as i32 - 3);
        assert!(!t.next && (t.will_throw - after).abs() < 1e-6);
        r[OBJ_STUN + 1] = 0;
        r[MONSTER_SHOT_COUNT] = 4;
        assert_eq!(goriya(&r, 1).unwrap().will_throw, 0.0);
        r[MONSTER_SHOT_COUNT] = 0;
        for s in 2..=0x0B {
            r[OBJ_TYPE + s] = 0x06;
        }
        assert_eq!(goriya(&r, 1).unwrap().will_throw, 0.0);
        assert!(goriya(&r, 0).is_none() && goriya(&room(0x01), 1).is_none());
    }

    #[test]
    fn a_knockback_across_its_axis_off_the_grid_is_dropped_and_one_ending_on_a_grid_point_decides_again() {
        let mut r = room(0x05);
        r[OBJ_DIR + 1] = 1;
        // A hit from above: shoved down on its first frame 8 px past its grid point walking right. The game drops the
        // shove (asm:Shove_Update) and the same update throws.
        (r[OBJ_SHOVE_DIR + 1], r[OBJ_SHOVE_DISTANCE + 1]) = (0x84, 0x40);
        let t = goriya(&r, 1).unwrap();
        assert!(t.next && t.will_throw == 1.0);
        // Along its axis it is kept: 1 + 16 + 1 frames, past the horizon; across it on a grid point too.
        r[OBJ_SHOVE_DIR + 1] = 0x82;
        assert_eq!((goriya(&r, 1).unwrap().next, goriya(&r, 1).unwrap().will_throw), (false, 0.0));
        (r[OBJ_SHOVE_DIR + 1], r[OBJ_GRID_OFFSET + 1]) = (0x84, 0);
        assert!(!goriya(&r, 1).unwrap().next);
        // 8 px left from 8 px past the grid point ends on it, on frame 3, and decides there: Link 40 px off, a throw.
        (r[OBJ_SHOVE_DIR + 1], r[OBJ_SHOVE_DISTANCE + 1], r[OBJ_GRID_OFFSET + 1]) = (0x02, 8, 8);
        let t = goriya(&r, 1).unwrap();
        assert_eq!((t.next, t.will_throw), (false, 1.0));
        // Link $51 px or more away there: the grid point clears the wish.
        (r[OBJ_X], r[chase::CHASE_X]) = (0xA0, 0xA0);
        assert_eq!(goriya(&r, 1).unwrap().will_throw, 0.0);
        // Ending 4 px off the grid it keeps the wish and throws once the shove is over.
        r[OBJ_SHOVE_DISTANCE + 1] = 4;
        assert_eq!(goriya(&r, 1).unwrap().will_throw, 1.0);
    }

    #[test]
    fn a_goriya_in_its_spawn_cloud_throws_six_frames_after_it_ends() {
        let mut r = room(0x05);
        (r[OBJ_METASTATE + 1], r[OBJ_TIMER + 1], r[OBJ_WANTS_TO_SHOOT + 1]) = (3, 1, 0);
        // The cloud ends next frame, the timer then at 6: it decides on its square and throws on frame 7.
        let t = goriya(&r, 1).unwrap();
        assert_eq!((t.loaded, t.throw_time, t.next, t.will_throw), (false, 0.0, false, 1.0));
        // Its second step with 4 frames left: live on frame 10, the throw on 16, the horizon's last; 5 left: past it.
        (r[OBJ_METASTATE + 1], r[OBJ_TIMER + 1]) = (2, 4);
        assert_eq!(goriya(&r, 1).unwrap().will_throw, 1.0);
        r[OBJ_TIMER + 1] = 5;
        assert_eq!(goriya(&r, 1).unwrap().will_throw, 0.0);
        // Link out of the Goriya rule's reach of its square: no wish, no throw.
        (r[OBJ_METASTATE + 1], r[OBJ_TIMER + 1]) = (3, 1);
        (r[OBJ_X], r[chase::CHASE_X]) = (0xA0, 0xA0);
        assert_eq!(goriya(&r, 1).unwrap().will_throw, 0.0);
    }

    #[test]
    fn a_walking_shooter_shoots_32_frames_into_its_stand_unless_hit_stunned_or_crowded() {
        for obj_type in [0x01u8, 0x02, 0x03, 0x04, 0x07, 0x08, 0x09, 0x0A] {
            let mut r = room(obj_type);
            // Walking (timer 0): its next stand starts next frame at the earliest, the shot 33 frames on.
            let t = walking(&r, 1).unwrap();
            assert_eq!((t.loaded, t.will_throw, t.next), (true, 0.0, false), "{obj_type:#x}");
            assert!((t.throw_time - (1.0 - 33.0 / TIMER_MAX)).abs() < 1e-6);
            assert_eq!(throw(&r, 1), Some(t));
            // In the stand: $11 shoots on the next frame, $20 on the horizon's last, $21 past it.
            r[OBJ_SHOOT_TIMER + 1] = 0x11;
            let t = walking(&r, 1).unwrap();
            assert_eq!((t.loaded, t.throw_time, t.will_throw, t.next), (true, 1.0 - 1.0 / TIMER_MAX, 1.0, true));
            r[OBJ_SHOOT_TIMER + 1] = 0x20;
            assert_eq!((walking(&r, 1).unwrap().will_throw, walking(&r, 1).unwrap().next), (1.0, false));
            r[OBJ_SHOOT_TIMER + 1] = 0x21;
            assert_eq!(walking(&r, 1).unwrap().will_throw, 0.0);
            // Standing after its shot ($10 to 1): the next stand once the timer is out, its shot 33 frames later.
            r[OBJ_SHOOT_TIMER + 1] = 0x10;
            let t = walking(&r, 1).unwrap();
            assert!(t.will_throw == 0.0 && (t.throw_time - (1.0 - 49.0 / TIMER_MAX)).abs() < 1e-6);
        }
        let mut r = room(0x03);
        r[OBJ_SHOOT_TIMER + 1] = 0x14;
        assert_eq!(walking(&r, 1).unwrap().will_throw, 1.0);
        // The wish cleared (a failed shot keeps it; nothing else clears it in a stand): no shot.
        r[OBJ_WANTS_TO_SHOOT + 1] = 0;
        assert_eq!(walking(&r, 1).unwrap().will_throw, 0.0);
        r[OBJ_WANTS_TO_SHOOT + 1] = 1;
        // A hit: the invincibility timer still running at the next update puts the timer back to 0. It counts down
        // on even frames before the update: 1 ends there on an even frame, holds on an odd one.
        r[OBJ_INVINCIBILITY + 1] = 1;
        r[FRAME_COUNTER] = 0x41;
        assert_eq!(walking(&r, 1).unwrap().will_throw, 1.0);
        r[FRAME_COUNTER] = 0x40;
        assert_eq!(walking(&r, 1).unwrap().will_throw, 0.0);
        r[OBJ_INVINCIBILITY + 1] = 0;
        // A stun reading 0 by the shot's frame lets it go (the timer runs while stunned), a longer one skips it.
        r[OBJ_STUN + 1] = 1;
        r[LONG_TIMER_CYCLE] = 3;
        let t = walking(&r, 1).unwrap();
        assert_eq!((t.loaded, t.throw_time, t.will_throw), (false, 0.0, 1.0));
        r[LONG_TIMER_CYCLE] = 4;
        assert_eq!(walking(&r, 1).unwrap().will_throw, 0.0);
        r[OBJ_STUN + 1] = 0;
        r[INV_CLOCK] = 1;
        assert_eq!(walking(&r, 1).unwrap(), Throw::default());
        r[INV_CLOCK] = 0;
        r[MONSTER_SHOT_COUNT] = 4;
        assert_eq!(walking(&r, 1).unwrap().will_throw, 0.0);
        r[MONSTER_SHOT_COUNT] = 0;
        r[OBJ_METASTATE + 1] = 2;
        assert_eq!(walking(&r, 1).unwrap(), Throw::default());
        assert!(walking(&room(0x05), 1).is_none() && walking(&room(0x11), 1).is_none());
    }

    #[test]
    fn a_zora_shoots_two_frames_into_its_surfaced_state() {
        let mut r = room(ZORA);
        // State 3 starts at $FF: $FE shoots next frame, $FD has shot (the timer cut to $20 after).
        (r[OBJ_STATE + 1], r[OBJ_TIMER + 1]) = (3, 0xFE);
        let t = zora(&r, 1).unwrap();
        assert_eq!((t.loaded, t.throw_time, t.will_throw, t.next), (true, 1.0 - 1.0 / TIMER_MAX, 1.0, true));
        r[OBJ_TIMER + 1] = 0x20;
        assert_eq!(zora(&r, 1).unwrap(), Throw::default());
        // Surfacing with 14 frames left: state 3 on frame 14, the shot on 16; 15 left: past the horizon.
        (r[OBJ_STATE + 1], r[OBJ_TIMER + 1]) = (2, 14);
        assert_eq!(zora(&r, 1).unwrap().will_throw, 1.0);
        r[OBJ_TIMER + 1] = 15;
        let t = zora(&r, 1).unwrap();
        assert!(t.loaded && t.will_throw == 0.0 && (t.throw_time - (1.0 - 17.0 / TIMER_MAX)).abs() < 1e-6);
        // The mound: 17 frames after it ends.
        (r[OBJ_STATE + 1], r[OBJ_TIMER + 1]) = (1, 0x20);
        assert!((zora(&r, 1).unwrap().throw_time - (1.0 - 49.0 / TIMER_MAX)).abs() < 1e-6);
        // Diving (4, 5) or made this frame (0): none to come from this object.
        for state in [0u8, 4, 5] {
            r[OBJ_STATE + 1] = state;
            assert_eq!(zora(&r, 1).unwrap(), Throw::default());
        }
        // A full object table holds the fireball; a full count of shots does not (fireballs are not counted).
        (r[OBJ_STATE + 1], r[OBJ_TIMER + 1]) = (3, 0xFF);
        r[MONSTER_SHOT_COUNT] = 4;
        assert_eq!(zora(&r, 1).unwrap().will_throw, 1.0);
        for s in 2..=0x0B {
            r[OBJ_TYPE + s] = 0x55;
        }
        assert_eq!(zora(&r, 1).unwrap().will_throw, 0.0);
        r = room(ZORA);
        (r[OBJ_STATE + 1], r[OBJ_TIMER + 1], r[INV_CLOCK]) = (3, 0xFF, 1);
        assert_eq!(zora(&r, 1).unwrap(), Throw::default());
    }

    #[test]
    fn aquamentus_gohma_and_the_red_wizzrobe_shoot_on_their_timers() {
        let mut r = room(AQUAMENTUS);
        r[OBJ_TIMER + 1] = 1;
        let t = aquamentus(&r, 1).unwrap();
        assert_eq!((t.loaded, t.will_throw, t.next), (true, 1.0, true));
        assert!((t.throw_time - (1.0 - 1.0 / TIMER_MAX)).abs() < 1e-6);
        r[OBJ_TIMER + 1] = 0x11;
        assert_eq!(aquamentus(&r, 1).unwrap().will_throw, 0.0);
        r[OBJ_TIMER + 1] = 0xF0;
        assert_eq!(aquamentus(&r, 1).unwrap().throw_time, 0.0);
        // The cloud's last step with 4 frames left: the volley 10 frames on; its second step: 6 more.
        (r[OBJ_METASTATE + 1], r[OBJ_TIMER + 1]) = (3, 4);
        let t = aquamentus(&r, 1).unwrap();
        assert_eq!((t.loaded, t.throw_time, t.will_throw), (false, 0.0, 1.0));
        r[OBJ_METASTATE + 1] = 2;
        assert_eq!(aquamentus(&r, 1).unwrap().will_throw, 1.0);
        r[OBJ_TIMER + 1] = 5;
        assert_eq!(aquamentus(&r, 1).unwrap().will_throw, 0.0);
        (r[OBJ_METASTATE + 1], r[OBJ_TIMER + 1], r[INV_CLOCK]) = (0, 1, 1);
        assert_eq!(aquamentus(&r, 1).unwrap(), Throw::default());

        for obj_type in [0x33u8, 0x34] {
            let mut r = room(obj_type);
            r[GOHMA_SHOOT_TIMER + 1] = 1;
            let t = gohma(&r, 1).unwrap();
            assert_eq!((t.loaded, t.will_throw, t.next), (true, 1.0, true));
            r[GOHMA_SHOOT_TIMER + 1] = 0x41;
            let t = gohma(&r, 1).unwrap();
            assert!(t.will_throw == 0.0 && (t.throw_time - (1.0 - 65.0 / TIMER_MAX)).abs() < 1e-6);
            // No clock test: it shoots under the clock too.
            (r[GOHMA_SHOOT_TIMER + 1], r[INV_CLOCK]) = (16, 1);
            assert_eq!(gohma(&r, 1).unwrap().will_throw, 1.0);
            // On entering its room, before its init (timer 0, metastate 1): the init sets 1, the fireball 2 frames on.
            (r[GOHMA_SHOOT_TIMER + 1], r[INV_CLOCK], r[OBJ_METASTATE + 1], r[OBJ_UNINITIALIZED + 1]) = (0, 0, 1, 0xFF);
            let t = gohma(&r, 1).unwrap();
            assert_eq!((t.loaded, t.throw_time, t.will_throw, t.next), (true, 1.0 - 2.0 / TIMER_MAX, 1.0, false));
        }

        let mut r = room(RED_WIZZROBE);
        r[OBJ_STATE + 1] = 0xB1;
        let t = red_wizzrobe(&r, 1).unwrap();
        assert_eq!((t.loaded, t.will_throw, t.next), (true, 1.0, true));
        r[OBJ_STATE + 1] = 0xFF;
        let t = red_wizzrobe(&r, 1).unwrap();
        assert!(t.loaded && t.will_throw == 0.0 && (t.throw_time - (1.0 - 79.0 / TIMER_MAX)).abs() < 1e-6);
        // Shot, fading out, hidden: not loaded.
        for state in [0xB0u8, 0x7F, 0x20, 0x00] {
            r[OBJ_STATE + 1] = state;
            assert_eq!(red_wizzrobe(&r, 1).unwrap(), Throw::default());
        }
        r[OBJ_STATE + 1] = 0xB8;
        r[MONSTER_SHOT_COUNT] = 4;
        assert_eq!(red_wizzrobe(&r, 1).unwrap().will_throw, 0.0);
        assert!(throw(&room(0x23), 1).is_none() && throw(&room(0x55), 1).is_none());
    }
}
