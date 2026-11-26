//! When a shooter fires next (the fresh generation's observation, `env::observation` `entity_shooter`, schema 10;
//! round40.md "The Goriya teacher"; the owner's three columns, Nov 26: "Knowing when a Goriya is loaded"). Filled for
//! the Goriyas `$05/$06` only for now (the owner: "If the training goes well, apply that (b) not only to goriya but
//! also all enemies that shoot"). Arithmetic on the current RAM only: no emulator step, no random byte read.
//!
//! **The game's rule** (asm:Goriya_Update, asm:Walker_SetInputDirAndTryThrowBoomerang, asm:Shooter_ShootIfWanted,
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

/// A shooter's throw ([`goriya`]).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Throw {
    /// It holds its boomerang and can act: live (ObjMetastate 0), not waiting for the boomerang, not stunned, no
    /// clock.
    pub loaded: bool,
    /// Its throw timer as it runs down: 1 − ObjTimer / [`TIMER_MAX`], at least 0; 0 when not `loaded`.
    pub throw_time: f32,
    /// The chance it throws within the next [`HORIZON`] frames, the room as it stands (the chase target where it is,
    /// no other monster moving): blue 1 or 0, red 1 − (1 − [`RED_CHANCE`])^n over the n frames the rest of the rule
    /// lets it.
    pub will_throw: f32,
    /// Every condition of the throw but a red one's random byte holds on the next frame: a blue one throws then.
    pub next: bool,
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
}
