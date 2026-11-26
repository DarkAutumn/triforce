//! An attack that drops the shield into a shot (round40.md "The Goriya teacher"; the owner, Nov 26: "We should
//! penalize attacking near projectiles if there's no enemies near the agent, and let the getting-hit penalty do the
//! rest if there are nearby enemies"): `reward.exposed_attack_penalty` (`env::config::RewardConfig`). Arithmetic on
//! the decision's state alone: no emulator step, nothing looked ahead.
//!
//! A swing or an item use puts Link's state at `$10` or more, which turns the shield off (asm:Link_CheckMonsterTouch_NoReset)
//! for frames 0-12 of a sword swing and 0-11 of a bomb, arrow or boomerang ([`lockout`]; zelda-asm items.md
//! "Shield"). The owner's two proximities, both from Link's whole box:
//!
//! - **A shot is near** ([`shot_near`]) when, flying straight at Link at its own speed, it would pass the game's
//!   touch test (middles within 9 pixels on both axes: Link's 16 x 16 box against the shot's) before the lockout ends:
//!   the gap the touch test leaves, max(|dx|, |dy|) − 9, under the shot's speed × the lockout's frames. Only a shot
//!   that can harm him (flying out, asm:Link_CheckMonsterTouch_NoReset), only when Link's invincibility timer and the
//!   clock let a touch count before the lockout ends; its direction is not read ("imagine the projectile was on target
//!   to hit me").
//! - **A monster is near** ([`monster_near`]) when, walking straight at Link at its own speed, it would be in the
//!   sword's reach on a hitting frame of a swing in one of the four directions: the sword's point (asm:Sword_LinkToWeaponOffsetsX/Y
//!   plus (6, 8) facing up or down, (8, 6) left or right) within `$10` pixels of the monster's middle along the
//!   swing and `$0C` across (asm:Collision_CheckStabVsMonster), by the swing's last hitting frame (11; the sword is
//!   out on frames 4-11). The four boxes each hold Link's middle, so a monster walking at him that is in one by frame
//!   11 is in it on a hitting frame.
//! - **The attack is aimed** ([`aimed`]): a sword shot, an arrow or a boomerang thrown in a direction a monster's middle
//!   lies along, within the weapon's hit threshold across (`$0C`, `$0B`, `$0A`) and, for the wooden boomerang, its
//!   reach: a ranged attack at a target is a trade the hit penalty prices, as one at a monster near.
//!
//! [`exposed`]: an attack ([`lockout`]) with a shot near, no monster near and, for a ranged attack, none aimed at.

use crate::contact::{self, TOUCH};
use crate::enums::{ActionKind, Direction};
use crate::game::GameView;

/// asm:ObjQSpeedFrac ($3BC + slot): pixels a frame × 64.
const OBJ_QSPEED: usize = 0x3BC;
/// asm:ObjAttr ($4BF + slot).
const OBJ_ATTR: usize = 0x4BF;
/// asm:InvClock ($66C): no touch is tested while it is held.
const INV_CLOCK: usize = 0x66C;
/// asm:InvMagicBoomerang ($675): the magic boomerang flies to the room's edge.
const INV_MAGIC_BOOMERANG: usize = 0x675;
/// The frames the shield is down from the press: a sword swing's Link state `$11`/`$31` on frames 0-12.
pub const SWORD_LOCKOUT: u32 = 13;
/// A bomb, arrow or boomerang's `$11`/`$31` on frames 0-11.
pub const ITEM_LOCKOUT: u32 = 12;
/// The sword's last hitting frame (state 2 on frames 4-11).
pub const STRIKE_LAST: f32 = 11.0;
/// The sword's point from Link's middle in each direction (asm:Sword_LinkToWeaponOffsetsX/Y, the stab's point) and
/// the half-sizes of its hit box along and across the swing (`$10`, `$0C`).
const SWORD_POINTS: [(Direction, (f32, f32)); 4] = [
    (Direction::N, (-3.0, -10.0)),
    (Direction::S, (-1.0, 13.0)),
    (Direction::W, (-11.0, 1.0)),
    (Direction::E, (11.0, 1.0)),
];
const STAB_ALONG: f32 = 16.0;
const STAB_ACROSS: f32 = 12.0;
/// The wooden boomerang's reach from Link's middle: `$10` ahead, `$31` at 3 px a frame (17 frames, 51 px), then
/// `$10` frames at 1 px a frame.
const WOODEN_BOOMERANG_REACH: f32 = 16.0 + 51.0 + 16.0;

/// Frames the shield is down for an attack of `kind`; `None` for a kind that is no attack (MOVE, WAIT, CLIP,
/// RECORDER, FOOD, CANDLE).
pub fn lockout(kind: ActionKind) -> Option<u32> {
    match kind {
        ActionKind::Sword | ActionKind::Beams => Some(SWORD_LOCKOUT),
        ActionKind::Bomb | ActionKind::Arrow | ActionKind::Boomerang => Some(ITEM_LOCKOUT),
        _ => None,
    }
}

/// Pixels a frame a shot of `obj_type` flies (zelda-asm enemies.md): rocks, sword and magic shots 3 (`$54` 3.5),
/// fireballs 1.75 along one axis, the arrow 2, the boomerang 2.5 out; `None` for a type that is no shot.
pub fn shot_speed(obj_type: u8) -> Option<f32> {
    Some(match obj_type {
        0x53 | 0x57..=0x5A => 3.0,
        0x54 => 3.5,
        0x55 | 0x56 => 1.75,
        0x5B => 2.0,
        0x5C => 2.5,
        _ => return None,
    })
}

/// Link's middle (corner + 8).
fn link_middle(view: &GameView) -> (f32, f32) {
    (view.link.position.x as f32 + 8.0, view.link.position.y as f32 + 8.0)
}

/// The middle of the object in RAM `slot` at corner `(x, y)` (asm:Obj_GetMiddle).
fn middle(view: &GameView, slot: u8, x: i32, y: i32) -> (f32, f32) {
    (x as f32 + contact::middle_x(view.ram()[OBJ_ATTR + slot as usize]), y as f32 + 8.0)
}

/// Whether some shot is near for a lockout of `frames` (the module's rule).
pub fn shot_near(view: &GameView, frames: u32) -> bool {
    let ram = view.ram();
    let invincible = ram[crate::threat::OBJ_INVINCIBILITY_TIMER] as u32;
    if ram[INV_CLOCK] != 0 || invincible >= frames {
        return false;
    }
    let (lx, ly) = link_middle(view);
    view.projectiles.iter().any(|p| {
        let Some(speed) = shot_speed(p.id) else {
            return false;
        };
        if !contact::harms_on_touch(ram, p.index as usize) {
            return false;
        }
        let (mx, my) = middle(view, p.index, p.position.x, p.position.y);
        let gap = (mx - lx).abs().max((my - ly).abs()) - TOUCH;
        gap < speed * frames as f32
    })
}

/// Whether some monster is near (the module's rule): one of `GameView::enemies` (the monsters and Gleeok's attached
/// heads), active (a surfaced Zora or Leever, a Wallmaster out of the wall, else spawned), not dying.
pub fn monster_near(view: &GameView) -> bool {
    let ram = view.ram();
    let (lx, ly) = link_middle(view);
    view.enemies.iter().any(|e| {
        if !e.is_active() || e.is_dying() {
            return false;
        }
        let (mx, my) = middle(view, e.index, e.position.x, e.position.y);
        let (dx, dy) = (mx - lx, my - ly);
        let dist = dx.hypot(dy);
        let speed = ram[OBJ_QSPEED + e.index as usize] as f32 / 64.0;
        let walk = (speed * STRIKE_LAST).min(dist);
        let (rx, ry) = if dist > 0.0 { (dx - dx / dist * walk, dy - dy / dist * walk) } else { (0.0, 0.0) };
        SWORD_POINTS.iter().any(|&(d, (px, py))| {
            let (along, across) = if matches!(d, Direction::N | Direction::S) { (ry - py, rx - px) } else { (rx - px, ry - py) };
            along.abs() < STAB_ALONG && across.abs() < STAB_ACROSS
        })
    })
}

/// Whether an attack of `kind` in `direction` is aimed at a monster (the module's rule; never for SWORD, BOMB).
pub fn aimed(view: &GameView, kind: ActionKind, direction: Direction) -> bool {
    let (threshold, reach) = match kind {
        ActionKind::Beams => (12.0, f32::INFINITY),
        ActionKind::Arrow => (11.0, f32::INFINITY),
        ActionKind::Boomerang if view.ram()[INV_MAGIC_BOOMERANG] != 0 => (10.0, f32::INFINITY),
        ActionKind::Boomerang => (10.0, WOODEN_BOOMERANG_REACH),
        _ => return false,
    };
    let d = direction.value();
    let ux = f32::from(d & 1 != 0) - f32::from(d & 2 != 0);
    let uy = f32::from(d & 4 != 0) - f32::from(d & 8 != 0);
    let norm = ux.hypot(uy);
    if norm == 0.0 {
        return false;
    }
    let (ux, uy) = (ux / norm, uy / norm);
    let (lx, ly) = link_middle(view);
    view.enemies.iter().any(|e| {
        if !e.is_active() || e.is_dying() {
            return false;
        }
        let (mx, my) = middle(view, e.index, e.position.x, e.position.y);
        let (dx, dy) = (mx - lx, my - ly);
        let along = dx * ux + dy * uy;
        let across = (dx * uy - dy * ux).abs();
        along > 0.0 && across < threshold && along <= reach
    })
}

/// The owner's penalty's condition: an attack of `kind` in `direction` chosen at `view` with a shot near for its
/// lockout, no monster near and, for a ranged attack, none aimed at. Play only (game mode 5, not a cave).
pub fn exposed(view: &GameView, kind: ActionKind, direction: Direction) -> bool {
    let Some(frames) = lockout(kind) else {
        return false;
    };
    view.mode() == 5
        && shot_near(view, frames)
        && !monster_near(view)
        && !aimed(view, kind, direction)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::RAM_SIZE;

    const OBJ_TYPE: usize = 0x34F;
    const OBJ_X: usize = 0x70;
    const OBJ_Y: usize = 0x84;
    const OBJ_STATE: usize = 0xAC;

    /// Dungeon 1 in play, Link's corner at (0x78, 0x8D), and the objects `(slot, type, x, y, state, q-speed)`.
    fn view(objects: &[(usize, u8, u8, u8, u8, u8)], vars: &[(usize, u8)]) -> GameView {
        let mut ram = [0u8; RAM_SIZE];
        ram[0x10] = 1;
        ram[0x12] = 5;
        (ram[OBJ_X], ram[OBJ_Y]) = (0x78, 0x8D);
        for &(slot, t, x, y, state, q) in objects {
            ram[OBJ_TYPE + slot] = t;
            (ram[OBJ_X + slot], ram[OBJ_Y + slot]) = (x, y);
            ram[OBJ_STATE + slot] = state;
            ram[OBJ_QSPEED + slot] = q;
        }
        for &(addr, value) in vars {
            ram[addr] = value;
        }
        GameView::new(&ram, 0, false, std::sync::Arc::new(crate::GameMap::builtin()))
    }

    #[test]
    fn a_boomerang_is_near_while_it_could_reach_links_box_before_the_shield_is_back() {
        // The gap the touch test leaves: 40 px between middles less 9 = 31 < 2.5 x 13 (a swing), not < 2.5 x 12.
        let v = view(&[(2, 0x5C, 0x78 + 40, 0x8D, 0x10, 0xA0)], &[]);
        assert!(shot_near(&v, SWORD_LOCKOUT) && !shot_near(&v, ITEM_LOCKOUT));
        assert!(exposed(&v, ActionKind::Sword, Direction::N));
        assert!(!exposed(&v, ActionKind::Bomb, Direction::N) && !exposed(&v, ActionKind::Move, Direction::N));
        // Its direction is not read: a diagonal offset counts by its longer axis.
        assert!(shot_near(&view(&[(2, 0x5C, 0x78 - 30, 0x8D + 40, 0x10, 0xA0)], &[]), SWORD_LOCKOUT));
        // Returning (state $5x) it harms nothing; under the clock or Link's invincibility nothing touches him.
        assert!(!shot_near(&view(&[(2, 0x5C, 0x78 + 20, 0x8D, 0x50, 0xA0)], &[]), SWORD_LOCKOUT));
        assert!(!shot_near(&view(&[(2, 0x5C, 0x78 + 20, 0x8D, 0x10, 0xA0)], &[(INV_CLOCK, 1)]), SWORD_LOCKOUT));
        let invincible = view(&[(2, 0x5C, 0x78 + 20, 0x8D, 0x10, 0xA0)], &[(0x4F0, 13)]);
        assert!(!shot_near(&invincible, SWORD_LOCKOUT));
    }

    #[test]
    fn a_monster_is_near_where_walking_at_link_it_meets_a_swing() {
        // A Red Goriya (0.5 px a frame) 40 px right of Link's middle: 34.5 after 11 frames, beyond the E stab's
        // 11 + 16; 32 px right: 26.5, inside.
        let far = view(&[(1, 0x06, 0x78 + 40, 0x8D, 0, 0x20), (2, 0x5C, 0x78, 0x8D - 30, 0x10, 0xA0)], &[]);
        assert!(!monster_near(&far) && exposed(&far, ActionKind::Sword, Direction::N));
        let near = view(&[(1, 0x06, 0x78 + 32, 0x8D, 0, 0x20), (2, 0x5C, 0x78, 0x8D - 30, 0x10, 0xA0)], &[]);
        assert!(monster_near(&near) && !exposed(&near, ActionKind::Sword, Direction::N));
        // Above Link the N stab reaches 10 + 16 from his middle; a standing monster 30 px up is out, 25 in.
        assert!(!monster_near(&view(&[(1, 0x06, 0x78, 0x8D - 30, 0, 0)], &[])));
        assert!(monster_near(&view(&[(1, 0x06, 0x78, 0x8D - 25, 0, 0)], &[])));
        // A monster in its spawn cloud (metastate 1) is none.
        assert!(!monster_near(&view(&[(1, 0x06, 0x78 + 20, 0x8D, 0, 0x20)], &[(0x405 + 1, 1)])));
    }

    #[test]
    fn a_ranged_attack_lined_up_on_a_monster_is_aimed() {
        let objects = [(1, 0x06, 0x78 + 100, 0x8D + 6, 0, 0x20), (2, 0x5C, 0x78, 0x8D - 30, 0x10, 0xA0)];
        let v = view(&objects, &[]);
        assert!(aimed(&v, ActionKind::Arrow, Direction::E) && !exposed(&v, ActionKind::Arrow, Direction::E));
        assert!(!aimed(&v, ActionKind::Arrow, Direction::W) && exposed(&v, ActionKind::Arrow, Direction::W));
        // The wooden boomerang falls short of it, the magic one does not; the sword's swing is never aimed.
        assert!(!aimed(&v, ActionKind::Boomerang, Direction::E));
        assert!(aimed(&view(&objects, &[(INV_MAGIC_BOOMERANG, 1)]), ActionKind::Boomerang, Direction::E));
        assert!(!aimed(&v, ActionKind::Sword, Direction::E) && exposed(&v, ActionKind::Sword, Direction::E));
    }
}
