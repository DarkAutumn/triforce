//! Round 40 (Darknut40; the owner's Darknut lesson, docs/owner/darknuts.md): the two predicates behind the switched-off
//! proposals for the fresh generation's darknut training, each read from the game state alone.
//!
//! - [`chasing`]: "You never try to get to a darknut moving away from you on your row by following them directly."
//!   A Darknut whose next grid-point decision would turn it straight back onto Link: [`crate::chase::TurnAt`]'s
//!   `reverse_link`, the same condition the observation's `entity_chase` shows (`obs.chase`). At each 16-pixel grid
//!   point a wanderer turns toward its chase target with a chance of 129 in 256 when the target is within 8 pixels of
//!   it on one axis (asm:Wanderer_MoveAndTurn: `CMP #$09`), so Link on its line behind it is met head-on. Read by
//!   `reward.chase_penalty` (`env::config::RewardConfig`).
//! - [`bomb_catch`]: "you always want to hit 2+ darknuts with one bomb." The monsters a bomb laid now would hurt if
//!   each walks on along its facing. Read by `items.bomb_catch` (the BOMB mask, `Session::set_bomb_catch`) and the
//!   count behind `reward.bomb_catch_min` (`env::critic`), which counts the hits the look-ahead credits instead.

use crate::bombs::landing;
use crate::chase::turn_at;
use crate::enums::{Direction, enemy};
use crate::game::{Enemy, GameView};

/// A bomb blasts from the 49th frame after its press (state `$11` for that frame, `$12` for `$30`) for `$18` frames
/// (asm:Bomb_StateTimes); a monster in the blast is hurt when the middles are under `$18` apart on both axes
/// (asm:Bomb_CheckBombOrFireVsMonster).
const BLAST_START: i32 = 0x31;
const BLAST_FRAMES: i32 = 0x18;
const BLAST_REACH: i32 = 0x18;

const OBJ_Q_SPEED: u16 = 0x3BC; // asm:ObjQSpeedFrac: q-speed, pixels a frame = value / 64
const OBJ_INVINCIBILITY_TIMER: u16 = 0x4F0; // asm:ObjInvincibilityTimer, counted every second frame
const OBJ_INVINCIBILITY_MASK: u16 = 0x4B2; // asm:ObjInvincibilityMask
const BOMB_DAMAGE_TYPE: u8 = 0x08; // asm:Bomb_CheckBombOrFireVsMonster
const INV_CLOCK: u16 = 0x66C; // asm:InvClock: monsters stand still (asm:Wanderer_Update)

fn is_darknut(e: &Enemy) -> bool {
    matches!(e.id, enemy::RED_DARKNUT | enemy::BLUE_DARKNUT)
}

/// Some Darknut would reverse onto Link at its next grid point ([`crate::chase::TurnAt::reverse_link`]): Link within
/// 8 pixels of its line there, behind it, at any distance. None while it cannot decide (the clock, a stun, a shove,
/// its spawn cloud or death); a flashing one counts (its turns go on, asm:Wanderer_Update).
pub fn chasing(state: &GameView) -> bool {
    let ram = state.ram();
    state
        .active_enemies()
        .filter(|e| is_darknut(e))
        .any(|e| turn_at(ram, usize::from(e.index)).is_some_and(|t| t.reverse_link()))
}

/// The monsters a BOMB laid now facing `d` (where [`landing`] puts it; 0 where it can't be laid) would hurt if each
/// walks on along its facing at its own q-speed (no turn at a grid point: what can be seen now), standing still with
/// the clock: in the blast's reach on some frame of the blast, out of its flash by then, not immune to bombs (its
/// invincibility mask), and for a Darknut not facing against `d` (its shield: a bomb's direction is Link's facing
/// when he laid it, asm:Damage_HandleWeaponHit@CheckDarknut). Bosses are left out (Manhandla's and Dodongo's bombs
/// are the route's, one at a time). Unlike `crate::beams::shot_meets` a walker is not stopped at a wall: within a
/// blast's 72 frames that walker turns there (its next facing is not visible now), and the count stays the one
/// `scripts/experiments/r40_darknut_proposals.py` reads again from replays.
pub fn bomb_catch(state: &GameView, d: Direction) -> u8 {
    let Some(bomb) = landing(state, d) else {
        return 0;
    };
    let still = state.peek(INV_CLOCK) != 0;
    let caught = state.active_enemies().filter(|e| {
        let slot = u16::from(e.index);
        if enemy::is_boss(e.id) || state.peek(OBJ_INVINCIBILITY_MASK + slot) & BOMB_DAMAGE_TYPE != 0 {
            return false;
        }
        let (vx, vy) = match e.direction {
            Direction::E => (1, 0),
            Direction::W => (-1, 0),
            Direction::S => (0, 1),
            Direction::N => (0, -1),
            _ => (0, 0),
        };
        let speed = if still { 0 } else { i32::from(state.peek(OBJ_Q_SPEED + slot)) };
        let flash = 2 * i32::from(state.peek(OBJ_INVINCIBILITY_TIMER + slot));
        let shielded = is_darknut(e) && e.direction == d.opposite();
        !shielded
            && (BLAST_START.max(flash)..BLAST_START + BLAST_FRAMES).any(|f| {
                let moved = speed * f / 64;
                let (x, y) = (e.position.x + vx * moved, e.position.y + vy * moved);
                (x - bomb.x).abs() < BLAST_REACH && (y - bomb.y).abs() < BLAST_REACH
            })
    });
    caught.count().min(255) as u8
}
