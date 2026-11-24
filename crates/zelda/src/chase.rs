//! Where the monsters are sent (the fresh generation's observation, `env::observation` `chase` and `entity_chase`;
//! docs/experiments/round40.md "The chase target"). Arithmetic on the current RAM only: no emulator step, nothing
//! looked ahead, no random byte read.
//!
//! **The chase target** (asm:Mode5Play_UpdateWorld@CheckChaseTarget): while `IsChasingOtherTarget` (`$60`) is 0 the
//! target (`ChaseTargetX/Y`, `$61/$62`) is copied from Link's corner every frame; each time `ChaseLongTimer` (`$4A`)
//! is found at 0 it is reloaded with `Random+1 AND 7` and the flag toggles, and on 0 -> 1 the target becomes Link's
//! corner mirrored, (X EOR $FF, Y EOR $FF) = (255 - X, 255 - Y), fixed until the flag returns to 0 (food on the
//! ground overrides both, asm:Boomerang_UpdateOrFood@AttractMonsters). The mirror's fixed point is (127.5, 127.5):
//! where Link stands near it, both targets are on him ([`HOTSPOT`]).
//!
//! **A monster may turn at the target at its next grid point** ([`turn_at`]): the wanderers
//! (asm:Wanderer_MoveAndTurn) and the Goriya walkers (asm:Goriya_Update) decide only on a frame whose move leaves
//! their grid offset a multiple of 16; the frames to that decision and the position it is made at follow from the
//! q-speed, the position fraction and the grid offset exactly as asm:Obj_MoveByQSpeed moves them (walls and room
//! bounds aside).

/// asm:IsChasingOtherTarget ($60): 0 the monsters chase Link, 1 the stored point.
pub const IS_CHASING_OTHER: usize = 0x60;
/// asm:ChaseTargetX / asm:ChaseTargetY ($61 / $62).
pub const CHASE_X: usize = 0x61;
pub const CHASE_Y: usize = 0x62;
/// asm:ChaseLongTimer ($4A): 10-frame units until the flag toggles.
pub const CHASE_LONG_TIMER: usize = 0x4A;
/// LongTimerCycle ($26): counts 9 down to 0; the long timers tick on the frame it wraps (asm:Nmi_Isr@UpdateTimers).
pub const LONG_TIMER_CYCLE: usize = 0x26;
/// The point the mirror leaves fixed, on each axis: 255 - v = v.
pub const HOTSPOT: f32 = 127.5;
/// The longest a flip can be away: a reload of 7 ticks, the first up to 10 frames off (asm:Nmi_Isr@UpdateTimers).
pub const FLIP_MAX: u32 = 70;

/// asm:ObjX / asm:ObjY ($70 / $84 + slot; slot 0 is Link).
const OBJ_X: usize = 0x70;
const OBJ_Y: usize = 0x84;
/// asm:ObjType ($34F + slot).
const OBJ_TYPE: usize = 0x34F;
/// asm:ObjDir ($98 + slot): 1 right, 2 left, 4 down, 8 up.
const OBJ_DIR: usize = 0x98;
/// asm:ObjInputDir ($3F8 + slot): the direction asm:Walker_Move moves in (its lowest set bit).
const OBJ_INPUT_DIR: usize = 0x3F8;
/// asm:ObjState ($AC + slot): a Goriya's bit 7 while its boomerang is out; a Vire's state.
const OBJ_STATE: usize = 0xAC;
/// asm:ObjMetastate ($405 + slot): 0 normal, else spawning or dying.
const OBJ_METASTATE: usize = 0x405;
/// asm:ObjStunTimer ($3D + slot), asm:InvClock ($66C), asm:ObjShoveDir ($C0 + slot).
const OBJ_STUN: usize = 0x3D;
const INV_CLOCK: usize = 0x66C;
const OBJ_SHOVE_DIR: usize = 0xC0;
/// asm:ObjQSpeedFrac ($3BC + slot), asm:ObjPosFrac ($3A8 + slot), asm:ObjGridOffset ($394 + slot).
const OBJ_QSPEED: usize = 0x3BC;
const OBJ_POS_FRAC: usize = 0x3A8;
const OBJ_GRID_OFFSET: usize = 0x394;
/// asm:Wanderer_ObjTurnTimer ($478 + slot): frames until a perpendicular turn is allowed.
const OBJ_TURN_TIMER: usize = 0x478;
/// asm:ObjCaptureTimer ($42C + slot): a Like-Like holding Link (nonzero) runs no wanderer update.
const LIKE_LIKE_CAPTURE: usize = 0x42C;
/// A grid row's Y modulo 16 (rows at $5D, $6D, ... in a dungeon, $3D, $4D, ... on the overworld).
const ROW_PHASE: u8 = 0x0D;

/// The most frames [`turn_at`] walks a monster forward: 16 pixels at the slowest q-speed ($10, a quarter pixel).
pub const MAX_FRAMES: u32 = 64;
/// The most frames counted at one grid point ([`GridPoint::lingers`]).
const MAX_LINGER: u32 = 16;

/// The chase target now ([`chase`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Chase {
    /// `$60`: the monsters chase the stored point, not Link.
    pub other: bool,
    /// `$61`, `$62`.
    pub x: u8,
    pub y: u8,
    /// Frames until the flag toggles ([`flip_frames`]).
    pub flip: u32,
}

/// The chase target and the frames to its next flip.
pub fn chase(ram: &[u8]) -> Chase {
    Chase { other: ram[IS_CHASING_OTHER] != 0, x: ram[CHASE_X], y: ram[CHASE_Y], flip: flip_frames(ram) }
}

/// Frames until the next flip of `$60`, counting the next frame as 1. The NMI ticks every long timer on a frame that
/// starts with LongTimerCycle at 0 (asm:Nmi_Isr@UpdateTimers), before the play update reads `ChaseLongTimer` and
/// flips when it is 0 (asm:Mode5Play_UpdateWorld@UpdateWeapons): timer 0 flips on the next frame; timer T on the
/// frame of its T-th tick, the first `cycle + 1` frames away, the others 10 apart. Play only: the submenu and a
/// pause stop the ticks, and Link's leaving the room skips the flip.
pub fn flip_frames(ram: &[u8]) -> u32 {
    let timer = ram[CHASE_LONG_TIMER] as u32;
    if timer == 0 {
        return 1;
    }
    let cycle = (ram[LONG_TIMER_CYCLE] as u32).min(9);
    cycle + 1 + 10 * (timer - 1)
}

/// How a monster turns at a grid point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Rule {
    /// asm:Wanderer_MoveAndTurn with this turn rate: toward the target when it is within 8 pixels on an axis and
    /// `rate >= Random+1,X` ((rate + 1) / 256 a decision); else perpendicular toward it when the turn timer is out.
    Wanderer(u8),
    /// asm:Goriya_Update: toward the target along the longer axis (the vertical on a tie) when that distance is
    /// under $51 pixels, every time.
    Goriya,
}

/// The grid-point rule of the object in RAM `slot`, when it decides by one now: the type, then its state.
/// - Turn rate $80: Darknuts `$0B/$0C` (asm:Darknut_Update), Gibdo `$30` (asm:Gibdo_Update), Stalfos `$2A`
///   (asm:Stalfos_Update), Like-Like `$17` while it holds no one (asm:LikeLike_Update), Vire `$12` in state 0
///   (asm:Vire_State0Wander).
/// - $40 Bubbles `$2B-$2D` (asm:Bubble_Update); $FF Ghini `$21` (asm:Ghini_Update); $A0 Moblins `$03/$04`
///   (asm:Moblin_Update), fast Octoroks `$09/$0A`, Blue Leever `$0F` (asm:Leever_UpdateBlue); $70 slow Octoroks
///   `$07/$08` (asm:Octorok_Update).
/// - The Goriya rule: Goriyas `$05/$06` (not while the boomerang is out, state bit 7) and Lynels `$01/$02`
///   (asm:Lynel_Update calls asm:Goriya_Update).
/// Left out (round40.md "The chase target"): Zols and Gels (they pause at grid points by a random timer,
/// asm:Zol_WalkAndPause), Dodongos (their X is shifted 16 during the decision, asm:Dodongo_State0Move), Armos.
pub fn rule(ram: &[u8], slot: usize) -> Option<Rule> {
    let state = ram[OBJ_STATE + slot];
    Some(match ram[OBJ_TYPE + slot] {
        0x0B | 0x0C | 0x30 | 0x2A => Rule::Wanderer(0x80),
        0x17 if ram[LIKE_LIKE_CAPTURE + slot] == 0 => Rule::Wanderer(0x80),
        0x12 if state == 0 => Rule::Wanderer(0x80),
        0x2B..=0x2D => Rule::Wanderer(0x40),
        0x21 => Rule::Wanderer(0xFF),
        0x03 | 0x04 | 0x09 | 0x0A | 0x0F => Rule::Wanderer(0xA0),
        0x07 | 0x08 => Rule::Wanderer(0x70),
        0x05 | 0x06 if state & 0x80 == 0 => Rule::Goriya,
        0x01 | 0x02 => Rule::Goriya,
        _ => return None,
    })
}

/// The direction bit an object moves in: the lowest set bit of `input` (asm:Walker_Move@TakeSingleInputDir), 0 none.
fn moving_dir(input: u8) -> u8 {
    let d = input & 0x0F;
    d & d.wrapping_neg()
}

/// The opposite direction bit.
pub fn opposite(dir: u8) -> u8 {
    match dir {
        1 => 2,
        2 => 1,
        4 => 8,
        8 => 4,
        _ => 0,
    }
}

/// The wanderers' homing turn at monster corner `(mx, my)` toward target `(tx, ty)` (asm:Wanderer_MoveAndTurn):
/// |dX| taken as a signed byte's magnitude (a difference of $F8-$FF wraps to 8-1) under 9: up if the target is
/// above, else down; else |dY| likewise under 9: right if the target is at the same X or right, else left.
pub fn homing(tx: u8, ty: u8, mx: u8, my: u8) -> Option<u8> {
    let near = |a: u8, b: u8| (a.wrapping_sub(b) as i8).unsigned_abs() < 9;
    if near(tx, mx) {
        Some(if ty < my { 8 } else { 4 })
    } else if near(ty, my) {
        Some(if tx >= mx { 1 } else { 2 })
    } else {
        None
    }
}

/// The wanderers' turn when the timer is out (asm:Wanderer_Return@TurnIfTime): perpendicular to the facing
/// `dir`, toward the target.
pub fn timer_turn(dir: u8, tx: u8, ty: u8, mx: u8, my: u8) -> u8 {
    if dir & 0x0C != 0 {
        if tx >= mx { 1 } else { 2 }
    } else if ty < my {
        8
    } else {
        4
    }
}

/// The Goriya rule's facing at `(mx, my)` (asm:Goriya_Update): the longer of |dY| and |dX| (unsigned; dY on a tie),
/// toward the target on that axis, when under $51.
pub fn goriya_facing(tx: u8, ty: u8, mx: u8, my: u8) -> Option<u8> {
    let (dy, vdir) = if ty >= my { (ty - my, 4) } else { (my - ty, 8) };
    let (dx, hdir) = if tx >= mx { (tx - mx, 1) } else { (mx - tx, 2) };
    let (d, dir) = if dy >= dx { (dy, vdir) } else { (dx, hdir) };
    (d < 0x51).then_some(dir)
}

/// One frame of asm:Obj_MoveByQSpeed along `dir` (1 right, 2 left, 4 down, 8 up): four sub-steps of the q-speed on
/// the fraction, a pixel on each carry (or borrow) unless the grid offset sits at +16 or -16.
fn step(pos: &mut u8, frac: &mut u8, offset: &mut u8, q: u8, dir: u8) {
    let plus = dir & 0x05 != 0;
    for _ in 0..4 {
        let carry = if plus {
            let (f, c) = frac.overflowing_add(q);
            *frac = f;
            c
        } else {
            let (f, b) = frac.overflowing_sub(q);
            *frac = f;
            b
        };
        if carry && *offset != 0x10 && *offset != 0xF0 {
            if plus {
                *pos = pos.wrapping_add(1);
                *offset = offset.wrapping_add(1);
            } else {
                *pos = pos.wrapping_sub(1);
                *offset = offset.wrapping_sub(1);
            }
        }
    }
}

/// A monster's next grid-point decision ([`next_grid_point`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridPoint {
    /// Frames until the frame of the decision (the next frame is 1).
    pub frames: u32,
    /// Its corner when it decides.
    pub x: u8,
    pub y: u8,
    /// The direction it moves in now.
    pub dir: u8,
    /// The frames it decides on there if it keeps that direction: a frame whose move gives no pixel leaves the offset
    /// at 0, and the update decides again (at q-speed $20 one frame in two).
    pub lingers: u32,
}

/// The next frame on which the object in RAM `slot` will find itself on a grid point, walking on in the direction
/// it moves now (asm:Walker_Move, asm:Obj_MoveByQSpeed; then asm:Wanderer_MoveAndTurn's test, the offset's low four
/// bits 0). `None` with q-speed 0 (no decision is made), or no direction and between grid points. Walls, room bounds
/// and the tile test that turns a blocked monster (asm:Walker_CheckTileCollision) are not walked.
pub fn next_grid_point(ram: &[u8], slot: usize) -> Option<GridPoint> {
    let q = ram[OBJ_QSPEED + slot];
    if q == 0 {
        return None;
    }
    let dir = moving_dir(ram[OBJ_INPUT_DIR + slot]);
    let (mut x, mut y) = (ram[OBJ_X + slot], ram[OBJ_Y + slot]);
    let mut frac = ram[OBJ_POS_FRAC + slot];
    let mut offset = ram[OBJ_GRID_OFFSET + slot];
    if dir == 0 {
        return (offset & 0x0F == 0).then_some(GridPoint { frames: 1, x, y, dir, lingers: MAX_LINGER });
    }
    let frame = |frac: &mut u8, offset: &mut u8, x: &mut u8, y: &mut u8| {
        let pos = if dir & 0x03 != 0 { x } else { y };
        step(pos, frac, offset, q, dir);
    };
    for frames in 1..=MAX_FRAMES {
        frame(&mut frac, &mut offset, &mut x, &mut y);
        if offset & 0x0F == 0 {
            let (at_x, at_y) = (x, y);
            let mut lingers = 1;
            offset = 0;
            while lingers < MAX_LINGER {
                frame(&mut frac, &mut offset, &mut x, &mut y);
                if offset & 0x0F != 0 {
                    break;
                }
                lingers += 1;
            }
            return Some(GridPoint { frames, x: at_x, y: at_y, dir, lingers });
        }
    }
    None
}

/// What the object in RAM `slot` may do at its next grid point ([`turn_at`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TurnAt {
    pub rule: Rule,
    pub point: GridPoint,
    /// The turn toward the current chase target there (its direction bit), when the rule allows one.
    pub toward: Option<u8>,
    /// The chance that turn is taken at that grid point: (rate + 1) / 256 a decision over [`GridPoint::lingers`]
    /// decisions (1 for the Goriya rule); 0 without `toward`.
    pub chance: f32,
    /// The wanderer's turn timer is out by then: without the homing turn it turns perpendicular, toward the target.
    pub timer_turn: Option<u8>,
    /// The same homing test with Link's own corner as the target (what it does there if the target is Link then).
    pub toward_link: Option<u8>,
}

impl TurnAt {
    /// The turn toward the target is a reversal (the target behind it on its line).
    pub fn reverse(&self) -> bool {
        self.toward.is_some_and(|d| d == opposite(self.point.dir))
    }

    /// The turn toward Link would be a reversal: Link behind it on its line (the owner's "don't chase").
    pub fn reverse_link(&self) -> bool {
        self.toward_link.is_some_and(|d| d == opposite(self.point.dir))
    }
}

/// The next grid-point decision of the object in RAM `slot` against the current chase target and against Link, for
/// a live monster with a [`rule`]: not spawning or dying, not knocked back (asm:Shove_Update moves it instead), not
/// frozen by the clock or a stun (asm:Wanderer_Update makes no decision then; the walkers that enter
/// asm:Wanderer_MoveAndTurn directly do not move then).
pub fn turn_at(ram: &[u8], slot: usize) -> Option<TurnAt> {
    let rule = rule(ram, slot)?;
    if ram[OBJ_METASTATE + slot] != 0
        || ram[OBJ_SHOVE_DIR + slot] != 0
        || ram[OBJ_STUN + slot] != 0
        || ram[INV_CLOCK] != 0
    {
        return None;
    }
    let mut point = next_grid_point(ram, slot)?;
    if ram[OBJ_TYPE + slot] == 0x12 && point.dir & 0x03 != 0 {
        // A Vire walking left or right hops: its Y is lifted up to 8 pixels between squares and back on its row at
        // each grid point (asm:Vire_HopYOffsets sum to 0), so the decision is made on the row at or below it.
        point.y = point.y.wrapping_add(ROW_PHASE.wrapping_sub(point.y) & 0x0F);
    }
    let (tx, ty) = (ram[CHASE_X], ram[CHASE_Y]);
    let (lx, ly) = (ram[OBJ_X], ram[OBJ_Y]);
    let (mx, my) = (point.x, point.y);
    Some(match rule {
        Rule::Wanderer(rate) => {
            let toward = homing(tx, ty, mx, my);
            let p = (rate as f32 + 1.0) / 256.0;
            let chance = if toward.is_some() { 1.0 - (1.0 - p).powi(point.lingers as i32) } else { 0.0 };
            let timer = ram[OBJ_TURN_TIMER + slot] as u32;
            let facing = ram[OBJ_DIR + slot];
            let timer_turn = (timer <= point.frames).then(|| timer_turn(facing, tx, ty, mx, my));
            TurnAt { rule, point, toward, chance, timer_turn, toward_link: homing(lx, ly, mx, my) }
        }
        Rule::Goriya => {
            let toward = goriya_facing(tx, ty, mx, my);
            let chance = if toward.is_some() { 1.0 } else { 0.0 };
            TurnAt { rule, point, toward, chance, timer_turn: None, toward_link: goriya_facing(lx, ly, mx, my) }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ram() -> Vec<u8> {
        vec![0u8; 0x800]
    }

    /// A Red Darknut in slot 1 at (x, y) walking `dir` at q-speed $20 with grid offset `offset`.
    fn darknut(r: &mut [u8], x: u8, y: u8, dir: u8, offset: u8) {
        r[OBJ_TYPE + 1] = 0x0B;
        r[OBJ_X + 1] = x;
        r[OBJ_Y + 1] = y;
        r[OBJ_DIR + 1] = dir;
        r[OBJ_INPUT_DIR + 1] = dir;
        r[OBJ_QSPEED + 1] = 0x20;
        r[OBJ_GRID_OFFSET + 1] = offset;
        r[OBJ_TURN_TIMER + 1] = 0xFF;
    }

    #[test]
    fn flips_count_the_ten_frame_ticks() {
        let mut r = ram();
        assert_eq!(flip_frames(&r), 1);
        r[CHASE_LONG_TIMER] = 1;
        r[LONG_TIMER_CYCLE] = 0;
        assert_eq!(flip_frames(&r), 1);
        r[LONG_TIMER_CYCLE] = 9;
        assert_eq!(flip_frames(&r), 10);
        r[CHASE_LONG_TIMER] = 7;
        assert_eq!(flip_frames(&r), FLIP_MAX);
    }

    #[test]
    fn the_homing_test_is_a_signed_byte() {
        assert_eq!(homing(0x80, 0x90, 0x88, 0x40), Some(4));
        assert_eq!(homing(0x80, 0x30, 0x88, 0x40), Some(8));
        assert_eq!(homing(0x80, 0x30, 0x89, 0x40), None);
        assert_eq!(homing(0x20, 0x48, 0x80, 0x40), Some(2));
        // A target at X $FF and a monster at X $00 differ by -1 as a signed byte: "in its column".
        assert_eq!(homing(0xFF, 0x90, 0x00, 0x40), Some(4));
    }

    #[test]
    fn the_goriya_faces_along_the_longer_axis() {
        assert_eq!(goriya_facing(0x80, 0x80, 0x80, 0x80), Some(4));
        assert_eq!(goriya_facing(0x30, 0x88, 0x80, 0x80), Some(2));
        assert_eq!(goriya_facing(0x80, 0x20, 0x80, 0x80), None);
    }

    #[test]
    fn a_half_pixel_walker_reaches_its_grid_point_and_lingers() {
        let mut r = ram();
        darknut(&mut r, 0x40, 0x5D, 1, 14);
        // 2 pixels left at 0.5 a frame: the fraction carries every other frame, from $00 after 2 frames.
        let p = next_grid_point(&r, 1).unwrap();
        assert_eq!((p.frames, p.x, p.y, p.dir), (4, 0x42, 0x5D, 1));
        // The next carry is 2 frames on: one more decision frame at the point.
        assert_eq!(p.lingers, 2);
        // Walking left the offset counts down to -16 ($F0).
        darknut(&mut r, 0x40, 0x5D, 2, 0xF1);
        r[OBJ_POS_FRAC + 1] = 0x00;
        let p = next_grid_point(&r, 1).unwrap();
        assert_eq!((p.frames, p.x), (1, 0x3F));
    }

    #[test]
    fn the_turn_toward_link_behind_it_is_a_reversal() {
        let mut r = ram();
        darknut(&mut r, 0x60, 0x8D, 2, 0xF8);
        // Link 22 px behind it on its row, the target on him.
        (r[OBJ_X], r[OBJ_Y]) = (0x76, 0x8D);
        (r[CHASE_X], r[CHASE_Y]) = (0x76, 0x8D);
        let t = turn_at(&r, 1).unwrap();
        assert_eq!(t.point.x, 0x58);
        assert_eq!(t.toward, Some(1));
        assert!(t.reverse() && t.reverse_link());
        assert!(t.chance > 129.0 / 256.0 - 1e-6);
        // 20 px off its row he is outside the 8: no turn toward him.
        r[OBJ_Y] = 0x8D + 20;
        r[CHASE_Y] = 0x8D + 20;
        let t = turn_at(&r, 1).unwrap();
        assert_eq!((t.toward, t.chance, t.reverse_link()), (None, 0.0, false));
        // With the target on his mirror the turn is toward the far side, and Link's own test still reads him.
        r[OBJ_Y] = 0x8D;
        (r[CHASE_X], r[CHASE_Y]) = (0x76 ^ 0xFF, 0x8D ^ 0xFF);
        r[IS_CHASING_OTHER] = 1;
        let t = turn_at(&r, 1).unwrap();
        assert_eq!(t.toward, None);
        assert_eq!(t.toward_link, Some(1));
    }

    #[test]
    fn a_frozen_or_shoved_monster_makes_no_decision() {
        let mut r = ram();
        darknut(&mut r, 0x60, 0x8D, 2, 0xF8);
        assert!(turn_at(&r, 1).is_some());
        r[INV_CLOCK] = 1;
        assert!(turn_at(&r, 1).is_none());
        r[INV_CLOCK] = 0;
        r[OBJ_SHOVE_DIR + 1] = 1;
        assert!(turn_at(&r, 1).is_none());
        r[OBJ_SHOVE_DIR + 1] = 0;
        r[OBJ_QSPEED + 1] = 0;
        assert!(turn_at(&r, 1).is_none());
    }

    #[test]
    fn the_timer_turn_is_perpendicular_toward_the_target() {
        let mut r = ram();
        darknut(&mut r, 0x60, 0x8D, 2, 0xF8);
        (r[CHASE_X], r[CHASE_Y]) = (0x20, 0x40);
        r[OBJ_TURN_TIMER + 1] = 3;
        let t = turn_at(&r, 1).unwrap();
        assert_eq!(t.toward, None);
        assert_eq!(t.timer_turn, Some(8));
    }
}
