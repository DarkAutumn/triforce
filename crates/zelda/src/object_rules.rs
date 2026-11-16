//! What an object does to Link and what can hurt it in its current phase, per object type: the rules of the
//! observation's schema 6 entity features `touch_harms`, `weapon_target`, `visible`, `phase`, `phase_time`,
//! `touch_damage` and `in_line` (round 26; docs/architecture.md "Observation", decisions 2026-11-16). Beside
//! [`crate::enums::ObjectClass`], so `GameView`, the objective and the critic can share the answers later.
//!
//! Values are 1, 0 or 0.5: 0.5 where the game tests or draws the object on every other frame only (a fading Red
//! Wizzrobe, a teleporting Blue one, brown Ganon's last frames), not the parity of the frame a decision falls on.
//! The schema 5 rules (`hurts_on_touch`, `killable`, `active`, frozen) come in as [`Legacy`]: a row that does not
//! correct them passes them on.

use crate::enums::{ObjectClass, enemy, object};

const OBJ_TYPE: usize = 0x34F;
const OBJ_X: usize = 0x70;
const OBJ_Y: usize = 0x84;
const OBJ_DIR: usize = 0x98;
const OBJ_STATE: usize = 0xAC;
/// ObjTimer: counts down once a frame.
const OBJ_TIMER: usize = 0x28;
/// ObjRemDistance: a Blue Wizzrobe's teleport pixels left (asm:Wizzrobe_Blue_WalkOrTeleport).
const OBJ_REM_DISTANCE: usize = 0x394;
/// Gohma_ObjEyeState: 3 = wide open (asm:Gohma_HandleWeaponCollision).
const GOHMA_EYE_STATE: usize = 0x46B;
/// Gohma_ObjEyeOpenTimer.
const GOHMA_EYE_OPEN_TIMER: usize = 0x444;
/// Gohma_ObjNextOpenEyeCounter, one down every second frame; the same byte is Ganon_ObjDyingCount.
const GOHMA_NEXT_OPEN: usize = 0x42C;
const GANON_DYING_COUNT: usize = 0x42C;
/// Ganon_ScenePhase: 0, 1 Link holds the piece up; 2 the fight.
const GANON_SCENE_PHASE: usize = 0x445;
/// The Triforce of Power appears at this dying count (asm:Ganon_UpdateDying).
const GANON_ITEM_AT: u8 = 0xA0;
/// Ganon's body is drawn below this dying count, then his ashes.
const GANON_ASHES_AT: u8 = 0x50;
/// `phase_time` frames that map to 1.
const PHASE_TIME_SCALE: f32 = 512.0;
/// The Manhandla base's NES slot (asm:Manhandla_CheckCollisions).
const MANHANDLA_BASE_SLOT: usize = 5;
/// Patra in NES slot 1, its children in 2-9 (asm:Patra_Update@LoopChildren).
const PATRA_SLOT: usize = 1;
const PATRA_CHILD_SLOTS: (usize, usize) = (2, 9);
/// asm:Damage_ObjTypeDamageToLink, types $00-$5C (entry $00 is the RTS opcode before it, $60, which the synthesized
/// Gleeok heads read: asm:Gleeok_CheckSegmentCollisions). Low nibble whole hearts, high nibble in place 1/256 hearts.
const DAMAGE_TO_LINK: [u8; 0x5D] = [
    0x60, 0x02, 0x01, 0x80, 0x80, 0x01, 0x80, 0x80, 0x80, 0x80, 0x80, 0x01, 0x02, 0x80, 0x80, 0x01, // $00
    0x80, 0x80, 0x01, 0x01, 0x80, 0x80, 0x02, 0x01, 0x02, 0x00, 0x80, 0x80, 0x80, 0x80, 0x01, 0x80, // $10
    0x80, 0x01, 0x01, 0x02, 0x01, 0x02, 0x02, 0x80, 0x80, 0x80, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, // $20
    0x02, 0x01, 0x01, 0x02, 0x02, 0x00, 0x00, 0x00, 0x02, 0x02, 0x02, 0x02, 0x01, 0x01, 0x04, 0x80, // $30
    0x80, 0x80, 0x01, 0x01, 0x01, 0x01, 0x01, 0x02, 0x02, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, // $40
    0x00, 0x00, 0x00, 0x80, 0x80, 0x80, 0x01, 0x02, 0x02, 0x04, 0x04, 0x80, 0x01, // $50
];
/// Objects whose position the game keeps while it is not drawn and that the owner's question names (a Red Wizzrobe
/// between appearances, Ganon while invisible): the types `obs.hidden_positions` acts on.
pub const HIDDEN_POSITION_TYPES: [u8; 2] = [enemy::RED_WIZZROBE, enemy::GANON];

/// The schema 5 rules of an object (env::observation, frozen): `hurts_on_touch`, and `killable` ∧ `active` ∧
/// ¬dying ∧ ¬spawning (`target`), `active` ∧ ¬dying (`active`), ¬dying ∧ ¬spawning (`settled`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Legacy {
    pub touch: bool,
    pub target: bool,
    pub active: bool,
    pub settled: bool,
}

/// The phase rules of one object (each in [0, 1]).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Rules {
    /// It harms Link when they overlap this frame (Link's own invincibility, halt and Like-Like hold aside).
    pub touch_harms: f32,
    /// Some weapon can damage it in its current phase (which one: its `immune_*` bits).
    pub weapon_target: f32,
    /// The game draws it.
    pub visible: f32,
    /// The type's phase, normalised; 0 for a type without one.
    pub phase: f32,
    /// Frames to the type's next event / 512, clamped.
    pub phase_time: f32,
}

fn b(v: bool) -> f32 {
    v as u8 as f32
}

fn time(frames: f32) -> f32 {
    (frames / PHASE_TIME_SCALE).clamp(0.0, 1.0)
}

impl Rules {
    fn new(touch_harms: f32, weapon_target: f32, visible: f32) -> Rules {
        Rules { touch_harms, weapon_target, visible, phase: 0.0, phase_time: 0.0 }
    }

    fn phased(mut self, phase: f32, phase_time: f32) -> Rules {
        (self.phase, self.phase_time) = (phase, time(phase_time));
        self
    }

    /// The legacy answers: a type this module has no rule for.
    fn legacy(legacy: Legacy) -> Rules {
        Rules::new(b(legacy.touch), b(legacy.target), 1.0)
    }

    /// One of Gleeok's attached heads (no object slot of its own; `game::gleeok_heads`).
    pub fn gleeok_head(legacy_touch: bool) -> Rules {
        Rules::new(b(legacy_touch), 1.0, 1.0)
    }

    /// The room item: drawn or not.
    pub fn treasure(visible: bool) -> Rules {
        Rules::new(0.0, 0.0, b(visible))
    }
}

/// The rules of the object in NES slot `slot` (1-$B) of `ram` (CPU RAM from $0000), `clock` the held clock
/// (InvClock), `legacy` its schema 5 answers. Rows are checked top to bottom (docs/architecture.md
/// "Observation", the schema 6 table).
pub fn object_rules(ram: &[u8], slot: usize, clock: bool, legacy: Legacy) -> Rules {
    let t = ram[OBJ_TYPE + slot];
    let s = ram[OBJ_STATE + slot];
    let timer = ram[OBJ_TIMER + slot];
    match t {
        // asm:Wizzrobe_Red_Update: with the clock it is drawn and hit-tested every frame, and Link is not touched
        // under the clock (asm:Link_CheckMonsterTouch).
        enemy::RED_WIZZROBE if clock => Rules::new(0.0, 1.0, 1.0).phased(f32::from(s) / 255.0, 0.0),
        enemy::RED_WIZZROBE => {
            // $80-$BF solid; $40-$7F fading out and $C0-$FE fading in, tested and drawn every other frame
            // (asm:Wizzrobe_DrawAndCheckCollisionsIntermittently); $00-$3F hidden and $FF choosing: never.
            let v = match s {
                0x80..=0xBF => 1.0,
                0x40..=0x7F | 0xC0..=0xFE => 0.5,
                _ => 0.0,
            };
            // Frames to its shot at $B0 (asm:Wizzrobe_Red_State2Solid): the state counts down a frame at a time,
            // $80 jumps to $4F, $00 to $FF, $FF to $B0 takes 79. A rejected spot repeats $00 a frame: a lower bound.
            let to_shot = if s >= 0xB0 {
                f32::from(s - 0xB0)
            } else {
                (if s >= 0x80 { f32::from(s) - 47.0 } else { f32::from(s) + 1.0 }) + 79.0
            };
            Rules::new(v, v, v).phased(f32::from(s) / 255.0, to_shot)
        }
        enemy::BLUE_WIZZROBE => {
            // Teleporting (ObjRemDistance ≠ 0): drawn and tested on even distances only
            // (asm:Wizzrobe_DrawAndCheckCollisionsIntermittently); its flash does not stop its touch
            // (asm:Wizzrobe_DrawAndCheckCollisions); the clock freezes it, drawn and hittable.
            let rem = ram[OBJ_REM_DISTANCE + slot];
            let teleporting = rem != 0;
            let v = if teleporting && !clock { 0.5 } else { 1.0 };
            let (phase, left) = if teleporting {
                (1.0, rem)
            } else if timer < 0x10 {
                (0.5, timer)
            } else {
                (0.0, timer)
            };
            Rules::new(if clock { 0.0 } else { v }, v, v).phased(phase, f32::from(left))
        }
        enemy::BLUE_GOHMA | enemy::RED_GOHMA => {
            // Only an arrow flying up into the open eye (state 3) hurts it (asm:Gohma_HandleWeaponCollision).
            let eye = ram[GOHMA_EYE_STATE + slot];
            let open = ram[GOHMA_EYE_OPEN_TIMER + slot];
            let left = if open != 0 { f32::from(open) } else { 2.0 * f32::from(ram[GOHMA_NEXT_OPEN + slot]) };
            Rules::new(b(legacy.touch), b(eye == 3), 1.0).phased(f32::from(eye.min(3)) / 3.0, left)
        }
        enemy::GANON => ganon(ram, slot, s, timer),
        enemy::PATRA | enemy::PATRA2 => {
            // While a child lives Patra is tested against Link only (asm:Patra_Update@LoopChildren).
            let children = (PATRA_CHILD_SLOTS.0..=PATRA_CHILD_SLOTS.1)
                .any(|c| matches!(ram[OBJ_TYPE + c], enemy::PATRA_CHILD1 | enemy::PATRA_CHILD2));
            Rules::new(b(legacy.touch), b(legacy.target && !children), 1.0).phased(f32::from(s.min(1)), 0.0)
        }
        enemy::PATRA_CHILD1 | enemy::PATRA_CHILD2 => {
            // A child is drawn from its own state 1 on, but tests no collision before Patra's state is 1 (the last
            // child out; asm:Patra_UpdateChild_State1Orbit).
            let ready = ram[OBJ_STATE + PATRA_SLOT] != 0;
            Rules::new(b(ready && legacy.touch), b(ready && legacy.target), b(s != 0)).phased(f32::from(s.min(1)), 0.0)
        }
        // A segment not yet in motion is neither drawn nor tested (asm:Lanmola_Update).
        enemy::RED_LAMNOLA | enemy::BLUE_LAMNOLA if ram[OBJ_DIR + slot] == 0 => Rules::default(),
        enemy::WALLMASTER => {
            let out = s == 1;
            Rules::new(b(legacy.touch), b(out), b(out)).phased(f32::from(s.min(1)), 0.0)
        }
        // The base takes hits but never dies (asm:Manhandla_CheckCollisions).
        enemy::MANHANDLA if slot == MANHANDLA_BASE_SLOT => Rules::new(b(legacy.touch), 0.0, 1.0),
        // Gleeok's neck base (slot 1), as `killable`.
        enemy::GLEEOK..=enemy::GLEEOK4 if slot == 1 => Rules::new(b(legacy.touch), 0.0, 1.0),
        // The big Digdogger, as `killable`; a flying Ghini is immune to everything (asm:Ghini_UpdateFlying).
        enemy::DIGDOGGER2 | enemy::DIGDOGGER3 | enemy::FLYING_GHINI => Rules::new(b(legacy.touch), 0.0, 1.0),
        // A Goriya stands from its throw until it catches its boomerang (ObjState bit 7, asm:Goriya_Update).
        enemy::BLUE_GORIYA | enemy::RED_GORIYA => Rules::legacy(legacy).phased(b(s & 0x80 != 0), 0.0),
        // Underground or under water unless surfaced.
        enemy::BLUE_LEEVER | enemy::RED_LEEVER | enemy::ZORA => {
            Rules::new(b(legacy.touch), b(legacy.target), b(legacy.active))
        }
        // A shot harms in states $10-$1F only (asm:Link_CheckMonsterTouch_NoReset), never under the clock.
        0x53..=0x5C => Rules::new(b(s & 0xF0 == 0x10 && !clock), 0.0, 1.0),
        // The guard fire dies to any damaging weapon (HP $10, asm:GuardFire_Update) though `killable` says 0.
        object::GUARD_FIRE => Rules::new(b(legacy.touch), b(legacy.active && legacy.settled), 1.0),
        _ => match ObjectClass::of(t) {
            ObjectClass::Monster => Rules::legacy(legacy),
            // The boulder spawner is an invisible generator (asm:Boulder_UpdateSpawner).
            ObjectClass::Hazard => Rules::new(b(legacy.touch), 0.0, b(t != object::BOULDER_SET)),
            _ => Rules::new(0.0, 0.0, 1.0),
        },
    }
}

/// Link at (`x`, `y`) stands where a Wallmaster waiting in the wall comes out (asm:Wallmaster_Update@CheckLinkAtWall):
/// against the W or E wall (X $20 or $D0) with Y in $6D-$B4, or against the N or S wall (Y $5D or $BD) with X in
/// $29-$C7; the corners let none out. (The shared timer and Link not halted are the game's other conditions.)
pub fn wallmaster_lets_one_out(x: u8, y: u8) -> bool {
    let corner = !(0x29..0xC8).contains(&x) && !(0x6D..0xB5).contains(&y);
    !corner && (matches!(x, 0x20 | 0xD0) || matches!(y, 0x5D | 0xBD))
}

/// Ganon (asm:Ganon_ScenePhase2_Fight and its handlers).
fn ganon(ram: &[u8], slot: usize, s: u8, timer: u8) -> Rules {
    let dying = ram[GANON_DYING_COUNT + slot];
    if ram[GANON_SCENE_PHASE] < 2 {
        // Link holds the Triforce piece up: no collision is tested.
        Rules::new(0.0, 0.0, 1.0)
    } else if dying != 0 {
        // Nothing but the death runs: the body below $50, then ashes; the item at $A0.
        Rules::new(0.0, 0.0, b(dying < GANON_ASHES_AT))
            .phased(1.0, f32::from(GANON_ITEM_AT.saturating_sub(dying)))
    } else if s != 0 {
        // Brown: the silver arrow only (asm:Ganon_CheckCollisions), drawn every frame from $30, then every other;
        // the state drops once every second frame.
        let v = if s >= 0x30 { 1.0 } else { 0.5 };
        Rules::new(1.0, 1.0, v).phased(0.75, 2.0 * f32::from(s))
    } else if timer != 0 {
        // Blue, shown after a sword hit: the sword passes through him now; he jumps away at timer 1.
        Rules::new(1.0, 0.0, 1.0).phased(0.5, f32::from(timer))
    } else {
        // Blue and invisible: tested every fight frame, the sword hits.
        Rules::new(1.0, 1.0, 0.0).phased(0.25, 0.0)
    }
}

/// Hearts a touch of `obj_type` costs Link with ring grade `ring` (InvRing: 0 none, 1 blue, 2 red): the damage byte
/// halved once per grade as a 16-bit amount (asm:Link_HarmByObjType, asm:Link_BeHarmed); 0 for types $5D and up.
pub fn touch_damage(obj_type: u8, ring: u8) -> f32 {
    let Some(&byte) = DAMAGE_TO_LINK.get(usize::from(obj_type)) else {
        return 0.0;
    };
    let amount = (u16::from(byte & 0x0F) << 8 | u16::from(byte & 0xF0)) >> ring.min(2);
    f32::from(amount) / 256.0
}

/// A shooter or a flying straight shot's direction of fire, when it has one: shooters (Lynels, Moblins, Goriyas,
/// Octoroks, Wizzrobes) along their facing, straight shots ($53, $57-$59, $5B, $5C) flying (states $10-$1F) along
/// theirs; a diagonal facing has none.
fn line_of_fire(ram: &[u8], slot: usize) -> Option<u8> {
    let t = ram[OBJ_TYPE + slot];
    let shooter = matches!(t, 0x01..=0x0A | enemy::BLUE_WIZZROBE | enemy::RED_WIZZROBE);
    let shot = matches!(t, 0x53 | 0x57 | 0x58 | 0x59 | 0x5B | 0x5C) && ram[OBJ_STATE + slot] & 0xF0 == 0x10;
    let d = ram[OBJ_DIR + slot];
    ((shooter || shot) && matches!(d, 1 | 2 | 4 | 8)).then_some(d)
}

/// Link (at his ObjX/ObjY) stands in the line of fire of the object in `slot`: his 16×16 box overlaps the 16-pixel
/// lane from the object along its direction (perpendicular offset under 16 pixels, Link ahead of it).
pub fn in_line(ram: &[u8], slot: usize) -> bool {
    let Some(d) = line_of_fire(ram, slot) else {
        return false;
    };
    let dx = i32::from(ram[OBJ_X]) - i32::from(ram[OBJ_X + slot]);
    let dy = i32::from(ram[OBJ_Y]) - i32::from(ram[OBJ_Y + slot]);
    match d {
        8 => dx.abs() < 16 && dy < 0,
        4 => dx.abs() < 16 && dy > 0,
        2 => dy.abs() < 16 && dx < 0,
        _ => dy.abs() < 16 && dx > 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RAM: usize = 0x800;

    fn ram(slot: usize, t: u8, s: u8) -> Vec<u8> {
        let mut r = vec![0u8; RAM];
        r[OBJ_TYPE + slot] = t;
        r[OBJ_STATE + slot] = s;
        r
    }

    fn rules(r: &[u8], slot: usize, clock: bool, legacy: Legacy) -> [f32; 5] {
        let x = object_rules(r, slot, clock, legacy);
        [x.touch_harms, x.weapon_target, x.visible, x.phase, x.phase_time]
    }

    const ALIVE: Legacy = Legacy { touch: true, target: true, active: true, settled: true };

    #[test]
    fn red_wizzrobe_by_state_and_clock() {
        for (s, v, to_shot) in [
            (0xFF, 0.0, 79.0),
            (0xC0, 0.5, 16.0),
            (0xBF, 1.0, 15.0),
            (0xB0, 1.0, 0.0),
            (0xAF, 1.0, 0xAF as f32 - 47.0 + 79.0),
            (0x80, 1.0, 81.0 + 79.0),
            (0x4F, 0.5, 80.0 + 79.0),
            (0x40, 0.5, 65.0 + 79.0),
            (0x3F, 0.0, 64.0 + 79.0),
            (0x00, 0.0, 1.0 + 79.0),
        ] {
            let r = ram(3, enemy::RED_WIZZROBE, s);
            assert_eq!(rules(&r, 3, false, ALIVE), [v, v, v, s as f32 / 255.0, time(to_shot)], "${s:02X}");
            assert_eq!(rules(&r, 3, true, ALIVE), [0.0, 1.0, 1.0, s as f32 / 255.0, 0.0], "${s:02X} clock");
        }
    }

    #[test]
    fn blue_wizzrobe_walks_stands_teleports() {
        let mut r = ram(2, enemy::BLUE_WIZZROBE, 0);
        r[OBJ_TIMER + 2] = 0x70;
        assert_eq!(rules(&r, 2, false, ALIVE), [1.0, 1.0, 1.0, 0.0, time(112.0)]);
        r[OBJ_TIMER + 2] = 0x0F;
        assert_eq!(rules(&r, 2, false, ALIVE), [1.0, 1.0, 1.0, 0.5, time(15.0)]);
        r[OBJ_TIMER + 2] = 0;
        r[OBJ_REM_DISTANCE + 2] = 32;
        assert_eq!(rules(&r, 2, false, ALIVE), [0.5, 0.5, 0.5, 1.0, time(32.0)]);
        assert_eq!(rules(&r, 2, true, ALIVE), [0.0, 1.0, 1.0, 1.0, time(32.0)]);
        // A flash (no legacy touch) does not stop its touch.
        assert_eq!(rules(&r, 2, false, Legacy { touch: false, ..ALIVE })[0], 0.5);
    }

    #[test]
    fn gohma_eye() {
        let mut r = ram(1, enemy::RED_GOHMA, 0);
        r[GOHMA_EYE_STATE + 1] = 3;
        r[GOHMA_EYE_OPEN_TIMER + 1] = 110;
        assert_eq!(rules(&r, 1, false, ALIVE), [1.0, 1.0, 1.0, 1.0, time(110.0)]);
        r[GOHMA_EYE_STATE + 1] = 2;
        assert_eq!(rules(&r, 1, false, ALIVE)[1], 0.0);
        r[GOHMA_EYE_STATE + 1] = 0;
        r[GOHMA_EYE_OPEN_TIMER + 1] = 0;
        r[GOHMA_NEXT_OPEN + 1] = 250;
        assert_eq!(rules(&r, 1, false, ALIVE), [1.0, 0.0, 1.0, 0.0, time(500.0)]);
    }

    #[test]
    fn ganon_by_phase() {
        let mut r = ram(1, enemy::GANON, 0);
        assert_eq!(rules(&r, 1, false, ALIVE), [0.0, 0.0, 1.0, 0.0, 0.0], "the entry scene");
        r[GANON_SCENE_PHASE] = 2;
        assert_eq!(rules(&r, 1, false, ALIVE), [1.0, 1.0, 0.0, 0.25, 0.0], "blue, invisible");
        r[OBJ_TIMER + 1] = 0x40;
        assert_eq!(rules(&r, 1, false, ALIVE), [1.0, 0.0, 1.0, 0.5, time(64.0)], "shown after a hit");
        r[OBJ_STATE + 1] = 0xFF;
        assert_eq!(rules(&r, 1, false, ALIVE), [1.0, 1.0, 1.0, 0.75, time(510.0)], "brown");
        r[OBJ_STATE + 1] = 0x2F;
        assert_eq!(rules(&r, 1, false, ALIVE)[2], 0.5, "brown, flickering");
        r[GANON_DYING_COUNT + 1] = 0x10;
        assert_eq!(rules(&r, 1, false, ALIVE), [0.0, 0.0, 1.0, 1.0, time(144.0)], "dying");
        r[GANON_DYING_COUNT + 1] = 0x50;
        assert_eq!(rules(&r, 1, false, ALIVE), [0.0, 0.0, 0.0, 1.0, time(80.0)], "ashes");
    }

    #[test]
    fn patra_and_children() {
        let mut r = ram(1, enemy::PATRA, 0);
        r[OBJ_TYPE + 4] = enemy::PATRA_CHILD1;
        assert_eq!(rules(&r, 1, false, ALIVE), [1.0, 0.0, 1.0, 0.0, 0.0]);
        assert_eq!(rules(&r, 4, false, ALIVE), [0.0, 0.0, 0.0, 0.0, 0.0], "waiting");
        r[OBJ_STATE + 4] = 1;
        assert_eq!(rules(&r, 4, false, ALIVE), [0.0, 0.0, 1.0, 1.0, 0.0], "out, the last not yet");
        r[OBJ_STATE + 1] = 1;
        assert_eq!(rules(&r, 4, false, ALIVE), [1.0, 1.0, 1.0, 1.0, 0.0], "all out");
        r[OBJ_TYPE + 4] = 0x5D;
        assert_eq!(rules(&r, 1, false, ALIVE), [1.0, 1.0, 1.0, 1.0, 0.0], "no child left");
    }

    #[test]
    fn the_other_rows() {
        let mut r = ram(4, enemy::RED_LAMNOLA, 0);
        assert_eq!(rules(&r, 4, false, ALIVE), [0.0; 5]);
        r[OBJ_DIR + 4] = 8;
        assert_eq!(rules(&r, 4, false, ALIVE), [1.0, 1.0, 1.0, 0.0, 0.0]);
        let r = ram(4, enemy::WALLMASTER, 0);
        assert_eq!(rules(&r, 4, false, Legacy::default()), [0.0, 0.0, 0.0, 0.0, 0.0]);
        let r = ram(4, enemy::WALLMASTER, 1);
        assert_eq!(rules(&r, 4, false, ALIVE), [1.0, 1.0, 1.0, 1.0, 0.0]);
        assert_eq!(rules(&ram(5, enemy::MANHANDLA, 0), 5, false, ALIVE), [1.0, 0.0, 1.0, 0.0, 0.0]);
        assert_eq!(rules(&ram(2, enemy::MANHANDLA, 0), 2, false, ALIVE), [1.0, 1.0, 1.0, 0.0, 0.0]);
        assert_eq!(rules(&ram(1, enemy::GLEEOK2, 0), 1, false, ALIVE), [1.0, 0.0, 1.0, 0.0, 0.0]);
        assert_eq!(rules(&ram(1, enemy::DIGDOGGER2, 0), 1, false, ALIVE), [1.0, 0.0, 1.0, 0.0, 0.0]);
        assert_eq!(rules(&ram(1, enemy::FLYING_GHINI, 0), 1, false, ALIVE), [1.0, 0.0, 1.0, 0.0, 0.0]);
        assert_eq!(rules(&ram(1, enemy::BLUE_GORIYA, 0x80), 1, false, ALIVE), [1.0, 1.0, 1.0, 1.0, 0.0]);
        assert_eq!(rules(&ram(1, enemy::BLUE_GORIYA, 0x00), 1, false, ALIVE), [1.0, 1.0, 1.0, 0.0, 0.0]);
        let under = Legacy { touch: false, target: false, active: false, settled: true };
        assert_eq!(rules(&ram(1, enemy::ZORA, 0), 1, false, under), [0.0; 5]);
        // A boomerang out (states $10-$1F) harms, coming home ($20 and up) not; never under the clock.
        assert_eq!(rules(&ram(6, 0x5C, 0x12), 6, false, ALIVE), [1.0, 0.0, 1.0, 0.0, 0.0]);
        assert_eq!(rules(&ram(6, 0x5C, 0x28), 6, false, ALIVE), [0.0, 0.0, 1.0, 0.0, 0.0]);
        assert_eq!(rules(&ram(6, 0x5C, 0x12), 6, true, ALIVE), [0.0, 0.0, 1.0, 0.0, 0.0]);
        assert_eq!(rules(&ram(3, enemy::BUBBLE, 0), 3, false, ALIVE), [1.0, 0.0, 1.0, 0.0, 0.0]);
        assert_eq!(rules(&ram(3, object::BOULDER_SET, 0), 3, false, Legacy::default()), [0.0; 5]);
        let fire = Legacy { touch: true, target: false, active: true, settled: true };
        assert_eq!(rules(&ram(3, object::GUARD_FIRE, 0), 3, false, fire), [1.0, 1.0, 1.0, 0.0, 0.0]);
        assert_eq!(rules(&ram(3, enemy::STALFOS, 0), 3, false, ALIVE), [1.0, 1.0, 1.0, 0.0, 0.0]);
        assert_eq!(rules(&ram(3, object::DROPPED_ITEM, 0), 3, false, Legacy::default()), [0.0, 0.0, 1.0, 0.0, 0.0]);
        assert_eq!(rules(&ram(3, 0x4B, 0), 3, false, Legacy::default()), [0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn damage_by_type_and_ring() {
        // Blue Goriya 1 heart, Red Goriya 1/2, small Digdogger 2, Ganon 4, Bubble 0, the Red's magic shot 4.
        for (t, hearts) in [(0x05, 1.0), (0x06, 0.5), (0x18, 2.0), (0x3E, 4.0), (0x2B, 0.0), (0x59, 4.0)] {
            assert_eq!(touch_damage(t, 0), hearts, "${t:02X}");
            assert_eq!(touch_damage(t, 1), hearts / 2.0, "${t:02X} blue ring");
            assert_eq!(touch_damage(t, 2), hearts / 4.0, "${t:02X} red ring");
        }
        assert_eq!(touch_damage(0, 0), 0.375, "the synthesized Gleeok heads");
        assert_eq!(touch_damage(0x5D, 0), 0.0);
        assert_eq!(touch_damage(0x60, 0), 0.0);
    }

    #[test]
    fn wallmaster_lure_spots() {
        // The walk's spot: Link still at ($20,$8D) against the W wall, hands came out.
        assert!(wallmaster_lets_one_out(0x20, 0x8D));
        assert!(wallmaster_lets_one_out(0xD0, 0x6D) && wallmaster_lets_one_out(0xD0, 0xB4));
        assert!(!wallmaster_lets_one_out(0x20, 0x6C) && !wallmaster_lets_one_out(0x20, 0xB5), "corners");
        assert!(wallmaster_lets_one_out(0x29, 0x5D) && wallmaster_lets_one_out(0xC7, 0xBD));
        assert!(!wallmaster_lets_one_out(0x28, 0x5D) && !wallmaster_lets_one_out(0xC8, 0xBD), "corners");
        assert!(!wallmaster_lets_one_out(0x21, 0x8D) && !wallmaster_lets_one_out(0x78, 0x8D), "off the wall");
    }

    #[test]
    fn lanes_in_four_directions() {
        let mut r = ram(3, enemy::BLUE_GORIYA, 0);
        r[OBJ_X + 3] = 100;
        r[OBJ_Y + 3] = 100;
        for (d, (x, y), yes) in [
            (8, (110, 40), true),
            (8, (116, 40), false),
            (8, (100, 120), false),
            (4, (90, 160), true),
            (2, (20, 85), true),
            (2, (20, 84), false),
            (1, (200, 115), true),
            (1, (60, 100), false),
            (9, (100, 40), false),
        ] {
            r[OBJ_DIR + 3] = d;
            (r[OBJ_X], r[OBJ_Y]) = (x, y);
            assert_eq!(in_line(&r, 3), yes, "dir {d} Link ({x},{y})");
        }
        // A boomerang flying (state $10-$1F) along its direction; coming home it is no shot line.
        let mut r = ram(6, 0x5C, 0x10);
        (r[OBJ_X + 6], r[OBJ_Y + 6], r[OBJ_DIR + 6], r[OBJ_X], r[OBJ_Y]) = (50, 100, 1, 150, 100);
        assert!(in_line(&r, 6));
        r[OBJ_STATE + 6] = 0x30;
        assert!(!in_line(&r, 6));
        // A Stalfos shoots nothing.
        let mut r = ram(3, enemy::STALFOS, 0);
        (r[OBJ_DIR + 3], r[OBJ_Y]) = (4, 200);
        assert!(!in_line(&r, 3));
    }
}
