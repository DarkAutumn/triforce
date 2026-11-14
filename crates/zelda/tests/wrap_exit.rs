//! The wrap exit in the real game (round 22; `zelda::clips::SkipKind::WrapExit`, `zelda-asm/mechanics.md` "After the
//! wrap, edge by edge"): `CLIP E` in room 0x75 wraps onto its solid W edge and Left there leaves into 0x74.
//!
//! No state starts in 0x75: every test walks in from `0_38w_0` (Link on its W bound, row Y `$8D`) with `RoomId` set to
//! the room E of the one wanted (SETUP, as `zelda-map.py` reaches rooms without states), and removes monsters by RAM
//! writes where it says so.
#![cfg(feature = "zcore")]

mod common;

use common::{Counted, load_named_state, open_core};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::BTreeSet;
use std::sync::Arc;
use zelda::clips::{ClipAbort, SkipKind, skip_available, skip_target};
use zelda::executor::buttons::{DOWN, LEFT, RIGHT, UP};
use zelda::{
    Action, ActionKind, Change, Direction, GameMap, LoadOptions, MapLocation, Nes, Session, SessionSnapshot, Var,
};

type S = Session<Counted>;
type Snap = SessionSnapshot<zcore::MachineState>;

/// The rows (Link's Y) of 0x75 Link can stand on at its E side: the wrap exit is offered on the first ten, the last
/// four are the softlock rows.
const SAFE: [i32; 10] = [0x65, 0x6D, 0x75, 0x7D, 0x85, 0x8D, 0x95, 0x9D, 0xA5, 0xAD];
const SOFTLOCK: [i32; 4] = [0x55, 0x5D, 0xB5, 0xBD];
const PEAHAT: u8 = 0x1A;
const OBJ_TYPE: u16 = 0x34F;
const ROOM_ID: u16 = 0xEB;

fn session() -> Option<S> {
    let core = open_core()?;
    let mut s = Session::new(Counted(core, 0), Arc::new(GameMap::builtin()));
    s.set_clips(true);
    Some(s)
}

fn ow(room: u8) -> MapLocation {
    MapLocation::new(0, room, false)
}

fn act(s: &mut S, text: &str) -> Change {
    s.act(Action::parse(text).unwrap()).unwrap()
}

/// (x, y, offset) of Link.
fn link(s: &S) -> (i32, i32, i32) {
    let st = s.state();
    (st.link.position.x, st.link.position.y, st.var(Var::LinkGridOffset))
}

fn describe(s: &S) -> String {
    let st = s.state();
    format!(
        "{:?} mode {} at ({:#04x}, {:#04x}) offset {} fraction {:#04x} facing {}",
        st.full_location(),
        st.mode(),
        st.link.position.x,
        st.link.position.y,
        st.var(Var::LinkGridOffset),
        st.var(Var::LinkPosFrac),
        st.var(Var::LinkDirection)
    )
}

/// Removes every monster but those of `keep` (SETUP: `ObjType` of slots 1-11 zeroed).
fn clear_monsters(s: &mut S, keep: Option<u8>) {
    for slot in 1..=11u16 {
        let kind = s.state().ram()[(OBJ_TYPE + slot) as usize];
        if Some(kind) != keep {
            s.nes_mut().poke(OBJ_TYPE + slot, 0);
        }
    }
    s.apply_ram(&[]);
}

/// Walks W from `0_38w_0` with `RoomId` = `east` (SETUP) into the room W of it, at (`$F0`, `$8D`) in play.
fn walk_into_west_of(s: &mut S, east: u8) {
    s.load(&load_named_state("0_38w_0"), &LoadOptions::default(), &mut ChaCha8Rng::seed_from_u64(0)).unwrap();
    assert_eq!((link(s).0, link(s).1), (0x00, 0x8D), "{}", describe(s));
    s.nes_mut().poke(ROOM_ID, east);
    s.apply_ram(&[]);
    let c = act(s, "MOVE W 8");
    assert!(c.changed_location(), "{}", describe(s));
    assert_eq!((s.state().full_location(), s.state().mode(), link(s).0), (ow(east - 1), 5, 0xF0), "{}", describe(s));
}

/// Holds `buttons` a frame at a time until `until` holds (at most 300 frames).
fn press_until(s: &mut S, buttons: u8, until: impl Fn(&S) -> bool) {
    for _ in 0..300 {
        if until(s) {
            return;
        }
        s.press(buttons, 1);
    }
    panic!("never got there: {}", describe(s));
}

/// 0x75 entered from 0x76, monsters removed but `keep`; Link walked to rest at (`$D0`, `y`).
fn at_row(s: &mut S, y: i32, keep: Option<u8>) -> Snap {
    walk_into_west_of(s, 0x76);
    clear_monsters(s, keep);
    press_until(s, LEFT, |s| link(s).0 == 0xD0 && link(s).2 == 0);
    let vertical = if y < 0x8D { UP } else { DOWN };
    press_until(s, vertical, |s| link(s).1 == y && link(s).2 == 0);
    s.snapshot()
}

/// Every start of the wrap zone on the row of `rest` reached by real walking: Right for each frame count that ends in
/// the zone (both cadences, offset of either sign or 0), each followed by 0-7 frames of Left.
fn zone_starts(s: &mut S, rest: &Snap) -> Vec<(String, Snap)> {
    let mut out = Vec::new();
    for frames in 1..40 {
        s.restore(rest);
        s.press(RIGHT, frames);
        let (x, ..) = link(s);
        if s.state().full_location() != ow(0x75) || s.state().mode() != 5 || !(0xD0..=0xF0).contains(&x) {
            continue;
        }
        let here = s.snapshot();
        for back in 0..8 {
            s.restore(&here);
            s.press(LEFT, back);
            if (0xD0..=0xF0).contains(&link(s).0) {
                out.push((format!("{frames} right, {back} left: {}", describe(s)), s.snapshot()));
            }
        }
    }
    out
}

/// CLIP E from every start of the zone on each safe row ends in 0x74 at (`$F0`, row), playing, the skip counted done
/// (the room change completes it); MOVE W then walks on. Both cadences: the Left on the landing frame meets the
/// 5-frame window from every one.
#[test]
fn the_wrap_exit_leaves_into_0x74_from_every_start_on_every_safe_row() {
    let Some(mut s) = session() else { return };
    let mut cadences = BTreeSet::new();
    for y in SAFE {
        let rest = at_row(&mut s, y, None);
        let starts = zone_starts(&mut s, &rest);
        assert!(starts.len() > 30, "Y {y:#04x}: {} starts", starts.len());
        for (label, snap) in &starts {
            s.restore(snap);
            cadences.insert(s.state().var(Var::LinkPosFrac));
            let target = skip_target(s.state(), Direction::E).unwrap_or_else(|| panic!("{label}: not offered"));
            assert_eq!(target.kind, SkipKind::WrapExit, "{label}");
            assert!(s.mask().is_allowed(&Action::parse("CLIP E").unwrap()), "{label}");
            let c = act(&mut s, "CLIP E");
            let clip = c.clip.expect("clip outcome");
            assert!(clip.completed(), "{label}: {clip:?}, now {}", describe(&s));
            assert_eq!(clip.presses, 1, "{label}");
            assert_eq!(c.interrupt, Some(zelda::Interrupt::Transition), "{label}");
            assert_eq!((s.state().full_location(), s.state().mode()), (ow(0x74), 5), "{label}: {}", describe(&s));
            assert_eq!((link(&s).0, link(&s).1), (0xF0, y), "{label}: {}", describe(&s));
            let mut stats = zelda::clips::ClipStats::default();
            stats.update(&c);
            assert_eq!((stats.done, stats.done_e, stats.aborted), (1, 1, 0), "{label}");
            act(&mut s, "MOVE W 8");
            assert!(s.state().full_location() == ow(0x74) && link(&s).0 < 0xF0, "{label}: then MOVE W: {}", describe(&s));
        }
    }
    assert_eq!(cadences.len(), 2, "both fractions: {cadences:?}");
}

/// The softlock rows are never offered; by buttons (the wrap, then Left on the landing frame) they leave Link in 0x74 at
/// X `$F0` where only Right moves him, back to 0x75 at X `$00`, where only Left does. The other rows of 0x75's E side
/// (Y `$3D`-`$4D`, `$C5`-`$DD`) are trees. Clips off: nothing is offered on any row.
#[test]
fn the_softlock_rows_are_never_offered() {
    let Some(mut s) = session() else { return };
    for y in SOFTLOCK {
        let rest = at_row(&mut s, y, None);
        for (label, snap) in zone_starts(&mut s, &rest) {
            s.restore(&snap);
            assert!(!skip_available(s.state(), Direction::E), "{label}");
            assert!(!s.mask().kind[ActionKind::Clip.index()], "{label}");
        }
        // The game, by buttons.
        s.restore(&rest);
        press_until(&mut s, RIGHT, |s| {
            let st = s.state();
            (link(s).0, link(s).2, st.var(Var::LinkPosFrac)) == (0xEB, 3, 0)
        });
        s.press(UP, 1);
        press_until(&mut s, RIGHT, |s| link(s).0 == 0x00 && link(s).2 == 0);
        press_until(&mut s, LEFT, |s| s.state().full_location() == ow(0x74) && s.state().mode() == 5);
        assert_eq!((link(&s).0, link(&s).1), (0xF0, y));
        let stuck = s.snapshot();
        for (d, buttons) in [("W", LEFT), ("N", UP), ("S", DOWN)] {
            s.restore(&stuck);
            s.press(buttons, 30);
            assert_eq!((s.state().full_location(), link(&s).0, link(&s).1), (ow(0x74), 0xF0, y), "Y {y:#04x} {d}");
        }
        s.restore(&stuck);
        press_until(&mut s, RIGHT, |s| s.state().full_location() == ow(0x75) && s.state().mode() == 5);
        assert_eq!((link(&s).0, link(&s).1), (0x00, y));
        let back = s.snapshot();
        for (d, buttons) in [("E", RIGHT), ("N", UP), ("S", DOWN)] {
            s.restore(&back);
            s.press(buttons, 30);
            assert_eq!((s.state().full_location(), link(&s).0, link(&s).1), (ow(0x75), 0x00, y), "Y {y:#04x} {d}");
        }
    }
    let rest = at_row(&mut s, 0x8D, None);
    s.restore(&rest);
    press_until(&mut s, RIGHT, |s| link(s).0 == 0xE0 && link(s).2 == 0);
    assert!(s.mask().is_allowed(&Action::parse("CLIP E").unwrap()));
    s.set_clips_allowed(false);
    assert!(!s.mask().kind[ActionKind::Clip.index()], "clips off: {}", describe(&s));
}

/// The macro's inputs from rest at (`$E0`, `y`), frame by frame (the game, by buttons), up to the room change.
fn macro_inputs(s: &mut S) -> Vec<u8> {
    let mut inputs = Vec::new();
    let mut pressed = false;
    for _ in 0..60 {
        let st = s.state();
        let at = (link(s).0, link(s).2, st.var(Var::LinkPosFrac));
        let buttons = if !pressed && at == (0xEB, 3, 0) {
            pressed = true;
            UP
        } else if pressed && (at.0, at.1) == (0x00, 0) {
            LEFT
        } else {
            RIGHT
        };
        inputs.push(buttons);
        s.press(buttons, 1);
        if s.state().mode() != 5 {
            assert!(pressed && buttons == LEFT, "the wrap was missed: {}", describe(s));
            return inputs;
        }
    }
    panic!("no room change: {}", describe(s));
}

/// From wherever a hit left Link (monsters gone), actions reach 0x74: CLIP E where it is offered, else MOVE W (from
/// X `$00`-`$07` it leaves by the W edge; from X `$F1`-`$FF` it walks back into the zone). Returns the number of
/// actions and whether a MOVE finished an aborted skip.
fn recover(s: &mut S, label: &str) -> (usize, bool) {
    let mut finished = false;
    for n in 0..8 {
        if s.state().full_location() == ow(0x74) {
            return (n, finished);
        }
        let text = if skip_available(s.state(), Direction::E) { "CLIP E" } else { "MOVE W 8" };
        assert!(s.mask().is_allowed(&Action::parse(text).unwrap()), "{label}: {text} not legal: {}", describe(s));
        finished |= act(s, text).clip_finished_by_move;
    }
    panic!("{label}: never reached 0x74: {}", describe(s));
}

/// A hit at every frame of the macro, from either side (a Peahat put next to Link, SETUP): the game ends the skip there
/// and Link is never stuck: the MOVE mask matches the game, and CLIP E or MOVE W take him into 0x74. Before the bound
/// he is knocked back into the zone; past it (X `$F1`-`$FF`) he stays on the row; on the landing frame (X `$00`) the
/// left bound stops the knockback and MOVE W leaves at once. The frame of the Left press starts the room change: no
/// hit lands after it.
#[test]
fn a_hit_at_any_phase_leaves_link_able_to_go_on() {
    let Some(mut s) = session() else { return };
    let mut phases = BTreeSet::new();
    for y in [0x65, 0x8D, 0xAD] {
        let rest = at_row(&mut s, y, Some(PEAHAT));
        let slot = (1..=11u16).find(|&k| s.state().ram()[(OBJ_TYPE + k) as usize] == PEAHAT).expect("a Peahat") as u16;
        let park = |s: &mut S| {
            s.nes_mut().poke(0x70 + slot, 0x80);
            s.nes_mut().poke(0x84 + slot, 0xC0);
        };
        s.restore(&rest);
        press_until(&mut s, RIGHT, |s| link(s).0 == 0xE0 && link(s).2 == 0);
        // SETUP: the fraction of the cadence the button sequence below follows (the macro handles both).
        s.nes_mut().poke(Var::LinkPosFrac.addr(), 0);
        s.apply_ram(&[]);
        let start = s.snapshot();
        let inputs = macro_inputs(&mut s);
        for k in 0..inputs.len() {
            for (side, dx) in [("W", -8i32), ("E", 8)] {
                s.restore(&start);
                for &b in &inputs[..k] {
                    park(&mut s);
                    s.press(b, 1);
                }
                let x0 = link(&s).0;
                let label = format!("Y {y:#04x} frame {k} X {x0:#04x} from {side}");
                let hp = s.state().health();
                let mut hit = false;
                for &b in &inputs[k..(k + 2).min(inputs.len())] {
                    let (lx, ly) = (link(&s).0, link(&s).1);
                    s.nes_mut().poke(0x70 + slot, (lx + dx) as u8);
                    s.nes_mut().poke(0x84 + slot, ly as u8);
                    s.press(b, 1);
                    if s.state().var(Var::LinkShoveDir) != 0 || s.state().health() < hp {
                        hit = true;
                        break;
                    }
                    if s.state().mode() != 5 {
                        break;
                    }
                }
                if !hit {
                    continue;
                }
                // The knockback runs out; the Peahat goes (SETUP).
                s.nes_mut().poke(OBJ_TYPE + slot, 0);
                press_until(&mut s, 0, |s| s.state().var(Var::LinkShoveDir) == 0 && s.state().link.status == 0);
                let (x, ..) = link(&s);
                assert_eq!((s.state().full_location(), s.state().mode(), link(&s).1), (ow(0x75), 5, y), "{label}");
                phases.insert(match x {
                    0xC0..=0xF0 => "before the bound",
                    0xF1..=0xFF => "past the bound",
                    0x00 => "landed",
                    _ => panic!("{label}: knocked to {}", describe(&s)),
                });
                assert_move_mask_matches_the_game(&mut s, &label);
                recover(&mut s, &label);
            }
        }
    }
    assert_eq!(phases.len(), 3, "{phases:?}");
}

/// The MOVE mask says what the game does: every legal direction moves Link (or changes the room) within 8 frames of
/// holding it, every illegal one does not.
fn assert_move_mask_matches_the_game(s: &mut S, label: &str) {
    let snap = s.snapshot();
    let mask = *s.mask();
    for d in [Direction::N, Direction::S, Direction::W, Direction::E] {
        s.restore(&snap);
        let (start, room) = (s.state().link.position, s.state().full_location());
        s.press(zelda::executor::direction_buttons(d), 8);
        let moved = s.state().link.position != start || s.state().full_location() != room || s.state().mode() != 5;
        let legal = mask.is_allowed(&Action::new(ActionKind::Move, d, 8).unwrap());
        assert_eq!(legal, moved, "{label}: MOVE {} legal {legal}, the game moved {moved}: {}", d.name(), describe(s));
    }
    s.restore(&snap);
}

/// A live Peahat on the row ahead of Link (SETUP: its position) cuts the macro: the skip ends with the damage
/// interrupt, aborted; actions take Link on into 0x74, and where the skip was cut past the bound the MOVE W that leaves
/// counts as finishing it.
#[test]
fn a_hit_during_clip_e_aborts_and_link_goes_on() {
    let Some(mut s) = session() else { return };
    let rest = at_row(&mut s, 0x8D, Some(PEAHAT));
    let slot = (1..=11u16).find(|&k| s.state().ram()[(OBJ_TYPE + k) as usize] == PEAHAT).expect("a Peahat") as u16;
    let mut aborted = 0;
    for px in (0xE8..=0x108).step_by(4) {
        s.restore(&rest);
        press_until(&mut s, RIGHT, |s| link(s).0 == 0xE0 && link(s).2 == 0);
        s.nes_mut().poke(0x70 + slot, (px & 0xFF) as u8);
        s.nes_mut().poke(0x84 + slot, 0x8D);
        s.apply_ram(&[]);
        let label = format!("Peahat at X {:#04x}", px & 0xFF);
        let c = act(&mut s, "CLIP E");
        let clip = c.clip.expect("clip outcome");
        if clip.completed() {
            assert_eq!(s.state().full_location(), ow(0x74), "{label}: {}", describe(&s));
            continue;
        }
        assert_eq!(c.interrupt, Some(zelda::Interrupt::Damage), "{label}: {clip:?} {}", describe(&s));
        assert_eq!(clip.abort, Some(ClipAbort::Interrupted(zelda::Interrupt::Damage)), "{label}");
        aborted += 1;
        s.nes_mut().poke(OBJ_TYPE + slot, 0);
        press_until(&mut s, 0, |s| s.state().var(Var::LinkShoveDir) == 0 && s.state().link.status == 0);
        let landed = link(&s).0 == 0x00;
        let (n, by_move) = recover(&mut s, &label);
        // Cut on the landing frame: the one MOVE W that leaves finishes the skip.
        if landed {
            assert_eq!((n, by_move), (1, true), "{label}");
        }
    }
    assert!(aborted > 0, "no placement hit Link");
}

/// Acts `text` until `done` (at most 40 times); counts the actions.
fn act_until(s: &mut S, actions: &mut usize, text: &str, done: &dyn Fn(&S) -> bool) {
    for _ in 0..40 {
        if done(s) {
            return;
        }
        act(s, text);
        *actions += 1;
    }
    panic!("{text}: never done: {}", describe(s));
}

/// MOVE `d` 4 until Link is at most 6 pixels short of the grid line `line` (X heading W or E, Y heading N or S): the
/// next perpendicular MOVE finishes the cell onto it (between grid points Link keeps his heading).
fn approach(s: &mut S, actions: &mut usize, d: Direction, line: i32) {
    let short = |s: &S| {
        let (x, y, _) = link(s);
        match d {
            Direction::W => x - line,
            Direction::E => line - x,
            Direction::N => y - line,
            _ => line - y,
        }
    };
    act_until(s, actions, &format!("MOVE {} 4", d.name()), &|s| short(s) <= 6);
    assert!(short(s) >= 0, "past {line:#04x}: {}", describe(s));
}

/// The stretch on actions alone: 0x76 entered from E (SETUP: from `0_38w_0`), W along Y `$8D` to X `$90`, N, W into
/// 0x75 (on row Y `$75`), `CLIP E` into 0x74, W to X `$D0`, S to Y `$8D`, W to X `$80` and N into the mouth: Dungeon
/// 3's entrance room plays. Monsters removed in each room (SETUP), so the count is the walk's own.
#[test]
fn the_clip_way_from_0x76_enters_dungeon_3() {
    let Some(mut s) = session() else { return };
    walk_into_west_of(&mut s, 0x77);
    assert_eq!(s.state().full_location(), ow(0x76));
    clear_monsters(&mut s, None);
    let start = s.frame();
    let mut actions = 0;
    act_until(&mut s, &mut actions, "MOVE W 8", &|s| link(s).0 <= 0x90);
    act_until(&mut s, &mut actions, "MOVE N 8", &|s| link(s).1 <= 0x7D);
    act_until(&mut s, &mut actions, "MOVE W 8", &|s| s.state().full_location() == ow(0x75));
    let at_75 = s.frame() - start;
    clear_monsters(&mut s, None);
    assert!(link(&s).0 == 0xF0 && SAFE.contains(&link(&s).1), "{}", describe(&s));
    act_until(&mut s, &mut actions, "CLIP E", &|s| s.state().full_location() == ow(0x74));
    let at_74 = s.frame() - start;
    clear_monsters(&mut s, None);
    approach(&mut s, &mut actions, Direction::W, 0xD0);
    approach(&mut s, &mut actions, Direction::S, 0x8D);
    approach(&mut s, &mut actions, Direction::W, 0x80);
    act_until(&mut s, &mut actions, "MOVE N 8", &|s| s.state().level() == 3);
    let frames = s.frame() - start;
    assert_eq!((s.state().full_location(), s.state().mode()), (MapLocation::new(3, 0x7C, false), 5), "{}", describe(&s));
    eprintln!("0x76 (E edge) to Dungeon 3: {actions} actions, {frames} frames (0x75 at {at_75}, 0x74 at {at_74})");
    assert!(frames < 900, "{frames} frames");
}
