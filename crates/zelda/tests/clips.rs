//! The grid-point skip in the real game (round 21; `zelda::clips`, `zelda-asm/mechanics.md`).
#![cfg(feature = "zcore")]

mod common;

use common::{Counted, load_named_state, open_core};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;
use zelda::clips::{ClipSettings, ClipTrigger, skip_available};
use zelda::executor::buttons::{LEFT, RIGHT};
use zelda::{
    Action, ActionKind, Change, Direction, GameMap, LoadOptions, MapLocation, Nes, Session, SessionSnapshot, Var,
};

type S = Session<Counted>;
type Snap = SessionSnapshot<zcore::MachineState>;

const ROW: i32 = 0x8D;

fn session() -> Option<S> {
    let core = open_core()?;
    Some(Session::new(Counted(core, 0), Arc::new(GameMap::builtin())))
}

fn load(s: &mut S, name: &str) {
    s.load(&load_named_state(name), &LoadOptions::default(), &mut ChaCha8Rng::seed_from_u64(0)).unwrap();
}

fn act(s: &mut S, text: &str) -> Change {
    s.act(Action::parse(text).unwrap()).unwrap()
}

fn ow(room: u8) -> MapLocation {
    MapLocation::new(0, room, false)
}

/// (x, offset, fraction) of Link.
fn link(s: &S) -> (i32, i32, i32) {
    let st = s.state();
    (st.link.position.x, st.var(Var::LinkGridOffset), st.var(Var::LinkPosFrac))
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

/// Every start of the wrap zone on `$77`'s row Y = `$8D` at the `side` edge, reached by real walking: walking toward
/// the edge for each frame count that ends in the zone (both cadences, offset of the heading's sign; stopped
/// mid-cell), each followed by 1-7 frames back (the other sign), and standing on the bound after entering from the
/// neighbour on that side.
fn wrap_starts(s: &mut S, side: Direction) -> Vec<(String, Snap)> {
    let (toward, away, neighbour_in) = match side {
        Direction::W => (LEFT, RIGHT, "MOVE E 8"),
        _ => (RIGHT, LEFT, "MOVE W 8"),
    };
    let in_zone = |x: i32| if side == Direction::W { x <= 0x20 } else { (0xD0..=0xF0).contains(&x) };
    let mut out = Vec::new();
    load(s, "start");
    assert_eq!((s.state().full_location(), s.state().link.position.y), (ow(0x77), ROW), "{}", describe(s));
    let rest = s.snapshot();
    for frames in 40..90 {
        s.restore(&rest);
        s.press(toward, frames);
        let (x, ..) = link(s);
        if s.state().full_location() != ow(0x77) || s.state().mode() != 5 || !in_zone(x) {
            continue;
        }
        let here = s.snapshot();
        out.push((format!("{} frames toward: {}", frames, describe(s)), here.clone()));
        for back in 1..8 {
            s.restore(&here);
            s.press(away, back);
            if in_zone(link(s).0) {
                out.push((format!("{frames} toward, {back} back: {}", describe(s)), s.snapshot()));
            }
        }
    }
    // On the bound right after entering from the neighbour on that side.
    s.restore(&rest);
    let leave = format!("MOVE {} 32", side.name());
    for _ in 0..10 {
        if act(s, &leave).changed_location() {
            break;
        }
    }
    assert_ne!(s.state().full_location(), ow(0x77), "left $77 {}", side.name());
    act(s, neighbour_in);
    assert_eq!(s.state().full_location(), ow(0x77), "{}", describe(s));
    out.push((format!("entered from the {} neighbour: {}", side.name(), describe(s)), s.snapshot()));
    out
}

/// CLIP W from every start of the zone ends at X $F0 offset 0 in $77, and MOVE E then enters $78 (the mirror for
/// CLIP E: X $00, MOVE W enters $76).
#[test]
fn wrap_completes_from_every_start_of_the_zone() {
    let Some(mut s) = session() else { return };
    for (side, end, exit, next) in [(Direction::W, 0xF0, "MOVE E 8", 0x78), (Direction::E, 0x00, "MOVE W 8", 0x76)] {
        let starts = wrap_starts(&mut s, side);
        assert!(starts.len() > 40, "{} starts", starts.len());
        let (mut cadences, mut signs) = (std::collections::BTreeSet::new(), std::collections::BTreeSet::new());
        for (label, snap) in &starts {
            s.restore(snap);
            let (_, offset, fraction) = link(&s);
            cadences.insert(fraction);
            signs.insert(offset.signum());
            assert!(skip_available(s.state(), side), "{label}");
            let c = act(&mut s, &format!("CLIP {}", side.name()));
            let clip = c.clip.expect("clip outcome");
            assert!(clip.completed(), "{label}: {clip:?}, now {}", describe(&s));
            assert_eq!(clip.presses, 1, "{label}: the decision table needs no second press");
            assert_eq!(s.state().full_location(), ow(0x77), "{label}");
            assert_eq!(s.state().mode(), 5, "{label}");
            assert_eq!((link(&s).0, link(&s).1), (end, 0), "{label}: {}", describe(&s));
            assert_eq!(s.state().link.position.y, ROW, "{label}");
            act(&mut s, exit);
            assert_eq!(s.state().full_location(), ow(next), "{label}: then {exit}: {}", describe(&s));
        }
        assert_eq!(cadences.len(), 2, "both fractions");
        assert_eq!(signs.len(), 3, "both offset signs and offset 0");
    }
}

#[test]
fn the_session_offers_clip_only_with_the_kind_trigger() {
    let Some(mut s) = session() else { return };
    load(&mut s, "start");
    s.press(LEFT, 80);
    assert!(skip_available(s.state(), Direction::W), "{}", describe(&s));
    let clip_w = Action::parse("CLIP W").unwrap();
    assert!(!s.mask().is_allowed(&clip_w));
    s.set_clip_settings(ClipSettings::new(ClipTrigger::Kind, 0));
    assert!(s.mask().is_allowed(&clip_w));
    assert!(!s.mask().is_allowed(&Action::parse("CLIP E").unwrap()));
    assert_eq!(s.mask().directions(ActionKind::Clip), vec![Direction::W]);
    for trigger in [ClipTrigger::None, ClipTrigger::Long, ClipTrigger::Edge] {
        s.set_clip_settings(ClipSettings::new(trigger, 0));
        assert!(!s.mask().kind[ActionKind::Clip.index()], "{trigger:?}");
    }
}

/// Holds `buttons` a frame at a time until `until` holds (at most 200 frames).
fn press_until(s: &mut S, buttons: u8, until: impl Fn(&S) -> bool) {
    for _ in 0..200 {
        if until(s) {
            return;
        }
        s.press(buttons, 1);
    }
    panic!("never got there: {}", describe(s));
}

/// Link walked from the start toward the W edge of $77 onto G = X $08 (in the zone, 8 pixels from the bound).
fn near_w(s: &mut S) -> Snap {
    load(s, "start");
    press_until(s, LEFT, |s| link(s).0 <= 0x08 && link(s).1 == 0);
    assert!(link(s).0 == 0x08 && skip_available(s.state(), Direction::W), "{}", describe(s));
    s.snapshot()
}

fn wrapped_w(s: &S, c: &Change, label: &str) {
    let clip = c.clip.unwrap_or_else(|| panic!("{label}: no skip"));
    assert!(clip.completed(), "{label}: {clip:?}");
    assert_eq!((s.state().full_location(), link(s).0, link(s).1), (ow(0x77), 0xF0, 0), "{label}: {}", describe(s));
}

/// `long`: MOVE W 32 in the zone wraps, MOVE W 16 leaves the room; with `none` or `kind` MOVE W 32 leaves too.
#[test]
fn the_long_trigger_wraps_on_the_longest_move_only() {
    let Some(mut s) = session() else { return };
    let at = near_w(&mut s);
    s.set_clip_settings(ClipSettings::new(ClipTrigger::Long, 0));
    let c = act(&mut s, "MOVE W 32");
    wrapped_w(&s, &c, "long MOVE W 32");
    s.restore(&at);
    let c = act(&mut s, "MOVE W 16");
    assert!(c.clip.is_none());
    assert_eq!(s.state().full_location(), ow(0x76), "long MOVE W 16: {}", describe(&s));
    for trigger in [ClipTrigger::None, ClipTrigger::Kind] {
        s.set_clip_settings(ClipSettings::new(trigger, 0));
        s.restore(&at);
        let c = act(&mut s, "MOVE W 32");
        assert!(c.clip.is_none(), "{trigger:?}");
        assert_eq!(s.state().full_location(), ow(0x76), "{trigger:?} MOVE W 32: {}", describe(&s));
    }
}

/// `edge`: with E the correct exit MOVE W 8 in the zone wraps and MOVE E at the east bound leaves; with W the
/// correct exit MOVE W leaves.
#[test]
fn the_edge_trigger_wraps_out_of_the_wrong_edge_only() {
    let Some(mut s) = session() else { return };
    let at = near_w(&mut s);
    s.set_clip_settings(ClipSettings::new(ClipTrigger::Edge, Direction::E.value()));
    let c = act(&mut s, "MOVE W 8");
    wrapped_w(&s, &c, "edge MOVE W 8, exit E");
    assert!(skip_available(s.state(), Direction::E), "the east bound is in the E zone");
    let c = act(&mut s, "MOVE E 8");
    assert!(c.clip.is_none());
    assert_eq!(s.state().full_location(), ow(0x78), "edge MOVE E out of the correct exit: {}", describe(&s));
    s.set_clip_settings(ClipSettings::new(ClipTrigger::Edge, Direction::W.value()));
    s.restore(&at);
    let c = act(&mut s, "MOVE W 8");
    assert!(c.clip.is_none());
    assert_eq!(s.state().full_location(), ow(0x76), "edge MOVE W out of the correct exit: {}", describe(&s));
}

/// The skip is offered only in its zone, on rows whose run-up and landing are open, in the overworld's play mode;
/// the mask follows it (with the `kind` trigger) and is off with `none`.
#[test]
fn availability_follows_the_zone_and_the_rows() {
    let Some(mut s) = session() else { return };
    let usable = |s: &S| [Direction::N, Direction::S, Direction::W, Direction::E].map(|d| skip_available(s.state(), d));
    let kind = ClipSettings::new(ClipTrigger::Kind, 0);
    s.set_clip_settings(kind);
    load(&mut s, "start");
    assert_eq!(usable(&s), [false; 4], "the middle of $77: {}", describe(&s));
    assert!(!s.mask().kind[ActionKind::Clip.index()]);
    press_until(&mut s, LEFT, |s| link(s).0 <= 0x22);
    assert!(link(&s).0 > 0x20 && usable(&s) == [false; 4], "just outside the zone: {}", describe(&s));
    press_until(&mut s, LEFT, |s| link(s).0 <= 0x20);
    assert!(usable(&s)[2], "the zone's first pixels: {}", describe(&s));
    assert_eq!(s.mask().directions(ActionKind::Clip), vec![Direction::W]);
    s.set_clip_settings(ClipSettings::default());
    assert!(!s.mask().kind[ActionKind::Clip.index()], "action none");
    s.set_clip_settings(kind);
    // Row Y $95 of $77: columns 0-2 and $1C-$1F are solid. Link stops at X $18 and $D0 there, in the zones.
    load(&mut s, "start");
    s.press(zelda::executor::buttons::DOWN, 6);
    assert_eq!(s.state().link.position.y, 0x95, "{}", describe(&s));
    let row = s.snapshot();
    s.press(LEFT, 100);
    assert_eq!((link(&s).0, usable(&s)), (0x18, [false; 4]), "{}", describe(&s));
    s.restore(&row);
    s.press(RIGHT, 100);
    assert_eq!((link(&s).0, usable(&s)), (0xD0, [false; 4]), "{}", describe(&s));
    // The softlock rows of mechanics.md: $38 Y $8D (a W wrap would stop at X $F8 in its solid right edge), $37 Y $8D
    // (an E wrap would stand inside its two solid squares at X $00).
    load(&mut s, "0_38w_0");
    assert_eq!((link(&s).0, s.state().link.position.y), (0x00, ROW), "{}", describe(&s));
    assert_eq!(usable(&s), [false; 4], "$38 row $8D: {}", describe(&s));
    load(&mut s, "0_37e_0");
    assert_eq!((link(&s).0, s.state().link.position.y), (0xF0, ROW), "{}", describe(&s));
    assert_eq!(usable(&s), [false; 4], "$37 row $8D: {}", describe(&s));
    // The same rows' other edges are open: $38's E wrap from X $D0..$F0 is no use here, but $37's W zone is.
    // Dungeons (no wrap; W/E are no block-clip headings) and caves.
    load(&mut s, "1_22e_0");
    assert!(!usable(&s)[2] && !usable(&s)[3], "{}", describe(&s));
    load(&mut s, "0_77cave_0");
    assert_eq!(usable(&s), [false; 4], "cave: {}", describe(&s));
    assert!(!s.mask().kind[ActionKind::Clip.index()]);
}

/// A hit during the macro ends it at once with the damage interrupt; from where it leaves Link MOVE works and a new
/// skip completes. Room $5B: a Moblin put on the back-off path (the test pokes its position).
#[test]
fn a_hit_aborts_the_skip_and_link_can_go_on() {
    let Some(mut s) = session() else { return };
    load(&mut s, "0_5bw_0");
    // Let the room's monsters appear, then put one on Link's row 12 pixels in: the back-off to G' ($10) walks into it.
    for _ in 0..20 {
        if s.state().enemies.iter().any(|e| e.spawn_state == 0) {
            break;
        }
        act(&mut s, "WAIT 8");
    }
    let st = s.state().clone();
    assert_eq!((link(&s).0, st.full_location()), (0x00, ow(0x5B)), "{}", describe(&s));
    assert!(skip_available(&st, Direction::W), "{}", describe(&s));
    let slot = st.enemies.iter().find(|e| e.spawn_state == 0).expect("a monster").index as u16;
    let y = st.link.position.y as u8;
    s.nes_mut().poke(0x70 + slot, 0x0C);
    s.nes_mut().poke(0x84 + slot, y);
    s.apply_ram(&[]);
    let c = act(&mut s, "CLIP W");
    let clip = c.clip.expect("clip outcome");
    assert_eq!(c.interrupt, Some(zelda::Interrupt::Damage), "{clip:?} {}", describe(&s));
    assert_eq!(clip.abort, Some(zelda::clips::ClipAbort::Interrupted(zelda::Interrupt::Damage)));
    assert!(c.health_lost > 0.0);
    assert_eq!((s.state().full_location(), s.state().mode()), (ow(0x5B), 5), "{}", describe(&s));
    let after_hit = describe(&s);
    let before = s.state().link.position;
    assert!(s.mask().is_allowed(&Action::parse("MOVE E 8").unwrap()), "{after_hit}");
    act(&mut s, "MOVE E 8");
    assert_ne!(s.state().link.position, before, "MOVE E after the abort ({after_hit}): {}", describe(&s));
    assert!(skip_available(s.state(), Direction::W), "after the abort ({after_hit}): {}", describe(&s));
    let c = act(&mut s, "CLIP W");
    let clip = c.clip.expect("clip outcome");
    assert!(clip.completed(), "{clip:?} {:?} {}", c.interrupt, describe(&s));
    assert_eq!((s.state().full_location(), link(&s).0, link(&s).1), (ow(0x5B), 0xF0, 0), "{}", describe(&s));
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

/// Wherever the macro can be cut on the way through a wrap (every frame from the press to the end point, past the
/// bound included: X $FF..$F1 and $F1..$FF), the MOVE mask matches the game.
#[test]
fn the_move_mask_is_true_on_every_frame_of_a_wrap() {
    let Some(mut s) = session() else { return };
    for side in [Direction::W, Direction::E] {
        load(&mut s, "start");
        let (toward, g) = if side == Direction::W { (LEFT, 0x08) } else { (RIGHT, 0xE8) };
        press_until(&mut s, toward, |s| link(s).0 == g && link(s).1 == 0);
        s.press(toward, 2);
        s.press(zelda::executor::buttons::UP, 1);
        let end = if side == Direction::W { 0xF0 } else { 0x00 };
        let mut past = 0;
        for frame in 0..24 {
            let label = format!("{} wrap, frame {frame} after the press", side.name());
            assert_move_mask_matches_the_game(&mut s, &label);
            let x = link(&s).0;
            past += i32::from(x > 0xF0);
            if x == end && link(&s).1 == 0 {
                break;
            }
            s.press(toward, 1);
        }
        assert!(past >= 8, "{} wrap: {past} frames past the bound: {}", side.name(), describe(&s));
        assert_eq!((link(&s).0, s.state().full_location()), (end, ow(0x77)), "{}", describe(&s));
    }
}

/// Level 1 room $22, the east block from below (column $A0, every Y of the zone $95..$A5 in both cadences): CLIP N
/// ends 8 pixels into the block at ($A0, $8D), and MOVE W then reaches the stairs (cellar $7F).
#[test]
fn block_clip_east_block_of_level_1_room_22() {
    let Some(mut s) = session() else { return };
    load(&mut s, "1_22e_0");
    let y = |s: &S| s.state().link.position.y;
    press_until(&mut s, LEFT, |s| link(s).0 == 0xB0 && link(s).1 == 0);
    press_until(&mut s, zelda::executor::buttons::DOWN, |s| y(s) == 0xAD && link(s).1 == 0);
    press_until(&mut s, LEFT, |s| link(s).0 == 0xA0 && link(s).1 == 0);
    assert!(!skip_available(s.state(), Direction::N), "$AD is outside the zone: {}", describe(&s));
    for d in [Direction::W, Direction::E] {
        assert!(!skip_available(s.state(), d), "no W/E block clip");
    }
    let base = s.snapshot();
    let mut starts = 0;
    for frames in 1..30 {
        s.restore(&base);
        s.press(zelda::executor::buttons::UP, frames);
        if !(0x95..=0xA5).contains(&y(&s)) {
            continue;
        }
        for back in [0, 1, 3] {
            s.restore(&base);
            s.press(zelda::executor::buttons::UP, frames);
            s.press(zelda::executor::buttons::DOWN, back);
            if !(0x95..=0xA5).contains(&y(&s)) {
                continue;
            }
            let label = format!("Up {frames}, Down {back}: {}", describe(&s));
            assert!(skip_available(s.state(), Direction::N), "{label}");
            starts += 1;
            let c = act(&mut s, "CLIP N");
            let clip = c.clip.expect("clip");
            assert!(clip.completed(), "{label}: {clip:?} now {}", describe(&s));
            assert_eq!((link(&s).0, y(&s), link(&s).1), (0xA0, 0x8D, 0), "{label}: {}", describe(&s));
            assert!(s.mask().is_allowed(&Action::parse("MOVE W 8").unwrap()), "{label}");
            for _ in 0..6 {
                if s.state().full_location().value == 0x7F {
                    break;
                }
                act(&mut s, "MOVE W 8");
            }
            assert_eq!(s.state().full_location().value, 0x7F, "{label}: the stairs: {}", describe(&s));
        }
    }
    assert!(starts >= 10, "{starts} starts");
}
