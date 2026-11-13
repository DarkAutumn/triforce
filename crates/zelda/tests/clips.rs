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

fn y(s: &S) -> i32 {
    s.state().link.position.y
}

/// Walks `d` a frame at a time until Link's coordinate on that axis is `to`, on a grid point.
fn walk(s: &mut S, d: Direction, to: i32) {
    let horizontal = matches!(d, Direction::W | Direction::E);
    press_until(s, zelda::executor::direction_buttons(d), |s| {
        (if horizontal { link(s).0 } else { y(s) }) == to && link(s).1 == 0
    });
}

/// MOVE `text` until Link is in cellar $7F (at most 8 times).
fn move_to_cellar(s: &mut S, text: &str, label: &str) {
    for _ in 0..8 {
        if s.state().full_location() == MapLocation::new(1, 0x7F, true) {
            return;
        }
        act(s, text);
    }
    assert_eq!(s.state().full_location(), MapLocation::new(1, 0x7F, true), "{label}: the stairs: {}", describe(s));
}

/// One square of level 1 room $22 (`zelda-asm/mechanics.md` "Block clip", "Reverse block clip"): Link at rest on
/// the square's column (`base`), the heading, the zone (G' to T), the end point 8 pixels into the square and the
/// way on from there to the stairs.
struct Square {
    label: &'static str,
    heading: Direction,
    zone: (i32, i32),
    end: (i32, i32),
    way_on: fn(&mut S, &str),
}

/// Every start of the zone reached from `base` by real walking (the heading held 0 frames up to the frame Link stops
/// on T, each followed by 0, 1 or 3 frames back: both cadences, both offset signs, stopped mid-cell): the skip is
/// offered in the heading only, never W or E; CLIP completes with one press at the end point, without damage (the
/// traps are in the room); the way on reaches the stairs (cellar $7F). Holding on against the square on T is not a
/// start: against the push block it pushes.
fn clip_from_every_start(s: &mut S, base: &Snap, square: &Square) {
    let Square { label, heading, zone: (from, t), end, way_on } = *square;
    let toward = zelda::executor::direction_buttons(heading);
    let away = zelda::executor::direction_buttons(heading.opposite());
    let zone = from.min(t)..=from.max(t);
    let mut starts = 0;
    for frames in 0..30 {
        s.restore(base);
        s.press(toward, frames);
        let on_t = y(s) == t && link(s).1 == 0;
        for back in [0, 1, 3] {
            s.restore(base);
            s.press(toward, frames);
            s.press(away, back);
            if !zone.contains(&y(s)) {
                continue;
            }
            let at = format!("{label}: {} {frames}, {} {back}: {}", heading.name(), heading.opposite().name(), describe(s));
            let offered = [Direction::N, Direction::S, Direction::W, Direction::E].map(|d| skip_available(s.state(), d));
            let want = [Direction::N, Direction::S, Direction::W, Direction::E].map(|d| d == heading);
            assert_eq!(offered, want, "{at}");
            starts += 1;
            let c = act(s, &format!("CLIP {}", heading.name()));
            let clip = c.clip.expect("clip");
            assert!(clip.completed() && clip.presses == 1, "{at}: {clip:?} now {}", describe(s));
            assert_eq!(c.health_lost, 0.0, "{at}");
            assert_eq!((link(s).0, y(s), link(s).1), (end.0, end.1, 0), "{at}: {}", describe(s));
            way_on(s, &at);
        }
        if on_t {
            break;
        }
    }
    assert!(starts >= 10, "{label}: {starts} starts");
}

/// Level 1 room $22, the east block from below (column $A0, zone $95..$A5): CLIP N ends 8 pixels into the block at
/// ($A0, $8D), and MOVE W then reaches the stairs.
#[test]
fn block_clip_east_block_of_level_1_room_22() {
    let Some(mut s) = session() else { return };
    load(&mut s, "1_22e_0");
    walk(&mut s, Direction::W, 0xB0);
    walk(&mut s, Direction::S, 0xAD);
    walk(&mut s, Direction::W, 0xA0);
    assert!(!skip_available(s.state(), Direction::N), "$AD is outside the zone: {}", describe(&s));
    let base = s.snapshot();
    let square = Square {
        label: "east block from below",
        heading: Direction::N,
        zone: (0xA5, 0x95),
        end: (0xA0, 0x8D),
        way_on: |s, at| {
            assert!(s.mask().is_allowed(&Action::parse("MOVE W 8").unwrap()), "{at}");
            move_to_cellar(s, "MOVE W 8", at);
        },
    };
    clip_from_every_start(&mut s, &base, &square);
}

/// Push block ($60, $90) at slot $0B, as the room set it up: not moved and still pending.
fn push_block_untouched(s: &S, at: &str) {
    let st = s.state();
    assert!(st.push_block_pending(), "{at}: push started");
    assert_eq!((st.peek(0x70 + 0x0B), st.peek(0x84 + 0x0B)), (0x60, 0x90), "{at}: the push block moved");
}

/// The west block, the room's push block, from below (from the cellar's exit spot ($60, $AD); zone $95..$A5): CLIP
/// N ends at ($60, $8D) without starting a push (the block stays, still pending, also after a WAIT), and MOVE E then
/// walks into the diamond and onto the stairs.
#[test]
fn block_clip_west_push_block_of_level_1_room_22() {
    let Some(mut s) = session() else { return };
    load(&mut s, "1_22c_0");
    assert_eq!((link(&s).0, y(&s)), (0x60, 0xAD), "{}", describe(&s));
    push_block_untouched(&s, "start");
    let base = s.snapshot();
    let square = Square {
        label: "west block from below",
        heading: Direction::N,
        zone: (0xA5, 0x95),
        end: (0x60, 0x8D),
        way_on: |s, at| {
            push_block_untouched(s, at);
            act(s, "WAIT 32");
            push_block_untouched(s, at);
            move_to_cellar(s, "MOVE E 8", at);
        },
    };
    clip_from_every_start(&mut s, &base, &square);
}

/// The west block's push spot below it (Link stopped on T = ($60, $95)) is inside its clip zone, so the trigger
/// decides between a push and a clip: a plain MOVE N 16 or 32 pushes the block ($10 frames against it), three MOVE N 8
/// don't (the count restarts between options); under `long` MOVE N 32 clips and MOVE N 16 still pushes; under `edge`
/// every MOVE N clips and the block can't be pushed from there.
#[test]
fn the_west_blocks_push_spot_is_in_its_clip_zone() {
    let Some(mut s) = session() else { return };
    load(&mut s, "1_22c_0");
    walk(&mut s, Direction::N, 0x95);
    let on_t = s.snapshot();
    for (trigger, moves, clipped) in [
        (ClipTrigger::None, &["MOVE N 16"][..], false),
        (ClipTrigger::None, &["MOVE N 32"][..], false),
        (ClipTrigger::None, &["MOVE N 8", "MOVE N 8", "MOVE N 8"][..], false),
        (ClipTrigger::Long, &["MOVE N 16"][..], false),
        (ClipTrigger::Long, &["MOVE N 32"][..], true),
        (ClipTrigger::Edge, &["MOVE N 8"][..], true),
        (ClipTrigger::Edge, &["MOVE N 16"][..], true),
        (ClipTrigger::Edge, &["MOVE N 32"][..], true),
    ] {
        s.set_clip_settings(ClipSettings::new(trigger, 0));
        s.restore(&on_t);
        let mut skips = 0;
        for text in moves {
            skips += u32::from(act(&mut s, text).clip.is_some_and(|c| c.completed()));
        }
        act(&mut s, "WAIT 32");
        let label = format!("{trigger:?} {moves:?}");
        assert_eq!(skips, u32::from(clipped), "{label}: {}", describe(&s));
        let pushed = !clipped && moves.len() == 1;
        assert_eq!(!s.state().push_block_pending(), pushed, "{label}: {}", describe(&s));
        if clipped {
            assert_eq!((link(&s).0, y(&s)), (0x60, 0x8D), "{label}: {}", describe(&s));
            push_block_untouched(&s, &label);
        }
    }
}

/// The lower diagonal blocks from below (zone $A5..$B5, the run-up within the bottom traps' reach): the lower-left
/// ($70, $A0) to ($70, $9D), then E to X $80 and N onto the stairs; the lower-right ($90, $A0) to ($90, $9D), then W
/// and N.
#[test]
fn block_clip_lower_diagonal_blocks_of_level_1_room_22() {
    let Some(mut s) = session() else { return };
    load(&mut s, "1_22c_0");
    walk(&mut s, Direction::E, 0x70);
    let base = s.snapshot();
    let square = Square {
        label: "lower-left block from below",
        heading: Direction::N,
        zone: (0xB5, 0xA5),
        end: (0x70, 0x9D),
        way_on: |s, at| {
            act(s, "MOVE E 16");
            assert_eq!((link(s).0, y(s)), (0x80, 0x9D), "{at}: stopped by the lower-right block: {}", describe(s));
            move_to_cellar(s, "MOVE N 8", at);
        },
    };
    clip_from_every_start(&mut s, &base, &square);
    load(&mut s, "1_22e_0");
    walk(&mut s, Direction::W, 0xB0);
    walk(&mut s, Direction::S, 0xAD);
    walk(&mut s, Direction::W, 0x90);
    let base = s.snapshot();
    let square = Square {
        label: "lower-right block from below",
        heading: Direction::N,
        zone: (0xB5, 0xA5),
        end: (0x90, 0x9D),
        way_on: |s, at| {
            act(s, "MOVE W 16");
            assert_eq!((link(s).0, y(s)), (0x80, 0x9D), "{at}: stopped by the lower-left block: {}", describe(s));
            move_to_cellar(s, "MOVE N 8", at);
        },
    };
    clip_from_every_start(&mut s, &base, &square);
}

/// The reverse block clip (heading S, zone $6D..$7D) from above the east block to ($A0, $85) and from above the west
/// block to ($60, $85); from either, sideways to X $80 and S onto the stairs.
#[test]
fn reverse_block_clip_east_and_west_blocks_of_level_1_room_22() {
    let Some(mut s) = session() else { return };
    load(&mut s, "1_22e_0");
    walk(&mut s, Direction::W, 0xB0);
    walk(&mut s, Direction::N, 0x6D);
    walk(&mut s, Direction::W, 0xA0);
    let base = s.snapshot();
    let square = Square {
        label: "east block from above",
        heading: Direction::S,
        zone: (0x6D, 0x7D),
        end: (0xA0, 0x85),
        way_on: |s, at| {
            walk(s, Direction::W, 0x80);
            move_to_cellar(s, "MOVE S 8", at);
        },
    };
    clip_from_every_start(&mut s, &base, &square);
    load(&mut s, "1_22c_0");
    walk(&mut s, Direction::W, 0x40);
    walk(&mut s, Direction::N, 0x6D);
    walk(&mut s, Direction::E, 0x60);
    let base = s.snapshot();
    let square = Square {
        label: "west block from above",
        heading: Direction::S,
        zone: (0x6D, 0x7D),
        end: (0x60, 0x85),
        way_on: |s, at| {
            push_block_untouched(s, at);
            walk(s, Direction::E, 0x80);
            move_to_cellar(s, "MOVE S 8", at);
        },
    };
    clip_from_every_start(&mut s, &base, &square);
}

/// Every grid point of a room where the block clip is offered, Link put there at rest (poked; availability reads the
/// state alone): (X, Y, heading).
fn offered_in_room(s: &mut S) -> Vec<(i32, i32, Direction)> {
    let base = s.snapshot();
    let mut out = Vec::new();
    for y in (0x5D..=0xBD).step_by(8) {
        for x in (0x18..=0xD8).step_by(8) {
            s.restore(&base);
            let nes = s.nes_mut();
            nes.poke(0x70, x as u8);
            nes.poke(0x84, y as u8);
            nes.poke(0x394, 0);
            s.apply_ram(&[]);
            for d in [Direction::N, Direction::S, Direction::W, Direction::E] {
                if skip_available(s.state(), d) {
                    out.push((x, y, d));
                }
            }
        }
    }
    s.restore(&base);
    out
}

/// What is offered in level 1 room $22 and level 2 room $7D, grid point by grid point. $22: the squares whose clip
/// leads into the diamond (from below the west, east and lower diagonal blocks, from above the upper diagonal, west
/// and east blocks) or out of it (from inside, the north block), each over its zone G'..T; nothing half a column off
/// a square (that clip leads only back out), not the south block from below (G' = $C5 is past the room bound; it too
/// only leads back out), not the south block from inside (G' = ($80, $8D) is the stairs spot) and never W or E. $7D:
/// nothing (its statues stand alone, 16 pixels apart: G' is inside the next statue and every tile beside one is
/// open from T).
#[test]
fn the_block_clip_is_offered_where_it_leads_somewhere_in_22_and_nowhere_in_7d() {
    let Some(mut s) = session() else { return };
    load(&mut s, "1_22c_0");
    let mut want = Vec::new();
    let zone = |want: &mut Vec<_>, x: i32, ys: [i32; 3], d: Direction| want.extend(ys.map(|y| (x, y, d)));
    // Heading N: west and east blocks, the lower diagonal blocks, the north block from inside the diamond.
    for x in [0x60, 0xA0] {
        zone(&mut want, x, [0x95, 0x9D, 0xA5], Direction::N);
    }
    for x in [0x70, 0x90] {
        zone(&mut want, x, [0xA5, 0xAD, 0xB5], Direction::N);
    }
    zone(&mut want, 0x80, [0x75, 0x7D, 0x85], Direction::N);
    // Heading S: the upper diagonal blocks, west and east blocks.
    for x in [0x70, 0x90] {
        zone(&mut want, x, [0x5D, 0x65, 0x6D], Direction::S);
    }
    for x in [0x60, 0xA0] {
        zone(&mut want, x, [0x6D, 0x75, 0x7D], Direction::S);
    }
    let key = |&(x, y, d): &(i32, i32, Direction)| (y, x, d.value());
    want.sort_by_key(key);
    let mut got = offered_in_room(&mut s);
    got.sort_by_key(key);
    assert_eq!(got, want);
    load(&mut s, "2_7ds_0");
    for _ in 0..3 {
        act(&mut s, "MOVE N 8");
    }
    assert_eq!((s.state().full_location(), s.state().var(Var::DoorwayDir)), (MapLocation::new(2, 0x7D, false), 0));
    assert_eq!(offered_in_room(&mut s), vec![], "$7D");
}

/// `long` and `edge` in a dungeon, at the east block of $22 (Link on G = ($A0, $9D) after a cell walked Left, or
/// stopped on T = $95): `long` clips with MOVE N 32 only (MOVE N 16, 8 and 4 are plain and never pass T); `edge`
/// with MOVE N of every duration; `none` and `kind` never with a MOVE. On T the mask has CLIP N under `kind` only,
/// and MOVE N (a wall for the plain walk) under `long` and `edge` only. From above the east block (Y $75) `edge`
/// clips S with MOVE S 4 and `long` with MOVE S 32.
#[test]
fn long_and_edge_trigger_the_block_clip_in_a_dungeon() {
    let Some(mut s) = session() else { return };
    load(&mut s, "1_22e_0");
    walk(&mut s, Direction::W, 0xB0);
    let corner = s.snapshot();
    walk(&mut s, Direction::S, 0x9D);
    walk(&mut s, Direction::W, 0xA0);
    let on_g = s.snapshot();
    walk(&mut s, Direction::N, 0x95);
    let on_t = s.snapshot();
    let clip_n = Action::parse("CLIP N").unwrap();
    for trigger in ClipTrigger::ALL {
        s.set_clip_settings(ClipSettings::new(trigger, 0));
        s.restore(&on_t);
        assert!(skip_available(s.state(), Direction::N), "{}", describe(&s));
        let mask = s.mask();
        assert_eq!(mask.is_allowed(&clip_n), trigger == ClipTrigger::Kind, "{trigger:?} CLIP N on T");
        let move_n = mask.is_allowed(&Action::parse("MOVE N 8").unwrap());
        assert_eq!(move_n, matches!(trigger, ClipTrigger::Long | ClipTrigger::Edge), "{trigger:?} MOVE N on T");
        for duration in [4, 8, 16, 32] {
            let clips = match trigger {
                ClipTrigger::Long => duration == 32,
                ClipTrigger::Edge => true,
                _ => false,
            };
            for (start, snap) in [("G", &on_g), ("T", &on_t)] {
                s.restore(snap);
                let label = format!("{trigger:?} MOVE N {duration} from {start}");
                let c = act(&mut s, &format!("MOVE N {duration}"));
                assert_eq!(c.clip.is_some(), clips, "{label}: {:?} {}", c.clip, describe(&s));
                if clips {
                    assert!(c.clip.unwrap().completed(), "{label}: {:?}", c.clip);
                    assert_eq!((link(&s).0, y(&s), link(&s).1), (0xA0, 0x8D, 0), "{label}: {}", describe(&s));
                } else {
                    assert!(y(&s) >= 0x95 && link(&s).0 == 0xA0, "{label}: {}", describe(&s));
                }
            }
        }
    }
    s.restore(&corner);
    walk(&mut s, Direction::N, 0x75);
    walk(&mut s, Direction::W, 0xA0);
    let above = s.snapshot();
    for (trigger, text) in [(ClipTrigger::Edge, "MOVE S 4"), (ClipTrigger::Long, "MOVE S 32")] {
        s.set_clip_settings(ClipSettings::new(trigger, 0));
        s.restore(&above);
        let c = act(&mut s, text);
        assert!(c.clip.is_some_and(|c| c.completed()), "{trigger:?} {text}: {:?}", c.clip);
        assert_eq!((link(&s).0, y(&s), link(&s).1), (0xA0, 0x85, 0), "{trigger:?} {text}: {}", describe(&s));
    }
}

/// The way through $22 with button-free actions only: from the E door MOVE W into the corner of the east block, MOVE
/// S 8 and MOVE W (Link turns on the next grid point and the lower-right block stops him on G = ($A0, $9D)), one
/// CLIP N (or, under `long`, MOVE N 32), MOVE W 32 onto the stairs: in cellar $7F 218 frames after the room's first
/// frame, no damage (mechanics.md: 117 frames from the doorway to the stairs by controller input, before the
/// stairs' transition).
#[test]
fn the_way_through_22_by_moves_and_one_clip() {
    let Some(mut s) = session() else { return };
    for (trigger, clip) in [(ClipTrigger::Kind, "CLIP N"), (ClipTrigger::Long, "MOVE N 32")] {
        load(&mut s, "1_22e_0");
        s.set_clip_settings(ClipSettings::new(trigger, 0));
        let (mut frames, mut skips, mut hurt) = (0, 0, 0.0);
        let plan = ["MOVE W 32", "MOVE W 32", "MOVE W 8", "MOVE S 8", "MOVE W 8", "MOVE W 8", clip, "MOVE W 32"];
        for text in plan {
            let action = Action::parse(text).unwrap();
            assert!(s.mask().is_allowed(&action), "{trigger:?} {text}: {}", describe(&s));
            let c = s.act(action).unwrap();
            frames += c.frames;
            hurt += c.health_lost;
            skips += u32::from(c.clip.is_some_and(|c| c.completed()));
            if text == clip {
                assert_eq!((link(&s).0, y(&s), c.frames), (0xA0, 0x8D, 14), "{trigger:?} {text}: {}", describe(&s));
            }
        }
        assert_eq!(s.state().full_location(), MapLocation::new(1, 0x7F, true), "{trigger:?}: {}", describe(&s));
        assert_eq!((frames, skips, hurt), (218, 1, 0.0), "{trigger:?}");
    }
}

const CARDINAL: [Direction; 4] = [Direction::N, Direction::S, Direction::W, Direction::E];

/// Empties the monster slots (round 20's Dungeon 4 rooms below): frame counts and the macro's runs don't depend on
/// where the Vires walk.
fn clear_monsters(s: &mut S) {
    for slot in 1..12 {
        s.nes_mut().poke((zelda::tables::OBJ_ID.0 + slot) as u16, 0);
    }
    s.apply_ram(&[]);
}

/// Link put at rest on (x, y) (offset 0), as `offered_in_room` does.
fn put(s: &mut S, x: i32, y: i32) {
    let nes = s.nes_mut();
    nes.poke(0x70, x as u8);
    nes.poke(0x84, y as u8);
    nes.poke(0x394, 0);
    s.apply_ram(&[]);
}

/// Holding `d` for 8 frames moves Link.
fn walks(s: &mut S, d: Direction) -> bool {
    let at = s.state().link.position;
    s.press(zelda::executor::direction_buttons(d), 8);
    s.state().link.position != at
}

/// Every Dungeon 4 start state (round 20), Link put on every grid point at rest out of the doorway: the block clip is
/// offered in room $03 (under its two inner blocks at X $60 and $90, into the side pockets of the triforce's
/// enclosure) and room $50 (its diagonal rows of blocks: 20 squares, each over its zone G'..T), nowhere else: not in
/// the water rooms $20, $21, $30, $31 (water is no block; the ladder crosses one square of it) nor in the cellar $60.
/// In $03 and $50 every offered clip is right in the game: CLIP from each grid point completes with one press 8
/// pixels past T, and from there a sideways walk goes on where Link stopped on T is walled off (useful clips only).
#[test]
fn the_block_clip_in_dungeon_4() {
    let Some(mut s) = session() else { return };
    let mut names: Vec<String> = std::fs::read_dir(common::repo().join("triforce/states"))
        .unwrap()
        .filter_map(|e| e.unwrap().file_name().into_string().unwrap().strip_suffix(".state").map(str::to_owned))
        .filter(|n| n.starts_with("4_"))
        .collect();
    names.sort();
    assert_eq!(names.len(), 54, "{names:?}");
    let mut want_03 = Vec::new();
    for y in [0x85, 0x8D, 0x95] {
        want_03.extend([(0x60, y, Direction::N), (0x90, y, Direction::N)]);
    }
    for name in &names {
        load(&mut s, name);
        s.apply_ram(&[(Var::DoorwayDir, 0)]);
        let got = offered_in_room(&mut s);
        match &name[..4] {
            "4_03" => assert_eq!(got, want_03, "{name}"),
            "4_50" => assert_eq!(got.len(), 60, "{name}: {got:?}"),
            _ => assert_eq!(got, vec![], "{name}"),
        }
    }
    for name in ["4_03s_0", "4_50e_0"] {
        load(&mut s, name);
        s.apply_ram(&[(Var::DoorwayDir, 0)]);
        clear_monsters(&mut s);
        let room = s.snapshot();
        let mut squares = std::collections::BTreeMap::new();
        for (x, gy, d) in offered_in_room(&mut s) {
            s.restore(&room);
            put(&mut s, x, gy);
            let t = i32::from(zelda::clips::skip_target(s.state(), d).expect("offered").t);
            let end = if d == Direction::N { t - 8 } else { t + 8 };
            let at = format!("{name} ({x:#04x}, {gy:#04x}) CLIP {} (T {t:#04x})", d.name());
            let c = act(&mut s, &format!("CLIP {}", d.name()));
            let clip = c.clip.expect("clip");
            assert!(clip.completed() && clip.presses == 1, "{at}: {clip:?} now {}", describe(&s));
            assert_eq!((link(&s).0, y(&s), link(&s).1), (x, end, 0), "{at}: {}", describe(&s));
            let clipped = s.snapshot();
            let way_on: Vec<Direction> = [Direction::W, Direction::E]
                .into_iter()
                .filter(|&side| {
                    s.restore(&clipped);
                    let on = walks(&mut s, side);
                    s.restore(&room);
                    put(&mut s, x, t);
                    on && !walks(&mut s, side)
                })
                .collect();
            assert!(!way_on.is_empty(), "{at}: no sideways way on that T doesn't have");
            squares.entry((x, t, d.name())).or_insert_with(Vec::new).push((gy, way_on));
        }
        for ((x, t, d), zone) in &squares {
            assert_eq!(zone.len(), 3, "{name} ({x:#04x}, T {t:#04x}) {d}: {zone:?}");
            println!("{name}: X {x:#04x} heading {d} T {t:#04x}: from {:?}, then {:?}", zone.iter().map(|z| z.0).collect::<Vec<_>>(), zone[0].1);
        }
        assert_eq!(squares.len(), if name == "4_03s_0" { 2 } else { 20 }, "{name}");
    }
}

/// Room 4:$50 (the route enters E and leaves N; diagonal rows of blocks, `zelda-asm/levels/level-4.md`): the plain
/// way goes round the bottom of the right diagonal (W, S, W, S, W, N, E, N, W, N: 14 MOVEs, 424 frames to 4:$40
/// from the room's first frame, the last MOVE's 191 frames of walking out and the scroll included); one CLIP N into
/// the diagonal's square above ($B0, $8D), where Link's MOVE W stops, cuts through it (W, CLIP N, W, N, W, N: 8
/// actions, 318 frames). Monsters cleared. In the E doorway nothing is offered and the sword keeps to the doorway's
/// axis (round 20); after the clip the MOVE mask matches the game and the attack masks follow
/// `GameView::sword_directions`.
#[test]
fn the_way_through_4_50_by_one_block_clip() {
    let Some(mut s) = session() else { return };
    let plain = [
        "MOVE W 32", "MOVE W 32", "MOVE S 32", "MOVE W 16", "MOVE W 16", "MOVE S 32", "MOVE W 16", "MOVE W 8",
        "MOVE N 32", "MOVE N 32", "MOVE E 8", "MOVE N 32", "MOVE W 8", "MOVE N 32",
    ];
    let clip = ["MOVE W 32", "MOVE W 32", "CLIP N", "MOVE W 8", "MOVE N 32", "MOVE W 16", "MOVE W 16", "MOVE N 32"];
    let mut frames = Vec::new();
    for (trigger, plan) in [(ClipTrigger::None, &plain[..]), (ClipTrigger::Kind, &clip[..])] {
        load(&mut s, "4_50e_0");
        clear_monsters(&mut s);
        s.set_clip_settings(ClipSettings::new(trigger, 0));
        assert_ne!(s.state().var(Var::DoorwayDir), 0, "{}", describe(&s));
        assert!(CARDINAL.iter().all(|&d| !skip_available(s.state(), d)), "in the doorway: {}", describe(&s));
        assert!(!s.mask().kind[ActionKind::Clip.index()]);
        for kind in [ActionKind::Sword, ActionKind::Beams] {
            assert!(s.mask().directions(kind).iter().all(|d| matches!(d, Direction::W | Direction::E)), "{kind:?}");
        }
        let (mut total, mut hurt) = (0, 0.0);
        for text in plan {
            let action = Action::parse(text).unwrap();
            assert!(s.mask().is_allowed(&action), "{trigger:?} {text}: {}", describe(&s));
            let c = s.act(action).unwrap();
            total += c.frames;
            hurt += c.health_lost;
            if action.kind == ActionKind::Clip {
                assert!(c.clip.is_some_and(|c| c.completed() && c.presses == 1), "{:?}", c.clip);
                assert_eq!((link(&s).0, y(&s), c.frames), (0xB0, 0x7D, 14), "{}", describe(&s));
                assert_move_mask_matches_the_game(&mut s, "after CLIP N");
                let attacks = s.state().sword_directions();
                assert_eq!(attacks.len(), 4, "inside the room's bounds: {attacks:?} {}", describe(&s));
                let sorted = |mut v: Vec<Direction>| {
                    v.sort_by_key(|d| d.value());
                    v
                };
                for kind in [ActionKind::Sword, ActionKind::Beams] {
                    if s.mask().kind[kind.index()] {
                        assert_eq!(sorted(s.mask().directions(kind)), sorted(attacks.clone()), "{kind:?}");
                    }
                }
            }
        }
        assert_eq!(s.state().full_location(), MapLocation::new(4, 0x40, false), "{trigger:?}: {}", describe(&s));
        assert_eq!(hurt, 0.0, "{trigger:?}");
        frames.push(total);
    }
    assert_eq!(frames, [424, 318]);
}

/// The skip is never offered in a dungeon doorway: every door entry state of levels 1-9, held into the room a frame
/// at a time while Link is in the doorway (`DoorwayDir`, or outside the room's bounds X $20..$D0, Y $5D..$BD),
/// offers no skip in any direction and no CLIP under `kind`.
#[test]
fn no_skip_in_a_doorway() {
    let Some(mut s) = session() else { return };
    let mut names: Vec<String> = std::fs::read_dir(common::repo().join("triforce/states"))
        .unwrap()
        .filter_map(|e| e.unwrap().file_name().into_string().unwrap().strip_suffix(".state").map(str::to_owned))
        .filter(|n| {
            let b = n.as_bytes();
            n.len() == 7 && (b'1'..=b'9').contains(&b[0]) && b[1] == b'_' && b"nsew".contains(&b[4]) && b[5] == b'_'
        })
        .collect();
    names.sort();
    assert!(names.len() > 150, "{} states", names.len());
    let mut frames = 0;
    for name in &names {
        load(&mut s, name);
        s.set_clip_settings(ClipSettings::new(ClipTrigger::Kind, 0));
        let inward = match name.as_bytes()[4] {
            b'n' => Direction::S,
            b's' => Direction::N,
            b'e' => Direction::W,
            _ => Direction::E,
        };
        let room = s.state().full_location();
        for _ in 0..64 {
            let st = s.state();
            let (x, y) = (st.link.position.x, st.link.position.y);
            let doorway = st.var(Var::DoorwayDir) != 0 || !(0x20..=0xD0).contains(&x) || !(0x5D..=0xBD).contains(&y);
            if !doorway || st.mode() != 5 || st.full_location() != room {
                break;
            }
            assert!(CARDINAL.iter().all(|&d| !skip_available(st, d)), "{name}: {}", describe(&s));
            assert!(!s.mask().kind[ActionKind::Clip.index()], "{name}: {}", describe(&s));
            s.press(zelda::executor::direction_buttons(inward), 1);
            frames += 1;
        }
    }
    assert!(frames > 1000, "{frames} doorway frames");
}

/// With the ladder out (`LadderSlot`) no skip is offered: 4:$30 from its E door with the ladder, Link walks W to X $80
/// and N onto the ladder over the water row; on every frame the ladder is out nothing is offered and CLIP is masked.
#[test]
fn no_skip_with_the_ladder_out() {
    let Some(mut s) = session() else { return };
    load(&mut s, "4_30e_0");
    clear_monsters(&mut s);
    s.set_clip_settings(ClipSettings::new(ClipTrigger::Kind, 0));
    assert!(s.state().ladder(), "{}", describe(&s));
    walk(&mut s, Direction::W, 0x80);
    let mut out = 0;
    for _ in 0..48 {
        s.press(zelda::executor::buttons::UP, 1);
        if s.state().var(Var::LadderSlot) != 0 {
            out += 1;
            assert!(CARDINAL.iter().all(|&d| !skip_available(s.state(), d)), "{}", describe(&s));
            assert!(!s.mask().kind[ActionKind::Clip.index()], "{}", describe(&s));
        }
    }
    assert!(out >= 8, "{out} frames with the ladder out");
    assert!(y(&s) < 0x75, "across the water: {}", describe(&s));
}
