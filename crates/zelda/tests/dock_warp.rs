//! Round 36: the dock warp (docs/routes/d3first.md "The wrong warps"). A whirlwind carrying Link, in the water on a
//! raft dock's row W of the dock, over the dock's trigger lands him one row above the whirlwind's stand-in room
//! (`zelda::game_map::dock_landing`): 0x55 -> 0x34 by Level 4's whirlwind, 0x3f -> 0x31 by Level 7's. The engine plays
//! it from the dock after the raft's ride down: MOVE S off the trigger, the block clip N into the water
//! (`zelda::clips::dock_clip_at`), RECORDER with every play the index needs (`Executor::run_dock_warp`), offered only
//! while the session has a dock warp (`Session::set_dock_warp`).
//!
//! SETUP (RAM writes, the tests are about the warp's mechanics): the room's monsters and their shots are removed
//! before every action (a Zora's fireball interrupts the clip or the plays; the live-monster tries are
//! docs/routes/d3first-walks-c.md §2).
#![cfg(feature = "zcore")]

mod common;

use common::{Counted, load_named_state, open_core};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;
use zelda::clips::{dock_clip_at, dock_spot};
use zelda::game_map::{dock_landing, dock_level};
use zelda::objectives::route_entry;
use zelda::whirlwind::LEVEL_INDEX;
use zelda::{Action, GameMap, LoadOptions, MapLocation, Session, Var};

type S = Session<Counted>;

fn ow(room: u8) -> MapLocation {
    MapLocation::new(0, room, false)
}

fn allowed(s: &S, text: &str) -> bool {
    s.mask().is_allowed(&Action::parse(text).unwrap())
}

/// Object types kept by [`tidy`] (as `zelda/tests/d8.rs`): the dock, the ladder, the rock wall, the whirlwind, a push
/// block, the pushed gravestone, the burned tree, the flute secret.
const KEEP: [u8; 9] = [0x61, 0x5F, 0x63, 0x2E, 0x68, 0x65, 0x64, 0x5E, 0x33];
const OBJ_TYPE: u16 = 0x34F;

/// Removes the room's monsters and shots (SETUP).
fn tidy(s: &mut S) {
    let types: Vec<(u16, u8)> =
        (1..=11u16).map(|slot| (OBJ_TYPE + slot, s.state().ram()[(OBJ_TYPE + slot) as usize])).collect();
    for (address, t) in types {
        if t != 0 && !KEEP.contains(&t) {
            zelda::Nes::poke(s.nes_mut(), address, 0);
        }
    }
}

fn act(s: &mut S, text: &str) {
    tidy(s);
    s.act(Action::parse(text).unwrap()).unwrap();
}

fn at(s: &S) -> (MapLocation, i32, i32) {
    let st = s.state();
    (st.full_location(), st.link.position.x, st.link.position.y)
}

/// The route's state `name` (Link on the raft into a dock room), with the corrected route's pieces and the index
/// written, played to the end of the ride down: Link on the dock below its trigger, in play.
fn off_the_raft(name: &str, pieces: u8, index: u8) -> Option<S> {
    let core = open_core()?;
    let mut s = Session::new(Counted(core, 0), Arc::new(GameMap::builtin()));
    s.load(&load_named_state(name), &LoadOptions::default(), &mut ChaCha8Rng::seed_from_u64(0)).unwrap();
    s.set_clips(true);
    for _ in 0..100 {
        let st = s.state();
        let (_, row) = st.room().dock().expect("a dock room");
        if st.mode() == 5 && st.link.status == 0 && st.link.position.y > row && st.link.position.y < row + 8 {
            zelda::Nes::poke(s.nes_mut(), LEVEL_INDEX, index);
            s.apply_ram(&[(Var::Triforce, pieces)]);
            return Some(s);
        }
        act(&mut s, "WAIT 4");
    }
    panic!("no end of the ride down: {:?}", at(&s));
}

/// The engine's dock warp from the raft's end to the landing; the frames it took.
fn dock_warp(s: &mut S, level: u8, spot: (i32, i32), landing: u8) {
    s.set_dock_warp(Some(level));
    // On the dock below the trigger: MOVE S only (any other step boards the raft back up).
    assert!(allowed(s, "MOVE S 8") && !allowed(s, "MOVE W 8") && !allowed(s, "MOVE N 8"), "{:?}", at(s));
    act(s, "MOVE S 8");
    // Walk W on the land to the clip's column; there the mask offers CLIP N only.
    for _ in 0..8 {
        if allowed(s, "CLIP N") {
            break;
        }
        act(s, "MOVE W 8");
    }
    assert!(allowed(s, "CLIP N") && !allowed(s, "MOVE W 8") && !allowed(s, "WAIT 4"), "{:?}", at(s));
    assert_eq!(s.state().link.position.x, spot.0, "the clip's column is the spot nearest the dock");
    act(s, "CLIP N");
    assert_eq!(at(s), (s.state().full_location(), spot.0, spot.1), "in the water on the dock's row");
    assert!(allowed(s, "RECORDER") && !allowed(s, "MOVE E 8") && !allowed(s, "WAIT 4"), "RECORDER only");
    act(s, "RECORDER");
    for _ in 0..100 {
        let st = s.state();
        if st.mode() == 5 && st.link.status == 0 && st.full_location() != ow(0x55) && st.full_location() != ow(0x3f) {
            break;
        }
        act(s, "WAIT 4");
    }
    let (location, x, _) = at(s);
    assert_eq!((location, x), (ow(landing), 0x80), "set down in the dock landing");
}

#[test]
fn the_dock_landings_are_the_stand_in_rooms_one_row_up() {
    let landings: Vec<Option<u8>> = (1..=8).map(dock_landing).collect();
    assert_eq!(landings, [Some(0x26), Some(0x2B), Some(0x63), Some(0x34), None, Some(0x11), Some(0x31), Some(0x5C)]);
    assert_eq!((dock_level(ow(0x34)), dock_level(ow(0x31)), dock_level(ow(0x45))), (Some(4), Some(7), None));
    assert_eq!(route_entry(ow(0x55), ow(0x34)), Some("recorder"));
    assert_eq!(route_entry(ow(0x3f), ow(0x31)), Some("recorder"));
    assert_eq!(route_entry(ow(0x45), ow(0x34)), None, "only out of a dock room");
}

/// The clip's tile rule holds exactly where the dock's row is water above land: 0x55's column W of the dock (`$70`;
/// the river beyond) and its W third behind the river (`$20`-`$40`, which the route never reaches), 0x3f's twelve
/// (`$00`-`$58`), from G' up to T, and nowhere E of a dock or in another room.
#[test]
fn the_dock_clip_is_legal_exactly_beside_the_docks() {
    let Some(s) = off_the_raft("d3first_0_55n_v2_0", 0x1F, 3) else { return };
    let room = s.state().room().clone();
    let legal: Vec<i32> = (0..0x100).step_by(8).filter(|&x| dock_clip_at(&room, x, 0x8D).is_some()).collect();
    assert_eq!(legal, [0x20, 0x28, 0x30, 0x38, 0x40, 0x70, 0x78]);
    assert_eq!(dock_spot(&room), Some((0x78, 0x7D)));
    assert!([0x85, 0x8D, 0x95].iter().all(|&y| dock_clip_at(&room, 0x78, y).is_some()));
    assert!(dock_clip_at(&room, 0x78, 0x9D).is_none() && dock_clip_at(&room, 0x78, 0x7D).is_none());
    let Some(s) = off_the_raft("d3first_0_3fn_0", 0x5F, 3) else { return };
    let room = s.state().room().clone();
    let legal: Vec<i32> = (0..0x100).step_by(8).filter(|&x| dock_clip_at(&room, x, 0x8D).is_some()).collect();
    assert_eq!(legal, (0..=0x58).step_by(8).collect::<Vec<i32>>());
    assert_eq!(dock_spot(&room), Some((0x58, 0x7D)));
}

/// 0x55 after the warp to Level 4 (index 3, pieces 1-5): the clip, then N, a turn W, N's opposite: 0x34 at ($80,$8D).
#[test]
fn the_dock_warp_out_of_0x55_lands_at_the_meat() {
    let Some(mut s) = off_the_raft("d3first_0_55n_v2_0", 0x1F, 3) else { return };
    dock_warp(&mut s, 4, (0x78, 0x7D), 0x34);
    assert_eq!(s.state().link.position.y, 0x8D);
    assert_eq!(s.state().peek(LEVEL_INDEX) & 7, 3, "the index back on Level 4");
}

/// 0x3f after 0x2f's heart (index 3, pieces 1-5 and 7): the clip beside the dock, two plays N: 0x31 at ($80,$AD),
/// the index on Level 7.
#[test]
fn the_dock_warp_out_of_0x3f_lands_in_the_graveyard() {
    let Some(mut s) = off_the_raft("d3first_0_3fn_0", 0x5F, 3) else { return };
    dock_warp(&mut s, 7, (0x58, 0x7D), 0x31);
    assert_eq!(s.state().link.position.y, 0xAD);
    assert_eq!(s.state().peek(LEVEL_INDEX) & 7, 6, "the index on Level 7");
}

/// Without a dock warp (every route but the step that plays one, d2first's dock rooms among them) no N clip is
/// offered on the overworld and the dock's mask is the game's.
#[test]
fn no_dock_clip_without_the_dock_warp() {
    let Some(mut s) = off_the_raft("d3first_0_55n_v2_0", 0x1F, 3) else { return };
    assert!(allowed(&s, "MOVE W 8") || allowed(&s, "MOVE N 8"), "the game's mask on the dock");
    act(&mut s, "MOVE S 8");
    for _ in 0..4 {
        act(&mut s, "MOVE W 8");
        assert!(!allowed(&s, "CLIP N"), "{:?}", at(&s));
    }
}

/// The dock warp gives way to a hit coming (round 36): on the dock, SWORD S only while a monster stands on the land
/// the step off it ends on (a red Octorok's touch there knocks Link back onto the dock and the raft takes him up); on
/// the water spot, WAIT only while a shot is in the air; the clip is never held back.
#[test]
fn the_dock_warp_gives_way_to_a_hit_coming() {
    const OBJ_X: u16 = 0x70;
    const OBJ_Y: u16 = 0x84;
    let Some(mut s) = off_the_raft("d3first_0_55n_v2_0", 0x1F, 3) else { return };
    tidy(&mut s);
    s.set_dock_warp(Some(4));
    let place = |s: &mut S, kind: u8, x: u8, y: u8| {
        for (address, value) in [(OBJ_TYPE + 1, kind), (OBJ_X + 1, x), (OBJ_Y + 1, y)] {
            zelda::Nes::poke(s.nes_mut(), address, value);
        }
        s.apply_ram(&[]);
    };
    place(&mut s, 0x07, 0x7E, 0x8D);
    assert!(allowed(&s, "SWORD S") && !allowed(&s, "MOVE S 8"), "a red Octorok below the dock: {:?}", at(&s));
    place(&mut s, 0, 0, 0);
    assert!(allowed(&s, "MOVE S 8") && !allowed(&s, "SWORD S"), "the land clear");
    act(&mut s, "MOVE S 8");
    for _ in 0..8 {
        if allowed(&s, "CLIP N") {
            break;
        }
        act(&mut s, "MOVE W 8");
    }
    place(&mut s, 0x55, 0xC0, 0x7D);
    assert!(allowed(&s, "CLIP N") && !allowed(&s, "WAIT 4"), "a shot in the air does not hold the clip: {:?}", at(&s));
    place(&mut s, 0, 0, 0);
    act(&mut s, "CLIP N");
    place(&mut s, 0x55, 0xC0, 0x7D);
    assert!(allowed(&s, "WAIT 4") && !allowed(&s, "RECORDER"), "on the water spot: {:?}", at(&s));
    place(&mut s, 0, 0, 0);
    assert!(allowed(&s, "RECORDER"));
}
