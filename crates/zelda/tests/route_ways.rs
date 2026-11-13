//! The round 22 route steps in the real game (`triforce/routes.yaml`, `docs/routes/overworld.md`, `d3.md`, `d4.md`):
//! 0x67's secret cave (the rock wall, the 30 rupees at X `$78`, both exits), both ways of the stretch `d3-approach`
//! into Dungeon 3, Dungeon 3's 0x5b E -> 0x5c -> 0x4c E -> 0x4d, Dungeon 4's 0x11 N -> 0x01 (the key) -> 0x02 ->
//! 0x12 (the block, the shutter) -> 0x13.  Each starts from the route's own start state (chain 0) and walks by
//! buttons, a frame at a time onto grid points, with the actions `BOMB` and `CLIP` where the route needs them.
//!
//! SETUP (RAM writes): every room's monsters, traps and the 0x11 rupee stash are removed on entry (object slots 1-11,
//! but the rock wall's tile object and a dungeon room's push block), and Link is given the bombs a wall needs where
//! the state holds fewer, so the frame counts are the walk's own.  Each test prints them.
#![cfg(feature = "zcore")]

mod common;

use common::{Counted, load_named_state, open_core};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;
use zelda::executor::direction_buttons;
use zelda::{Action, Change, Direction, GameMap, LoadOptions, MapLocation, Session, Var};

type S = Session<Counted>;

const OBJ_TYPE: u16 = 0x34F;
/// The overworld rock wall's tile object (asm:RockWall_Update) and a dungeon room's push block (slot `$0B`,
/// asm:PushBlock_FindAndCreate): kept by `clear_room`.
const KEEP: [u8; 2] = [0x63, 0x68];

fn ow(room: u8) -> MapLocation {
    MapLocation::new(0, room, false)
}

fn uw(level: u8, room: u8) -> MapLocation {
    MapLocation::new(level, room, false)
}

/// The start state `name` loaded (no randomization) to its first playable frame, the clips switch `clips`.
fn start(name: &str, clips: bool) -> Option<S> {
    let core = open_core()?;
    let mut s = Session::new(Counted(core, 0), Arc::new(GameMap::builtin()));
    s.set_clips(clips);
    s.load(&load_named_state(name), &LoadOptions::default(), &mut ChaCha8Rng::seed_from_u64(0)).unwrap();
    assert_eq!(s.state().mode(), 5, "{name}: {}", describe(&s));
    Some(s)
}

fn act(s: &mut S, text: &str) -> Change {
    s.act(Action::parse(text).unwrap()).unwrap()
}

/// (x, y, grid offset) of Link.
fn link(s: &S) -> (i32, i32, i32) {
    let st = s.state();
    (st.link.position.x, st.link.position.y, st.var(Var::LinkGridOffset))
}

fn describe(s: &S) -> String {
    let st = s.state();
    format!(
        "{:?} mode {} at ({:#04x}, {:#04x}) offset {} frame {}",
        st.full_location(),
        st.mode(),
        st.link.position.x,
        st.link.position.y,
        st.var(Var::LinkGridOffset),
        s.frame()
    )
}

/// Removes the room's monsters, traps and items (SETUP), keeping `KEEP`, and lets the game notice.
fn clear_room(s: &mut S) {
    for slot in 1..=11u16 {
        if !KEEP.contains(&s.state().ram()[(OBJ_TYPE + slot) as usize]) {
            zelda::Nes::poke(s.nes_mut(), OBJ_TYPE + slot, 0);
        }
    }
    s.apply_ram(&[]);
}

/// Holds `buttons` a frame at a time until `until` holds (at most 900 frames).
fn press_until(s: &mut S, buttons: u8, until: impl Fn(&S) -> bool) {
    for _ in 0..900 {
        if until(s) {
            return;
        }
        s.press(buttons, 1);
    }
    panic!("never got there: {}", describe(s));
}

/// Walks `d` until Link's coordinate on that axis is `to`, on a grid point.
fn walk(s: &mut S, d: Direction, to: i32) {
    let horizontal = matches!(d, Direction::W | Direction::E);
    press_until(s, direction_buttons(d), |s| {
        let (x, y, offset) = link(s);
        (if horizontal { x } else { y }) == to && offset == 0
    });
}

/// Walks `d` until Link plays in `room` (the scroll or the stairs done), then clears it.
fn walk_into(s: &mut S, d: Direction, room: MapLocation) {
    press_until(s, direction_buttons(d), |s| s.state().full_location() == room && s.state().mode() == 5);
    clear_room(s);
}

/// Places a bomb `d` (Link given one if he has none, SETUP) and waits until `open` holds.
fn bomb(s: &mut S, d: Direction, open: impl Fn(&S) -> bool) {
    if s.state().bombs() == 0 {
        s.apply_ram(&[(Var::Bombs, 1)]);
    }
    assert!(!open(s), "already open: {}", describe(s));
    act(s, &format!("BOMB {}", d.name()));
    for _ in 0..16 {
        if open(s) {
            return;
        }
        act(s, "WAIT 8");
    }
    panic!("the bomb opened nothing: {}", describe(s));
}

/// Frames since `from`, printed under `label`.
fn lap(s: &S, from: u64, label: &str) -> u64 {
    let frames = s.frame() - from;
    eprintln!("{label}: {frames} frames");
    frames
}

/// 0x67 entered from 0x68 (E): N to row Y `$7D`, W to X `$70`, N to Y `$5D` facing the rock wall, `BOMB N`, into the
/// cave; there E to X `$78` and N onto the middle ware: the 30 rupees, once.  Walking straight up at X `$70` takes
/// nothing.  Out of the cave Link stands in the mouth (`$70`, `$4D`).  The stretch parts there: S into 0x77 (the clip
/// way), or S to row `$7D` and W into 0x66 (the plain way).
#[test]
fn the_0x67_secret_cave_pays_30_rupees_at_x_78_and_both_ways_leave() {
    let Some(mut s) = start("0_67e_0", true) else { return };
    assert_eq!(s.state().full_location(), ow(0x67));
    clear_room(&mut s);
    let t0 = s.frame();
    assert!(s.state().ow_bomb_secret_closed());
    walk(&mut s, Direction::N, 0x7D);
    walk(&mut s, Direction::W, 0x70);
    walk(&mut s, Direction::N, 0x5D);
    bomb(&mut s, Direction::N, |s| !s.state().ow_bomb_secret_closed());
    let opened = lap(&s, t0, "0x67: E to the wall open");
    press_until(&mut s, direction_buttons(Direction::N), |s| s.state().in_cave() && s.state().mode() == 0x0B);
    clear_room(&mut s);
    let cave = s.snapshot();
    let rupees = s.state().rupees();
    walk(&mut s, Direction::E, 0x78);
    press_until(&mut s, direction_buttons(Direction::N), |s| s.state().rupees() > rupees);
    for _ in 0..8 {
        act(&mut s, "WAIT 8");
    }
    assert_eq!(s.state().rupees(), rupees + 30, "{}", describe(&s));
    walk_into(&mut s, Direction::S, ow(0x67));
    assert_eq!((link(&s).0, link(&s).1), (0x70, 0x4D), "{}", describe(&s));
    assert!(!s.state().ow_bomb_secret_closed(), "the wall stays open");
    let out = lap(&s, t0, "0x67: E to out of the cave with the rupees");
    assert!(opened < 400 && out < 1050, "{opened} / {out} frames");

    let mouth = s.snapshot();
    let t1 = s.frame();
    walk_into(&mut s, Direction::S, ow(0x77));
    let south = lap(&s, t1, "0x67 cave -> S into 0x77 (clip way)");
    s.restore(&mouth);
    let t1 = s.frame();
    walk(&mut s, Direction::S, 0x7D);
    walk_into(&mut s, Direction::W, ow(0x66));
    let west = lap(&s, t1, "0x67 cave -> W into 0x66 (plain way)");
    assert!(south < 300 && west < 400, "{south} / {west} frames");

    // Straight up from the entrance (X $70) Link walks past the ware to Y $8D beside it and takes nothing.
    s.restore(&cave);
    for _ in 0..300 {
        s.press(direction_buttons(Direction::N), 1);
    }
    assert_eq!((link(&s).0, s.state().rupees()), (0x70, rupees), "{}", describe(&s));
    assert_eq!(link(&s).1, 0x8D, "{}", describe(&s));
}

/// The clip way (switch on) from 0x67 out of the cave: S into 0x77, S to row `$8D`, W into 0x76, W to X `$90`, N to
/// row `$7D` (0x76's straight lanes end on 0x75's softlock rows), W into 0x75, `CLIP E` (the wrap exit) into 0x74, W
/// to X `$D0`, S to `$8D`, W to `$80`, N into the mouth: Dungeon 3's 0x7c entered from S.
#[test]
fn the_clip_way_from_0x67_enters_dungeon_3_by_the_wrap_exit() {
    let Some(mut s) = start("0_67c_0", true) else { return };
    assert_eq!((s.state().full_location(), link(&s).0, link(&s).1), (ow(0x67), 0x70, 0x4D), "{}", describe(&s));
    clear_room(&mut s);
    let t0 = s.frame();
    walk_into(&mut s, Direction::S, ow(0x77));
    walk(&mut s, Direction::S, 0x8D);
    walk_into(&mut s, Direction::W, ow(0x76));
    walk(&mut s, Direction::W, 0x90);
    walk(&mut s, Direction::N, 0x7D);
    walk_into(&mut s, Direction::W, ow(0x75));
    let at_75 = lap(&s, t0, "clip way: 0x75");
    let clip = act(&mut s, "CLIP E");
    assert!(clip.changed_location(), "{}", describe(&s));
    walk_into(&mut s, Direction::W, ow(0x74));
    assert_eq!((link(&s).0, link(&s).1), (0xF0, 0x7D), "entered by the wrap exit: {}", describe(&s));
    walk(&mut s, Direction::W, 0xD0);
    walk(&mut s, Direction::S, 0x8D);
    walk(&mut s, Direction::W, 0x80);
    walk_into(&mut s, Direction::N, uw(3, 0x7C));
    let frames = lap(&s, t0, "clip way: 0x67 cave mouth to Dungeon 3");
    assert!(at_75 < 800 && frames < 1300, "{at_75} / {frames} frames");
}

/// The plain way (switch off) from 0x67 out of the cave: S to row `$7D`, W into 0x66; W to X `$B0`, N to `$5D`
/// (below its shop's mouth), W into 0x65; W to `$90`, S to its ford (`$8D`), W into 0x64, W along `$8D` past the
/// potion shop into 0x63; W to `$D0`, S into 0x73; S to `$8D`, E into 0x74 from W; E to `$80`, N into the mouth.
#[test]
fn the_plain_way_from_0x67_enters_dungeon_3_along_row_6() {
    let Some(mut s) = start("0_67c_0", false) else { return };
    clear_room(&mut s);
    let t0 = s.frame();
    walk(&mut s, Direction::S, 0x7D);
    walk_into(&mut s, Direction::W, ow(0x66));
    walk(&mut s, Direction::W, 0xB0);
    walk(&mut s, Direction::N, 0x5D);
    walk_into(&mut s, Direction::W, ow(0x65));
    walk(&mut s, Direction::W, 0x90);
    walk(&mut s, Direction::S, 0x8D);
    walk_into(&mut s, Direction::W, ow(0x64));
    walk_into(&mut s, Direction::W, ow(0x63));
    walk(&mut s, Direction::W, 0xD0);
    walk_into(&mut s, Direction::S, ow(0x73));
    walk(&mut s, Direction::S, 0x8D);
    walk_into(&mut s, Direction::E, ow(0x74));
    let at_74 = lap(&s, t0, "plain way: 0x74 from W");
    walk(&mut s, Direction::E, 0x80);
    walk_into(&mut s, Direction::N, uw(3, 0x7C));
    let frames = lap(&s, t0, "plain way: 0x67 cave mouth to Dungeon 3");
    assert!(at_74 < 1700 && frames < 1900, "{at_74} / {frames} frames");
}

/// Dungeon 3 from 0x5b entered from W (its last visit): E to X `$C0` on the doorway row, `BOMB E`, E into 0x5c; N to
/// row `$5D`, E to X `$78`, N through the open doorway into 0x4c (from S); N to `$8D`, E to `$C0`, `BOMB E`, E into
/// Manhandla's room 0x4d.  Two walls, two bombs (0x5b's own four bombs, its room item once the Darknuts are gone,
/// lie on the way).
#[test]
fn dungeon_3_bombs_0x5b_east_goes_north_through_0x5c_and_bombs_0x4c_east() {
    let Some(mut s) = start("3_5bw_0", true) else { return };
    assert_eq!(s.state().full_location(), uw(3, 0x5B));
    clear_room(&mut s);
    let t0 = s.frame();
    let open = |d: Direction| move |s: &S| !s.state().bomb_walls().iter().any(|w| w.wall_direction == d);
    walk(&mut s, Direction::E, 0xC0);
    bomb(&mut s, Direction::E, open(Direction::E));
    walk_into(&mut s, Direction::E, uw(3, 0x5C));
    let at_5c = lap(&s, t0, "D3: 0x5c");
    walk(&mut s, Direction::N, 0x5D);
    walk(&mut s, Direction::E, 0x78);
    walk_into(&mut s, Direction::N, uw(3, 0x4C));
    let at_4c = lap(&s, t0, "D3: 0x4c from S");
    assert_eq!(s.state().bomb_walls().iter().map(|w| w.wall_direction).collect::<Vec<_>>(), [Direction::E]);
    walk(&mut s, Direction::N, 0x8D);
    walk(&mut s, Direction::E, 0xC0);
    bomb(&mut s, Direction::E, open(Direction::E));
    walk_into(&mut s, Direction::E, uw(3, 0x4D));
    let frames = lap(&s, t0, "D3: 0x5b W to Manhandla");
    assert!(at_5c < 600 && at_4c < 900 && frames < 1400, "{at_5c} / {at_4c} / {frames} frames");
}

/// Dungeon 4 from 0x11 entered from S: N to row `$5D` (taking the rupee stash's line), `BOMB N`, N into the dark
/// Keese room 0x01; N over the water square by the ladder to `$8D`, E to X `$80`, N to `$7D`: the key (+1); S to
/// `$8D`, E over the second square into the dark trap room 0x02; E to `$20`, S to `$9D`, E to `$30`, S to `$AD`, E to
/// `$78`, S into the Vire room 0x12 (from N); S to `$6D`, W to `$60`, S against the block: it slides and the E
/// shutter opens; N to `$7D`, E to `$D0`, S to `$8D`, E into Gleeok's room 0x13.
#[test]
fn dungeon_4_takes_0x01s_key_crosses_0x02_and_opens_0x12s_shutter() {
    let Some(mut s) = start("4_11s_0", true) else { return };
    assert_eq!(s.state().full_location(), uw(4, 0x11));
    assert!(s.state().ladder(), "the ladder is in hand");
    clear_room(&mut s);
    let t0 = s.frame();
    walk(&mut s, Direction::N, 0x5D);
    bomb(&mut s, Direction::N, |s| !s.state().bomb_walls().iter().any(|w| w.wall_direction == Direction::N));
    walk_into(&mut s, Direction::N, uw(4, 0x01));
    let at_01 = lap(&s, t0, "D4: 0x01");
    let keys = s.state().keys();
    walk(&mut s, Direction::N, 0x8D);
    walk(&mut s, Direction::E, 0x80);
    walk(&mut s, Direction::N, 0x7D);
    assert_eq!(s.state().keys(), keys + 1, "0x01's key: {}", describe(&s));
    walk(&mut s, Direction::S, 0x8D);
    walk_into(&mut s, Direction::E, uw(4, 0x02));
    let at_02 = lap(&s, t0, "D4: 0x02");
    walk(&mut s, Direction::E, 0x20);
    walk(&mut s, Direction::S, 0x9D);
    walk(&mut s, Direction::E, 0x30);
    walk(&mut s, Direction::S, 0xAD);
    walk(&mut s, Direction::E, 0x78);
    walk_into(&mut s, Direction::S, uw(4, 0x12));
    let at_12 = lap(&s, t0, "D4: 0x12 from N");
    assert!(s.state().is_door_barred(Direction::E) && s.state().push_block_pending());
    walk(&mut s, Direction::S, 0x6D);
    walk(&mut s, Direction::W, 0x60);
    press_until(&mut s, direction_buttons(Direction::S), |s| !s.state().is_door_barred(Direction::E));
    let pushed = lap(&s, t0, "D4: 0x12's block pushed, the shutter open");
    walk(&mut s, Direction::N, 0x7D);
    walk(&mut s, Direction::E, 0xD0);
    walk(&mut s, Direction::S, 0x8D);
    walk_into(&mut s, Direction::E, uw(4, 0x13));
    let frames = lap(&s, t0, "D4: 0x11 S to Gleeok");
    assert!(
        at_01 < 600 && at_02 < 900 && at_12 < 1300 && pushed < 1450 && frames < 1700,
        "{at_01} / {at_02} / {at_12} / {pushed} / {frames} frames"
    );
}
