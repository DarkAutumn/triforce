//! Round 40 (2.0.1.0; the lead's rule: "an item's target tiles are the tiles Link can take it from, nothing wider"):
//! `zelda::game::take_tiles` against the game's own take tests, one item kind at a time. For each, from the game:
//! on every target tile Link takes the item (from every position of the tile on an axis the tile lies wholly
//! inside the take test, from the tile's grid point on an axis no tile does), and on the tile one beyond the targets
//! on either side of either axis some position does not take it (so it is rightly no target).
#![cfg(feature = "zcore")]

mod common;

use common::{load_named_state, open_core};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;
use zcore::ZCore;
use zelda::game::{LinkBounds, OW_LINK_BOUNDS, Take, UW_LINK_BOUNDS, take_tiles};
use zelda::room_state::room_flags_addr;
use zelda::{Action, ActionKind, GameMap, LoadOptions, Position, Session, Tile};

const OBJ_X: u16 = 0x70;
const OBJ_Y: u16 = 0x84;
const OBJ_STATE: u16 = 0xAC;
const OBJ_TYPE: u16 = 0x34F;
const ITEM_LIFETIME: u16 = 0x3A8;
const ROOM_ITEM: u16 = 0x13;
const ROOM_ITEM_ID: u16 = 0xAB;
const CAVE_ITEM_IDS: u16 = 0x422;
const ITEM_TAKEN: u8 = 0x10;
const DROP_SLOT: u16 = 5;
const DROP_TYPE: u8 = 0x60;
const GRID_OFFSET: u16 = 0x394;

/// Link's X and Y on his grid lines within a tile, and the tiles' origin (as `take_tiles` counts them).
const GRID_IN_TILE: [i32; 2] = [0, 5];
const TILE_ORIGIN: [i32; 2] = [0, 56];

/// The state loaded and played to its first frame where Link can move, the room's monsters gone.
fn start(state: &str) -> Session<ZCore> {
    let nes = open_core().expect("ROM");
    let mut s = Session::new(nes, Arc::new(GameMap::builtin()));
    s.load(&load_named_state(state), &LoadOptions::default(), &mut ChaCha8Rng::seed_from_u64(0)).unwrap();
    for _ in 0..40 {
        if s.state().peek(0x12) == 5 && s.mask().kind[ActionKind::Move.index()] {
            break;
        }
        s.act(Action::parse("WAIT 8").unwrap()).unwrap();
    }
    for slot in 1..=11u16 {
        s.nes_mut().poke(OBJ_TYPE + slot, 0);
    }
    s
}

/// The pixels of one tile on `axis` within `bound`.
fn span(t: i32, axis: usize, bound: (i32, i32)) -> std::ops::RangeInclusive<i32> {
    (TILE_ORIGIN[axis] + 8 * t).max(bound.0)..=(TILE_ORIGIN[axis] + 8 * t + 7).min(bound.1)
}

fn grid(t: i32, axis: usize) -> i32 {
    TILE_ORIGIN[axis] + 8 * t + GRID_IN_TILE[axis]
}

/// Asserts the rule for one item: `taken` tells, two frames after Link is put at a position, whether he took it.
/// `fallback[axis]`: the axis on which no tile lies wholly inside the take test (its grid points stand in).
fn check(s: &mut Session<ZCore>, what: &str, at: Position, take: Take, bounds: LinkBounds, fallback: [bool; 2], taken: impl Fn(&mut Session<ZCore>) -> bool) {
    let snap = s.snapshot();
    let takes = |s: &mut Session<ZCore>, x: i32, y: i32| {
        s.restore(&snap);
        // Link off the grid carries his distance into the grid step (ObjGridOffset $394): with it the game keeps him
        // where he is put (at 0 its tile collision test moves him back onto a grid line).
        let offset = if x % 8 != 0 { x % 8 } else { (y - GRID_IN_TILE[1] - TILE_ORIGIN[1]).rem_euclid(8) };
        let nes = s.nes_mut();
        nes.poke(OBJ_X, x as u8);
        nes.poke(OBJ_Y, y as u8);
        nes.poke(GRID_OFFSET, offset as u8);
        // Two frames: a busy room updates its objects (a drop among them) every other frame.
        s.press(0, 2);
        taken(s)
    };
    let tiles = take_tiles(at, take, bounds);
    assert!(!tiles.is_empty(), "{what}: no tile");
    let cols: Vec<i32> = tiles.iter().map(|t| t.0).collect();
    let rows: Vec<i32> = tiles.iter().map(|t| t.1).collect();
    // Every target tile takes: the whole of its span on a full axis, its grid point on a fallback axis.
    for &(c, r) in &tiles {
        let xs: Vec<i32> = if fallback[0] { vec![grid(c, 0)] } else { span(c, 0, bounds[0]).collect() };
        let ys: Vec<i32> = if fallback[1] { vec![grid(r, 1)] } else { span(r, 1, bounds[1]).collect() };
        for &x in &xs {
            for &y in &ys {
                assert!(takes(s, x, y), "{what} at ({:#04x}, {:#04x}): tile {:?} of {tiles:?} does not take it", at.x, at.y, (c, r));
            }
        }
    }
    // One tile beyond on either side of either axis: some position of it within the bounds does not take.
    let beyond: Vec<(Tile, usize)> = [
        ((cols.iter().min().unwrap() - 1, rows[0]), 0),
        ((cols.iter().max().unwrap() + 1, rows[0]), 0),
        ((cols[0], rows.iter().min().unwrap() - 1), 1),
        ((cols[0], rows.iter().max().unwrap() + 1), 1),
    ]
    .into_iter()
    .filter(|&((c, r), _)| !span(c, 0, bounds[0]).is_empty() && !span(r, 1, bounds[1]).is_empty())
    .collect();
    for ((c, r), axis) in beyond {
        let (xs, ys): (Vec<i32>, Vec<i32>) = if axis == 0 {
            (span(c, 0, bounds[0]).collect(), vec![grid(r, 1).clamp(bounds[1].0, bounds[1].1)])
        } else {
            (vec![grid(c, 0).clamp(bounds[0].0, bounds[0].1)], span(r, 1, bounds[1]).collect())
        };
        let refused = xs.iter().any(|&x| ys.iter().any(|&y| !takes(s, x, y)));
        assert!(refused, "{what} at ({:#04x}, {:#04x}): tile {:?} beyond {tiles:?} takes it from all of it", at.x, at.y, (c, r));
    }
    s.restore(&snap);
}

/// The room item (slot $13) made `id` at `at`, the room's "item taken" flag ($10) clear: what the game makes of a
/// dungeon room's key, heart container, Triforce piece or the Triforce of Power, or of 0x5f's heart container
/// (asm:Room_SetUpItemAndPushBlock; asm:Item_TryTakeRoomItem takes it).
fn room_item(s: &mut Session<ZCore>, id: u8, at: Position) {
    let state = s.state();
    let flags = room_flags_addr(state.ram(), state.location()).expect("the level's room flags") as u16;
    let cleared = state.peek(flags) & !ITEM_TAKEN;
    let nes = s.nes_mut();
    nes.poke(flags, cleared);
    nes.poke(ROOM_ITEM_ID, id);
    nes.poke(OBJ_STATE + ROOM_ITEM, 0);
    nes.poke(OBJ_X + ROOM_ITEM, at.x as u8);
    nes.poke(OBJ_Y + ROOM_ITEM, at.y as u8);
    nes.poke(ITEM_LIFETIME + ROOM_ITEM, 0);
}

/// Taken: the pickup set the room's "item taken" flag (asm:Item_TryTake, asm:RoomFlags_SetItemTaken).
fn room_item_taken(s: &mut Session<ZCore>) -> bool {
    let state = s.state();
    let flags = room_flags_addr(state.ram(), state.location()).expect("the level's room flags") as u16;
    state.peek(flags) & ITEM_TAKEN != 0
}

#[test]
fn a_room_item_is_taken_from_its_target_tiles_and_no_wider() {
    let Some(_) = open_core() else { return };
    let mut s = start("1_33n_0");
    // (kind, id, position): the level's item positions (zelda-asm data/levelinfo.md), a Triforce piece 8 left of
    // its position, one low in the room (Y $C0: its lower rows are below Link's $BD), the Triforce of Power where
    // Ganon's ashes left it in the d9 walk ($77, $C5).
    for (what, id, at, fallback) in [
        ("key", 0x19, Position::new(0x80, 0x90), [false, false]),
        ("heart container", 0x1A, Position::new(0xD0, 0x60), [false, false]),
        ("Triforce piece", 0x1B, Position::new(0x78, 0x90), [false, false]),
        ("heart container low", 0x1A, Position::new(0xA0, 0xC0), [false, false]),
        ("Triforce of Power", 0x0E, Position::new(0x77, 0xC5), [false, true]),
    ] {
        room_item(&mut s, id, at);
        check(&mut s, what, at, Take::Box, UW_LINK_BOUNDS, fallback, room_item_taken);
    }
}

#[test]
fn a_cellar_item_is_taken_from_its_target_tiles_and_no_wider() {
    let Some(_) = open_core() else { return };
    // The route's cellar items where the game lays them (Dungeon 1's bow, 5's recorder, 7's cellar): the room item
    // of the cellar (slot $13), taken as in a room.
    for state in ["1_7fcave_0", "5_04cave_0", "7_4acave_0"] {
        let nes = open_core().expect("ROM");
        let mut s = Session::new(nes, Arc::new(GameMap::builtin()));
        s.load(&load_named_state(state), &LoadOptions::default(), &mut ChaCha8Rng::seed_from_u64(0)).unwrap();
        for _ in 0..120 {
            if s.state().treasure().is_some() {
                break;
            }
            s.press(0, 1);
        }
        let item = s.state().treasure().unwrap_or_else(|| panic!("{state}: the cellar's item"));
        check(&mut s, state, item.position, Take::Box, UW_LINK_BOUNDS, [false, false], room_item_taken);
    }
}

#[test]
fn a_drop_is_taken_from_its_target_tiles_and_no_wider() {
    let Some(_) = open_core() else { return };
    let mut s = start("1_33n_0");
    // A dropped rupee (type $60, its id in ObjState; asm:Drop_UpdateItem takes it through asm:Item_TryTake), at
    // positions off the tile grid on both axes.
    for at in [Position::new(0x5B, 0xA3), Position::new(0x9F, 0x7A), Position::new(0x44, 0x8D)] {
        let nes = s.nes_mut();
        nes.poke(OBJ_TYPE + DROP_SLOT, DROP_TYPE);
        nes.poke(OBJ_STATE + DROP_SLOT, 0x18);
        nes.poke(OBJ_X + DROP_SLOT, at.x as u8);
        nes.poke(OBJ_Y + DROP_SLOT, at.y as u8);
        nes.poke(ITEM_LIFETIME + DROP_SLOT, 0x80);
        let taken = |s: &mut Session<ZCore>| {
            let st = s.state();
            st.peek(OBJ_TYPE + DROP_SLOT) != DROP_TYPE || st.peek(OBJ_STATE + DROP_SLOT) == 0xFF
        };
        check(&mut s, "drop", at, Take::Box, UW_LINK_BOUNDS, [false, false], taken);
    }
}

#[test]
fn the_overworld_heart_container_is_taken_from_its_target_tiles_and_no_wider() {
    let Some(_) = open_core() else { return };
    let mut s = start("0_5fe_0");
    let at = Position::new(0xC0, 0x90);
    room_item(&mut s, 0x1A, at);
    check(&mut s, "0x5f's heart container", at, Take::Box, OW_LINK_BOUNDS, [false, false], room_item_taken);
}

#[test]
fn a_cave_ware_is_taken_from_its_target_tiles_and_no_wider() {
    let Some(_) = open_core() else { return };
    let nes = open_core().expect("ROM");
    let mut s = Session::new(nes, Arc::new(GameMap::builtin()));
    s.load(&load_named_state("0_2ccave_0"), &LoadOptions::default(), &mut ChaCha8Rng::seed_from_u64(0)).unwrap();
    // The person's text first (state 2: Link may take a ware; asm:Cave_State2TalkOrShopOrDoorCharge).
    for _ in 0..600 {
        if s.state().peek(OBJ_STATE + 1) == 2 {
            break;
        }
        s.press(0, 1);
    }
    assert_eq!(s.state().peek(OBJ_STATE + 1), 2, "the wares are offered");
    // 0x2c's right ware (X $98, the heart container): taken, its id goes ($FF).
    let at = Position::new(0x98, 0x98);
    let taken = |s: &mut Session<ZCore>| s.state().peek(CAVE_ITEM_IDS + 2) == 0xFF;
    check(&mut s, "0x2c's right ware", at, Take::Ware, UW_LINK_BOUNDS, [true, true], taken);
}
