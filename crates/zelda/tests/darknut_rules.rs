//! Round 40 (Darknut40's switched-off proposals; the owner's Darknut lesson, docs/owner/darknuts.md):
//! `zelda::darknut_rules::chasing` on Link's place against a Darknut's line, and `bomb_catch` against the game's own
//! blast (the Olive Garden's Blue Darknuts held by the clock, a bomb laid, their HP read after it) and in the BOMB
//! mask (`Session::set_bomb_catch`).
#![cfg(feature = "zcore")]

mod common;

use common::{load_named_state, open_core};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;
use zcore::ZCore;
use zelda::darknut_rules::{bomb_catch, chasing};
use zelda::{Action, ActionKind, Direction, GameMap, LoadOptions, Session, Var};

const OBJ_X: u16 = 0x70;
const OBJ_Y: u16 = 0x84;
const OBJ_DIR: u16 = 0x98;
const OBJ_TYPE: u16 = 0x34F;
const OBJ_HP: u16 = 0x485;
const INV_CLOCK: u16 = 0x66C;
const OBJ_INPUT_DIR: u16 = 0x3F8;
const OBJ_Q_SPEED: u16 = 0x3BC;
const OBJ_POS_FRAC: u16 = 0x3A8;
const OBJ_GRID_OFFSET: u16 = 0x394;
const BLUE_DARKNUT: u8 = 0x0C;

/// The Olive Garden from E, played to its first frame where Link can move and every Darknut is out of its cloud.
fn garden() -> Session<ZCore> {
    let nes = open_core().expect("ROM");
    let mut s = Session::new(nes, Arc::new(GameMap::builtin()));
    s.load(&load_named_state("d3first_5_05e_0"), &LoadOptions::default(), &mut ChaCha8Rng::seed_from_u64(0)).unwrap();
    for _ in 0..40 {
        if s.state().peek(0x12) == 5 && s.mask().kind[ActionKind::Move.index()] && s.state().active_enemies().count() == 6
        {
            break;
        }
        s.act(Action::parse("WAIT 8").unwrap()).unwrap();
    }
    s
}

/// Darknuts at (x, y) walking `dir` in slots 1.. (q-speed $20, on a grid point: the next one 16 pixels on; the rest
/// of the room's monsters gone), Link at (lx, ly) facing E with 4 bombs and the clock (test setup: nothing moves,
/// nothing touches Link).
fn place(s: &mut Session<ZCore>, nuts: &[(u8, u8, u8)], link: (u8, u8), clock: bool) {
    for slot in 1..=11u16 {
        let nut = nuts.get(usize::from(slot) - 1);
        let nes = s.nes_mut();
        nes.poke(OBJ_TYPE + slot, if nut.is_some() { BLUE_DARKNUT } else { 0 });
        if let Some(&(x, y, dir)) = nut {
            nes.poke(OBJ_X + slot, x);
            nes.poke(OBJ_Y + slot, y);
            nes.poke(OBJ_DIR + slot, dir);
            nes.poke(OBJ_INPUT_DIR + slot, dir);
            nes.poke(OBJ_Q_SPEED + slot, 0x20);
            nes.poke(OBJ_POS_FRAC + slot, 0);
            nes.poke(OBJ_GRID_OFFSET + slot, 0);
        }
    }
    s.nes_mut().poke(INV_CLOCK, u8::from(clock));
    s.apply_ram(&[(Var::LinkX, link.0), (Var::LinkY, link.1), (Var::LinkDirection, 1), (Var::Bombs, 4)]);
}

#[test]
fn chasing_is_a_darknuts_reversal_onto_link_at_its_next_grid_point() {
    let mut s = garden();
    // A Darknut at (128, 125) walking E (direction 1), its next grid point (144, 125); Link at each place, the
    // expected answer (`zelda::chase::TurnAt::reverse_link`).
    let cases: [((u8, u8), bool, &str); 8] = [
        ((112, 125), true, "on its row, 16 behind"),
        ((112, 133), true, "8 below its row"),
        ((112, 134), false, "9 below its row: outside the homing turn's 8"),
        ((112, 117), true, "8 above its row"),
        ((32, 125), true, "far behind on its row: the turn does not depend on the distance"),
        ((136, 125), false, "8 short of its grid point: in its column there, the turn is N or S"),
        ((128, 125 + 16), false, "level with it, below"),
        ((160, 125), false, "ahead of it"),
    ];
    for ((x, y), want, what) in cases {
        place(&mut s, &[(128, 125, 1)], (x, y), false);
        assert_eq!(chasing(s.state()), want, "Link at ({x}, {y}): {what}");
    }
    // The clock: it decides nothing.
    place(&mut s, &[(128, 125, 1)], (112, 125), true);
    assert!(!chasing(s.state()), "the clock");
    // A Darknut walking N (direction 8), its next grid point (128, 109): behind it is below.
    place(&mut s, &[(128, 125, 8)], (132, 141), false);
    assert!(chasing(s.state()), "4 beside its column, 16 below");
    place(&mut s, &[(128, 125, 8)], (128, 93), false);
    assert!(!chasing(s.state()), "ahead of it, above");
}

#[test]
fn bomb_catch_counts_what_the_blast_hurts_and_the_mask_follows_it() {
    let mut s = garden();
    // Link at (160, 125) facing E: BOMB E lands at (176, 125). Held by the clock: two Darknuts in the blast's reach
    // not facing against E (N, S), one in reach facing W (its shield to the bomb), one far away.
    let nuts = [(176, 125, 8), (184, 133, 4), (170, 120, 2), (208, 173, 8)];
    place(&mut s, &nuts, (160, 125), true);
    assert_eq!(bomb_catch(s.state(), Direction::E), 2);
    assert_eq!(bomb_catch(s.state(), Direction::W), 0);

    // The mask: BOMB offered facing exactly where two or more are caught.
    s.set_bomb_catch(2);
    let offered: Vec<Direction> = s.mask().directions(ActionKind::Bomb);
    assert!(offered.contains(&Direction::E) && !offered.contains(&Direction::W), "{offered:?}");
    for d in [Direction::N, Direction::S, Direction::W, Direction::E] {
        assert_eq!(offered.contains(&d), bomb_catch(s.state(), d) >= 2, "{d:?}");
    }
    s.set_bomb_catch(0);

    // The game: the bomb hurts exactly the two counted (a Blue Darknut's 8 hits, a bomb's 4).
    s.act(Action::new(ActionKind::Bomb, Direction::E, 0).unwrap()).unwrap();
    for _ in 0..12 {
        s.act(Action::parse("WAIT 8").unwrap()).unwrap();
    }
    let hp: Vec<u8> = (1..=4u16).map(|slot| s.state().peek(OBJ_HP + slot) >> 4).collect();
    assert_eq!(hp, vec![4, 4, 8, 8]);
}
