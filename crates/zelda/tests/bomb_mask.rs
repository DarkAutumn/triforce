//! Round 22 (4:0x21's bomb waste): BOMB kept to the bombs that open the route's wall (`Session::set_bomb_targets`,
//! `zelda::bombs::BombTarget`, `zelda::bombs::landing`).  Played in the game for every room the rule touches on the
//! route (the `requires_bombs` rooms of the `required` families): around each wall, from every grid point and in
//! every direction the game lets Link place a bomb, a BOMB the mask offers opens the wall with that one bomb, and one
//! it doesn't offer leaves the wall shut.
mod common;

use common::{load_named_state, open_core};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;
use zcore::ZCore;
use zelda::bombs::BombTarget;
use zelda::game::GameView;
use zelda::{Action, ActionKind, Direction, GameMap, LoadOptions, Session, Var};

/// Object type of the overworld rock wall tile object (asm:RockWall_Update).
const ROCK_WALL: u8 = 0x63;

const CARDINALS: [Direction; 4] = [Direction::N, Direction::S, Direction::W, Direction::E];

fn act(s: &mut Session<ZCore>, text: &str) {
    s.act(Action::parse(text).unwrap()).unwrap();
}

/// The target is still shut.
fn shut(state: &GameView, target: BombTarget) -> bool {
    match target {
        BombTarget::Wall(d) => state.bomb_walls().iter().any(|w| w.wall_direction == d),
        BombTarget::Rock(_) => state.ow_bomb_secret_closed(),
    }
}

/// Link placed at (x, y) facing N with bombs, the room's monsters and traps gone, a rock wall's tile object kept
/// (test setup).
fn place(s: &mut Session<ZCore>, x: u8, y: u8) {
    for slot in 1..=11u16 {
        if s.state().peek(0x34F + slot) != ROCK_WALL {
            zelda::Nes::poke(s.nes_mut(), 0x34F + slot, 0);
        }
    }
    s.apply_ram(&[
        (Var::LinkX, x),
        (Var::LinkY, y),
        (Var::LinkDirection, 8),
        (Var::Bombs, 4),
        (Var::HeartsAndContainers, 0x55),
        (Var::PartialHearts, 0xFF),
    ]);
}

/// Sweeps the grid points within 48 px of `centre`: (offered, opened) pairs, all equal.  Returns the number of
/// BOMBs offered and the number refused that the game would have placed.
fn sweep(state: &str, target: BombTarget, centre: (i32, i32)) -> (usize, usize) {
    let nes = open_core().expect("ROM");
    let mut s = Session::new(nes, Arc::new(GameMap::builtin()));
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    s.load(&load_named_state(state), &LoadOptions::default(), &mut rng).unwrap();
    for _ in 0..40 {
        if s.state().peek(0x12) == 5 && s.mask().kind[ActionKind::Move.index()] {
            break;
        }
        act(&mut s, "WAIT 8");
    }
    assert!(shut(s.state(), target), "{state}: {target:?} is shut at the start");
    let start = s.snapshot();
    let (mut offered, mut refused) = (0, 0);
    for dy in (-48..=48).step_by(8) {
        for dx in (-48..=48).step_by(8) {
            let (x, y) = (centre.0 + dx, centre.1 + dy);
            if !(0x10..=0xE0).contains(&x) || !(0x45..=0xD5).contains(&y) {
                continue;
            }
            for d in CARDINALS {
                s.restore(&start);
                s.set_bomb_targets(None);
                place(&mut s, x as u8, y as u8);
                let bomb = Action::new(ActionKind::Bomb, d, 0).unwrap();
                if !s.mask().is_allowed(&bomb) {
                    continue;
                }
                s.set_bomb_targets(Some(&[target]));
                let allowed = s.mask().is_allowed(&bomb);
                s.act(bomb).unwrap();
                for _ in 0..8 {
                    act(&mut s, "WAIT 16");
                }
                let opened = !shut(s.state(), target);
                assert_eq!(
                    allowed, opened,
                    "{state}: Link at (${x:02X}, ${y:02X}), BOMB {d:?}: offered {allowed}, the bomb opened {target:?}: {opened} (room flags ${:02X}, cave tile {:?}, link {:?})",
                    s.state().peek(0x67F + u16::from(s.state().location())),
                    s.state().room().cave_tile,
                    s.state().link.position
                );
                if allowed { offered += 1 } else { refused += 1 }
            }
        }
    }
    (offered, refused)
}

/// One room on the route where a `required` family bombs: it offers some BOMBs, refuses some the game would place,
/// and the two agree with the game (`sweep`).
fn room(state: &str, target: BombTarget, centre: (i32, i32)) {
    if open_core().is_none() {
        return;
    }
    let (offered, refused) = sweep(state, target, centre);
    println!("{state}: {offered} BOMBs offered, {refused} refused");
    assert!(offered > 0 && refused > 0, "{state}: {offered} offered, {refused} refused");
}

/// ow-tektite: 0x2c's rock wall (the take-any cave's heart).
#[test]
fn bombs_at_0x2c_open_its_rock_wall() {
    room("0_2ce_0", BombTarget::rock((18, 12)), (0x90, 0xAD));
}

/// ow-octorok: 0x67's rock wall (the secret rupees, round 22).
#[test]
fn bombs_at_0x67_open_its_rock_wall() {
    room("0_67s_0", BombTarget::rock((14, 2)), (0x70, 0x6D));
}

/// trap-room: Dungeon 3's 0x4c E wall into Manhandla's room.
#[test]
fn bombs_at_3_0x4c_open_its_east_wall() {
    room("3_4cw_0", BombTarget::Wall(Direction::E), (0xC0, 0x8D));
}

/// trap-room: Dungeon 4's rupee room 0x11, its N wall into the Keese room 0x01 (round 22).
#[test]
fn bombs_at_4_0x11_open_its_north_wall() {
    room("4_11s_0", BombTarget::Wall(Direction::N), (0x78, 0x6D));
}

/// stalfos-keese-gel: 4:0x21's N wall, from its W door (the plain way) and from its island (the clip way, where
/// round 21's chains threw 3.4-3.8 bombs a visit).
#[test]
fn bombs_at_4_0x21_open_its_north_wall_from_both_entries() {
    room("4_21w_0", BombTarget::Wall(Direction::N), (0x78, 0x6D));
    room("4_21s_0", BombTarget::Wall(Direction::N), (0x78, 0x6D));
}
