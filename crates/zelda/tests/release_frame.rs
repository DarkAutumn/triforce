//! Round 40 (decisions 2026-11-23, "The fresh generation plays without the closing no-button frame"): a MOVE closes
//! with no frame of no buttons (every other action keeps it: measured, the docks need it after a WAIT). The game reads
//! the directions as held and A, B, Select and Start as new presses (asm:Input_ReadOneController;
//! asm:Mode5Play_Update@CheckMenu takes Link's direction from `ButtonsDown`, asm:Link_HandleInput the sword and the B
//! item from `ButtonsPressed`), so the executor releases a button for one frame only before an action presses it
//! while the game still reads it as held.
#![cfg(feature = "zcore")]

mod common;

use common::{Counted, load_named_state, open_core};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;
use zelda::executor::buttons;
use zelda::{Action, GameMap, LoadOptions, Session, Var};

/// 0:0x1f entered from W (Link at X 0 on its row, nothing near for the first frames).
fn session() -> Option<Session<Counted>> {
    let core = open_core()?;
    let mut s = Session::new(Counted(core, 0), Arc::new(GameMap::builtin()));
    s.load(&load_named_state("d3first_0_1fw_0"), &LoadOptions::default(), &mut ChaCha8Rng::seed_from_u64(0)).unwrap();
    Some(s)
}

fn act(s: &mut Session<Counted>, action: &str) -> zelda::Change {
    s.act(Action::parse(action).unwrap()).unwrap()
}

/// A MOVE lasts its own frames, the rest one more: two MOVEs, a WAIT, a turn.
#[test]
fn no_move_closes_with_a_frame_of_no_buttons() {
    let Some(mut s) = session() else { return };
    let frames: Vec<u32> = ["MOVE E 8", "MOVE E 8", "WAIT 4", "MOVE N 4"].iter().map(|a| act(&mut s, a).frames).collect();
    assert_eq!(frames, [8, 8, 5, 4]);
}

/// After a MOVE, a WAIT, a swing and a bomb, the next swing (A) and the next bomb (B) are new presses: the swing
/// starts (8 frames or more; an unstarted one ends after its two pressed frames), the bomb leaves the bag.
#[test]
fn every_press_registers_after_every_action() {
    for first in ["MOVE E 8", "MOVE S 4", "WAIT 4", "SWORD E", "BOMB E"] {
        for second in ["SWORD E", "SWORD S", "BOMB E", "BOMB S"] {
            let Some(mut s) = session() else { return };
            act(&mut s, "MOVE E 16");
            act(&mut s, first);
            let bombs = s.state().bombs();
            let change = act(&mut s, second);
            if second.starts_with("SWORD") {
                assert!(change.control_frames >= 8, "{first} then {second}: {} frames", change.control_frames);
            } else if !first.starts_with("BOMB") {
                // The game makes one bomb at a time here: a bomb right after a bomb waits for the first.
                assert_eq!(bombs - s.state().bombs(), 1, "{first} then {second}");
            }
        }
    }
}

/// The rule on the game: A (B) held past the swing (the bomb) it started is no new press when pressed on, and the
/// executor's swing (bomb) still registers because it lets go for a frame first; a held direction is walked on.
#[test]
fn a_press_of_a_held_button_is_released_first() {
    let Some(mut s) = session() else { return };
    act(&mut s, "MOVE E 16");
    s.press(buttons::A, 40);
    let raw = s.press(buttons::A, 12).var(Var::LinkStatus);
    assert_eq!(raw, 0, "A held on: no swing");
    assert!(act(&mut s, "SWORD E").control_frames >= 8, "released first: it swings");

    let Some(mut s) = session() else { return };
    act(&mut s, "MOVE E 16");
    act(&mut s, "BOMB E");
    for _ in 0..5 {
        act(&mut s, "WAIT 32");
    }
    s.press(buttons::B, 200);
    let bombs = s.state().bombs();
    s.press(buttons::B, 4);
    assert_eq!(s.state().bombs(), bombs, "B held on: no bomb");
    act(&mut s, "BOMB E");
    assert_eq!(bombs - s.state().bombs(), 1, "released first: a bomb");

    let Some(mut s) = session() else { return };
    act(&mut s, "MOVE E 16");
    s.press(buttons::RIGHT, 4);
    let x = s.state().var(Var::LinkX);
    assert_eq!(act(&mut s, "MOVE E 8").frames, 8, "a held direction needs no release");
    assert!(s.state().var(Var::LinkX) > x);
}
