//! Round 44 (the real walk, `entry_states walk`): an explorer whose resets load the start state exactly as saved
//! (`Explorer::set_exact_starts`), and the kills a path may make (`KillSpec`).

mod common;

use common::{TfzStore, open_core};
use env::explore::{Explorer, KillSpec};

/// The random register, the spawn cycle and the edge cell at the first controllable frame after a reset.
fn drawn(explorer: &mut Explorer<common::ZCore, TfzStore>, seed: u64, state: &str) -> Vec<u8> {
    explorer.env.reset(Some(seed), Some(state)).unwrap();
    let view = explorer.env.session().state();
    (0x18..=0x24).chain([0x524, 0x525]).map(|a| view.peek(a)).collect()
}

#[test]
fn an_exact_start_draws_nothing() {
    let Some(nes) = open_core() else { return };
    let data = common::game_data();
    let config = data.specialists["ow-zora"].clone();
    let mut explorer = Explorer::new(&config, data.clone(), nes, TfzStore::new(), 0).unwrap();
    // A reset draws the random phase and, before a room entry, the spawn cycle and edge cell: two seeds differ.
    assert_ne!(drawn(&mut explorer, 1, "0_38n_0"), drawn(&mut explorer, 2, "0_38n_0"));
    // An exact reset plays the saved machine's entry: every seed meets the same bytes.
    explorer.set_exact_starts(true);
    let first = drawn(&mut explorer, 1, "0_38n_0");
    assert_eq!(first, drawn(&mut explorer, 2, "0_38n_0"));
    assert_eq!(first, drawn(&mut explorer, 3, "0_38n_0"));
}

#[test]
fn a_kill_spec_counts_the_rooms_kills_and_the_living() {
    let Some(nes) = open_core() else { return };
    let data = common::game_data();
    let config = data.specialists["ow-zora"].clone();
    let mut explorer = Explorer::new(&config, data.clone(), nes, TfzStore::new(), 0).unwrap();
    explorer.set_exact_starts(true);
    explorer.env.reset(Some(0), Some("0_38n_0")).unwrap();
    let view = explorer.env.session().state().clone();
    assert_eq!(view.room_kills(), 0);
    let crossing = KillSpec { max: Some(0), ..Default::default() };
    assert!(!crossing.too_many(&view) && crossing.met(&view));
    assert!(KillSpec { exact: Some(0), ..Default::default() }.met(&view));
    assert!(!KillSpec { exact: Some(1), ..Default::default() }.met(&view));
    let living = view.enemies.iter().any(|e| !e.is_dying());
    assert_eq!(KillSpec { clear: true, ..Default::default() }.met(&view), !living);
    // The search holds to it: a crossing that may kill nothing still finds the room's exit.
    explorer.set_kills(Some(crossing));
    let result = explorer.explore("0_38n_0", 400, 25, 50, None).unwrap();
    assert!(result.success, "{result:?}");
}
