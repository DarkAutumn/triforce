//! Go-Explore's cells (`env::explore::cell_key`) tell a room whose item is taken from the same spot before the pickup
//! (round 22: 0x67's secret rupees change nothing else the key reads; with one cell for both, the shorter path,
//! without the rupees, kept it, and the state builder never solved the cave: 52 cells, no success in 3000
//! iterations).

mod common;

use common::{TfzStore, open_core};
use env::explore::{Explorer, cell_key, progress_events, room_item_taken};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;
use zelda::executor::buttons::{DOWN, RIGHT, UP};
use zelda::{GameMap, LoadOptions, Session};

/// In 0x67's cave from its entrance (`$70`, `$D5`): Right to X `$78`, Up onto the middle ware (the 30 rupees), Down
/// back to the entrance row. The cell there differs from the entrance's only by the room's item-taken flag, and the
/// pickup frame counts as progress.
#[test]
fn taking_0x67s_secret_rupees_makes_a_new_cell() {
    let Some(nes) = open_core() else { return };
    let mut s = Session::new(nes, Arc::new(GameMap::builtin()));
    s.load(&common::load_named_state("0_67cave_0").unwrap(), &LoadOptions::default(), &mut ChaCha8Rng::seed_from_u64(0))
        .unwrap();
    let entrance = s.state().clone();
    assert!(entrance.in_cave() && !room_item_taken(&entrance));
    let (x, y) = (entrance.link.position.x, entrance.link.position.y);
    assert_eq!((x, y), (0x70, 0xD5));
    for _ in 0..300 {
        if s.state().link.position.x == 0x78 {
            break;
        }
        s.press(RIGHT, 1);
    }
    let rupees = s.state().rupees();
    let mut picked = None;
    for _ in 0..600 {
        let before = s.state().clone();
        s.press(UP, 1);
        if room_item_taken(s.state()) {
            picked = Some(progress_events(&before, s.state()));
            break;
        }
    }
    assert_eq!(picked, Some(1), "the pickup is one progress event");
    for _ in 0..300 {
        if s.state().link.position.y == y {
            break;
        }
        s.press(DOWN, 1);
    }
    let back = s.state().clone();
    assert!(back.rupees() > rupees && back.in_cave());
    let (a, b) = (cell_key(&entrance), cell_key(&back));
    assert_ne!(a, b);
    assert_eq!((a.0, a.1, a.2, a.3, a.4, a.5, a.6, a.7, a.8, a.9, a.10), (b.0, b.1, b.2, b.3, b.4, b.5, b.6, b.7, b.8, b.9, b.10));
    assert!(!a.11 && b.11);
}

/// ow-octorok's explorer takes the rupees and leaves the cave S (`exit-correct`; without them leaving is `missed`).
#[test]
fn the_explorer_takes_the_rupees_and_leaves_0x67s_cave() {
    let Some(nes) = open_core() else { return };
    let data = common::game_data();
    let config = data.specialists["ow-octorok"].clone();
    let mut explorer = Explorer::new(&config, data.clone(), nes, TfzStore::new(), 7).unwrap();
    let result = explorer.explore("0_67cave_0", 3000, 25, 20, None).unwrap();
    assert!(result.success, "{result:?}");
    assert_eq!(result.ending.as_deref(), Some("exit-correct"));
}
