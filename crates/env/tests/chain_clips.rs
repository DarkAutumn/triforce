//! The run-time clips switch in a chain run (round 22) through Dungeon 4's stretch `d4-island`: the start step
//! follows the switch, a flip mid-room re-picks the step and is recorded, the plain way holds once taken, and a flip
//! on 0x21's island re-plans back to the S door.

mod common;

use common::{TfzStore, ZCore, open_core};
use env::chain::{ChainEnv, ChainStatus, ClipsFlip, Route};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::BTreeSet;
use zelda::{Action, MapLocation};

type Chain = ChainEnv<ZCore, TfzStore>;

/// A chain from `state` with the switch `on` (and `off_from`), reset with seed 0.
fn start(nes: ZCore, state: &str, on: bool, off_from: Option<ClipsFlip>) -> Chain {
    let data = common::game_data();
    let route = Route::load(&data.map, &data.states_dir, "d2first", None, Some(state))
        .unwrap()
        .with_clips(&data.map, on, off_from)
        .unwrap();
    let mut chain = ChainEnv::new(data, nes, TfzStore::new(), route);
    let status = chain.reset(Some(0), &mut ChaCha8Rng::seed_from_u64(0)).unwrap();
    assert!(matches!(status, ChainStatus::Act { .. }), "{status:?}");
    chain
}

fn next_rooms(chain: &Chain) -> BTreeSet<u8> {
    chain.objective().unwrap().next_rooms.iter().map(|l| l.value).collect()
}

fn step_way(chain: &Chain) -> Option<&'static str> {
    let data = common::game_data();
    let step = chain.context().0?;
    data.map.routes["d2first"][step].way.map(|w| w.name())
}

fn wait(chain: &mut Chain) {
    assert!(matches!(chain.step(Action::parse("WAIT 4").unwrap()).unwrap(), ChainStatus::Act { .. }));
}

/// From 4_31e (back W from the Breakfast Room with the ladder) the switch picks the way: on, the clip way's N; off,
/// the plain way's W. Turned off mid-room the plain step takes over at once (recorded as a flip of the visit);
/// turned on again on the plain way, the plain way holds to the rejoin.
#[test]
fn the_switch_picks_and_re_picks_the_way_in_0x31() {
    let Some(nes) = open_core() else { return };
    let mut chain = start(nes, "4_31e_0", true, None);
    assert_eq!((next_rooms(&chain), step_way(&chain)), ([0x21].into(), Some("clips")));
    wait(&mut chain);
    chain.set_clips_allowed(false);
    wait(&mut chain);
    assert_eq!((next_rooms(&chain), step_way(&chain), chain.clips_allowed()), ([0x30].into(), Some("plain"), false));
    chain.set_clips_allowed(true);
    wait(&mut chain);
    assert_eq!((next_rooms(&chain), step_way(&chain)), ([0x30].into(), Some("plain")), "the plain way holds");
    let record = chain.result().unwrap().rooms.last().unwrap().clone();
    assert_eq!((record.clips_allowed, record.clips_flips.clone()), (true, vec![(2, false), (3, true)]));
    let json = record.to_json();
    assert_eq!(json["clips_flips"][0]["clips"], false);
    let nes = chain.into_parts().0.into_nes();
    let chain = start(nes, "4_31e_0", false, None);
    assert_eq!((next_rooms(&chain), step_way(&chain)), ([0x30].into(), Some("plain")), "off from the start");
    assert!(!chain.result().unwrap().clips);
}

/// `clips_off_from` 0x21's room, from its island (`4_21s`): the switch goes off at the first decision and the
/// objective re-plans to the island's S door (0x31), the only exit Link can walk to; the room's record starts with
/// the switch off and lists the flip at step 0. A stop room only the plain way reaches is refused with clips on.
#[test]
fn a_flip_on_the_island_replans_to_its_south_door() {
    let Some(nes) = open_core() else { return };
    let island = MapLocation::new(4, 0x21, false);
    let chain = start(nes, "4_21s_0", true, Some(ClipsFlip::Room(island)));
    assert_eq!((next_rooms(&chain), chain.clips_allowed(), chain.context().0), ([0x31].into(), false, None));
    let record = &chain.result().unwrap().rooms[0];
    assert_eq!((record.clips_allowed, record.clips_flips.clone()), (false, vec![(0, false)]));
    let data = common::game_data();
    let room_20 = Some(MapLocation::new(4, 0x20, false));
    let only_plain = Route::load(&data.map, &data.states_dir, "d2first", room_20, Some("4_31e_0")).unwrap();
    assert!(only_plain.clone().with_clips(&data.map, true, None).is_err());
    assert!(only_plain.with_clips(&data.map, false, None).is_ok());
}
