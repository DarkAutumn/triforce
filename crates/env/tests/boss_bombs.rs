//! The two rooms beaten with bombs never leave Link without one (game.yaml `bomb_weapon`; `zelda::cheats`): Dodongo's
//! 2:0x0e and Manhandla's 3:0x4d, on both routes.  The owner, Nov 23: "On a speedrunner's route we always kill
//! Manhandla with bombs and reset if not. So I'm fine with cheating bombs when needed here, I don't ever want to
//! fight them with a wood sword."
//!
//! From round 24 (Nov 14) to Nov 23 Manhandla's room granted nothing: it has a bomb wall (the one Link enters
//! through), and "a room's need ends when its bomb walls are open" read that open wall as the need being met; 246 of
//! 638 visits of a 1000-run Dungeon 3 chain died there, with no cheated bomb in any of them.  These tests enter each
//! room with an EMPTY bag and lay three bombs: each must be there, be offered, be replaced when used and be counted
//! as cheated.  Skipped without the ROM.

mod common;

use common::{TfzStore, ZCore, open_core};
use env::StateStore;
use env::chain::{ChainEnv, ChainStatus, Route};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use zelda::cheats::CheatItem;
use zelda::{Action, ActionKind, MapLocation};

/// `$658`: Link's bombs.
const BOMBS: u16 = 0x658;

/// The first BOMB the mask offers now.
fn bomb(c: &ChainEnv<ZCore, TfzStore>) -> Option<Action> {
    ["BOMB N", "BOMB S", "BOMB W", "BOMB E"]
        .into_iter()
        .map(|text| Action::parse(text).unwrap())
        .find(|action| c.session().mask().is_allowed(action))
}

/// Enters `room` from `state` on `route` with no bombs, walks in off the doorway (`inward`: B is not read in a
/// doorway's border band, so BOMB is offered only inside) and lays three.
fn an_empty_bag_is_never_empty_here(nes: ZCore, route_name: &str, state: &str, room: MapLocation, inward: &str) -> ZCore {
    let what = format!("{route_name} {state}");
    let data = common::game_data();
    assert!(data.map.requires_bombs(room) && data.map.bomb_weapon(room), "{what}: the room's bomb is its weapon");
    assert!(!zelda::cheats::bomb_opens(&data.map, room), "{what}: its granted bomb is bound to no wall");
    let dir = tempfile::tempdir().unwrap();
    let mut source = common::read_state_file(&common::state_path(state)).unwrap();
    source.machine.poke(BOMBS, 0);
    let path = dir.path().join(format!("{state}.state"));
    TfzStore::new().save(&path, &source).unwrap();
    let route = Route::load(&data.map, &data.states_dir, route_name, None, Some(path.to_str().unwrap())).unwrap();
    let mut c = ChainEnv::new(data.clone(), nes, TfzStore::new(), route);
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let ChainStatus::Act { .. } = c.reset(Some(0), &mut rng).unwrap() else { panic!("{what}: the run ended at reset") };
    assert_eq!(c.session().state().full_location(), room, "{what}");
    assert_eq!(c.session().state().bombs(), 1, "{what}: an empty bag is topped up on entering");
    assert_eq!(c.cheated()[CheatItem::Bombs], 0, "{what}: a bomb merely granted is not counted");

    let wait = Action::parse("WAIT 4").unwrap();
    let step_in = Action::parse(inward).unwrap();
    for _ in 0..12 {
        if bomb(&c).is_some() {
            break;
        }
        assert!(c.session().mask().is_allowed(&step_in), "{what}: {inward} off the doorway");
        if let ChainStatus::Done(result) = c.step(step_in).unwrap() {
            panic!("{what}: the run ended walking in: {result:?}");
        }
        assert_eq!(c.session().state().bombs(), 1, "{what}: the bag on the way in");
    }
    assert!(bomb(&c).is_some(), "{what}: BOMB is offered inside the room, with the granted bomb, away from any wall");
    for laid in 1..=3u32 {
        // BOMB is masked while a bomb of Link's is in the air: wait for it, the bag still holding the granted one.
        let mut action = bomb(&c);
        for _ in 0..80 {
            if action.is_some() {
                break;
            }
            assert_eq!(c.session().state().bombs(), 1, "{what}: the bag while bomb {} burns", laid - 1);
            if let ChainStatus::Done(result) = c.step(wait).unwrap() {
                panic!("{what}: the run ended while waiting to lay bomb {laid}: {result:?}");
            }
            action = bomb(&c);
        }
        let action = action.unwrap_or_else(|| panic!("{what}: BOMB is never offered for bomb {laid}"));
        assert_eq!(action.kind, ActionKind::Bomb);
        if let ChainStatus::Done(result) = c.step(action).unwrap() {
            panic!("{what}: the run ended laying bomb {laid}: {result:?}");
        }
        assert_eq!(c.session().state().bombs(), 1, "{what}: bomb {laid} used, the bag topped up again");
        assert_eq!(c.cheated()[CheatItem::Bombs], laid, "{what}: bomb {laid} is counted as cheated");
    }
    c.into_parts().0.into_nes()
}

#[test]
fn dodongo_and_manhandla_are_never_fought_without_a_bomb() {
    let Some(mut nes) = open_core() else { return };
    let dodongo = MapLocation::new(2, 0x0e, false);
    let manhandla = MapLocation::new(3, 0x4d, false);
    for (route, state, room, inward) in [
        ("d2first", "2_0es_0", dodongo, "MOVE N"),
        ("d3first", "d3first_2_0es_0", dodongo, "MOVE N"),
        ("d2first", "3_4dw_0", manhandla, "MOVE E"),
        ("d3first", "d3first_3_4dw_0", manhandla, "MOVE E"),
    ] {
        nes = an_empty_bag_is_never_empty_here(nes, route, state, room, inward);
    }
}

/// The data says which rooms are beaten with bombs, and only those: a room that merely has a bomb wall keeps the
/// round 24 rule (no bomb once its walls are open).
#[test]
fn the_weapon_rooms_are_named_in_the_room_data() {
    let data = common::game_data();
    let weapon: Vec<MapLocation> =
        data.map.rooms().values().filter(|room| room.bomb_weapon).map(|room| room.location).collect();
    let mut expected = vec![MapLocation::new(2, 0x0e, false), MapLocation::new(3, 0x4d, false)];
    let mut found = weapon.clone();
    found.sort();
    expected.sort();
    assert_eq!(found, expected);
    assert!(zelda::cheats::bomb_opens(&data.map, MapLocation::new(3, 0x4c, false)), "the trap room's E wall");
}
