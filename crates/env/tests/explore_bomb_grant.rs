//! The explorer's search and its replay agree in a bomb-wall room entered with an empty bag: d3first's 3:0x5b W -> E
//! (`bomb_reserve: 0`), where Link arrives with no bombs, the cheat ledger grants the wall's one and BOMB is kept to
//! the route's E wall while that grant is outstanding (`ItemsConfig::bomb_targets`).  The search used to start a new
//! episode at every cell it picked (`SpecialistEnv::restore`): the ledger began again and took the granted bomb for
//! Link's own, so BOMB was free there, the search bombed a Darknut, and the replay from the start refused its track
//! (`InvalidActionError: BOMB N is masked`, `python -m triforce.explore --specialist darknut --states
//! d3first_3_5bw_0`).  A cell now continues its episode exactly (`SpecialistEnv::save`/`load`).  Skipped without
//! the ROM.

mod common;

use common::{TfzStore, open_core};
use env::curriculum::TRAJECTORY_FILE;
use env::explore::{Explorer, explore_config};
use env::specialist_env::{EnvOptions, SpecialistEnv};
use std::sync::Arc;
use zelda::{Action, ActionKind};

const STATE: &str = "d3first_3_5bw_0";
/// A staged start (`python -m triforce.explore --room-kills 3:0x5b=2`): two of the three Darknuts banked, so the
/// search ends in about a thousand iterations rather than five (one Darknut left to bomb or to fight).
const KILLS: ((u8, u8), u8) = ((3, 0x5b), 2);
/// The search's seed (the explorer's and its episodes'): seed 10 clears the room in 143 iterations, seed 0 in 1152
/// (Nov 23, seeds 0-11 at patience 0: 143-1152); the test proved the same with either, in a tenth of the time.
const SEED: u64 = 10;

/// The explorer (at most 3000 iterations of 25 steps, seed `SEED`, stopping at its first success: any success leaves
/// by the bombed wall) clears the room and leaves by its wall; the track it writes replays, and its recorded actions
/// (`trajectory.json`) replay in an env of their own: every action legal where the track plays it, the same success.
#[test]
fn a_track_through_a_bomb_wall_room_with_an_empty_bag_replays() {
    let (Some(nes), Some(nes2)) = (open_core(), open_core()) else { return };
    let data = common::game_data();
    let config = data.specialists["darknut"].clone();
    let opts = EnvOptions { explicit_starts: true, skip_look_ahead: true, ..Default::default() };
    let mut env =
        SpecialistEnv::new(Arc::new(explore_config(&config)), data.clone(), nes2, TfzStore::new(), opts).unwrap();
    env.set_banked_kills(Some(vec![KILLS]));
    env.reset(Some(SEED), Some(STATE)).unwrap();
    assert_eq!(env.route().0, "d3first");
    assert_eq!(env.session().state().bombs(), 1, "the wall's bomb");
    assert_eq!(env.context().1.room.bombs, 1, "an empty bag: the bomb is granted (by the room)");

    let mut explorer = Explorer::new(&config, data.clone(), nes, TfzStore::new(), SEED).unwrap();
    explorer.set_banked_kills(Some(vec![KILLS]));
    let result = explorer.explore(STATE, 3000, 25, 0, None).unwrap();
    assert!(result.success, "{result:?}");
    let dir = tempfile::tempdir().unwrap();
    assert!(explorer.write(&result, dir.path()).unwrap(), "the search's track replays from the start");
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.path().join(TRAJECTORY_FILE)).unwrap()).unwrap();
    assert_eq!(meta["success"], true, "{meta}");
    let actions: Vec<Action> =
        meta["actions"].as_array().unwrap().iter().map(|a| Action::parse(a.as_str().unwrap()).unwrap()).collect();
    assert_eq!(actions.len(), result.actions.len());
    assert!(actions.iter().any(|a| a.kind == ActionKind::Bomb), "the wall is bombed: {actions:?}");

    // The recorded track in an env of its own (the curriculum's and the state builder's view).
    env.reset(Some(SEED), Some(STATE)).unwrap();
    for (i, action) in actions.iter().enumerate() {
        assert!(env.session().mask().is_allowed(action), "step {i}: {action} is masked");
        let step = env.step(*action).unwrap();
        if step.terminated || step.truncated {
            assert_eq!(i + 1, actions.len(), "step {i} ends the episode early");
            assert_eq!(step.info.outcome(), Some(env::rewards::SUCCESS));
            assert_eq!(step.info.ending().map(str::to_owned), result.ending);
            return;
        }
    }
    panic!("the track ends before the episode");
}
