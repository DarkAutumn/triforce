//! Later visits (round 24): a route that enters a room from the same side more than once names each visit's start
//! states apart (`<level>_<room><side>_v<visit>_<chain>`, `zelda::game_map::route_visits`), and a state starts the
//! step its name spells. The test route `revisit` (`fixtures/revisit_routes.yaml`) goes on from the D4 triforce
//! through 0x5c and 0x5d from W a second time; d2first leaves 0x5d N, `revisit` S. The states below are made from
//! `0_5dw_0` (the first visit): `0_5dw_v2_0` with the second visit's progress written into it (four triforce pieces,
//! the white sword, bow, raft and ladder), `0_5dw_v2_1` a plain copy, whose name alone says the second visit.

mod common;

use common::{TfzStore, ZCore, open_core};
use env::chain::Route;
use env::config::{GameData, RamRandomization};
use env::specialist_env::{EnvOptions, SpecialistEnv};
use std::path::Path;
use std::sync::Arc;
use zelda::game_map::{GAME_YAML, ROUTES_YAML};
use zelda::objectives::{StepProgress, route_progress};
use zelda::room_state::level_bit;
use zelda::{GameMap, MapLocation, Nes, Var};

const FIXTURE: &str = include_str!("fixtures/revisit_routes.yaml");
const ROUTE: &str = "revisit";

fn map() -> Arc<GameMap> {
    Arc::new(GameMap::from_yaml(GAME_YAML, &format!("{ROUTES_YAML}\n{FIXTURE}")).expect("the test route loads"))
}

fn ow(room: u8) -> MapLocation {
    MapLocation::new(0, room, false)
}

/// The visits of the test route's own steps (`tests/test_entry_states.py` uses the same list): 0x5c and 0x5d from W
/// a second time; the stretch's two 0x6f-from-W parts and two 0x5f-from-S steps are each one visit on two ways; 0x6f
/// from S, first entered on the plain way, again on a shared step; D1's bow cellar a second time.
#[test]
fn the_test_route_numbers_its_visits() {
    let map = map();
    let first = map.routes["d2first"].len();
    let visits = &map.visits[ROUTE];
    assert!(visits[..first].iter().all(|&v| v == 1));
    assert_eq!(visits[..first], map.visits["d2first"][..]);
    assert_eq!(visits[first..], [2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2]);
}

/// Writes `0_5dw_0` into `dir` as each of `names`; a name of the second visit gets that visit's progress (route
/// step `step`) unless it is in `copies`.
fn make_states(map: &GameMap, dir: &Path, step: usize, names: &[&str], copies: &[&str]) {
    let progress = route_progress(map, &map.routes[ROUTE])[step];
    for name in names {
        let mut snap = common::read_state_file(&common::state_path("0_5dw_0")).unwrap();
        if name.contains("_v2_") && !copies.contains(name) {
            let triforces = (1..=8u8).filter(|l| progress.triforces & 1 << (l - 1) != 0).map(level_bit).sum::<u32>();
            for (var, value) in [
                (Var::Triforce, triforces as u8),
                (Var::Sword, progress.sword as u8),
                (Var::Bow, progress.bow as u8),
                (Var::Raft, progress.raft as u8),
                (Var::StepLadder, progress.ladder as u8),
            ] {
                <ZCore as Nes>::poke_state(&mut snap.machine, var.addr(), value);
            }
        }
        std::fs::write(dir.join(format!("{name}.state")), common::encode_state(&snap).unwrap()).unwrap();
    }
}

/// A single-room env (the real game) starts the step the state's name spells, on every reset: the bare name the
/// first visit's (out N, towards 0x4d), `_v2` the second's (out S, towards 0x6d).  The second visit's state plays
/// that step's objective; the plain copy (the first visit's progress) starts on it too, its name decides.
#[test]
fn a_later_visit_state_starts_its_own_step() {
    let Some(nes) = open_core() else { return };
    let map = map();
    let data = Arc::new(GameData { map: map.clone(), ..(*common::game_data()).clone() });
    let mut config = (*data.specialists["ow-moblin"]).clone();
    config.route = Some(ROUTE.into());
    config.ram_randomization = RamRandomization::default();
    let steps = &map.routes[ROUTE];
    let second = (0..steps.len()).find(|&i| map.visits[ROUTE][i] == 2 && steps[i].location() == ow(0x5d)).unwrap();
    let dir = tempfile::tempdir().unwrap();
    make_states(&map, dir.path(), second, &["0_5dw_0", "0_5dw_v2_0", "0_5dw_v2_1"], &["0_5dw_v2_1"]);
    let mut env = SpecialistEnv::new(Arc::new(config), data, nes, TfzStore::new(), EnvOptions::default()).unwrap();
    let progress = route_progress(&map, steps);
    for (name, visit, exit, next) in
        [("0_5dw_0", 1, "N", Some(0x4d)), ("0_5dw_v2_0", 2, "S", Some(0x6d)), ("0_5dw_v2_1", 2, "S", None)]
    {
        let path = dir.path().join(format!("{name}.state"));
        for seed in 0..8 {
            env.reset(Some(seed), Some(path.to_str().unwrap())).unwrap();
            let (route, step) = env.route();
            let step = step.unwrap_or_else(|| panic!("{name}: no route step"));
            assert_eq!(route, ROUTE);
            assert_eq!((steps[step].location(), steps[step].entry.as_str()), (ow(0x5d), "W"), "{name}");
            assert_eq!((map.visits[ROUTE][step], steps[step].exit.as_str()), (visit, exit), "{name}");
            let Some(next) = next else { continue };
            assert_eq!(StepProgress::of(env.session().state()), progress[step], "{name}");
            let objective = &env.objective().expect("an objective").objective;
            assert_eq!(objective.next_rooms.iter().copied().collect::<Vec<_>>(), [ow(next)], "{name} seed {seed}");
        }
    }
}

/// A chain's `--from` reads the visit too: from the second visit of 0x5d there is no 0x5c left to stop at, from the
/// first there is (the second visit's).
#[test]
fn a_chain_from_a_later_visit_starts_after_the_first() {
    let map = map();
    let dir = tempfile::tempdir().unwrap();
    for name in ["0_5dw_0", "0_5dw_v2_0"] {
        std::fs::copy(common::state_path("0_5dw_0"), dir.path().join(format!("{name}.state"))).unwrap();
    }
    assert!(Route::load(&map, dir.path(), ROUTE, Some(ow(0x5c)), Some("0_5dw_0")).is_ok());
    let err = Route::load(&map, dir.path(), ROUTE, Some(ow(0x5c)), Some("0_5dw_v2_0")).unwrap_err();
    assert!(err.contains("not a room of route 'revisit' after the start step"), "{err}");
    assert!(Route::load(&map, dir.path(), ROUTE, Some(ow(0x6d)), Some("0_5dw_v2_0")).is_ok());
}
