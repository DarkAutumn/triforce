//! ChainEnv (`tests/test_runner.py`) and Explorer (`tests/test_explore.py`) on zcore with scripted
//! policies in place of networks.

mod common;

use common::{MachineState, TfzStore, ZCore, open_core};
use env::StateStore;
use env::chain::{ChainEnv, ChainStatus, Route, RunResult};
use env::config::{GameData, SpecialistConfig};
use env::curriculum::{Curriculum, CurriculumSpec, load_trajectories};
use env::explore::{Explorer, random_action};
use env::specialist_env::{EnvOptions, SpecialistEnv};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;
use zelda::cheats::ItemCounts;
use zelda::{Action, GameMap, LoadOptions, MapLocation, Nes, Session, power_on};

fn data() -> Arc<GameData> {
    common::game_data()
}

/// Runs one chain to its end with `policy`.
fn drive<St: StateStore<MachineState>>(
    chain: &mut ChainEnv<ZCore, St>,
    seed: u64,
    mut policy: impl FnMut(&Session<ZCore>, &mut ChaCha8Rng) -> Action,
) -> RunResult {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut status = chain.reset(Some(seed), &mut rng).expect("reset");
    loop {
        match status {
            ChainStatus::Done(r) => return r,
            ChainStatus::Act { .. } => {
                let action = policy(chain.session(), &mut rng);
                status = chain.step(action).expect("step");
            }
        }
    }
}

fn random_policy(session: &Session<ZCore>, rng: &mut ChaCha8Rng) -> Action {
    random_action(session.mask(), rng, None)
}

fn chain(nes: ZCore, data: &Arc<GameData>, route: Route) -> ChainEnv<ZCore, TfzStore> {
    ChainEnv::new(data.clone(), nes, TfzStore::new(), route)
}

fn d2first(data: &GameData) -> Route {
    Route::load(&data.map, &data.states_dir, "d2first", None, None).unwrap()
}

/// The config of `name` with `edit` applied.
fn config(
    data: &GameData,
    name: &str,
    edit: impl FnOnce(&mut SpecialistConfig),
) -> Arc<SpecialistConfig> {
    let mut c = (*data.specialists[name]).clone();
    edit(&mut c);
    Arc::new(c)
}

fn parse(a: &str) -> Action {
    Action::parse(a).unwrap()
}

#[test]
fn route_definition() {
    let data = data();
    let (map, states) = (&data.map, &data.states_dir);
    let route = Route::load(map, states, "d2first", None, None).unwrap();
    assert_eq!(route.start_state, "start");
    assert_eq!((route.terminal, route.final_triforce), (None, Some(3)));
    let stop = MapLocation::new(1, 0x73, false);
    assert_eq!(
        Route::load(map, states, "d2first", Some(stop), None)
            .unwrap()
            .terminal,
        Some(stop)
    );
    assert!(Route::load(map, states, "no-such-route", None, None).is_err());
}

#[test]
fn every_route_room_has_a_specialist() {
    let data = data();
    for (name, steps) in &data.map.routes {
        for step in steps {
            assert_eq!(
                data.specialist_for(step.location()),
                step.specialist.as_deref(),
                "{name} {:?}",
                step
            );
        }
    }
}

/// The chain tests, one test each (`chain_runs::<case>`), so they run in parallel.  A case takes a powered-on
/// emulator and hands it back.
macro_rules! chain_runs {
    ($($case:ident),* $(,)?) => {
        mod chain_runs {
            $(
                #[test]
                fn $case() {
                    let Some(nes) = super::open_core() else { return };
                    super::$case(nes, &super::data());
                }
            )*
        }
    };
}

chain_runs!(
    untagged_room_stops_with_clear_error,
    wrong_exit_into_untagged_room_is_a_failure,
    missing_model_is_an_error,
    run_records_rooms_and_ends_cleanly,
    same_seed_same_run,
    no_room_progress_threshold_replaces_flat_budget,
    no_room_progress_preserves_global_guard_and_disabled_budget,
    room_exit_resets_visit_clock,
    lynel_retreat_continues_the_run,
    cave_bounce_hits_the_wrong_exit_limit,
    back_and_forth_hits_the_bounce_limit,
    mid_route_triforce_is_the_correct_exit,
    harvested_states_hold_no_granted_bomb,
    one_room_entries_reset_the_observation,
    target_observation_follows_the_acting_specialist,
    a_stop_step_ends_the_run_and_a_checkpoint_continues_it,
    a_stop_step_after_a_triforce_ends_the_run_on_the_overworld,
    a_power_on_start_boots_the_game_by_inputs_alone,
    chain_rewards_are_the_training_rewards,
);

fn untagged_room_stops_with_clear_error(nes: ZCore, data: &Arc<GameData>) -> ZCore {
    let mut map = GameMap::builtin();
    map.get_mut(MapLocation::new(0, 0x77, false))
        .unwrap()
        .specialist = None;
    let untagged = Arc::new(GameData {
        root: data.root.clone(),
        map: Arc::new(map),
        families: data.families.clone(),
        chain_hp: data.chain_hp.clone(),
        specialists: data.specialists.clone(),
        states_dir: data.states_dir.clone(),
        trajectories_dir: data.trajectories_dir.clone(),
    });
    let mut c = chain(nes, &untagged, d2first(data));
    let r = drive(&mut c, 0, random_policy);
    assert_eq!(r.result, "error");
    assert!(
        r.reason.contains("no specialist") && r.reason.contains("0x77"),
        "{}",
        r.reason
    );
    c.into_parts().0.into_nes()
}

fn wrong_exit_into_untagged_room_is_a_failure(nes: ZCore, data: &Arc<GameData>) -> ZCore {
    let mut c = chain(nes, data, d2first(data));
    c.max_steps = 200;
    let r = drive(&mut c, 0, |_, _| parse("MOVE W 16"));
    assert_eq!(r.result, "failure");
    assert!(
        r.reason.contains("untagged") && r.reason.contains("0x76"),
        "{}",
        r.reason
    );
    let last = r.rooms.last().unwrap();
    assert_eq!(
        (last.room.as_str(), last.outcome.as_str()),
        ("0x77", "exit-wrong")
    );
    c.into_parts().0.into_nes()
}

fn missing_model_is_an_error(nes: ZCore, data: &Arc<GameData>) -> ZCore {
    let mut c = chain(nes, data, d2first(data));
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let status = c.reset(Some(0), &mut rng).unwrap();
    let ChainStatus::Act {
        specialist,
        new_room,
        ..
    } = status
    else {
        panic!("expected Act")
    };
    assert!(new_room);
    let r = c.fail(format!("no promoted model for specialist {specialist}"));
    assert_eq!(r.result, "error");
    assert_eq!(&*specialist, "ow-start", "the route's first step");
    assert_eq!(r.rooms[0].outcome, "error");
    assert!(
        c.step(parse("WAIT 8")).is_err(),
        "a finished run takes no more steps"
    );
    c.into_parts().0.into_nes()
}

/// The chain fills `target` only for specialists with `obs.target` on, from the same wavefront their env uses.
fn target_observation_follows_the_acting_specialist(mut nes: ZCore, data: &Arc<GameData>) -> ZCore {
    for on in [false, true] {
        let mut c = chain(nes, data, d2first(data));
        c.configs.insert(
            "ow-start".into(),
            config(data, "ow-start", |c| c.obs.target = on),
        );
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let mut status = c.reset(Some(0), &mut rng).unwrap();
        for step in 0..3 {
            let ChainStatus::Act { obs, .. } = status else {
                panic!("expected Act")
            };
            if on {
                // The start room's exit is reachable: a walking distance and one first step.
                assert!(obs.target[4] > 0.0 && obs.target[4] < 1.0, "{step}: {:?}", obs.target);
                assert_eq!(obs.target[..4].iter().sum::<f32>(), 1.0, "{step}: {:?}", obs.target);
            } else {
                assert!(obs.target.iter().all(|&x| x == 0.0), "{step}: {:?}", obs.target);
            }
            status = c.step(parse("WAIT 8")).unwrap();
        }
        nes = c.into_parts().0.into_nes();
    }
    nes
}

fn run_records_rooms_and_ends_cleanly(nes: ZCore, data: &Arc<GameData>) -> ZCore {
    let mut c = chain(nes, data, d2first(data));
    c.max_steps = 150;
    let r = drive(&mut c, 3, random_policy);
    assert!(r.result == "failure" || r.result == "success", "{r:?}");
    assert_eq!(r.rooms[0].specialist, "ow-start");
    assert_eq!(r.steps, r.rooms.iter().map(|x| x.steps).sum::<u64>());
    assert!(r.rooms.iter().all(|x| x.outcome != "in-progress"));
    assert!(["stuck", "death", "milestone"].contains(&r.rooms.last().unwrap().outcome.as_str()));
    let room = &r.to_json()["rooms"][0];
    assert_eq!(room["specialist"], "ow-start");
    let n = |k: &str| room[k].as_u64().unwrap();
    assert_eq!(
        n("hearts_dropped"),
        n("hearts_taken") + n("hearts_left") + n("hearts_expired")
    );
    c.into_parts().0.into_nes()
}

fn same_seed_same_run(nes: ZCore, data: &Arc<GameData>) -> ZCore {
    let mut c = chain(nes, data, d2first(data));
    c.max_steps = 60;
    let first = drive(&mut c, 7, random_policy);
    let second = drive(&mut c, 7, random_policy);
    assert_eq!(first.to_json(), second.to_json());
    assert_eq!(first.steps, 60);
    c.into_parts().0.into_nes()
}

fn no_room_progress_threshold_replaces_flat_budget(
    nes: ZCore,
    data: &Arc<GameData>,
) -> ZCore {
    let mut c = chain(nes, data, d2first(data));
    c.min_room_steps = 2;
    c.max_steps = 20;
    c.configs.insert(
        "ow-start".into(),
        config(data, "ow-start", |c| {
            c.no_progress_steps = Some(4);
            c.max_steps = 1;
        }),
    );
    let r = drive(&mut c, 0, |_, _| parse("WAIT 8"));
    assert_eq!(r.steps, 4);
    assert_eq!(r.result, "failure");
    let last = r.rooms.last().unwrap();
    assert_eq!(last.outcome, "stuck");
    assert!(r.reason.contains("no-room-progress"), "{}", r.reason);
    let linger = last.linger.unwrap();
    assert_eq!(linger.tail_steps, 4);
    assert_eq!(linger.tail_frames as u64, last.frames);
    c.into_parts().0.into_nes()
}

fn no_room_progress_preserves_global_guard_and_disabled_budget(
    mut nes: ZCore,
    data: &Arc<GameData>,
) -> ZCore {
    for enabled in [false, true] {
        let mut c = chain(nes, data, d2first(data));
        c.min_room_steps = 3;
        c.max_steps = 5;
        c.configs.insert(
            "ow-start".into(),
            config(data, "ow-start", |c| {
                c.no_progress_steps = enabled.then_some(20);
                c.max_steps = 1;
            }),
        );
        let r = drive(&mut c, 0, |_, _| parse("WAIT 8"));
        assert_eq!(r.result, "failure");
        assert_eq!(r.rooms.last().unwrap().outcome, "stuck");
        assert_eq!(r.steps, if enabled { 5 } else { 3 });
        // The step budgets end it, not the (longer) no-progress threshold.
        assert!(!r.reason.contains("no-room-progress"), "{}", r.reason);
        assert_eq!(
            r.rooms.last().unwrap().linger.unwrap().tail_steps as u64,
            r.steps
        );
        nes = c.into_parts().0.into_nes();
    }
    nes
}

fn room_exit_resets_visit_clock(nes: ZCore, data: &Arc<GameData>) -> ZCore {
    let route = Route::load(&data.map, &data.states_dir, "d2first", None, Some("0_77cave_0")).unwrap();
    let mut c = chain(nes, data, route);
    c.max_steps = 100;
    c.configs.insert(
        "ow-start".into(),
        config(data, "ow-start", |c| c.no_progress_steps = Some(20)),
    );
    let r = drive(&mut c, 0, |s, _| {
        if s.state().in_cave() {
            parse("MOVE S 16")
        } else {
            parse("WAIT 8")
        }
    });
    assert_eq!(r.result, "failure");
    assert!(r.reason.contains("no-room-progress"), "{}", r.reason);
    assert_eq!(r.rooms.len(), 2);
    let (old, new) = (&r.rooms[0], &r.rooms[1]);
    assert!(old.in_cave && old.linger.unwrap().tail_steps == 0);
    let linger = new.linger.unwrap();
    assert!(!new.in_cave);
    assert_eq!(
        (new.steps, linger.steps as u64, linger.tail_steps as u64),
        (20, 20, 20)
    );
    c.into_parts().0.into_nes()
}

/// Leaving 0x0a S before the white sword (routes.yaml `retreat: S`, Lee Nov 8) doesn't end the run: the visit ends as
/// `retreat`, the run goes on in 0x1a, whose objective sends Link back N (a correct exit).  A retreat beyond the
/// run's cap (`MAX_RETREATS` = 3; 1 here) ends the run as a failure, `retreat-limit`.
fn lynel_retreat_continues_the_run(nes: ZCore, data: &Arc<GameData>) -> ZCore {
    assert_eq!(env::chain::MAX_RETREATS, 3);
    let route = Route::load(&data.map, &data.states_dir, "d2first", None, Some("0_0as_0")).unwrap();
    let mut c = chain(nes, data, route);
    c.max_steps = 400;
    c.max_retreats = 1;
    let (south, north, wait) = (parse("MOVE S 16"), parse("MOVE N 16"), parse("WAIT 8"));
    let r = drive(&mut c, 0, |s, _| {
        let want = if s.state().full_location() == MapLocation::new(0, 0x0a, false) { south } else { north };
        if s.mask().is_allowed(&want) { want } else { wait }
    });
    let rooms: Vec<_> = r.rooms.iter().map(|x| (x.room.as_str(), x.specialist.as_str(), x.outcome.as_str())).collect();
    assert_eq!(rooms[0], ("0x0a", "ow-lynel", "retreat"), "{r:?}");
    assert_eq!(rooms[1], ("0x1a", "ow-tektite", "exit-correct"), "{r:?}");
    assert_eq!(r.retreats, 2, "{r:?}");
    assert_eq!(rooms.last().unwrap(), &("0x0a", "ow-lynel", "retreat-limit"), "{r:?}");
    assert_eq!(r.result, "failure");
    assert!(r.reason.starts_with("retreat-limit"), "{}", r.reason);
    assert_eq!(r.to_json()["retreats"], 2);
    c.into_parts().0.into_nes()
}

/// F157: wrong exits into tagged rooms (here out of the sword cave S without the sword, and random walking after)
/// are counted per run; the `max_wrong_exits`-th ends the run `stuck`, not the run cap.
fn cave_bounce_hits_the_wrong_exit_limit(nes: ZCore, data: &Arc<GameData>) -> ZCore {
    assert_eq!(env::chain::MAX_WRONG_EXITS, 20);
    let route = Route::load(&data.map, &data.states_dir, "d2first", None, Some("0_77cave_0")).unwrap();
    let mut c = chain(nes, data, route);
    c.max_steps = 5000;
    c.max_wrong_exits = 3;
    let south = parse("MOVE S 16");
    let r = drive(&mut c, 0, |s, rng| {
        if s.state().in_cave() && s.mask().is_allowed(&south) { south } else { random_policy(s, rng) }
    });
    let rooms: Vec<_> = r.rooms.iter().map(|x| (&x.room, x.in_cave, &x.outcome, x.steps)).collect();
    assert_eq!(r.result, "failure", "{}", r.reason);
    assert!(r.reason.starts_with("wrong-exit-limit"), "{} {rooms:?}", r.reason);
    assert!(r.steps < 5000);
    assert_eq!(r.wrong_exits, 3);
    assert_eq!(r.to_json()["wrong_exits"], 3);
    assert_eq!(r.rooms.last().unwrap().outcome, "stuck");
    assert_eq!(r.rooms.iter().filter(|x| x.outcome == "exit-wrong").count(), 2);
    c.into_parts().0.into_nes()
}

/// Lee, Nov 11: walking back into the room before the last and forward again once or twice is fine; the 5th return
/// in a row (`MAX_BOUNCES`) ends the run `stuck` (`bounce-limit`), before the wrong-exit limit.  Here Link walks
/// between 0x4d and 0x4c from 0x4d's route entry.
fn back_and_forth_hits_the_bounce_limit(nes: ZCore, data: &Arc<GameData>) -> ZCore {
    assert_eq!(env::chain::MAX_BOUNCES, 5);
    let route = Route::load(&data.map, &data.states_dir, "d2first", None, Some("0_4ds_0")).unwrap();
    let mut c = chain(nes, data, route);
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let ChainStatus::Act { .. } = c.reset(Some(0), &mut rng).unwrap() else { panic!("run ended at reset") };
    let (east, west) = (MapLocation::new(0, 0x4d, false), MapLocation::new(0, 0x4c, false));
    let mut done = None;
    while done.is_none() {
        let target = if c.session().state().full_location() == east { west } else { east };
        for a in walk_to(&mut c, target) {
            if let ChainStatus::Done(r) = c.step(a).unwrap() {
                done = Some(r);
                break;
            }
        }
    }
    let r = done.unwrap();
    let rooms: Vec<_> = r.rooms.iter().map(|x| (x.room.as_str(), x.outcome.as_str())).collect();
    assert_eq!(r.result, "failure", "{rooms:?}");
    assert!(r.reason.starts_with("bounce-limit"), "{} {rooms:?}", r.reason);
    assert_eq!(rooms.len(), 6, "0x4d, then 0x4c and 4 returns: {rooms:?}");
    assert_eq!(rooms.last().unwrap(), &("0x4c", "stuck"));
    assert!(r.wrong_exits < env::chain::MAX_WRONG_EXITS);
    c.into_parts().0.into_nes()
}

/// F135: taking a triforce the route takes before its last (Dungeon 2's in 2:0x0d) is that room's correct exit,
/// and the handoff it causes (0:0x3c out of the dungeon) is harvested.
fn mid_route_triforce_is_the_correct_exit(nes: ZCore, data: &Arc<GameData>) -> ZCore {
    let route = Route::load(&data.map, &data.states_dir, "d2first", None, Some("2_0de_0")).unwrap();
    let mut c = chain(nes, data, route);
    let dir = tempfile::tempdir().unwrap();
    c.set_harvest(Some(dir.path().to_path_buf()));
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let ChainStatus::Act { .. } = c.reset(Some(0), &mut rng).unwrap() else { panic!("run ended at reset") };
    if let Some(r) = walk_onto_the_triforce(&mut c) {
        panic!("run ended on the walk: {r:?}");
    }
    // The run goes on out of the dungeon (only the route's last triforce ends it); wait for the 0x3c entry.
    while c.result().unwrap().rooms.len() < 2 {
        if let ChainStatus::Done(r) = c.step(parse("WAIT 8")).unwrap() {
            panic!("run ended before 0x3c: {r:?}");
        }
    }
    let r = c.result().unwrap();
    let rooms: Vec<_> = r.rooms.iter().map(|x| (x.level, x.room.as_str(), x.outcome.as_str())).collect();
    assert_eq!(rooms, [(2, "0x0d", "exit-correct"), (0, "0x3c", "in-progress")]);
    assert!(dir.path().join("ow-octorok/0_3cc_h0_1.state").is_file());
    c.into_parts().0.into_nes()
}

/// Walks the chain onto the triforce of the room it is in: a breadth-first search over Link's tiles finds the
/// walk, the chain then plays it.  Returns the run's result if the walk ended it (the step that takes the
/// triforce goes on until Link stands outside the dungeon).
fn walk_onto_the_triforce(c: &mut ChainEnv<ZCore, TfzStore>) -> Option<RunResult> {
    let start = c.session().snapshot();
    let mut queue = std::collections::VecDeque::from([(start.clone(), Vec::new())]);
    let mut seen = std::collections::HashSet::from([c.session().state().link.tile()]);
    let mut path = None;
    while let Some((snap, actions)) = queue.pop_front() {
        c.session_mut().restore(&snap);
        for d in c.session().mask().directions(zelda::ActionKind::Move) {
            let a = zelda::Action::new(zelda::ActionKind::Move, d, 8).unwrap();
            c.session_mut().restore(&snap);
            let change = c.session_mut().act(a).unwrap();
            let mut next = actions.clone();
            next.push(a);
            if change.gained_triforce() {
                path = Some(next);
                break;
            }
            if !change.changed_location() && seen.insert(change.state.link.tile()) {
                queue.push_back((c.session().snapshot(), next));
            }
        }
        if path.is_some() {
            break;
        }
    }
    c.session_mut().restore(&start);
    for a in path.expect("a walk onto the triforce") {
        if let ChainStatus::Done(r) = c.step(a).unwrap() {
            return Some(r);
        }
    }
    None
}

/// A route with a stop step ends the run, a success, on entering that step's room by the route's exit; a
/// route with a checkpoint starts from the state another run ended in, exactly as it is, on its route step.
fn a_stop_step_ends_the_run_and_a_checkpoint_continues_it(nes: ZCore, data: &Arc<GameData>) -> ZCore {
    let load = |start: Option<&str>| Route::load(&data.map, &data.states_dir, "d2first", None, start).unwrap();
    let steps = data.map.routes["d2first"].len();
    // 0x3c, where Link comes out of Dungeon 2 (step 24): straight south is 0x4c (step 25).
    let mut c = chain(nes, data, load(Some("0_3cc_0")).with_stop_step(&data.map, 25).unwrap());
    let r = drive(&mut c, 0, |_, _| parse("MOVE S 16"));
    assert_eq!((r.result.as_str(), r.reason.as_str()), ("success", "reached route step 25 (level 0, room 0x4c)"));
    let rooms: Vec<_> = r.rooms.iter().map(|x| (x.room.as_str(), x.specialist.as_str(), x.outcome.as_str())).collect();
    assert_eq!(rooms, [("0x3c", "ow-octorok", "exit-correct"), ("0x4c", "-", "milestone")]);
    assert_eq!(c.context(), (Some(25), ItemCounts::default()));
    let end: Arc<[u8]> = Arc::from(common::encode_state(&c.session().snapshot()).unwrap());
    let (ram, prg_ram) = (*c.session().nes().ram(), *c.session().nes().prg_ram());

    // A wrong exit that lands in the stop step's room is not reaching the step: Dungeon 1's 0x72 (step 47, a key
    // to take) left at once the way Link came is 0x73 with step 48's progress, and the run goes on.
    let mut c = chain(c.into_parts().0.into_nes(), data, load(Some("1_72e_0")).with_stop_step(&data.map, 48).unwrap());
    assert!(matches!(c.reset(Some(0), &mut ChaCha8Rng::seed_from_u64(0)).unwrap(), ChainStatus::Act { .. }));
    assert!(matches!(c.step(parse("MOVE E 16")).unwrap(), ChainStatus::Act { new_room: true, .. }));
    let rooms: Vec<_> = c.result().unwrap().rooms.iter().map(|x| (x.room.as_str(), x.outcome.as_str())).collect();
    assert_eq!((rooms, c.context().0), (vec![("0x72", "exit-wrong"), ("0x73", "in-progress")], Some(48)));

    // What a route refuses: a stop step not after the start step or off the route, two ways to stop, two starts.
    let error = |route: Result<Route, String>| route.expect_err("refused");
    assert!(error(load(Some("0_3cc_0")).with_stop_step(&data.map, 24)).contains("after its start step 24"));
    assert!(error(load(None).with_stop_step(&data.map, 0)).contains("after its start step 0"));
    assert!(error(load(None).with_stop_step(&data.map, steps)).contains("is not a step of route 'd2first'"));
    let stop_at = Route::load(&data.map, &data.states_dir, "d2first", Some(MapLocation::new(1, 0x73, false)), None);
    assert!(error(stop_at.unwrap().with_stop_step(&data.map, 3)).contains("exclusive"));
    assert!(error(load(Some("0_3cc_0")).with_power_on("LINK", Some(0), 0)).contains("replaces the game start"));
    assert!(error(load(Some("0_3cc_0")).with_checkpoint(&data.map, end.clone(), 25, ItemCounts::default()))
        .contains("replaces the game start"));
    assert!(error(load(None).with_checkpoint(&data.map, end.clone(), steps, ItemCounts::default()))
        .contains("is not a step"));
    assert!(error(load(None).with_power_on("ZELDA", None, 0)).contains("second quest"));
    assert!(error(load(None).with_power_on("LINK", None, power_on::MAX_DELAY + 1)).contains("at most 32766"));
    let checkpoint = load(None).with_checkpoint(&data.map, end.clone(), 25, ItemCounts::default()).unwrap();
    assert!(error(checkpoint.clone().with_stop_step(&data.map, 25)).contains("after its start step 25"));

    // From the checkpoint: no frame runs and nothing is written, whatever the seed.
    let mut c = chain(c.into_parts().0.into_nes(), data, checkpoint.with_stop_step(&data.map, 26).unwrap());
    let acting = data.specialist_for(MapLocation::new(0, 0x4c, false)).unwrap();
    for seed in [5, 6] {
        let status = c.reset(Some(seed), &mut ChaCha8Rng::seed_from_u64(seed)).unwrap();
        let ChainStatus::Act { specialist, new_room, .. } = status else { panic!("run ended at reset") };
        assert_eq!((&*specialist, new_room), (acting, true));
        assert_eq!(c.session().frame(), 0, "the checkpoint runs no frame");
        assert!(c.session().nes().ram() == &ram && c.session().nes().prg_ram() == &prg_ram);
        assert_eq!(c.context().0, Some(25));
        let rooms = &c.result().unwrap().rooms;
        assert_eq!((rooms.len(), rooms[0].level, rooms[0].room.as_str()), (1, 0, "0x4c"));
    }
    // The run goes on from there (the stop step is not the room it starts in).
    assert!(matches!(c.step(parse("WAIT 8")).unwrap(), ChainStatus::Act { .. }));

    // With the RNG redrawn, the checkpoint is another game per seed (models acting by argmax would otherwise
    // play the same one every time): the game's RNG bytes ($18-$24) differ, and nothing else.
    let redrawn = load(None).with_checkpoint(&data.map, end.clone(), 25, ItemCounts::default()).unwrap();
    let mut c = chain(c.into_parts().0.into_nes(), data, redrawn.redrawing_rng(true));
    let mut draw = |seed: u64| {
        assert!(matches!(c.reset(Some(seed), &mut ChaCha8Rng::seed_from_u64(seed)).unwrap(), ChainStatus::Act { .. }));
        assert_eq!((c.session().frame(), c.context().0), (0, Some(25)));
        (c.session().nes().ram().to_vec(), c.session().nes().prg_ram().to_vec())
    };
    let ((five, prg_five), (again, _), (six, _)) = (draw(5), draw(5), draw(6));
    assert!(five == again && prg_five == prg_ram.to_vec(), "the same seed is the same game");
    assert_ne!(five[0x18..=0x24], six[0x18..=0x24], "another seed is another one");
    assert_ne!(five[0x18..=0x24], ram[0x18..=0x24]);
    let differ: Vec<usize> = (0..ram.len()).filter(|&at| five[at] != ram[at]).collect();
    assert!(differ.iter().all(|at| (0x18..=0x24).contains(at)), "only the RNG bytes are written: {differ:?}");

    // Bytes that are no state end the run as an error, like a start state that can't be loaded.
    let junk = load(None).with_checkpoint(&data.map, Arc::from(&b"not a state"[..]), 25, ItemCounts::default());
    let mut c = chain(c.into_parts().0.into_nes(), data, junk.unwrap());
    let ChainStatus::Done(r) = c.reset(Some(0), &mut ChaCha8Rng::seed_from_u64(0)).unwrap() else { panic!() };
    assert_eq!(r.result, "error");
    assert!(r.reason.contains("TFZSTATE"), "{}", r.reason);
    c.into_parts().0.into_nes()
}

/// The golden path's checkpoints: the step after a mid-route triforce is the overworld room Link comes out
/// in.  A run stopped on that step ends there with Link controllable, and its end state is a checkpoint.
fn a_stop_step_after_a_triforce_ends_the_run_on_the_overworld(nes: ZCore, data: &Arc<GameData>) -> ZCore {
    let steps = &data.map.routes["d2first"];
    let triforce = steps.iter().position(|s| s.exit == "triforce").unwrap();
    assert_eq!((steps[triforce].level, steps[triforce].room, triforce), (2, 0x0d, 23));
    let route = Route::load(&data.map, &data.states_dir, "d2first", None, Some("2_0de_0")).unwrap();
    let mut c = chain(nes, data, route.with_stop_step(&data.map, triforce + 1).unwrap());
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let ChainStatus::Act { .. } = c.reset(Some(0), &mut rng).unwrap() else { panic!("run ended at reset") };
    let r = walk_onto_the_triforce(&mut c).expect("taking the triforce ends the run outside the dungeon");
    assert_eq!((r.result.as_str(), r.reason.as_str()), ("success", "reached route step 24 (level 0, room 0x3c)"));
    let rooms: Vec<_> = r.rooms.iter().map(|x| (x.level, x.room.as_str(), x.outcome.as_str())).collect();
    assert_eq!(rooms, [(2, "0x0d", "exit-correct"), (0, "0x3c", "milestone")]);
    let state = c.session().state().clone();
    assert_eq!((c.session().nes().peek(0x12), state.level(), state.location()), (5, 0, 0x3c));
    assert!(state.has_triforce(2));
    assert_eq!(c.context().0, Some(triforce + 1));
    c.into_parts().0.into_nes()
}

/// A power-on start plays the boot inputs and nothing else: the machine is the one a bare emulator reaches
/// with `power_on::boot` (and control gained), on the route's first step; a random delay is the run seed's.
fn a_power_on_start_boots_the_game_by_inputs_alone(nes: ZCore, data: &Arc<GameData>) -> ZCore {
    let mut c = chain(nes, data, d2first(data).with_power_on("link", Some(0), 0).unwrap());
    let status = c.reset(Some(9), &mut ChaCha8Rng::seed_from_u64(9)).unwrap();
    let ChainStatus::Act { specialist, .. } = status else { panic!("run ended at reset") };
    assert_eq!(&*specialist, "ow-start");
    assert_eq!(c.booted(), Some(("LINK", 0)));
    assert_eq!(c.session().state().full_location(), MapLocation::new(0, 0x77, false));
    assert_eq!(c.context(), (Some(0), ItemCounts::default()));
    let mut bare = open_core().unwrap();
    power_on::boot(&mut bare, "LINK", 0).unwrap();
    let mut session = Session::new(bare, data.map.clone());
    let live = session.nes().snapshot();
    session.load(&live, &LoadOptions::default(), &mut ChaCha8Rng::seed_from_u64(0)).unwrap();
    assert!(session.nes().ram() == c.session().nes().ram(), "RAM");
    assert!(session.nes().prg_ram() == c.session().nes().prg_ram(), "PRG-RAM");

    // A delay left to chance is drawn from the run's generator: the same run for the same seed.
    let mut c = chain(c.into_parts().0.into_nes(), data, d2first(data).with_power_on("LINK", None, 50).unwrap());
    let mut boot = |seed: u64| {
        let status = c.reset(Some(seed), &mut ChaCha8Rng::seed_from_u64(seed)).unwrap();
        assert!(matches!(status, ChainStatus::Act { .. }));
        (c.booted().unwrap().1, c.session().nes().ram().to_vec())
    };
    let ((delay, ram), (again_delay, again), (other_delay, other)) = (boot(1), boot(1), boot(2));
    assert!(delay <= 50 && other_delay <= 50);
    assert!((delay, &ram) == (again_delay, &again), "seed 1 twice");
    assert_ne!(delay, other_delay, "seeds 1 and 2 draw different delays");
    assert_ne!(ram[0x18..=0x24], other[0x18..=0x24], "another delay is another RNG phase");
    c.into_parts().0.into_nes()
}

/// A chain's reward recording pays what training pays: from the same state, on the same route step, a
/// specialist env and a chain with rewards on score the same actions with the same terms, to the episode's end.
fn chain_rewards_are_the_training_rewards(nes: ZCore, data: &Arc<GameData>) -> ZCore {
    let location = MapLocation::new(0, 0x3c, false);
    let step = data.map.routes["d2first"].iter().position(|s| s.location() == location).unwrap();
    let mut compared = 0;
    let mut nes = Some(nes);
    for seed in 0..4u64 {
        // A controllable snapshot of 0x3c (where Link comes out of Dungeon 2).
        let mut session = Session::new(nes.take().unwrap(), data.map.clone());
        let start = common::load_named_state("0_3cc_0").unwrap();
        session.load(&start, &LoadOptions::default(), &mut ChaCha8Rng::seed_from_u64(seed)).unwrap();
        let snapshot = session.snapshot();
        assert_eq!(session.state().full_location(), location);
        let name = data.specialist_for(location).unwrap().to_owned();

        let opts = EnvOptions {
            explicit_starts: true,
            ..Default::default()
        };
        let config = data.specialist(&name).unwrap();
        let mut env = SpecialistEnv::new(config, data.clone(), session.into_nes(), TfzStore::new(), opts).unwrap();
        env.set_route("d2first", Some(step));
        env.restore(&snapshot, "0_3cc_0").unwrap();

        let state: Arc<[u8]> = Arc::from(common::encode_state(&snapshot).unwrap());
        let route = d2first(data).with_checkpoint(&data.map, state, step, ItemCounts::default()).unwrap();
        let mut c = chain(open_core().unwrap(), data, route);
        c.set_rewards(true);
        let status = c.reset(Some(seed), &mut ChaCha8Rng::seed_from_u64(seed)).unwrap();
        assert!(matches!(status, ChainStatus::Act { ref specialist, .. } if **specialist == *name));
        assert!(c.take_scored().is_none(), "nothing is scored before the first step");

        let mut rng = ChaCha8Rng::seed_from_u64(40 + seed);
        for _ in 0..40 {
            assert!(env.session().mask() == c.session().mask(), "the same legal actions");
            let action = random_action(env.session().mask(), &mut rng, None);
            let trained = env.step(action).unwrap();
            let status = c.step(action).unwrap();
            let scored = c.take_scored().expect("every step is scored");
            assert_eq!(scored.rewards.terms().collect::<Vec<_>>(), trained.info.rewards, "{action}");
            assert_eq!((scored.terminated, scored.truncated), (trained.terminated, trained.truncated));
            let ended = trained.info.terminal.as_ref().map(|t| (t.ending.clone(), t.outcome));
            assert_eq!(scored.ending.clone().zip(scored.outcome), ended);
            compared += 1;
            if ended.is_some() || matches!(status, ChainStatus::Done(_)) {
                break;
            }
        }
        nes = Some(env.into_parts().0.into_nes());
    }
    assert!(compared >= 40, "{compared} steps compared");
    nes.unwrap()
}

/// `test_explore_finds_replays_and_writes_a_curriculum_trajectory`: 0x38 from the north, the W exit past Octoroks
/// and the Zora; the written trajectory feeds a curriculum env.
#[test]
fn explore_finds_replays_and_writes_a_curriculum_trajectory() {
    let Some(nes) = open_core() else { return };
    let data = data();
    let tmp = tempfile::tempdir().unwrap();
    let config = data.specialists["ow-zora"].clone();
    let mut explorer = Explorer::new(&config, data.clone(), nes, TfzStore::new(), 0).unwrap();
    assert!(!explorer.env.session().look_ahead, "the explorer pays no weapon look-ahead (F176)");
    let result = explorer.explore("0_38n_0", 400, 25, 20, None).unwrap();
    assert!(result.success, "{result:?}");
    assert_eq!(result.ending.as_deref(), Some("exit-correct"));
    assert!(!result.actions.is_empty() && result.actions.len() as u64 <= result.env_steps);

    let dir = tmp.path().join("0_38n_0");
    assert!(
        explorer.write(&result, &dir).unwrap(),
        "replaying the actions reproduces the success"
    );
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("trajectory.json")).unwrap())
            .unwrap();
    assert_eq!(meta["success"], true);
    assert_eq!(meta["steps"], result.actions.len());
    let paths = &load_trajectories(tmp.path()).unwrap()["0_38n_0"];
    assert_eq!(paths.len(), result.actions.len());
    assert!(paths[0].ends_with("0000.state"));

    let (info, pre_entry) = explorer.replay(&result, true).unwrap();
    let info = info.unwrap();
    assert_eq!(info.ending(), Some("exit-correct"));
    assert!(pre_entry.is_some(), "the exit's room entry was captured");

    // Curriculum starts load trajectory states, and their episodes report the track.
    let nes = explorer.env.into_parts().0.into_nes();
    let spec = CurriculumSpec::new(
        tmp.path().to_path_buf(),
        tmp.path().join("run/curriculum.json"),
        1.0,
        0.6,
    );
    let curriculum = Curriculum::new(spec).unwrap().shared();
    let opts = EnvOptions {
        curriculum: Some(curriculum),
        ..Default::default()
    };
    let mut env = SpecialistEnv::new(config, data, nes, TfzStore::new(), opts).unwrap();
    let (_, info) = env.reset(Some(0), None).unwrap();
    let start = info.start_state.unwrap();
    let tail = &paths[paths.len() - (paths.len() / 10 + 1)..];
    assert!(
        tail.iter().any(|p| p.to_string_lossy() == start),
        "{start} not in {tail:?}"
    );
    let (west, wait) = (parse("MOVE W 16"), parse("WAIT 8"));
    let mut terminal = None;
    for _ in 0..200 {
        let action = if env.session().mask().is_allowed(&west) {
            west
        } else {
            wait
        };
        let step = env.step(action).unwrap();
        if step.terminated || step.truncated {
            terminal = step.info.terminal;
            break;
        }
    }
    let terminal = terminal.expect("episode ended");
    assert_eq!(terminal.curriculum.unwrap().track, "0_38n_0");
}

/// The moves a best-first search over Link's positions finds from the chain's current state into the neighbouring
/// room `target` (the session is put back where it was): positions nearer `target`'s side of the room first, so
/// the search doesn't flood the whole room before walking out.
fn walk_to<St: StateStore<MachineState>>(c: &mut ChainEnv<ZCore, St>, target: MapLocation) -> Vec<Action> {
    use std::cmp::Reverse;
    let here = c.session().state().full_location();
    let side = here.direction_to(target);
    let distance = |p: zelda::Position| match side {
        zelda::Direction::N => p.y,
        zelda::Direction::S => 255 - p.y,
        zelda::Direction::W => p.x,
        zelda::Direction::E => 255 - p.x,
        other => panic!("{target:?} is not beside {here:?} ({other:?})"),
    };
    let start = c.session().snapshot();
    let position = c.session().state().link.position;
    let mut nodes = vec![(start.clone(), Vec::new())];
    let mut queue = std::collections::BinaryHeap::from([Reverse((distance(position), 0usize))]);
    let mut seen = std::collections::HashSet::from([position]);
    while let Some(Reverse((_, node))) = queue.pop() {
        let (snap, actions) = nodes[node].clone();
        c.session_mut().restore(&snap);
        for d in c.session().mask().directions(zelda::ActionKind::Move) {
            let a = Action::new(zelda::ActionKind::Move, d, 4).unwrap();
            c.session_mut().restore(&snap);
            let hc = c.session().state().hearts_and_containers();
            c.session_mut().apply_ram(&[(zelda::Var::HeartsAndContainers, (hc & 0xF0) | (hc >> 4))]);
            let change = c.session_mut().act(a).unwrap();
            let mut next = actions.clone();
            next.push(a);
            if change.state.full_location() == target {
                c.session_mut().restore(&start);
                return next;
            }
            let p = change.state.link.position;
            if !change.changed_location() && seen.insert(p) {
                queue.push(Reverse((distance(p), nodes.len())));
                nodes.push((c.session().snapshot(), next));
            }
        }
    }
    panic!("no walk to {target:?} ({} positions reached)", seen.len());
}

/// F148: leaving 0x2c (a `requires_bombs` room) with a granted bomb outstanding: the harvested pre-entry state of
/// 0x1c holds the bombs the run really carried (0), not the granted one the live run takes back afterwards.
fn harvested_states_hold_no_granted_bomb(nes: ZCore, data: &Arc<GameData>) -> ZCore {
    let dir = tempfile::tempdir().unwrap();
    let mut source = common::read_state_file(&common::state_path("0_2cc_0")).unwrap();
    source.machine.poke(0x658, 0);
    let start = dir.path().join("0_2cc_0.state");
    TfzStore::new().save(&start, &source).unwrap();
    let route = Route::load(&data.map, &data.states_dir, "d2first", None, Some(start.to_str().unwrap())).unwrap();
    let mut c = chain(nes, data, route);
    let harvest = dir.path().join("harvested");
    c.set_harvest(Some(harvest.clone()));
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let ChainStatus::Act { .. } = c.reset(Some(0), &mut rng).unwrap() else { panic!("run ended at reset") };
    assert_eq!(c.session().state().bombs(), 1, "the room's granted bomb");
    let n1c = MapLocation::new(0, 0x1c, false);
    for a in walk_to(&mut c, n1c) {
        if let ChainStatus::Done(r) = c.step(a).unwrap() {
            panic!("run ended on the walk: {r:?}");
        }
    }
    assert_eq!(c.session().state().full_location(), n1c);
    assert_eq!(c.session().state().bombs(), 0, "taken back on leaving");
    let files: Vec<_> = glob_files(&harvest);
    assert_eq!(files.len(), 1, "{files:?}");
    let saved = common::read_state_file(&files[0]).unwrap();
    assert_eq!(saved.machine.peek(0x658), 0, "{}", files[0].display());
    c.into_parts().0.into_nes()
}

fn glob_files(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let p = e.path();
        if p.is_dir() {
            out.extend(glob_files(&p));
        } else {
            out.push(p);
        }
    }
    out
}

/// F040: a one_room specialist entering a room of its own family in a chain sees a reset observation, as at the
/// start of its training episodes: the history block is empty but the entry side.
fn one_room_entries_reset_the_observation(nes: ZCore, data: &Arc<GameData>) -> ZCore {
    let route = Route::load(&data.map, &data.states_dir, "d2first", None, Some("1_73s_0")).unwrap();
    let mut c = chain(nes, data, route);
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let ChainStatus::Act { .. } = c.reset(Some(0), &mut rng).unwrap() else { panic!("run ended at reset") };
    let west = MapLocation::new(1, 0x72, false);
    let walk = walk_to(&mut c, west);
    let mut entry = None;
    for a in walk {
        match c.step(a).unwrap() {
            ChainStatus::Done(r) => panic!("run ended on the walk: {r:?}"),
            ChainStatus::Act { specialist, obs, new_room } => {
                if new_room && c.session().state().full_location() == west {
                    assert_eq!(&*specialist, "stalfos-keese-gel");
                    entry = Some(obs);
                }
            }
        }
    }
    let obs = entry.expect("entered 0x72");
    let names = &env::observation::feature_names().history;
    for (name, &v) in names.iter().zip(obs.history.iter()) {
        if !name.starts_with("entered_from_") {
            assert_eq!(v, 0.0, "{name}");
        }
    }
    c.into_parts().0.into_nes()
}
