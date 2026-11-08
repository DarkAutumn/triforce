//! ChainEnv (`tests/test_runner.py`) and Explorer (`tests/test_explore.py`) on zcore with scripted
//! policies in place of networks.

mod common;

use common::{MachineState, MemStore, TfzStore, ZCore, open_core};
use env::StateStore;
use env::chain::{ChainEnv, ChainStatus, RoomRecord, Route, RunResult, chain_summary};
use env::config::{GameData, SpecialistConfig};
use env::curriculum::{Curriculum, CurriculumSpec, load_trajectories};
use env::explore::{Explorer, random_action};
use env::specialist_env::{EnvOptions, SpecialistEnv};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;
use zelda::{Action, GameMap, LoadOptions, MapLocation, Session, Var};

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

fn milestone1(data: &GameData) -> Route {
    Route::load(&data.map, "milestone1", None, None).unwrap()
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
fn milestone_route_definition() {
    let map = GameMap::builtin();
    let route = Route::load(&map, "milestone1", None, None).unwrap();
    assert_eq!(route.start_state, "start");
    assert_eq!(route.terminal, Some(MapLocation::new(2, 0x0e, false)));
    let stop = MapLocation::new(1, 0x73, false);
    assert_eq!(
        Route::load(&map, "milestone1", Some(stop), None)
            .unwrap()
            .terminal,
        Some(stop)
    );
    assert!(Route::load(&map, "no-such-route", None, None).is_err());
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

#[test]
fn chain_summary_counts_failures_by_room() {
    let run = |result: &str, room: &str, outcome: &str, cheated: u32| RunResult {
        route: "milestone1".into(),
        seed: Some(0),
        result: result.into(),
        reason: String::new(),
        steps: 0,
        frames: 0,
        rooms: vec![RoomRecord {
            index: 0,
            level: 1,
            room: room.into(),
            in_cave: false,
            specialist: "stalfos-keese-gel".into(),
            steps: 0,
            frames: 0,
            health_in: 0.0,
            health_out: 0.0,
            max_health: 0,
            keys_in: 0,
            keys_out: 0,
            outcome: outcome.into(),
            hearts_dropped: 0,
            hearts_taken: 0,
            hearts_healed: 0.0,
            hearts_wasted: 0.0,
            hearts_left: 0,
            hearts_expired: 0,
            linger: None,
            blocks: 0,
            no_damage_max: 0,
            bombs_used: cheated,
            bombs_cheated: cheated,
            boss_stuns: 0,
            boss_eats: 0,
        }],
    };
    let s = chain_summary(&[
        run("success", "0x0e", "milestone", 2),
        run("failure", "0x44", "death", 0),
        run("failure", "0x44", "death", 1),
        run("failure", "0x73", "stuck", 0),
    ]);
    assert_eq!(s["runs"], 4);
    assert_eq!(s["successes"], 1);
    assert_eq!(s["success_rate"], 0.25);
    assert_eq!(
        s["failures"],
        serde_json::json!({"1:0x44 stalfos-keese-gel death": 2, "1:0x73 stalfos-keese-gel stuck": 1})
    );
    assert_eq!((s["bombs_cheated"].as_u64(), s["runs_with_cheated_bombs"].as_u64()), (Some(3), Some(2)));
}

/// The chain tests, run in sequence on one emulator.
#[test]
fn chain_runs() {
    let Some(mut nes) = open_core() else { return };
    let data = data();
    for case in [
        untagged_room_stops_with_clear_error,
        wrong_exit_into_untagged_room_is_a_failure,
        missing_model_is_an_error,
        run_records_rooms_and_ends_cleanly,
        same_seed_same_run,
        no_room_progress_threshold_replaces_flat_budget,
        no_room_progress_preserves_global_guard_and_disabled_budget,
        room_exit_resets_visit_clock,
        step_budget_records_health_after_native_fairy_healing,
        target_observation_follows_the_acting_specialist,
    ] {
        nes = case(nes, &data);
    }
}

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
    let mut c = chain(nes, &untagged, milestone1(data));
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
    let mut c = chain(nes, data, milestone1(data));
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
    let mut c = chain(nes, data, milestone1(data));
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
    assert!(r.reason.contains("ow-start"));
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
        let mut c = chain(nes, data, milestone1(data));
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
    let mut c = chain(nes, data, milestone1(data));
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
    let mut c = chain(nes, data, milestone1(data));
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
    let mut c = chain(nes, data, milestone1(data));
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
    assert!(
        r.reason.contains("no-room-progress") && r.reason.contains("4 quiet actions"),
        "{}",
        r.reason
    );
    let linger = last.linger.unwrap();
    assert_eq!(linger.tail_steps, 4);
    assert_eq!(linger.tail_frames as u64, last.frames);
    assert_eq!(chain_summary(&[r])["successes"], 0);
    c.into_parts().0.into_nes()
}

fn no_room_progress_preserves_global_guard_and_disabled_budget(
    mut nes: ZCore,
    data: &Arc<GameData>,
) -> ZCore {
    for enabled in [false, true] {
        let mut c = chain(nes, data, milestone1(data));
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
        let reason = if enabled {
            "run exceeded 5 steps"
        } else {
            "ow-start exceeded 3 steps"
        };
        assert_eq!(r.reason, reason);
        assert_eq!(
            r.rooms.last().unwrap().linger.unwrap().tail_steps as u64,
            r.steps
        );
        nes = c.into_parts().0.into_nes();
    }
    nes
}

fn room_exit_resets_visit_clock(nes: ZCore, data: &Arc<GameData>) -> ZCore {
    let route = Route::load(&data.map, "milestone1", None, Some("0_77cave_0")).unwrap();
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

fn step_budget_records_health_after_native_fairy_healing(
    mut nes: ZCore,
    data: &Arc<GameData>,
) -> ZCore {
    for room_budget in [true, false] {
        // A pond state at one heart of four.
        let mut session = Session::new(nes, data.map.clone());
        let state = TfzStore::new().load("0_39s_2").unwrap();
        let opts = LoadOptions {
            randomize_rng: false,
            ..Default::default()
        };
        session
            .load(&state, &opts, &mut ChaCha8Rng::seed_from_u64(0))
            .unwrap();
        session.apply_ram(&[(Var::HeartsAndContainers, 0x30), (Var::PartialHearts, 0xFF)]);
        let store = MemStore::default();
        store
            .0
            .borrow_mut()
            .insert("0_39s_2".into(), zelda::Nes::snapshot(session.nes()));
        nes = session.into_nes();

        let route = Route::load(&data.map, "milestone1-fairy", None, Some("0_39s_2")).unwrap();
        let mut c = ChainEnv::new(data.clone(), nes, store, route);
        c.min_room_steps = if room_budget { 12 } else { 100 };
        c.max_steps = if room_budget { 100 } else { 12 };
        c.configs.insert(
            "ow-start".into(),
            config(data, "ow-start", |c| {
                c.no_progress_steps = None;
                c.max_steps = if room_budget { 12 } else { 100 };
            }),
        );
        let (north, wait) = (parse("MOVE N 8"), parse("WAIT 8"));
        let r = drive(&mut c, 0, |s, _| {
            if s.mask().is_allowed(&north) {
                north
            } else {
                wait
            }
        });
        let record = r.rooms.last().unwrap();
        assert_eq!(
            (r.result.as_str(), record.outcome.as_str(), r.steps),
            ("failure", "stuck", 12)
        );
        let state = c.session().state();
        assert_eq!((state.health(), state.max_health()), (4.0, 4));
        assert_eq!((record.health_in, record.health_out), (1.0, 4.0));
        nes = c.into_parts().0.into_nes();
    }
    nes
}

/// `test_explore_finds_replays_and_writes_a_curriculum_trajectory`: 0x38 from the south, the W exit ~15 steps
/// away past Octoroks and the Zora; the written trajectory feeds a curriculum env.
#[test]
fn explore_finds_replays_and_writes_a_curriculum_trajectory() {
    let Some(nes) = open_core() else { return };
    let data = data();
    let tmp = tempfile::tempdir().unwrap();
    let config = data.specialists["ow-zora"].clone();
    let mut explorer = Explorer::new(&config, data.clone(), nes, TfzStore::new(), 0).unwrap();
    let result = explorer.explore("0_38s_0", 400, 25, 20, None).unwrap();
    assert!(result.success, "{result:?}");
    assert_eq!(result.ending.as_deref(), Some("exit-correct"));
    assert!(!result.actions.is_empty() && result.actions.len() as u64 <= result.env_steps);

    let dir = tmp.path().join("0_38s_0");
    assert!(
        explorer.write(&result, &dir).unwrap(),
        "replaying the actions reproduces the success"
    );
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("trajectory.json")).unwrap())
            .unwrap();
    assert_eq!(meta["success"], true);
    assert_eq!(meta["steps"], result.actions.len());
    let paths = &load_trajectories(tmp.path()).unwrap()["0_38s_0"];
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
    assert_eq!(terminal.curriculum.unwrap().track, "0_38s_0");
}
