//! ChainEnv (`tests/test_runner.py`) and Explorer (`tests/test_explore.py`) on zcore with scripted
//! policies in place of networks.

mod common;

use common::{MachineState, TfzStore, ZCore, open_core};
use env::StateStore;
use env::chain::{ChainEnv, ChainStatus, RoomRecord, Route, RunResult, chain_summary};
use env::config::{GameData, SpecialistConfig};
use env::curriculum::{Curriculum, CurriculumSpec, load_trajectories};
use env::explore::{Explorer, random_action};
use env::specialist_env::{EnvOptions, SpecialistEnv};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;
use zelda::cheats::ItemCounts;
use zelda::{Action, GameMap, MapLocation, Session};

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

#[test]
fn chain_summary_counts_failures_by_room() {
    let run = |result: &str, room: &str, outcome: &str, bombs: u32, keys: u32| RunResult {
        route: "d2first".into(),
        seed: Some(0),
        result: result.into(),
        reason: String::new(),
        steps: 0,
        frames: 0,
        retreats: 0,
        wrong_exits: 0,
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
            bombs_used: bombs,
            cheated: ItemCounts {
                bombs,
                keys,
                rupees: 0,
            },
            boss_stuns: 0,
            boss_eats: 0,
        }],
    };
    let s = chain_summary(&[
        run("success", "0x0e", "milestone", 2, 0),
        run("failure", "0x44", "death", 0, 1),
        run("failure", "0x44", "death", 1, 1),
        run("failure", "0x73", "stuck", 0, 0),
    ]);
    assert_eq!(s["runs"], 4);
    assert_eq!(s["successes"], 1);
    assert_eq!(s["success_rate"], 0.25);
    assert_eq!(
        s["failures"],
        serde_json::json!({"1:0x44 stalfos-keese-gel death": 2, "1:0x73 stalfos-keese-gel stuck": 1})
    );
    let counts = |item: &str| {
        let n = |key: String| s[key.as_str()].as_u64();
        (n(format!("cheated_{item}")), n(format!("runs_with_cheated_{item}")))
    };
    assert_eq!(counts("bombs"), (Some(3), Some(2)));
    assert_eq!(counts("keys"), (Some(2), Some(2)));
    assert_eq!(counts("rupees"), (Some(0), Some(0)));
    assert!(s.get("bombs_cheated").is_none(), "the old name is gone");
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
        lynel_retreat_continues_the_run,
        cave_bounce_hits_the_wrong_exit_limit,
        mid_route_triforce_is_the_correct_exit,
        harvested_states_hold_no_granted_bomb,
        one_room_entries_reset_the_observation,
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
    assert_eq!(chain_summary(&[r])["successes"], 0);
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

/// F135: taking a triforce the route takes before its last (Dungeon 2's in 2:0x0d) is that room's correct exit,
/// and the handoff it causes (0:0x3c out of the dungeon) is harvested.
fn mid_route_triforce_is_the_correct_exit(nes: ZCore, data: &Arc<GameData>) -> ZCore {
    let route = Route::load(&data.map, &data.states_dir, "d2first", None, Some("2_0de_0")).unwrap();
    let mut c = chain(nes, data, route);
    let dir = tempfile::tempdir().unwrap();
    c.set_harvest(Some(dir.path().to_path_buf()));
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let ChainStatus::Act { .. } = c.reset(Some(0), &mut rng).unwrap() else { panic!("run ended at reset") };
    // A breadth-first search over Link's tiles finds the walk onto the triforce; the chain then plays it.
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
            panic!("run ended on the walk: {r:?}");
        }
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

/// The moves a breadth-first search over Link's tiles finds from the chain's current state to the first step
/// `goal` accepts (the session is put back where it was).
fn bfs_walk<St: StateStore<MachineState>>(
    c: &mut ChainEnv<ZCore, St>,
    goal: impl Fn(&zelda::Change) -> bool,
) -> Vec<Action> {
    let start = c.session().snapshot();
    let mut queue = std::collections::VecDeque::from([(start.clone(), Vec::new())]);
    let mut seen = std::collections::HashSet::from([c.session().state().link.position]);
    while let Some((snap, actions)) = queue.pop_front() {
        c.session_mut().restore(&snap);
        for d in c.session().mask().directions(zelda::ActionKind::Move) {
            let a = Action::new(zelda::ActionKind::Move, d, 4).unwrap();
            c.session_mut().restore(&snap);
            let hc = c.session().state().hearts_and_containers();
            c.session_mut().apply_ram(&[(zelda::Var::HeartsAndContainers, (hc & 0xF0) | (hc >> 4))]);
            let change = c.session_mut().act(a).unwrap();
            let mut next = actions.clone();
            next.push(a);
            if goal(&change) {
                c.session_mut().restore(&start);
                return next;
            }
            if !change.changed_location() && seen.insert(change.state.link.position) {
                queue.push_back((c.session().snapshot(), next));
            }
        }
    }
    panic!("no walk to the goal ({} tiles reached)", seen.len());
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
    for a in bfs_walk(&mut c, |ch| ch.state.full_location() == n1c) {
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
    let walk = bfs_walk(&mut c, |ch| ch.state.full_location() == west);
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
