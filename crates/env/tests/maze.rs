//! The Lost Hills (overworld 0x1b, asm:Maze_CheckStep) played through the route (round 24): the scouting route
//! `d5maze` leaves 0x1b entered from E by `exit: maze`, N four times in a row into 0x0b entered from S. Each of the
//! first three Ns loops back into 0x1b (`zelda::MazeLoop`), a room transition for the objective, the endings, the
//! rewards, the observation, linger and the chain's records, though the room id never changes; a wrong edge resets
//! the sequence and is a wrong exit of the room; W is the free edge out to 0x1a.
//!
//! The runs start from `0_1be_0` holding what the route has there (the D4 triforce's progress: triforces 1-4, the
//! white sword, the bow, the raft and the ladder), written into a copy of the state: no played state of the leg to
//! Dungeon 5 exists yet, and `d2first`'s own 0x1b step (W) is picked for the state as it is.

mod common;

use common::{MachineState, TfzStore, ZCore, open_core};
use env::chain::{ChainEnv, ChainStatus, Route};
use env::config::{GameData, RamRandomization};
use env::explore::{Explorer, cell_key, maze_step, progress_events};
use env::observation::{Observation, feature_names};
use env::specialist_env::{EnvOptions, SpecialistEnv};
use env::StateStore;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::{BTreeSet, BinaryHeap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use zelda::game_map::MAZE_STEP;
use zelda::{Action, ActionKind, Change, Direction, GameView, MapLocation, Session};

const HILLS: MapLocation = MapLocation::new(0, 0x1b, false);
const TOP: MapLocation = MapLocation::new(0, 0x0b, false);
const D5: MapLocation = MapLocation::new(5, 0x76, false);

/// `name` (a d2first state) with the progress the route has after the D4 triforce, as `<dir>/<name>.state`.
fn leg_state(dir: &Path, name: &str) -> PathBuf {
    let mut state = common::read_state_file(&common::state_path(name)).unwrap();
    // Var::Triforce (levels 1-4), Var::Sword (white), Var::Bow, Var::Raft, Var::StepLadder.
    for (addr, value) in [(0x671, 0x0F), (0x657, 2), (0x65A, 1), (0x660, 1), (0x663, 1)] {
        state.machine.poke(addr, value);
    }
    let path = dir.join(format!("{name}.state"));
    TfzStore::new().save(&path, &state).unwrap();
    path
}

/// The index of `d5maze`'s step in `room` entered from `entry`.
fn d5maze_step(data: &GameData, room: MapLocation, entry: &str) -> usize {
    let steps = &data.map.routes["d5maze"];
    let first = steps.len() - 6;
    first + steps[first..].iter().position(|s| s.location() == room && s.entry == entry).unwrap()
}

/// MOVEs a best-first search over Link's positions finds from the session's state to `done` (positions nearer
/// `distance`'s zero first); it stops at the first room transition, so a search never runs through a loop it doesn't
/// want. The session is put back where it was.
fn search(
    s: &mut Session<ZCore>,
    distance: impl Fn(&GameView) -> i32,
    done: impl Fn(&Change) -> bool,
) -> Vec<Action> {
    let start = s.snapshot();
    let mut nodes = vec![(start.clone(), Vec::new())];
    let mut queue = BinaryHeap::from([std::cmp::Reverse((distance(s.state()), 0usize))]);
    let mut seen = HashSet::from([s.state().link.position]);
    while let Some(std::cmp::Reverse((_, node))) = queue.pop() {
        let (snap, actions) = nodes[node].clone();
        s.restore(&snap);
        for d in s.mask().directions(ActionKind::Move) {
            let a = Action::new(ActionKind::Move, d, 4).unwrap();
            s.restore(&snap);
            let change = s.act(a).unwrap();
            let mut next = actions.clone();
            next.push(a);
            if done(&change) {
                s.restore(&start);
                return next;
            }
            let p = change.state.link.position;
            if !change.entered_room() && !change.state.game_over() && seen.insert(p) {
                queue.push(std::cmp::Reverse((distance(&change.state), nodes.len())));
                nodes.push((s.snapshot(), next));
            }
        }
    }
    panic!("no walk found ({} positions)", seen.len());
}

/// The walk out of Link's room through its `edge` (a loop, or the room the edge leads to).
fn walk_out(s: &mut Session<ZCore>, edge: Direction) -> Vec<Action> {
    let distance = move |v: &GameView| {
        let p = v.link.position;
        match edge {
            Direction::N => p.y,
            Direction::S => 255 - p.y,
            Direction::W => p.x,
            _ => 255 - p.x,
        }
    };
    search(s, distance, move |c| c.maze_loop().is_some_and(|l| l.direction == edge) || c.changed_location())
}

fn history(obs: &Observation, name: &str) -> f32 {
    obs.history[feature_names().history.iter().position(|n| n == name).unwrap()]
}

/// The observation's `maze_step` (room_state): `MazeStep` / 4 in the Lost Hills, 0 elsewhere.
fn maze_feature(obs: &Observation) -> f32 {
    obs.room_state[feature_names().room_state.iter().position(|n| n == "maze_step").unwrap()]
}

type Chain = ChainEnv<ZCore, TfzStore>;

/// A `d5maze` chain run from `start` to Dungeon 5's first room, reset with seed 0.
fn chain(nes: ZCore, data: &Arc<GameData>, start: &Path) -> Chain {
    let route = Route::load(&data.map, &data.states_dir, "d5maze", Some(D5), Some(start.to_str().unwrap())).unwrap();
    let mut c = ChainEnv::new(data.clone(), nes, TfzStore::new(), route);
    let status = c.reset(Some(0), &mut ChaCha8Rng::seed_from_u64(0)).unwrap();
    assert!(matches!(status, ChainStatus::Act { .. }), "{status:?}");
    c
}

/// Plays `actions` in the chain; the last status.
fn play(c: &mut Chain, actions: &[Action]) -> ChainStatus {
    let mut last = None;
    for (i, &a) in actions.iter().enumerate() {
        let status = c.step(a).unwrap();
        if let ChainStatus::Done(r) = &status {
            assert_eq!(i, actions.len() - 1, "the run ended early: {r:?}");
        }
        last = Some(status);
    }
    last.expect("an action")
}

/// The four Ns from 0x1b entered from E (MOVE actions only), in a chain of `d5maze`: each of the first three is a
/// correct loop (the objective still leads N into 0x0b, the observation says the room changed and Link came in from
/// S, the visit's record counts it), the fourth enters 0x0b from S (the record's exit is correct, ow-leever takes
/// over), and down 0x0b's cave mouth the run reaches Dungeon 5 (its stop room). Whether 0x0b's own objective leads
/// into the cave (a route that holds no later triforce) is the leg's business (`r24-leg`), not checked here.
#[test]
fn four_norths_through_the_lost_hills_reach_dungeon_5() {
    let Some(nes) = open_core() else { return };
    let data = common::game_data();
    let dir = tempfile::tempdir().unwrap();
    let mut c = chain(nes, &data, &leg_state(dir.path(), "0_1be_0"));
    assert_eq!(c.context().0, Some(d5maze_step(&data, HILLS, "E")), "the maze step, not d2first's W");
    assert_eq!(c.objective().unwrap().next_rooms, BTreeSet::from([TOP]));
    for k in 1..=3u8 {
        let walk = walk_out(c.session_mut(), Direction::N);
        let ChainStatus::Act { obs, new_room, .. } = play(&mut c, &walk) else { panic!("run ended") };
        let state = c.session().state().clone();
        assert_eq!((state.full_location(), state.peek(MAZE_STEP)), (HILLS, k), "loop {k}");
        assert!(!new_room, "a loop continues the visit");
        assert_eq!((history(&obs, "room_changed"), history(&obs, "entered_from_S")), (1.0, 1.0), "loop {k}");
        assert_eq!(maze_feature(&obs), k as f32 / 4.0, "the observation tells loop {k} from the others");
        assert_eq!(c.objective().unwrap().next_rooms, BTreeSet::from([TOP]), "loop {k}");
        let record = c.result().unwrap().rooms.last().unwrap().clone();
        assert_eq!((record.room.as_str(), record.maze_loops, record.maze_wrong_loops), ("0x1b", k as u32, 0));
        assert_eq!(c.result().unwrap().wrong_exits, 0);
    }
    let walk = walk_out(c.session_mut(), Direction::N);
    let ChainStatus::Act { specialist, new_room, obs } = play(&mut c, &walk) else { panic!("run ended") };
    assert_eq!((c.session().state().full_location(), &*specialist, new_room), (TOP, "ow-leever", true));
    assert_eq!(maze_feature(&obs), 0.0, "0x0b is no maze, though MazeStep stays 3");
    assert_eq!(c.context().0, Some(d5maze_step(&data, TOP, "S")));
    let rooms = &c.result().unwrap().rooms;
    let hills = &rooms[rooms.len() - 2];
    assert_eq!((hills.outcome.as_str(), hills.maze_loops, hills.maze_wrong_loops), ("exit-correct", 3, 0));
    assert_eq!(hills.to_json()["maze_loops"], 3);
    // Down 0x0b's cave mouth (game.yaml `entrance.tile` [14, 8], the room's cave tile) into Dungeon 5.
    let mouth = c.session().state().room().cave_tile.expect("0x0b's cave mouth");
    assert_eq!(mouth, (14, 8));
    let distance = move |v: &GameView| {
        let t = v.link.tile();
        (mouth.0 - t.0).abs() + (mouth.1 - t.1).abs()
    };
    let walk = search(c.session_mut(), distance, |c| c.state.full_location() == D5);
    let ChainStatus::Done(r) = play(&mut c, &walk) else { panic!("the run goes on in {:?}", c.session().state().full_location()) };
    assert_eq!((r.result.as_str(), r.reason.as_str()), ("success", "entered (level 5, room 0x76)"));
    let rooms: Vec<_> = r.rooms.iter().map(|x| (x.room.as_str(), x.maze_loops)).collect();
    assert_eq!(&rooms[rooms.len() - 3..], [("0x1b", 3), ("0x0b", 0), ("0x76", 0)]);
}

/// A wrong edge on the second loop (S: the Lost Hills' W is its free edge, out to 0x1a) is a wrong loop: the record
/// and the run count it as a wrong exit of 0x1b, `MazeStep` is back at 0, the objective still leads N and four more Ns
/// reach 0x0b. W leaves the room as any wrong exit does.
#[test]
fn a_wrong_edge_restarts_the_sequence() {
    let Some(nes) = open_core() else { return };
    let data = common::game_data();
    let dir = tempfile::tempdir().unwrap();
    let start = leg_state(dir.path(), "0_1be_0");
    let mut c = chain(nes, &data, &start);
    let walk = walk_out(c.session_mut(), Direction::N);
    play(&mut c, &walk);
    assert_eq!(c.session().state().peek(MAZE_STEP), 1);
    let walk = walk_out(c.session_mut(), Direction::S);
    let ChainStatus::Act { obs, .. } = play(&mut c, &walk) else { panic!("run ended") };
    let state = c.session().state().clone();
    assert_eq!((state.full_location(), state.peek(MAZE_STEP)), (HILLS, 0));
    assert_eq!(history(&obs, "entered_from_N"), 1.0, "a S loop brings Link in at the top");
    assert_eq!(c.objective().unwrap().next_rooms, BTreeSet::from([TOP]));
    let record = c.result().unwrap().rooms.last().unwrap().clone();
    assert_eq!((record.maze_loops, record.maze_wrong_loops, record.outcome.as_str()), (1, 1, "in-progress"));
    assert_eq!(c.result().unwrap().wrong_exits, 1);
    for k in 1..=4u8 {
        let walk = walk_out(c.session_mut(), Direction::N);
        play(&mut c, &walk);
        let state = c.session().state().clone();
        let want = if k < 4 { (HILLS, k) } else { (TOP, 3) };
        assert_eq!((state.full_location(), state.peek(MAZE_STEP)), want, "N {k} after the reset");
    }
    let rooms = &c.result().unwrap().rooms;
    let hills = &rooms[rooms.len() - 2];
    assert_eq!((hills.outcome.as_str(), hills.maze_loops, hills.maze_wrong_loops), ("exit-correct", 4, 1));

    let mut c = chain(c.into_parts().0.into_nes(), &data, &start);
    let walk = walk_out(c.session_mut(), Direction::W);
    play(&mut c, &walk);
    assert_eq!(c.session().state().full_location(), MapLocation::new(0, 0x1a, false));
    let rooms = &c.result().unwrap().rooms;
    assert_eq!((rooms[0].outcome.as_str(), rooms[0].maze_wrong_loops, c.result().unwrap().wrong_exits), ("exit-wrong", 0, 1));
}

type Env = SpecialistEnv<ZCore, TfzStore>;

/// ow-start's env on 0x1b alone (one_room), episodes from `start`.
fn hills_env(nes: ZCore, data: &Arc<GameData>) -> Env {
    let mut config = (*data.specialists["ow-start"]).clone();
    config.ram_randomization = RamRandomization::default();
    config.one_room = true;
    let opts = EnvOptions { explicit_starts: true, seed: 0, ..Default::default() };
    SpecialistEnv::new(Arc::new(config), data.clone(), nes, TfzStore::new(), opts).unwrap()
}

/// A single-room training env on 0x1b: the first three Ns go on (each pays as a new room, no PBRS fall), the fourth
/// ends the episode `exit-correct`; a wrong edge ends it `exit-wrong`. With d2first's progress (the state as it is)
/// the step leaves W, and an N loop is a wrong exit too.
#[test]
fn a_one_room_episode_on_0x1b_ends_correct_only_after_the_fourth_north() {
    let Some(nes) = open_core() else { return };
    let data = common::game_data();
    let dir = tempfile::tempdir().unwrap();
    let start = leg_state(dir.path(), "0_1be_0");
    let start = start.to_str().unwrap();
    let mut env = hills_env(nes, &data);
    let (obs, _) = env.reset(Some(0), Some(start)).unwrap();
    assert_eq!(maze_feature(&obs), 0.0, "entered from E: step 0");
    assert_eq!(env.route(), ("d5maze", Some(d5maze_step(&data, HILLS, "E"))));
    for k in 1..=4 {
        let walk = walk_out(env.session_mut(), Direction::N);
        let (last, steps) = walk.split_last().unwrap();
        for &a in steps {
            assert!(!env.step(a).unwrap().terminated);
        }
        let step = env.step(*last).unwrap();
        let rewards: Vec<&str> = step.info.rewards.iter().map(|(n, _)| *n).collect();
        assert!(rewards.contains(&"reward-new-location"), "N {k}: {rewards:?}");
        assert!(!rewards.contains(&"penalty-pbrs-movement"), "N {k}: {rewards:?}");
        if k < 4 {
            assert!(!step.terminated && !step.truncated, "N {k}: {:?}", step.info.ending());
        } else {
            assert!(step.terminated);
            assert_eq!((step.info.ending(), step.info.outcome()), (Some("exit-correct"), Some("success")));
        }
    }
    env.reset(Some(1), Some(start)).unwrap();
    let walk = walk_out(env.session_mut(), Direction::N);
    for &a in &walk {
        env.step(a).unwrap();
    }
    let walk = walk_out(env.session_mut(), Direction::E);
    let step = walk.iter().map(|&a| env.step(a).unwrap()).last().unwrap();
    assert_eq!((step.terminated, step.info.ending()), (true, Some("exit-wrong")));
    let rewards: Vec<&str> = step.info.rewards.iter().map(|(n, _)| *n).collect();
    assert!(rewards.contains(&"penalty-wrong-location") && !rewards.contains(&"reward-new-location"), "{rewards:?}");

    env.reset(Some(2), Some("0_1be_0")).unwrap();
    assert_eq!(env.route().0, "d2first");
    let walk = walk_out(env.session_mut(), Direction::N);
    let step = walk.iter().map(|&a| env.step(a).unwrap()).last().unwrap();
    assert_eq!((step.terminated, step.info.ending()), (true, Some("exit-wrong")), "d2first leaves 0x1b W");
}

/// Go-Explore from 0x1b entered from E (the leg's progress) finds the four Ns unaided (the maze step is in its cells
/// and a correct loop is progress), and the trajectory it writes replays to the same success.
#[test]
fn the_explorer_finds_the_way_through_the_lost_hills() {
    let Some(nes) = open_core() else { return };
    let data = common::game_data();
    let dir = tempfile::tempdir().unwrap();
    let start = leg_state(dir.path(), "0_1be_0");
    let start = start.to_str().unwrap();
    let mut config = (*data.specialists["ow-start"]).clone();
    config.one_room = true;
    let mut explorer = Explorer::new(&config, data.clone(), nes, TfzStore::new(), 0).unwrap();
    let result = explorer.explore(start, 4000, 25, 200, None).unwrap();
    assert!(result.success, "{} iterations, {} cells", result.iterations, result.cells);
    assert_eq!(result.ending.as_deref(), Some("exit-correct"));
    let track = dir.path().join("track");
    assert!(explorer.write(&result, &track).unwrap(), "the trajectory replays");
    assert!(track.join(env::curriculum::TRAJECTORY_FILE).is_file());
    assert_eq!(explorer.env.session().state().full_location(), TOP);
}

/// The cell key and the progress count see the maze step: the same spot of 0x1b after one N and after none are two
/// cells, and a correct loop is progress; outside a maze the step reads 0.
#[test]
fn explorer_cells_tell_the_loops_apart() {
    let Some(nes) = open_core() else { return };
    let data = common::game_data();
    let dir = tempfile::tempdir().unwrap();
    let start = leg_state(dir.path(), "0_1be_0");
    let mut env = hills_env(nes, &data);
    env.reset(Some(0), Some(start.to_str().unwrap())).unwrap();
    let before = env.session().state().clone();
    let walk = walk_out(env.session_mut(), Direction::N);
    for &a in &walk {
        env.step(a).unwrap();
    }
    let after = env.session().state().clone();
    assert_eq!((maze_step(&before), maze_step(&after)), (0, 1));
    assert_eq!(progress_events(&before, &after), 1);
    assert_ne!(cell_key(&before).11, cell_key(&after).11);
    let mut machine: MachineState = common::load_named_state("0_1cs_0").unwrap();
    machine.poke(MAZE_STEP, 2);
    let s = env.session_mut();
    s.restore(&zelda::SessionSnapshot { machine, was_in_cave: false, just_exited_cave: false, total_frames: 0, steps: 0 });
    assert_eq!(maze_step(s.state()), 0, "MazeStep means nothing outside a maze");
}

/// Random play in 0x1b: every room transition that keeps the room is a loop through the edge Link left by (he stands
/// at the opposite edge after it), every loop that raised `MazeStep` matched the sequence (N), and no other step is a
/// loop.
#[test]
fn loops_are_exactly_the_room_transitions_in_the_maze() {
    let Some(nes) = open_core() else { return };
    let data = common::game_data();
    let dir = tempfile::tempdir().unwrap();
    let start = leg_state(dir.path(), "0_1be_0");
    let mut env = hills_env(nes, &data);
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let (mut loops, mut steps) = (0, 0);
    for episode in 0..20 {
        env.reset(Some(episode), Some(start.to_str().unwrap())).unwrap();
        for _ in 0..200 {
            let action = env::explore::random_action(env.session().mask(), &mut rng, None);
            let s = env.session_mut();
            let snap = s.snapshot();
            let change = s.act(action).unwrap();
            s.restore(&snap);
            steps += 1;
            if let Some(l) = change.maze_loop() {
                loops += 1;
                let p = change.state.link.position;
                let at_far_edge = match l.direction {
                    Direction::N => p.y >= 0xD5,
                    Direction::S => p.y <= 0x45,
                    Direction::E => p.x <= 0x08,
                    _ => p.x >= 0xE8,
                };
                assert!(at_far_edge, "{l:?} at {p:?}");
                assert_eq!(l.advanced, l.direction == Direction::N, "{l:?}");
            } else {
                assert!(change.changed_location() || change.interrupt != Some(zelda::Interrupt::Transition));
            }
            let r = env.step(action).unwrap();
            if r.terminated || r.truncated {
                break;
            }
        }
    }
    assert!(loops > 0, "{steps} random steps made no loop");
}
