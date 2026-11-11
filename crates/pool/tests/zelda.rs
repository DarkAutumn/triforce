//! The Zelda adapters in a pool, on zcore with the regenerated `TFZSTATE` start states in `triforce/states`
//! (skipped when the ROM is missing).

use env::chain::{ChainEnv, ChainStatus, Route, RunResult};
use env::config::{GameData, HpBound, RamRandomization};
use env::explore::random_action;
use env::observation::Observation;
use env::specialist_env::{EnvOptions, SpecialistEnv};
use env::{EnvError, StateStore};
use pool::rollout::{Actor, Decision, EnvRollout, Record};
use pool::zelda::{
    ACTION_MASK, ChainRunEnv, EvalEnv, EvalJob, TrainEnv, ZeldaAction, ZeldaEvent, interrupt_code, obs_layout,
    write_observation,
};
use parking_lot::Mutex;
use pool::{Column, EnvControl, EnvInit, Jobs, ObsData, Pool, PoolEnv, PoolOptions, SearchEnv};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use replay::{Event, Player, RecordOptions, Replay, Source, StateKind};
use std::sync::{Arc, LazyLock};
use zcore::{MachineState, Rom, ZCore, state_file};
use zelda::{ActionKind, ActionMask, MASK_SIZE, Nes, SessionSnapshot};

/// zcore as a `zelda::Nes`; `.1` counts every frame run (look-ahead included; restores don't rewind it).
struct Counted(ZCore, u64);

impl Nes for Counted {
    type State = MachineState;

    fn run_frame(&mut self, buttons: u8, render: bool) {
        self.0.run_frame(buttons, render);
        self.1 += 1;
    }

    fn ram(&self) -> &[u8; 0x800] {
        self.0.ram()
    }

    fn prg_ram(&self) -> &[u8; 0x2000] {
        self.0.prg_ram()
    }

    fn poke(&mut self, addr: u16, value: u8) {
        self.0.poke(addr, value);
    }

    fn snapshot(&self) -> MachineState {
        self.0.snapshot()
    }

    fn restore(&mut self, state: &MachineState) {
        self.0.restore(state);
    }

    fn poke_state(state: &mut MachineState, addr: u16, value: u8) {
        state.poke(addr, value);
    }
}

/// Palette indices of the last rendered frame as gray RGB (false when it wasn't rendered).
fn rgb(nes: &Counted, out: &mut [u8]) -> bool {
    let Some(pixels) = nes.0.pixels() else {
        return false;
    };
    for (o, &p) in out.as_chunks_mut::<3>().0.iter_mut().zip(pixels.iter()) {
        o.fill(p);
    }
    true
}

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The ROM (`$TRIFORCE_ROM`, else `roms/zelda.nes`). Missing: panics (`test_rom`), or None with
/// `TRIFORCE_ALLOW_NO_ROM=1`.
fn rom() -> Option<Arc<Rom>> {
    test_rom::path()?; // before the LazyLock, so every test fails with the ROM message, not a poisoned lock
    static ROM: LazyLock<Option<Arc<Rom>>> =
        LazyLock::new(|| test_rom::bytes().map(|bytes| Rom::from_ines(&bytes).expect("valid Zelda ROM")));
    ROM.clone()
}

fn open_core() -> Option<Counted> {
    Some(Counted(ZCore::power_on(rom()?), 0))
}

/// A named state of `triforce/states`, or a path as given.
fn state_path(name: &str) -> PathBuf {
    if Path::new(name).is_absolute() {
        return PathBuf::from(name);
    }
    repo().join(format!("triforce/states/{name}.state"))
}

fn read_state(name: &str) -> std::io::Result<Vec<u8>> {
    std::fs::read(state_path(name))
}

fn state_error(e: impl std::fmt::Display) -> EnvError {
    EnvError::State(e.to_string())
}

/// The regenerated start states, `triforce/states/<name>.state` (read-only).
struct TfzStore;

impl StateStore<MachineState> for TfzStore {
    fn load(&self, name: &str) -> Result<MachineState, EnvError> {
        Ok(self.decode(&self.bytes(name)?)?.machine)
    }

    fn save(&self, _: &Path, _: &SessionSnapshot<MachineState>) -> Result<(), EnvError> {
        Err(EnvError::State("read-only store".into()))
    }

    fn bytes(&self, name: &str) -> Result<Vec<u8>, EnvError> {
        read_state(name).map_err(|e| state_error(format!("{}: {e}", state_path(name).display())))
    }

    fn encode(&self, snapshot: &SessionSnapshot<MachineState>) -> Result<Vec<u8>, EnvError> {
        let rom = rom().ok_or_else(|| state_error("ROM not found"))?;
        let flags = snapshot.was_in_cave as u8 | (snapshot.just_exited_cave as u8) << 1;
        Ok(state_file::encode(&rom, &snapshot.machine, flags))
    }

    fn decode(&self, bytes: &[u8]) -> Result<SessionSnapshot<MachineState>, EnvError> {
        let rom = rom().ok_or_else(|| state_error("ROM not found"))?;
        let file = state_file::decode(&rom, bytes).map_err(state_error)?;
        Ok(SessionSnapshot {
            machine: file.machine,
            was_in_cave: file.flags & 1 != 0,
            just_exited_cave: file.flags & 2 != 0,
            total_frames: 0,
            steps: 0,
        })
    }

    fn power_on(&self) -> Result<MachineState, EnvError> {
        Ok(ZCore::power_on(rom().ok_or_else(|| state_error("ROM not found"))?).snapshot())
    }
}

const SPECIALIST: &str = "stalfos-keese-gel";
const START: &str = "1_73s_0";

fn data() -> Arc<GameData> {
    GameData::load(&repo()).expect("game data")
}

fn start_opts() -> EnvOptions {
    EnvOptions {
        start_state: Some(START.into()),
        ..Default::default()
    }
}

fn config(data: &GameData, max_steps: i64) -> Arc<env::SpecialistConfig> {
    specialist_config(data, SPECIALIST, max_steps)
}

fn specialist_config(data: &GameData, name: &str, max_steps: i64) -> Arc<env::SpecialistConfig> {
    let mut c = (*data.specialists[name]).clone();
    c.max_steps = max_steps;
    Arc::new(c)
}

fn obs_data(obs: &Observation) -> ObsData {
    let mut data = ObsData::new(Arc::new(obs_layout()));
    write_observation(obs, &mut data.row());
    data
}

/// Whether row `t` of a rollout's observations holds `expected`.
fn row_is(r: &EnvRollout<ZeldaAction>, t: usize, expected: &ObsData) -> bool {
    r.obs.columns.iter().enumerate().all(|(k, c)| {
        let n = expected.layout().keys()[k].len();
        let range = t * n..(t + 1) * n;
        match c {
            Column::I64(v) => v[range] == *expected.i64(k),
            Column::F32(v) => v[range] == *expected.f32(k),
            Column::Bool(v) => v[range] == *expected.bool(k),
        }
    })
}

/// A test actor: `pick(mask, job)` chooses every action (behind a lock; one env keeps the order fixed). Every
/// value is 1, so a bootstrapped truncation would show.
struct Picks<F>(Mutex<F>);

fn picks<F: FnMut(&ActionMask, i64) -> zelda::Action + Send>(pick: F) -> Arc<Picks<F>> {
    Arc::new(Picks(Mutex::new(pick)))
}

impl<F: FnMut(&ActionMask, i64) -> zelda::Action + Send> Actor<ZeldaAction> for Picks<F> {
    type Scratch = ();

    fn scratch(&self) {}

    fn act(&self, obs: &ObsData, _: i64, job: i64, _: i64, _: &mut ()) -> Result<(ZeldaAction, Decision), String> {
        let flat: [bool; MASK_SIZE] = obs.bool(ACTION_MASK).try_into().unwrap();
        let action = (self.0.lock())(&ActionMask::from_flat(&flat), job);
        Ok((action.to_indices().map(|i| i as i64), Decision::default()))
    }

    fn value(&self, _: &ObsData, _: i64, _: &mut ()) -> Result<f32, String> {
        Ok(1.0)
    }
}

struct Expected {
    obs: ObsData,
    reward: f32,
    frames: i64,
    interrupt: i64,
    terminated: bool,
    truncated: bool,
}

#[test]
fn train_env_rows_match_the_env_played_directly() {
    let Some(nes) = open_core() else { return };
    let data = data();
    let config = config(&data, 12);
    let seed = 9;

    // Directly: the SpecialistEnv on the bare emulator, seeded like pool env 0.
    let mut direct =
        SpecialistEnv::new(config.clone(), data.clone(), nes, TfzStore, start_opts()).unwrap();
    direct.rng = pool::env_rng(seed, 0);
    let mut action_rng = ChaCha8Rng::seed_from_u64(1);
    let first = obs_data(&direct.reset(None, None).unwrap().0);
    let mut actions = VecDeque::new();
    let mut expected = Vec::new();
    for _ in 0..40 {
        let action = random_action(direct.session().mask(), &mut action_rng, None);
        actions.push_back(action);
        let r = direct.step(action).unwrap();
        let done = r.terminated || r.truncated;
        let obs = if done {
            obs_data(&direct.reset(None, None).unwrap().0)
        } else {
            obs_data(&r.obs)
        };
        expected.push(Expected {
            obs,
            reward: r.reward as f32,
            frames: r.info.frames as i64,
            interrupt: interrupt_code(r.info.interrupt),
            terminated: r.terminated,
            truncated: r.truncated,
        });
    }
    let episodes = expected.iter().filter(|e| e.terminated || e.truncated).count();
    assert!(episodes >= 2, "max_steps 12 ends episodes");

    // In a pool.
    let mut nes = Some(direct.into_parts().0.into_nes());
    let pool = Pool::new(PoolOptions::new(1, 1, seed), |init| {
        let nes = nes.take().unwrap();
        TrainEnv::new(
            config.clone(),
            data.clone(),
            nes,
            TfzStore,
            start_opts(),
            init,
            None,
        )
    })
    .unwrap();
    let mut rollouts = pool.rollout(40, picks(move |_, _| actions.pop_front().unwrap())).unwrap();
    let (_, r) = rollouts.pop().unwrap();
    assert!(row_is(&r, 0, &first));
    for (i, e) in expected.iter().enumerate() {
        assert!(row_is(&r, i + 1, &e.obs), "step {i} observation");
        assert_eq!(
            (r.reward[i], r.frames[i], r.interrupt[i], r.terminated[i], r.truncated[i]),
            (e.reward, e.frames, e.interrupt, e.terminated, e.truncated),
            "step {i}"
        );
        assert_eq!(r.bootstrap_value[i], 0.0, "max-steps truncations don't bootstrap");
    }
    let ended: Vec<_> = pool
        .poll_events()
        .into_iter()
        .filter_map(|(_, e)| match e {
            ZeldaEvent::Episode { info, job } => Some((info, job)),
            _ => None,
        })
        .collect();
    assert_eq!(ended.len(), episodes);
    assert!(ended.iter().all(|(i, _)| i.terminal.is_some()));
    // Each episode event carries the job key of its rows (the terminal row's).
    let terminal_jobs: Vec<i64> = expected
        .iter()
        .enumerate()
        .filter(|(_, e)| e.terminated || e.truncated)
        .map(|(i, _)| r.job[i])
        .collect();
    assert_eq!(ended.iter().map(|(_, j)| *j).collect::<Vec<_>>(), terminal_jobs);
    assert!(terminal_jobs.windows(2).all(|w| w[0] != w[1]));
}

#[test]
fn chain_runs_in_a_pool_match_direct_runs() {
    let Some(nes) = open_core() else { return };
    let data = data();
    let route = Route::load(&data.map, &data.states_dir, "d2first", None, None).unwrap();
    let seeds = [3u64, 4];
    let cap = 80;

    let mut direct = ChainEnv::new(data.clone(), nes, TfzStore, route.clone());
    direct.max_steps = cap;
    let mut runs: Vec<RunResult> = Vec::new();
    for &seed in &seeds {
        let mut action_rng = ChaCha8Rng::seed_from_u64(100 + seed);
        let mut status = direct
            .reset(Some(seed), &mut ChaCha8Rng::seed_from_u64(seed))
            .unwrap();
        loop {
            match status {
                ChainStatus::Done(r) => break runs.push(r),
                ChainStatus::Act { obs, .. } => {
                    let mask = ActionMask::from_flat(&obs.action_mask);
                    status = direct
                        .step(random_action(&mask, &mut action_rng, None))
                        .unwrap();
                }
            }
        }
    }

    let policies: Vec<Arc<str>> = data
        .families
        .keys()
        .map(|k| Arc::from(k.as_str()))
        .collect();
    let mut nes = Some(direct.into_parts().0.into_nes());
    let jobs = Jobs::new(seeds);
    let pool = Pool::new(PoolOptions::new(1, 1, 0), |init| {
        let nes = nes.take().unwrap();
        let mut env = ChainRunEnv::new(
            data.clone(),
            nes,
            TfzStore,
            route.clone(),
            jobs.clone(),
            policies.clone(),
            init,
            None,
        );
        env.chain_mut().max_steps = cap;
        Ok::<_, EnvError>(env)
    })
    .unwrap();
    let mut rngs: HashMap<i64, ChaCha8Rng> = HashMap::new();
    let actor = picks(move |mask, job| {
        let rng = rngs.entry(job).or_insert_with(|| ChaCha8Rng::seed_from_u64(100 + job as u64));
        random_action(mask, rng, None)
    });
    pool.play(actor, Record::Off).unwrap();
    assert!(pool.wait_idle(None).unwrap());
    let events: Vec<ZeldaEvent> = pool.poll_events().into_iter().map(|(_, e)| e).collect();
    let run = events.iter().filter(|e| matches!(e, ZeldaEvent::Run(_))).count();
    assert_eq!(run, seeds.len());
    assert_eq!(pool.active(), 0);
    let pooled: Vec<RunResult> = events
        .iter()
        .filter_map(|e| match e {
            ZeldaEvent::Run(r) => Some((**r).clone()),
            _ => None,
        })
        .collect();
    assert_eq!(pooled, runs);
    // Room events per run: each run reports its start room first, then only changes.
    let mut per_run = vec![Vec::new()];
    for e in &events {
        match e {
            ZeldaEvent::Room(r) => per_run.last_mut().unwrap().push(r),
            ZeldaEvent::Run(_) => per_run.push(Vec::new()),
            _ => {}
        }
    }
    assert!(per_run.pop().unwrap().is_empty());
    assert_eq!(per_run.len(), seeds.len());
    for (rooms, run) in per_run.iter().zip(&runs) {
        assert_eq!((rooms[0].level, rooms[0].room), (0, 0x77));
        assert_eq!(&*rooms[0].specialist, run.rooms[0].specialist.as_str());
        assert!(
            rooms.windows(2).all(|w| w[0] != w[1]),
            "only changes are reported"
        );
    }
}

/// A test actor that records (policy, job, step) of every decision: the zora policy walks E (MOVE E 16), any
/// other waits (WAIT 8), each falling back to a legal action where the mask forbids it.
struct Recorder {
    zora: i64,
    decisions: Mutex<Vec<(i64, i64, i64)>>,
}

impl Actor<ZeldaAction> for Recorder {
    type Scratch = ();

    fn scratch(&self) {}

    fn act(&self, obs: &ObsData, policy: i64, job: i64, step: i64, _: &mut ()) -> Result<(ZeldaAction, Decision), String> {
        self.decisions.lock().push((policy, job, step));
        let flat: [bool; MASK_SIZE] = obs.bool(ACTION_MASK).try_into().unwrap();
        let mask = ActionMask::from_flat(&flat);
        let wanted = zelda::Action::parse(if policy == self.zora { "MOVE E 16" } else { "WAIT 8" }).unwrap();
        let action = if mask.is_allowed(&wanted) {
            wanted
        } else {
            random_action(&mask, &mut ChaCha8Rng::seed_from_u64(0), None)
        };
        Ok((action.to_indices().map(|i| i as i64), Decision::default()))
    }

    fn value(&self, _: &ObsData, _: i64, _: &mut ()) -> Result<f32, String> {
        Ok(1.0)
    }
}

/// F153: past a handoff (ow-zora walks E out of 0x5a into ow-moblin's 0x5b) every decision is acted on by the
/// room's specialist's policy index, keyed by the run's seed, and `step` counts the run's decisions from 0 across
/// rooms (the keyed noise of a sampled policy depends on it). Chain 1's entry state: walking straight E from it
/// reaches 0x5b (chain 0's Link enters in line with an obstacle).
#[test]
fn chain_decisions_carry_the_room_specialists_policy_and_the_run_step() {
    let Some(nes) = open_core() else { return };
    let data = data();
    let route = Route::load(&data.map, &data.states_dir, "d2first", None, Some("0_5aw_1")).unwrap();
    let seeds = [5u64, 6];
    let policies: Vec<Arc<str>> = vec![Arc::from("ow-moblin"), Arc::from("ow-zora")];
    let mut nes = Some(nes);
    let jobs = Jobs::new(seeds);
    let pool = Pool::new(PoolOptions::new(1, 1, 0), |init| {
        let mut env = ChainRunEnv::new(
            data.clone(),
            nes.take().unwrap(),
            TfzStore,
            route.clone(),
            jobs.clone(),
            policies.clone(),
            init,
            None,
        );
        env.chain_mut().max_steps = 40;
        Ok::<_, EnvError>(env)
    })
    .unwrap();
    let actor = Arc::new(Recorder {
        zora: 1,
        decisions: Mutex::new(Vec::new()),
    });
    pool.play(actor.clone(), Record::Off).unwrap();
    assert!(pool.wait_idle(None).unwrap());
    let runs: Vec<RunResult> = pool
        .poll_events()
        .into_iter()
        .filter_map(|(_, e)| match e {
            ZeldaEvent::Run(r) => Some((*r).clone()),
            _ => None,
        })
        .collect();
    assert_eq!(runs.len(), seeds.len());
    let decisions = actor.decisions.lock();
    for run in &runs {
        let job = run.seed.unwrap() as i64;
        let names: Vec<&str> = run.rooms.iter().map(|r| r.specialist.as_str()).collect();
        assert_eq!(names[..2], ["ow-zora", "ow-moblin"], "{run:?}");
        let expected: Vec<(i64, i64, i64)> = run
            .rooms
            .iter()
            .flat_map(|room| {
                let p = policies.iter().position(|n| **n == *room.specialist).unwrap() as i64;
                std::iter::repeat_n(p, room.steps as usize)
            })
            .enumerate()
            .map(|(step, p)| (p, job, step as i64))
            .collect();
        let recorded: Vec<(i64, i64, i64)> = decisions.iter().copied().filter(|d| d.1 == job).collect();
        assert_eq!(recorded.len() as u64, run.steps);
        assert_eq!(recorded, expected);
    }
}

#[test]
fn a_chain_specialist_without_a_model_ends_the_run_as_error() {
    let Some(nes) = open_core() else { return };
    let data = data();
    let route = Route::load(&data.map, &data.states_dir, "d2first", None, None).unwrap();
    let policies: Vec<Arc<str>> = data
        .families
        .keys()
        .map(|k| Arc::from(k.as_str()))
        .collect();
    let mut env = ChainRunEnv::new(
        data.clone(),
        nes,
        TfzStore,
        route,
        Jobs::new([0u64]),
        policies,
        init(),
        None,
    );
    let first = data
        .specialist_for(zelda::MapLocation::new(0, 0x77, false))
        .unwrap()
        .to_owned();
    env.missing_mut()
        .insert(first.clone(), format!("no promoted model for {first}"));
    let mut obs = ObsData::new(Arc::new(obs_layout()));
    let mut events = Vec::new();
    assert_eq!(env.reset(obs.row(), &mut events).unwrap(), None);
    let [ZeldaEvent::Run(r)] = &events[..] else {
        panic!("{events:?}")
    };
    assert_eq!(
        (r.result.as_str(), r.reason.clone()),
        ("error", format!("no promoted model for {first}"))
    );
}

fn init() -> EnvInit {
    EnvInit {
        index: 0,
        rng: pool::env_rng(0, 0),
        control: EnvControl::standalone(0, 0.0, 240 * 256 * 3),
    }
}

#[test]
fn frames_publish_only_while_rendering() {
    let Some(nes) = open_core() else { return };
    let data = data();
    // The Goriya room at full health, no moves: Link stays in the room (no episode ends mid-test), waits for the
    // Goriyas and then attacks; beams make the session run look-ahead frames, which are never shown.
    let mut config = (*specialist_config(&data, "goriya", 1000)).clone();
    config.ram_randomization = RamRandomization {
        hp: Some((HpBound::Max, HpBound::Max)),
        full_hp: Some(1.0),
        ..Default::default()
    };
    let config = Arc::new(config);
    let opts = || EnvOptions {
        start_state: Some("1_44w_0".into()),
        ..Default::default()
    };
    let mut nes = Some(nes);
    let pool = Pool::new(PoolOptions::new(1, 1, 0), |init| {
        TrainEnv::new(
            config.clone(),
            data.clone(),
            nes.take().unwrap(),
            TfzStore,
            opts(),
            init,
            Some(rgb),
        )
    })
    .unwrap();
    let mut frame = vec![0u8; pool.frame_bytes()];
    let mut rng = ChaCha8Rng::seed_from_u64(5);
    let kinds = Arc::new(Mutex::new(Vec::new()));
    let script = Arc::new(Mutex::new(VecDeque::from(["MOVE E 16", "MOVE E 16"])));
    let actor = {
        let (kinds, script) = (kinds.clone(), script.clone());
        picks(move |mask, _| {
            let action = match script.lock().pop_front() {
                Some(text) => zelda::Action::parse(text).unwrap(),
                None => {
                    let mut mask = prefer_attacks(*mask);
                    mask.disallow_kind(ActionKind::Move);
                    random_action(&mask, &mut rng, None)
                }
            };
            kinds.lock().push(action.kind);
            action
        })
    };
    let step = || {
        let (_, r) = pool.rollout(1, actor.clone()).unwrap().pop().unwrap();
        assert!(!(r.terminated[0] || r.truncated[0]));
        r.frames[0] as u64
    };
    // Off the door tile (attacks are masked there).
    step();
    assert_eq!(pool.frame_counter(0).unwrap(), 0);
    pool.set_render(0, true).unwrap();
    let mut shown = step();
    for _ in 0..12 {
        shown += step();
        // Every real frame of the step, no look-ahead frames.
        assert_eq!(pool.frame_counter(0).unwrap(), shown);
    }
    assert_eq!(pool.frame(0, 0, &mut frame).unwrap(), Some(shown));
    assert!(
        frame
            .as_chunks::<3>()
            .0
            .iter()
            .all(|p| p[0] == p[1] && p[1] == p[2])
    );
    assert!(
        frame.iter().any(|&v| v != frame[0]),
        "a dungeon room isn't one color"
    );
    pool.set_render(0, false).unwrap();
    step();
    assert_eq!(pool.frame_counter(0).unwrap(), shown);
    assert_eq!(pool.frame(0, shown, &mut frame).unwrap(), None);
    let kinds = kinds.lock();
    assert!(kinds.contains(&ActionKind::Beams), "beams fired: {kinds:?}");
}

/// Attack kinds only when any is legal, else the whole mask.
fn prefer_attacks(mask: ActionMask) -> ActionMask {
    let mut attacks = mask;
    attacks.disallow_kind(ActionKind::Move);
    attacks.disallow_kind(ActionKind::Wait);
    if attacks.to_flat().iter().any(|&a| a) {
        attacks
    } else {
        mask
    }
}

/// Verifies `replay` on a bare zcore (every recorded state is the machine the replay reaches) and returns the
/// RAM and PRG-RAM it ends with.
fn replayed(replay: &Replay) -> ([u8; 0x800], [u8; 0x2000]) {
    let rom = rom().expect("the ROM");
    replay::verify(rom.clone(), replay.clone(), &read_state).expect("the replay verifies");
    let mut player = Player::open(rom, replay.clone(), &read_state).unwrap();
    while player.step(false).unwrap() {}
    player.finish().unwrap();
    (*player.nes().ram(), *player.nes().prg_ram())
}

fn recording(embed_start: bool) -> Option<Arc<RecordOptions>> {
    Some(Arc::new(RecordOptions {
        embed_start,
        ..Default::default()
    }))
}

/// What one recorded episode left: the machine's RAM and PRG-RAM at its end, every frame zcore ran since power-on
/// (look-ahead included), the episode's replay, and its other events.
type Recorded = ([u8; 0x800], [u8; 0x2000], u64, Replay, Vec<ZeldaEvent>);

/// Plays one episode of `env` with attacks preferred, recording it with `options`.
fn play_recorded<E>(
    env: &mut E,
    control: &EnvControl,
    options: Option<Arc<RecordOptions>>,
    nes: impl Fn(&E) -> &Counted,
) -> Recorded
where
    E: PoolEnv<Action = ZeldaAction, Event = ZeldaEvent, Error = EnvError>,
{
    control.set_recording(options);
    let mut obs = ObsData::new(Arc::new(obs_layout()));
    let mut events = Vec::new();
    let mut rng = ChaCha8Rng::seed_from_u64(2);
    let start = env.reset(obs.row(), &mut events).unwrap().expect("an episode");
    loop {
        let flat: [bool; MASK_SIZE] = obs.bool(ACTION_MASK).try_into().unwrap();
        let action = random_action(&prefer_attacks(ActionMask::from_flat(&flat)), &mut rng, None);
        let outcome = env
            .step(action.to_indices().map(|i| i as i64), obs.row(), &mut events)
            .unwrap();
        if outcome.terminated || outcome.truncated {
            let nes = nes(env);
            let (ram, prg_ram, emulated) = (*nes.ram(), *nes.prg_ram(), nes.1);
            let at = events.iter().position(|e| matches!(e, ZeldaEvent::Replay { .. })).expect("a replay");
            let ZeldaEvent::Replay { job, data } = events.remove(at) else { unreachable!() };
            assert_eq!(job, start.job, "the replay carries its episode's job key");
            return (ram, prg_ram, emulated, Replay::parse(data).expect("a readable replay"), events);
        }
    }
}

/// The Goriya room at full health (the NES full-health encoding), 30 steps: with attacks preferred, sword beams
/// fire; the health pokes land after the gain-control frames.
fn goriya_full_health(data: &GameData) -> Arc<env::SpecialistConfig> {
    let mut config = (*specialist_config(data, "goriya", 30)).clone();
    config.ram_randomization = RamRandomization {
        hp: Some((HpBound::Max, HpBound::Max)),
        full_hp: Some(1.0),
        ..Default::default()
    };
    Arc::new(config)
}

#[test]
fn recorded_eval_episodes_replay_to_the_same_machine() {
    let Some(nes) = open_core() else { return };
    let data = data();
    // Eval: an explicit start and seed per episode.
    let start = "1_44w_0";
    let opts = EnvOptions {
        explicit_starts: true,
        ..Default::default()
    };
    let jobs = Jobs::new([EvalJob {
        start_state: start.into(),
        seed: 11,
    }]);
    let init = init();
    let control = init.control.clone();
    let config = goriya_full_health(&data);
    let mut env = EvalEnv::new(config, data.clone(), nes, TfzStore, opts, jobs, init, None).unwrap();
    let (ram, prg_ram, emulated, replay, events) =
        play_recorded(&mut env, &control, recording(false), |e| e.env().session().nes().inner());
    let record = events
        .into_iter()
        .find_map(|e| match e {
            ZeldaEvent::Eval(r) => Some(*r),
            _ => None,
        })
        .expect("an eval record");
    // Recorded by name: the replay names the start state and pins its content.
    let file_hash = replay::file_hash(&read_state(start).unwrap());
    assert_eq!(replay.source().unwrap(), Source::File { name: start.into(), file_hash });
    assert!(replay.start_state().is_none());
    let meta = replay.meta().unwrap();
    assert_eq!(meta["seed"], 11);
    assert_eq!(meta["ending"]["outcome"], record.ending.as_str());
    assert_eq!(replay.ending_name() == "win", record.success);
    assert_eq!((record.start_state.as_str(), record.seed), (start, 11));
    assert!(record.steps > 0 && record.frames > 0);
    // Room visits cover the episode: they start in the reset room and their steps, frames and health add up.
    let (first, last) = (&record.visits[0], record.visits.last().unwrap());
    let location = first.location;
    assert_eq!((location.level, location.value, location.in_cave), record.room);
    assert_eq!(record.visits.iter().map(|v| v.steps as u64).sum::<u64>(), record.steps);
    assert_eq!(record.visits.iter().map(|v| v.frames).sum::<u64>(), record.frames);
    assert_eq!((first.health_in, last.health_out), (record.start_health, record.health));
    assert_eq!(record.to_json()["visits"].as_array().unwrap().len(), record.visits.len());
    // Load randomization edits come first, the models take over once Link is controllable; the replay covers
    // the gain-control frames and every step.
    assert!(replay.edits().any(|e| e.unwrap().frame == 0));
    let control_frame = replay.events().next().unwrap().unwrap();
    assert_eq!(control_frame.event, Event::Control);
    assert!(u64::from(replay.frames() - control_frame.frame) >= record.frames);
    // Evaluation runs no look-ahead: zcore ran exactly the recorded frames since it powered on.
    assert_eq!(emulated, u64::from(replay.frames()));
    let mut obs = ObsData::new(Arc::new(obs_layout()));
    assert_eq!(env.reset(obs.row(), &mut Vec::new()).unwrap(), None, "one job");
    let (replayed_ram, replayed_prg_ram) = replayed(&replay);
    assert!(replayed_ram == ram, "RAM after replay");
    assert!(replayed_prg_ram == prg_ram, "PRG-RAM after replay");
}

#[test]
fn recorded_training_episodes_replay_past_their_look_aheads() {
    let Some(nes) = open_core() else { return };
    let data = data();
    let opts = EnvOptions {
        start_state: Some("1_44w_0".into()),
        ..Default::default()
    };
    let init = init();
    let control = init.control.clone();
    let config = goriya_full_health(&data);
    let mut env = TrainEnv::new(config, data.clone(), nes, TfzStore, opts, init, None).unwrap();
    let (ram, prg_ram, emulated, replay, _) =
        play_recorded(&mut env, &control, recording(true), |e| e.env().session().nes().inner());
    // zcore ran every frame since it powered on, look-ahead included; the replay kept only the real ones.
    assert!(emulated > u64::from(replay.frames()), "look-ahead ran: {emulated} frames emulated");
    // Embedded: the replay carries the start state file (and still says which it was).
    assert_eq!(replay.source().unwrap(), Source::Embedded);
    assert_eq!(replay.start_state().unwrap().blob, read_state("1_44w_0").unwrap());
    assert_eq!(replay.meta().unwrap()["source"], serde_json::json!({"kind": "state", "name": "1_44w_0"}));
    let (replayed_ram, replayed_prg_ram) = replayed(&replay);
    assert!(replayed_ram == ram, "RAM after replay");
    assert!(replayed_prg_ram == prg_ram, "PRG-RAM after replay");
}

/// The names of every family, in the order chain pools index their policies.
fn all_policies(data: &GameData) -> Vec<Arc<str>> {
    data.families.keys().map(|k| Arc::from(k.as_str())).collect()
}

/// Plays the next chain run of `env` to its end with `pick`, recorded with `options`: (the run, its replay).
fn play_chain_recorded<St>(
    env: &mut ChainRunEnv<Counted, St>,
    control: &EnvControl,
    options: RecordOptions,
    mut pick: impl FnMut(&ActionMask) -> zelda::Action,
) -> (RunResult, Replay)
where
    St: StateStore<MachineState> + Send + 'static,
{
    control.set_recording(Some(Arc::new(options)));
    let mut obs = ObsData::new(Arc::new(obs_layout()));
    let mut events = Vec::new();
    let start = env.reset(obs.row(), &mut events).unwrap().expect("a run");
    loop {
        let flat: [bool; MASK_SIZE] = obs.bool(ACTION_MASK).try_into().unwrap();
        let action = pick(&ActionMask::from_flat(&flat)).to_indices().map(|i| i as i64);
        if env.step(action, obs.row(), &mut events).unwrap().terminated {
            break;
        }
    }
    let (mut run, mut replay) = (None, None);
    for event in events {
        match event {
            ZeldaEvent::Run(r) => run = Some(*r),
            ZeldaEvent::Replay { job, data } => {
                assert_eq!(job, start.job);
                replay = Some(Replay::parse(data).expect("a readable replay"));
            }
            _ => {}
        }
    }
    (run.expect("a run result"), replay.expect("a replay"))
}

/// A recorded chain run carries its rooms (each with the policy that acted and the outcome of the visit), a
/// keyframe at the first action boundary after every `keyframe_frames` frames, and verifies: every keyframe is
/// the machine the replay reaches there.
#[test]
fn recorded_chain_runs_carry_rooms_outcomes_and_keyframes() {
    let Some(nes) = open_core() else { return };
    let data = data();
    let route = Route::load(&data.map, &data.states_dir, "d2first", None, None).unwrap();
    let policies = all_policies(&data);
    let init = init();
    let control = init.control.clone();
    let mut env = ChainRunEnv::new(data.clone(), nes, TfzStore, route, Jobs::new([3u64]), policies.clone(), init, None);
    env.chain_mut().max_steps = 150;
    let options = RecordOptions {
        embed_start: true,
        keyframe_frames: 120,
        ..Default::default()
    };
    let mut rng = ChaCha8Rng::seed_from_u64(103);
    let (run, replay) = play_chain_recorded(&mut env, &control, options, |mask| random_action(mask, &mut rng, None));
    assert_eq!((run.result.as_str(), replay.ending_name()), ("failure", "stuck"), "{}", run.reason);
    assert_eq!(replay.meta().unwrap()["ending"]["reason"], run.reason.as_str());
    replayed(&replay);

    let states: Vec<_> = replay.states().collect();
    let frames: Vec<u32> = states.iter().map(|s| s.frame).collect();
    let kinds: Vec<StateKind> = states.iter().map(|s| s.kind).collect();
    assert_eq!((kinds[0], *kinds.last().unwrap()), (StateKind::Start, StateKind::End));
    assert!(kinds.len() >= 4 && kinds[1..kinds.len() - 1].iter().all(|&k| k == StateKind::Keyframe), "{kinds:?}");
    let keyframes = &frames[..frames.len() - 1];
    assert!(keyframes.windows(2).all(|w| w[1] - w[0] >= 120), "{frames:?}");
    // The longest action is 32 frames (plus a room change's scroll): keyframes come as soon as they are due.
    assert!(keyframes.windows(2).all(|w| w[1] - w[0] < 120 + 400), "{frames:?}");
    // The run starts on the route's first step; a random walk leaves the route (no step) sooner or later.
    assert_eq!(states[0].context.step, Some(0));

    let events: Vec<_> = replay.events().map(|e| e.unwrap().event).collect();
    assert_eq!(events[0], Event::Control);
    let rooms: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            Event::Room { level, room, in_cave, policy, .. } => Some((*level, *room, *in_cave, *policy)),
            _ => None,
        })
        .collect();
    let expected: Vec<_> = run
        .rooms
        .iter()
        .map(|r| {
            let location = r.location();
            let policy = policies.iter().position(|p| **p == *r.specialist).map(|p| p as u8);
            (location.level, location.value, location.in_cave, policy)
        })
        .collect();
    assert_eq!(rooms, expected);
    assert_eq!(rooms[0], (0, 0x77, false, policies.iter().position(|p| &**p == "ow-start").map(|p| p as u8)));
    let outcomes: Vec<&str> = events
        .iter()
        .filter_map(|e| match e {
            Event::Outcome(outcome) => Some(outcome.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(outcomes, run.rooms.iter().map(|r| r.outcome.as_str()).collect::<Vec<_>>());
    assert_eq!(replay.cheats().count(), 0, "no room on the way needs an item");
}

/// The cheat ledger of a recorded run: every granted bomb placed in 0x2c (a `requires_bombs` room entered with
/// none) is in the replay, at the frame after the action that placed it.
#[test]
fn recorded_chain_runs_carry_the_cheated_items_spent() {
    let Some(nes) = open_core() else { return };
    let data = data();
    let mut source = TfzStore.decode(&read_state("0_2cc_0").unwrap()).unwrap();
    source.machine.poke(0x658, 0); // no bombs
    let start = std::env::temp_dir().join(format!("pool-cheats-{}-0_2cc_0.state", std::process::id()));
    std::fs::write(&start, TfzStore.encode(&source).unwrap()).unwrap();
    let route = Route::load(&data.map, &data.states_dir, "d2first", None, Some(start.to_str().unwrap())).unwrap();
    let init = init();
    let control = init.control.clone();
    let mut env = ChainRunEnv::new(data.clone(), nes, TfzStore, route, Jobs::new([0u64]), all_policies(&data), init, None);
    env.chain_mut().max_steps = 40;
    let options = RecordOptions {
        embed_start: true,
        ..Default::default()
    };
    let (run, replay) = play_chain_recorded(&mut env, &control, options, |mask| {
        match mask.directions(ActionKind::Bomb).first() {
            Some(&direction) => zelda::Action::new(ActionKind::Bomb, direction, 0).unwrap(),
            None => zelda::Action::parse("WAIT 8").unwrap(),
        }
    });
    std::fs::remove_file(&start).unwrap();
    replayed(&replay);
    let placed: u32 = run.rooms.iter().map(|r| r.cheated.bombs).sum();
    assert!(placed >= 2, "{run:?}");
    let cheats: Vec<_> = replay.cheats().map(|c| c.unwrap()).collect();
    assert!(cheats.iter().all(|c| replay::CHEAT_ITEMS[c.item as usize] == "bombs"), "{cheats:?}");
    assert_eq!(cheats.iter().map(|c| c.count).sum::<u32>(), placed);
    assert!(cheats.windows(2).all(|w| w[0].frame < w[1].frame) && cheats[0].frame > 0, "{cheats:?}");
    // The granted bomb Link holds at the end is the end state's context, not a spent one.
    assert_eq!(replay.end_state().context.granted, [u8::from(run.rooms.last().unwrap().outcome != "death"), 0, 0]);
}

/// The tables a replay's streams index are the game's: the action space of the decision stream and the cheat
/// items.
#[test]
fn the_replay_tables_are_the_games() {
    use replay::actions;
    assert_eq!(actions::KINDS.to_vec(), ActionKind::ALL.map(|k| k.name()).to_vec());
    assert_eq!(actions::DIRECTIONS.to_vec(), zelda::actions::DIRECTIONS.map(|d| d.name()).to_vec());
    assert_eq!(actions::DURATIONS, zelda::actions::DURATIONS);
    assert_eq!(actions::DISTRIBUTION, MASK_SIZE);
    assert_eq!(replay::CHEAT_ITEMS.to_vec(), zelda::cheats::CheatItem::ALL.map(|i| i.name()).to_vec());
    assert_eq!(replay::OUTCOME_CLASSES.to_vec(), env::rewards::OUTCOMES.to_vec());
}

fn wait_action() -> [i64; 3] {
    zelda::Action::parse("WAIT 8")
        .unwrap()
        .to_indices()
        .map(|i| i as i64)
}

fn eval_jobs() -> Vec<EvalJob> {
    [5u64, 6, 7]
        .map(|seed| EvalJob {
            start_state: START.into(),
            seed,
        })
        .to_vec()
}

fn eval_env(
    nes: Counted,
    data: &Arc<GameData>,
    jobs: Jobs<EvalJob>,
    init: EnvInit,
) -> EvalEnv<Counted, TfzStore> {
    let opts = EnvOptions {
        explicit_starts: true,
        ..Default::default()
    };
    EvalEnv::new(
        config(data, 4),
        data.clone(),
        nes,
        TfzStore,
        opts,
        jobs,
        init,
        None,
    )
    .unwrap()
}

#[test]
fn eval_jobs_are_shared_and_each_result_is_independent_of_the_env() {
    let (Some(a), Some(b), Some(c)) = (open_core(), open_core(), open_core()) else {
        return;
    };
    let data = data();
    // Directly: one env plays every job in order.
    let mut direct = eval_env(a, &data, Jobs::new(eval_jobs()), init());
    let mut obs = ObsData::new(Arc::new(obs_layout()));
    let mut events = Vec::new();
    let mut expected = Vec::new();
    while direct.reset(obs.row(), &mut events).unwrap().is_some() {
        while !direct
            .step(wait_action(), obs.row(), &mut events)
            .map(|o| o.terminated || o.truncated)
            .unwrap()
        {}
        for e in events.drain(..) {
            match e {
                ZeldaEvent::Eval(r) => expected.push(*r),
                ZeldaEvent::Room(_) => panic!("room events are off by default"),
                _ => {}
            }
        }
    }
    assert_eq!(expected.len(), 3);
    // F095: the record keeps the hearts at reset; no container is taken by waiting, so it equals the end's.
    for r in &expected {
        assert!(r.start_max_health >= 3, "{r:?}");
        assert_eq!(r.start_max_health, r.max_health);
        assert_eq!(r.to_json()["start_max_health"], r.start_max_health);
    }

    // In a pool: two envs take jobs from one list; only env 0 reports rooms.
    let jobs = Jobs::new(eval_jobs());
    let mut cores = vec![b, c];
    let pool = Pool::new(PoolOptions::new(2, 2, 0), |init| {
        let index = init.index;
        let mut env = eval_env(cores.remove(0), &data, jobs.clone(), init);
        env.set_room_events(index == 0);
        Ok::<_, EnvError>(pool::zelda::TaskEnv::Eval(env))
    })
    .unwrap();
    pool.play(picks(|_, _| zelda::Action::parse("WAIT 8").unwrap()), Record::Off).unwrap();
    assert!(pool.wait_idle(None).unwrap());
    assert_eq!(pool.active(), 0, "both envs finish once the list is empty");
    assert_eq!(jobs.remaining(), 0);
    let (mut records, mut rooms) = (Vec::new(), Vec::new());
    for (env, e) in pool.poll_events() {
        match e {
            ZeldaEvent::Eval(r) => records.push(*r),
            ZeldaEvent::Room(r) => rooms.push((env, r)),
            _ => {}
        }
    }
    records.sort_by_key(|r| r.seed);
    assert_eq!(
        records, expected,
        "every job played once, with the same result as directly"
    );
    assert!(
        !rooms.is_empty() && rooms.iter().all(|(env, _)| *env == 0),
        "{rooms:?}"
    );
    let (_, first) = &rooms[0];
    assert_eq!(
        (first.level, first.room, &*first.specialist),
        (1, 0x73, SPECIALIST)
    );
}

#[test]
fn a_chain_specialist_outside_the_pool_policies_ends_the_run_as_error() {
    let Some(nes) = open_core() else { return };
    let data = data();
    let route = Route::load(&data.map, &data.states_dir, "d2first", None, None).unwrap();
    let only: Vec<Arc<str>> = vec![Arc::from("boss-aquamentus")];
    let mut env = ChainRunEnv::new(
        data.clone(),
        nes,
        TfzStore,
        route,
        Jobs::new([0u64, 1]),
        only,
        init(),
        None,
    );
    let first = data
        .specialist_for(zelda::MapLocation::new(0, 0x77, false))
        .unwrap()
        .to_owned();
    let mut obs = ObsData::new(Arc::new(obs_layout()));
    let mut events = Vec::new();
    // Every seed's run fails at once, so one reset drains the list: a run event per seed, then finished.
    assert_eq!(env.reset(obs.row(), &mut events).unwrap(), None);
    let reason = format!("specialist {first} has no policy in this pool");
    let runs: Vec<(String, String)> = events
        .iter()
        .map(|e| match e {
            ZeldaEvent::Run(r) => (r.result.clone(), r.reason.clone()),
            other => panic!("{other:?}"),
        })
        .collect();
    assert_eq!(runs, vec![("error".to_owned(), reason); 2]);
    events.clear();
    assert_eq!(env.reset(obs.row(), &mut events).unwrap(), None);
    assert!(events.is_empty(), "no seeds left: no run");
}

/// One search's worth of simulation in `env` (search's `SearchEnv` calls): `futures` futures of up to `depth` random
/// steps each, attacks preferred, from the saved state with the RNG bytes redrawn.
fn simulate_futures<E>(env: &mut E, obs: &ObsData, rng: &mut ChaCha8Rng, futures: u64, depth: usize)
where
    E: SearchEnv<Action = ZeldaAction, Error = EnvError>,
{
    let mut sim = ObsData::new(Arc::new(obs_layout()));
    let snapshot = env.begin_search().unwrap();
    for f in 0..futures {
        env.load(&snapshot);
        env.perturb(f);
        let mut mask: [bool; MASK_SIZE] = obs.bool(ACTION_MASK).try_into().unwrap();
        for _ in 0..depth {
            let action = random_action(&prefer_attacks(ActionMask::from_flat(&mask)), rng, None);
            let o = env.simulate(action.to_indices().map(|i| i as i64), sim.row()).unwrap();
            if o.terminated || o.truncated {
                break;
            }
            mask = sim.bool(ACTION_MASK).try_into().unwrap();
        }
    }
    env.end_search(&snapshot);
}

#[test]
fn a_search_leaves_the_env_exactly_where_it_was() {
    let (Some(nes_a), Some(nes_b)) = (open_core(), open_core()) else { return };
    let data = data();
    let opts = || EnvOptions {
        start_state: Some("1_44w_0".into()),
        ..Default::default()
    };
    let config = goriya_full_health(&data);
    let (init_a, init_b) = (init(), init());
    let (control_a, control_b) = (init_a.control.clone(), init_b.control.clone());
    for control in [&control_a, &control_b] {
        control.set_recording(recording(true));
        control.set_render(true);
    }
    let mut a = TrainEnv::new(config.clone(), data.clone(), nes_a, TfzStore, opts(), init_a, Some(rgb)).unwrap();
    let mut b = TrainEnv::new(config, data.clone(), nes_b, TfzStore, opts(), init_b, Some(rgb)).unwrap();
    let (mut obs_a, mut obs_b) = (ObsData::new(Arc::new(obs_layout())), ObsData::new(Arc::new(obs_layout())));
    let (mut events_a, mut events_b) = (Vec::new(), Vec::new());
    a.reset(obs_a.row(), &mut events_a).unwrap();
    b.reset(obs_b.row(), &mut events_b).unwrap();
    let (mut sim_rng, mut rng) = (ChaCha8Rng::seed_from_u64(5), ChaCha8Rng::seed_from_u64(6));
    let (mut predicted, mut episodes) = (0, 0);
    for step in 0..60 {
        predicted += !a.env().session().ledger().predictions().is_empty() as usize;
        let published = control_a.frame_counter();
        simulate_futures(&mut a, &obs_a, &mut sim_rng, 3, 4);
        assert_eq!(control_a.frame_counter(), published, "simulated frames are never published");
        let flat: [bool; MASK_SIZE] = obs_b.bool(ACTION_MASK).try_into().unwrap();
        let action = random_action(&prefer_attacks(ActionMask::from_flat(&flat)), &mut rng, None);
        let action = action.to_indices().map(|i| i as i64);
        let oa = a.step(action, obs_a.row(), &mut events_a).unwrap();
        let ob = b.step(action, obs_b.row(), &mut events_b).unwrap();
        assert_eq!(oa, ob, "step {step}");
        assert!(control_a.frame_counter() > published, "real frames are published");
        assert!(obs_a == obs_b, "step {step} observation");
        let (sa, sb) = (a.env().session(), b.env().session());
        assert_eq!(format!("{:?}", sa.ledger()), format!("{:?}", sb.ledger()), "step {step} ledger");
        assert_eq!((sa.frame(), sa.total_frames, sa.steps), (sb.frame(), sb.total_frames, sb.steps));
        assert!(sa.nes().ram() == sb.nes().ram(), "step {step} RAM");
        assert_eq!(format!("{events_a:?}"), format!("{events_b:?}"), "step {step} events");
        if oa.terminated || oa.truncated {
            episodes += 1;
            a.reset(obs_a.row(), &mut events_a).unwrap();
            b.reset(obs_b.row(), &mut events_b).unwrap();
            assert!(obs_a == obs_b);
        }
        events_a.clear();
        events_b.clear();
    }
    assert!(predicted > 0, "searches started with look-ahead credit pending");
    assert!(episodes > 0, "an episode ended (its replay and terminal info matched)");
}
