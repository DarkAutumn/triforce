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
    ACTION_MASK, ChainRunEnv, EvalEnv, EvalJob, InputLog, TrainEnv, ZeldaAction, ZeldaEvent, interrupt_code,
    obs_layout, write_observation,
};
use parking_lot::Mutex;
use pool::{Column, EnvControl, EnvInit, Jobs, ObsData, Pool, PoolEnv, PoolOptions, SearchEnv};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock};
use zcore::{MachineState, Rom, ZCore};
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

/// The ROM (`$TRIFORCE_ROM`, else `roms/zelda.nes`), or None when it is missing.
fn rom() -> Option<Arc<Rom>> {
    static ROM: LazyLock<Option<Arc<Rom>>> = LazyLock::new(|| {
        let path = std::env::var_os("TRIFORCE_ROM")
            .map(PathBuf::from)
            .unwrap_or_else(|| repo().join("roms/zelda.nes"));
        let bytes = std::fs::read(path).ok()?;
        Some(Rom::from_ines(&bytes).expect("valid Zelda ROM"))
    });
    ROM.clone()
}

fn open_core() -> Option<Counted> {
    let Some(rom) = rom() else {
        eprintln!("ROM not found: set TRIFORCE_ROM or create roms/zelda.nes");
        return None;
    };
    Some(Counted(ZCore::power_on(rom), 0))
}

/// The regenerated start states, `triforce/states/<name>.state` (read-only).
struct TfzStore;

impl StateStore<MachineState> for TfzStore {
    fn load(&self, name: &str) -> Result<MachineState, EnvError> {
        let path = repo().join(format!("triforce/states/{name}.state"));
        let err = |e: &dyn std::fmt::Display| EnvError::State(format!("{}: {e}", path.display()));
        let bytes = std::fs::read(&path).map_err(|e| err(&e))?;
        let rom = rom().ok_or_else(|| EnvError::State("ROM not found".into()))?;
        Ok(zcore::state_file::decode(&rom, &bytes)
            .map_err(|e| err(&e))?
            .machine)
    }

    fn save(&self, _: &Path, _: &SessionSnapshot<MachineState>) -> Result<(), EnvError> {
        Err(EnvError::State("read-only store".into()))
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
            ZeldaEvent::Episode(info) => Some(info),
            _ => None,
        })
        .collect();
    assert_eq!(ended.len(), episodes);
    assert!(ended.iter().all(|i| i.terminal.is_some()));
}

#[test]
fn chain_runs_in_a_pool_match_direct_runs() {
    let Some(nes) = open_core() else { return };
    let data = data();
    let route = Route::load(&data.map, "d2first", None, None).unwrap();
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

#[test]
fn a_chain_specialist_without_a_model_ends_the_run_as_error() {
    let Some(nes) = open_core() else { return };
    let data = data();
    let route = Route::load(&data.map, "d2first", None, None).unwrap();
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

/// Replays a log on the bare emulator from its origin state (TFMOVIE poke order).
fn replay(nes: &mut Counted, log: &InputLog) {
    nes.restore(&TfzStore.load(log.origin.as_deref().unwrap()).unwrap());
    let mut pokes = log.pokes.iter().peekable();
    for (frame, &buttons) in log.buttons.iter().enumerate() {
        while let Some(&(_, addr, value)) = pokes.next_if(|p| p.0 as usize == frame) {
            nes.poke(addr, value);
        }
        nes.run_frame(buttons, false);
    }
    for &(_, addr, value) in pokes {
        nes.poke(addr, value);
    }
}

/// What one recorded episode left: the machine's RAM and PRG-RAM at its end, every frame zcore ran since power-on
/// (look-ahead included), the episode's input log, and its other events.
type Recorded = ([u8; 0x800], [u8; 0x2000], u64, InputLog, Vec<ZeldaEvent>);

/// Plays one episode of `env` with attacks preferred, recording it.
fn play_recorded<E>(env: &mut E, control: &EnvControl, nes: impl Fn(&E) -> &Counted) -> Recorded
where
    E: PoolEnv<Action = ZeldaAction, Event = ZeldaEvent, Error = EnvError>,
{
    control.set_recording(true);
    let mut obs = ObsData::new(Arc::new(obs_layout()));
    let mut events = Vec::new();
    let mut rng = ChaCha8Rng::seed_from_u64(2);
    assert!(env.reset(obs.row(), &mut events).unwrap().is_some());
    loop {
        let flat: [bool; MASK_SIZE] = obs.bool(ACTION_MASK).try_into().unwrap();
        let action = random_action(&prefer_attacks(ActionMask::from_flat(&flat)), &mut rng, None);
        let outcome = env
            .step(action.to_indices().map(|i| i as i64), obs.row(), &mut events)
            .unwrap();
        if outcome.terminated || outcome.truncated {
            let nes = nes(env);
            let (ram, prg_ram, emulated) = (*nes.ram(), *nes.prg_ram(), nes.1);
            let at = events.iter().position(|e| matches!(e, ZeldaEvent::Movie(_))).expect("a movie");
            let ZeldaEvent::Movie(log) = events.remove(at) else { unreachable!() };
            return (ram, prg_ram, emulated, *log, events);
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
    let (ram, prg_ram, emulated, log, events) = play_recorded(&mut env, &control, |e| e.env().session().nes().inner());
    let record = events
        .into_iter()
        .find_map(|e| match e {
            ZeldaEvent::Eval(r) => Some(*r),
            _ => None,
        })
        .expect("an eval record");
    assert_eq!(log.origin.as_deref(), Some(start));
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
    // Load randomization pokes come first; the log covers the gain-control frames and every step.
    assert!(log.pokes.iter().any(|p| p.0 == 0));
    assert!(log.buttons.len() as u64 >= record.frames);
    // Evaluation runs no look-ahead: zcore ran exactly the logged frames since it powered on.
    assert_eq!(emulated, log.buttons.len() as u64);
    let mut obs = ObsData::new(Arc::new(obs_layout()));
    assert_eq!(env.reset(obs.row(), &mut Vec::new()).unwrap(), None, "one job");
    let mut nes = env.into_env().into_parts().0.into_nes().into_inner();
    replay(&mut nes, &log);
    assert!(nes.ram() == &ram, "RAM after replay");
    assert!(nes.prg_ram() == &prg_ram, "PRG-RAM after replay");
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
    let (ram, prg_ram, emulated, log, _) = play_recorded(&mut env, &control, |e| e.env().session().nes().inner());
    // zcore ran every frame since it powered on, look-ahead included; the log kept only the real ones.
    assert!(emulated > log.buttons.len() as u64, "look-ahead ran: {emulated} frames emulated");
    let mut nes = env.into_env().into_parts().0.into_nes().into_inner();
    replay(&mut nes, &log);
    assert!(nes.ram() == &ram, "RAM after replay");
    assert!(nes.prg_ram() == &prg_ram, "PRG-RAM after replay");
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
    let route = Route::load(&data.map, "d2first", None, None).unwrap();
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
        control.set_recording(true);
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
    assert!(episodes > 0, "an episode ended (its movie and terminal info matched)");
}
