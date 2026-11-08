//! The Zelda adapters in a pool, on zcore with the regenerated `TFZSTATE` start states in `triforce/states`
//! (skipped when the ROM is missing).

use env::chain::{ChainEnv, ChainStatus, Route, RunResult};
use env::config::{GameData, HpBound, RamRandomization};
use env::explore::random_action;
use env::observation::Observation;
use env::specialist_env::{EnvOptions, SpecialistEnv};
use env::{EnvError, StateStore};
use pool::zelda::{
    ACTION_MASK, ChainRunEnv, EvalEnv, EvalJob, InputLog, TrainEnv, ZeldaEvent, interrupt_code,
    obs_layout, write_observation,
};
use pool::{Buffers, Column, EnvControl, EnvInit, Jobs, ObsData, Pool, PoolEnv, PoolOptions};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
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

/// Whether batch row `row` of `columns` holds `expected`.
fn row_is(columns: &[Column], b: &Buffers, row: usize, expected: &ObsData) -> bool {
    columns.iter().enumerate().all(|(k, c)| {
        let r = b.row_range(k, row);
        match c {
            Column::I64(v) => v[r] == *expected.i64(k),
            Column::F32(v) => v[r] == *expected.f32(k),
            Column::Bool(v) => v[r] == *expected.bool(k),
        }
    })
}

fn row_mask(b: &Buffers, row: usize) -> ActionMask {
    let flat: [bool; MASK_SIZE] = b.obs[ACTION_MASK].bool()[b.row_range(ACTION_MASK, row)]
        .try_into()
        .unwrap();
    ActionMask::from_flat(&flat)
}

struct Expected {
    obs: ObsData,
    final_obs: Option<ObsData>,
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
    let mut actions = Vec::new();
    let mut expected = Vec::new();
    for _ in 0..40 {
        let action = random_action(direct.session().mask(), &mut action_rng, None);
        actions.push(action.to_indices().map(|i| i as i64));
        let r = direct.step(action).unwrap();
        let done = r.terminated || r.truncated;
        let (obs, final_obs) = if done {
            let reset = obs_data(&direct.reset(None, None).unwrap().0);
            (reset, Some(obs_data(&r.obs)))
        } else {
            (obs_data(&r.obs), None)
        };
        expected.push(Expected {
            obs,
            final_obs,
            reward: r.reward as f32,
            frames: r.info.frames as i64,
            interrupt: interrupt_code(r.info.interrupt),
            terminated: r.terminated,
            truncated: r.truncated,
        });
    }
    let episodes = expected.iter().filter(|e| e.final_obs.is_some()).count();
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
    let mut b = pool.buffers();
    assert_eq!(pool.recv(&mut b.batch(), 1, None).unwrap(), 1);
    assert!(row_is(&b.obs, &b, 0, &first));
    for (i, (action, e)) in actions.iter().zip(&expected).enumerate() {
        pool.send(&[0], &[*action]).unwrap();
        assert_eq!(pool.recv(&mut b.batch(), 1, None).unwrap(), 1);
        assert!(row_is(&b.obs, &b, 0, &e.obs), "step {i} observation");
        assert_eq!(b.final_valid[0], e.final_obs.is_some(), "step {i}");
        if let Some(f) = &e.final_obs {
            assert!(row_is(&b.final_obs, &b, 0, f), "step {i} final observation");
        }
        assert_eq!(
            (
                b.reward[0],
                b.frames[0],
                b.interrupt[0],
                b.terminated[0],
                b.truncated[0]
            ),
            (e.reward, e.frames, e.interrupt, e.terminated, e.truncated),
            "step {i}"
        );
        assert!(!b.bootstrap[0], "max-steps truncations don't bootstrap");
        assert_eq!(b.policy[0], 0);
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
    let route = Route::load(&data.map, "milestone1", None, None).unwrap();
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
    let mut b = pool.buffers();
    let mut run = 0;
    let mut action_rng = ChaCha8Rng::seed_from_u64(100 + seeds[0]);
    let mut events = Vec::new();
    loop {
        assert_eq!(pool.recv(&mut b.batch(), 1, None).unwrap(), 1);
        events.extend(pool.poll_events().into_iter().map(|(_, e)| e));
        if b.terminated[0] {
            assert_eq!(b.reward[0], 0.0);
            run += 1;
            if run < seeds.len() {
                action_rng = ChaCha8Rng::seed_from_u64(100 + seeds[run]);
            }
        }
        if b.finished[0] {
            break;
        }
        let specialist = &policies[b.policy[0] as usize];
        assert!(data.families.contains_key(&**specialist));
        let action = random_action(&row_mask(&b, 0), &mut action_rng, None);
        pool.send(&[0], &[action.to_indices().map(|i| i as i64)])
            .unwrap();
    }
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
    let route = Route::load(&data.map, "milestone1", None, None).unwrap();
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
    let mut b = pool.buffers();
    let mut frame = vec![0u8; pool.frame_bytes()];
    let mut rng = ChaCha8Rng::seed_from_u64(5);
    let mut kinds = Vec::new();
    let mut step = |b: &mut Buffers, scripted: Option<&str>| {
        let action = match scripted {
            Some(text) => zelda::Action::parse(text).unwrap(),
            None => {
                let mut mask = prefer_attacks(row_mask(b, 0));
                mask.disallow_kind(ActionKind::Move);
                random_action(&mask, &mut rng, None)
            }
        };
        kinds.push(action.kind);
        pool.send(&[0], &[action.to_indices().map(|i| i as i64)])
            .unwrap();
        pool.recv(&mut b.batch(), 1, None).unwrap();
        assert!(!b.final_valid[0]);
        b.frames[0] as u64
    };
    pool.recv(&mut b.batch(), 1, None).unwrap();
    // Off the door tile (attacks are masked there).
    step(&mut b, Some("MOVE E 16"));
    assert_eq!(pool.frame_counter(0).unwrap(), 0);
    pool.set_render(0, true).unwrap();
    let mut shown = step(&mut b, Some("MOVE E 16"));
    for _ in 0..12 {
        shown += step(&mut b, None);
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
    step(&mut b, None);
    assert_eq!(pool.frame_counter(0).unwrap(), shown);
    assert_eq!(pool.frame(0, shown, &mut frame).unwrap(), None);
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

#[test]
fn recorded_episodes_replay_to_the_same_machine() {
    let Some(nes) = open_core() else { return };
    let data = data();
    // Eval: an explicit start and seed per episode. The Goriya room at full health (the NES full-health encoding)
    // with attacks preferred: sword beams fire, so the session runs its look-ahead; the health pokes land after
    // the gain-control frames.
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
    let mut config = (*specialist_config(&data, "goriya", 30)).clone();
    config.ram_randomization = RamRandomization {
        hp: Some((HpBound::Max, HpBound::Max)),
        full_hp: Some(1.0),
        ..Default::default()
    };
    let mut env = EvalEnv::new(
        Arc::new(config),
        data.clone(),
        nes,
        TfzStore,
        opts,
        jobs,
        init,
        None,
    )
    .unwrap();
    control.set_recording(true);
    let mut obs = ObsData::new(Arc::new(obs_layout()));
    let mut events = Vec::new();
    let mut rng = ChaCha8Rng::seed_from_u64(2);
    assert!(env.reset(obs.row(), &mut events).unwrap().is_some());
    let (ram, prg_ram, emulated, log, record) = loop {
        let flat: [bool; MASK_SIZE] = obs.bool(ACTION_MASK).try_into().unwrap();
        let action = random_action(
            &prefer_attacks(ActionMask::from_flat(&flat)),
            &mut rng,
            None,
        );
        let outcome = env
            .step(
                action.to_indices().map(|i| i as i64),
                obs.row(),
                &mut events,
            )
            .unwrap();
        if outcome.terminated || outcome.truncated {
            let nes = env.env().session().nes().inner();
            let (ram, prg_ram, emulated) = (*nes.ram(), *nes.prg_ram(), nes.1);
            let mut log = None;
            let mut record = None;
            for e in events.drain(..) {
                match e {
                    ZeldaEvent::Movie(l) => log = Some(*l),
                    ZeldaEvent::Eval(r) => record = Some(*r),
                    _ => {}
                }
            }
            break (
                ram,
                prg_ram,
                emulated,
                log.expect("a movie"),
                record.expect("an eval record"),
            );
        }
    };
    assert_eq!(log.origin.as_deref(), Some(start));
    assert_eq!((record.start_state.as_str(), record.seed), (start, 11));
    assert!(record.steps > 0 && record.frames > 0);
    // Room visits cover the episode: they start in the reset room and their steps, frames and health add up.
    let (first, last) = (&record.visits[0], record.visits.last().unwrap());
    let location = first.location;
    assert_eq!(
        (location.level, location.value, location.in_cave),
        record.room
    );
    assert_eq!(
        record.visits.iter().map(|v| v.steps as u64).sum::<u64>(),
        record.steps
    );
    assert_eq!(
        record.visits.iter().map(|v| v.frames).sum::<u64>(),
        record.frames
    );
    assert_eq!(
        (first.health_in, last.health_out),
        (record.start_health, record.health)
    );
    assert_eq!(
        record.to_json()["visits"].as_array().unwrap().len(),
        record.visits.len()
    );
    // Load randomization pokes come first; the log covers the gain-control frames and every step.
    assert!(log.pokes.iter().any(|p| p.0 == 0));
    assert!(log.buttons.len() as u64 >= record.frames);
    // zcore ran every frame since it powered on, look-ahead included; the log kept only the real ones.
    assert!(
        emulated > log.buttons.len() as u64,
        "look-ahead ran: {emulated} frames emulated"
    );
    assert_eq!(env.reset(obs.row(), &mut events).unwrap(), None, "one job");
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
    let mut buffers = pool.buffers();
    let mut finished = Vec::new();
    while pool.active() > 0 {
        let n = pool.recv(&mut buffers.batch(), 1, None).unwrap();
        let mut send = Vec::new();
        for row in 0..n {
            if buffers.finished[row] {
                finished.push(buffers.env_ids[row]);
            } else {
                send.push(buffers.env_ids[row] as usize);
            }
        }
        pool.send(&send, &vec![wait_action(); send.len()]).unwrap();
    }
    finished.sort();
    assert_eq!(finished, [0, 1], "both envs finish once the list is empty");
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
    let route = Route::load(&data.map, "milestone1", None, None).unwrap();
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
