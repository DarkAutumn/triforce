//! The Zelda adapters in a pool, on the dev-only FCEUmm oracle (skipped when the oracle library, ROM or legacy
//! states are missing). FCEUmm keeps its state in globals, so each test uses one env and holds `oracle::lock()`.

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
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use zelda::{ActionKind, ActionMask, MASK_SIZE, Nes, SessionSnapshot};

/// The oracle as a `zelda::Nes`; `.1` counts every frame run (look-ahead included). FCEUmm's globals have no
/// thread affinity; tests hold `oracle::lock()` and use one instance at a time, so moving it to a worker thread is
/// sound.
struct OracleNes(oracle::Oracle, u64);

unsafe impl Send for OracleNes {}

impl Nes for OracleNes {
    type State = Vec<u8>;

    fn run_frame(&mut self, buttons: u8, _render: bool) {
        self.0.run_frame(buttons);
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

    fn snapshot(&self) -> Vec<u8> {
        self.0.save_state()
    }

    fn restore(&mut self, state: &Vec<u8>) {
        self.0.load_state(state).expect("oracle load_state");
    }
}

/// Palette indices as gray RGB.
fn rgb(nes: &OracleNes, out: &mut [u8]) -> bool {
    let mut pixels = [0u8; 256 * 240];
    nes.0.pixels(&mut pixels);
    for (o, &p) in out.as_chunks_mut::<3>().0.iter_mut().zip(pixels.iter()) {
        o.fill(p);
    }
    true
}

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn open_oracle() -> Option<OracleNes> {
    if !oracle::available() {
        eprintln!("oracle library not built (oracle/fceumm/build.sh): skipping");
        return None;
    }
    let rom = std::env::var_os("TRIFORCE_ROM")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo().join("roms/zelda.nes"));
    if !rom.exists() {
        eprintln!("ROM not found: set TRIFORCE_ROM or create roms/zelda.nes");
        return None;
    }
    if !GzipStore.path("start").exists() {
        eprintln!("legacy states not found: skipping");
        return None;
    }
    Some(OracleNes(
        oracle::Oracle::open(&rom).expect("open oracle"),
        0,
    ))
}

/// The legacy gzip states (FCEUmm bodies the oracle loads, env's test fixtures) plus the oracle's 8-byte frame
/// counter.
struct GzipStore;

impl GzipStore {
    fn path(&self, name: &str) -> PathBuf {
        repo()
            .join("crates/env/tests/fixtures/legacy-states")
            .join(format!("{name}.state"))
    }
}

impl StateStore<Vec<u8>> for GzipStore {
    fn load(&self, name: &str) -> Result<Vec<u8>, EnvError> {
        let path = self.path(name);
        let file = std::fs::File::open(&path)
            .map_err(|e| EnvError::State(format!("{}: {e}", path.display())))?;
        let mut body = Vec::new();
        flate2::read::GzDecoder::new(file).read_to_end(&mut body)?;
        body.extend_from_slice(&0u64.to_le_bytes());
        Ok(body)
    }

    fn save(&self, _: &Path, _: &SessionSnapshot<Vec<u8>>) -> Result<(), EnvError> {
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
    let _lock = oracle::lock();
    let Some(nes) = open_oracle() else { return };
    let data = data();
    let config = config(&data, 12);
    let seed = 9;

    // Directly: the SpecialistEnv on the bare oracle, seeded like pool env 0.
    let mut direct =
        SpecialistEnv::new(config.clone(), data.clone(), nes, GzipStore, start_opts()).unwrap();
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
            GzipStore,
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
    let _lock = oracle::lock();
    let Some(nes) = open_oracle() else { return };
    let data = data();
    let route = Route::load(&data.map, "milestone1", None, None).unwrap();
    let seeds = [3u64, 4];
    let cap = 80;

    let mut direct = ChainEnv::new(data.clone(), nes, GzipStore, route.clone());
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
            GzipStore,
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
    let _lock = oracle::lock();
    let Some(nes) = open_oracle() else { return };
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
        GzipStore,
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
    let _lock = oracle::lock();
    let Some(nes) = open_oracle() else { return };
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
        start_state: Some("1_44w".into()),
        ..Default::default()
    };
    let mut nes = Some(nes);
    let pool = Pool::new(PoolOptions::new(1, 1, 0), |init| {
        TrainEnv::new(
            config.clone(),
            data.clone(),
            nes.take().unwrap(),
            GzipStore,
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

/// Replays a log on the bare oracle from its origin state (TFMOVIE poke order).
fn replay(nes: &mut OracleNes, log: &InputLog) {
    nes.restore(&GzipStore.load(log.origin.as_deref().unwrap()).unwrap());
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
    let _lock = oracle::lock();
    let Some(nes) = open_oracle() else { return };
    let data = data();
    // Eval: an explicit start and seed per episode. The Goriya room at full health (the NES full-health encoding)
    // with attacks preferred: sword beams fire, so the session runs its look-ahead; the health pokes land after
    // the gain-control frames.
    let start = "1_44w";
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
        GzipStore,
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
    // Load randomization pokes come first; the log covers the gain-control frames and every step.
    assert!(log.pokes.iter().any(|p| p.0 == 0));
    assert!(log.buttons.len() as u64 >= record.frames);
    // The oracle ran every frame since it opened, look-ahead included; the log kept only the real ones.
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
