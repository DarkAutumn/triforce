//! The pool on a fake env whose rows depend only on its seeded generator and its actions.

use pool::{
    Buffers, Dtype, EnvControl, EnvInit, KeySpec, ObsLayout, ObsRow, Outcome, Pool, PoolEnv,
    PoolError, PoolOptions, Start,
};
use rand::RngExt;
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;
use std::time::{Duration, Instant};

const ID: usize = 0;
const NOISE: usize = 1;
const MASK: usize = 2;

#[derive(Clone)]
struct Config {
    /// Fixed episode length (else drawn from the generator, 2..6).
    episode_len: Option<u32>,
    /// None: unlimited.
    max_episodes: Option<u32>,
    frames_per_step: u32,
    frame_delay: Duration,
    /// Sleep a few hundred microseconds of wall-clock-dependent jitter per step.
    jitter: bool,
    panic_at: Option<(usize, u64)>,
    error_at: Option<(usize, u64)>,
    /// Per-env delay per frame (env, delay).
    slow: Option<(usize, Duration)>,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            episode_len: None,
            max_episodes: None,
            frames_per_step: 1,
            frame_delay: Duration::ZERO,
            jitter: false,
            panic_at: None,
            error_at: None,
            slow: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Event {
    Ended {
        episode: u32,
        t: u32,
    },
    /// Frame `frame` of step `step` was rendered.
    Rendered {
        step: u64,
        frame: u32,
    },
}

struct Fake {
    index: usize,
    rng: ChaCha8Rng,
    control: EnvControl,
    cfg: Config,
    episode: u32,
    t: u32,
    len: u32,
    acc: i64,
    steps: u64,
    frames: u64,
}

fn layout() -> ObsLayout {
    ObsLayout::new(vec![
        KeySpec::new("id", &[5], Dtype::I64),
        KeySpec::new("noise", &[2, 3], Dtype::F32),
        KeySpec::new("mask", &[4], Dtype::Bool),
    ])
}

impl Fake {
    fn new(init: EnvInit, cfg: Config) -> Fake {
        Fake {
            index: init.index,
            rng: init.rng,
            control: init.control,
            cfg,
            episode: 0,
            t: 0,
            len: 0,
            acc: 0,
            steps: 0,
            frames: 0,
        }
    }

    fn write(&mut self, mut obs: ObsRow<'_>) {
        let id = [
            self.index as i64,
            self.episode as i64,
            self.t as i64,
            self.acc,
            self.rng.random_range(0..1000),
        ];
        obs.i64(ID).copy_from_slice(&id);
        for v in obs.f32(NOISE) {
            *v = self.rng.random();
        }
        for m in obs.bool(MASK) {
            *m = self.rng.random();
        }
    }
}

impl PoolEnv for Fake {
    type Action = i64;
    type Event = Event;
    type Error = String;

    fn layout(&self) -> ObsLayout {
        layout()
    }

    fn reset(&mut self, obs: ObsRow<'_>, _: &mut Vec<Event>) -> Result<Option<Start>, String> {
        if self.cfg.max_episodes.is_some_and(|m| self.episode >= m) {
            return Ok(None);
        }
        self.episode += 1;
        self.t = 0;
        self.acc = 0;
        self.len = match self.cfg.episode_len {
            Some(n) => n,
            None => self.rng.random_range(2..6),
        };
        self.write(obs);
        Ok(Some(Start {
            policy: (self.index % 2) as i64,
        }))
    }

    fn step(
        &mut self,
        action: i64,
        obs: ObsRow<'_>,
        events: &mut Vec<Event>,
    ) -> Result<Outcome, String> {
        self.steps += 1;
        if self.cfg.panic_at == Some((self.index, self.steps)) {
            panic!("boom in env {}", self.index);
        }
        if self.cfg.error_at == Some((self.index, self.steps)) {
            return Err(format!("bad step {}", self.steps));
        }
        if self.cfg.jitter {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .subsec_nanos();
            std::thread::sleep(Duration::from_micros((nanos % 400) as u64));
        }
        let delay = match self.cfg.slow {
            Some((env, d)) if env == self.index => d,
            _ => self.cfg.frame_delay,
        };
        for frame in 0..self.cfg.frames_per_step {
            let render = self.control.render();
            if !delay.is_zero() {
                std::thread::sleep(delay);
            }
            self.frames += 1;
            if render {
                events.push(Event::Rendered {
                    step: self.steps,
                    frame,
                });
            }
            let value = self.frames as u8;
            self.control.end_frame(render, |buf| {
                buf.fill(value);
                true
            });
        }
        self.acc = self.acc.wrapping_mul(31).wrapping_add(action);
        self.t += 1;
        let done = self.t >= self.len;
        let terminated = done && self.rng.random::<bool>();
        let truncated = done && !terminated;
        let outcome = Outcome {
            reward: self.rng.random::<f32>() + action as f32,
            frames: self.cfg.frames_per_step as i64,
            interrupt: action.rem_euclid(3),
            terminated,
            truncated,
            bootstrap: truncated && self.acc % 2 == 0,
            policy: (self.index % 2) as i64,
        };
        if done {
            events.push(Event::Ended {
                episode: self.episode,
                t: self.t,
            });
        }
        self.write(obs);
        Ok(outcome)
    }
}

fn pool(num_envs: usize, threads: usize, seed: u64, cfg: Config) -> Pool<Fake> {
    pool_with(PoolOptions::new(num_envs, threads, seed), cfg)
}

fn pool_with(opts: PoolOptions, cfg: Config) -> Pool<Fake> {
    Pool::new(opts, |init| Ok(Fake::new(init, cfg.clone()))).unwrap()
}

fn ids(b: &Buffers, n: usize) -> Vec<usize> {
    b.env_ids[..n].iter().map(|&i| i as usize).collect()
}

/// Row `row`'s id key.
fn id(b: &Buffers, row: usize) -> Vec<i64> {
    b.obs[ID].i64()[b.row_range(ID, row)].to_vec()
}

fn final_id(b: &Buffers, row: usize) -> Vec<i64> {
    b.final_obs[ID].i64()[b.row_range(ID, row)].to_vec()
}

/// An action that depends on the row's contents only.
fn policy(b: &Buffers, row: usize) -> i64 {
    let id = id(b, row);
    (id[3].rem_euclid(7) + id[4] + b.env_ids[row]) % 5
}

type Trace = (Vec<Buffers>, Vec<Vec<Event>>);

fn lockstep(threads: usize, seed: u64, iterations: usize) -> Trace {
    let n = 8;
    let cfg = Config {
        jitter: true,
        ..Config::default()
    };
    let pool = pool(n, threads, seed, cfg);
    assert_eq!(pool.threads(), threads.min(n));
    let mut buffers = pool.buffers();
    let mut trace = Vec::new();
    let mut events = vec![Vec::new(); n];
    for _ in 0..iterations {
        let got = pool.recv(&mut buffers.batch(), n, None).unwrap();
        assert_eq!(got, n);
        assert_eq!(ids(&buffers, n), (0..n).collect::<Vec<_>>());
        for (env, e) in pool.poll_events() {
            events[env].push(e);
        }
        trace.push(buffers.clone());
        let actions: Vec<i64> = (0..n).map(|r| policy(&buffers, r)).collect();
        pool.send(&ids(&buffers, n), &actions).unwrap();
    }
    (trace, events)
}

#[test]
fn lockstep_is_deterministic_across_thread_counts() {
    let one = lockstep(1, 42, 40);
    assert!(
        one.1.iter().all(|e| !e.is_empty()),
        "every env ended an episode"
    );
    assert!(one.0.iter().any(|b| b.final_valid.iter().any(|&v| v)));
    for threads in [3, 8] {
        let other = lockstep(threads, 42, 40);
        assert!(one.0 == other.0, "buffers differ with {threads} threads");
        assert_eq!(one.1, other.1, "events differ with {threads} threads");
    }
    let reseeded = lockstep(3, 43, 40);
    assert!(one.0 != reseeded.0, "the seed changes the rows");
}

#[test]
fn envs_get_distinct_streams_of_the_pool_seed() {
    let a = pool::env_rng(5, 0);
    let b = pool::env_rng(5, 1);
    assert_eq!(a.get_seed(), b.get_seed());
    assert_ne!(a.get_stream(), b.get_stream());
    let mut a2 = pool::env_rng(5, 0);
    let mut a = a;
    assert_eq!(a.random::<u64>(), a2.random::<u64>());
}

#[test]
fn recv_returns_as_soon_as_min_ready_envs_are_ready() {
    let cfg = Config {
        slow: Some((1, Duration::from_millis(600))),
        ..Config::default()
    };
    let pool = pool(2, 2, 0, cfg);
    let mut b = pool.buffers();
    assert_eq!(pool.recv(&mut b.batch(), 2, None).unwrap(), 2);
    pool.send(&[0, 1], &[1, 1]).unwrap();
    let t0 = Instant::now();
    assert_eq!(pool.recv(&mut b.batch(), 1, None).unwrap(), 1);
    assert!(
        t0.elapsed() < Duration::from_millis(300),
        "{:?}",
        t0.elapsed()
    );
    assert_eq!(b.env_ids[0], 0);
    // Env 0 now awaits an action, so only env 1 can become ready.
    assert_eq!(pool.recv(&mut b.batch(), 2, None).unwrap(), 1);
    assert_eq!(b.env_ids[0], 1);
    assert!(t0.elapsed() >= Duration::from_millis(500));
}

#[test]
fn recv_and_wait_time_out() {
    let cfg = Config {
        frame_delay: Duration::from_millis(400),
        ..Config::default()
    };
    let pool = pool(1, 1, 0, cfg);
    let mut b = pool.buffers();
    assert_eq!(pool.recv(&mut b.batch(), 1, None).unwrap(), 1);
    pool.send(&[0], &[0]).unwrap();
    let t0 = Instant::now();
    assert!(!pool.wait(1, Some(Duration::from_millis(30))).unwrap());
    assert_eq!(
        pool.recv(&mut b.batch(), 1, Some(Duration::from_millis(30)))
            .unwrap(),
        0
    );
    assert!(t0.elapsed() < Duration::from_millis(300));
    // min_ready 0 polls.
    assert_eq!(pool.recv(&mut b.batch(), 0, None).unwrap(), 0);
    assert!(pool.wait(1, None).unwrap());
    assert_eq!(
        pool.recv(&mut b.batch(), 1, Some(Duration::ZERO)).unwrap(),
        1
    );
}

#[test]
fn auto_reset_keeps_the_final_observation_and_returns_the_reset_one() {
    let cfg = Config {
        episode_len: Some(3),
        ..Config::default()
    };
    let pool = pool(1, 1, 7, cfg);
    let mut b = pool.buffers();
    assert_eq!(pool.recv(&mut b.batch(), 1, None).unwrap(), 1);
    assert_eq!(&id(&b, 0)[..4], [0, 1, 0, 0]);
    assert!(!b.final_valid[0] && !b.terminated[0] && !b.truncated[0]);
    assert_eq!((b.reward[0], b.frames[0], b.policy[0]), (0.0, 0, 0));
    let mut acc = 0i64;
    for (t, action) in [(1, 4), (2, 2)] {
        pool.send(&[0], &[action]).unwrap();
        pool.recv(&mut b.batch(), 1, None).unwrap();
        acc = acc * 31 + action;
        assert_eq!(&id(&b, 0)[..4], [0, 1, t, acc]);
        assert!(!b.final_valid[0] && !b.terminated[0] && !b.truncated[0]);
        assert_eq!(b.frames[0], 1);
        assert_eq!(b.interrupt[0], action % 3);
    }
    pool.send(&[0], &[3]).unwrap();
    pool.recv(&mut b.batch(), 1, None).unwrap();
    assert!(b.terminated[0] != b.truncated[0]);
    assert!(b.final_valid[0]);
    assert_eq!(&final_id(&b, 0)[..4], [0, 1, 3, acc * 31 + 3]);
    assert_eq!(&id(&b, 0)[..4], [0, 2, 0, 0]);
    assert!(b.reward[0] >= 3.0);
    assert_eq!(pool.poll_events(), [(0, Event::Ended { episode: 1, t: 3 })]);
    pool.send(&[0], &[1]).unwrap();
    pool.recv(&mut b.batch(), 1, None).unwrap();
    assert!(!b.final_valid[0]);
    assert_eq!(&id(&b, 0)[..4], [0, 2, 1, 1]);
}

fn rendered(events: &[(usize, Event)]) -> Vec<u32> {
    events
        .iter()
        .filter_map(|(_, e)| match e {
            Event::Rendered { frame, .. } => Some(*frame),
            _ => None,
        })
        .collect()
}

#[test]
fn set_render_takes_effect_at_the_next_frame() {
    let cfg = Config {
        frames_per_step: 100,
        frame_delay: Duration::from_millis(2),
        ..Config::default()
    };
    let pool = pool(1, 1, 0, cfg);
    let mut b = pool.buffers();
    let mut frame = vec![0u8; pool.frame_bytes()];
    assert_eq!(pool.frame_bytes(), 240 * 256 * 3);
    pool.recv(&mut b.batch(), 1, None).unwrap();
    // Off: no frames.
    pool.send(&[0], &[0]).unwrap();
    pool.recv(&mut b.batch(), 1, None).unwrap();
    assert!(rendered(&pool.poll_events()).is_empty());
    assert_eq!(pool.frame_counter(0).unwrap(), 0);
    assert_eq!(pool.frame(0, 0, &mut frame).unwrap(), None);
    // On mid-step: from the next frame to the end of the step.
    pool.send(&[0], &[0]).unwrap();
    std::thread::sleep(Duration::from_millis(60));
    pool.set_render(0, true).unwrap();
    pool.recv(&mut b.batch(), 1, None).unwrap();
    let on = rendered(&pool.poll_events());
    assert!(!on.is_empty() && on[0] > 0, "{on:?}");
    assert_eq!(on, (on[0]..100).collect::<Vec<_>>());
    let counter = pool.frame_counter(0).unwrap();
    assert_eq!(counter, on.len() as u64);
    assert_eq!(pool.frame(0, 0, &mut frame).unwrap(), Some(counter));
    assert!(frame.iter().all(|&v| v == 200), "the last frame of step 2");
    assert_eq!(pool.frame(0, counter, &mut frame).unwrap(), None);
    // Off mid-step: frames stop at the next frame.
    pool.send(&[0], &[0]).unwrap();
    std::thread::sleep(Duration::from_millis(60));
    pool.set_render(0, false).unwrap();
    pool.recv(&mut b.batch(), 1, None).unwrap();
    let off = rendered(&pool.poll_events());
    assert!(!off.is_empty() && *off.last().unwrap() < 99, "{off:?}");
    assert_eq!(off, (0..off.len() as u32).collect::<Vec<_>>());
    let newer = pool.frame(0, counter, &mut frame).unwrap().unwrap();
    assert_eq!(newer, counter + off.len() as u64);
    assert!(frame.iter().all(|&v| v == (200 + off.len()) as u8));
    assert!(matches!(
        pool.frame(0, 0, &mut [0u8; 3]),
        Err(PoolError::Usage(_))
    ));
    assert!(matches!(pool.set_render(1, true), Err(PoolError::Usage(_))));
}

fn timed_steps(fps: f64, steps: usize) -> Duration {
    let cfg = Config {
        frames_per_step: 30,
        ..Config::default()
    };
    let mut opts = PoolOptions::new(1, 1, 0);
    opts.fps = fps;
    let pool = pool_with(opts, cfg);
    let mut b = pool.buffers();
    pool.recv(&mut b.batch(), 1, None).unwrap();
    let t0 = Instant::now();
    for _ in 0..steps {
        pool.send(&[0], &[0]).unwrap();
        pool.recv(&mut b.batch(), 1, None).unwrap();
    }
    t0.elapsed()
}

#[test]
fn pacing_caps_the_frame_rate() {
    // 60 frames at 100 fps: ~0.6 s.
    let paced = timed_steps(100.0, 2);
    assert!(
        paced >= Duration::from_millis(500) && paced < Duration::from_secs(3),
        "{paced:?}"
    );
    let free = timed_steps(0.0, 2);
    assert!(free < Duration::from_millis(200), "{free:?}");
}

#[test]
fn set_fps_changes_pacing_from_the_next_frame() {
    let cfg = Config {
        frames_per_step: 20,
        ..Config::default()
    };
    let pool = pool(1, 1, 0, cfg);
    let mut b = pool.buffers();
    pool.recv(&mut b.batch(), 1, None).unwrap();
    pool.set_fps(0, 50.0).unwrap();
    assert_eq!(pool.control(0).unwrap().fps(), 50.0);
    let t0 = Instant::now();
    pool.send(&[0], &[0]).unwrap();
    pool.recv(&mut b.batch(), 1, None).unwrap();
    let paced = t0.elapsed();
    assert!(paced >= Duration::from_millis(300), "{paced:?}");
    pool.set_fps(0, 0.0).unwrap();
    let t0 = Instant::now();
    pool.send(&[0], &[0]).unwrap();
    pool.recv(&mut b.batch(), 1, None).unwrap();
    assert!(t0.elapsed() < Duration::from_millis(100));
}

#[test]
fn drop_stops_paced_and_busy_workers_promptly() {
    let cfg = Config {
        frames_per_step: 100,
        ..Config::default()
    };
    // Paced at 10 fps a step would take 10 s; closing stops the pacing sleeps.
    let mut opts = PoolOptions::new(4, 2, 0);
    opts.fps = 10.0;
    let pool = pool_with(opts, cfg);
    let mut b = pool.buffers();
    assert_eq!(pool.recv(&mut b.batch(), 4, None).unwrap(), 4);
    pool.send(&[0, 1, 2, 3], &[0, 0, 0, 0]).unwrap();
    std::thread::sleep(Duration::from_millis(50));
    let t0 = Instant::now();
    std::thread::scope(|s| {
        let waiter = s.spawn(|| {
            let mut b = pool.buffers();
            pool.recv(&mut b.batch(), 4, None)
        });
        std::thread::sleep(Duration::from_millis(50));
        pool.close();
        assert!(matches!(waiter.join().unwrap(), Err(PoolError::Closed)));
    });
    assert!(t0.elapsed() < Duration::from_secs(2), "{:?}", t0.elapsed());
    assert!(matches!(pool.send(&[0], &[0]), Err(PoolError::Closed)));
    drop(pool);

    // A busy (unpaced) step finishes before the worker exits.
    let cfg = Config {
        frames_per_step: 100,
        frame_delay: Duration::from_millis(2),
        ..Config::default()
    };
    let pool = pool_with(PoolOptions::new(2, 2, 0), cfg);
    pool.recv(&mut b.batch(), 2, None).unwrap();
    pool.send(&[0, 1], &[0, 0]).unwrap();
    let t0 = Instant::now();
    drop(pool);
    assert!(t0.elapsed() < Duration::from_secs(2));

    // Dropping right after creation (resets still queued) doesn't hang either.
    for threads in [1, 4] {
        drop(pool_with(
            PoolOptions::new(8, threads, 0),
            Config::default(),
        ));
    }
}

#[test]
fn an_env_panic_is_an_error_not_a_hang() {
    let cfg = Config {
        panic_at: Some((1, 2)),
        ..Config::default()
    };
    let pool = pool(3, 2, 0, cfg);
    let mut b = pool.buffers();
    let mut error = None;
    for _ in 0..5 {
        match pool.recv(&mut b.batch(), 3, None) {
            Ok(n) => {
                let actions = vec![1; n];
                pool.send(&ids(&b, n), &actions).unwrap();
            }
            Err(e) => {
                error = Some(e);
                break;
            }
        }
    }
    match error {
        Some(PoolError::Panic { env: 1, message }) => {
            assert!(message.contains("boom"), "{message}")
        }
        other => panic!("expected env 1's panic, got {other:?}"),
    }
    assert_eq!(pool.active(), 2);
    // The others keep going; min_ready is capped by what can still become ready.
    let n = pool
        .recv(&mut b.batch(), 3, Some(Duration::from_secs(5)))
        .unwrap();
    assert_eq!(ids(&b, n), [0, 2]);
    assert!(matches!(pool.send(&[1], &[0]), Err(PoolError::Usage(_))));
    pool.send(&[0, 2], &[0, 0]).unwrap();
    assert_eq!(pool.recv(&mut b.batch(), 3, None).unwrap(), 2);
}

#[test]
fn an_env_error_fails_that_env() {
    let cfg = Config {
        error_at: Some((0, 1)),
        ..Config::default()
    };
    let pool = pool(2, 1, 0, cfg);
    let mut b = pool.buffers();
    assert_eq!(pool.recv(&mut b.batch(), 2, None).unwrap(), 2);
    pool.send(&[0, 1], &[0, 0]).unwrap();
    match pool.recv(&mut b.batch(), 2, None) {
        Err(PoolError::Env { env: 0, error }) => assert_eq!(error, "bad step 1"),
        other => panic!("expected env 0's error, got {other:?}"),
    }
    assert_eq!(pool.recv(&mut b.batch(), 2, None).unwrap(), 1);
    assert_eq!(b.env_ids[0], 1);
}

#[test]
fn a_factory_error_fails_the_pool() {
    let made = Pool::<Fake>::new(PoolOptions::new(3, 2, 0), |init| {
        if init.index == 2 {
            Err("no env 2".to_owned())
        } else {
            Ok(Fake::new(init, Config::default()))
        }
    });
    assert!(matches!(made, Err(PoolError::Env { env: 2, .. })));
    assert!(matches!(
        Pool::<Fake>::new(PoolOptions::new(0, 1, 0), |init| Ok(Fake::new(
            init,
            Config::default()
        ))),
        Err(PoolError::Usage(_))
    ));
}

#[test]
fn finished_envs_leave_the_pool() {
    let cfg = Config {
        episode_len: Some(2),
        max_episodes: Some(1),
        ..Config::default()
    };
    let pool = pool(3, 2, 0, cfg);
    let mut b = pool.buffers();
    assert_eq!(pool.active(), 3);
    let mut finished = Vec::new();
    while pool.active() > 0 {
        let n = pool.recv(&mut b.batch(), 3, None).unwrap();
        let mut send = Vec::new();
        for row in 0..n {
            if b.finished[row] {
                assert!(b.final_valid[row]);
                assert_eq!(b.policy[row], -1);
                finished.push(b.env_ids[row]);
            } else {
                send.push(b.env_ids[row] as usize);
            }
        }
        pool.send(&send, &vec![0; send.len()]).unwrap();
    }
    finished.sort();
    assert_eq!(finished, [0, 1, 2]);
    assert_eq!(pool.recv(&mut b.batch(), 3, None).unwrap(), 0);
    assert!(matches!(pool.send(&[0], &[0]), Err(PoolError::Usage(_))));
}

#[test]
fn send_and_recv_check_their_arguments() {
    let pool = pool(3, 2, 0, Config::default());
    let mut b = pool.buffers();
    assert_eq!(pool.recv(&mut b.batch(), 3, None).unwrap(), 3);
    for (envs, actions) in [
        (&[0, 0][..], &[0, 0][..]),
        (&[0, 5][..], &[0, 0][..]),
        (&[0][..], &[0, 1][..]),
    ] {
        assert!(matches!(pool.send(envs, actions), Err(PoolError::Usage(_))));
    }
    // A rejected send changes nothing.
    pool.send(&[0, 1, 2], &[0, 0, 0]).unwrap();
    assert!(matches!(pool.send(&[1], &[0]), Err(PoolError::Usage(_))));
    // Small batches take the oldest rows first, written in env order.
    let mut small = Buffers::new(pool.layout().clone(), 2);
    let n = pool.recv(&mut small.batch(), 3, None).unwrap();
    assert_eq!(n, 2);
    assert!(small.env_ids[0] < small.env_ids[1]);
    assert_eq!(pool.recv(&mut small.batch(), 1, None).unwrap(), 1);
    let mut bad = pool.buffers();
    bad.reward.pop();
    assert!(matches!(
        pool.recv(&mut bad.batch(), 1, None),
        Err(PoolError::Usage(_))
    ));
    let mut wrong = Buffers::new(Arc::new(ObsLayout::new(vec![])), 3);
    assert!(matches!(
        pool.recv(&mut wrong.batch(), 1, None),
        Err(PoolError::Usage(_))
    ));
    let spec = pool.buffer_spec(3);
    assert_eq!(spec.obs, layout().keys());
    assert_eq!(spec.scalars[0], ("env_ids", Dtype::I64));
}
