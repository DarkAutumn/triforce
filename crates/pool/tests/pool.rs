//! The pool on a fake env whose rows depend only on its seeded generator and its actions, acting with a test
//! actor whose actions depend only on the observation.

use pool::rollout::{Actor, Decision, EnvRollout, Play, Record};
use pool::{
    Dtype, EnvControl, EnvInit, KeySpec, ObsData, ObsLayout, ObsRow, Outcome, Pool, PoolEnv, PoolError,
    PoolOptions, Start,
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
    /// Hold the env's frame at the end of every episode (the watch player's result pause).
    hold_after_episode: bool,
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
            hold_after_episode: false,
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
            job: self.index as i64 * 1000 + i64::from(self.episode),
        }))
    }

    fn step(&mut self, action: i64, obs: ObsRow<'_>, events: &mut Vec<Event>) -> Result<Outcome, String> {
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
        for frame in 0..self.cfg.frames_per_step {
            let render = self.control.render();
            if !self.cfg.frame_delay.is_zero() {
                std::thread::sleep(self.cfg.frame_delay);
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
            menu_frames: 0,
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
            if self.cfg.hold_after_episode {
                self.control.set_hold(true);
            }
        }
        self.write(obs);
        Ok(outcome)
    }
}

/// Actions from the observation only; the decision reports (job, step, policy) so tests can read them back, and
/// the value of an observation is its `acc`.
struct TestActor;

impl Actor<i64> for TestActor {
    type Scratch = ();

    fn scratch(&self) {}

    fn act(&self, obs: &ObsData, policy: i64, job: i64, step: i64, _: &mut ()) -> Result<(i64, Decision), String> {
        let id = obs.i64(ID);
        let action = (id[3].rem_euclid(7) + id[4] + id[0]) % 5;
        let decision = Decision {
            log_prob: job as f32,
            entropy: step as f32,
            value: policy as f32,
        };
        Ok((action, decision))
    }

    fn value(&self, obs: &ObsData, _: i64, _: &mut ()) -> Result<f32, String> {
        Ok(obs.i64(ID)[3] as f32)
    }
}

fn pool(num_envs: usize, threads: usize, seed: u64, cfg: Config) -> Pool<Fake> {
    pool_with(PoolOptions::new(num_envs, threads, seed), cfg)
}

fn pool_with(opts: PoolOptions, cfg: Config) -> Pool<Fake> {
    Pool::new(opts, |init| Ok(Fake::new(init, cfg.clone()))).unwrap()
}

fn actor() -> Arc<TestActor> {
    Arc::new(TestActor)
}

/// Row `t`'s id key.
fn id(r: &EnvRollout<i64>, t: usize) -> &[i64] {
    &r.obs.columns[ID].i64()[t * 5..(t + 1) * 5]
}

type Trace = (Vec<Vec<(usize, EnvRollout<i64>)>>, Vec<Vec<Event>>);

fn rollouts(threads: usize, seed: u64, count: usize, steps: usize) -> Trace {
    let n = 8;
    let cfg = Config {
        jitter: true,
        ..Config::default()
    };
    let pool = pool(n, threads, seed, cfg);
    assert_eq!(pool.threads(), threads.min(n));
    let mut trace = Vec::new();
    let mut events = vec![Vec::new(); n];
    for _ in 0..count {
        let got = pool.rollout(steps, actor()).unwrap();
        assert_eq!(got.iter().map(|(env, _)| *env).collect::<Vec<_>>(), (0..n).collect::<Vec<_>>());
        trace.push(got);
        for (env, e) in pool.poll_events() {
            events[env].push(e);
        }
    }
    (trace, events)
}

#[test]
fn rollouts_are_deterministic_across_thread_counts() {
    let one = rollouts(1, 42, 3, 12);
    assert!(one.1.iter().all(|e| !e.is_empty()), "every env ended an episode");
    for threads in [3, 8] {
        let other = rollouts(threads, 42, 3, 12);
        assert!(one.0 == other.0, "rollouts differ with {threads} threads");
        assert_eq!(one.1, other.1, "events differ with {threads} threads");
    }
    let reseeded = rollouts(3, 43, 3, 12);
    assert!(one.0 != reseeded.0, "the seed changes the rollouts");
}

#[test]
fn rollouts_carry_keys_auto_reset_bootstrap_values_and_resume() {
    let (trace, _) = rollouts(3, 7, 2, 15);
    let mut restarted = false;
    let mut bootstrapped = false;
    for (env, r) in &trace[0] {
        assert_eq!(r.steps(), 15);
        for t in 0..15 {
            let row = id(r, t);
            assert_eq!(row[0], *env as i64);
            assert_eq!(r.job[t], row[0] * 1000 + row[1], "the episode's key");
            assert_eq!(r.step[t], row[2], "decisions taken before the row's observation");
            assert_eq!(r.starts[t], row[2] == 0);
            let d = r.decisions[t];
            assert_eq!((d.log_prob, d.entropy), (r.job[t] as f32, r.step[t] as f32), "the actor saw the keys");
            assert_eq!(r.interrupt[t], r.actions[t] % 3);
            let ended = r.terminated[t] || r.truncated[t];
            assert_eq!(id(r, t + 1)[2], if ended { 0 } else { row[2] + 1 }, "a terminal step resets at once");
            restarted |= ended;
            // A bootstrapped truncation gets the actor's value of the final observation (its acc).
            let final_acc = row[3].wrapping_mul(31).wrapping_add(r.actions[t]);
            let expected = if r.truncated[t] && final_acc % 2 == 0 { final_acc as f32 } else { 0.0 };
            assert_eq!(r.bootstrap_value[t], expected);
            bootstrapped |= expected != 0.0;
        }
    }
    assert!(restarted && bootstrapped);
    // The next rollout resumes from each env's last row.
    for ((_, a), (_, b)) in trace[0].iter().zip(&trace[1]) {
        assert_eq!(id(a, 15), id(b, 0));
    }
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

fn plays(threads: usize, record: Record) -> Vec<Play<i64>> {
    let cfg = Config {
        max_episodes: Some(3),
        jitter: true,
        ..Config::default()
    };
    let pool = pool(5, threads, 9, cfg);
    pool.play(actor(), record).unwrap();
    assert!(pool.wait_idle(None).unwrap());
    assert_eq!(pool.active(), 0, "every env ran out of episodes");
    let mut plays: Vec<Play<i64>> = pool.poll_plays().into_iter().map(|(_, p)| p).collect();
    plays.sort_by_key(|p| p.job);
    plays
}

#[test]
fn plays_record_every_episode_whatever_the_thread_count() {
    let recorded = plays(1, Record::Observations);
    assert_eq!(recorded.iter().map(|p| p.job).collect::<Vec<_>>(),
               (0..5).flat_map(|e| (1..=3).map(move |k| e * 1000 + k)).collect::<Vec<_>>());
    for p in &recorded {
        let obs = p.obs.as_ref().unwrap();
        assert_eq!((obs.rows, p.decisions.len(), p.policy.len()), (p.steps(), p.steps(), p.steps()));
        for t in 0..p.steps() {
            let row = &obs.columns[ID].i64()[t * 5..(t + 1) * 5];
            assert_eq!((p.job, t as i64), (row[0] * 1000 + row[1], row[2]), "decision t of this episode");
            assert_eq!(p.decisions[t].entropy, t as f32);
            assert_eq!(p.policy[t], row[0] % 2);
        }
    }
    assert!(recorded == plays(4, Record::Observations));
    let actions = plays(2, Record::Actions);
    assert!(actions.iter().all(|p| p.obs.is_none()));
    assert!(recorded.iter().zip(&actions).all(|(a, b)| a.actions == b.actions && a.decisions == b.decisions));
}

#[test]
fn a_held_env_parks_after_its_episode_until_released() {
    let cfg = Config {
        max_episodes: Some(2),
        episode_len: Some(3),
        hold_after_episode: true,
        ..Config::default()
    };
    let pool = pool(2, 2, 0, cfg);
    pool.play(actor(), Record::Actions).unwrap();
    assert!(pool.wait_idle(Some(Duration::from_secs(10))).unwrap(), "both envs parked");
    assert_eq!((pool.active(), pool.poll_plays().len()), (2, 2), "one episode each, then parked");
    std::thread::sleep(Duration::from_millis(50));
    assert!(pool.poll_plays().is_empty(), "a parked env doesn't act");
    pool.set_hold(1, false).unwrap();
    assert!(pool.wait_idle(Some(Duration::from_secs(10))).unwrap());
    let plays = pool.poll_plays();
    assert_eq!(plays.iter().map(|(env, p)| (*env, p.job)).collect::<Vec<_>>(), [(1, 1002)]);
    assert_eq!(pool.active(), 1, "env 1 ran out of episodes; env 0 is still parked");
    pool.set_hold(0, false).unwrap();
    assert!(pool.wait_idle(Some(Duration::from_secs(10))).unwrap());
    assert_eq!(pool.active(), 0);
    assert!(matches!(pool.rollout(1, actor()), Err(PoolError::Usage(_))), "a playing pool doesn't roll out");
    assert!(matches!(pool.play(actor(), Record::Off), Err(PoolError::Usage(_))));
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

/// One decision with `between` run on this thread while it is in flight.
fn step_while(pool: &Pool<Fake>, between: impl FnOnce()) {
    std::thread::scope(|s| {
        let step = s.spawn(|| pool.rollout(1, actor()).unwrap());
        between();
        step.join().unwrap();
    });
}

#[test]
fn set_render_takes_effect_at_the_next_frame() {
    let cfg = Config {
        frames_per_step: 100,
        frame_delay: Duration::from_millis(2),
        ..Config::default()
    };
    let pool = pool(1, 1, 0, cfg);
    let mut frame = vec![0u8; pool.frame_bytes()];
    assert_eq!(pool.frame_bytes(), 240 * 256 * 3);
    // Off: no frames.
    pool.rollout(1, actor()).unwrap();
    assert!(rendered(&pool.poll_events()).is_empty());
    assert_eq!(pool.frame_counter(0).unwrap(), 0);
    assert_eq!(pool.frame(0, 0, &mut frame).unwrap(), None);
    // On mid-step: from the next frame to the end of the step.
    step_while(&pool, || {
        std::thread::sleep(Duration::from_millis(60));
        pool.set_render(0, true).unwrap();
    });
    let on = rendered(&pool.poll_events());
    assert!(!on.is_empty() && on[0] > 0, "{on:?}");
    assert_eq!(on, (on[0]..100).collect::<Vec<_>>());
    let counter = pool.frame_counter(0).unwrap();
    assert_eq!(counter, on.len() as u64);
    assert_eq!(pool.frame(0, 0, &mut frame).unwrap(), Some(counter));
    assert!(frame.iter().all(|&v| v == 200), "the last frame of step 2");
    assert_eq!(pool.frame(0, counter, &mut frame).unwrap(), None);
    // Off mid-step: frames stop at the next frame.
    step_while(&pool, || {
        std::thread::sleep(Duration::from_millis(60));
        pool.set_render(0, false).unwrap();
    });
    let off = rendered(&pool.poll_events());
    assert!(!off.is_empty() && *off.last().unwrap() < 99, "{off:?}");
    assert_eq!(off, (0..off.len() as u32).collect::<Vec<_>>());
    let newer = pool.frame(0, counter, &mut frame).unwrap().unwrap();
    assert_eq!(newer, counter + off.len() as u64);
    assert!(frame.iter().all(|&v| v == (200 + off.len()) as u8));
    assert!(matches!(pool.frame(0, 0, &mut [0u8; 3]), Err(PoolError::Usage(_))));
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
    assert!(pool.wait_idle(None).unwrap());
    let t0 = Instant::now();
    pool.rollout(steps, actor()).unwrap();
    t0.elapsed()
}

#[test]
fn pacing_caps_the_frame_rate() {
    // 60 frames at 100 fps: ~0.6 s.
    let paced = timed_steps(100.0, 2);
    assert!(paced >= Duration::from_millis(500) && paced < Duration::from_secs(3), "{paced:?}");
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
    pool.set_fps(0, 50.0).unwrap();
    assert_eq!(pool.control(0).unwrap().fps(), 50.0);
    assert!(pool.wait_idle(None).unwrap());
    let t0 = Instant::now();
    pool.rollout(1, actor()).unwrap();
    assert!(t0.elapsed() >= Duration::from_millis(300), "{:?}", t0.elapsed());
    pool.set_fps(0, 0.0).unwrap();
    let t0 = Instant::now();
    pool.rollout(1, actor()).unwrap();
    assert!(t0.elapsed() < Duration::from_millis(100));
}

#[test]
fn emulated_frames_count_every_played_decision() {
    let cfg = Config {
        frames_per_step: 7,
        max_episodes: Some(2),
        episode_len: Some(3),
        ..Config::default()
    };
    let pool = pool(2, 1, 0, cfg);
    pool.play(actor(), Record::Off).unwrap();
    assert!(pool.wait_idle(None).unwrap());
    assert_eq!(pool.control(1).unwrap().emulated_frames(), 2 * 3 * 7);
}

#[test]
fn close_stops_paced_and_busy_workers_promptly() {
    let cfg = Config {
        frames_per_step: 100,
        ..Config::default()
    };
    // Paced at 10 fps a step would take 10 s; closing stops the pacing sleeps.
    let mut opts = PoolOptions::new(4, 2, 0);
    opts.fps = 10.0;
    let pool = pool_with(opts, cfg);
    let t0 = Instant::now();
    std::thread::scope(|s| {
        let waiter = s.spawn(|| pool.rollout(5, actor()));
        std::thread::sleep(Duration::from_millis(100));
        pool.close();
        assert!(matches!(waiter.join().unwrap(), Err(PoolError::Closed)));
    });
    assert!(t0.elapsed() < Duration::from_secs(2), "{:?}", t0.elapsed());
    assert!(matches!(pool.rollout(1, actor()), Err(PoolError::Closed)));
    assert!(matches!(pool.wait_idle(None), Err(PoolError::Closed)));
    drop(pool);

    // A playing pool closes too, while its envs act and while one is parked.
    let cfg = Config {
        frames_per_step: 100,
        frame_delay: Duration::from_millis(1),
        episode_len: Some(1),
        hold_after_episode: true,
        ..Config::default()
    };
    let pool = pool_with(PoolOptions::new(2, 2, 0), cfg);
    pool.play(actor(), Record::Actions).unwrap();
    std::thread::sleep(Duration::from_millis(150));
    let t0 = Instant::now();
    drop(pool);
    assert!(t0.elapsed() < Duration::from_secs(2));

    // Dropping right after creation (resets still queued) doesn't hang either.
    for threads in [1, 4] {
        drop(pool_with(PoolOptions::new(8, threads, 0), Config::default()));
    }
}

#[test]
fn an_env_panic_is_an_error_not_a_hang() {
    let cfg = Config {
        panic_at: Some((1, 2)),
        ..Config::default()
    };
    let pool = pool(3, 2, 0, cfg);
    match pool.rollout(5, actor()) {
        Err(PoolError::Panic { env: 1, message }) => assert!(message.contains("boom"), "{message}"),
        other => panic!("expected env 1's panic, got {other:?}"),
    }
    assert_eq!(pool.active(), 2);
    // The others keep going.
    let got = pool.rollout(5, actor()).unwrap();
    assert_eq!(got.iter().map(|(env, _)| *env).collect::<Vec<_>>(), [0, 2]);
}

#[test]
fn an_env_error_fails_that_env() {
    let cfg = Config {
        error_at: Some((0, 1)),
        ..Config::default()
    };
    let pool = pool(2, 1, 0, cfg);
    pool.play(actor(), Record::Off).unwrap();
    match pool.wait_idle(Some(Duration::from_secs(10))) {
        Err(PoolError::Env { env: 0, error }) => assert_eq!(error, "bad step 1"),
        other => panic!("expected env 0's error, got {other:?}"),
    }
    assert_eq!(pool.active(), 1, "env 1 plays on");
}

#[test]
fn an_actor_error_fails_the_call_not_the_env() {
    struct Refuses;
    impl Actor<i64> for Refuses {
        type Scratch = ();
        fn scratch(&self) {}
        fn act(&self, _: &ObsData, policy: i64, _: i64, _: i64, _: &mut ()) -> Result<(i64, Decision), String> {
            if policy == 1 { Err("no policy 1".into()) } else { Ok((0, Decision::default())) }
        }
        fn value(&self, _: &ObsData, _: i64, _: &mut ()) -> Result<f32, String> {
            Ok(0.0)
        }
    }
    let pool = pool(2, 2, 0, Config::default());
    match pool.rollout(3, Arc::new(Refuses)) {
        Err(PoolError::Usage(message)) => assert_eq!(message, "env 1: no policy 1"),
        other => panic!("expected env 1's actor error, got {other:?}"),
    }
    assert_eq!(pool.active(), 2, "an actor error leaves the env usable");
    assert!(pool.wait_idle(None).unwrap(), "the rollout reported the error; nothing is left queued");
    let got = pool.rollout(3, actor()).unwrap();
    assert_eq!(got.iter().map(|(env, _)| *env).collect::<Vec<_>>(), [0, 1]);
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
        Pool::<Fake>::new(PoolOptions::new(0, 1, 0), |init| Ok(Fake::new(init, Config::default()))),
        Err(PoolError::Usage(_))
    ));
}

#[test]
fn finished_envs_leave_the_pool() {
    let cfg = Config {
        max_episodes: Some(0),
        ..Config::default()
    };
    let pool = pool(3, 2, 0, cfg);
    assert!(pool.wait_idle(None).unwrap());
    assert_eq!(pool.active(), 0, "no env had an episode");
    assert!(matches!(pool.rollout(4, actor()), Err(PoolError::Usage(_))), "no env to roll out is an error");
}
