//! The pool's error and shutdown paths under concurrent callers: a rollout reports its own envs' failures whoever
//! else waits on the pool (F077), `close` stops a rollout before its next step (F082), a timed `wait_idle` never
//! reports idle with a failure unreported (F128), and `set_hold` on a closed pool is an error (F072).

use pool::rollout::{Actor, Decision, Record};
use pool::{
    Dtype, EnvControl, EnvInit, KeySpec, ObsData, ObsLayout, ObsRow, Outcome, Pool, PoolEnv, PoolError,
    PoolOptions, Start,
};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Clone, Default)]
struct Config {
    step_delay: Duration,
    /// (env, step) pairs that fail.
    errors: Vec<(usize, u64)>,
    /// Hold the frame after every 3-step episode (the watch player's result pause).
    hold_after_episode: bool,
}

struct Slow {
    index: usize,
    control: EnvControl,
    cfg: Config,
    steps: u64,
    episode: i64,
    t: u32,
}

impl PoolEnv for Slow {
    type Action = i64;
    type Event = ();
    type Error = String;

    fn layout(&self) -> ObsLayout {
        ObsLayout::new(vec![KeySpec::new("id", &[1], Dtype::I64)])
    }

    fn reset(&mut self, mut obs: ObsRow<'_>, _: &mut Vec<()>) -> Result<Option<Start>, String> {
        self.episode += 1;
        self.t = 0;
        obs.i64(0)[0] = self.index as i64;
        Ok(Some(Start {
            policy: 0,
            job: self.episode,
        }))
    }

    fn step(&mut self, _: i64, mut obs: ObsRow<'_>, _: &mut Vec<()>) -> Result<Outcome, String> {
        self.steps += 1;
        if self.cfg.errors.contains(&(self.index, self.steps)) {
            return Err(format!("bad step {}", self.steps));
        }
        if !self.cfg.step_delay.is_zero() {
            std::thread::sleep(self.cfg.step_delay);
        }
        self.t += 1;
        let done = self.t >= 3;
        if done && self.cfg.hold_after_episode {
            self.control.set_hold(true);
        }
        obs.i64(0)[0] = self.index as i64;
        Ok(Outcome {
            truncated: done,
            frames: 1,
            ..Outcome::default()
        })
    }
}

struct Zero;

impl Actor<i64> for Zero {
    type Scratch = ();
    fn scratch(&self) {}
    fn act(&self, _: &ObsData, _: i64, _: i64, _: i64, _: &mut ()) -> Result<(i64, Decision), String> {
        Ok((0, Decision::default()))
    }
    fn value(&self, _: &ObsData, _: i64, _: &mut ()) -> Result<f32, String> {
        Ok(0.0)
    }
}

fn pool(num_envs: usize, threads: usize, cfg: Config) -> Pool<Slow> {
    Pool::new(PoolOptions::new(num_envs, threads, 0), |init: EnvInit| {
        Ok(Slow {
            index: init.index,
            control: init.control,
            cfg: cfg.clone(),
            steps: 0,
            episode: 0,
            t: 0,
        })
    })
    .unwrap()
}

#[test]
fn a_rollout_reports_its_failed_env_even_when_another_caller_took_the_error() {
    for _ in 0..20 {
        let cfg = Config {
            step_delay: Duration::from_millis(1),
            errors: vec![(1, 3)],
            ..Config::default()
        };
        let pool = pool(2, 2, cfg);
        let rolled = std::thread::scope(|scope| {
            let rolling = scope.spawn(|| pool.rollout(50, Arc::new(Zero)));
            let _ = pool.wait_idle(None); // may take env 1's error first
            rolling.join()
        });
        let result = rolled.expect("the rollout must not panic");
        assert!(result.is_err(), "env 1 failed during the rollout");
        assert_eq!(pool.active(), 1);
    }
}

#[test]
fn every_failure_of_one_rollout_is_reported_by_that_rollout() {
    let cfg = Config {
        errors: vec![(0, 2), (1, 2)],
        ..Config::default()
    };
    let pool = pool(3, 3, cfg);
    assert!(matches!(pool.rollout(5, Arc::new(Zero)), Err(PoolError::Env { env: 0 | 1, .. })));
    // env 1's error is not left for the next call: it rolls out the healthy env.
    let next = pool.rollout(5, Arc::new(Zero)).unwrap();
    assert_eq!(next.iter().map(|(env, _)| *env).collect::<Vec<_>>(), [2]);
    assert!(pool.wait_idle(None).unwrap());
}

#[test]
fn close_stops_a_rollout_before_its_next_step() {
    let cfg = Config {
        step_delay: Duration::from_millis(1),
        ..Config::default()
    };
    let pool = pool(2, 1, cfg);
    std::thread::scope(|scope| {
        let rolling = scope.spawn(|| pool.rollout(2000, Arc::new(Zero)));
        std::thread::sleep(Duration::from_millis(100));
        let t0 = Instant::now();
        pool.close();
        let took = t0.elapsed();
        assert!(took < Duration::from_millis(500), "close took {took:?}");
        assert!(matches!(rolling.join().unwrap(), Err(PoolError::Closed)));
    });
}

#[test]
fn a_timed_wait_idle_never_reports_idle_over_an_unreported_failure() {
    for trial in 0..300 {
        let cfg = Config {
            errors: vec![(0, 1)],
            ..Config::default()
        };
        let pool = pool(1, 1, cfg);
        pool.play(Arc::new(Zero), Record::Off).unwrap();
        let result = loop {
            match pool.wait_idle(Some(Duration::ZERO)) {
                Ok(false) => continue,
                other => break other,
            }
        };
        assert!(matches!(result, Err(PoolError::Env { env: 0, .. })), "trial {trial}: {result:?}");
    }
}

#[test]
fn set_hold_on_a_closed_pool_is_an_error() {
    let cfg = Config {
        hold_after_episode: true,
        ..Config::default()
    };
    let pool = pool(2, 2, cfg);
    pool.play(Arc::new(Zero), Record::Off).unwrap();
    assert!(pool.wait_idle(Some(Duration::from_secs(10))).unwrap(), "both envs park after an episode");
    pool.close();
    assert!(matches!(pool.set_hold(1, false), Err(PoolError::Closed)));
    assert!(matches!(pool.set_hold(0, true), Err(PoolError::Closed)));
}
