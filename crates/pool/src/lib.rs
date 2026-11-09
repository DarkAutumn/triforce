//! Threaded environment pool: many envs on persistent worker threads that act with an in-pool policy (the
//! PyO3 module's `triforce._native.EnvPool`; docs/rust-actor.md).
//!
//! The core ([`Pool`], [`PoolEnv`], [`layout`], [`control`], [`rollout`]) knows nothing about Zelda:
//! - `num_envs` envs are pinned round-robin to `threads` OS threads (env `i` on thread `i % threads`), built on
//!   the caller's thread by a factory that gets each env's seeded generator ([`env_rng`]: ChaCha8 from the pool
//!   seed, stream `i`) and its [`EnvControl`].
//! - Every env resets at creation. An [`rollout::Actor`] picks the actions on the workers:
//!   [`Pool::rollout`] steps every env a fixed number of decisions and returns the rollouts (training);
//!   [`Pool::play`] lets every env act on its own until its jobs run out, optionally recording each episode
//!   (evaluation, chains, the watch player). A terminal step resets the env at once.
//! - Rollouts and plays are reproducible: an env's episodes depend only on its seed, its jobs and the actor (keyed
//!   per episode and decision), whatever the thread count.
//! - [`EnvControl`] carries the per-env render flag (read before every emulated frame), a double-buffered frame
//!   with a counter ([`Pool::frame`]), [`FramePacer`] pacing, the hold and pause flags and the record flag.
//!   Events (episode infos, run results, room changes, recorded inputs) come back through [`Pool::poll_events`].
//! - An env error or panic fails that env only and is returned once by `rollout`/`wait_idle`; dropping the pool
//!   stops and joins the workers.
//!
//! With the `zelda` feature, [`zelda`] adapts `env::SpecialistEnv` (training, evaluation) and `env::ChainEnv`
//! (chain runs) to [`PoolEnv`] over any `zelda::Nes`, and [`zelda::PolicyActor`] acts with `policy::Policy`.

pub mod control;
pub mod jobs;
pub mod layout;
mod pool;
pub mod rollout;
#[cfg(feature = "zelda")]
pub mod zelda;

pub use control::{EnvControl, FramePacer};
pub use jobs::Jobs;
pub use layout::{Column, Dtype, KeySpec, ObsColumns, ObsData, ObsLayout, ObsRow};
pub use pool::{
    EnvInit, FRAME_SHAPE, Outcome, Pool, PoolEnv, PoolError, PoolOptions, Start, env_rng,
};
