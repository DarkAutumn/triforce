//! Threaded environment pool (plan step 14): many envs on persistent worker threads, stepped in batches by a
//! caller that runs central batched inference (the PyO3 module, `triforce._native.EnvPool`).
//!
//! The core ([`Pool`], [`PoolEnv`], [`layout`], [`control`]) knows nothing about Zelda:
//! - `num_envs` envs are pinned round-robin to `threads` OS threads (env `i` on thread `i % threads`), built on
//!   the caller's thread by a factory that gets each env's seeded generator ([`env_rng`]: ChaCha8 from the pool
//!   seed, stream `i`) and its [`EnvControl`].
//! - Every env resets at creation. [`Pool::recv`] blocks until enough envs are ready and copies their rows into
//!   caller-owned [`Batch`] buffers; [`Pool::send`] queues one action per returned env. A terminal step resets
//!   the env in the same row (`final_obs`/`final_valid` keep the terminal observation).
//! - Lockstep (`recv(min_ready = num_envs)`, then `send` to all) is reproducible: an env's rows depend only on
//!   its seed and actions, and rows come back in env id order, whatever the thread count.
//! - [`EnvControl`] carries the per-env render flag (read before every emulated frame), a double-buffered frame
//!   with a counter ([`Pool::frame`]), [`FramePacer`] pacing and the record flag. Events (episode infos, run
//!   results, room changes, recorded inputs) come back through [`Pool::poll_events`].
//! - An env error or panic fails that env only and is returned once by `recv`/`wait`; dropping the pool stops
//!   and joins the workers.
//!
//! With the `zelda` feature, [`zelda`] adapts `env::SpecialistEnv` (training, evaluation) and `env::ChainEnv`
//! (chain runs) to [`PoolEnv`] over any `zelda::Nes`.

pub mod control;
pub mod jobs;
pub mod layout;
mod pool;
pub mod rollout;
#[cfg(feature = "zelda")]
pub mod zelda;

pub use control::{EnvControl, FramePacer};
pub use jobs::Jobs;
pub use layout::{
    Batch, BufferSpec, Buffers, Column, ColumnMut, Dtype, KeySpec, ObsData, ObsLayout, ObsRow,
    SCALAR_COLUMNS,
};
pub use pool::{
    EnvInit, FRAME_SHAPE, Outcome, Pool, PoolEnv, PoolError, PoolOptions, Start, env_rng,
};
