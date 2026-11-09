//! Specialist environments over the `zelda` game model (plan step 13): configs and game data, rewards and
//! critic, end conditions, observations (schema v1), start states, the backward curriculum, the single-
//! specialist env, the chain runner and Go-Explore.
//!
//! Module map (Python source → module): `specialist.py` + game.yaml/routes.yaml/chain_hp.json → [`config`];
//! `rewards.py` → [`rewards`]; `critics.py` → [`critic`]; `end_conditions.py` → [`end_conditions`];
//! `observation.py` → [`observation`]; `start_states.py` → [`start_states`]; `curriculum.py` →
//! [`curriculum`]; `env.py` → [`specialist_env`]; `runner.py` (`Runner._run`) → [`chain`]; `explore.py` →
//! [`explore`].  Emulator states are opaque (`zelda::Nes::State`); a [`start_states::StateStore`] loads and
//! saves them.

pub mod chain;
pub mod config;
pub mod critic;
pub mod curriculum;
pub mod end_conditions;
pub mod error;
pub mod explore;
pub mod objective;
pub mod observation;
pub mod rewards;
pub mod specialist_env;
pub mod start_states;

pub use chain::{ChainEnv, ChainStatus, RoomRecord, Route, RunResult};
pub use config::{ConfigError, GameData, SpecialistConfig};
pub use curriculum::{Curriculum, CurriculumSpec, SharedCurriculum};
pub use error::EnvError;
pub use explore::{AcceptSpec, ExploreResult, Explorer};
pub use observation::{OBS_SPEC, Observation, ObservationBuilder};
pub use specialist_env::{SpecialistEnv, StepInfo, StepResult, TerminalInfo};
pub use start_states::StateStore;
