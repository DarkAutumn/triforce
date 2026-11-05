//! Errors of the env layer (each maps to one Python exception class in `triforce._native`).

use crate::config::ConfigError;
use crate::end_conditions::UnclassifiedEndingError;
use std::fmt;
use zelda::ZeldaError;

#[derive(Debug)]
pub enum EnvError {
    /// Bad config or game data (`ConfigError(ValueError)`).
    Config(ConfigError),
    /// A start state outside the specialist's rooms, or no start states (`StartStateError(ValueError)`).
    StartState(String),
    /// `step` with an action the mask forbids (`InvalidActionError(ValueError)`).
    InvalidAction(String),
    /// An ending the specialist doesn't classify (`UnclassifiedEndingError(RuntimeError)`).
    Unclassified(UnclassifiedEndingError),
    /// `step` on a finished episode (`EpisodeFinishedError(RuntimeError)`).
    EpisodeFinished,
    /// Unreadable/unwritable state or trajectory file (`StateFormatError(ValueError)` / OSError).
    State(String),
    /// The emulator never gave Link control, or the session wasn't loaded.
    Zelda(ZeldaError),
}

impl fmt::Display for EnvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EnvError::Config(e) => write!(f, "{e}"),
            EnvError::StartState(m) | EnvError::InvalidAction(m) | EnvError::State(m) => {
                f.write_str(m)
            }
            EnvError::Unclassified(e) => write!(f, "{e}"),
            EnvError::EpisodeFinished => {
                f.write_str("step() called on a finished episode; call reset()")
            }
            EnvError::Zelda(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for EnvError {}

impl From<ConfigError> for EnvError {
    fn from(e: ConfigError) -> Self {
        EnvError::Config(e)
    }
}

impl From<UnclassifiedEndingError> for EnvError {
    fn from(e: UnclassifiedEndingError) -> Self {
        EnvError::Unclassified(e)
    }
}

impl From<ZeldaError> for EnvError {
    fn from(e: ZeldaError) -> Self {
        EnvError::Zelda(e)
    }
}

impl From<std::io::Error> for EnvError {
    fn from(e: std::io::Error) -> Self {
        EnvError::State(e.to_string())
    }
}
