//! Python exception classes and the Rust error mappings.

use env::EnvError;
use pool::PoolError;
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

create_exception!(triforce._native, InvalidActionError, PyValueError);
create_exception!(triforce._native, StartStateError, PyValueError);
create_exception!(triforce._native, UnclassifiedEndingError, PyRuntimeError);
create_exception!(triforce._native, ConfigError, PyValueError);
create_exception!(triforce._native, StateFormatError, PyValueError);
create_exception!(triforce._native, RomError, PyException);
create_exception!(triforce._native, RenderDisabledError, PyRuntimeError);
create_exception!(triforce._native, EpisodeFinishedError, PyRuntimeError);
create_exception!(triforce._native, ReplayError, PyValueError);

/// The Python exception of an env error.
pub fn env_err(e: EnvError) -> PyErr {
    let message = e.to_string();
    match e {
        EnvError::Config(_) => ConfigError::new_err(message),
        EnvError::StartState(_) => StartStateError::new_err(message),
        EnvError::InvalidAction(_) => InvalidActionError::new_err(message),
        EnvError::Unclassified(_) => UnclassifiedEndingError::new_err(message),
        EnvError::EpisodeFinished => EpisodeFinishedError::new_err(message),
        EnvError::State(_) => StateFormatError::new_err(message),
        EnvError::Zelda(_) => PyRuntimeError::new_err(message),
    }
}

/// The Python exception of a pool error: an env's own error keeps its class (prefixed with the env id).
pub fn pool_err(e: PoolError<EnvError>) -> PyErr {
    match e {
        PoolError::Env { env, error } => {
            let inner = env_err(error);
            Python::attach(|py| {
                let message = format!("env {env}: {}", inner.value(py));
                PyErr::from_type(inner.get_type(py), message)
            })
        }
        PoolError::Usage(m) => PyValueError::new_err(m),
        other => PyRuntimeError::new_err(other.to_string()),
    }
}
