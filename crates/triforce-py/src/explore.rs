//! Go-Explore (`triforce/explore.py`): `Explorer` over one specialist's env, `AcceptSpec` for route-state builds,
//! and `explore_states`, which explores many start states on threads and writes their trajectories.

use crate::config::extract_config;
use crate::context::{self, TfzStore};
use crate::envs::info_dict;
use crate::errors::env_err;
use crate::game::{PyGameState, PySnapshot};
use env::config::{GameData, SpecialistConfig};
use env::explore::{AcceptSpec, ExploreResult, Explorer};
use env::start_states::state_label;
use parking_lot::Mutex;
use pyo3::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use zcore::{Rom, ZCore};

/// Success for route-state builds: Link entered `target` (level, room, in_cave) holding at least `min_keys`,
/// and (when set) the item of `item_taken_room` (level, room) is taken.
#[pyclass(
    module = "triforce._native",
    name = "AcceptSpec",
    frozen,
    get_all,
    from_py_object
)]
#[derive(Clone)]
pub struct PyAcceptSpec {
    pub target: (u8, u8, bool),
    pub min_keys: u8,
    pub item_taken_room: Option<(u8, u8)>,
}

#[pymethods]
impl PyAcceptSpec {
    #[new]
    #[pyo3(signature = (target, min_keys=0, item_taken_room=None))]
    fn new(target: (u8, u8, bool), min_keys: u8, item_taken_room: Option<(u8, u8)>) -> Self {
        PyAcceptSpec {
            target,
            min_keys,
            item_taken_room,
        }
    }

    /// Whether `state` passes.
    fn accepts(&self, state: &Bound<'_, PyGameState>) -> bool {
        self.spec().accepts(&state.get().view)
    }

    fn __repr__(&self) -> String {
        format!(
            "AcceptSpec(target={:?}, min_keys={}, item_taken_room={:?})",
            self.target, self.min_keys, self.item_taken_room
        )
    }
}

impl PyAcceptSpec {
    fn spec(&self) -> AcceptSpec {
        AcceptSpec {
            target: self.target,
            min_keys: self.min_keys,
            item_taken_room: self.item_taken_room,
        }
    }
}

/// The outcome of exploring one start state.
#[pyclass(module = "triforce._native", name = "ExploreResult", frozen)]
pub struct PyExploreResult {
    pub result: ExploreResult,
}

#[pymethods]
impl PyExploreResult {
    #[getter]
    fn start_state(&self) -> &str {
        &self.result.start_state
    }
    #[getter]
    fn success(&self) -> bool {
        self.result.success
    }
    /// `[kind, direction, duration]` head indices of the best path.
    #[getter]
    fn actions(&self) -> Vec<(usize, usize, usize)> {
        self.result
            .actions
            .iter()
            .map(|a| (a[0], a[1], a[2]))
            .collect()
    }
    #[getter]
    fn health(&self) -> f64 {
        self.result.health
    }
    #[getter]
    fn ending(&self) -> Option<&str> {
        self.result.ending.as_deref()
    }
    #[getter]
    fn iterations(&self) -> u32 {
        self.result.iterations
    }
    #[getter]
    fn env_steps(&self) -> u64 {
        self.result.env_steps
    }
    #[getter]
    fn cells(&self) -> usize {
        self.result.cells
    }
    #[getter]
    fn seconds(&self) -> f64 {
        self.result.seconds
    }

    fn __repr__(&self) -> String {
        format!(
            "ExploreResult(start_state={:?}, success={}, steps={}, health={}, ending={:?}, iterations={}, \
             env_steps={}, cells={})",
            self.result.start_state,
            self.result.success,
            self.result.actions.len(),
            self.result.health,
            self.result.ending,
            self.result.iterations,
            self.result.env_steps,
            self.result.cells
        )
    }
}

type ZExplorer = Explorer<ZCore, TfzStore>;

fn new_explorer(
    config: &SpecialistConfig,
    data: Arc<GameData>,
    rom: Arc<Rom>,
    seed: u64,
) -> Result<ZExplorer, env::EnvError> {
    let store = TfzStore::new(&data, rom.clone());
    Explorer::new(config, data, ZCore::power_on(rom), store, seed)
}

/// Go-Explore over one specialist's env (no RAM randomization).
#[pyclass(module = "triforce._native", name = "Explorer", frozen)]
pub struct PyExplorer {
    inner: Mutex<ZExplorer>,
}

#[pymethods]
impl PyExplorer {
    #[new]
    #[pyo3(signature = (config, seed=0, states_dir=None))]
    fn new(
        py: Python<'_>,
        config: &Bound<'_, PyAny>,
        seed: u64,
        states_dir: Option<PathBuf>,
    ) -> PyResult<Self> {
        let config = extract_config(py, config)?;
        let data = context::data(py, states_dir.as_deref())?;
        let rom = context::rom(py)?;
        Ok(PyExplorer {
            inner: Mutex::new(new_explorer(&config, data, rom, seed).map_err(env_err)?),
        })
    }

    /// Searches for a successful path from `start_state` (GIL released); returns the best one found.
    #[pyo3(signature = (start_state, iterations=3000, explore_steps=25, patience=300, accept=None))]
    fn explore(
        &self,
        py: Python<'_>,
        start_state: &str,
        iterations: u32,
        explore_steps: u32,
        patience: u32,
        accept: Option<PyAcceptSpec>,
    ) -> PyResult<PyExploreResult> {
        let accept = accept.map(|a| a.spec());
        let result = py
            .detach(|| {
                self.inner.lock().explore(
                    start_state,
                    iterations,
                    explore_steps,
                    patience,
                    accept.as_ref(),
                )
            })
            .map_err(env_err)?;
        Ok(PyExploreResult { result })
    }

    /// Replays a successful result, writing the state before every step plus `trajectory.json`; returns whether
    /// the replay ends in the same success.
    fn write(
        &self,
        py: Python<'_>,
        result: &Bound<'_, PyExploreResult>,
        directory: PathBuf,
    ) -> PyResult<bool> {
        let result = result.get().result.clone();
        py.detach(|| self.inner.lock().write(&result, &directory))
            .map_err(env_err)
    }

    /// Replays a result from the start of the last `explore`; returns (final step info or None, the pre-entry
    /// snapshot of its last room entry when `capture_pre_entry`).
    #[pyo3(signature = (result, capture_pre_entry=false))]
    fn replay<'py>(
        &self,
        py: Python<'py>,
        result: &Bound<'py, PyExploreResult>,
        capture_pre_entry: bool,
    ) -> PyResult<(Option<Bound<'py, PyAny>>, Option<PySnapshot>)> {
        let result = result.get().result.clone();
        let (info, pre_entry) = py
            .detach(|| self.inner.lock().replay(&result, capture_pre_entry))
            .map_err(env_err)?;
        Ok((
            info.map(|i| info_dict(py, &i)).transpose()?,
            pre_entry.map(PySnapshot::new),
        ))
    }

    /// The explorer's current game state.
    #[getter]
    fn state(&self) -> PyGameState {
        PyGameState::new(self.inner.lock().env.session().state().clone())
    }

    fn snapshot(&self) -> PySnapshot {
        PySnapshot::new(self.inner.lock().env.session().snapshot())
    }

    #[getter]
    fn seed(&self) -> u64 {
        self.inner.lock().seed
    }
}

type Explored = Result<(ExploreResult, bool), env::EnvError>;

/// Explores one start state with a fresh explorer and writes a success to `<out>/<state>`.
fn explore_one(
    config: &SpecialistConfig,
    data: &Arc<GameData>,
    rom: &Arc<Rom>,
    state: &str,
    out: &Path,
    budget: (u32, u32, u32),
    seed: u64,
) -> Explored {
    let mut explorer = new_explorer(config, data.clone(), rom.clone(), seed)?;
    let (iterations, explore_steps, patience) = budget;
    let result = explorer.explore(state, iterations, explore_steps, patience, None)?;
    let replayed = if result.success {
        explorer.write(&result, &out.join(state))?
    } else {
        false
    };
    Ok((result, replayed))
}

/// Explores every seed start state of `specialist` (or `states`), writing each success to
/// `<out>/<specialist>/<state>` (`out` is the trajectories root). Each state gets a fresh explorer seeded with
/// `seed`, so results don't depend on `parallel`. Returns [(ExploreResult, replayed)] in state order.
#[pyfunction]
#[pyo3(signature = (specialist, out, states=None, iterations=3000, explore_steps=25, patience=300, parallel=1,
                    seed=0, states_dir=None))]
#[allow(clippy::too_many_arguments)]
pub fn explore_states(
    py: Python<'_>,
    specialist: &Bound<'_, PyAny>,
    out: PathBuf,
    states: Option<Vec<String>>,
    iterations: u32,
    explore_steps: u32,
    patience: u32,
    parallel: usize,
    seed: u64,
    states_dir: Option<PathBuf>,
) -> PyResult<Vec<(PyExploreResult, bool)>> {
    let config = extract_config(py, specialist)?;
    let data = context::data(py, states_dir.as_deref())?;
    let rom = context::rom(py)?;
    let states = match states {
        Some(s) => s,
        None => {
            let mut s: Vec<String> = config
                .resolve_start_states(&data.states_dir)
                .iter()
                .map(|p| state_label(&p.to_string_lossy()).to_owned())
                .collect();
            s.sort();
            s
        }
    };
    let out = out.join(&config.name);
    let budget = (iterations, explore_steps, patience);
    let results: Vec<Mutex<Option<Explored>>> = states.iter().map(|_| Mutex::new(None)).collect();
    let next = AtomicUsize::new(0);
    let workers = parallel.clamp(1, states.len().max(1));
    py.detach(|| {
        std::thread::scope(|scope| {
            for _ in 0..workers {
                scope.spawn(|| {
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(state) = states.get(i) else { break };
                        let r = explore_one(&config, &data, &rom, state, &out, budget, seed);
                        *results[i].lock() = Some(r);
                    }
                });
            }
        });
    });
    results
        .into_iter()
        .map(|r| {
            let (result, replayed) = r
                .into_inner()
                .expect("every state explored")
                .map_err(env_err)?;
            Ok((PyExploreResult { result }, replayed))
        })
        .collect()
}
