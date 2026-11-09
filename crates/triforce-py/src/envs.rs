//! `Env`: one specialist env (`ZeldaEnv`'s gym-like reset/step/restore) for demos, behavior cloning and tests;
//! `CurriculumSpec`: backward-curriculum settings for `Env` and `TrainTask`.

use crate::config::{PySpecialistConfig, extract_config};
use crate::context::{self, TfzStore};
use crate::convert::{extract_action, json_to_py, obs_to_dict};
use crate::errors::env_err;
use crate::game::{PyGameState, PySnapshot, mask_array, screen};
use env::SharedCurriculum;
use env::curriculum::{Curriculum, CurriculumSpec};
use env::specialist_env::{EnvOptions, SpecialistEnv, StepInfo};
use numpy::PyArray1;
use parking_lot::Mutex;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use std::path::{Path, PathBuf};
use zcore::ZCore;

/// Backward-curriculum settings (`curriculum.CurriculumSpec`): `trajectories` is one specialist's explore
/// output; progress lives in `state_path` (a `TrainTask` defaults it to `<run dir>/curriculum.json`).
#[pyclass(
    module = "triforce._native",
    name = "CurriculumSpec",
    frozen,
    get_all,
    from_py_object
)]
#[derive(Clone)]
pub struct PyCurriculumSpec {
    pub trajectories: PathBuf,
    pub state_path: Option<PathBuf>,
    pub fraction: f64,
    pub stages: usize,
    pub window: u32,
    pub threshold: f64,
}

#[pymethods]
impl PyCurriculumSpec {
    #[new]
    #[pyo3(signature = (trajectories, state_path=None, fraction=0.5, stages=10, window=10, threshold=0.6))]
    fn new(
        trajectories: PathBuf,
        state_path: Option<PathBuf>,
        fraction: f64,
        stages: usize,
        window: u32,
        threshold: f64,
    ) -> Self {
        PyCurriculumSpec {
            trajectories,
            state_path,
            fraction,
            stages,
            window,
            threshold,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "CurriculumSpec(trajectories={:?}, state_path={:?}, fraction={}, stages={}, window={}, threshold={})",
            self.trajectories,
            self.state_path,
            self.fraction,
            self.stages,
            self.window,
            self.threshold
        )
    }
}

impl PyCurriculumSpec {
    /// The shared curriculum, with progress in `state_path` or `default_state`.
    pub fn open(&self, default_state: Option<&Path>) -> PyResult<SharedCurriculum> {
        let state_path = self
            .state_path
            .clone()
            .or_else(|| default_state.map(Path::to_path_buf))
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("CurriculumSpec needs a state_path")
            })?;
        let spec = CurriculumSpec {
            trajectories: self.trajectories.clone(),
            state_path,
            fraction: self.fraction,
            stages: self.stages,
            window: self.window,
            threshold: self.threshold,
        };
        Ok(Curriculum::new(spec).map_err(env_err)?.shared())
    }
}

pub fn info_dict<'py>(py: Python<'py>, info: &StepInfo) -> PyResult<Bound<'py, PyAny>> {
    json_to_py(py, &info.to_json())
}

struct EnvInner {
    env: SpecialistEnv<ZCore, TfzStore>,
    render: bool,
}

/// One specialist env with gym-like `reset`/`step` (observations as `{key: ndarray}`).
#[pyclass(module = "triforce._native", name = "Env", frozen)]
pub struct PyEnv {
    inner: Mutex<EnvInner>,
    config: Py<PySpecialistConfig>,
}

#[pymethods]
impl PyEnv {
    /// `config`: a `SpecialistConfig` or name. `start_state` pins every reset; `explicit_starts` builds no
    /// start-state pool (every reset names its state). `capture_dir` saves first-entry states;
    /// `strict_actions=False` executes masked actions instead of raising.
    #[new]
    #[pyo3(signature = (config, start_state=None, capture_dir=None, strict_actions=true, curriculum=None,
                        seed=0, states_dir=None, render=false, explicit_starts=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        config: &Bound<'_, PyAny>,
        start_state: Option<String>,
        capture_dir: Option<PathBuf>,
        strict_actions: bool,
        curriculum: Option<PyCurriculumSpec>,
        seed: u64,
        states_dir: Option<PathBuf>,
        render: bool,
        explicit_starts: bool,
    ) -> PyResult<Self> {
        let config = extract_config(py, config)?;
        let rom = context::rom(py)?;
        let data = context::data(py, states_dir.as_deref())?;
        let curriculum = curriculum.map(|c| c.open(None)).transpose()?;
        let opts = EnvOptions {
            start_state,
            explicit_starts,
            capture_dir,
            lenient_actions: !strict_actions,
            curriculum,
            seed,
            track_visits: false,
            skip_look_ahead: false,
            shaping: None,
        };
        let store = TfzStore::new(&data, rom.clone());
        let mut env = SpecialistEnv::new(config.clone(), data, ZCore::power_on(rom), store, opts)
            .map_err(env_err)?;
        env.session_mut().set_render(render);
        Ok(PyEnv {
            inner: Mutex::new(EnvInner { env, render }),
            config: Py::new(py, PySpecialistConfig { inner: config })?,
        })
    }

    /// Starts an episode: `seed` reseeds the env RNG; `options={'start_state': ...}` overrides the pool.
    #[pyo3(signature = (*, seed=None, options=None))]
    fn reset<'py>(
        &self,
        py: Python<'py>,
        seed: Option<u64>,
        options: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let start: Option<String> = match options {
            Some(o) => o
                .get_item("start_state")?
                .map(|s| s.extract())
                .transpose()?,
            None => None,
        };
        let (obs, info) = py
            .detach(|| self.inner.lock().env.reset(seed, start.as_deref()))
            .map_err(env_err)?;
        (obs_to_dict(py, &obs)?, info_dict(py, &info)?).into_pyobject(py)
    }

    /// Executes an action (`'MOVE N 16'`, head indices or an `Action`); returns
    /// (obs, reward, terminated, truncated, info).
    fn step<'py>(
        &self,
        py: Python<'py>,
        action: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let action = extract_action(action)?;
        let r = py
            .detach(|| self.inner.lock().env.step(action))
            .map_err(env_err)?;
        (
            obs_to_dict(py, &r.obs)?,
            r.reward,
            r.terminated,
            r.truncated,
            info_dict(py, &r.info)?,
        )
            .into_pyobject(py)
    }

    /// Continues from a snapshot as a new episode labelled `name` (no RNG or RAM changes).
    fn restore<'py>(
        &self,
        py: Python<'py>,
        snapshot: &Bound<'py, PySnapshot>,
        name: &str,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let snap = snapshot.get().snap.clone();
        let (obs, info) = self
            .inner
            .lock()
            .env
            .restore(&snap, name)
            .map_err(env_err)?;
        (obs_to_dict(py, &obs)?, info_dict(py, &info)?).into_pyobject(py)
    }

    fn snapshot(&self) -> PySnapshot {
        PySnapshot::new(self.inner.lock().env.session().snapshot())
    }

    #[getter]
    fn config(&self, py: Python<'_>) -> Py<PySpecialistConfig> {
        self.config.clone_ref(py)
    }

    #[getter]
    fn state(&self) -> PyGameState {
        PyGameState::new(self.inner.lock().env.session().state().clone())
    }

    #[getter]
    fn mask<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        mask_array(py, self.inner.lock().env.session().mask())
    }

    /// The episode ended (call reset).
    #[getter]
    fn done(&self) -> bool {
        self.inner.lock().env.is_done()
    }

    /// (route, step) the episode's objectives follow.
    #[getter]
    fn route(&self) -> (String, Option<usize>) {
        let g = self.inner.lock();
        let (route, step) = g.env.route();
        (route.to_owned(), step)
    }

    /// Sets the route for the next `restore` (a reset draws its own).
    #[pyo3(signature = (route, step=None))]
    fn set_route(&self, route: &str, step: Option<usize>) {
        self.inner.lock().env.set_route(route, step);
    }

    #[getter]
    fn get_render(&self) -> bool {
        self.inner.lock().render
    }

    #[setter]
    fn set_render(&self, on: bool) {
        let mut g = self.inner.lock();
        g.render = on;
        g.env.session_mut().set_render(on);
    }

    fn screen<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let g = self.inner.lock();
        screen(py, g.env.session().nes(), g.render)
    }

    /// No-op (the env owns nothing that needs closing); kept for gym compatibility.
    fn close(&self) {}
}
