//! `EnvPool` over `pool::Pool` with the Zelda adapters: `TrainTask`, `EvalTask` and `ChainTask` envs on
//! worker threads, acting with the Rust policy inside the pool (`rollout`, `play`; docs/rust-actor.md).

use crate::config::{PySpecialistConfig, extract_config, room_value};
use crate::context::{self, TfzStore};
use crate::convert::{LAYOUT, json_to_py, room_of};
use crate::envs::PyCurriculumSpec;
use crate::errors::{ConfigError, pool_err};
use crate::game::{FRAME_BYTES, FRAME_SHAPE, rgb};
use env::SharedCurriculum;
use env::chain::Route;
use env::config::{GameData, SpecialistConfig, parse_room};
use env::specialist_env::{EnvOptions, ShapingDial, cheated_json};
use env::start_states::{StartWeights, state_label};
use numpy::{
    PyArray, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyReadwriteArrayDyn,
    PyUntypedArrayMethods,
};
use crate::policy::PyPolicy;
use pool::rollout::{EnvRollout, Play, Record};
use pool::zelda::{
    ChainRunEnv, EvalEnv, EvalJob, GumbelSearch, PolicyActor, RgbFn, SearchConfig, TaskEnv, TrainEnv, ZeldaAction,
    ZeldaEvent,
};
use pool::{Column, Dtype, EnvInit, Jobs, Pool, PoolOptions};
use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use zcore::ZCore;

type Task = TaskEnv<ZCore, TfzStore>;

const RGB: Option<RgbFn<ZCore>> = Some(rgb);

/// One specialist's training episodes: resets draw from its start-state pool (or `start_state`), or from the
/// curriculum. `room_events` reports room changes.
#[pyclass(module = "triforce._native", name = "TrainTask", frozen)]
pub struct PyTrainTask {
    config: Arc<SpecialistConfig>,
    run_dir: PathBuf,
    start_state: Option<String>,
    curriculum: Option<PyCurriculumSpec>,
    states_dir: Option<PathBuf>,
    room_events: bool,
}

#[pymethods]
impl PyTrainTask {
    #[new]
    #[pyo3(signature = (config, run_dir, start_state=None, curriculum=None, states_dir=None, room_events=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        config: &Bound<'_, PyAny>,
        run_dir: PathBuf,
        start_state: Option<String>,
        curriculum: Option<PyCurriculumSpec>,
        states_dir: Option<PathBuf>,
        room_events: bool,
    ) -> PyResult<Self> {
        Ok(PyTrainTask {
            config: extract_config(py, config)?,
            run_dir,
            start_state,
            curriculum,
            states_dir,
            room_events,
        })
    }

    #[getter]
    fn config(&self) -> PySpecialistConfig {
        PySpecialistConfig {
            inner: self.config.clone(),
        }
    }

    #[getter]
    fn run_dir(&self) -> PathBuf {
        self.run_dir.clone()
    }
}

/// Evaluation episodes: one per `(start_state, seed)` job, shared by the pool's envs.
#[pyclass(module = "triforce._native", name = "EvalTask", frozen)]
pub struct PyEvalTask {
    config: Arc<SpecialistConfig>,
    jobs: Vec<EvalJob>,
    states_dir: Option<PathBuf>,
    room_events: bool,
}

#[pymethods]
impl PyEvalTask {
    #[new]
    #[pyo3(signature = (config, jobs, states_dir=None, room_events=false))]
    fn new(
        py: Python<'_>,
        config: &Bound<'_, PyAny>,
        jobs: Vec<(String, u64)>,
        states_dir: Option<PathBuf>,
        room_events: bool,
    ) -> PyResult<Self> {
        Ok(PyEvalTask {
            config: extract_config(py, config)?,
            jobs: jobs
                .into_iter()
                .map(|(start_state, seed)| EvalJob { start_state, seed })
                .collect(),
            states_dir,
            room_events,
        })
    }

    #[getter]
    fn jobs(&self) -> Vec<(String, u64)> {
        self.jobs
            .iter()
            .map(|j| (j.start_state.clone(), j.seed))
            .collect()
    }
}

/// Chain runs of `route`, one per seed. `configs` are the specialists that have policies (names or configs; a
/// config replaces the canonical YAML), in policy-index order: each row's `policy` is the acting specialist's
/// index here. A specialist in `missing` (name -> reason, e.g. no promoted model) ends a run that needs it as
/// `error`; any other unlisted specialist does too. `stall_steps` (the watch player's stall budget, off by
/// default): room events carry `stalled` once a room has gone that many actions without damage dealt or taken,
/// and the run ends `stuck` at twice as many; room events also carry `cheated` (`{bombs, keys, rupees}`), the run's
/// cheated items used so far (`zelda::cheats`). `hold_finished_frame`: a finished run keeps its env's last frame
/// published (the next run doesn't publish) until `EnvPool.set_hold(env, False)`. `harvest` (a directory, off by
/// default): every correct room entry where the acting specialist changes saves its pre-entry state as
/// `harvest/<specialist>/<level>_<room><side>_h<seed>_<n>.state` (`ChainEnv::set_harvest`).
#[pyclass(module = "triforce._native", name = "ChainTask", frozen)]
pub struct PyChainTask {
    configs: Vec<Arc<SpecialistConfig>>,
    route: String,
    seeds: Vec<u64>,
    stop_at: Option<zelda::MapLocation>,
    start_from: Option<String>,
    max_steps: u64,
    stall_steps: Option<u64>,
    hold_finished_frame: bool,
    missing: HashMap<String, String>,
    states_dir: Option<PathBuf>,
    harvest: Option<PathBuf>,
}

#[pymethods]
impl PyChainTask {
    #[new]
    #[pyo3(signature = (configs, route, seeds, stop_at=None, start_from=None, max_steps=50000, missing=None,
                        states_dir=None, stall_steps=None, hold_finished_frame=false, harvest=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        configs: Vec<Bound<'_, PyAny>>,
        route: String,
        seeds: Vec<u64>,
        stop_at: Option<&Bound<'_, PyAny>>,
        start_from: Option<String>,
        max_steps: u64,
        missing: Option<HashMap<String, String>>,
        states_dir: Option<PathBuf>,
        stall_steps: Option<u64>,
        hold_finished_frame: bool,
        harvest: Option<PathBuf>,
    ) -> PyResult<Self> {
        let configs = configs
            .iter()
            .map(|c| extract_config(py, c))
            .collect::<PyResult<Vec<_>>>()?;
        let stop_at = stop_at
            .map(|s| -> PyResult<_> {
                let serde_json::Value::String(text) = room_value(s)? else {
                    unreachable!()
                };
                parse_room(&text).map_err(|e| ConfigError::new_err(e.0))
            })
            .transpose()?;
        let data = context::data(py, states_dir.as_deref())?;
        Route::load(&data.map, &data.states_dir, &route, stop_at, start_from.as_deref())
            .map_err(ConfigError::new_err)?;
        Ok(PyChainTask {
            configs,
            route,
            seeds,
            stop_at,
            start_from,
            max_steps,
            stall_steps,
            hold_finished_frame,
            missing: missing.unwrap_or_default(),
            states_dir,
            harvest,
        })
    }

    /// The policy names, in policy-index order.
    #[getter]
    fn policies(&self) -> Vec<String> {
        self.configs.iter().map(|c| c.name.clone()).collect()
    }
}

/// What a pool was built from.
enum Kind {
    Train,
    Eval,
    Chain,
}

/// A threaded pool of Zelda envs whose agent acts inside the pool with the Rust policy.
///
/// `rollout(policies, steps, out)` steps every env `steps` decisions and writes them into Python-owned numpy
/// buffers (training); `play(policies)` lets every env act on its own until the task's jobs run out (evaluation,
/// chains, the watch player), with `wait_idle`, `poll_events` and `poll_plays` to follow it. Both release the
/// GIL. Terminal steps reset the env at once.
#[pyclass(module = "triforce._native", name = "EnvPool", frozen)]
pub struct PyEnvPool {
    pool: Pool<Task>,
    curriculum: Option<SharedCurriculum>,
    policies: Vec<String>,
    kind: Kind,
    /// The train pool's shared shaping scale (None for eval and chain pools).
    shaping: Option<ShapingDial>,
    /// The train pool's shared seed-state weights and its seed-state labels (None for eval and chain pools and a
    /// train pool pinned to one `start_state`).
    start_weights: Option<(StartWeights, Vec<String>)>,
}

fn build(py: Python<'_>, task: &Bound<'_, PyAny>, opts: PoolOptions) -> PyResult<PyEnvPool> {
    let rom = context::rom(py)?;
    if let Ok(t) = task.cast::<PyTrainTask>() {
        let t = t.get();
        let data = context::data(py, t.states_dir.as_deref())?;
        let curriculum = t
            .curriculum
            .as_ref()
            .map(|c| c.open(Some(&t.run_dir.join("curriculum.json"))))
            .transpose()?;
        let store = TfzStore::new(&data, rom.clone());
        let shaping = ShapingDial::new(t.config.reward.shaping_scale);
        let start_weights = t.start_state.is_none().then(|| {
            let states = t
                .config
                .resolve_start_states(&data.states_dir)
                .iter()
                .map(|p| state_label(&p.to_string_lossy()).to_owned())
                .collect();
            (StartWeights::default(), states)
        });
        let make = |init: EnvInit| {
            let opts = EnvOptions {
                start_state: t.start_state.clone(),
                curriculum: curriculum.clone(),
                shaping: Some(shaping.clone()),
                start_weights: start_weights.as_ref().map(|(w, _)| w.clone()),
                ..Default::default()
            };
            let mut env = TrainEnv::new(
                t.config.clone(),
                data.clone(),
                ZCore::power_on(rom.clone()),
                store.clone(),
                opts,
                init,
                RGB,
            )?;
            env.set_room_events(t.room_events);
            Ok(TaskEnv::Train(env))
        };
        let pool = py.detach(|| Pool::new(opts, make)).map_err(pool_err)?;
        return Ok(PyEnvPool {
            pool,
            curriculum,
            policies: vec![t.config.name.clone()],
            kind: Kind::Train,
            shaping: Some(shaping),
            start_weights,
        });
    }
    if let Ok(t) = task.cast::<PyEvalTask>() {
        let t = t.get();
        let data = context::data(py, t.states_dir.as_deref())?;
        let store = TfzStore::new(&data, rom.clone());
        let jobs = Jobs::new(t.jobs.iter().cloned());
        let make = |init: EnvInit| {
            let opts = EnvOptions {
                explicit_starts: true,
                ..Default::default()
            };
            let mut env = EvalEnv::new(
                t.config.clone(),
                data.clone(),
                ZCore::power_on(rom.clone()),
                store.clone(),
                opts,
                jobs.clone(),
                init,
                RGB,
            )?;
            env.set_room_events(t.room_events);
            Ok(TaskEnv::Eval(env))
        };
        let pool = py.detach(|| Pool::new(opts, make)).map_err(pool_err)?;
        return Ok(PyEnvPool {
            pool,
            curriculum: None,
            policies: vec![t.config.name.clone()],
            kind: Kind::Eval,
            shaping: None,
            start_weights: None,
        });
    }
    if let Ok(t) = task.cast::<PyChainTask>() {
        let t = t.get();
        let data: Arc<GameData> = context::data(py, t.states_dir.as_deref())?;
        let store = TfzStore::new(&data, rom.clone());
        let route = Route::load(&data.map, &data.states_dir, &t.route, t.stop_at, t.start_from.as_deref())
            .map_err(ConfigError::new_err)?;
        let seeds = Jobs::new(t.seeds.iter().copied());
        let policies: Vec<Arc<str>> = t
            .configs
            .iter()
            .map(|c| Arc::from(c.name.as_str()))
            .collect();
        let configs: HashMap<String, Arc<SpecialistConfig>> = t
            .configs
            .iter()
            .map(|c| (c.name.clone(), c.clone()))
            .collect();
        let make = |init: EnvInit| {
            let mut env = ChainRunEnv::new(
                data.clone(),
                ZCore::power_on(rom.clone()),
                store.clone(),
                route.clone(),
                seeds.clone(),
                policies.clone(),
                init,
                RGB,
            );
            let chain = env.chain_mut();
            chain.configs = configs.clone();
            chain.max_steps = t.max_steps;
            chain.stall_steps = t.stall_steps;
            chain.set_harvest(t.harvest.clone());
            env.missing_mut().extend(t.missing.clone());
            env.set_hold_finished(t.hold_finished_frame);
            Ok(TaskEnv::Chain(env))
        };
        let pool = py.detach(|| Pool::new(opts, make)).map_err(pool_err)?;
        return Ok(PyEnvPool {
            pool,
            curriculum: None,
            policies: t.configs.iter().map(|c| c.name.clone()).collect(),
            kind: Kind::Chain,
            shaping: None,
            start_weights: None,
        });
    }
    Err(PyTypeError::new_err(
        "task must be a TrainTask, EvalTask or ChainTask",
    ))
}

/// `obj` as a C-contiguous int64 array: a non-empty array of another kind than integers is refused (floats would be
/// truncated to other actions), and an array a running call holds is a ValueError, not a panic.
fn as_int64<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
    what: &str,
) -> PyResult<PyReadonlyArrayDyn<'py, i64>> {
    let np = py.import("numpy")?;
    let arr = np.getattr("asarray")?.call1((obj,))?;
    let kind: String = arr.getattr("dtype")?.getattr("kind")?.extract()?;
    let size: usize = arr.getattr("size")?.extract()?;
    if size > 0 && kind != "i" && kind != "u" {
        return Err(PyValueError::new_err(format!("{what} must be integers, got dtype kind '{kind}'")));
    }
    let arr = np.getattr("ascontiguousarray")?.call1((arr, "int64"))?;
    arr.cast_into::<PyArrayDyn<i64>>()?
        .try_readonly()
        .map_err(|e| PyValueError::new_err(format!("{what}: {e}")))
}

/// A numpy array of `out[key]` with exactly `shape` and dtype `T`, borrowed for writing.
fn rollout_array<'py, T: numpy::Element>(
    d: &Bound<'py, PyDict>,
    key: &str,
    shape: &[usize],
) -> PyResult<Option<PyReadwriteArrayDyn<'py, T>>> {
    let Some(item) = d.get_item(key)? else {
        return Ok(None);
    };
    let array = item.cast_into::<PyArrayDyn<T>>().map_err(|_| {
        PyTypeError::new_err(format!("rollout buffer {key:?} has the wrong dtype"))
    })?;
    if array.shape() != shape {
        return Err(PyValueError::new_err(format!(
            "rollout buffer {key:?} is {:?}, expected {shape:?}",
            array.shape()
        )));
    }
    let rw = array
        .try_readwrite()
        .map_err(|e| PyValueError::new_err(format!("rollout buffer {key:?}: {e}")))?;
    Ok(Some(rw))
}

/// `EnvPool.rollout`'s `search` dict as a [`SearchConfig`] (missing keys keep their defaults).
fn search_config(d: &Bound<'_, PyDict>) -> PyResult<SearchConfig> {
    let mut c = SearchConfig::default();
    for (key, value) in d.iter() {
        let key: String = key.extract()?;
        match key.as_str() {
            "candidates" => c.candidates = value.extract()?,
            "simulations" => c.simulations = value.extract()?,
            "futures" => c.futures = value.extract()?,
            "depth" => c.depth = value.extract()?,
            "gamma" => c.gamma = value.extract()?,
            "c_visit" => c.c_visit = value.extract()?,
            "c_scale" => c.c_scale = value.extract()?,
            "fraction" => c.fraction = value.extract()?,
            "perturb_rng" => c.perturb_rng = value.extract()?,
            other => return Err(PyValueError::new_err(format!("unknown search setting {other:?}"))),
        }
    }
    c.validate().map_err(PyValueError::new_err)?;
    Ok(c)
}

fn required<T>(v: Option<T>, key: &str) -> PyResult<T> {
    v.ok_or_else(|| PyKeyError::new_err(format!("rollout buffers need {key:?}")))
}

/// One observation key's rollout buffer.
enum ObsArray<'py> {
    I64(PyReadwriteArrayDyn<'py, i64>),
    F32(PyReadwriteArrayDyn<'py, f32>),
    Bool(PyReadwriteArrayDyn<'py, bool>),
}

/// The rollout buffers of `EnvPool.rollout`, checked and borrowed before the rollout runs.
struct RolloutOut<'py> {
    steps: usize,
    obs: Vec<ObsArray<'py>>,
    starts: PyReadwriteArrayDyn<'py, f32>,
    actions: PyReadwriteArrayDyn<'py, i64>,
    lev: PyReadwriteArrayDyn<'py, f32>,
    rewards: PyReadwriteArrayDyn<'py, f32>,
    frames: PyReadwriteArrayDyn<'py, f32>,
    boot: PyReadwriteArrayDyn<'py, f32>,
    interrupts: Option<PyReadwriteArrayDyn<'py, i64>>,
    jobs: Option<PyReadwriteArrayDyn<'py, i64>>,
    decision_steps: Option<PyReadwriteArrayDyn<'py, i64>>,
    terminated: Option<PyReadwriteArrayDyn<'py, bool>>,
    truncated: Option<PyReadwriteArrayDyn<'py, bool>>,
    search: SearchOut<'py>,
}

/// Joint actions of the search's improved policy: `kind * 32 + direction * 4 + duration` (7 kinds × 8 directions
/// × 4 durations).
const JOINT: usize = zelda::actions::NUM_KINDS * zelda::actions::NUM_DIRECTIONS * zelda::actions::NUM_DURATIONS;

/// The optional search buffers (`search_*`); `k` candidates per row.
struct SearchOut<'py> {
    k: usize,
    mask: Option<PyReadwriteArrayDyn<'py, bool>>,
    policy: Option<PyReadwriteArrayDyn<'py, f32>>,
    q: Option<PyReadwriteArrayDyn<'py, f32>>,
    actions: Option<PyReadwriteArrayDyn<'py, i64>>,
    value: Option<PyReadwriteArrayDyn<'py, f32>>,
}

impl<'py> SearchOut<'py> {
    /// `k`: the search's candidates; None (no search) takes it from the `search_q`/`search_actions` buffers.
    fn borrow(out: &Bound<'py, PyDict>, n: usize, steps: usize, k: Option<usize>) -> PyResult<SearchOut<'py>> {
        let k = match k {
            Some(k) => k,
            None => {
                let mut k = 0;
                for key in ["search_q", "search_actions"] {
                    if let Some(item) = out.get_item(key)? {
                        let shape: Vec<usize> = item.getattr("shape")?.extract()?;
                        k = shape.get(2).copied().unwrap_or(0);
                    }
                }
                k
            }
        };
        Ok(SearchOut {
            k,
            mask: rollout_array(out, "search_mask", &[n, steps])?,
            policy: rollout_array(out, "search_policy", &[n, steps, JOINT])?,
            q: rollout_array(out, "search_q", &[n, steps, k])?,
            actions: rollout_array(out, "search_actions", &[n, steps, k, 3])?,
            value: rollout_array(out, "search_value", &[n, steps])?,
        })
    }

    /// Env `env`'s rows: a searched row its `SearchRow`, every other row unsearched (mask false, policy 0, Q NaN,
    /// candidates -1, value 0).
    fn write(&mut self, env: usize, steps: usize, r: &EnvRollout<ZeldaAction>) -> PyResult<()> {
        let k = self.k;
        let row = |t: usize| r.search.get(t).and_then(Option::as_ref);
        if let Some(a) = self.mask.as_mut() {
            let dst = &mut a.as_slice_mut()?[env * steps..(env + 1) * steps];
            for (t, d) in dst.iter_mut().enumerate() {
                *d = row(t).is_some();
            }
        }
        if let Some(a) = self.policy.as_mut() {
            let dst = &mut a.as_slice_mut()?[env * steps * JOINT..(env + 1) * steps * JOINT];
            for (t, d) in dst.as_chunks_mut::<JOINT>().0.iter_mut().enumerate() {
                match row(t) {
                    Some(s) if s.policy.len() == JOINT => d.copy_from_slice(&s.policy),
                    Some(s) => {
                        return Err(PyValueError::new_err(format!(
                            "the policy has {} joint actions, search_policy holds {JOINT}",
                            s.policy.len()
                        )));
                    }
                    None => d.fill(0.0),
                }
            }
        }
        if k == 0 {
            return self.write_value(env, steps, r);
        }
        if let Some(a) = self.q.as_mut() {
            let dst = &mut a.as_slice_mut()?[env * steps * k..(env + 1) * steps * k];
            for (t, d) in dst.chunks_exact_mut(k).enumerate() {
                d.fill(f32::NAN);
                if let Some(s) = row(t) {
                    d[..s.q.len()].copy_from_slice(&s.q);
                }
            }
        }
        if let Some(a) = self.actions.as_mut() {
            let dst = &mut a.as_slice_mut()?[env * steps * k * 3..(env + 1) * steps * k * 3];
            for (t, d) in dst.chunks_exact_mut(k * 3).enumerate() {
                d.fill(-1);
                if let Some(s) = row(t) {
                    d[..s.candidates.len() * 3].copy_from_slice(s.candidates.as_flattened());
                }
            }
        }
        self.write_value(env, steps, r)
    }

    fn write_value(&mut self, env: usize, steps: usize, r: &EnvRollout<ZeldaAction>) -> PyResult<()> {
        if let Some(a) = self.value.as_mut() {
            let dst = &mut a.as_slice_mut()?[env * steps..(env + 1) * steps];
            for (t, d) in dst.iter_mut().enumerate() {
                *d = r.search.get(t).and_then(Option::as_ref).map_or(0.0, |s| s.value);
            }
        }
        Ok(())
    }
}

impl<'py> RolloutOut<'py> {
    fn borrow(out: &Bound<'py, PyDict>, n: usize, steps: usize, search_k: Option<usize>) -> PyResult<RolloutOut<'py>> {
        let obs = out
            .get_item("obs")?
            .ok_or_else(|| PyKeyError::new_err("rollout buffers need 'obs'"))?
            .cast_into::<PyDict>()?;
        let mut columns = Vec::with_capacity(LAYOUT.keys().len());
        for key in LAYOUT.keys() {
            let mut shape = vec![n, steps + 1];
            shape.extend(&key.shape);
            let name = &key.name;
            columns.push(match key.dtype {
                Dtype::I64 => ObsArray::I64(required(rollout_array(&obs, name, &shape)?, name)?),
                Dtype::F32 => ObsArray::F32(required(rollout_array(&obs, name, &shape)?, name)?),
                Dtype::Bool => ObsArray::Bool(required(rollout_array(&obs, name, &shape)?, name)?),
            });
        }
        Ok(RolloutOut {
            steps,
            obs: columns,
            starts: required(rollout_array(out, "starts", &[n, steps + 1])?, "starts")?,
            actions: required(rollout_array(out, "actions", &[n, steps, 3])?, "actions")?,
            lev: required(rollout_array(out, "logp_ent_val", &[n, steps, 3])?, "logp_ent_val")?,
            rewards: required(rollout_array(out, "rewards", &[n, steps])?, "rewards")?,
            frames: required(rollout_array(out, "frames", &[n, steps])?, "frames")?,
            boot: required(rollout_array(out, "bootstrap_value", &[n, steps])?, "bootstrap_value")?,
            interrupts: rollout_array(out, "interrupts", &[n, steps])?,
            jobs: rollout_array(out, "job", &[n, steps])?,
            decision_steps: rollout_array(out, "step", &[n, steps])?,
            terminated: rollout_array(out, "terminated", &[n, steps])?,
            truncated: rollout_array(out, "truncated", &[n, steps])?,
            search: SearchOut::borrow(out, n, steps, search_k)?,
        })
    }

    /// Copies each env's rollout into its env id row.
    fn write(&mut self, rollouts: &[(usize, EnvRollout<ZeldaAction>)]) -> PyResult<()> {
        fn rows<T: Copy>(dst: &mut [T], env: usize, width: usize, src: &[T]) {
            dst[env * width..env * width + src.len()].copy_from_slice(src);
        }
        let steps = self.steps;
        for (k, key) in LAYOUT.keys().iter().enumerate() {
            let width = (steps + 1) * key.len();
            for (env, r) in rollouts {
                match (&mut self.obs[k], &r.obs.columns[k]) {
                    (ObsArray::I64(a), Column::I64(c)) => rows(a.as_slice_mut()?, *env, width, c),
                    (ObsArray::F32(a), Column::F32(c)) => rows(a.as_slice_mut()?, *env, width, c),
                    (ObsArray::Bool(a), Column::Bool(c)) => rows(a.as_slice_mut()?, *env, width, c),
                    _ => unreachable!("the pool's layout is LAYOUT"),
                }
            }
        }
        for (env, r) in rollouts {
            let starts: Vec<f32> = r.starts.iter().map(|&s| s as u8 as f32).collect();
            rows(self.starts.as_slice_mut()?, *env, steps + 1, &starts);
            let flat: Vec<i64> = r.actions.iter().flatten().copied().collect();
            rows(self.actions.as_slice_mut()?, *env, steps * 3, &flat);
            let d: Vec<f32> = r.decisions.iter().flat_map(|d| [d.log_prob, d.entropy, d.value]).collect();
            rows(self.lev.as_slice_mut()?, *env, steps * 3, &d);
            rows(self.rewards.as_slice_mut()?, *env, steps, &r.reward);
            let f: Vec<f32> = r.frames.iter().map(|&f| f as f32).collect();
            rows(self.frames.as_slice_mut()?, *env, steps, &f);
            rows(self.boot.as_slice_mut()?, *env, steps, &r.bootstrap_value);
            if let Some(a) = self.interrupts.as_mut() {
                rows(a.as_slice_mut()?, *env, steps, &r.interrupt);
            }
            if let Some(a) = self.jobs.as_mut() {
                rows(a.as_slice_mut()?, *env, steps, &r.job);
            }
            if let Some(a) = self.decision_steps.as_mut() {
                rows(a.as_slice_mut()?, *env, steps, &r.step);
            }
            if let Some(a) = self.terminated.as_mut() {
                rows(a.as_slice_mut()?, *env, steps, &r.terminated);
            }
            if let Some(a) = self.truncated.as_mut() {
                rows(a.as_slice_mut()?, *env, steps, &r.truncated);
            }
            self.search.write(*env, steps, r)?;
        }
        Ok(())
    }
}

fn event_dict<'py>(
    py: Python<'py>,
    env: usize,
    event: &ZeldaEvent,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("env", env)?;
    match event {
        ZeldaEvent::Episode { info, job } => {
            d.set_item("type", "episode")?;
            d.set_item("info", json_to_py(py, &info.to_json())?)?;
            d.set_item("job", job)?;
        }
        ZeldaEvent::Eval(record) => {
            d.set_item("type", "eval")?;
            d.set_item("record", json_to_py(py, &record.to_json())?)?;
        }
        ZeldaEvent::Run(result) => {
            d.set_item("type", "run")?;
            d.set_item("result", json_to_py(py, &result.to_json())?)?;
        }
        ZeldaEvent::Room(r) => {
            d.set_item("type", "room")?;
            d.set_item("specialist", &*r.specialist)?;
            d.set_item("level", r.level)?;
            d.set_item("room", r.room)?;
            d.set_item("in_cave", r.in_cave)?;
            d.set_item("rooms", r.rooms)?;
            d.set_item("health", r.health)?;
            d.set_item("max_health", r.max_health)?;
            d.set_item("stalled", r.stalled)?;
            d.set_item("cheated", json_to_py(py, &cheated_json(&r.cheated))?)?;
        }
        ZeldaEvent::Movie(log) => {
            d.set_item("type", "movie")?;
            d.set_item("origin", log.origin.as_deref())?;
            d.set_item("buttons", PyBytes::new(py, &log.buttons))?;
            d.set_item("pokes", log.pokes.clone())?;
        }
    }
    Ok(d)
}

/// A recorded episode as a dict (see `EnvPool.poll_plays`).
fn play_dict<'py>(py: Python<'py>, env: usize, play: &Play<ZeldaAction>) -> PyResult<Bound<'py, PyDict>> {
    let n = play.steps();
    let d = PyDict::new(py);
    d.set_item("env", env)?;
    d.set_item("job", play.job)?;
    d.set_item("policy", PyArray::from_slice(py, &play.policy))?;
    let actions: Vec<i64> = play.actions.iter().flatten().copied().collect();
    d.set_item("actions", PyArray::from_vec(py, actions).reshape([n, 3])?)?;
    let column = |f: fn(&pool::rollout::Decision) -> f32| -> Vec<f32> { play.decisions.iter().map(f).collect() };
    d.set_item("log_prob", PyArray::from_vec(py, column(|d| d.log_prob)))?;
    d.set_item("entropy", PyArray::from_vec(py, column(|d| d.entropy)))?;
    d.set_item("value", PyArray::from_vec(py, column(|d| d.value)))?;
    if let Some(obs) = &play.obs {
        let o = PyDict::new(py);
        for (key, column) in LAYOUT.keys().iter().zip(&obs.columns) {
            let mut shape = vec![n];
            shape.extend(&key.shape);
            let array = match column {
                Column::I64(c) => PyArray::from_slice(py, c).reshape(shape)?.into_any(),
                Column::F32(c) => PyArray::from_slice(py, c).reshape(shape)?.into_any(),
                Column::Bool(c) => PyArray::from_slice(py, c).reshape(shape)?.into_any(),
            };
            o.set_item(&key.name, array)?;
        }
        d.set_item("obs", o)?;
    }
    Ok(d)
}

#[pymethods]
impl PyEnvPool {
    /// `num_envs` envs of `task` on `threads` worker threads (env `i` on thread `i % threads`), env `i` seeded
    /// from (`seed`, `i`); `fps > 0` paces every env's real frames; `record` records every env's inputs from
    /// its first episode on (as `set_record(env, True)` for all, but in time for the episodes started here).
    #[new]
    #[pyo3(signature = (task, num_envs, threads=None, seed=0, fps=0.0, record=false))]
    fn new(
        py: Python<'_>,
        task: &Bound<'_, PyAny>,
        num_envs: usize,
        threads: Option<usize>,
        seed: u64,
        fps: f64,
        record: bool,
    ) -> PyResult<Self> {
        let threads = threads.unwrap_or_else(|| {
            num_envs.min(std::thread::available_parallelism().map_or(1, |n| n.get()))
        });
        let mut opts = PoolOptions::new(num_envs, threads, seed);
        opts.fps = fps;
        opts.frame_shape = FRAME_SHAPE;
        opts.record = record;
        build(py, task, opts)
    }

    #[getter]
    fn num_envs(&self) -> usize {
        self.pool.num_envs()
    }

    #[getter]
    fn threads(&self) -> usize {
        self.pool.threads()
    }

    /// Envs neither finished (no job left) nor failed.
    #[getter]
    fn active(&self) -> usize {
        self.pool.active()
    }

    /// Policy names in policy-index order (the chain's specialists; the task's specialist otherwise).
    #[getter]
    fn policies(&self) -> Vec<String> {
        self.policies.clone()
    }

    /// The train pool's shaping scale (every non-terminal reward and penalty is multiplied by it); None for eval
    /// and chain pools.
    #[getter]
    fn shaping(&self) -> Option<f64> {
        self.shaping.as_ref().map(ShapingDial::get)
    }

    /// Sets the train pool's shaping scale (in [0, 1]); steps taken after the call use it.
    fn set_shaping(&self, scale: f64) -> PyResult<()> {
        let dial = self
            .shaping
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("set_shaping needs a TrainTask pool"))?;
        if !(0.0..=1.0).contains(&scale) {
            return Err(PyValueError::new_err(format!("shaping scale {scale} is not in [0, 1]")));
        }
        dial.set(scale);
        Ok(())
    }

    /// The seed start states (labels) a train pool's resets draw from (what `set_start_weights` weighs); empty for
    /// eval and chain pools and a train pool pinned to one `start_state`.
    #[getter]
    fn start_states(&self) -> Vec<String> {
        self.start_weights
            .as_ref()
            .map_or_else(Vec::new, |(_, states)| states.clone())
    }

    /// Sets the train pool's seed-state weights (prioritized level replay): {seed state label: weight}, every weight
    /// finite and ≥ 0, not all 0 (unlisted states get 0); None goes back to each env's failure-weighted draws.
    /// Resets after the call draw their seed state by these weights; harvested states and curriculum starts don't
    /// change, and health/items stay randomized per reset.
    #[pyo3(signature = (weights))]
    fn set_start_weights(&self, weights: Option<HashMap<String, f64>>) -> PyResult<()> {
        let (shared, states) = self.start_weights.as_ref().ok_or_else(|| {
            PyValueError::new_err("set_start_weights needs a TrainTask pool that draws its start states")
        })?;
        if let Some(weights) = &weights {
            for (state, w) in weights {
                if !states.contains(state) {
                    return Err(PyValueError::new_err(format!(
                        "{state:?} is not a seed start state of this pool ({states:?})"
                    )));
                }
                if !(w.is_finite() && *w >= 0.0) {
                    return Err(PyValueError::new_err(format!(
                        "start weight {w} of {state:?} is not a finite number >= 0"
                    )));
                }
            }
            if !weights.values().any(|w| *w > 0.0) {
                return Err(PyValueError::new_err("start weights are all 0"));
            }
        }
        shared.set(weights);
        Ok(())
    }

    /// 'train', 'eval' or 'chain'.
    #[getter]
    fn kind(&self) -> &'static str {
        match self.kind {
            Kind::Train => "train",
            Kind::Eval => "eval",
            Kind::Chain => "chain",
        }
    }

    /// The observation layout: key → `{'shape': [...], 'dtype': 'int64' | 'float32' | 'bool'}` (what
    /// `rollout` buffers and `poll_plays` observations hold, per row).
    #[getter]
    fn obs_spec<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for key in LAYOUT.keys() {
            let spec = PyDict::new(py);
            spec.set_item("shape", key.shape.clone())?;
            spec.set_item("dtype", key.dtype.numpy())?;
            out.set_item(&key.name, spec)?;
        }
        Ok(out)
    }

    /// Rolls every active env out `steps` decisions on the worker threads, the Rust `policies` (one
    /// `_native.Policy` per policy index) acting inside the pool with keyed noise (argmax with
    /// `deterministic`), GIL released. Writes each env at its env id row of `out`:
    /// `obs` {key: `[n, steps+1, *shape]`} for every observation key (row `steps`: where the next rollout
    /// resumes), `starts` `[n, steps+1]` float32 (1 where the row starts an episode) and, per decision,
    /// `actions` `[n, steps, 3]` int64, `logp_ent_val` `[n, steps, 3]` float32 (the Rust policy's log-prob,
    /// entropy, value), `rewards`, `frames`, `bootstrap_value` `[n, steps]` float32, and optionally
    /// `interrupts`, `job`, `step` int64 and `terminated`, `truncated` bool. Episode infos arrive through
    /// `poll_events`. Returns the number of envs rolled out.
    ///
    /// `search` (train pools only; docs/rust-actor.md "Search"): None, or a dict of Gumbel search settings
    /// (`candidates` 8, `simulations` 16, `futures` 4, `depth` 6, `gamma` 0.99, `c_visit` 50.0, `c_scale` 0.1,
    /// `fraction` 1.0, `perturb_rng` True). A searched decision executes the search's action (`actions`;
    /// `logp_ent_val` is the policy's log-prob, entropy and value of it). Optional search buffers, written whenever
    /// present (unsearched rows: false, 0, NaN, -1, 0): `search_mask` `[n, steps]` bool, `search_policy`
    /// `[n, steps, 224]` float32 (the improved policy over joint actions `kind * 32 + direction * 4 + duration`),
    /// `search_q` `[n, steps, k]` float32 (each candidate's mean return, NaN if never evaluated),
    /// `search_actions` `[n, steps, k, 3]` int64 (the candidates), `search_value` `[n, steps]` float32 (the
    /// search's root value).
    #[pyo3(signature = (policies, steps, out, deterministic=false, search=None))]
    fn rollout(
        &self,
        py: Python<'_>,
        policies: Vec<PyRef<'_, PyPolicy>>,
        steps: usize,
        out: &Bound<'_, PyDict>,
        deterministic: bool,
        search: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<usize> {
        let nets: Vec<_> = policies.iter().map(|p| p.inner.clone()).collect();
        let flags = vec![deterministic; nets.len()];
        let actor = Arc::new(PolicyActor::new(&LAYOUT, nets, flags).map_err(PyValueError::new_err)?);
        let search = search.map(search_config).transpose()?;
        if search.is_some() && !matches!(self.kind, Kind::Train) {
            return Err(PyValueError::new_err("search needs a TrainTask pool"));
        }
        let mut buffers = RolloutOut::borrow(out, self.pool.num_envs(), steps, search.as_ref().map(|s| s.candidates))?;
        let rollouts = match search {
            None => py.detach(|| self.pool.rollout(steps, actor)),
            Some(config) => {
                let planner = GumbelSearch::new(&LAYOUT, actor.clone(), config).map_err(PyValueError::new_err)?;
                py.detach(|| self.pool.rollout_planned(steps, actor, Arc::new(planner)))
            }
        }
        .map_err(pool_err)?;
        buffers.write(&rollouts)?;
        Ok(rollouts.len())
    }

    /// Lets every env act on its own with the Rust `policies` (one `_native.Policy` per policy index), decision
    /// after decision on the worker threads, until the task's jobs run out (eval jobs, chain seeds). Sampled
    /// with keyed noise (job, decision) unless `deterministic` (a bool for every policy, or one per policy).
    /// Returns at once; follow it with `poll_events`,
    /// `poll_plays`, `active` and `wait_idle`. A chain run that ends with `hold_finished_frame` parks its env
    /// until `set_hold(env, False)`. `record`: None, 'actions' or 'observations' (see `poll_plays`).
    /// `replay` {job: [n, 3] actions}: replay those recorded episodes instead (the policies score the recorded
    /// actions; a job outside `replay` or a replay running past its actions fails the env). After `play` the
    /// pool only plays (`rollout` and another `play` raise).
    #[pyo3(signature = (policies, deterministic=None, record=None, replay=None))]
    fn play(
        &self,
        py: Python<'_>,
        policies: Vec<PyRef<'_, PyPolicy>>,
        deterministic: Option<&Bound<'_, PyAny>>,
        record: Option<&str>,
        replay: Option<HashMap<i64, Bound<'_, PyAny>>>,
    ) -> PyResult<()> {
        let record = match record {
            None => Record::Off,
            Some("actions") => Record::Actions,
            Some("observations") => Record::Observations,
            Some(other) => {
                return Err(PyValueError::new_err(format!(
                    "record must be None, 'actions' or 'observations', not {other:?}"
                )));
            }
        };
        let nets: Vec<_> = policies.iter().map(|p| p.inner.clone()).collect();
        let flags: Vec<bool> = match deterministic {
            None => vec![false; nets.len()],
            Some(d) => match d.extract::<bool>() {
                Ok(all) => vec![all; nets.len()],
                Err(_) => d.extract()?,
            },
        };
        let actor = match replay {
            None => PolicyActor::new(&LAYOUT, nets, flags),
            Some(plays) => {
                let mut actions = HashMap::new();
                for (job, a) in plays {
                    let a = as_int64(py, &a, &format!("replay actions of job {job}"))?;
                    if a.shape().len() != 2 || a.shape()[1] != 3 {
                        return Err(PyValueError::new_err(format!(
                            "replay actions of job {job} must be [n, 3], got {:?}",
                            a.shape()
                        )));
                    }
                    let flat = a.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
                    actions.insert(job, flat.as_chunks::<3>().0.to_vec());
                }
                PolicyActor::replay(&LAYOUT, nets, actions)
            }
        }
        .map_err(PyValueError::new_err)?;
        py.detach(|| self.pool.play(Arc::new(actor), record))
            .map_err(pool_err)
    }

    /// Waits (GIL released) until no env is running (all finished, failed or parked) or `timeout` seconds pass;
    /// returns whether none is. Raises an env's failure.
    #[pyo3(signature = (timeout=None))]
    fn wait_idle(&self, py: Python<'_>, timeout: Option<f64>) -> PyResult<bool> {
        let deadline = timeout.map(|t| Instant::now() + Duration::from_secs_f64(t.max(0.0)));
        let tick = Duration::from_millis(50);
        loop {
            let wait = match deadline {
                Some(d) => d.saturating_duration_since(Instant::now()).min(tick),
                None => tick,
            };
            if py.detach(|| self.pool.wait_idle(Some(wait))).map_err(pool_err)? {
                return Ok(true);
            }
            if deadline.is_some_and(|d| Instant::now() >= d) {
                return Ok(false);
            }
            py.check_signals()?;
        }
    }

    /// Episodes `play(record=...)` finished since the last call, as dicts: `env`, `job` (the episode's key: an
    /// eval job's seed, a chain run's seed), per decision `policy` [n] int64, `actions` [n, 3] int64,
    /// `log_prob`, `entropy`, `value` [n] float32 (the acting policy's), and with 'observations' `obs`
    /// {key: [n, *shape]} (what each decision saw).
    fn poll_plays<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for (env, play) in self.pool.poll_plays() {
            list.append(play_dict(py, env, &play)?)?;
        }
        Ok(list)
    }

    /// Renders env `env`'s frames from its next emulated frame on (or stops).
    fn set_render(&self, env: usize, on: bool) -> PyResult<()> {
        self.pool.set_render(env, on).map_err(pool_err)
    }

    /// Holds env `env`'s latest frame (no new frames publish) until released with `on=False`.
    fn set_hold(&self, env: usize, on: bool) -> PyResult<()> {
        self.pool.set_hold(env, on).map_err(pool_err)
    }

    /// Pauses (or resumes) every env before its next real emulated frame (a step in progress stops mid-step).
    fn set_paused(&self, on: bool) {
        self.pool.set_paused(on);
    }

    /// Paces env `env` at `fps` real frames per second (`<= 0`: unthrottled).
    fn set_fps(&self, env: usize, fps: f64) -> PyResult<()> {
        self.pool.set_fps(env, fps).map_err(pool_err)
    }

    /// Records env `env`'s episode inputs from its next episode on (`movie` events).
    fn set_record(&self, env: usize, on: bool) -> PyResult<()> {
        self.pool.set_record(env, on).map_err(pool_err)
    }

    /// Env `env`'s latest frame if newer than counter `since`: `(counter, frame)` with a (240, 256, 3) uint8
    /// array (written into `out` when given), else None.
    #[pyo3(signature = (env, since=0, out=None))]
    fn frame<'py>(
        &self,
        py: Python<'py>,
        env: usize,
        since: u64,
        out: Option<Bound<'py, PyArrayDyn<u8>>>,
    ) -> PyResult<Option<(u64, Bound<'py, PyAny>)>> {
        let array = match out {
            Some(a) => a,
            None => PyArray::<u8, _>::zeros(py, FRAME_SHAPE, false)
                .to_dyn()
                .clone(),
        };
        let counter = {
            let mut rw = array
                .try_readwrite()
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let buf = rw.as_slice_mut().map_err(|e| {
                PyValueError::new_err(format!("frame buffer must be C-contiguous: {e}"))
            })?;
            if buf.len() != FRAME_BYTES {
                return Err(PyValueError::new_err(format!(
                    "frame buffer has {} bytes, frames have {FRAME_BYTES}",
                    buf.len()
                )));
            }
            self.pool.frame(env, since, buf).map_err(pool_err)?
        };
        Ok(counter.map(|c| (c, array.into_any())))
    }

    /// Frames env `env` has published (0: none yet).
    fn frame_counter(&self, env: usize) -> PyResult<u64> {
        self.pool.frame_counter(env).map_err(pool_err)
    }

    /// Frames env `env` has emulated in `play` so far, shown or not.
    fn emulated_frames(&self, env: usize) -> PyResult<u64> {
        Ok(self.pool.control(env).map_err(pool_err)?.emulated_frames())
    }

    /// Events since the last call, as dicts with `env` and `type`: `episode` (`info`: the terminal step's info, `job`:
    /// the episode's job key, its rollout rows' `job`),
    /// `eval` (`record`), `run` (`result`: the RunResult dict), `room` (`specialist`, `level`, `room`, `in_cave`,
    /// `rooms`: rooms entered so far in the episode or run, `health`, `max_health`, `stalled`, `cheated`) and
    /// `movie` (`origin`, `buttons`, `pokes`).
    fn poll_events<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for (env, event) in self.pool.poll_events() {
            list.append(event_dict(py, env, &event)?)?;
        }
        Ok(list)
    }

    /// The training curriculum's progress (`curriculum.json`), or None without a curriculum.
    fn curriculum_progress<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let Some(c) = &self.curriculum else {
            return Ok(None);
        };
        let value = serde_json::to_value(c.lock().progress()).expect("progress serializes");
        json_to_py(py, &value).map(Some)
    }

    /// Stops and joins the workers (idempotent).
    fn close(&self, py: Python<'_>) {
        py.detach(|| self.pool.close());
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __exit__(
        &self,
        py: Python<'_>,
        _ty: Option<&Bound<'_, PyAny>>,
        _value: Option<&Bound<'_, PyAny>>,
        _tb: Option<&Bound<'_, PyAny>>,
    ) -> bool {
        self.close(py);
        false
    }
}

/// (level, room, in_cave) tuples, for Python.
pub fn rooms(locations: &[zelda::MapLocation]) -> Vec<(u8, u8, bool)> {
    locations.iter().map(|l| room_of(*l)).collect()
}
