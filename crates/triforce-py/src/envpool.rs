//! `EnvPool` over `pool::Pool` with the Zelda adapters: `TrainTask`, `EvalTask` and `ChainTask` envs on
//! worker threads, stepped in lockstep or asynchronously by a caller that runs batched inference.

use crate::config::{PySpecialistConfig, extract_config, room_value};
use crate::context::{self, TfzStore};
use crate::convert::{LAYOUT, json_to_py, room_of};
use crate::envs::PyCurriculumSpec;
use crate::errors::{ConfigError, pool_err};
use crate::game::{FRAME_BYTES, FRAME_SHAPE, rgb};
use env::SharedCurriculum;
use env::chain::Route;
use env::config::{GameData, SpecialistConfig, parse_room};
use env::specialist_env::EnvOptions;
use numpy::{
    PyArray, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyReadwriteArrayDyn,
    PyUntypedArrayMethods,
};
use pool::zelda::{
    ChainRunEnv, EvalEnv, EvalJob, RgbFn, TaskEnv, TrainEnv, ZeldaAction, ZeldaEvent,
};
use pool::{Batch, ColumnMut, Dtype, EnvInit, Jobs, Pool, PoolOptions, SCALAR_COLUMNS};
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
/// curriculum. `capture_dir` saves first-entry states; `room_events` reports room changes.
#[pyclass(module = "triforce._native", name = "TrainTask", frozen)]
pub struct PyTrainTask {
    config: Arc<SpecialistConfig>,
    run_dir: PathBuf,
    start_state: Option<String>,
    curriculum: Option<PyCurriculumSpec>,
    capture_dir: Option<PathBuf>,
    states_dir: Option<PathBuf>,
    room_events: bool,
}

#[pymethods]
impl PyTrainTask {
    #[new]
    #[pyo3(signature = (config, run_dir, start_state=None, curriculum=None, capture_dir=None, states_dir=None,
                        room_events=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        config: &Bound<'_, PyAny>,
        run_dir: PathBuf,
        start_state: Option<String>,
        curriculum: Option<PyCurriculumSpec>,
        capture_dir: Option<PathBuf>,
        states_dir: Option<PathBuf>,
        room_events: bool,
    ) -> PyResult<Self> {
        Ok(PyTrainTask {
            config: extract_config(py, config)?,
            run_dir,
            start_state,
            curriculum,
            capture_dir,
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
/// `error`; any other unlisted specialist does too.
#[pyclass(module = "triforce._native", name = "ChainTask", frozen)]
pub struct PyChainTask {
    configs: Vec<Arc<SpecialistConfig>>,
    route: String,
    seeds: Vec<u64>,
    stop_at: Option<zelda::MapLocation>,
    start_from: Option<String>,
    max_steps: u64,
    missing: HashMap<String, String>,
    states_dir: Option<PathBuf>,
}

#[pymethods]
impl PyChainTask {
    #[new]
    #[pyo3(signature = (configs, route, seeds, stop_at=None, start_from=None, max_steps=50000, missing=None,
                        states_dir=None))]
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
        let data = context::data(py, None)?;
        Route::load(&data.map, &route, stop_at, start_from.as_deref())
            .map_err(ConfigError::new_err)?;
        Ok(PyChainTask {
            configs,
            route,
            seeds,
            stop_at,
            start_from,
            max_steps,
            missing: missing.unwrap_or_default(),
            states_dir,
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

/// A threaded pool of Zelda envs with caller-driven batched actions.
///
/// Buffers come from `make_buffers()` (Python-owned numpy arrays; nothing is referenced after a call
/// returns). `recv(out, min_ready)` releases the GIL until `min_ready` envs are ready, writes their rows in
/// env-id order and returns the row count; `send(env_ids, actions)` gives each returned env its next action
/// (`[kind, direction, duration]` head indices). Terminal steps reset in the same row (`final_obs` /
/// `final_valid` keep the terminal observation).
#[pyclass(module = "triforce._native", name = "EnvPool", frozen)]
pub struct PyEnvPool {
    pool: Pool<Task>,
    curriculum: Option<SharedCurriculum>,
    policies: Vec<String>,
    kind: Kind,
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
        let make = |init: EnvInit| {
            let opts = EnvOptions {
                start_state: t.start_state.clone(),
                capture_dir: t.capture_dir.clone(),
                curriculum: curriculum.clone(),
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
        });
    }
    if let Ok(t) = task.cast::<PyChainTask>() {
        let t = t.get();
        let data: Arc<GameData> = context::data(py, t.states_dir.as_deref())?;
        let store = TfzStore::new(&data, rom.clone());
        let route = Route::load(&data.map, &t.route, t.stop_at, t.start_from.as_deref())
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
            env.missing_mut().extend(t.missing.clone());
            Ok(TaskEnv::Chain(env))
        };
        let pool = py.detach(|| Pool::new(opts, make)).map_err(pool_err)?;
        return Ok(PyEnvPool {
            pool,
            curriculum: None,
            policies: t.configs.iter().map(|c| c.name.clone()).collect(),
            kind: Kind::Chain,
        });
    }
    Err(PyTypeError::new_err(
        "task must be a TrainTask, EvalTask or ChainTask",
    ))
}

fn as_int64<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
) -> PyResult<PyReadonlyArrayDyn<'py, i64>> {
    let np = py.import("numpy")?;
    let arr = np.getattr("ascontiguousarray")?.call1((obj, "int64"))?;
    Ok(arr.cast_into::<PyArrayDyn<i64>>()?.readonly())
}

fn get_array<'py, T: numpy::Element>(
    d: &Bound<'py, PyDict>,
    key: &str,
) -> PyResult<Bound<'py, PyArrayDyn<T>>> {
    let item = d
        .get_item(key)?
        .ok_or_else(|| PyKeyError::new_err(format!("buffers have no {key:?}")))?;
    item.cast_into::<PyArrayDyn<T>>().map_err(|_| {
        PyTypeError::new_err(format!(
            "buffer {key:?} must be a numpy array of the dtype make_buffers() uses"
        ))
    })
}

/// The readwrite borrows of a buffer dict, by dtype, plus where each batch column comes from.
struct Borrowed<'py> {
    i64s: Vec<PyReadwriteArrayDyn<'py, i64>>,
    f32s: Vec<PyReadwriteArrayDyn<'py, f32>>,
    bools: Vec<PyReadwriteArrayDyn<'py, bool>>,
    /// Column order: obs keys, final_obs keys, then `SCALAR_COLUMNS`; (dtype, index in its vector).
    order: Vec<(Dtype, usize)>,
}

impl<'py> Borrowed<'py> {
    fn push(&mut self, d: &Bound<'py, PyDict>, key: &str, dtype: Dtype) -> PyResult<()> {
        let rw = |e: numpy::BorrowError| PyValueError::new_err(format!("buffer {key:?}: {e}"));
        let slot = match dtype {
            Dtype::I64 => {
                self.i64s
                    .push(get_array::<i64>(d, key)?.try_readwrite().map_err(rw)?);
                self.i64s.len() - 1
            }
            Dtype::F32 => {
                self.f32s
                    .push(get_array::<f32>(d, key)?.try_readwrite().map_err(rw)?);
                self.f32s.len() - 1
            }
            Dtype::Bool => {
                self.bools
                    .push(get_array::<bool>(d, key)?.try_readwrite().map_err(rw)?);
                self.bools.len() - 1
            }
        };
        self.order.push((dtype, slot));
        Ok(())
    }

    fn new(out: &Bound<'py, PyDict>) -> PyResult<Borrowed<'py>> {
        let mut b = Borrowed {
            i64s: Vec::new(),
            f32s: Vec::new(),
            bools: Vec::new(),
            order: Vec::new(),
        };
        for group in ["obs", "final_obs"] {
            let d = out
                .get_item(group)?
                .ok_or_else(|| PyKeyError::new_err(format!("buffers have no {group:?}")))?;
            let d = d.cast_into::<PyDict>()?;
            for key in LAYOUT.keys() {
                b.push(&d, &key.name, key.dtype)?;
            }
        }
        for (name, dtype) in SCALAR_COLUMNS {
            b.push(out, name, dtype)?;
        }
        Ok(b)
    }

    /// The batch over the borrowed arrays.
    fn batch(&mut self) -> PyResult<Batch<'_>> {
        let contiguous = |e: numpy::AsSliceError| {
            PyValueError::new_err(format!("buffers must be C-contiguous: {e}"))
        };
        let mut i64s = self
            .i64s
            .iter_mut()
            .map(|a| a.as_slice_mut().map(Some))
            .collect::<Result<Vec<_>, _>>()
            .map_err(contiguous)?;
        let mut f32s = self
            .f32s
            .iter_mut()
            .map(|a| a.as_slice_mut().map(Some))
            .collect::<Result<Vec<_>, _>>()
            .map_err(contiguous)?;
        let mut bools = self
            .bools
            .iter_mut()
            .map(|a| a.as_slice_mut().map(Some))
            .collect::<Result<Vec<_>, _>>()
            .map_err(contiguous)?;
        let mut columns: Vec<ColumnMut<'_>> = self
            .order
            .iter()
            .map(|&(dtype, slot)| match dtype {
                Dtype::I64 => ColumnMut::I64(i64s[slot].take().expect("column used once")),
                Dtype::F32 => ColumnMut::F32(f32s[slot].take().expect("column used once")),
                Dtype::Bool => ColumnMut::Bool(bools[slot].take().expect("column used once")),
            })
            .collect();
        let keys = LAYOUT.keys().len();
        let mut scalars = columns.split_off(2 * keys).into_iter();
        let final_obs = columns.split_off(keys);
        let obs = columns;
        let mut next_i64 = || match scalars.next() {
            Some(ColumnMut::I64(c)) => c,
            _ => unreachable!("SCALAR_COLUMNS order"),
        };
        let env_ids = next_i64();
        let reward = match scalars.next() {
            Some(ColumnMut::F32(c)) => c,
            _ => unreachable!("SCALAR_COLUMNS order"),
        };
        let mut i64_col = || match scalars.next() {
            Some(ColumnMut::I64(c)) => c,
            _ => unreachable!("SCALAR_COLUMNS order"),
        };
        let frames = i64_col();
        let interrupt = i64_col();
        let mut bool_col = || match scalars.next() {
            Some(ColumnMut::Bool(c)) => c,
            _ => unreachable!("SCALAR_COLUMNS order"),
        };
        let terminated = bool_col();
        let truncated = bool_col();
        let bootstrap = bool_col();
        let final_valid = bool_col();
        let policy = match scalars.next() {
            Some(ColumnMut::I64(c)) => c,
            _ => unreachable!("SCALAR_COLUMNS order"),
        };
        let finished = match scalars.next() {
            Some(ColumnMut::Bool(c)) => c,
            _ => unreachable!("SCALAR_COLUMNS order"),
        };
        Ok(Batch {
            env_ids,
            obs,
            final_obs,
            reward,
            frames,
            interrupt,
            terminated,
            truncated,
            bootstrap,
            final_valid,
            policy,
            finished,
        })
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
        ZeldaEvent::Episode(info) => {
            d.set_item("type", "episode")?;
            d.set_item("info", json_to_py(py, &info.to_json())?)?;
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
            d.set_item("health", r.health)?;
            d.set_item("max_health", r.max_health)?;
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

    /// 'train', 'eval' or 'chain'.
    #[getter]
    fn kind(&self) -> &'static str {
        match self.kind {
            Kind::Train => "train",
            Kind::Eval => "eval",
            Kind::Chain => "chain",
        }
    }

    /// Zeroed buffers of `rows` rows (default `num_envs`): `obs` and `final_obs` dicts of `[rows, *shape]`
    /// arrays per observation key, the `[rows]` scalar columns, and `mask` (the same array as
    /// `obs['action_mask']`).
    #[pyo3(signature = (rows=None))]
    fn make_buffers<'py>(
        &self,
        py: Python<'py>,
        rows: Option<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let rows = rows.unwrap_or(self.pool.num_envs());
        let out = PyDict::new(py);
        for group in ["obs", "final_obs"] {
            let d = PyDict::new(py);
            for key in LAYOUT.keys() {
                let mut shape = vec![rows];
                shape.extend(&key.shape);
                let array = match key.dtype {
                    Dtype::I64 => PyArray::<i64, _>::zeros(py, shape, false).into_any(),
                    Dtype::F32 => PyArray::<f32, _>::zeros(py, shape, false).into_any(),
                    Dtype::Bool => PyArray::<bool, _>::zeros(py, shape, false).into_any(),
                };
                d.set_item(&key.name, array)?;
            }
            out.set_item(group, d)?;
        }
        for (name, dtype) in SCALAR_COLUMNS {
            let array = match dtype {
                Dtype::I64 => PyArray::<i64, _>::zeros(py, [rows], false).into_any(),
                Dtype::F32 => PyArray::<f32, _>::zeros(py, [rows], false).into_any(),
                Dtype::Bool => PyArray::<bool, _>::zeros(py, [rows], false).into_any(),
            };
            out.set_item(name, array)?;
        }
        let obs = out.get_item("obs")?.expect("obs");
        out.set_item("mask", obs.get_item("action_mask")?)?;
        Ok(out)
    }

    /// Waits (GIL released) until `min_ready` envs (default: all rows) are ready or `timeout` seconds pass,
    /// writes every ready env's row (ascending env id) into `out` and returns how many it wrote.
    #[pyo3(signature = (out, min_ready=None, timeout=None))]
    fn recv(
        &self,
        py: Python<'_>,
        out: &Bound<'_, PyDict>,
        min_ready: Option<usize>,
        timeout: Option<f64>,
    ) -> PyResult<usize> {
        let mut borrowed = Borrowed::new(out)?;
        let rows = borrowed.i64s[borrowed.order[2 * LAYOUT.keys().len()].1].len();
        let target = min_ready.unwrap_or(rows).min(rows);
        let deadline = timeout.map(|t| Instant::now() + Duration::from_secs_f64(t.max(0.0)));
        let tick = Duration::from_millis(50);
        loop {
            let wait = match deadline {
                Some(d) => d.saturating_duration_since(Instant::now()).min(tick),
                None => tick,
            };
            if py
                .detach(|| self.pool.wait(target, Some(wait)))
                .map_err(pool_err)?
            {
                break;
            }
            if deadline.is_some_and(|d| Instant::now() >= d) {
                break;
            }
            py.check_signals()?;
        }
        let mut batch = borrowed.batch()?;
        py.detach(|| self.pool.recv(&mut batch, target, Some(Duration::ZERO)))
            .map_err(pool_err)
    }

    /// Queues one action per env: `env_ids` [n] and `actions` [n, 3] (`[kind, direction, duration]`).
    fn send(
        &self,
        py: Python<'_>,
        env_ids: &Bound<'_, PyAny>,
        actions: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let ids = as_int64(py, env_ids)?;
        let acts = as_int64(py, actions)?;
        let ids = ids
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let shape = acts.shape();
        if shape.len() != 2 || shape[1] != 3 || shape[0] != ids.len() {
            return Err(PyValueError::new_err(format!(
                "actions must be [{}, 3], got {shape:?}",
                ids.len()
            )));
        }
        let flat = acts
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ids: Vec<usize> = ids
            .iter()
            .map(|&i| {
                usize::try_from(i).map_err(|_| PyValueError::new_err(format!("bad env id {i}")))
            })
            .collect::<PyResult<_>>()?;
        let actions: Vec<ZeldaAction> = flat.as_chunks::<3>().0.to_vec();
        self.pool.send(&ids, &actions).map_err(pool_err)
    }

    /// Renders env `env`'s frames from its next emulated frame on (or stops).
    fn set_render(&self, env: usize, on: bool) -> PyResult<()> {
        self.pool.set_render(env, on).map_err(pool_err)
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

    /// Events since the last call, as dicts with `env` and `type`: `episode` (`info`: the terminal step's info),
    /// `eval` (`record`), `run` (`result`: the RunResult dict), `room` (`specialist`, `level`, `room`, `health`,
    /// `max_health`) and `movie` (`origin`, `buttons`, `pokes`).
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
