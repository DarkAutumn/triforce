//! `_native.ExternalQueue` / `_native.ExternalPolicy`: networks served from Python act inside the pool
//! (`pool::zelda::external`, docs/lab.md). `EnvPool.play` and `EnvPool.rollout` take an `ExternalPolicy` wherever
//! they take a `Policy`; a server thread (`triforce.nets.serve`) loops `take` → its networks → `answer`.

use crate::convert::LAYOUT;
use numpy::{PyArray, PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use policy::logits::Heads;
use pool::zelda::{ExternalQueue, Request};
use pool::{Column, ObsColumns};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[pyclass(name = "ExternalQueue", module = "triforce._native", frozen)]
pub struct PyExternalQueue {
    pub(crate) inner: Arc<ExternalQueue>,
    /// The server's buffers (`take` swaps them with the queue's pending ones: no allocation per batch in Rust).
    taken: Mutex<(ObsColumns, Vec<Request>)>,
}

/// One network of an `ExternalQueue`'s server, as a policy of `EnvPool.play` / `EnvPool.rollout`; `hash` identifies
/// its weights in records (the server's choice: `serve.weights_hash`).
#[pyclass(name = "ExternalPolicy", module = "triforce._native", frozen)]
pub struct PyExternalPolicy {
    pub(crate) queue: Arc<ExternalQueue>,
    pub(crate) model: usize,
    #[pyo3(get)]
    hash: u64,
}

fn duration(seconds: f64) -> Duration {
    Duration::try_from_secs_f64(seconds.max(0.0)).unwrap_or(Duration::MAX)
}

#[pymethods]
impl PyExternalQueue {
    /// A queue for networks with the Zelda heads (kinds, directions, durations) on the pool's observation layout.
    #[new]
    fn new() -> PyExternalQueue {
        let heads = Heads {
            kinds: zelda::actions::NUM_KINDS,
            directions: zelda::actions::NUM_DIRECTIONS,
            durations: zelda::actions::NUM_DURATIONS,
        };
        PyExternalQueue {
            inner: Arc::new(ExternalQueue::new(&LAYOUT, heads)),
            taken: Mutex::new((ObsColumns::new(&LAYOUT, 0), Vec::new())),
        }
    }

    /// The policy that acts with the server's network `model` (`hash`: its weights' identity).
    #[pyo3(signature = (model, hash=0))]
    fn policy(&self, model: usize, hash: u64) -> PyExternalPolicy {
        PyExternalPolicy {
            queue: self.inner.clone(),
            model,
            hash,
        }
    }

    /// Values per row of the logits `answer` takes (kind, every kind's directions, every kind's durations).
    #[getter]
    fn logits_size(&self) -> usize {
        self.inner.heads().size()
    }

    /// Waits (GIL released) up to `timeout` seconds for a request, then up to `gather` more while fewer than
    /// `target` are pending, and takes every pending one: a dict with `obs` (key -> `[n, *shape]` array, the pool's
    /// layout), `slot`, `model`, `job`, `step` (int64 `[n]`) and `bootstrap` (bool `[n]`: a value only, no decision);
    /// None when nothing came or the queue is closed.
    #[pyo3(signature = (timeout=0.1, gather=0.0, target=1))]
    fn take<'py>(
        &self,
        py: Python<'py>,
        timeout: f64,
        gather: f64,
        target: usize,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        let mut taken = self.taken.lock().unwrap_or_else(|e| e.into_inner());
        let (obs, requests) = &mut *taken;
        let got = py.detach(|| self.inner.take(duration(timeout), duration(gather), target, obs, requests));
        if !got {
            return Ok(None);
        }
        let n = requests.len();
        let d = PyDict::new(py);
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
        let ints = |f: fn(&Request) -> i64| -> Vec<i64> { requests.iter().map(f).collect() };
        d.set_item("slot", PyArray1::from_vec(py, ints(|r| r.slot as i64)))?;
        d.set_item("model", PyArray1::from_vec(py, ints(|r| r.model as i64)))?;
        d.set_item("job", PyArray1::from_vec(py, ints(|r| r.job)))?;
        d.set_item("step", PyArray1::from_vec(py, ints(|r| r.step)))?;
        let bootstrap: Vec<bool> = requests.iter().map(|r| r.bootstrap).collect();
        d.set_item("bootstrap", PyArray1::from_vec(py, bootstrap))?;
        Ok(Some(d))
    }

    /// Answers the rows of the last `take` for `slots`: `logits` `[n, logits_size]` float32, `values` `[n]`.
    fn answer(
        &self,
        slots: PyReadonlyArray1<'_, i64>,
        logits: PyReadonlyArray2<'_, f32>,
        values: PyReadonlyArray1<'_, f32>,
    ) -> PyResult<()> {
        let slots: Vec<usize> = slots
            .as_slice()?
            .iter()
            .map(|&s| usize::try_from(s).map_err(|_| PyValueError::new_err(format!("bad slot {s}"))))
            .collect::<PyResult<_>>()?;
        if logits.shape()[1] != self.inner.heads().size() {
            return Err(PyValueError::new_err(format!(
                "logits rows have {} values, expected {}",
                logits.shape()[1],
                self.inner.heads().size()
            )));
        }
        self.inner
            .answer(&slots, logits.as_slice()?, values.as_slice()?)
            .map_err(PyValueError::new_err)
    }

    /// Fails every waiting and later request (the server stopped): their envs fail with `reason`.
    fn close(&self, reason: String) {
        self.inner.close(reason);
    }

    #[getter]
    fn closed(&self) -> bool {
        self.inner.is_closed()
    }
}

#[pymethods]
impl PyExternalPolicy {
    #[getter]
    fn model(&self) -> usize {
        self.model
    }
}
