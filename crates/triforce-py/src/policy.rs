//! `_native.Policy`: the Rust forward pass of the specialist network (`crates/policy`), for parity tests,
//! replays and one-off scoring from Python. Rollouts use the same `policy::Policy` inside the pool.

use numpy::{PyArray1, PyArray2, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use policy::{Input, Output, Pick, Policy, Tensor};
use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use std::sync::Arc;

#[pyclass(name = "Policy", module = "triforce._native", frozen)]
pub struct PyPolicy {
    pub inner: Arc<Policy>,
}

fn policy_err(e: policy::PolicyError) -> PyErr {
    PyValueError::new_err(e.0)
}

fn contiguous<'py, T: numpy::Element>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
    dtype: &str,
    what: &str,
) -> PyResult<PyReadonlyArrayDyn<'py, T>> {
    let np = py.import("numpy")?;
    let arr = np.getattr("ascontiguousarray")?.call1((obj, dtype))?;
    arr.cast_into::<PyArrayDyn<T>>()
        .map_err(|_| PyTypeError::new_err(format!("{what} must convert to a {dtype} array")))?
        .try_readonly()
        .map_err(|e| PyValueError::new_err(format!("{what}: {e}")))
}

/// Given actions as int64, refusing a non-integer dtype (floats would be truncated to other actions).
fn integer_actions<'py>(py: Python<'py>, obj: &Bound<'py, PyAny>) -> PyResult<PyReadonlyArrayDyn<'py, i64>> {
    let arr = py.import("numpy")?.getattr("asarray")?.call1((obj,))?;
    let kind: String = arr.getattr("dtype")?.getattr("kind")?.extract()?;
    if kind != "i" && kind != "u" {
        return Err(PyValueError::new_err(format!("actions must be integers, got dtype kind '{kind}'")));
    }
    contiguous(py, &arr, "int64", "actions")
}

/// One given action index as usize: a negative one is an error, not a huge index.
fn action_index(value: i64) -> PyResult<usize> {
    usize::try_from(value).map_err(|_| PyValueError::new_err(format!("action index {value} is negative")))
}

/// Batched observation arrays borrowed from a dict of numpy arrays; every key has the mask's batch size.
struct Batch<'py> {
    room: PyReadonlyArrayDyn<'py, i64>,
    types: PyReadonlyArrayDyn<'py, i64>,
    features: PyReadonlyArrayDyn<'py, f32>,
    grid: PyReadonlyArrayDyn<'py, i64>,
    vectors: Vec<PyReadonlyArrayDyn<'py, f32>>,
    mask: PyReadonlyArrayDyn<'py, bool>,
    rows: usize,
}

impl<'py> Batch<'py> {
    fn new(py: Python<'py>, policy: &Policy, obs: &Bound<'py, PyDict>) -> PyResult<Batch<'py>> {
        let get = |key: &str| {
            obs.get_item(key)?
                .ok_or_else(|| PyKeyError::new_err(format!("observation has no '{key}'")))
        };
        // Any numpy-truthy mask byte is True: normalise to 0/1 bools (a Rust bool holding 2 is undefined).
        let truthy = py.import("numpy")?.getattr("not_equal")?.call1((get("action_mask")?, 0))?;
        let mask = contiguous::<bool>(py, &truthy, "bool", "action_mask")?;
        let rows = mask.shape().first().copied().unwrap_or(0);
        let lead = |what: &str, shape: &[usize]| -> PyResult<()> {
            match shape.first() {
                Some(&n) if n == rows => Ok(()),
                n => Err(PyValueError::new_err(format!(
                    "{what} has {} rows, action_mask has {rows}",
                    n.map_or("no".to_owned(), |n| n.to_string())
                ))),
            }
        };
        let mut vectors = Vec::new();
        for key in policy.vector_keys() {
            let v = contiguous::<f32>(py, &get(key)?, "float32", key)?;
            lead(key, v.shape())?;
            vectors.push(v);
        }
        let room = contiguous::<i64>(py, &get("room_id")?, "int64", "room_id")?;
        lead("room_id", room.shape())?;
        if room.len() != rows {
            return Err(PyValueError::new_err(format!("room_id has {} values for {rows} rows", room.len())));
        }
        let types = contiguous::<i64>(py, &get("entity_types")?, "int64", "entity_types")?;
        lead("entity_types", types.shape())?;
        let features = contiguous::<f32>(py, &get("entity_features")?, "float32", "entity_features")?;
        lead("entity_features", features.shape())?;
        let grid = contiguous::<i64>(py, &get("grid")?, "int64", "grid")?;
        lead("grid", grid.shape())?;
        Ok(Batch { room, types, features, grid, vectors, mask, rows })
    }

    /// Row `i` (`Policy::check` validates each row's lengths).
    fn row<'a>(&'a self, i: usize, vectors: &'a mut Vec<&'a [f32]>) -> PyResult<Input<'a>> {
        let per = |a: usize| a / self.rows.max(1);
        let room = self.room.as_slice()?;
        let types = self.types.as_slice()?;
        let features = self.features.as_slice()?;
        let grid = self.grid.as_slice()?;
        let mask = self.mask.as_slice()?;
        let (t, f, g, m) = (per(types.len()), per(features.len()), per(grid.len()), per(mask.len()));
        for v in &self.vectors {
            let v = v.as_slice()?;
            let n = per(v.len());
            vectors.push(&v[i * n..(i + 1) * n]);
        }
        Ok(Input {
            room_id: room[i],
            entity_types: &types[i * t..(i + 1) * t],
            entity_features: &features[i * f..(i + 1) * f],
            grid: &grid[i * g..(i + 1) * g],
            vectors: vectors.as_slice(),
            mask: &mask[i * m..(i + 1) * m],
        })
    }
}

#[pymethods]
impl PyPolicy {
    /// `config_json`: the checkpoint's `model_config` as JSON (with `obs_spec`); `weights`: the `state_dict` as
    /// name → float32 numpy array.
    #[new]
    fn new(py: Python<'_>, config_json: &str, weights: &Bound<'_, PyDict>) -> PyResult<PyPolicy> {
        let config = policy::ModelConfig::from_json(config_json).map_err(policy_err)?;
        let mut tensors = HashMap::new();
        for (name, value) in weights.iter() {
            let name: String = name.extract()?;
            let arr = contiguous::<f32>(py, &value, "float32", &name)?;
            tensors.insert(
                name,
                Tensor {
                    shape: arr.shape().to_vec(),
                    data: arr.as_slice()?.to_vec(),
                },
            );
        }
        let inner = py.detach(|| Policy::new(config, tensors)).map_err(policy_err)?;
        Ok(PyPolicy { inner: Arc::new(inner) })
    }

    /// Identifies the model (config and every weight's bits).
    #[getter]
    fn hash(&self) -> u64 {
        self.inner.hash()
    }

    #[getter]
    fn vector_keys(&self) -> Vec<String> {
        self.inner.vector_keys().to_vec()
    }

    #[getter]
    fn noise_size(&self) -> usize {
        self.inner.noise_size()
    }

    /// (kinds, directions, durations).
    #[getter]
    fn head_sizes(&self) -> (usize, usize, usize) {
        let [k, d, u] = self.inner.head_sizes();
        (k, d, u)
    }

    /// Evaluates a batch (dict of `[B, ...]` arrays incl. `action_mask`). Picks by `noise` (`[B, noise_size]`
    /// Gumbel samples), else by argmax with `deterministic`, or scores `actions` (`[B, 3]`). Returns a dict:
    /// `actions` `[B,3]`, `log_prob`, `entropy`, `value` `[B]`, `kind_log_probs` `[B,K]`, `direction_log_probs`
    /// `[B,D]`, `duration_log_probs` `[B,U]` (given the chosen kind), `head_applies` `[B,3]`, `head_entropy`
    /// `[B,3]`, `head_log_prob` `[B,3]`.
    #[pyo3(signature = (obs, noise=None, actions=None, deterministic=false))]
    fn evaluate<'py>(
        &self,
        py: Python<'py>,
        obs: &Bound<'py, PyDict>,
        noise: Option<&Bound<'py, PyAny>>,
        actions: Option<&Bound<'py, PyAny>>,
        deterministic: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let batch = Batch::new(py, &self.inner, obs)?;
        let rows = batch.rows;
        if rows == 0 {
            return Err(PyValueError::new_err("empty batch"));
        }
        let noise = noise.map(|n| contiguous::<f32>(py, n, "float32", "noise")).transpose()?;
        let actions = actions.map(|a| integer_actions(py, a)).transpose()?;
        if noise.is_none() && actions.is_none() && !deterministic {
            return Err(PyValueError::new_err("pass noise, actions or deterministic=True"));
        }
        let ns = self.inner.noise_size();
        let noise_slice = noise.as_ref().map(|n| n.as_slice()).transpose()?;
        let action_slice = actions.as_ref().map(|a| a.as_slice()).transpose()?;
        if noise_slice.is_some_and(|n| n.len() != rows * ns) {
            return Err(PyValueError::new_err(format!("noise must be [{rows}, {ns}]")));
        }
        if action_slice.is_some_and(|a| a.len() != rows * 3) {
            return Err(PyValueError::new_err(format!("actions must be [{rows}, 3]")));
        }
        let mut scratch = self.inner.scratch();
        let mut outs: Vec<Output> = Vec::with_capacity(rows);
        for i in 0..rows {
            let mut vectors = Vec::new();
            let input = batch.row(i, &mut vectors)?;
            let pick = if let Some(a) = action_slice {
                Pick::Given([action_index(a[i * 3])?, action_index(a[i * 3 + 1])?, action_index(a[i * 3 + 2])?])
            } else if let Some(n) = noise_slice {
                Pick::Sample(&n[i * ns..(i + 1) * ns])
            } else {
                Pick::Argmax
            };
            outs.push(self.inner.evaluate(&input, pick, &mut scratch).map_err(policy_err)?);
        }
        output_dict(py, &self.inner, &outs)
    }

    /// Value estimates `[B]`.
    fn value<'py>(&self, py: Python<'py>, obs: &Bound<'py, PyDict>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let batch = Batch::new(py, &self.inner, obs)?;
        let mut scratch = self.inner.scratch();
        let mut values = Vec::with_capacity(batch.rows);
        for i in 0..batch.rows {
            let mut vectors = Vec::new();
            let input = batch.row(i, &mut vectors)?;
            values.push(self.inner.value(&input, &mut scratch).map_err(policy_err)?);
        }
        Ok(PyArray1::from_vec(py, values))
    }

    /// Joint log-probs `[B, J]` (J = kinds × directions × durations = 224; entry `kind * 32 + direction * 4 +
    /// duration`; `-inf` where illegal) and values `[B]`, one trunk pass per row (`Policy::joint_log_probs`: what
    /// `evaluate(actions=...)` gives each joint action, bit for bit).
    fn joint_log_probs<'py>(
        &self,
        py: Python<'py>,
        obs: &Bound<'py, PyDict>,
    ) -> PyResult<JointOut<'py>> {
        let batch = Batch::new(py, &self.inner, obs)?;
        let joint = self.inner.joint_size();
        let mut scratch = self.inner.scratch();
        let mut out = vec![0f32; batch.rows * joint];
        let mut values = Vec::with_capacity(batch.rows);
        for (i, row) in out.chunks_exact_mut(joint).enumerate() {
            let mut vectors = Vec::new();
            let input = batch.row(i, &mut vectors)?;
            values.push(self.inner.joint_log_probs(&input, &mut scratch, row).map_err(policy_err)?);
        }
        let out = PyArray1::from_vec(py, out).reshape([batch.rows, joint])?;
        Ok((out, PyArray1::from_vec(py, values)))
    }
}

/// `Policy.joint_log_probs`: (`[B, J]` log-probs, `[B]` values).
type JointOut<'py> = (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<f32>>);

fn output_dict<'py>(py: Python<'py>, policy: &Policy, outs: &[Output]) -> PyResult<Bound<'py, PyDict>> {
    let [k, d, u] = policy.head_sizes();
    let d_out = PyDict::new(py);
    let actions: Vec<Vec<i64>> = outs.iter().map(|o| o.action.iter().map(|&a| a as i64).collect()).collect();
    d_out.set_item("actions", PyArray2::from_vec2(py, &actions)?)?;
    d_out.set_item("log_prob", PyArray1::from_iter(py, outs.iter().map(|o| o.log_prob)))?;
    d_out.set_item("entropy", PyArray1::from_iter(py, outs.iter().map(|o| o.entropy)))?;
    d_out.set_item("value", PyArray1::from_iter(py, outs.iter().map(|o| o.value)))?;
    for (h, (name, n)) in [("kind_log_probs", k), ("direction_log_probs", d), ("duration_log_probs", u)]
        .into_iter()
        .enumerate()
    {
        let rows_v: Vec<Vec<f32>> = outs.iter().map(|o| o.head_log_probs[h][..n].to_vec()).collect();
        d_out.set_item(name, PyArray2::from_vec2(py, &rows_v)?)?;
    }
    let applies: Vec<Vec<bool>> = outs.iter().map(|o| o.head_applies.to_vec()).collect();
    d_out.set_item("head_applies", PyArray2::from_vec2(py, &applies)?)?;
    let ent: Vec<Vec<f32>> = outs.iter().map(|o| o.head_entropy.to_vec()).collect();
    d_out.set_item("head_entropy", PyArray2::from_vec2(py, &ent)?)?;
    let lp: Vec<Vec<f32>> = outs.iter().map(|o| o.head_log_prob.to_vec()).collect();
    d_out.set_item("head_log_prob", PyArray2::from_vec2(py, &lp)?)?;
    Ok(d_out)
}

/// Keyed Gumbel noise `[n, size]` for decisions `steps[i]` of jobs `jobs[i]` (`policy::noise::keyed_noise`).
#[pyfunction]
pub fn keyed_noise<'py>(
    py: Python<'py>,
    jobs: Vec<i64>,
    steps: Vec<i64>,
    size: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    if jobs.len() != steps.len() {
        return Err(PyValueError::new_err("jobs and steps must have the same length"));
    }
    let rows: Vec<Vec<f32>> = jobs
        .iter()
        .zip(&steps)
        .map(|(&j, &s)| {
            let mut row = vec![0f32; size];
            policy::noise::keyed_noise(j, s, &mut row);
            row
        })
        .collect();
    if rows.is_empty() {
        return Ok(PyArray2::zeros(py, [0, size], false));
    }
    PyArray2::from_vec2(py, &rows).map_err(|e| PyValueError::new_err(e.to_string()))
}
