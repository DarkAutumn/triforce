//! Conversions between Rust values and Python objects: JSON values, observations, actions.

use crate::errors::InvalidActionError;
use env::observation::Observation;
use numpy::{PyArray, PyArrayMethods};
use pool::zelda::{obs_layout, write_observation};
use pool::{Dtype, ObsData, ObsLayout};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use serde_json::{Map, Number, Value};
use std::sync::{Arc, LazyLock};
use zelda::Action;

pub static LAYOUT: LazyLock<Arc<ObsLayout>> = LazyLock::new(|| Arc::new(obs_layout()));

/// A JSON value as plain Python objects (dicts keep key order).
pub fn json_to_py<'py>(py: Python<'py>, v: &Value) -> PyResult<Bound<'py, PyAny>> {
    Ok(match v {
        Value::Null => py.None().into_bound(py),
        Value::Bool(b) => PyBool::new(py, *b).to_owned().into_any(),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into_pyobject(py)?.into_any()
            } else if let Some(u) = n.as_u64() {
                u.into_pyobject(py)?.into_any()
            } else {
                n.as_f64().unwrap_or(f64::NAN).into_pyobject(py)?.into_any()
            }
        }
        Value::String(s) => PyString::new(py, s).into_any(),
        Value::Array(a) => {
            let items = a
                .iter()
                .map(|x| json_to_py(py, x))
                .collect::<PyResult<Vec<_>>>()?;
            PyList::new(py, items)?.into_any()
        }
        Value::Object(m) => {
            let d = PyDict::new(py);
            for (k, x) in m {
                d.set_item(k, json_to_py(py, x)?)?;
            }
            d.into_any()
        }
    })
}

/// A JSON value with objects as `types.SimpleNamespace` (attribute access for nested config blocks).
pub fn json_to_namespace<'py>(py: Python<'py>, v: &Value) -> PyResult<Bound<'py, PyAny>> {
    match v {
        Value::Object(m) => {
            let kwargs = PyDict::new(py);
            for (k, x) in m {
                kwargs.set_item(k, json_to_namespace(py, x)?)?;
            }
            py.import("types")?
                .getattr("SimpleNamespace")?
                .call((), Some(&kwargs))
        }
        Value::Array(a) => {
            let items = a
                .iter()
                .map(|x| json_to_namespace(py, x))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PyList::new(py, items)?.into_any())
        }
        other => json_to_py(py, other),
    }
}

/// A Python value as JSON: dicts, lists, tuples, scalars, None, and objects with a `__dict__` (namespaces).
pub fn py_to_json(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_none() {
        return Ok(Value::Null);
    }
    if let Ok(b) = obj.cast::<PyBool>() {
        return Ok(Value::Bool(b.is_true()));
    }
    if obj.is_instance_of::<PyInt>() {
        if let Ok(i) = obj.extract::<i64>() {
            return Ok(Value::from(i));
        }
        return Ok(Value::from(obj.extract::<u64>()?));
    }
    if obj.is_instance_of::<PyFloat>() {
        let f: f64 = obj.extract()?;
        return Number::from_f64(f)
            .map(Value::Number)
            .ok_or_else(|| PyTypeError::new_err(format!("{f} is not a JSON number")));
    }
    if let Ok(s) = obj.cast::<PyString>() {
        return Ok(Value::String(s.to_str()?.to_owned()));
    }
    if let Ok(d) = obj.cast::<PyDict>() {
        let mut m = Map::new();
        for (k, v) in d.iter() {
            m.insert(k.str()?.to_str()?.to_owned(), py_to_json(&v)?);
        }
        return Ok(Value::Object(m));
    }
    if obj.is_instance_of::<PyList>() || obj.is_instance_of::<PyTuple>() {
        let items = obj
            .try_iter()?
            .map(|x| py_to_json(&x?))
            .collect::<PyResult<Vec<_>>>()?;
        return Ok(Value::Array(items));
    }
    if let Ok(d) = obj.getattr("__dict__")
        && let Ok(d) = d.cast::<PyDict>()
    {
        return py_to_json(d.as_any());
    }
    // numpy scalars and other numbers.
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(Value::from(i));
    }
    if let Ok(f) = obj.extract::<f64>() {
        return Ok(Number::from_f64(f).map_or(Value::Null, Value::Number));
    }
    Err(PyTypeError::new_err(format!(
        "can't convert {} to a config value",
        obj.get_type().name()?
    )))
}

/// An observation as `{key: ndarray}` (`OBS_SPEC` shapes and dtypes).
pub fn obs_to_dict<'py>(py: Python<'py>, obs: &Observation) -> PyResult<Bound<'py, PyDict>> {
    let mut data = ObsData::new(LAYOUT.clone());
    write_observation(obs, &mut data.row());
    obs_data_to_dict(py, &data)
}

pub fn obs_data_to_dict<'py>(py: Python<'py>, data: &ObsData) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (i, key) in data.layout().keys().iter().enumerate() {
        let shape = key.shape.clone();
        let array = match key.dtype {
            Dtype::I64 => PyArray::from_slice(py, data.i64(i))
                .reshape(shape)?
                .into_any(),
            Dtype::F32 => PyArray::from_slice(py, data.f32(i))
                .reshape(shape)?
                .into_any(),
            Dtype::Bool => PyArray::from_slice(py, data.bool(i))
                .reshape(shape)?
                .into_any(),
        };
        d.set_item(&key.name, array)?;
    }
    Ok(d)
}

/// An action from Python: `'MOVE N 16'`, `(kind, direction, duration)` head indices, or an object with
/// `to_indices()` (the Python `Action`).
pub fn extract_action(obj: &Bound<'_, PyAny>) -> PyResult<Action> {
    if let Ok(s) = obj.cast::<PyString>() {
        return Action::parse(s.to_str()?).map_err(|e| InvalidActionError::new_err(e.0));
    }
    let indices = if let Ok(f) = obj.getattr("to_indices") {
        f.call0()?
    } else {
        obj.clone()
    };
    let v: Vec<i64> = indices
        .try_iter()?
        .map(|x| x?.extract::<i64>())
        .collect::<PyResult<_>>()?;
    let [k, d, u]: [i64; 3] = v.try_into().map_err(|v: Vec<i64>| {
        InvalidActionError::new_err(format!("an action has 3 head indices, got {}", v.len()))
    })?;
    let idx = |x: i64| {
        usize::try_from(x)
            .map_err(|_| InvalidActionError::new_err(format!("action head index {x} is negative")))
    };
    Action::from_indices(idx(k)?, idx(d)?, idx(u)?).map_err(|e| InvalidActionError::new_err(e.0))
}

/// `[kind, direction, duration]` head indices of an action.
pub fn action_indices(action: &Action) -> (usize, usize, usize) {
    let [k, d, u] = action.to_indices();
    (k, d, u)
}

/// (level, room, in_cave).
pub type Room = (u8, u8, bool);

pub fn room_of(location: zelda::MapLocation) -> Room {
    (location.level, location.value, location.in_cave)
}
