//! `SpecialistConfig`: a validated, immutable specialist config (`specialists/<name>.yaml`).

use crate::context;
use crate::convert::{json_to_namespace, json_to_py, py_to_json};
use crate::errors::ConfigError;
use env::config::{SpecialistConfig as Config, format_room};
use pyo3::exceptions::PyAttributeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString, PyTuple, PyType};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::Arc;

#[pyclass(module = "triforce._native", name = "SpecialistConfig", frozen)]
pub struct PySpecialistConfig {
    pub inner: Arc<Config>,
}

/// A room from Python: `'(1, 0x73)'`, `(level, room[, in_cave])`, or an object with `level`/`value`
/// (or `room`)/`in_cave` attributes (the Python `MapLocation`), as the config's string form.
pub fn room_value(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if let Ok(s) = obj.cast::<PyString>() {
        return Ok(Value::String(s.to_str()?.to_owned()));
    }
    let (level, room, cave): (u8, u8, bool) = if let Ok(level) = obj.getattr("level") {
        let room = obj.getattr("value").or_else(|_| obj.getattr("room"))?;
        let cave = obj.getattr("in_cave").map_or(Ok(false), |c| c.extract())?;
        (level.extract()?, room.extract()?, cave)
    } else {
        let parts: Vec<Bound<'_, PyAny>> = obj.try_iter()?.collect::<PyResult<_>>()?;
        match parts.as_slice() {
            [l, r] => (l.extract()?, r.extract()?, false),
            [l, r, c] => {
                let cave = match c.cast::<PyString>() {
                    Ok(s) => s.to_str()? == "cave",
                    Err(_) => c.extract()?,
                };
                (l.extract()?, r.extract()?, cave)
            }
            _ => {
                return Err(ConfigError::new_err(
                    "a room is (level, room) or (level, room, in_cave)",
                ));
            }
        }
    };
    Ok(Value::String(format_room(zelda::MapLocation::new(
        level, room, cave,
    ))))
}

/// Config fields from keyword arguments (rooms in any accepted form).
fn fields_json(
    fields: &Bound<'_, PyDict>,
    into: &mut serde_json::Map<String, Value>,
) -> PyResult<()> {
    for (key, value) in fields.iter() {
        let key: String = key.extract()?;
        let value = if key == "rooms" {
            let rooms = value
                .try_iter()?
                .map(|r| room_value(&r?))
                .collect::<PyResult<Vec<_>>>()?;
            Value::Array(rooms)
        } else {
            py_to_json(&value)?
        };
        into.insert(key, value);
    }
    Ok(())
}

pub fn build(py: Python<'_>, value: Value) -> PyResult<Arc<Config>> {
    let data = context::data(py, None)?;
    let config = Config::from_json(value, Some(&data.map)).map_err(|e| ConfigError::new_err(e.0))?;
    data.check_chain_hp(&config).map_err(|e| ConfigError::new_err(e.0))?;
    Ok(Arc::new(config))
}

/// A config argument: a `SpecialistConfig`, a specialist name or a `.yaml` path.
pub fn extract_config(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Arc<Config>> {
    if let Ok(c) = obj.cast::<PySpecialistConfig>() {
        return Ok(c.get().inner.clone());
    }
    let name: String = obj.extract()?;
    load(py, &name)
}

pub fn load(py: Python<'_>, name_or_path: &str) -> PyResult<Arc<Config>> {
    context::data(py, None)?
        .specialist(name_or_path)
        .map_err(|e| ConfigError::new_err(e.0))
}

#[pymethods]
impl PySpecialistConfig {
    /// Builds and validates a config from fields (`SpecialistConfig(name=..., rooms=[(1, 0x73)], ...)`).
    #[new]
    #[pyo3(signature = (**fields))]
    fn new(py: Python<'_>, fields: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut m = serde_json::Map::new();
        if let Some(f) = fields {
            fields_json(f, &mut m)?;
        }
        Ok(PySpecialistConfig {
            inner: build(py, Value::Object(m))?,
        })
    }

    /// A specialist by name (`specialists/<name>.yaml`) or a config loaded from a `.yaml` path.
    #[classmethod]
    fn load(_cls: &Bound<'_, PyType>, py: Python<'_>, name_or_path: &str) -> PyResult<Self> {
        Ok(PySpecialistConfig {
            inner: load(py, name_or_path)?,
        })
    }

    /// Every specialist name (sorted).
    #[classmethod]
    fn names(_cls: &Bound<'_, PyType>, py: Python<'_>) -> PyResult<Vec<String>> {
        Ok(context::data(py, None)?
            .specialists
            .keys()
            .cloned()
            .collect())
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, py: Python<'_>, text: &str) -> PyResult<Self> {
        let value: Value =
            serde_json::from_str(text).map_err(|e| ConfigError::new_err(e.to_string()))?;
        Ok(PySpecialistConfig {
            inner: build(py, value)?,
        })
    }

    /// A copy with `fields` replaced, re-validated.
    #[pyo3(signature = (**fields))]
    fn replace(&self, py: Python<'_>, fields: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let Value::Object(mut m) = self.inner.to_json() else {
            unreachable!("a config serializes to an object")
        };
        if let Some(f) = fields {
            fields_json(f, &mut m)?;
        }
        Ok(PySpecialistConfig {
            inner: build(py, Value::Object(m))?,
        })
    }

    /// The config as plain Python values (rooms in their string form).
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_py(py, &self.inner.to_json())
    }

    fn to_json(&self) -> String {
        self.inner.to_json().to_string()
    }

    /// Rooms as (level, room, in_cave).
    #[getter]
    fn rooms(&self) -> Vec<(u8, u8, bool)> {
        self.inner
            .rooms
            .iter()
            .map(|r| crate::convert::room_of(*r))
            .collect()
    }

    /// Paths of the seed start states (`start_states` globs under the states directory).
    #[pyo3(signature = (states_dir=None))]
    fn resolve_start_states(
        &self,
        py: Python<'_>,
        states_dir: Option<PathBuf>,
    ) -> PyResult<Vec<String>> {
        let data = context::data(py, states_dir.as_deref())?;
        Ok(self
            .inner
            .resolve_start_states(&data.states_dir)
            .into_iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect())
    }

    /// The harvested start-state directory (`harvested_states`, default `<states dir>/harvested/<name>`).
    #[pyo3(signature = (states_dir=None))]
    fn harvested_dir(&self, py: Python<'_>, states_dir: Option<PathBuf>) -> PyResult<String> {
        let data = context::data(py, states_dir.as_deref())?;
        Ok(self.inner.harvested_dir(&data.states_dir).to_string_lossy().into_owned())
    }

    /// Paths of the harvested start states on disk (`<harvested_dir>/*.state`, sorted); training draws them at
    /// `harvested_fraction` (0 by default), or always when there are no seed states.
    #[pyo3(signature = (states_dir=None))]
    fn resolve_harvested_states(&self, py: Python<'_>, states_dir: Option<PathBuf>) -> PyResult<Vec<String>> {
        let data = context::data(py, states_dir.as_deref())?;
        Ok(self
            .inner
            .resolve_harvested_states(&data.states_dir)
            .into_iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect())
    }

    /// Action kind names masked out by `items`.
    fn disabled_kinds(&self) -> Vec<&'static str> {
        self.inner
            .items
            .disabled_kinds()
            .iter()
            .map(|k| k.name())
            .collect()
    }

    /// Other fields; nested blocks (`reward`, `obs`, `items`, `curriculum`, `ram_randomization`) as namespaces.
    fn __getattr__<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
        match self.inner.to_json().get(name) {
            Some(v) => json_to_namespace(py, v),
            None => Err(PyAttributeError::new_err(format!(
                "'SpecialistConfig' object has no attribute '{name}'"
            ))),
        }
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .cast::<PySpecialistConfig>()
            .is_ok_and(|o| *o.get().inner == *self.inner)
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.inner.to_json().to_string().hash(&mut h);
        h.finish()
    }

    fn __repr__(&self) -> String {
        format!("SpecialistConfig({})", self.inner.to_json())
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let from_json = py.get_type::<PySpecialistConfig>().getattr("from_json")?;
        (from_json, (self.to_json(),)).into_pyobject(py)
    }
}
