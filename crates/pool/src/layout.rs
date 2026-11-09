//! Observation layouts and storage.
//!
//! An [`ObsLayout`] lists the observation keys (name, per-row shape, dtype). [`ObsData`] holds one observation
//! (an env's latest), [`ObsColumns`] many, key by key (`[rows, *shape]` per key: a rollout's or a recorded
//! play's). Integers are `int64`, floats `float32`, flags numpy `bool` (one byte, 0 or 1), so every column maps
//! to a numpy array.

use std::sync::Arc;

/// Element type of a column.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Dtype {
    I64,
    F32,
    Bool,
}

impl Dtype {
    /// The numpy dtype name.
    pub const fn numpy(self) -> &'static str {
        match self {
            Dtype::I64 => "int64",
            Dtype::F32 => "float32",
            Dtype::Bool => "bool",
        }
    }

    pub fn from_numpy(name: &str) -> Option<Dtype> {
        match name {
            "int64" => Some(Dtype::I64),
            "float32" => Some(Dtype::F32),
            "bool" => Some(Dtype::Bool),
            _ => None,
        }
    }

    const fn slot(self) -> usize {
        match self {
            Dtype::I64 => 0,
            Dtype::F32 => 1,
            Dtype::Bool => 2,
        }
    }
}

/// One observation key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeySpec {
    pub name: String,
    /// Per-row shape (the batch column is `[rows, *shape]`).
    pub shape: Vec<usize>,
    pub dtype: Dtype,
}

impl KeySpec {
    pub fn new(name: impl Into<String>, shape: &[usize], dtype: Dtype) -> KeySpec {
        KeySpec {
            name: name.into(),
            shape: shape.to_vec(),
            dtype,
        }
    }

    /// Elements per row.
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The observation keys of a pool, in column order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObsLayout {
    keys: Vec<KeySpec>,
    /// Offset of each key inside its dtype's staging storage.
    offsets: Vec<usize>,
    /// Elements per row of each dtype (I64, F32, Bool).
    totals: [usize; 3],
}

impl ObsLayout {
    /// Panics on duplicate key names.
    pub fn new(keys: Vec<KeySpec>) -> ObsLayout {
        let mut totals = [0; 3];
        let mut offsets = Vec::with_capacity(keys.len());
        for (i, key) in keys.iter().enumerate() {
            assert!(
                keys[..i].iter().all(|k| k.name != key.name),
                "duplicate observation key {:?}",
                key.name
            );
            let slot = key.dtype.slot();
            offsets.push(totals[slot]);
            totals[slot] += key.len();
        }
        ObsLayout {
            keys,
            offsets,
            totals,
        }
    }

    pub fn keys(&self) -> &[KeySpec] {
        &self.keys
    }

    /// Index of the key named `name`.
    pub fn key(&self, name: &str) -> Option<usize> {
        self.keys.iter().position(|k| k.name == name)
    }

    fn range(&self, key: usize, dtype: Dtype) -> std::ops::Range<usize> {
        let spec = &self.keys[key];
        assert!(
            spec.dtype == dtype,
            "observation key {:?} is {}, not {}",
            spec.name,
            spec.dtype.numpy(),
            dtype.numpy()
        );
        self.offsets[key]..self.offsets[key] + spec.len()
    }
}

/// One observation in a layout (the pool's per-env staging row).
#[derive(Clone, Debug, PartialEq)]
pub struct ObsData {
    layout: Arc<ObsLayout>,
    i64s: Vec<i64>,
    f32s: Vec<f32>,
    bools: Vec<bool>,
}

impl ObsData {
    /// A zeroed observation.
    pub fn new(layout: Arc<ObsLayout>) -> ObsData {
        let [i, f, b] = layout.totals;
        ObsData {
            layout,
            i64s: vec![0; i],
            f32s: vec![0.0; f],
            bools: vec![false; b],
        }
    }

    pub fn layout(&self) -> &ObsLayout {
        &self.layout
    }

    /// The writer an env fills.
    pub fn row(&mut self) -> ObsRow<'_> {
        ObsRow {
            layout: &self.layout,
            i64s: &mut self.i64s,
            f32s: &mut self.f32s,
            bools: &mut self.bools,
        }
    }

    pub fn i64(&self, key: usize) -> &[i64] {
        &self.i64s[self.layout.range(key, Dtype::I64)]
    }

    pub fn f32(&self, key: usize) -> &[f32] {
        &self.f32s[self.layout.range(key, Dtype::F32)]
    }

    pub fn bool(&self, key: usize) -> &[bool] {
        &self.bools[self.layout.range(key, Dtype::Bool)]
    }
}

/// Observations appended row by row, stored key by key (`[rows, *shape]` per key): a rollout's observations,
/// copied out per key in one piece.
#[derive(Clone, Debug, PartialEq)]
pub struct ObsColumns {
    pub columns: Vec<Column>,
    pub rows: usize,
}

impl ObsColumns {
    pub fn new(layout: &ObsLayout, capacity: usize) -> ObsColumns {
        let columns = layout
            .keys
            .iter()
            .map(|k| match k.dtype {
                Dtype::I64 => Column::I64(Vec::with_capacity(capacity * k.len())),
                Dtype::F32 => Column::F32(Vec::with_capacity(capacity * k.len())),
                Dtype::Bool => Column::Bool(Vec::with_capacity(capacity * k.len())),
            })
            .collect();
        ObsColumns { columns, rows: 0 }
    }

    pub fn push(&mut self, obs: &ObsData) {
        for (key, column) in self.columns.iter_mut().enumerate() {
            match column {
                Column::I64(c) => c.extend_from_slice(obs.i64(key)),
                Column::F32(c) => c.extend_from_slice(obs.f32(key)),
                Column::Bool(c) => c.extend_from_slice(obs.bool(key)),
            }
        }
        self.rows += 1;
    }
}

/// Write access to one observation, key by key (indices follow [`ObsLayout::keys`]). Each accessor panics if
/// the key has another dtype. Keys the env doesn't write keep their previous contents.
pub struct ObsRow<'a> {
    layout: &'a ObsLayout,
    i64s: &'a mut [i64],
    f32s: &'a mut [f32],
    bools: &'a mut [bool],
}

impl ObsRow<'_> {
    pub fn layout(&self) -> &ObsLayout {
        self.layout
    }

    pub fn i64(&mut self, key: usize) -> &mut [i64] {
        &mut self.i64s[self.layout.range(key, Dtype::I64)]
    }

    pub fn f32(&mut self, key: usize) -> &mut [f32] {
        &mut self.f32s[self.layout.range(key, Dtype::F32)]
    }

    pub fn bool(&mut self, key: usize) -> &mut [bool] {
        &mut self.bools[self.layout.range(key, Dtype::Bool)]
    }
}

/// An owned column (one observation key of an [`ObsColumns`]).
#[derive(Clone, Debug, PartialEq)]
pub enum Column {
    I64(Vec<i64>),
    F32(Vec<f32>),
    Bool(Vec<bool>),
}

impl Column {
    /// Panics unless this is an int64 column.
    pub fn i64(&self) -> &[i64] {
        match self {
            Column::I64(c) => c,
            _ => panic!("not an int64 column"),
        }
    }

    /// Panics unless this is a float32 column.
    pub fn f32(&self) -> &[f32] {
        match self {
            Column::F32(c) => c,
            _ => panic!("not a float32 column"),
        }
    }

    /// Panics unless this is a bool column.
    pub fn bool(&self) -> &[bool] {
        match self {
            Column::Bool(c) => c,
            _ => panic!("not a bool column"),
        }
    }
}

/// An env's latest step, staged on its worker.
#[derive(Clone, Debug)]
pub(crate) struct Staged {
    pub obs: ObsData,
    pub final_obs: ObsData,
    pub outcome: crate::Outcome,
    pub final_valid: bool,
    pub policy: i64,
    pub finished: bool,
    /// [`crate::Start::job`] of the current episode; -1 when finished.
    pub job: i64,
    /// Decisions taken in the current episode.
    pub step: i64,
}

impl Staged {
    pub fn new(layout: &Arc<ObsLayout>) -> Staged {
        Staged {
            obs: ObsData::new(layout.clone()),
            final_obs: ObsData::new(layout.clone()),
            outcome: crate::Outcome::default(),
            final_valid: false,
            policy: 0,
            finished: false,
            job: 0,
            step: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two keys per dtype, so offsets inside a dtype's storage matter; one scalar key (shape []).
    fn layout() -> Arc<ObsLayout> {
        Arc::new(ObsLayout::new(vec![
            KeySpec::new("a", &[2], Dtype::I64),
            KeySpec::new("x", &[2, 2], Dtype::F32),
            KeySpec::new("m", &[3], Dtype::Bool),
            KeySpec::new("b", &[], Dtype::I64),
            KeySpec::new("y", &[1], Dtype::F32),
            KeySpec::new("n", &[2], Dtype::Bool),
        ]))
    }

    fn filled(layout: &Arc<ObsLayout>, base: i64) -> ObsData {
        let mut d = ObsData::new(layout.clone());
        let mut row = d.row();
        row.i64(0).copy_from_slice(&[base, base + 1]);
        row.f32(1).copy_from_slice(&[0.5, 1.5, 2.5, base as f32]);
        row.bool(2).copy_from_slice(&[true, false, true]);
        row.i64(3)[0] = base + 2;
        row.f32(4)[0] = -1.0;
        row.bool(5).copy_from_slice(&[false, true]);
        d
    }

    #[test]
    fn numpy_names_round_trip() {
        for d in [Dtype::I64, Dtype::F32, Dtype::Bool] {
            assert_eq!(Dtype::from_numpy(d.numpy()), Some(d));
        }
        assert_eq!(Dtype::from_numpy("float64"), None);
        assert_eq!(Dtype::from_numpy("uint8"), None);
    }

    #[test]
    fn keys_of_one_dtype_get_disjoint_ranges() {
        let layout = layout();
        assert_eq!(layout.key("y"), Some(4));
        assert_eq!(layout.key("missing"), None);
        assert_eq!(layout.keys()[3].len(), 1, "a scalar key holds one element");
        assert!(KeySpec::new("e", &[0, 4], Dtype::F32).is_empty());
        let d = filled(&layout, 10);
        assert_eq!(d.i64(0), [10, 11]);
        assert_eq!(d.i64(3), [12]);
        assert_eq!(d.f32(1), [0.5, 1.5, 2.5, 10.0]);
        assert_eq!(d.f32(4), [-1.0]);
        assert_eq!(d.bool(2), [true, false, true]);
        assert_eq!(d.bool(5), [false, true]);
        // Keys a writer leaves alone keep their contents.
        let mut again = d.clone();
        again.row().i64(3)[0] = 99;
        assert_eq!((again.i64(0), again.i64(3)), (&[10, 11][..], &[99][..]));
    }

    #[test]
    #[should_panic(expected = "observation key \"x\" is float32, not int64")]
    fn reading_a_key_as_another_dtype_panics() {
        let mut d = ObsData::new(layout());
        d.row().i64(1);
    }

    #[test]
    #[should_panic(expected = "duplicate observation key \"a\"")]
    fn duplicate_keys_panic() {
        ObsLayout::new(vec![
            KeySpec::new("a", &[1], Dtype::I64),
            KeySpec::new("a", &[1], Dtype::F32),
        ]);
    }

    #[test]
    #[should_panic(expected = "not a bool column")]
    fn owned_columns_check_their_dtype() {
        Column::I64(vec![1]).bool();
    }

    #[test]
    fn columns_store_rows_key_by_key() {
        let layout = layout();
        let mut c = ObsColumns::new(&layout, 2);
        c.push(&filled(&layout, 10));
        c.push(&filled(&layout, 20));
        assert_eq!(c.rows, 2);
        assert_eq!(c.columns[0].i64(), [10, 11, 20, 21]);
        assert_eq!(c.columns[3].i64(), [12, 22]);
        assert_eq!(c.columns[1].f32(), [0.5, 1.5, 2.5, 10.0, 0.5, 1.5, 2.5, 20.0]);
        assert_eq!(c.columns[5].bool(), [false, true, false, true]);
    }
}
