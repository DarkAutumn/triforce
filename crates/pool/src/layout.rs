//! Observation layouts and the caller-owned batch buffers [`crate::Pool::recv`] fills.
//!
//! An [`ObsLayout`] lists the observation keys (name, per-row shape, dtype). A batch holds one C-contiguous
//! column per observation key, shaped `[rows, *shape]`, a second set of columns for final observations, and the
//! per-row scalar columns of [`SCALAR_COLUMNS`]. Integers are `int64`, floats `float32`, flags numpy `bool`
//! (one byte, 0 or 1), so every column maps to a numpy array the PyO3 layer allocates once ([`BufferSpec`]).

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

    /// Copies key `key` into row `row` of `column`.
    fn copy_to(&self, key: usize, column: &mut ColumnMut<'_>, row: usize) {
        let n = self.layout.keys[key].len();
        let dst = row * n..(row + 1) * n;
        match column {
            ColumnMut::I64(c) => c[dst].copy_from_slice(self.i64(key)),
            ColumnMut::F32(c) => c[dst].copy_from_slice(self.f32(key)),
            ColumnMut::Bool(c) => c[dst].copy_from_slice(self.bool(key)),
        }
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

/// A borrowed batch column (`rows * key.len()` elements, row-major).
#[derive(Debug)]
pub enum ColumnMut<'a> {
    I64(&'a mut [i64]),
    F32(&'a mut [f32]),
    Bool(&'a mut [bool]),
}

impl ColumnMut<'_> {
    pub fn dtype(&self) -> Dtype {
        match self {
            ColumnMut::I64(_) => Dtype::I64,
            ColumnMut::F32(_) => Dtype::F32,
            ColumnMut::Bool(_) => Dtype::Bool,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            ColumnMut::I64(c) => c.len(),
            ColumnMut::F32(c) => c.len(),
            ColumnMut::Bool(c) => c.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The scalar columns of a batch: name, dtype (each `[rows]`). Names match the [`Batch`] fields.
pub const SCALAR_COLUMNS: [(&str, Dtype); 12] = [
    ("env_ids", Dtype::I64),
    ("reward", Dtype::F32),
    ("frames", Dtype::I64),
    ("interrupt", Dtype::I64),
    ("terminated", Dtype::Bool),
    ("truncated", Dtype::Bool),
    ("bootstrap", Dtype::Bool),
    ("final_valid", Dtype::Bool),
    ("policy", Dtype::I64),
    ("finished", Dtype::Bool),
    ("job", Dtype::I64),
    ("step", Dtype::I64),
];

/// Caller-owned output of [`crate::Pool::recv`]: `rows` = `env_ids.len()`, the most rows one call writes.
/// Rows `0..n` (n returned by `recv`) are written in ascending env id order; the rest keep their contents.
///
/// Per row: `obs` is the observation the env's next action answers (after a terminal step, the first
/// observation of the next episode). `final_obs` is written only where `final_valid` is set: the observation the
/// terminal step produced, before the automatic reset. `reward`/`frames`/`interrupt`/`terminated`/`truncated`/
/// `bootstrap` describe the step that produced the row (zero/false on an env's first row). `policy` is the
/// policy (e.g. chain specialist) that acts on `obs`, -1 when finished. `finished`: the env has no episode left
/// (its `obs` row is not written and it takes no more actions). `job` is the episode's job key
/// ([`crate::Start::job`], -1 when finished) and `step` how many decisions the episode took before `obs`.
#[derive(Debug)]
pub struct Batch<'a> {
    pub env_ids: &'a mut [i64],
    /// One column per [`ObsLayout`] key, in order.
    pub obs: Vec<ColumnMut<'a>>,
    /// Same shapes as `obs`.
    pub final_obs: Vec<ColumnMut<'a>>,
    pub reward: &'a mut [f32],
    pub frames: &'a mut [i64],
    pub interrupt: &'a mut [i64],
    pub terminated: &'a mut [bool],
    pub truncated: &'a mut [bool],
    pub bootstrap: &'a mut [bool],
    pub final_valid: &'a mut [bool],
    pub policy: &'a mut [i64],
    pub finished: &'a mut [bool],
    pub job: &'a mut [i64],
    pub step: &'a mut [i64],
}

impl Batch<'_> {
    pub fn rows(&self) -> usize {
        self.env_ids.len()
    }

    /// Checks every column's dtype and length against `layout` and `rows()`.
    pub fn validate(&self, layout: &ObsLayout) -> Result<(), String> {
        let rows = self.rows();
        let scalars = [
            ("reward", self.reward.len()),
            ("frames", self.frames.len()),
            ("interrupt", self.interrupt.len()),
            ("terminated", self.terminated.len()),
            ("truncated", self.truncated.len()),
            ("bootstrap", self.bootstrap.len()),
            ("final_valid", self.final_valid.len()),
            ("policy", self.policy.len()),
            ("finished", self.finished.len()),
            ("job", self.job.len()),
            ("step", self.step.len()),
        ];
        for (name, len) in scalars {
            if len != rows {
                return Err(format!("{name} has {len} rows, env_ids has {rows}"));
            }
        }
        for (group, columns) in [("obs", &self.obs), ("final_obs", &self.final_obs)] {
            if columns.len() != layout.keys.len() {
                return Err(format!(
                    "{group} has {} columns, the layout has {} keys",
                    columns.len(),
                    layout.keys.len()
                ));
            }
            for (key, column) in layout.keys.iter().zip(columns) {
                if column.dtype() != key.dtype || column.len() != rows * key.len() {
                    return Err(format!(
                        "{group}[{:?}] is {} x {}, expected {} x {} ({rows} rows of {:?})",
                        key.name,
                        column.dtype().numpy(),
                        column.len(),
                        key.dtype.numpy(),
                        rows * key.len(),
                        key.shape
                    ));
                }
            }
        }
        Ok(())
    }

    /// Writes one staged row (pool internal).
    pub(crate) fn write(&mut self, row: usize, env: usize, staged: &Staged) {
        self.env_ids[row] = env as i64;
        let o = &staged.outcome;
        self.reward[row] = o.reward;
        self.frames[row] = o.frames;
        self.interrupt[row] = o.interrupt;
        self.terminated[row] = o.terminated;
        self.truncated[row] = o.truncated;
        self.bootstrap[row] = o.bootstrap;
        self.final_valid[row] = staged.final_valid;
        self.policy[row] = staged.policy;
        self.finished[row] = staged.finished;
        self.job[row] = staged.job;
        self.step[row] = staged.step;
        let keys = staged.obs.layout.keys.len();
        if !staged.finished {
            for (key, column) in (0..keys).zip(self.obs.iter_mut()) {
                staged.obs.copy_to(key, column, row);
            }
        }
        if staged.final_valid {
            for (key, column) in (0..keys).zip(self.final_obs.iter_mut()) {
                staged.final_obs.copy_to(key, column, row);
            }
        }
    }
}

/// An owned column.
#[derive(Clone, Debug, PartialEq)]
pub enum Column {
    I64(Vec<i64>),
    F32(Vec<f32>),
    Bool(Vec<bool>),
}

impl Column {
    fn zeroed(dtype: Dtype, len: usize) -> Column {
        match dtype {
            Dtype::I64 => Column::I64(vec![0; len]),
            Dtype::F32 => Column::F32(vec![0.0; len]),
            Dtype::Bool => Column::Bool(vec![false; len]),
        }
    }

    fn as_mut(&mut self) -> ColumnMut<'_> {
        match self {
            Column::I64(c) => ColumnMut::I64(c),
            Column::F32(c) => ColumnMut::F32(c),
            Column::Bool(c) => ColumnMut::Bool(c),
        }
    }

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

/// Owned batch buffers (Rust callers and tests; the PyO3 layer uses numpy arrays shaped by [`BufferSpec`]).
#[derive(Clone, Debug, PartialEq)]
pub struct Buffers {
    layout: Arc<ObsLayout>,
    pub env_ids: Vec<i64>,
    pub obs: Vec<Column>,
    pub final_obs: Vec<Column>,
    pub reward: Vec<f32>,
    pub frames: Vec<i64>,
    pub interrupt: Vec<i64>,
    pub terminated: Vec<bool>,
    pub truncated: Vec<bool>,
    pub bootstrap: Vec<bool>,
    pub final_valid: Vec<bool>,
    pub policy: Vec<i64>,
    pub finished: Vec<bool>,
    pub job: Vec<i64>,
    pub step: Vec<i64>,
}

impl Buffers {
    /// Zeroed buffers of `rows` rows.
    pub fn new(layout: Arc<ObsLayout>, rows: usize) -> Buffers {
        let columns = |layout: &ObsLayout| -> Vec<Column> {
            layout
                .keys
                .iter()
                .map(|k| Column::zeroed(k.dtype, rows * k.len()))
                .collect()
        };
        Buffers {
            obs: columns(&layout),
            final_obs: columns(&layout),
            layout,
            env_ids: vec![0; rows],
            reward: vec![0.0; rows],
            frames: vec![0; rows],
            interrupt: vec![0; rows],
            terminated: vec![false; rows],
            truncated: vec![false; rows],
            bootstrap: vec![false; rows],
            final_valid: vec![false; rows],
            policy: vec![0; rows],
            finished: vec![false; rows],
            job: vec![0; rows],
            step: vec![0; rows],
        }
    }

    pub fn layout(&self) -> &ObsLayout {
        &self.layout
    }

    pub fn rows(&self) -> usize {
        self.env_ids.len()
    }

    pub fn batch(&mut self) -> Batch<'_> {
        Batch {
            env_ids: &mut self.env_ids,
            obs: self.obs.iter_mut().map(Column::as_mut).collect(),
            final_obs: self.final_obs.iter_mut().map(Column::as_mut).collect(),
            reward: &mut self.reward,
            frames: &mut self.frames,
            interrupt: &mut self.interrupt,
            terminated: &mut self.terminated,
            truncated: &mut self.truncated,
            bootstrap: &mut self.bootstrap,
            final_valid: &mut self.final_valid,
            policy: &mut self.policy,
            finished: &mut self.finished,
            job: &mut self.job,
            step: &mut self.step,
        }
    }

    /// Elements of `key` in batch row `row`.
    pub fn row_range(&self, key: usize, row: usize) -> std::ops::Range<usize> {
        let n = self.layout.keys[key].len();
        row * n..(row + 1) * n
    }
}

/// Everything the PyO3 layer needs to allocate a batch: `rows`, the observation keys (`obs` and `final_obs`
/// columns are `[rows, *key.shape]` of `key.dtype`) and the `[rows]` scalar columns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BufferSpec {
    pub rows: usize,
    pub obs: Vec<KeySpec>,
    pub scalars: &'static [(&'static str, Dtype)],
}

/// Step results the pool stages per env until `recv` copies them out.
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

    fn staged(layout: &Arc<ObsLayout>, base: i64, final_valid: bool, finished: bool) -> Staged {
        Staged {
            obs: filled(layout, base),
            final_obs: filled(layout, base + 100),
            outcome: crate::Outcome {
                reward: 0.25,
                frames: 9,
                interrupt: 2,
                terminated: final_valid,
                truncated: false,
                bootstrap: false,
                policy: 0,
            },
            final_valid,
            policy: if finished { -1 } else { 3 },
            finished,
            job: if finished { -1 } else { base + 1000 },
            step: base / 10,
        }
    }

    #[test]
    fn written_rows_land_at_their_offsets_for_every_dtype() {
        let layout = layout();
        let mut b = Buffers::new(layout.clone(), 3);
        assert_eq!(b.rows(), 3);
        assert_eq!(b.layout(), &*layout);
        b.batch().validate(&layout).unwrap();
        b.batch().write(1, 7, &staged(&layout, 20, false, false));
        assert_eq!(b.env_ids, [0, 7, 0]);
        assert_eq!(
            (b.reward[1], b.frames[1], b.interrupt[1], b.policy[1]),
            (0.25, 9, 2, 3)
        );
        assert_eq!((b.job[1], b.step[1]), (1020, 2));
        assert_eq!(b.obs[0].i64(), [0, 0, 20, 21, 0, 0]);
        assert_eq!(b.obs[3].i64(), [0, 22, 0]);
        assert_eq!(&b.obs[1].f32()[4..8], [0.5, 1.5, 2.5, 20.0]);
        assert_eq!(b.obs[4].f32(), [0.0, -1.0, 0.0]);
        assert_eq!(
            b.obs[2].bool(),
            [false, false, false, true, false, true, false, false, false]
        );
        assert_eq!(b.obs[5].bool(), [false, false, false, true, false, false]);
        // No final observation unless it is valid.
        assert_eq!(b.final_obs, Buffers::new(layout.clone(), 3).final_obs);

        // A terminal row also writes its final observation.
        b.batch().write(2, 8, &staged(&layout, 30, true, false));
        assert_eq!(b.final_obs[0].i64(), [0, 0, 0, 0, 130, 131]);
        assert!(b.final_valid[2] && b.terminated[2]);
        assert_eq!(b.obs[0].i64()[4..], [30, 31]);

        // A finished env's row keeps the old observation; its final observation is still written.
        b.batch().write(1, 7, &staged(&layout, 40, true, true));
        assert_eq!(b.obs[0].i64()[2..4], [20, 21]);
        assert_eq!(b.final_obs[0].i64()[2..4], [140, 141]);
        assert!(b.finished[1] && b.policy[1] == -1 && b.job[1] == -1);
    }

    #[test]
    fn validate_names_the_mismatched_column() {
        let layout = layout();
        let mut b = Buffers::new(layout.clone(), 2);
        b.policy.pop();
        assert_eq!(
            b.batch().validate(&layout).unwrap_err(),
            "policy has 1 rows, env_ids has 2"
        );

        let mut b = Buffers::new(layout.clone(), 2);
        b.final_obs.pop();
        assert_eq!(
            b.batch().validate(&layout).unwrap_err(),
            "final_obs has 5 columns, the layout has 6 keys"
        );

        let mut b = Buffers::new(layout.clone(), 2);
        b.obs[1] = Column::I64(vec![0; 8]);
        assert_eq!(
            b.batch().validate(&layout).unwrap_err(),
            "obs[\"x\"] is int64 x 8, expected float32 x 8 (2 rows of [2, 2])"
        );

        let mut b = Buffers::new(layout.clone(), 2);
        b.final_obs[2] = Column::Bool(vec![false; 3]);
        assert_eq!(
            b.batch().validate(&layout).unwrap_err(),
            "final_obs[\"m\"] is bool x 3, expected bool x 6 (2 rows of [3])"
        );
        assert_eq!(b.row_range(2, 1), 3..6);
    }
}
