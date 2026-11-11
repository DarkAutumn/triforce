//! `TFREPLAY` files for Python: `Replay` (reading), `ReplayPlayer` (playing on zcore), `replay_verify`,
//! `replay_stitch` and `replay_truncate` (`crates/replay`).

use crate::context::{self, TfzStore};
use crate::convert::json_to_py;
use crate::errors::{ReplayError, StateFormatError};
use crate::game::screen;
use ::replay::{Event, OUTCOME_CLASSES, Player, Replay};
use numpy::{PyArray, PyArray1, PyArrayMethods};
use parking_lot::Mutex;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use serde_json::Value;
use std::path::PathBuf;
use zcore::state_file;

fn replay_err(e: ::replay::ReplayError) -> PyErr {
    ReplayError::new_err(e.to_string())
}

fn parse(data: &[u8]) -> PyResult<Replay> {
    Replay::parse(data.to_vec()).map_err(replay_err)
}

/// Reads state files where the native state store finds them (`states_dir`, default the states directory).
fn resolver(py: Python<'_>, states_dir: Option<PathBuf>) -> PyResult<impl Fn(&str) -> std::io::Result<Vec<u8>>> {
    let store = TfzStore::new(&*context::data(py, states_dir.as_deref())?, context::rom(py)?);
    Ok(move |name: &str| std::fs::read(store.path(name)))
}

/// A parsed replay file (`crates/replay` documents the format).  `data`: the file's bytes.
#[pyclass(module = "triforce._native", name = "Replay", frozen)]
pub struct PyReplay {
    replay: Replay,
}

#[pymethods]
impl PyReplay {
    #[new]
    fn new(data: &[u8]) -> PyResult<Self> {
        Ok(PyReplay { replay: parse(data)? })
    }

    /// The file's bytes.
    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, self.replay.bytes())
    }

    #[getter]
    fn version(&self) -> u32 {
        self.replay.version()
    }

    /// Input frames.
    #[getter]
    fn frames(&self) -> u32 {
        self.replay.frames()
    }

    /// 'unknown', 'win', 'death', 'stuck', 'other' or 'error'.
    #[getter]
    fn ending(&self) -> &'static str {
        self.replay.ending_name()
    }

    /// The META object: what played (`triforce`, `code`, `data`, `kind`, `route`, `models`, ...), `seed`,
    /// `source`, `ending`, and the tables the streams index (`buttons`, `cheat_items`, `reward_terms`, `actions`).
    #[getter]
    fn meta<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_py(py, &Value::Object(self.replay.meta().map_err(replay_err)?))
    }

    /// Where the replay starts: `{'kind': 'power-on', 'name', 'delay'}`, `{'kind': 'state', 'name'}` (a state
    /// file: embedded when `states()` begins with a `start` entry, else found by name) or
    /// `{'kind': 'checkpoint'}` (an embedded state that came from no file).
    #[getter]
    fn source<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let meta = self.replay.meta().map_err(replay_err)?;
        json_to_py(py, meta.get("source").unwrap_or(&Value::Null))
    }

    /// The controller byte of every frame (META `buttons` names the bits).
    fn inputs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        PyArray::from_vec(py, self.replay.inputs())
    }

    /// Every memory edit as an int64 [n, 4] array of (frame, address, previous byte, new byte); an edit at
    /// frame `k` is made before input frame `k` runs.
    fn edits<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let mut flat = Vec::with_capacity(4 * self.replay.edits().remaining() as usize);
        for edit in self.replay.edits() {
            let e = edit.map_err(replay_err)?;
            flat.extend([i64::from(e.frame), i64::from(e.addr), i64::from(e.prev), i64::from(e.value)]);
        }
        let rows = flat.len() / 4;
        Ok(PyArray::from_vec(py, flat).reshape([rows, 4])?.into_any())
    }

    /// The cheated items spent, as (frame, item name, count).
    fn cheats(&self) -> PyResult<Vec<(u32, &'static str, u32)>> {
        self.replay
            .cheats()
            .map(|c| c.map(|c| (c.frame, ::replay::CHEAT_ITEMS[usize::from(c.item)], c.count)).map_err(replay_err))
            .collect()
    }

    /// The events as dicts: `frame`, `type` ('control', 'room', 'outcome', 'episode' or 'segment') and the
    /// type's fields (room: `level`, `room`, `in_cave`, `step`, `policy`; outcome: `outcome`; episode: `ending`,
    /// `outcome`, `terminated`, `truncated`).
    fn events<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let out = PyList::empty(py);
        for record in self.replay.events() {
            let record = record.map_err(replay_err)?;
            let d = PyDict::new(py);
            d.set_item("frame", record.frame)?;
            match record.event {
                Event::Control => d.set_item("type", "control")?,
                Event::Room { level, room, in_cave, step, policy } => {
                    d.set_item("type", "room")?;
                    d.set_item("level", level)?;
                    d.set_item("room", room)?;
                    d.set_item("in_cave", in_cave)?;
                    d.set_item("step", step)?;
                    d.set_item("policy", policy)?;
                }
                Event::Outcome(outcome) => {
                    d.set_item("type", "outcome")?;
                    d.set_item("outcome", outcome)?;
                }
                Event::Episode { ending, class, terminated, truncated } => {
                    d.set_item("type", "episode")?;
                    d.set_item("ending", ending)?;
                    d.set_item("outcome", OUTCOME_CLASSES.get(usize::from(class)).copied())?;
                    d.set_item("terminated", terminated)?;
                    d.set_item("truncated", truncated)?;
                }
                Event::Segment => d.set_item("type", "segment")?,
            }
            out.append(d)?;
        }
        Ok(out)
    }

    /// The reward terms as (frame, term name, value); empty when the replay has no reward stream.
    fn rewards(&self) -> PyResult<Vec<(u32, String, f32)>> {
        let meta = self.replay.meta().map_err(replay_err)?;
        let terms: Vec<&str> = meta
            .get("reward_terms")
            .and_then(Value::as_array)
            .map_or_else(Vec::new, |terms| terms.iter().filter_map(Value::as_str).collect());
        self.replay
            .rewards()
            .map(|r| {
                let r = r.map_err(replay_err)?;
                let term = terms.get(usize::from(r.term)).ok_or_else(|| {
                    ReplayError::new_err("REWARDS section: a term that META reward_terms does not name")
                })?;
                Ok((r.frame, (*term).to_owned(), r.value))
            })
            .collect()
    }

    /// The model decisions: `frame`, `policy` int64 [n], `action` int64 [n, 3] (head indices), `value`
    /// float32 [n] and `dist` float32 [n, N], every decision's whole distribution (META `actions` gives the
    /// layout).  n = 0 when the replay has no decision stream.
    fn decisions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let width = self.replay.distribution().map_err(replay_err)?;
        let records = self.replay.decisions().map_err(replay_err)?;
        let n = records.remaining() as usize;
        let (mut frame, mut policy, mut value) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
        let (mut action, mut dist) = (Vec::with_capacity(3 * n), Vec::with_capacity(width * n));
        for decision in records {
            let d = decision.map_err(replay_err)?;
            frame.push(i64::from(d.frame));
            policy.push(i64::from(d.policy));
            action.extend(d.action.map(i64::from));
            value.push(d.value);
            dist.extend(d.distribution());
        }
        let n = frame.len();
        let out = PyDict::new(py);
        out.set_item("frame", PyArray::from_vec(py, frame))?;
        out.set_item("policy", PyArray::from_vec(py, policy))?;
        out.set_item("action", PyArray::from_vec(py, action).reshape([n, 3])?)?;
        out.set_item("value", PyArray::from_vec(py, value))?;
        out.set_item("dist", PyArray::from_vec(py, dist).reshape([n, width])?)?;
        Ok(out)
    }

    /// The states in frame order as dicts: `frame`, `kind` ('start', 'end', 'keyframe' or 'checkpoint'), `step`
    /// (the route step, or None), `granted` (cheat-granted [bombs, keys, rupees] still held), `edits` (the edits
    /// made before it) and `state` (a TFZSTATE file's bytes).  The last is the end state.  A chain can continue
    /// from any of them (`ChainTask(checkpoint=entry)`).
    fn states<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let out = PyList::empty(py);
        for state in self.replay.states() {
            let d = PyDict::new(py);
            d.set_item("frame", state.frame)?;
            d.set_item("kind", state.kind.name())?;
            d.set_item("step", state.context.step)?;
            d.set_item("granted", state.context.granted.map(u32::from).to_vec())?;
            d.set_item("edits", state.edits)?;
            d.set_item("state", PyBytes::new(py, state.blob))?;
            out.append(d)?;
        }
        Ok(out)
    }

    /// RAM then PRG-RAM of the end state (as `Game.ram()`): the machine after the last frame and edit.
    fn end_ram<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u8>>> {
        let rom = context::rom(py)?;
        let end = state_file::decode(&rom, self.replay.end_state().blob)
            .map_err(|e| StateFormatError::new_err(e.0))?;
        let ram = (0x0000..0x0800u16).chain(0x6000..0x8000).map(|addr| end.machine.peek(addr));
        Ok(PyArray::from_vec(py, ram.collect()))
    }
}

/// `a` followed by `b`, which must start (embedded) from the state `a` ends in; raises `ReplayError` when the
/// two don't join.  Returns the joined file's bytes.
#[pyfunction]
pub fn replay_stitch<'py>(py: Python<'py>, a: &[u8], b: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
    let rom = context::rom(py)?;
    // The same machine under other bytes (zstd output isn't part of the state format) still joins.
    let same = |end: &[u8], start: &[u8]| {
        end.get(30) == start.get(30) && state_file::same_machine(&rom, end, start).unwrap_or(false)
    };
    let joined = ::replay::stitch(&parse(a)?, &parse(b)?, &same).map_err(replay_err)?;
    Ok(PyBytes::new(py, &joined))
}

/// The replay up to its state `state_index` (a keyframe or a checkpoint), which becomes its end state: a run
/// can continue from it and be stitched on.
#[pyfunction]
pub fn replay_truncate<'py>(py: Python<'py>, data: &[u8], state_index: usize) -> PyResult<Bound<'py, PyBytes>> {
    let cut = ::replay::truncate(&parse(data)?, state_index).map_err(replay_err)?;
    Ok(PyBytes::new(py, &cut))
}

/// Plays the replay on the emulator, comparing the machine with every state it recorded; raises `ReplayError`
/// where it diverges.  Returns `{'frames', 'edits', 'states'}`.  `states_dir`: where a replay recorded from
/// a named state file finds it (default the states directory).
#[pyfunction]
#[pyo3(signature = (data, states_dir=None))]
pub fn replay_verify<'py>(py: Python<'py>, data: &[u8], states_dir: Option<PathBuf>) -> PyResult<Bound<'py, PyDict>> {
    let (rom, replay, resolve) = (context::rom(py)?, parse(data)?, resolver(py, states_dir)?);
    let report = py.detach(|| ::replay::verify(rom, replay, &resolve)).map_err(replay_err)?;
    let out = PyDict::new(py);
    out.set_item("frames", report.frames)?;
    out.set_item("edits", report.edits)?;
    out.set_item("states", report.states)?;
    Ok(out)
}

/// A replay being played on the emulator alone (no env, reward or model code).
#[pyclass(module = "triforce._native", name = "ReplayPlayer", frozen)]
pub struct PyReplayPlayer {
    player: Mutex<Player>,
}

#[pymethods]
impl PyReplayPlayer {
    /// `states_dir`: where a replay recorded from a named state file finds it (default the states directory).
    #[new]
    #[pyo3(signature = (data, states_dir=None))]
    fn new(py: Python<'_>, data: &[u8], states_dir: Option<PathBuf>) -> PyResult<Self> {
        let player = Player::open(context::rom(py)?, parse(data)?, &resolver(py, states_dir)?).map_err(replay_err)?;
        Ok(PyReplayPlayer { player: Mutex::new(player) })
    }

    /// Input frames run so far.
    #[getter]
    fn frame(&self) -> u32 {
        self.player.lock().frame()
    }

    /// Input frames of the replay.
    #[getter]
    fn frames(&self) -> u32 {
        self.player.lock().frames()
    }

    /// Runs up to `frames` input frames (rendering each with `render`); returns how many ran (0 at the end).
    /// Reaching the last frame also applies the edits recorded after it.
    #[pyo3(signature = (frames=1, render=false))]
    fn step(&self, py: Python<'_>, frames: u32, render: bool) -> PyResult<u32> {
        py.detach(|| {
            let mut player = self.player.lock();
            let mut ran = 0;
            while ran < frames && player.step(render)? {
                ran += 1;
            }
            if ran > 0 && player.frame() == player.frames() {
                player.finish()?;
            }
            Ok(ran)
        })
        .map_err(replay_err)
    }

    /// Moves to `frame` (that many input frames run), from the nearest recorded state when that is closer.
    /// At the last frame the edits recorded after it are applied, as `step` does.
    fn seek(&self, py: Python<'_>, frame: u32) -> PyResult<()> {
        py.detach(|| {
            let mut player = self.player.lock();
            player.seek(frame)?;
            if frame == player.frames() {
                player.finish()?;
            }
            Ok(())
        })
        .map_err(replay_err)
    }

    /// RAM then PRG-RAM (as `Game.ram()`).
    fn ram<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        let player = self.player.lock();
        let mut out = Vec::with_capacity(0x2800);
        out.extend_from_slice(player.nes().ram());
        out.extend_from_slice(player.nes().prg_ram());
        PyArray::from_vec(py, out)
    }

    /// The last frame as a (240, 256, 3) uint8 array; it must have been run with `render=True`.
    fn screen<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        screen(py, self.player.lock().nes(), true)
    }

    /// The machine as a TFZSTATE file's bytes.
    fn save_state<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let player = self.player.lock();
        PyBytes::new(py, &state_file::encode(player.nes().rom(), &player.nes().snapshot(), 0))
    }
}
