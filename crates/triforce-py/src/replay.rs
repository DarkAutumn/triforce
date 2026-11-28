//! `TFREPLAY` files for Python: `Replay` (reading), `ReplayPlayer` (playing on zcore), `replay_verify`,
//! `replay_stitch`, `replay_truncate`, `replay_rewrite` and `replay_cut_states` (`crates/replay`).

use crate::context::{self, TfzStore};
use crate::convert::{json_to_py, py_to_json};
use crate::errors::{ReplayError, StateFormatError};
use crate::game::screen_with;
use ::replay::{ENDINGS, EVENT_NAMES, Ending, Event, Names, OUTCOME_CLASSES, Player, Replay, Rewrite, Window, section};
use numpy::{PyArray, PyArray1, PyArrayMethods};
use parking_lot::Mutex;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use serde_json::{Map, Value};
use std::path::PathBuf;
use zcore::state_file;
use zcore::tracks::Mix;

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

    /// The events as dicts: `frame`, `type` ('control', 'room', 'outcome', 'episode', 'segment', 'visit', 'reset':
    /// the console's RESET button before the frame, or 'load': a savestate loaded) and the
    /// type's fields (room: `level`, `room`, `in_cave`, `step`, `policy`; outcome: `outcome`; episode: `ending`,
    /// `outcome`, `terminated`, `truncated`; visit: `record`, the room visit's record as the env judged it; load:
    /// `state`, the index in `states()` of the state loaded, `edits` and `states`, the edits and states before it).
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
                Event::Reset => d.set_item("type", "reset")?,
                Event::Load { state, edits, states } => {
                    d.set_item("type", "load")?;
                    d.set_item("state", state)?;
                    d.set_item("edits", edits)?;
                    d.set_item("states", states)?;
                }
                Event::Visit(text) => {
                    d.set_item("type", "visit")?;
                    // Python's parser: serde_json's default float parsing can be one ulp off (no
                    // `float_roundtrip`, which would change every crate's parsing), and the record must read back
                    // as the chain wrote it (`replay.chain_records`).
                    let record = py.import("json")?.call_method1("loads", (text,)).map_err(|e| {
                        ReplayError::new_err(format!("EVENTS section: a visit record: {e}"))
                    })?;
                    d.set_item("record", record)?;
                }
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
    /// float32 [n], `dist` float32 [n, N], every decision's whole distribution (META `actions` gives the
    /// layout), and from version 2 `draw` (how the act mode drew it, META `tables.draws`; None before), `low` bool
    /// [n] (its low-health part), `danger` (per decision None or `{joint index: P(hit)}` over the legal joint
    /// actions) and `foresight` (per decision None or `{outcome, trials, frames, vetoed: [joint indices]}`).
    /// n = 0 when the replay has no decision stream.
    fn decisions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let width = self.replay.distribution().map_err(replay_err)?;
        let names = Names::of(&self.replay.meta().map_err(replay_err)?);
        let records = self.replay.decisions().map_err(replay_err)?;
        let n = records.remaining() as usize;
        let (mut frame, mut policy, mut value) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
        let (mut action, mut dist) = (Vec::with_capacity(3 * n), Vec::with_capacity(width * n));
        let (draw, danger, foresight) = (PyList::empty(py), PyList::empty(py), PyList::empty(py));
        let mut low = Vec::with_capacity(n);
        for decision in records {
            let d = decision.map_err(replay_err)?;
            frame.push(i64::from(d.frame));
            policy.push(i64::from(d.policy));
            action.extend(d.action.map(i64::from));
            value.push(d.value);
            dist.extend(d.distribution());
            draw.append(d.draw.map(|i| names.draw(usize::from(i))))?;
            low.push(d.low);
            match d.danger {
                Some(_) => {
                    let row = PyDict::new(py);
                    for (joint, p) in d.dangers() {
                        row.set_item(joint, p)?;
                    }
                    danger.append(row)?;
                }
                None => danger.append(py.None())?,
            }
            match d.foresight {
                Some(f) => {
                    let row = PyDict::new(py);
                    row.set_item("outcome", ["", "safe", "replaced", "unsafe"].get(usize::from(f.outcome)).copied())?;
                    row.set_item("trials", f.trials)?;
                    row.set_item("frames", f.frames)?;
                    row.set_item("vetoed", f.vetoed().collect::<Vec<_>>())?;
                    foresight.append(row)?;
                }
                None => foresight.append(py.None())?,
            }
        }
        let n = frame.len();
        let out = PyDict::new(py);
        out.set_item("frame", PyArray::from_vec(py, frame))?;
        out.set_item("policy", PyArray::from_vec(py, policy))?;
        out.set_item("action", PyArray::from_vec(py, action).reshape([n, 3])?)?;
        out.set_item("value", PyArray::from_vec(py, value))?;
        out.set_item("dist", PyArray::from_vec(py, dist).reshape([n, width])?)?;
        out.set_item("draw", draw)?;
        out.set_item("low", PyArray::from_vec(py, low))?;
        out.set_item("danger", danger)?;
        out.set_item("foresight", foresight)?;
        Ok(out)
    }

    /// Every decision read for people, named by META `actions` (a kind only with the heads it has: `MOVE N 16`,
    /// `SWORD E`, `WAIT 8`): per decision a dict of `chosen` (label, p, danger: P(hit) or None), `top` (the `top`
    /// most likely legal actions, the same triples, most likely first), `legal` (how many were legal) and `vetoed`
    /// (the planner's vetoed actions' labels, in the order tried).
    #[pyo3(signature = (top=5))]
    fn choices<'py>(&self, py: Python<'py>, top: usize) -> PyResult<Bound<'py, PyList>> {
        let names = Names::of(&self.replay.meta().map_err(replay_err)?);
        let out = PyList::empty(py);
        for decision in self.replay.decisions().map_err(replay_err)? {
            let d = decision.map_err(replay_err)?;
            let choices = names.choices(&d);
            let [k, di, u] = d.action.map(usize::from);
            let chosen = choices.iter().find(|c| (c.kind, c.direction, c.duration) == (k, di, u));
            let row = PyDict::new(py);
            row.set_item(
                "chosen",
                chosen.map_or_else(|| (names.action_text(k, di, u), None, None), |c| (c.label.clone(), Some(c.p), c.danger)),
            )?;
            let best: Vec<_> = choices.iter().take(top).map(|c| (c.label.clone(), c.p, c.danger)).collect();
            row.set_item("top", best)?;
            row.set_item("legal", choices.len())?;
            let vetoed: Vec<String> =
                d.foresight.map_or_else(Vec::new, |f| f.vetoed().map(|j| names.joint_text(usize::from(j))).collect());
            row.set_item("vetoed", vetoed)?;
            out.append(row)?;
        }
        Ok(out)
    }

    /// The objective's changes (version 2; empty before), named by META `tables`: dicts of `frame`, `objective`
    /// ('FIGHT', ...), `flags`, `route`, `step`, `way`, `stretch`, `next_rooms` and `retreats` ('8:0x3c') and
    /// `targets` (each `{role, obs, reward_only, <the role's fields>}`: `{'role': 'monster', 'slot': 3, 'type': 42,
    /// ...}`, `{'role': 'bomb-wall', 'col': 15, 'row': 2, 'dir': 'N', ...}`).
    fn targets<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let names = Names::of(&self.replay.meta().map_err(replay_err)?);
        let out = PyList::empty(py);
        for record in self.replay.targets() {
            out.append(json_to_py(py, &names.targets_json(&record.map_err(replay_err)?))?)?;
        }
        Ok(out)
    }

    /// The action mask's changes (version 2; empty before): dicts of `frame`, `kinds` ({'MOVE': {'directions':
    /// ['N', ...], 'durations': [4, 8, 16, 32]}, ...}: the legal kinds) and `blocked` ({'BOMB': ['tenth'], ...}: the
    /// rules behind a kind's block, or that keep it to some directions).
    fn masks<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let names = Names::of(&self.replay.meta().map_err(replay_err)?);
        let out = PyList::empty(py);
        for record in self.replay.masks() {
            out.append(json_to_py(py, &names.mask_json(&record.map_err(replay_err)?))?)?;
        }
        Ok(out)
    }

    /// The states in frame order as dicts: `frame`, `kind` ('start', 'end', 'keyframe' or 'checkpoint'), `step`
    /// (the route step, or None), `granted` (cheat-granted [bombs, keys, rupees] still held: a room's grants),
    /// `route_granted` (a route step's grants still held, `zelda::cheats`), `edits` (the edits
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
            d.set_item("route_granted", state.context.route_granted.map(u32::from).to_vec())?;
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

/// A window of `replay_rewrite` as given: owned bytes the [`Window`] borrows.
struct Given {
    from: u32,
    to: u32,
    start: Vec<u8>,
    end: Option<Vec<u8>>,
    about: Map<String, Value>,
}

fn given(d: &Bound<'_, PyDict>) -> PyResult<Given> {
    let need = |key: &str| d.get_item(key)?.ok_or_else(|| PyValueError::new_err(format!("a window needs '{key}'")));
    let bytes = |v: Bound<'_, PyAny>| -> PyResult<Vec<u8>> { Ok(v.cast::<PyBytes>()?.as_bytes().to_vec()) };
    let about = match d.get_item("about")? {
        None => Map::new(),
        Some(about) if about.is_none() => Map::new(),
        Some(about) => match py_to_json(&about)? {
            Value::Object(about) => about,
            _ => return Err(PyValueError::new_err("a window's 'about' is a dict")),
        },
    };
    let end = d.get_item("end")?.filter(|end| !end.is_none()).map(bytes).transpose()?;
    Ok(Given { from: need("from")?.extract()?, to: need("to")?.extract()?, start: bytes(need("start")?)?, end, about })
}

/// The replay rewritten (`crates/replay` `rewrite`): only some of its streams and only some stretches of it.
/// `watch`: start from what watching needs (no REWARDS, DECISIONS, TARGETS, MASKS, keyframes, OUTCOME, EPISODE or
/// VISIT events) instead of everything; `sections` (names of REWARDS, DECISIONS, TARGETS, MASKS), `events` (event
/// type names; 'reset' and 'load' are always kept) and `keyframes` say what is kept instead.  `windows`: the
/// stretches kept, in order, each a dict `{'from', 'to'}` (input frames, `to` excluded), `'start'` (the machine
/// before input `from` after its RESET, a TFZSTATE's bytes: `replay_cut_states`), optional `'end'` (before input
/// `to`'s RESET; found by playing when left out) and `'about'` (META `clip` entries); the stretches after the first
/// are loaded where the one before ends (LOAD events).  `ending`: `{'kind': one of ENDINGS, 'result', 'outcome',
/// 'reason'}` for the output (default: the source's when the last stretch reaches its end).  Returns the bytes.
#[pyfunction]
#[pyo3(signature = (data, *, watch=false, sections=None, events=None, keyframes=None, windows=None, ending=None))]
#[allow(clippy::too_many_arguments)]
pub fn replay_rewrite<'py>(
    py: Python<'py>,
    data: &[u8],
    watch: bool,
    sections: Option<Vec<String>>,
    events: Option<Vec<String>>,
    keyframes: Option<bool>,
    windows: Option<Vec<Bound<'py, PyDict>>>,
    ending: Option<Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyBytes>> {
    let replay = parse(data)?;
    let base = if watch { Rewrite::watching() } else { Rewrite::default() };
    let sections = match sections {
        None => base.sections,
        Some(names) => names
            .iter()
            .map(|name| {
                section::OPTIONAL.into_iter().find(|&k| section::NAMES[k as usize] == name).ok_or_else(|| {
                    PyValueError::new_err(format!("{name}: not an optional section (REWARDS, DECISIONS, TARGETS, MASKS)"))
                })
            })
            .collect::<PyResult<_>>()?,
    };
    let events = match events {
        None => base.events,
        Some(names) => names
            .iter()
            .map(|name| match EVENT_NAMES.iter().position(|n| !n.is_empty() && n == name) {
                Some(code) => Ok(code as u8),
                None => Err(PyValueError::new_err(format!("{name}: not an event ({})", EVENT_NAMES[1..].join(", ")))),
            })
            .collect::<PyResult<_>>()?,
    };
    let given: Vec<Given> = windows.unwrap_or_default().iter().map(given).collect::<PyResult<_>>()?;
    let ending = match ending {
        None => None,
        Some(d) => {
            let text = |key: &str| -> PyResult<String> {
                Ok(d.get_item(key)?.map(|v| v.extract()).transpose()?.unwrap_or_default())
            };
            let kind = text("kind")?;
            let Some(kind) = ENDINGS.iter().position(|&e| e == kind) else {
                return Err(PyValueError::new_err(format!("ending kind {kind:?}: one of {}", ENDINGS.join(", "))));
            };
            Some(Ending { kind: kind as u8, result: text("result")?, outcome: text("outcome")?, reason: text("reason")? })
        }
    };
    let spec = Rewrite {
        sections,
        events,
        keyframes: keyframes.unwrap_or(base.keyframes),
        windows: given
            .iter()
            .map(|w| Window { from: w.from, to: w.to, start: &w.start, end: w.end.as_deref(), about: w.about.clone() })
            .collect(),
        ending,
    };
    let rom = context::rom(py)?;
    let out = py.detach(|| ::replay::rewrite_played(rom, &replay, &spec)).map_err(replay_err)?;
    Ok(PyBytes::new(py, &out))
}

/// The machine at each of `frames` (input frames, at most the replay's), in one pass over the replay: `{frame:
/// (start, end)}`, TFZSTATE bytes of the machine before that input, `start` after the frame's RESET (a
/// `replay_rewrite` window from it starts there), `end` before it (a window to it ends there).
#[pyfunction]
#[pyo3(signature = (data, frames, states_dir=None))]
pub fn replay_cut_states<'py>(
    py: Python<'py>,
    data: &[u8],
    frames: Vec<u32>,
    states_dir: Option<PathBuf>,
) -> PyResult<Bound<'py, PyDict>> {
    let (rom, replay, resolve) = (context::rom(py)?, parse(data)?, resolver(py, states_dir)?);
    let cuts = py.detach(|| ::replay::cut_states(rom, replay, &resolve, &frames)).map_err(replay_err)?;
    let out = PyDict::new(py);
    for (frame, (start, end)) in cuts {
        out.set_item(frame, (PyBytes::new(py, &start), PyBytes::new(py, &end)))?;
    }
    Ok(out)
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

/// The owner's two bomb metrics over a replay (`env::bomb_windows::BombFrames`): every frame played from the
/// replay's start, its EDITS (cheated bombs are not Link's own), and `facts`, the env's per-decision facts
/// `[(frame, kill_all, bomb_allowed, bomb_never), ...]` in frame order (the replay's targets stream: the objective
/// FIGHT, BOMB allowed by the bomb rule, the mode `never`; `frame` the input frames run before the decision). Returns
/// the run's windows: `[{'metric': 'tenth'|'carrier', 'open_frame', 'open_room', 'close_frame', 'close_room'
/// ('3:0x5b', '0:0x77:cave'), 'outcome', 'killed_type', 'killed_slot' (None: nothing), 'full_bag', 'drop' (None, or
/// 'none', 'pending', 'fell', 'taken', 'wasted', 'lost')}]`.
#[pyfunction]
#[pyo3(signature = (data, facts, states_dir=None))]
pub fn replay_bomb_windows<'py>(
    py: Python<'py>,
    data: &[u8],
    facts: Vec<(u64, bool, bool, bool)>,
    states_dir: Option<PathBuf>,
) -> PyResult<Bound<'py, PyList>> {
    use env::bomb_windows::{BombFrames, StepFacts};
    let (rom, replay, resolve) = (context::rom(py)?, parse(data)?, resolver(py, states_dir)?);
    let edits: Vec<::replay::Edit> = replay.edits().collect::<Result<_, _>>().map_err(replay_err)?;
    let windows = py
        .detach(|| {
            let mut frames = BombFrames::new();
            for (frame, kill_all, bomb_allowed, bomb_never) in facts {
                frames.facts(frame, StepFacts { kill_all, bomb_allowed, bomb_never });
            }
            let mut player = Player::open(rom, replay, &resolve)?;
            let mut edit = edits.iter().peekable();
            let mut image = Box::new([0u8; zelda::tables::RAM_SIZE]);
            loop {
                // The edits tagged with the next input frame are applied before it runs.
                while let Some(e) = edit.next_if(|e| e.frame <= player.frame()) {
                    frames.edit(e.addr, e.prev, e.value);
                }
                if !player.step(false)? {
                    break;
                }
                image[..0x800].copy_from_slice(player.nes().ram());
                image[0x800..].copy_from_slice(player.nes().prg_ram());
                frames.push(&image);
            }
            Ok(frames.finish())
        })
        .map_err(replay_err)?;
    let room = |l: zelda::MapLocation| format!("{}:0x{:02x}{}", l.level, l.value, if l.in_cave { ":cave" } else { "" });
    let list = PyList::empty(py);
    for w in windows {
        let d = PyDict::new(py);
        d.set_item("metric", w.metric.name())?;
        d.set_item("open_frame", w.open_frame)?;
        d.set_item("open_room", room(w.open_room))?;
        d.set_item("close_frame", w.close_frame)?;
        d.set_item("close_room", room(w.close_room))?;
        d.set_item("outcome", w.outcome.name())?;
        d.set_item("killed_type", w.killed.map(|k| k.0))?;
        d.set_item("killed_slot", w.killed.map(|k| k.1))?;
        d.set_item("full_bag", w.full_bag)?;
        d.set_item("drop", w.drop.map(|f| f.name()))?;
        list.append(d)?;
    }
    Ok(list)
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

    /// The last frame as a (240, 256, 3) uint8 array; it must have been run with `render=True`.  With
    /// `controller` the pad is drawn into it, the buttons held on that frame lit (as `ReplayScreen` draws it).
    #[pyo3(signature = (controller=false))]
    fn screen<'py>(&self, py: Python<'py>, controller: bool) -> PyResult<Bound<'py, PyAny>> {
        let player = self.player.lock();
        let buttons = player.buttons();
        screen_with(py, player.nes(), true, |frame| {
            if controller {
                ::theater::draw_controller(frame, buttons);
            }
        })
    }

    /// The controller byte of the last frame run (A 1, B 2, SELECT 4, START 8, UP 16, DOWN 32, LEFT 64,
    /// RIGHT 128; 0 before the first frame).
    #[getter]
    fn buttons(&self) -> u8 {
        self.player.lock().buttons()
    }

    /// Turns sound on at `rate` samples per second, or off (None).  It starts from whatever the game is
    /// playing at that moment; every frame stepped from now on adds its samples to what `audio()` returns.
    /// A `seek` is silent: the sound picks up where it lands.  Playback is the same with or without sound.
    /// `split`: `audio()` gives music and sound effects as two tracks (`set_mix` says which parts are heard).
    #[pyo3(signature = (rate=None, split=false))]
    fn set_sound(&self, rate: Option<f64>, split: bool) -> PyResult<()> {
        if rate.is_some_and(|rate| !(8000.0..=384_000.0).contains(&rate)) {
            return Err(pyo3::exceptions::PyValueError::new_err("rate must be 8000 to 384000 samples per second"));
        }
        let mut player = self.player.lock();
        player.set_split(split, false);
        player.set_sound(rate);
        Ok(())
    }

    /// Which parts of the sound are heard from the next frame on: the music (songs), the sound effects (tunes,
    /// noise effects, samples, the item fanfare and the end-of-level tune) and, among the effects, the
    /// low-hearts beep.  All on is the game's own sound.  Best set before `set_sound`.
    #[pyo3(signature = (music=true, effects=true, heart_beep=true))]
    fn set_mix(&self, music: bool, effects: bool, heart_beep: bool) {
        self.player.lock().set_mix(Mix { music, effects, beep: heart_beep });
    }

    /// The sound of the frames stepped since the last call (or since `set_sound`): a float32 array of mono
    /// samples in -1..1, or with `set_sound(split=True)` of shape (samples, 2): music, sound effects.  A frame
    /// is 1/60.0988 s: about 798.7 samples at 48 kHz.
    fn audio<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let mut player = self.player.lock();
        let samples = std::mem::take(player.audio());
        let pairs = samples.len() / 2;
        let array = PyArray::from_vec(py, samples);
        if player.split() {
            return Ok(array.reshape([pairs, 2])?.into_any());
        }
        Ok(array.into_any())
    }

    /// The machine as a TFZSTATE file's bytes.
    fn save_state<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let player = self.player.lock();
        PyBytes::new(py, &state_file::encode(player.nes().rom(), &player.nes().snapshot(), 0))
    }
}
