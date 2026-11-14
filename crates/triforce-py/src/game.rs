//! `Game`: one game session for tools and tests (`GameSession` + the emulator surface `play` needs), and the
//! frozen `GameState` / `Change` / `Snapshot` copies it hands out.

use crate::context::{self, TfzStore};
use crate::convert::{action_indices, extract_action, room_of};
use crate::describe;
use crate::errors::{RenderDisabledError, StateFormatError};
use crate::signals::interruptible;
use numpy::{PyArray, PyArray1, PyArrayMethods};
use parking_lot::Mutex;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyString, PyTuple};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use zcore::{MachineState, ZCore, state_file};
use zelda::enums::Direction;
use zelda::executor::buttons;
use zelda::game::GameView;
use zelda::{
    ActionKind, ActionMask, Change, LoadOptions, MASK_SIZE, Session, SessionSnapshot, Var,
    ZeldaError,
};

pub const FRAME_SHAPE: [usize; 3] = [240, 256, 3];
pub const FRAME_BYTES: usize = 240 * 256 * 3;
/// `Game.press` emulates this many frames (about a minute of play) between Ctrl-C checks.
const PRESS_CHUNK: u32 = 3600;

pub fn zelda_err(e: ZeldaError) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// RGB of `core`'s last rendered frame into `out` (`FRAME_BYTES`); false if it wasn't rendered.
pub fn rgb(core: &ZCore, out: &mut [u8]) -> bool {
    let out: &mut [u8; FRAME_BYTES] = out.try_into().expect("frame buffer size");
    core.rgb(out)
}

/// The last rendered frame as a (240, 256, 3) uint8 array.
pub fn screen<'py>(py: Python<'py>, core: &ZCore, render: bool) -> PyResult<Bound<'py, PyAny>> {
    screen_with(py, core, render, |_| {})
}

/// [`screen`], after `draw` has drawn over the frame's RGB bytes.
pub fn screen_with<'py>(
    py: Python<'py>,
    core: &ZCore,
    render: bool,
    draw: impl FnOnce(&mut [u8]),
) -> PyResult<Bound<'py, PyAny>> {
    if !render {
        return Err(RenderDisabledError::new_err(
            "rendering is off: set render = True and run a frame first",
        ));
    }
    let mut buf = vec![0u8; FRAME_BYTES];
    if !rgb(core, &mut buf) {
        return Err(RenderDisabledError::new_err(
            "no rendered frame yet: run a frame with rendering on",
        ));
    }
    draw(&mut buf);
    Ok(PyArray::from_vec(py, buf).reshape(FRAME_SHAPE)?.into_any())
}

pub fn mask_array<'py>(py: Python<'py>, mask: &ActionMask) -> Bound<'py, PyArray1<bool>> {
    PyArray::from_slice(py, &mask.to_flat())
}

/// A session snapshot (machine state, executor flags, total frames, steps). Picklable.
#[pyclass(module = "triforce._native", name = "Snapshot", frozen)]
pub struct PySnapshot {
    pub snap: SessionSnapshot<MachineState>,
}

impl PySnapshot {
    pub fn new(snap: SessionSnapshot<MachineState>) -> PySnapshot {
        PySnapshot { snap }
    }

    /// `u64 total_frames, u64 steps`, then the `TFZSTATE` file.
    fn encode(&self, py: Python<'_>) -> PyResult<Vec<u8>> {
        let rom = context::rom(py)?;
        let mut out = Vec::new();
        out.extend_from_slice(&self.snap.total_frames.to_le_bytes());
        out.extend_from_slice(&self.snap.steps.to_le_bytes());
        out.extend(state_file::encode(
            &rom,
            &self.snap.machine,
            context::flags(&self.snap),
        ));
        Ok(out)
    }
}

pub fn snapshot_from_file(file: state_file::StateFile) -> SessionSnapshot<MachineState> {
    SessionSnapshot {
        machine: file.machine,
        was_in_cave: file.flags & 1 != 0,
        just_exited_cave: file.flags & 2 != 0,
        total_frames: 0,
        steps: 0,
    }
}

#[pymethods]
impl PySnapshot {
    #[staticmethod]
    fn from_bytes(py: Python<'_>, data: &[u8]) -> PyResult<PySnapshot> {
        if data.len() < 16 {
            return Err(StateFormatError::new_err("snapshot bytes are truncated"));
        }
        let rom = context::rom(py)?;
        let file =
            state_file::decode(&rom, &data[16..]).map_err(|e| StateFormatError::new_err(e.0))?;
        let mut snap = snapshot_from_file(file);
        snap.total_frames = u64::from_le_bytes(data[0..8].try_into().unwrap());
        snap.steps = u64::from_le_bytes(data[8..16].try_into().unwrap());
        Ok(PySnapshot { snap })
    }

    /// A snapshot of a `TFZSTATE` file as it is on disk (no frames or steps counted): `Game.restore` of it is the
    /// raw machine state, without the load-time randomization or control gain of `Game.load`.
    #[staticmethod]
    fn from_state_file(py: Python<'_>, data: &[u8]) -> PyResult<PySnapshot> {
        let rom = context::rom(py)?;
        let file = state_file::decode(&rom, data).map_err(|e| StateFormatError::new_err(e.0))?;
        Ok(PySnapshot {
            snap: snapshot_from_file(file),
        })
    }

    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(py, &self.encode(py)?))
    }

    /// The machine state as a `TFZSTATE` file.
    fn state_file<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let rom = context::rom(py)?;
        Ok(PyBytes::new(
            py,
            &state_file::encode(&rom, &self.snap.machine, context::flags(&self.snap)),
        ))
    }

    #[getter]
    fn total_frames(&self) -> u64 {
        self.snap.total_frames
    }

    #[getter]
    fn steps(&self) -> u64 {
        self.snap.steps
    }

    #[getter]
    fn was_in_cave(&self) -> bool {
        self.snap.was_in_cave
    }

    #[getter]
    fn just_exited_cave(&self) -> bool {
        self.snap.just_exited_cave
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let from_bytes = py.get_type::<PySnapshot>().getattr("from_bytes")?;
        (from_bytes, (PyBytes::new(py, &self.encode(py)?),)).into_pyobject(py)
    }
}

fn direction(name: &str) -> PyResult<Direction> {
    Direction::from_name(name)
        .ok_or_else(|| PyValueError::new_err(format!("unknown direction {name:?}")))
}

fn tile_of(t: zelda::Tile) -> (i32, i32) {
    t
}

/// A decoded game state (immutable).
#[pyclass(module = "triforce._native", name = "GameState", frozen)]
pub struct PyGameState {
    pub view: Arc<GameView>,
}

impl PyGameState {
    pub fn new(view: Arc<GameView>) -> PyGameState {
        PyGameState { view }
    }
}

#[pymethods]
impl PyGameState {
    #[getter]
    fn frame(&self) -> u64 {
        self.view.frame
    }
    #[getter]
    fn level(&self) -> u8 {
        self.view.level()
    }
    #[getter]
    fn location(&self) -> u8 {
        self.view.location()
    }
    #[getter]
    fn in_cave(&self) -> bool {
        self.view.in_cave()
    }
    /// (level, room, in_cave).
    #[getter]
    fn full_location(&self) -> (u8, u8, bool) {
        room_of(self.view.full_location())
    }
    #[getter]
    fn mode(&self) -> u8 {
        self.view.mode()
    }
    #[getter]
    fn game_over(&self) -> bool {
        self.view.game_over()
    }
    #[getter]
    fn just_exited_cave(&self) -> bool {
        self.view.just_exited_cave
    }
    #[getter]
    fn health(&self) -> f64 {
        self.view.health()
    }
    #[getter]
    fn max_health(&self) -> i32 {
        self.view.max_health()
    }
    #[getter]
    fn keys(&self) -> i32 {
        self.view.keys()
    }
    #[getter]
    fn rupees(&self) -> i32 {
        self.view.rupees()
    }
    #[getter]
    fn bombs(&self) -> i32 {
        self.view.bombs()
    }
    #[getter]
    fn sword(&self) -> i32 {
        self.view.sword() as i32
    }
    #[getter]
    fn arrows(&self) -> i32 {
        self.view.arrows() as i32
    }
    #[getter]
    fn boomerang(&self) -> i32 {
        self.view.boomerang() as i32
    }
    #[getter]
    fn bow(&self) -> bool {
        self.view.bow()
    }
    #[getter]
    fn magic_key(&self) -> bool {
        self.view.magic_key()
    }
    #[getter]
    fn magic_shield(&self) -> bool {
        self.view.magic_shield()
    }
    #[getter]
    fn has_beams(&self) -> bool {
        self.view.has_beams()
    }
    #[getter]
    fn triforce_pieces(&self) -> u32 {
        self.view.triforce_pieces()
    }
    #[getter]
    fn room_kills(&self) -> i32 {
        self.view.room_kills()
    }

    /// {x, y, tile, direction, status}.
    #[getter]
    fn link<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let l = &self.view.link;
        let d = PyDict::new(py);
        d.set_item("x", l.position.x)?;
        d.set_item("y", l.position.y)?;
        d.set_item("tile", tile_of(l.tile()))?;
        d.set_item("direction", l.direction.name())?;
        d.set_item("status", l.status)?;
        Ok(d)
    }

    /// Enemies by distance from Link.
    #[getter]
    fn enemies<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for e in &self.view.enemies {
            let d = PyDict::new(py);
            d.set_item("index", e.index)?;
            d.set_item("id", e.id)?;
            d.set_item("name", describe::enemy_name(e.id))?;
            d.set_item("x", e.position.x)?;
            d.set_item("y", e.position.y)?;
            d.set_item("tile", tile_of(e.tile()))?;
            d.set_item("direction", e.direction.name())?;
            d.set_item("health", e.health)?;
            d.set_item("stun_timer", e.stun_timer)?;
            d.set_item("spawn_state", e.spawn_state)?;
            d.set_item("status", e.status)?;
            d.set_item("distance", e.distance)?;
            d.set_item("active", e.is_active())?;
            d.set_item("dying", e.is_dying())?;
            list.append(d)?;
        }
        Ok(list)
    }

    #[getter]
    fn items<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for i in &self.view.items {
            list.append(item_dict(py, i)?)?;
        }
        Ok(list)
    }

    #[getter]
    fn treasure<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        self.view.treasure().map(|t| item_dict(py, &t)).transpose()
    }

    #[getter]
    fn projectiles<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for p in &self.view.projectiles {
            let d = PyDict::new(py);
            d.set_item("index", p.index)?;
            d.set_item("id", p.id)?;
            d.set_item("x", p.position.x)?;
            d.set_item("y", p.position.y)?;
            d.set_item("tile", tile_of(p.tile()))?;
            d.set_item("direction", p.direction.name())?;
            d.set_item("distance", p.distance)?;
            d.set_item("blockable", p.blockable())?;
            list.append(d)?;
        }
        Ok(list)
    }

    /// RAM then PRG-RAM (10,240 bytes, the old `emu.ram` index layout).
    #[getter]
    fn ram<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, self.view.ram())
    }

    /// A byte by CPU address ($0000-$07FF, $6000-$7FFF).
    fn read(&self, addr: u16) -> PyResult<u8> {
        check_addr(addr)?;
        Ok(self.view.peek(addr))
    }

    /// A named RAM variable (`data.json` names), decoded (signed where declared).
    fn var(&self, name: &str) -> PyResult<i32> {
        Ok(self.view.var(var(name)?))
    }

    fn is_door_locked(&self, direction_name: &str) -> PyResult<bool> {
        Ok(self.view.is_door_locked(direction(direction_name)?))
    }

    fn is_door_barred(&self, direction_name: &str) -> PyResult<bool> {
        Ok(self.view.is_door_barred(direction(direction_name)?))
    }

    fn describe(&self) -> String {
        describe::describe_state(&self.view, None)
    }

    fn render_grid(&self) -> String {
        describe::render_grid(&self.view)
    }

    fn __repr__(&self) -> String {
        format!("GameState({:?})", self.view)
    }
}

fn item_dict<'py>(py: Python<'py>, i: &zelda::game::Item) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("index", i.index)?;
    d.set_item("id", i.id)?;
    d.set_item("name", describe::item_name(i.id))?;
    d.set_item("x", i.position.x)?;
    d.set_item("y", i.position.y)?;
    d.set_item("tile", tile_of(i.tile()))?;
    d.set_item("timer", i.timer)?;
    d.set_item("distance", i.distance)?;
    Ok(d)
}

fn check_addr(addr: u16) -> PyResult<()> {
    if addr < 0x800 || (0x6000..0x8000).contains(&addr) {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "address ${addr:04X} is outside $0000-$07FF/$6000-$7FFF"
        )))
    }
}

fn var(name: &str) -> PyResult<Var> {
    Var::from_name(name)
        .ok_or_else(|| PyValueError::new_err(format!("unknown RAM variable {name:?}")))
}

/// What changed over one action (immutable).
#[pyclass(module = "triforce._native", name = "Change", frozen)]
pub struct PyChange {
    pub change: Change,
}

#[pymethods]
impl PyChange {
    #[getter]
    fn action(&self) -> String {
        self.change.action.to_string()
    }
    #[getter]
    fn action_indices(&self) -> (usize, usize, usize) {
        action_indices(&self.change.action)
    }
    /// The frames that count against the action (the item sub-screen's left out).
    #[getter]
    fn frames(&self) -> u32 {
        self.change.frames
    }
    /// Frames on the item sub-screen choosing the B item (the game paused; not in `frames`).
    #[getter]
    fn menu_frames(&self) -> u32 {
        self.change.menu_frames
    }
    #[getter]
    fn interrupt(&self) -> Option<&'static str> {
        self.change.interrupt.map(|i| i.name())
    }
    #[getter]
    fn control_frames(&self) -> u32 {
        self.change.control_frames
    }
    /// Confirmed shield blocks during the option.
    #[getter]
    fn blocks(&self) -> usize {
        self.change.blocks()
    }
    /// The grid-point skip the option performed (`zelda::clips`): `{'direction', 'kind' ('wrap'/'block'/'ladder',
    /// None when unavailable), 'completed', 'abort' (None, 'unavailable', 'interrupted', 'missed', 'stalled',
    /// 'frame-cap'), 'presses'}`, or None.
    #[getter]
    fn clip<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(clip) = self.change.clip else {
            return Ok(None);
        };
        let d = PyDict::new(py);
        d.set_item("direction", clip.direction.name())?;
        d.set_item("kind", clip.kind.map(|k| k.name()))?;
        d.set_item("completed", clip.completed())?;
        d.set_item("abort", clip.abort.map(|a| a.name()))?;
        d.set_item("presses", clip.presses)?;
        Ok(Some(d))
    }
    /// This MOVE finished a skip an earlier CLIP of the room left cut short (round 22; `zelda::clips::OpenClip`).
    #[getter]
    fn clip_finished_by_move(&self) -> bool {
        self.change.clip_finished_by_move
    }
    #[getter]
    fn health_lost(&self) -> f64 {
        self.change.health_lost
    }
    #[getter]
    fn health_gained(&self) -> f64 {
        self.change.health_gained
    }
    /// (slot, damage) pairs, observed plus predicted.
    #[getter]
    fn enemies_hit(&self) -> Vec<(u8, i32)> {
        self.change.enemies_hit.clone()
    }
    #[getter]
    fn enemies_stunned(&self) -> Vec<u8> {
        self.change.enemies_stunned.clone()
    }
    #[getter]
    fn items_gained(&self) -> Vec<i16> {
        self.change.items_gained.clone()
    }
    #[getter]
    fn changed_location(&self) -> bool {
        self.change.changed_location()
    }
    #[getter]
    fn gained_triforce(&self) -> bool {
        self.change.gained_triforce()
    }
    #[getter]
    fn damage_dealt(&self) -> i32 {
        self.change.damage_dealt()
    }
    #[getter]
    fn hits(&self) -> usize {
        self.change.hits()
    }
    #[getter]
    fn previous(&self) -> PyGameState {
        PyGameState::new(self.change.previous.clone())
    }
    #[getter]
    fn state(&self) -> PyGameState {
        PyGameState::new(self.change.state.clone())
    }
    /// The new state's flat action mask.
    #[getter]
    fn action_mask<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        mask_array(py, &self.change.action_mask)
    }

    fn describe(&self) -> String {
        describe::describe_change(&self.change)
    }

    fn __repr__(&self) -> String {
        format!("Change({})", describe::describe_change(&self.change))
    }
}

/// Controller byte from `'UP+A'`, `['UP', 'A']`, or an int.
fn extract_buttons(obj: &Bound<'_, PyAny>) -> PyResult<u8> {
    if let Ok(b) = obj.extract::<u8>() {
        return Ok(b);
    }
    let names: Vec<String> = if let Ok(s) = obj.cast::<PyString>() {
        s.to_str()?
            .split('+')
            .map(str::trim)
            .filter(|n| !n.is_empty() && *n != "NONE" && *n != "-")
            .map(str::to_owned)
            .collect()
    } else {
        obj.extract()?
    };
    names.iter().try_fold(0u8, |acc, name| {
        let bit = match name.to_ascii_uppercase().as_str() {
            "A" => buttons::A,
            "B" => buttons::B,
            "SELECT" => buttons::SELECT,
            "START" => buttons::START,
            "UP" => buttons::UP,
            "DOWN" => buttons::DOWN,
            "LEFT" => buttons::LEFT,
            "RIGHT" => buttons::RIGHT,
            other => return Err(PyValueError::new_err(format!("unknown button {other:?}"))),
        };
        Ok(acc | bit)
    })
}

/// `movie::Playback` over a session's emulator; once `stop` is set (Ctrl-C) the remaining frames are skipped.
struct Player<'a>(&'a mut Session<ZCore>, bool, &'a AtomicBool);

impl movie::Playback for Player<'_> {
    fn run_frame(&mut self, buttons: u8) {
        if !self.2.load(Ordering::Relaxed) {
            zelda::Nes::run_frame(self.0.nes_mut(), buttons, self.1);
        }
    }

    fn poke(&mut self, addr: u16, value: u8) {
        zelda::Nes::poke(self.0.nes_mut(), addr, value);
    }
}

struct GameInner {
    session: Session<ZCore>,
    rng: ChaCha8Rng,
    store: TfzStore,
    render: bool,
    loaded: bool,
}

impl GameInner {
    fn state(&self) -> PyResult<Arc<GameView>> {
        if !self.loaded {
            return Err(PyRuntimeError::new_err(
                "call load(), press() or play_movie() first",
            ));
        }
        Ok(self.session.state().clone())
    }

    /// Rebuilds the state after direct emulator writes or frames.
    fn rebuild(&mut self) -> Arc<GameView> {
        self.loaded = true;
        self.session.apply_ram(&[]).clone()
    }
}

/// The machine state named by `state`: a state name or path, `TFZSTATE` bytes, or a `Snapshot`.
pub fn resolve_state(
    py: Python<'_>,
    store: &TfzStore,
    state: &Bound<'_, PyAny>,
) -> PyResult<SessionSnapshot<MachineState>> {
    if let Ok(s) = state.cast::<PySnapshot>() {
        return Ok(s.get().snap.clone());
    }
    if let Ok(b) = state.cast::<PyBytes>() {
        let rom = context::rom(py)?;
        let file =
            state_file::decode(&rom, b.as_bytes()).map_err(|e| StateFormatError::new_err(e.0))?;
        return Ok(snapshot_from_file(file));
    }
    let name: PathBuf = state.extract().map_err(|_| {
        PyTypeError::new_err("a state is a name, a path, TFZSTATE bytes or a Snapshot")
    })?;
    let file = store.read(&store.path(&name.to_string_lossy()))?;
    Ok(snapshot_from_file(file))
}

/// A RAM variable's value as its byte: 0..=255, or -128..=-1 as the two's-complement spelling of a signed one.
fn ram_byte(name: &str, value: i64) -> PyResult<u8> {
    u8::try_from(value)
        .or_else(|_| i8::try_from(value).map(|v| v as u8))
        .map_err(|_| PyValueError::new_err(format!("{name} = {value} does not fit a byte")))
}

fn overrides(values: Option<&Bound<'_, PyDict>>) -> PyResult<Vec<(Var, u8)>> {
    let Some(values) = values else {
        return Ok(Vec::new());
    };
    values
        .iter()
        .map(|(k, v)| {
            let name = k.extract::<String>()?;
            Ok((var(&name)?, ram_byte(&name, v.extract::<i64>()?)?))
        })
        .collect()
}

/// The saved idle states of a power-on start (`zelda::power_on`), made in one pass from power on: the machine
/// after every `POWER_ON_IDLE_STEP` idle frames on the file-select screen, as `(name, TFZSTATE bytes)` with
/// names `idle_02000` ... `idle_30000`.  `python -m triforce.entry_states idle` writes them under
/// `<states>/power_on/`, where a power-on run finds the one nearest below its delay.
#[pyfunction]
pub fn power_on_idle_states(py: Python<'_>) -> PyResult<Vec<(String, Bound<'_, PyBytes>)>> {
    use zelda::power_on::{IDLE_STATES, IDLE_STEP, idle, to_file_select};
    let rom = context::rom(py)?;
    let mut nes = ZCore::power_on(rom.clone());
    to_file_select(&mut nes).map_err(zelda_err)?;
    let mut states = Vec::new();
    for index in 1..=IDLE_STATES {
        idle(&mut nes, IDLE_STEP);
        let bytes = state_file::encode(&rom, &nes.snapshot(), 0);
        states.push((format!("idle_{:05}", index * IDLE_STEP), PyBytes::new(py, &bytes)));
    }
    Ok(states)
}

/// The name a power-on start registers for `name`: upper-cased.  ValueError if it can't be registered (1 to 8
/// of A-Z, 0-9 and space, not all spaces) or starts the second quest (ZELDA).
#[pyfunction]
pub fn power_on_name(name: &str) -> PyResult<String> {
    zelda::power_on::name_cells(name).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(name.to_ascii_uppercase())
}

/// One game session: load states, act, poke, render, snapshot.
#[pyclass(module = "triforce._native", name = "Game", frozen)]
pub struct PyGame {
    inner: Mutex<GameInner>,
}

#[pymethods]
impl PyGame {
    /// A powered-on machine. `seed` seeds load randomization (a natural RNG state, spawn cycle, edge cell);
    /// `states_dir` overrides where bare state names resolve.
    #[new]
    #[pyo3(signature = (seed=0, render=false, states_dir=None, look_ahead=true))]
    fn new(
        py: Python<'_>,
        seed: u64,
        render: bool,
        states_dir: Option<PathBuf>,
        look_ahead: bool,
    ) -> PyResult<Self> {
        let rom = context::rom(py)?;
        let data = context::data(py, states_dir.as_deref())?;
        let mut session = Session::new(ZCore::power_on(rom.clone()), data.map.clone());
        session.set_render(render);
        session.look_ahead = if look_ahead { zelda::LookAhead::All } else { zelda::LookAhead::Off };
        Ok(PyGame {
            inner: Mutex::new(GameInner {
                session,
                rng: ChaCha8Rng::seed_from_u64(seed),
                store: TfzStore::new(&data, rom),
                render,
                loaded: false,
            }),
        })
    }

    /// Loads a state (name, path, TFZSTATE bytes or Snapshot), applies randomization and overrides, and gains
    /// control of Link (unless `controllable`: the state already is an action boundary).
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (state, randomize_rng=true, fresh_room=false, controllable=false, ram_overrides=None, seed=None))]
    fn load(
        &self,
        py: Python<'_>,
        state: &Bound<'_, PyAny>,
        randomize_rng: bool,
        fresh_room: bool,
        controllable: bool,
        ram_overrides: Option<&Bound<'_, PyDict>>,
        seed: Option<u64>,
    ) -> PyResult<PyGameState> {
        let mut g = self.inner.lock();
        let snap = resolve_state(py, &g.store, state)?;
        let opts = LoadOptions {
            randomize_rng,
            fresh_room,
            controllable,
            overrides: overrides(ram_overrides)?,
        };
        if let Some(seed) = seed {
            g.rng = ChaCha8Rng::seed_from_u64(seed);
        }
        let GameInner { session, rng, .. } = &mut *g;
        let view = session
            .load(&snap.machine, &opts, rng)
            .map_err(zelda_err)?
            .clone();
        g.loaded = true;
        Ok(PyGameState::new(view))
    }

    /// Boots the game from console power on by controller inputs alone (`zelda::power_on`): the file-select
    /// screen, `delay` idle frames there (0..=32766: the game RNG's phase at the start), `name` registered in
    /// slot 1 (1 to 8 of A-Z, 0-9 and space; not ZELDA), the game started, and control of Link gained.  No
    /// memory is written.  The machine must be at power on (a fresh `Game`).
    #[pyo3(signature = (name="LINK", delay=0))]
    fn boot(&self, name: &str, delay: u32) -> PyResult<PyGameState> {
        let mut g = self.inner.lock();
        if g.loaded || g.session.nes().frame() != 0 {
            return Err(PyValueError::new_err(
                "boot() needs a machine at power on: this one has run frames or loaded a state",
            ));
        }
        let GameInner { session, rng, .. } = &mut *g;
        zelda::power_on::boot(session.nes_mut(), name, delay).map_err(|e| match e {
            ZeldaError::Boot(_) => PyValueError::new_err(e.to_string()),
            e => zelda_err(e),
        })?;
        let live = session.nes().snapshot();
        let view = session
            .load(&live, &LoadOptions::default(), rng)
            .map_err(zelda_err)?
            .clone();
        g.loaded = true;
        Ok(PyGameState::new(view))
    }

    /// Executes one option (`'MOVE N 16'`, head indices or an `Action`) and returns what changed.
    fn act(&self, action: &Bound<'_, PyAny>) -> PyResult<PyChange> {
        let action = extract_action(action)?;
        let mut g = self.inner.lock();
        g.state()?;
        let change = g.session.act(action).map_err(zelda_err)?;
        Ok(PyChange { change })
    }

    /// Writes named RAM variables at the current frame (no emulation) and rebuilds the state and mask.
    fn apply_ram(&self, values: &Bound<'_, PyDict>) -> PyResult<PyGameState> {
        let values = overrides(Some(values))?;
        let mut g = self.inner.lock();
        g.loaded = true;
        Ok(PyGameState::new(g.session.apply_ram(&values).clone()))
    }

    fn set_value(&self, name: &str, value: i64) -> PyResult<PyGameState> {
        let values = [(var(name)?, ram_byte(name, value)?)];
        let mut g = self.inner.lock();
        g.loaded = true;
        Ok(PyGameState::new(g.session.apply_ram(&values).clone()))
    }

    /// A byte by CPU address ($0000-$07FF, $6000-$7FFF).
    fn read(&self, addr: u16) -> PyResult<u8> {
        check_addr(addr)?;
        Ok(zelda::Nes::peek(self.inner.lock().session.nes(), addr))
    }

    /// Writes a byte ($0000-$07FF, $6000-$7FFF) and rebuilds the state.
    fn write(&self, addr: u16, value: u8) -> PyResult<()> {
        check_addr(addr)?;
        let mut g = self.inner.lock();
        zelda::Nes::poke(g.session.nes_mut(), addr, value);
        g.rebuild();
        Ok(())
    }

    /// A copy of RAM then PRG-RAM (10,240 bytes, the old `emu.ram` layout).
    fn ram<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        let g = self.inner.lock();
        let nes = g.session.nes();
        let mut out = Vec::with_capacity(0x2800);
        out.extend_from_slice(zelda::Nes::ram(nes));
        out.extend_from_slice(zelda::Nes::prg_ram(nes));
        PyArray::from_vec(py, out)
    }

    /// Holds raw buttons (`'UP+A'`, `['UP', 'A']` or a controller byte) for `frames` frames, bypassing the
    /// option executor (GIL released; Ctrl-C stops it between chunks of `PRESS_CHUNK` frames).
    #[pyo3(signature = (buttons, frames=1))]
    fn press(&self, py: Python<'_>, buttons: &Bound<'_, PyAny>, frames: u32) -> PyResult<PyGameState> {
        let b = extract_buttons(buttons)?;
        let mut remaining = frames;
        loop {
            let n = remaining.min(PRESS_CHUNK);
            let state = py.detach(|| {
                let mut g = self.inner.lock();
                g.loaded = true;
                g.session.press(b, n).clone()
            });
            remaining -= n;
            if remaining == 0 {
                return Ok(PyGameState::new(state));
            }
            py.check_signals()?;
        }
    }

    /// Plays a `TFMOVIE` (its `base` movies first, from the same directory) on the current machine: from
    /// power-on for corpus movies. Returns the state at the end.
    fn play_movie(&self, py: Python<'_>, path: PathBuf) -> PyResult<PyGameState> {
        let m = movie::Movie::load(&path).map_err(|e| PyValueError::new_err(format!("{e:#}")))?;
        let rom = context::rom(py)?;
        if m.rom_sha1 != rom.payload_sha1() {
            return Err(PyValueError::new_err(format!(
                "{} was recorded with another ROM",
                path.display()
            )));
        }
        let dir = path.parent().map(PathBuf::from).unwrap_or_default();
        let stop = AtomicBool::new(false);
        let played = interruptible(
            py,
            || {
                let mut g = self.inner.lock();
                let render = g.render;
                movie::play(&mut Player(&mut g.session, render, &stop), &m, &dir)
                    .map_err(|e| format!("{e:#}"))?;
                Ok::<_, String>(g.rebuild())
            },
            || stop.store(true, Ordering::Relaxed),
        )?;
        Ok(PyGameState::new(played.map_err(PyValueError::new_err)?))
    }

    fn snapshot(&self) -> PySnapshot {
        PySnapshot::new(self.inner.lock().session.snapshot())
    }

    /// Resumes a snapshot without stepping.
    fn restore(&self, snapshot: &Bound<'_, PySnapshot>) -> PyGameState {
        let mut g = self.inner.lock();
        g.loaded = true;
        PyGameState::new(g.session.restore(&snapshot.get().snap).clone())
    }

    /// The current machine as a `TFZSTATE` file.
    fn save_state<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let g = self.inner.lock();
        PyBytes::new(py, &g.store.encode(&g.session.snapshot()))
    }

    /// Writes the current machine as a `TFZSTATE` file (atomically).
    fn save_state_file(&self, path: PathBuf) -> PyResult<()> {
        let g = self.inner.lock();
        let bytes = g.store.encode(&g.session.snapshot());
        g.store
            .write(&path, &bytes)
            .map_err(|e| pyo3::exceptions::PyOSError::new_err(format!("{}: {e}", path.display())))
    }

    #[getter]
    fn state(&self) -> PyResult<PyGameState> {
        Ok(PyGameState::new(self.inner.lock().state()?))
    }

    /// The flat action mask of the current state.
    #[getter]
    fn mask<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        mask_array(py, self.inner.lock().session.mask())
    }

    /// The legal actions as text (`MOVE[N,S] WAIT ...`).
    #[getter]
    fn legal(&self) -> String {
        self.inner.lock().session.mask().to_string()
    }

    fn is_allowed(&self, action: &Bound<'_, PyAny>) -> PyResult<bool> {
        let action = extract_action(action)?;
        Ok(self.inner.lock().session.mask().is_allowed(&action))
    }

    /// Action kinds removed from every mask (names, e.g. `['BOMB']`).
    fn set_disabled_kinds(&self, kinds: Vec<String>) -> PyResult<()> {
        let kinds = kinds
            .iter()
            .map(|k| {
                ActionKind::from_name(k)
                    .ok_or_else(|| PyValueError::new_err(format!("unknown action kind {k:?}")))
            })
            .collect::<PyResult<Vec<_>>>()?;
        self.inner.lock().session.set_disabled_kinds(&kinds);
        Ok(())
    }

    /// Offer the CLIP kind where a grid-point skip is available (`zelda::clips`; a specialist's `clips: true`) and the
    /// run-time switch allows it ([`set_clips_allowed`], default on). A `CLIP <dir>` action runs the skip whatever
    /// this says (the mask only offers it when on).
    fn set_clips(&self, enabled: bool) {
        self.inner.lock().session.set_clips(enabled);
    }

    /// The run-time clips switch (round 22; default on): off masks CLIP whatever `set_clips` says, on the mask this
    /// call recomputes.
    fn set_clips_allowed(&self, allowed: bool) {
        self.inner.lock().session.set_clips_allowed(allowed);
    }

    #[getter]
    fn clips_allowed(&self) -> bool {
        self.inner.lock().session.clips_allowed()
    }

    /// Whether the grid-point skip is available in `direction` now (`zelda::clips::skip_available`).
    fn skip_available(&self, direction: &str) -> PyResult<bool> {
        let direction = Direction::from_name(direction)
            .ok_or_else(|| PyValueError::new_err(format!("unknown direction {direction:?}")))?;
        let state = self.inner.lock().state()?;
        Ok(zelda::clips::skip_available(&state, direction))
    }

    /// Render emulated frames (from the next one on).
    #[getter]
    fn get_render(&self) -> bool {
        self.inner.lock().render
    }

    #[setter]
    fn set_render(&self, on: bool) {
        let mut g = self.inner.lock();
        g.render = on;
        g.session.set_render(on);
    }

    /// The last rendered frame as (240, 256, 3) uint8 RGB; `RenderDisabledError` when rendering is off.
    fn screen<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let g = self.inner.lock();
        screen(py, g.session.nes(), g.render)
    }

    #[getter]
    fn get_capture_pre_entry(&self) -> bool {
        self.inner.lock().session.capture_pre_entry()
    }

    #[setter]
    fn set_capture_pre_entry(&self, on: bool) {
        self.inner.lock().session.set_capture_pre_entry(on);
    }

    /// The snapshot of the last frame before the most recent room entry (with `capture_pre_entry`).
    #[getter]
    fn last_pre_entry(&self) -> Option<PySnapshot> {
        self.inner
            .lock()
            .session
            .last_pre_entry()
            .cloned()
            .map(PySnapshot::new)
    }

    /// Real frames emulated by this game (look-ahead excluded).
    #[getter]
    fn frame(&self) -> u64 {
        self.inner.lock().session.frame()
    }

    /// Frames since the last load.
    #[getter]
    fn total_frames(&self) -> u64 {
        self.inner.lock().session.total_frames
    }

    /// Actions since the last load.
    #[getter]
    fn steps(&self) -> u64 {
        self.inner.lock().session.steps
    }

    /// Every fired weapon is simulated forward (`zelda::LookAhead::All`); False: none.
    #[getter]
    fn get_look_ahead(&self) -> bool {
        self.inner.lock().session.look_ahead == zelda::LookAhead::All
    }

    #[setter]
    fn set_look_ahead(&self, on: bool) {
        self.inner.lock().session.look_ahead = if on { zelda::LookAhead::All } else { zelda::LookAhead::Off };
    }

    /// Location, Link, inventory, room, entities and legal actions as text.
    fn describe_state(&self) -> PyResult<String> {
        let g = self.inner.lock();
        let state = g.state()?;
        Ok(describe::describe_state(&state, Some(g.session.mask())))
    }

    #[staticmethod]
    fn describe_change(change: &Bound<'_, PyChange>) -> String {
        describe::describe_change(&change.get().change)
    }

    /// ASCII map of the room's tile grid.
    fn render_grid(&self) -> PyResult<String> {
        let state = self.inner.lock().state()?;
        Ok(describe::render_grid(&state))
    }

    /// The masked-out width of the flat mask (`MASK_SIZE`).
    #[classattr]
    const MASK_SIZE: usize = MASK_SIZE;
}
