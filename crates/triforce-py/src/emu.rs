//! `LiveGame`: a game a person plays frame by frame (`triforce.emu`, `emu-zelda.py`).  A start is loaded as a
//! chain run loads one (a natural RNG phase drawn from the seed, the overrides written), and from that load on
//! every frame's buttons and every memory write go to a TFREPLAY recording (`PoolNes`, as a pool env records), so
//! what was played replays bit for bit and `tf replay verify` passes.  Each frame is rendered; with sound on its
//! samples go to the default sound device (`theater::Speaker`), as `play-zelda.py`'s heard cell does.

use crate::context::{self, TfzStore};
use crate::game::{PyGameState, overrides, screen, zelda_err};
use env::StateStore;
use parking_lot::Mutex;
use pool::EnvControl;
use pool::zelda::{PoolNes, PoolStore};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use replay::{Context, ENDINGS, Ending, Event, RecordOptions, Source};
use std::path::PathBuf;
use std::sync::Arc;
use theater::Speaker;
use zcore::ZCore;
use zcore::tracks::{Mix, Tracks};
use zelda::{LoadOptions, Session, power_on};

struct Inner {
    session: Session<PoolNes<ZCore>>,
    store: PoolStore<TfzStore>,
    loaded: bool,
    seed: u64,
    /// Round 43, the reset rule of the route's doors (None: no route, or a route without an `entry_rule` file).
    entry_rule: Option<Arc<zelda::room_entry::EntryRule>>,
    /// The sound device, opened the first time sound is turned on (None: never, or it failed: `sound_error`).
    speaker: Option<Speaker>,
    sound_error: Option<String>,
    /// Sound asked for ([`PyLiveGame::set_sound`]) and the speed the frames are played at against the
    /// console's: each frame makes `1 / speed` times its samples, so the sound runs on without gaps, lower or
    /// higher, as a tape played slow or fast.
    hear: bool,
    speed: f64,
    /// The sound of the frames being played while sound is on.
    tracks: Option<Tracks>,
    audio: Vec<f32>,
}

impl Inner {
    fn core(&self) -> &ZCore {
        self.session.nes().inner()
    }

    fn view(&self) -> PyResult<PyGameState> {
        if !self.loaded {
            return Err(PyRuntimeError::new_err("call load() first"));
        }
        Ok(PyGameState::new(self.session.state().clone()))
    }

    /// Sound picks up from the machine as it stands (after a load, or turned on): no samples of before.
    fn restart_sound(&mut self) {
        let speaker = self.speaker.as_ref().filter(|_| self.hear);
        self.session.nes_mut().inner_mut().set_sound(speaker.is_some());
        self.audio.clear();
        self.tracks = speaker.map(|speaker| {
            Tracks::new(self.session.nes().inner(), f64::from(speaker.ring().rate()), Mix::ALL, false, false)
        });
    }

    fn step(&mut self, buttons: u8) {
        let Inner { session, tracks, speaker, audio, speed, .. } = self;
        if let (Some(tracks), Some(speaker)) = (tracks.as_mut(), speaker.as_ref()) {
            tracks.set_rate(speaker.ring().feed_rate() / *speed);
            tracks.prepare(session.nes().inner());
        }
        session.press(buttons, 1);
        if let (Some(tracks), Some(speaker)) = (tracks.as_mut(), speaker.as_ref()) {
            tracks.run(session.nes().inner(), audio);
            speaker.ring().push(audio);
            audio.clear();
        }
    }
}

/// A game played by hand and recorded: `load` a start, `step(buttons)` one frame at a time, `screen()` its
/// picture, `replay(...)` the recording so far as TFREPLAY bytes (the recording goes on).  `set_sound(True)`
/// plays each frame's sound on the default device from the next frame on.
#[pyclass(module = "triforce._native", name = "LiveGame", frozen)]
pub struct PyLiveGame {
    inner: Mutex<Inner>,
}

#[pymethods]
impl PyLiveGame {
    /// `route`: the rooms as that route plays them (game.yaml; None: the rooms every route plays alike);
    /// `states_dir` overrides where bare state names resolve.
    #[new]
    #[pyo3(signature = (route=None, states_dir=None))]
    fn new(py: Python<'_>, route: Option<&str>, states_dir: Option<PathBuf>) -> PyResult<Self> {
        let rom = context::rom(py)?;
        let data = context::data(py, states_dir.as_deref())?;
        let map = match route {
            Some(route) if !data.map.routes.contains_key(route) => {
                let known: Vec<&String> = data.map.routes.keys().collect();
                return Err(crate::errors::ConfigError::new_err(format!(
                    "Unknown route '{route}' (routes.yaml has {known:?})"
                )));
            }
            Some(route) => data.map.route_map(route),
            None => data.map.clone(),
        };
        let control = EnvControl::standalone(0, 60.0, crate::game::FRAME_BYTES);
        control.set_render(true);
        let mut session = Session::new(PoolNes::new(ZCore::power_on(rom.clone()), Some(control)), map);
        session.set_render(true);
        session.look_ahead = zelda::LookAhead::Off;
        let entry_rule = route.and_then(|route| data.entry_rule(data.map.namespace(route)));
        Ok(PyLiveGame {
            inner: Mutex::new(Inner {
                session,
                store: PoolStore(TfzStore::new(&data, rom)),
                loaded: false,
                seed: 0,
                entry_rule,
                speaker: None,
                sound_error: None,
                hear: false,
                speed: 1.0,
                tracks: None,
                audio: Vec::new(),
            }),
        })
    }

    /// Loads `state` (a state name or path, embedded in the recording under its name; or `TFZSTATE` bytes,
    /// embedded) and starts a new recording from it: the natural RNG phase drawn from `seed` and `ram_overrides`
    /// (`Game.set_value` names) are its frame-0 edits, the frames until Link is controllable its first inputs.
    /// `banked_kills` ((level, room), kills): the kills the game keeps for the room about to be entered
    /// (`LoadOptions::banked_kills`, as a staged training reset banks them), so it makes that many monsters fewer.
    /// `layout` (round 43, the reset rule of the route's doors): 'route' (default) the layout the route's plan
    /// brings to the door, 'valid' any layout some play of the game can bring there (the route's own on half the
    /// seeds, as training draws them); the chase state, `$4A`, the divider and the edge cell as the rule draws them,
    /// all from `seed`.  The previous recording is dropped.
    #[pyo3(signature = (state, seed, ram_overrides=None, banked_kills=Vec::new(), layout="route"))]
    fn load(
        &self,
        py: Python<'_>,
        state: &Bound<'_, PyAny>,
        seed: u64,
        ram_overrides: Option<&Bound<'_, PyDict>>,
        banked_kills: Vec<((u8, u8), u8)>,
        layout: &str,
    ) -> PyResult<PyGameState> {
        let layout = match layout {
            "route" => zelda::room_entry::Layout::Route,
            "valid" => zelda::room_entry::Layout::Valid,
            other => return Err(PyValueError::new_err(format!("layout {other:?}: 'route' or 'valid'"))),
        };
        let opts = LoadOptions {
            randomize_rng: true,
            fresh_room: false,
            controllable: false,
            overrides: overrides(ram_overrides)?,
            banked_kills,
            entry: self.inner.lock().entry_rule.clone().map(|rule| zelda::room_entry::EntryDraw::new(rule, layout)),
        };
        let (name, bytes) = if let Ok(b) = state.cast::<PyBytes>() {
            (None, b.as_bytes().to_vec())
        } else if let Ok(name) = state.extract::<PathBuf>() {
            (Some(name.to_string_lossy().into_owned()), Vec::new())
        } else {
            return Err(PyTypeError::new_err("a state is a name, a path or TFZSTATE bytes"));
        };
        let mut g = self.inner.lock();
        let g = &mut *g;
        let machine = match &name {
            Some(name) => g.store.load(name),
            None => g.store.decode(&bytes).map(|s| s.machine),
        }
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let nes = g.session.nes_mut();
        nes.take_recorder();
        nes.set_recording(Some(Arc::new(RecordOptions {
            embed_start: true,
            rewards: false,
            keyframe_frames: 0,
            meta: serde_json::Value::Null,
        })));
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        py.detach(|| g.session.load(&machine, &opts, &mut rng).map(|_| ())).map_err(zelda_err)?;
        let phase = g.session.rng_phase();
        let start = match &name {
            Some(name) => Some(g.store.bytes(name).map_err(|e| PyValueError::new_err(e.to_string()))?),
            None => None,
        };
        let rec = g.session.nes_mut().recorder_mut().expect("a load from a store state records");
        rec.set_rng_phase(phase);
        if let (Some(name), Some(start)) = (&name, &start) {
            rec.embed_start(Some(name), Context::default(), start);
        }
        rec.event(Event::Control);
        g.loaded = true;
        g.seed = seed;
        g.restart_sound();
        g.view()
    }

    /// Starts a new recording at console power on (TFREPLAY source `power-on`, every frame an input): the title
    /// screen, the file-select screen, `name` registered in slot 1 with no idle and the game started, to its first
    /// gameplay frame in overworld 0x77 (`zelda::power_on::to_file_select`, `register_and_start`; unrendered:
    /// the next `step` draws the picture).  `name`: 1 to 8 of A-Z, 0-9 and space (`power_on::board_cells`),
    /// `ZELDA` too (the second quest).  The previous recording is dropped; `hand_over()` marks where the player
    /// takes over.
    fn power_on(&self, py: Python<'_>, name: &str) -> PyResult<PyGameState> {
        power_on::board_cells(name).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let name = name.to_ascii_uppercase();
        let mut g = self.inner.lock();
        let g = &mut *g;
        let machine = g.store.power_on().map_err(|e| PyValueError::new_err(e.to_string()))?;
        let nes = g.session.nes_mut();
        nes.take_recorder();
        nes.set_recording(Some(Arc::new(RecordOptions {
            embed_start: false,
            rewards: false,
            keyframe_frames: 0,
            meta: serde_json::Value::Null,
        })));
        zelda::Nes::restore(nes, &machine);
        py.detach(|| {
            power_on::to_file_select(nes)?;
            power_on::register_and_start(nes, &name)
        })
        .map_err(zelda_err)?;
        nes.recorder_mut()
            .expect("a power-on restore records")
            .set_source(Source::PowerOn { name: name.clone(), delay: 0 });
        let live = zelda::Nes::snapshot(g.session.nes());
        let opts = LoadOptions { controllable: true, ..LoadOptions::default() };
        g.session.load(&live, &opts, &mut ChaCha8Rng::seed_from_u64(0)).map_err(zelda_err)?;
        g.loaded = true;
        g.seed = 0;
        g.restart_sound();
        g.view()
    }

    /// Presses the console's RESET button before the next frame (`zcore::ZCore::reset`: CPU RAM and the battery
    /// RAM kept, the game's own reset code runs from the next frame), recorded as a RESET event.
    fn reset_console(&self) -> PyResult<PyGameState> {
        let mut g = self.inner.lock();
        if !g.loaded {
            return Err(PyRuntimeError::new_err("call load() or power_on() first"));
        }
        let nes = g.session.nes_mut();
        nes.inner_mut().reset();
        if let Some(rec) = nes.recorder_mut() {
            rec.event(Event::Reset);
        }
        g.session.apply_ram(&[]);
        g.view()
    }

    /// Records the CONTROL event at this frame: the player takes over here (`power_on` leaves it to the caller,
    /// who may run a scripted stretch first).
    fn hand_over(&self) -> PyResult<()> {
        let mut g = self.inner.lock();
        if !g.loaded {
            return Err(PyRuntimeError::new_err("call load() or power_on() first"));
        }
        if let Some(rec) = g.session.nes_mut().recorder_mut() {
            rec.event(Event::Control);
        }
        Ok(())
    }

    /// The machine as it stands as a `TFZSTATE` file (with the session's flags), as `Game.load` and
    /// `LiveGame.load` read it.
    fn state_file<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let g = self.inner.lock();
        if !g.loaded {
            return Err(PyRuntimeError::new_err("call load() or power_on() first"));
        }
        let bytes = g.store.encode(&g.session.snapshot()).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// Runs one frame with the controller byte `buttons` (A 0x01, B 0x02, SELECT 0x04, START 0x08, UP 0x10, DOWN
    /// 0x20, LEFT 0x40, RIGHT 0x80), recorded, rendered and heard.
    fn step(&self, py: Python<'_>, buttons: u8) -> PyResult<PyGameState> {
        let mut g = self.inner.lock();
        if !g.loaded {
            return Err(PyRuntimeError::new_err("call load() first"));
        }
        let inner = &mut *g;
        py.detach(|| inner.step(buttons));
        g.view()
    }

    #[getter]
    fn state(&self) -> PyResult<PyGameState> {
        self.inner.lock().view()
    }

    /// The last frame as a (240, 256, 3) uint8 RGB array.
    fn screen<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let g = self.inner.lock();
        screen(py, g.core(), true)
    }

    /// Writes a byte ($0000-$07FF, $6000-$7FFF) before the next frame, recorded as an EDIT (a scripted kill, a
    /// forced hit in a test): the take still verifies.
    fn poke(&self, addr: u16, value: u8) -> PyResult<()> {
        if !matches!(addr, 0x0000..0x0800 | 0x6000..0x8000) {
            return Err(PyValueError::new_err(format!("${addr:04X} is outside $0000-$07FF/$6000-$7FFF")));
        }
        let mut g = self.inner.lock();
        if !g.loaded {
            return Err(PyRuntimeError::new_err("call load() first"));
        }
        zelda::Nes::poke(g.session.nes_mut(), addr, value);
        g.session.apply_ram(&[]);
        Ok(())
    }

    /// Input frames recorded since the load (the load's own frames to control included).
    #[getter]
    fn frames(&self) -> u32 {
        self.inner.lock().session.nes().recorder().map_or(0, |rec| rec.frames())
    }

    /// The natural RNG phase the load wrote (META `rng_phase`).
    #[getter]
    fn rng_phase(&self) -> Option<u32> {
        self.inner.lock().session.rng_phase()
    }

    /// The recording so far as TFREPLAY bytes, ending in the machine as it stands: `meta` the META object (JSON
    /// text) beside what the recorder writes, `ending` one of `ENDINGS` with its `result` and `reason` texts.
    /// The recording goes on.
    #[pyo3(signature = (meta, ending, result="", reason=""))]
    fn replay<'py>(
        &self,
        py: Python<'py>,
        meta: &str,
        ending: &str,
        result: &str,
        reason: &str,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let meta: serde_json::Value =
            serde_json::from_str(meta).map_err(|e| PyValueError::new_err(format!("meta: {e}")))?;
        let kind = ENDINGS
            .iter()
            .position(|&e| e == ending)
            .ok_or_else(|| PyValueError::new_err(format!("unknown ending {ending:?}; endings: {ENDINGS:?}")))?;
        let g = self.inner.lock();
        let Some(rec) = g.session.nes().recorder() else {
            return Err(PyRuntimeError::new_err("nothing is recorded: call load() first"));
        };
        let mut rec = rec.clone();
        if let serde_json::Value::Object(meta) = meta {
            for (key, value) in meta {
                rec.set_meta(&key, value);
            }
        }
        let end = g.store.encode(&g.session.snapshot()).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ending = Ending {
            kind: u8::try_from(kind).expect("few endings"),
            result: result.to_owned(),
            outcome: String::new(),
            reason: reason.to_owned(),
        };
        let seed = i64::try_from(g.seed).unwrap_or(i64::MAX);
        Ok(PyBytes::new(py, &rec.finish(seed, ending, Context::default(), &end)))
    }

    /// Plays each frame's sound on the default sound device from the next frame on (on), or stops it (off);
    /// `speed`: the frames' rate against the console's (each frame then makes `1 / speed` times its samples).
    /// Turned on, the sound takes the machine's as it stands: nothing made earlier is played late.  A call that
    /// only changes the speed goes on without a break.  Returns why there is no sound (no device), else None.
    #[pyo3(signature = (on, speed=1.0))]
    fn set_sound(&self, on: bool, speed: f64) -> PyResult<Option<String>> {
        if !(speed.is_finite() && speed > 0.0) {
            return Err(PyValueError::new_err(format!("speed {speed}: a positive number")));
        }
        let mut g = self.inner.lock();
        if on && g.speaker.is_none() && g.sound_error.is_none() {
            match Speaker::open() {
                Ok(speaker) => g.speaker = Some(speaker),
                Err(error) => g.sound_error = Some(error),
            }
        }
        g.speed = speed;
        if on != g.hear {
            g.hear = on;
            g.restart_sound();
        }
        Ok(g.sound_error.clone())
    }

    /// Whether frames are heard now.
    #[getter]
    fn sound(&self) -> bool {
        self.inner.lock().tracks.is_some()
    }
}
