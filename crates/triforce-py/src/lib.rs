//! `triforce._native`: the Rust game, env and pool stack as a Python extension (plan step 15).
//!
//! The emulator is `zcore::ZCore`; the ROM is `$TRIFORCE_ROM`, else `<repo>/roms/zelda.nes` (the repo being the
//! directory holding the `triforce` package), validated once per process. Start states are `TFZSTATE` files
//! under the states directory (`$TRIFORCE_STATES_DIR`, else `triforce/states`; every constructor also takes
//! `states_dir=`).

mod config;
mod context;
mod convert;
mod describe;
mod envpool;
mod envs;
mod errors;
mod explore;
mod game;
mod policy;
mod replay;
mod signals;
mod theater;

use pyo3::prelude::*;

/// The route's room visits as dicts (`index`, `level`, `room`, `in_cave`, `entry`, `exit`, `specialist`,
/// `waypoint`, `retreat`, `treasure`: None to take the room's treasure, `skip`, `clear` or `if_bombs_below: N`,
/// `bomb_reserve`: the step's routes.yaml `bomb_reserve` or None).
#[pyfunction]
#[pyo3(signature = (route=zelda::game_map::DEFAULT_ROUTE))]
fn route_steps<'py>(
    py: Python<'py>,
    route: &str,
) -> PyResult<Vec<Bound<'py, pyo3::types::PyDict>>> {
    let data = context::data(py, None)?;
    let steps = data.map.routes.get(route).ok_or_else(|| {
        let known: Vec<&String> = data.map.routes.keys().collect();
        errors::ConfigError::new_err(format!(
            "Unknown route '{route}' (routes.yaml has {known:?})"
        ))
    })?;
    steps
        .iter()
        .enumerate()
        .map(|(i, s)| step_dict(py, i, s))
        .collect()
}

fn step_dict<'py>(
    py: Python<'py>,
    index: usize,
    s: &zelda::game_map::RouteStep,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let d = pyo3::types::PyDict::new(py);
    d.set_item("index", index)?;
    d.set_item("level", s.level)?;
    d.set_item("room", s.room)?;
    d.set_item("in_cave", s.in_cave)?;
    d.set_item("entry", &s.entry)?;
    d.set_item("exit", &s.exit)?;
    let specialist = s.specialist.clone().or_else(|| {
        let data = context::data(py, None).ok()?;
        data.specialist_for(s.location()).map(str::to_owned)
    });
    d.set_item("specialist", specialist)?;
    d.set_item("waypoint", s.waypoint.as_deref())?;
    d.set_item("retreat", s.retreat.as_deref())?;
    d.set_item("treasure", s.treasure.describe())?;
    d.set_item("bomb_reserve", s.bomb_reserve)?;
    Ok(d)
}

/// routes.yaml `alternates` as dicts: `replaces` [(level, room, in_cave)] and `with` [step dicts, `index`
/// counting within the alternate].
#[pyfunction]
fn route_alternates<'py>(py: Python<'py>) -> PyResult<Vec<Bound<'py, pyo3::types::PyDict>>> {
    let data = context::data(py, None)?;
    data.map
        .alternates
        .iter()
        .map(|a| {
            let d = pyo3::types::PyDict::new(py);
            d.set_item("replaces", envpool::rooms(&a.replaces))?;
            let with = a
                .with
                .iter()
                .enumerate()
                .map(|(i, s)| step_dict(py, i, s))
                .collect::<PyResult<Vec<_>>>()?;
            d.set_item("with", with)?;
            Ok(d)
        })
        .collect()
}

/// Route names (routes.yaml order).
#[pyfunction]
fn routes(py: Python<'_>) -> PyResult<Vec<String>> {
    Ok(context::data(py, None)?
        .map
        .routes
        .keys()
        .cloned()
        .collect())
}

/// routes.yaml `families`: specialist -> [(level, room, in_cave)].
#[pyfunction]
fn families(py: Python<'_>) -> PyResult<Bound<'_, pyo3::types::PyDict>> {
    let data = context::data(py, None)?;
    let d = pyo3::types::PyDict::new(py);
    for (name, rooms) in &data.families {
        d.set_item(name, envpool::rooms(rooms))?;
    }
    Ok(d)
}

/// game.yaml rooms: `(level, room, in_cave)` -> `{'enemies': {name: count}, 'name': str or None (what people call
/// the room: "Breakfast Room"), 'treasure': str or None, 'requires_bombs': bool, 'cave': (level, room, in_cave) or
/// None}`.  `cave` is where the room's cave exit leads: the room's own interior on the overworld, a separate room
/// id for a dungeon cellar (1:0x22 -> 1:0x7f).
#[pyfunction]
fn game_rooms(py: Python<'_>) -> PyResult<Bound<'_, pyo3::types::PyDict>> {
    let data = context::data(py, None)?;
    let d = pyo3::types::PyDict::new(py);
    for (location, room) in data.map.rooms() {
        let info = pyo3::types::PyDict::new(py);
        let enemies = pyo3::types::PyDict::new(py);
        for (name, count) in &room.enemies {
            enemies.set_item(name, count)?;
        }
        info.set_item("enemies", enemies)?;
        info.set_item("name", room.name.as_deref())?;
        info.set_item("treasure", room.treasure.as_deref())?;
        info.set_item("requires_bombs", room.requires_bombs)?;
        let cave = room.exits.get("cave").map(|exit| {
            let to = &exit.destination;
            (to.level, to.value, to.in_cave)
        });
        info.set_item("cave", cave)?;
        d.set_item((location.level, location.value, location.in_cave), info)?;
    }
    Ok(d)
}

/// The specialist game.yaml tags on a room (`(level, room[, in_cave])` or `'(1, 0x44)'`), or None.
#[pyfunction]
fn specialist_for(py: Python<'_>, location: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    let serde_json::Value::String(text) = config::room_value(location)? else {
        unreachable!("rooms format as strings")
    };
    let location = env::config::parse_room(&text).map_err(|e| errors::ConfigError::new_err(e.0))?;
    Ok(context::data(py, None)?
        .specialist_for(location)
        .map(str::to_owned))
}

/// The repo root, ROM path and default states directory this module uses.
#[pyfunction]
fn paths(py: Python<'_>) -> PyResult<Bound<'_, pyo3::types::PyDict>> {
    let root = context::root(py)?;
    let data = context::data(py, None)?;
    let d = pyo3::types::PyDict::new(py);
    d.set_item("root", &root)?;
    d.set_item("rom", context::rom_path(&root))?;
    d.set_item("states_dir", &data.states_dir)?;
    d.set_item("trajectories_dir", &data.trajectories_dir)?;
    Ok(d)
}

#[pymodule]
mod _native {
    use super::*;
    use pyo3::types::PyDict;

    #[pymodule_export]
    use super::{families, game_rooms, paths, route_alternates, route_steps, routes, specialist_for};
    #[pymodule_export]
    use crate::config::PySpecialistConfig;
    #[pymodule_export]
    use crate::envpool::{PyChainTask, PyEnvPool, PyEvalTask, PyTrainTask};
    #[pymodule_export]
    use crate::envs::{PyCurriculumSpec, PyEnv};
    #[pymodule_export]
    use crate::errors::{
        ConfigError, EpisodeFinishedError, InvalidActionError, RenderDisabledError, ReplayError, RomError,
        StartStateError, StateFormatError, UnclassifiedEndingError,
    };
    #[pymodule_export]
    use crate::explore::{PyAcceptSpec, PyExploreResult, PyExplorer, explore_states};
    #[pymodule_export]
    use crate::game::{PyChange, PyGame, PyGameState, PySnapshot, power_on_idle_states, power_on_name};
    #[pymodule_export]
    use crate::policy::{PyPolicy, keyed_noise};
    #[pymodule_export]
    use crate::replay::{PyReplay, PyReplayPlayer, replay_stitch, replay_truncate, replay_verify};
    #[pymodule_export]
    use crate::theater::{PyReplayFeed, PyReplayScreen};

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        use env::observation as obs;
        let py = m.py();
        m.add("VERSION", include_str!("../../../VERSION").trim())?;
        m.add("MAX_POWER_ON_DELAY", zelda::power_on::MAX_DELAY)?;
        m.add("POWER_ON_IDLE_STEP", zelda::power_on::IDLE_STEP)?;
        m.add("OBS_SCHEMA_VERSION", obs::OBS_SCHEMA_VERSION)?;
        m.add(
            "ACTION_SCHEMA_VERSION",
            zelda::actions::ACTION_SCHEMA_VERSION,
        )?;
        m.add("MASK_SIZE", zelda::MASK_SIZE)?;
        m.add("DEFAULT_ROUTE", zelda::game_map::DEFAULT_ROUTE)?;
        m.add("FRAME_SHAPE", (240usize, 256usize, 3usize))?;
        let spec = PyDict::new(py);
        for k in obs::OBS_SPEC {
            let d = PyDict::new(py);
            d.set_item("shape", pyo3::types::PyTuple::new(py, k.shape)?)?;
            d.set_item("vocab", k.vocab)?;
            d.set_item("dtype", k.dtype)?;
            d.set_item("low", k.low)?;
            d.set_item("high", k.high)?;
            spec.set_item(k.key, d)?;
        }
        m.add("OBS_SPEC", spec)?;
        let names = obs::feature_names();
        m.add(
            "ROOM_STATE_FEATURES",
            pyo3::types::PyTuple::new(py, &names.room_state)?,
        )?;
        m.add("LINK_FEATURES", pyo3::types::PyTuple::new(py, &names.link)?)?;
        m.add(
            "ENTITY_FEATURES",
            pyo3::types::PyTuple::new(py, &names.entity)?,
        )?;
        m.add(
            "HISTORY_FEATURES",
            pyo3::types::PyTuple::new(py, &names.history)?,
        )?;
        m.add(
            "OBJECTIVE_FEATURES",
            pyo3::types::PyTuple::new(py, &names.objective)?,
        )?;
        m.add(
            "TARGET_FEATURES",
            pyo3::types::PyTuple::new(py, &names.target)?,
        )?;
        m.add("NUM_ENTITY_SLOTS", obs::NUM_ENTITY_SLOTS)?;
        m.add("ROOM_VOCAB", obs::ROOM_VOCAB)?;
        m.add("ENTITY_VOCAB", obs::ENTITY_VOCAB)?;
        m.add("NUM_TILE_CLASSES", zelda::tiles::NUM_TILE_CLASSES)?;
        m.add(
            "INTERRUPTS",
            pyo3::types::PyTuple::new(py, ::pool::zelda::INTERRUPT_NAMES)?,
        )?;
        m.add(
            "REWARD_TERMS",
            pyo3::types::PyTuple::new(py, env::rewards::REWARD_TERMS)?,
        )?;
        m.add(
            "OUTCOMES",
            pyo3::types::PyTuple::new(py, env::rewards::OUTCOMES)?,
        )?;
        m.add(
            "CHEAT_ITEMS",
            pyo3::types::PyTuple::new(py, zelda::cheats::CheatItem::ALL.iter().map(|i| i.name()))?,
        )?;
        m.add(
            "BOMB_STATS",
            pyo3::types::PyTuple::new(py, zelda::bombs::BombStats::default().fields().map(|(name, _)| name))?,
        )?;
        m.add(
            "ACTION_KINDS",
            pyo3::types::PyTuple::new(py, zelda::ActionKind::ALL.iter().map(|k| k.name()))?,
        )?;
        m.add(
            "DIRECTIONS",
            pyo3::types::PyTuple::new(py, zelda::actions::DIRECTIONS.iter().map(|d| d.name()))?,
        )?;
        m.add(
            "DURATIONS",
            pyo3::types::PyTuple::new(py, zelda::actions::DURATIONS)?,
        )?;
        // The kinds each head applies to (zelda::actions::{is_timed, is_directional, is_diagonal}).
        for (name, applies) in [
            ("TIMED_KINDS", zelda::actions::is_timed as fn(zelda::ActionKind) -> bool),
            ("DIRECTIONAL_KINDS", zelda::actions::is_directional),
            ("DIAGONAL_KINDS", zelda::actions::is_diagonal),
        ] {
            let kinds: Vec<&str> = zelda::ActionKind::ALL
                .iter()
                .filter(|&&k| applies(k))
                .map(|k| k.name())
                .collect();
            m.add(name, pyo3::types::PyTuple::new(py, kinds)?)?;
        }
        Ok(())
    }
}
