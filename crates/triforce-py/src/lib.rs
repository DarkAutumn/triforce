//! `triforce._native`: the Rust game, env and pool stack as a Python extension (plan step 15).
//!
//! The emulator is `zcore::ZCore`; the ROM is `$TRIFORCE_ROM`, else the shared store's `rom/zelda.nes`, else
//! `<repo>/roms/zelda.nes` (the repo being the directory holding the `triforce` package; the `paths` crate),
//! validated once per process. Start states are `TFZSTATE` files
//! under the states directory (`$TRIFORCE_STATES_DIR`, else `triforce/states`; every constructor also takes
//! `states_dir=`).

mod config;
mod context;
mod convert;
mod describe;
mod emu;
mod envpool;
mod envs;
mod errors;
mod explore;
mod external;
mod game;
mod policy;
mod replay;
mod signals;
mod theater;

use pyo3::prelude::*;

/// The route's room visits as dicts (`index`, `level`, `room`, `in_cave`, `entry`, `exit`, `specialist`,
/// `waypoint`, `retreat`, `treasure`: None to take the room's treasure, `skip`, `refuse`, `clear` or `if_bombs_below: N`,
/// `bomb_reserve`: the step's routes.yaml `bomb_reserve` or None, `way`: 'clips' or 'plain' for a step of one way of
/// a `stretch:` (None: both ways'), `stretch`: its name, `visit`: the route's visit to the room from that side,
/// `zelda::game_map::route_visits`, which a later visit's state names spell, `carriers`: the step's `carriers:`,
/// `goals`: what the step expects of a visit, `zelda::goals::step_goals`'s names). Both ways' steps are listed,
/// numbered in file order.
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
    let visits = &data.map.visits[route];
    let map = data.map.route_map(route);
    steps
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let d = step_dict(py, route, i, s, visits[i])?;
            d.set_item("goals", zelda::goals::step_goals(&map, route, i).names())?;
            Ok(d)
        })
        .collect()
}

fn step_dict<'py>(
    py: Python<'py>,
    route: &str,
    index: usize,
    s: &zelda::game_map::RouteStep,
    visit: u32,
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
        data.route_specialist_for(route, s.location()).map(str::to_owned)
    });
    d.set_item("specialist", specialist)?;
    d.set_item("waypoint", s.waypoint.as_deref())?;
    d.set_item("retreat", s.retreat.as_deref())?;
    d.set_item("treasure", s.treasure.describe())?;
    d.set_item("bomb_reserve", s.bomb_reserve)?;
    d.set_item("bombs", s.bombs.as_deref())?;
    d.set_item("drops", s.drops_first.then_some("first"))?;
    d.set_item("rupee_reserve", s.rupee_reserve)?;
    d.set_item("banked_kills", s.banked_kills)?;
    d.set_item("sword", s.sword)?;
    d.set_item("carriers", s.carriers)?;
    d.set_item("carrier_fight", s.carrier_fight)?;
    // Round 44: `clear:` as written: True, False, a kill count, or None (a route without it).
    match s.clear {
        Some(zelda::game_map::StepClear::All) => d.set_item("clear", true)?,
        Some(zelda::game_map::StepClear::None) => d.set_item("clear", false)?,
        Some(zelda::game_map::StepClear::Kills(k)) => d.set_item("clear", k)?,
        None => d.set_item("clear", py.None())?,
    }
    d.set_item("way", s.way.map(|w| w.name()))?;
    d.set_item("stretch", s.stretch.as_deref())?;
    d.set_item("visit", visit)?;
    Ok(d)
}

/// The route's `stretch:` blocks (round 22) as dicts: `name`, `clips` and `plain` (each way's step indices) and
/// `rejoin` (the shared step both lead into; the ways part at the first step of each).
#[pyfunction]
#[pyo3(signature = (route=zelda::game_map::DEFAULT_ROUTE))]
fn route_stretches<'py>(py: Python<'py>, route: &str) -> PyResult<Vec<Bound<'py, pyo3::types::PyDict>>> {
    let data = context::data(py, None)?;
    let stretches = data.map.stretches.get(route).ok_or_else(|| {
        errors::ConfigError::new_err(format!("Unknown route '{route}'"))
    })?;
    stretches
        .iter()
        .map(|s| {
            let d = pyo3::types::PyDict::new(py);
            d.set_item("name", &s.name)?;
            d.set_item("clips", s.clips.clone().collect::<Vec<_>>())?;
            d.set_item("plain", s.plain.clone().collect::<Vec<_>>())?;
            d.set_item("rejoin", s.rejoin)?;
            Ok(d)
        })
        .collect()
}

/// routes.yaml `alternates` as dicts: `replaces` [(level, room, in_cave)] and `with` [step dicts, `index`
/// counting within the alternate, `visit` as if it stood at the default route's step of its replaced room].
#[pyfunction]
fn route_alternates<'py>(py: Python<'py>) -> PyResult<Vec<Bound<'py, pyo3::types::PyDict>>> {
    let data = context::data(py, None)?;
    let steps = &data.map.routes[zelda::game_map::DEFAULT_ROUTE];
    let stretches = &data.map.stretches[zelda::game_map::DEFAULT_ROUTE];
    data.map
        .alternates
        .iter()
        .map(|a| {
            let d = pyo3::types::PyDict::new(py);
            d.set_item("replaces", envpool::rooms(&a.replaces))?;
            let at = a.replaces.first().and_then(|r| {
                steps.iter().position(|s| (s.level, s.room) == (r.level, r.value))
            });
            let with = a
                .with
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    let at = at.map(|at| (at + i).min(steps.len()));
                    let visit = at.map_or(1, |at| zelda::game_map::visit_at(steps, stretches, at, s));
                    step_dict(py, zelda::game_map::DEFAULT_ROUTE, i, s, visit)
                })
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
/// None, 'exits': {key: {'to': (level, room, in_cave), 'locked', 'bomb_wall', 'shutter', 'raft', 'wrap'}},
/// 'trigger': str or None, 'dark': bool}`.  `cave` is where the room's cave exit leads: the room's own interior on
/// the overworld, a separate room id for a dungeon cellar (1:0x22 -> 1:0x7f).  `exits` are the room's game.yaml
/// exits by key ('N', 'S', 'E', 'W', 'cave'; a maze's 'maze') with each door's kind.  `route`: the rooms as that route
/// plays them (its own game.yaml rooms over the others, phase X); None: the rooms every route plays alike.
#[pyfunction]
#[pyo3(signature = (route=None))]
fn game_rooms<'py>(py: Python<'py>, route: Option<&str>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let data = context::data(py, None)?;
    let map = match route {
        Some(route) => known_route(&data, route)?.map.for_route(route),
        None => &data.map,
    };
    let d = pyo3::types::PyDict::new(py);
    for (location, room) in map.rooms() {
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
        let exits = pyo3::types::PyDict::new(py);
        for (key, exit) in &room.exits {
            let to = &exit.destination;
            let e = pyo3::types::PyDict::new(py);
            e.set_item("to", (to.level, to.value, to.in_cave))?;
            e.set_item("locked", exit.locked)?;
            e.set_item("bomb_wall", exit.bomb_wall)?;
            e.set_item("shutter", exit.shutter)?;
            e.set_item("raft", exit.raft)?;
            e.set_item("wrap", exit.wrap)?;
            exits.set_item(key, e)?;
        }
        info.set_item("exits", exits)?;
        info.set_item("trigger", room.trigger.as_deref())?;
        info.set_item("dark", room.dark)?;
        d.set_item((location.level, location.value, location.in_cave), info)?;
    }
    Ok(d)
}

/// The specialist game.yaml tags on a room (`(level, room[, in_cave])` or `'(1, 0x44)'`), or None; `route`: as that
/// route plays the room (its own game.yaml rooms over the others, phase X).
#[pyfunction]
#[pyo3(signature = (location, route=None))]
fn specialist_for(py: Python<'_>, location: &Bound<'_, PyAny>, route: Option<&str>) -> PyResult<Option<String>> {
    let serde_json::Value::String(text) = config::room_value(location)? else {
        unreachable!("rooms format as strings")
    };
    let location = env::config::parse_room(&text).map_err(|e| errors::ConfigError::new_err(e.0))?;
    let data = context::data(py, None)?;
    Ok(match route {
        Some(route) => known_route(&data, route)?.route_specialist_for(route, location),
        None => data.specialist_for(location),
    }
    .map(str::to_owned))
}

/// `data` if routes.yaml has `route`, else the error naming the routes it has.
fn known_route<'a>(data: &'a env::config::GameData, route: &str) -> PyResult<&'a env::config::GameData> {
    if data.map.routes.contains_key(route) {
        return Ok(data);
    }
    let known: Vec<&String> = data.map.routes.keys().collect();
    Err(errors::ConfigError::new_err(format!("Unknown route '{route}' (routes.yaml has {known:?})")))
}

/// What a start state's name or path says (`env::start_states::StateName`, docs/architecture.md "Start states"):
/// `(level, room, in_cave, entry, visit)` of `[<ns>_]<level>_<room><side>[_v<visit>]_<chain>` (harvested
/// `..._h<seed>_<n>` alike; `entry` the routes.yaml entry N/S/E/W or 'cave', `visit` 1 for the bare name), else None.
/// A route's state namespace before it (phase X) is read past: `state_namespace` says which.
#[pyfunction]
fn state_name(name: &str) -> Option<(u8, u8, bool, &'static str, u32)> {
    env::start_states::StateName::parse(name)
        .map(|n| (n.location.level, n.location.value, n.location.in_cave, n.entry, n.visit))
}

/// The route state namespace a start state's name or path spells (`d3first_3_7cs_0` -> 'd3first'), None for a bare
/// name (`zelda::game_map::state_namespace`).
#[pyfunction]
fn state_namespace(name: &str) -> Option<&str> {
    zelda::game_map::state_namespace(name)
}

/// The start-state namespace of `route` (routes.yaml `states:`, phase X), None for bare names (`d2first`).
#[pyfunction]
fn route_namespace(py: Python<'_>, route: &str) -> PyResult<Option<String>> {
    let data = context::data(py, None)?;
    Ok(known_route(&data, route)?.map.namespace(route).map(str::to_owned))
}

/// The index of the step up to which `route`'s own runs play (routes.yaml `own_models_until`, round 32), None for
/// a route whose own runs play everywhere (`GameMap::own_models_until`).
#[pyfunction]
fn own_models_until(py: Python<'_>, route: &str) -> PyResult<Option<usize>> {
    let data = context::data(py, None)?;
    Ok(known_route(&data, route)?.map.own_models_until.get(route).copied())
}

/// The policy key of `route`'s step `step` played by `family` (`GameMap::step_policy`, round 32): `family@route`
/// before the route's `own_models_until`, else `family`.
#[pyfunction]
fn step_policy(py: Python<'_>, route: &str, step: usize, family: &str) -> PyResult<String> {
    let data = context::data(py, None)?;
    Ok(known_route(&data, route)?.map.step_policy(route, step, family))
}

/// A start state's name on `route` (`GameMap::route_state`): a bare name in the route's namespace, a name already in
/// it, `start` and paths as given; ValueError for a name in another route's namespace.
#[pyfunction]
fn route_state(py: Python<'_>, route: &str, name: &str) -> PyResult<String> {
    let data = context::data(py, None)?;
    data.map.route_state(route, name).map_err(pyo3::exceptions::PyValueError::new_err)
}

/// The repo root, ROM path (None when no ROM is found) and default states directory this module uses.
#[pyfunction]
fn paths(py: Python<'_>) -> PyResult<Bound<'_, pyo3::types::PyDict>> {
    let root = context::root(py)?;
    let data = context::data(py, None)?;
    let d = pyo3::types::PyDict::new(py);
    d.set_item("root", &root)?;
    d.set_item("rom", context::rom_path(&root).ok())?;
    d.set_item("states_dir", &data.states_dir)?;
    d.set_item("trajectories_dir", &data.trajectories_dir)?;
    Ok(d)
}

/// Whether the start state `name` (a name or path, harvested `..._h<seed>_<n>` alike), whose machine holds `state`, is
/// a state `route` can be in (`zelda::objectives::start_on_route`, round 42: Link's triforce pieces and items the
/// route's on entering a step of the room the name spells; the test a training reset applies to a harvested start).
/// False for a name that isn't an entry state's.
#[pyfunction]
fn start_on_route(py: Python<'_>, route: &str, name: &str, state: &game::PyGameState) -> PyResult<bool> {
    let data = context::data(py, None)?;
    let data = known_route(&data, route)?;
    Ok(env::start_states::StateName::parse(name).is_some_and(|n| {
        zelda::objectives::start_on_route(&data.map, n.location, n.entry, n.visit, route, &state.view)
    }))
}

#[pymodule]
mod _native {
    use super::*;
    use pyo3::types::PyDict;

    #[pymodule_export]
    use super::{
        families, game_rooms, own_models_until, paths, route_alternates, route_namespace, route_state, route_steps,
        route_stretches, routes, specialist_for, start_on_route, state_name, state_namespace, step_policy,
    };
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
    use crate::external::{PyExternalPolicy, PyExternalQueue};
    #[pymodule_export]
    use crate::emu::PyLiveGame;
    #[pymodule_export]
    use crate::game::{
        PyChange, PyGame, PyGameState, PySnapshot, chase_blocks, chase_probe, danger_blocks, exposed_probe,
        hearts_encoding, observe, power_on_idle_states, power_on_name, lane_probe, shooter_blocks, shooter_probe,
        threat_blocks,
    };
    #[pymodule_export]
    use crate::policy::{PyPolicy, act_mode, act_probabilities, keyed_noise};
    #[pymodule_export]
    use crate::replay::{
        PyReplay, PyReplayPlayer, replay_bomb_windows, replay_stitch, replay_truncate, replay_verify,
    };
    #[pymodule_export]
    use crate::theater::{PyReplayFeed, PyReplayScreen};

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        use env::observation as obs;
        let py = m.py();
        m.add("VERSION", include_str!("../../../VERSION").trim())?;
        m.add("REPLAY_VERSION", ::replay::REPLAY_VERSION)?;
        m.add("MAX_POWER_ON_DELAY", zelda::power_on::MAX_DELAY)?;
        m.add("MIN_HARVESTED_ENTRY", env::start_states::MIN_HARVESTED_ENTRY)?;
        m.add("POWER_ON_IDLE_STEP", zelda::power_on::IDLE_STEP)?;
        m.add("OBS_SCHEMA_VERSION", obs::OBS_SCHEMA_VERSION)?;
        m.add("OLDEST_OBS_SCHEMA", obs::OLDEST_OBS_SCHEMA)?;
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
            "ENTITY_REL_FEATURES",
            pyo3::types::PyTuple::new(py, &names.entity_rel)?,
        )?;
        m.add(
            "ENTITY_CONTACT_FEATURES",
            pyo3::types::PyTuple::new(py, &names.entity_contact)?,
        )?;
        m.add(
            "ENTITY_THREAT_FEATURES",
            pyo3::types::PyTuple::new(py, &names.entity_threat)?,
        )?;
        m.add(
            "CHASE_FEATURES",
            pyo3::types::PyTuple::new(py, &names.chase)?,
        )?;
        m.add(
            "ENTITY_CHASE_FEATURES",
            pyo3::types::PyTuple::new(py, &names.entity_chase)?,
        )?;
        m.add(
            "ENTITY_SHOOTER_FEATURES",
            pyo3::types::PyTuple::new(py, &names.entity_shooter)?,
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
            "CLIP_STATS",
            pyo3::types::PyTuple::new(py, zelda::clips::ClipStats::default().fields().map(|(name, _)| name))?,
        )?;
        m.add(
            "AIMLESS_FIELDS",
            pyo3::types::PyTuple::new(
                py,
                zelda::aimless::AimlessSummary::default().fields().map(|(name, _)| name),
            )?,
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
