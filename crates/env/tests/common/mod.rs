//! Shared test helpers: zcore over the ROM, the named-state loader (`TFZSTATE` files in `triforce/states`) and
//! the observation-space check.
#![allow(dead_code)]

use env::observation::{OBS_SPEC, Observation};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock};
use zcore::Rom;
pub use zcore::{MachineState, ZCore};
use zelda::ActionMask;

pub fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The ROM (`$TRIFORCE_ROM`, else `roms/zelda.nes`). Missing: panics (`test_rom`), or None with
/// `TRIFORCE_ALLOW_NO_ROM=1`.
pub fn rom() -> Option<Arc<Rom>> {
    test_rom::path()?; // before the LazyLock, so every test fails with the ROM message, not a poisoned lock
    static ROM: LazyLock<Option<Arc<Rom>>> =
        LazyLock::new(|| test_rom::bytes().map(|bytes| Rom::from_ines(&bytes).expect("valid Zelda ROM")));
    ROM.clone()
}

/// A powered-on zcore, or None when the ROM is missing.
pub fn open_core() -> Option<ZCore> {
    rom().map(ZCore::power_on)
}

/// In-memory `StateStore` (tests): `load` finds a state by name or by the label of a saved path; `save` keeps
/// the machine state under the path.
#[derive(Default)]
pub struct MemStore(pub std::cell::RefCell<HashMap<String, MachineState>>);

impl env::StateStore<MachineState> for MemStore {
    fn load(&self, name: &str) -> Result<MachineState, env::EnvError> {
        let states = self.0.borrow();
        states
            .get(name)
            .or_else(|| {
                let label = env::start_states::state_label(name);
                states
                    .iter()
                    .find(|(k, _)| env::start_states::state_label(k) == label)
                    .map(|(_, v)| v)
            })
            .cloned()
            .ok_or_else(|| env::EnvError::State(format!("no state {name}")))
    }

    fn save(
        &self,
        path: &Path,
        snapshot: &zelda::SessionSnapshot<MachineState>,
    ) -> Result<(), env::EnvError> {
        std::fs::write(path, b"mem")?;
        self.0
            .borrow_mut()
            .insert(path.display().to_string(), snapshot.machine);
        Ok(())
    }
}

/// Directory of the named start states: the regenerated `TFZSTATE` states, `triforce/states`.
pub fn states_dir() -> PathBuf {
    repo().join("triforce/states")
}

/// The repo's game data (its states directory is [`states_dir`]).
pub fn game_data() -> Arc<env::config::GameData> {
    env::config::GameData::load(&repo()).expect("game data")
}

/// File of a named state, resolved like `Game.load` (bare name -> `<states dir>/<name>.state`; an existing or
/// absolute path is used as is).
pub fn state_path(name: &str) -> PathBuf {
    let p = Path::new(name);
    if p.is_absolute() || p.exists() {
        return p.to_path_buf();
    }
    let file = if name.ends_with(".state") {
        name.to_owned()
    } else {
        format!("{name}.state")
    };
    states_dir().join(file)
}

/// A `TFZSTATE` file's bytes as a session snapshot (executor flags from the file, no frames or steps).
pub fn decode_state(bytes: &[u8]) -> Result<zelda::SessionSnapshot<MachineState>, env::EnvError> {
    let rom = rom().ok_or_else(|| env::EnvError::State("ROM not found".into()))?;
    let file = zcore::state_file::decode(&rom, bytes).map_err(|e| env::EnvError::State(e.to_string()))?;
    Ok(zelda::SessionSnapshot {
        machine: file.machine,
        was_in_cave: file.flags & 1 != 0,
        just_exited_cave: file.flags & 2 != 0,
        total_frames: 0,
        steps: 0,
    })
}

/// A session snapshot as a `TFZSTATE` file's bytes.
pub fn encode_state(snapshot: &zelda::SessionSnapshot<MachineState>) -> Result<Vec<u8>, env::EnvError> {
    let rom = rom().ok_or_else(|| env::EnvError::State("ROM not found".into()))?;
    let flags = snapshot.was_in_cave as u8 | (snapshot.just_exited_cave as u8) << 1;
    Ok(zcore::state_file::encode(&rom, &snapshot.machine, flags))
}

/// A decoded `TFZSTATE` file as a session snapshot.
pub fn read_state_file(path: &Path) -> Result<zelda::SessionSnapshot<MachineState>, env::EnvError> {
    let err = |e: &dyn std::fmt::Display| env::EnvError::State(format!("{}: {e}", path.display()));
    decode_state(&std::fs::read(path).map_err(|e| err(&e))?).map_err(|e| err(&e))
}

/// THE state loader of the env tests: every test that starts from a named state (directly or through
/// `TfzStore`) reads `triforce/states/<name>.state` here.
pub fn load_named_state(name: &str) -> Result<MachineState, env::EnvError> {
    read_state_file(&state_path(name)).map(|s| s.machine)
}

/// Every named state in the states directory, sorted.
pub fn named_states() -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(states_dir())
        .expect("states dir")
        .filter_map(|e| {
            let name = e.unwrap().file_name().into_string().unwrap();
            name.strip_suffix(".state").map(str::to_owned)
        })
        .collect();
    names.sort();
    names
}

/// `StateStore` over the named states (`load_named_state`, like the native `TfzStore`); `save` writes a
/// `TFZSTATE` file atomically.
#[derive(Default)]
pub struct TfzStore;

impl TfzStore {
    pub fn new() -> TfzStore {
        TfzStore
    }
}

impl env::StateStore<MachineState> for TfzStore {
    fn load(&self, name: &str) -> Result<MachineState, env::EnvError> {
        load_named_state(name)
    }

    fn save(
        &self,
        path: &Path,
        snapshot: &zelda::SessionSnapshot<MachineState>,
    ) -> Result<(), env::EnvError> {
        let bytes = encode_state(snapshot)?;
        let tmp = PathBuf::from(format!("{}.{}.tmp", path.display(), std::process::id()));
        std::fs::write(&tmp, bytes)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    fn bytes(&self, name: &str) -> Result<Vec<u8>, env::EnvError> {
        let path = state_path(name);
        std::fs::read(&path).map_err(|e| env::EnvError::State(format!("{}: {e}", path.display())))
    }

    fn encode(&self, snapshot: &zelda::SessionSnapshot<MachineState>) -> Result<Vec<u8>, env::EnvError> {
        encode_state(snapshot)
    }

    fn decode(&self, bytes: &[u8]) -> Result<zelda::SessionSnapshot<MachineState>, env::EnvError> {
        decode_state(bytes)
    }

    fn power_on(&self) -> Result<MachineState, env::EnvError> {
        open_core()
            .map(|core| core.snapshot())
            .ok_or_else(|| env::EnvError::State("ROM not found".into()))
    }
}

/// (key, flattened values) of an observation.
pub fn obs_values(o: &Observation) -> Vec<(&'static str, Vec<f64>)> {
    let f = |v: &[f32]| v.iter().map(|&x| x as f64).collect::<Vec<_>>();
    vec![
        ("room_id", vec![o.room_id as f64]),
        ("room_state", f(&o.room_state)),
        ("link", f(&o.link)),
        (
            "entity_types",
            o.entity_types.iter().map(|&x| x as f64).collect(),
        ),
        (
            "entity_features",
            o.entity_features.iter().flat_map(|r| f(r)).collect(),
        ),
        ("grid", o.grid.iter().flatten().map(|&x| x as f64).collect()),
        ("history", f(&o.history)),
        ("objective", f(&o.objective)),
        ("target", f(&o.target)),
        (
            "action_mask",
            o.action_mask.iter().map(|&x| x as u8 as f64).collect(),
        ),
    ]
}

/// The OBS_SPEC keys whose shape, bounds or integer vocabulary `o` violates (`not space[k].contains(v)`).
pub fn obs_violations(o: &Observation) -> Vec<String> {
    let values = obs_values(o);
    assert_eq!(values.len(), OBS_SPEC.len());
    let mut bad = Vec::new();
    for spec in OBS_SPEC {
        let (_, v) = values
            .iter()
            .find(|(k, _)| *k == spec.key)
            .expect("obs key");
        let n: usize = spec.shape.iter().product();
        if v.len() != n {
            bad.push(format!("{} shape {}", spec.key, v.len()));
            continue;
        }
        for &x in v {
            if !(x.is_finite() && spec.low <= x && x <= spec.high) {
                bad.push(format!(
                    "{} value {x} outside [{}, {}]",
                    spec.key, spec.low, spec.high
                ));
                break;
            }
            if let Some(vocab) = spec.vocab
                && !(x.fract() == 0.0 && (x as usize) < vocab)
            {
                bad.push(format!("{} id {x}", spec.key));
                break;
            }
        }
    }
    bad
}

/// `observation_space.contains(obs)` for every key; the mask is the session's.
pub fn assert_obs_valid(o: &Observation, mask: &ActionMask) {
    let bad = obs_violations(o);
    assert!(bad.is_empty(), "{bad:?}");
    assert_eq!(
        o.action_mask,
        mask.to_flat(),
        "obs mask is the session mask"
    );
}
