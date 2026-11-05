//! Shared test helpers: zcore over the ROM, the named-state loader (`TFZSTATE` files in `triforce/states`), an
//! `.npz` reader, the parity fixtures (replayed from their corpus movies) and the observation-space check.
#![allow(dead_code)]

use env::observation::{OBS_SPEC, Observation};
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock};
use zcore::Rom;
pub use zcore::{MachineState, ZCore};
use zelda::ActionMask;

pub fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The ROM (`$TRIFORCE_ROM`, else `roms/zelda.nes`), or None (with the reason on stderr) when it is missing.
pub fn rom() -> Option<Arc<Rom>> {
    static ROM: LazyLock<Option<Arc<Rom>>> = LazyLock::new(|| {
        let path = std::env::var_os("TRIFORCE_ROM")
            .map(PathBuf::from)
            .unwrap_or_else(|| repo().join("roms/zelda.nes"));
        let bytes = std::fs::read(path).ok()?;
        Some(Rom::from_ines(&bytes).expect("valid Zelda ROM"))
    });
    if ROM.is_none() {
        eprintln!("ROM not found: set TRIFORCE_ROM or create roms/zelda.nes");
    }
    ROM.clone()
}

/// A powered-on zcore, or None when the ROM is missing.
pub fn open_core() -> Option<ZCore> {
    rom().map(ZCore::power_on)
}

/// One array of an `.npz`.
#[derive(Clone, Debug)]
pub struct Npy {
    pub shape: Vec<usize>,
    pub data: Data,
}

#[derive(Clone, Debug)]
pub enum Data {
    I64(Vec<i64>),
    F64(Vec<f64>),
    F32(Vec<f32>),
    I16(Vec<i16>),
    U8(Vec<u8>),
    Bool(Vec<bool>),
    Str(Vec<String>),
}

impl Npy {
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// Elements per leading index.
    pub fn row(&self) -> usize {
        self.shape.iter().skip(1).product()
    }

    pub fn i64(&self) -> Vec<i64> {
        match &self.data {
            Data::I64(v) => v.clone(),
            Data::I16(v) => v.iter().map(|&x| x as i64).collect(),
            Data::U8(v) => v.iter().map(|&x| x as i64).collect(),
            Data::Bool(v) => v.iter().map(|&x| x as i64).collect(),
            d => panic!("not an integer array: {d:?}"),
        }
    }

    pub fn f64(&self) -> Vec<f64> {
        match &self.data {
            Data::F64(v) => v.clone(),
            Data::F32(v) => v.iter().map(|&x| x as f64).collect(),
            _ => self.i64().into_iter().map(|x| x as f64).collect(),
        }
    }

    pub fn bools(&self) -> Vec<bool> {
        match &self.data {
            Data::Bool(v) => v.clone(),
            _ => self.i64().into_iter().map(|x| x != 0).collect(),
        }
    }

    pub fn bytes(&self) -> Vec<u8> {
        match &self.data {
            Data::U8(v) => v.clone(),
            d => panic!("not a uint8 array: {d:?}"),
        }
    }

    pub fn strings(&self) -> Vec<String> {
        match &self.data {
            Data::Str(v) => v.clone(),
            d => panic!("not a string array: {d:?}"),
        }
    }

    pub fn scalar_i64(&self) -> i64 {
        self.i64()[0]
    }

    pub fn string(&self) -> String {
        self.strings()[0].clone()
    }
}

fn le<const N: usize, T>(body: &[u8], n: usize, f: fn([u8; N]) -> T) -> Vec<T> {
    body.as_chunks::<N>()
        .0
        .iter()
        .take(n)
        .map(|c| f(*c))
        .collect()
}

fn parse_npy(bytes: &[u8]) -> Npy {
    assert_eq!(&bytes[..6], b"\x93NUMPY", "not an npy file");
    let major = bytes[6];
    let (header_len, start) = if major == 1 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10)
    } else {
        (
            u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize,
            12,
        )
    };
    let header = std::str::from_utf8(&bytes[start..start + header_len]).unwrap();
    let body = &bytes[start + header_len..];
    let field = |name: &str| {
        let i = header.find(&format!("'{name}':")).unwrap() + name.len() + 3;
        header[i..].trim_start()
    };
    let descr = field("descr");
    let descr = &descr[1..descr[1..].find('\'').unwrap() + 1];
    assert!(field("fortran_order").starts_with("False"), "fortran order");
    let shape_text = field("shape");
    let shape_text = &shape_text[1..shape_text.find(')').unwrap()];
    let shape: Vec<usize> = shape_text
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse().unwrap())
        .collect();
    let n: usize = shape.iter().product();
    let data = match descr {
        "<i8" => Data::I64(le(body, n, i64::from_le_bytes)),
        "<f8" => Data::F64(le(body, n, f64::from_le_bytes)),
        "<f4" => Data::F32(le(body, n, f32::from_le_bytes)),
        "<i2" => Data::I16(le(body, n, i16::from_le_bytes)),
        "|u1" => Data::U8(body[..n].to_vec()),
        "|b1" => Data::Bool(body[..n].iter().map(|&b| b != 0).collect()),
        d if d.starts_with("<U") => {
            let width: usize = d[2..].parse().unwrap();
            Data::Str(
                body.chunks_exact(width * 4)
                    .take(n)
                    .map(|c| {
                        le(c, width, u32::from_le_bytes)
                            .into_iter()
                            .take_while(|&u| u != 0)
                            .map(|u| char::from_u32(u).unwrap())
                            .collect()
                    })
                    .collect(),
            )
        }
        d => panic!("unsupported npy dtype {d}"),
    };
    Npy { shape, data }
}

/// Reads every array of an `.npz`.
pub fn read_npz(path: &Path) -> HashMap<String, Npy> {
    let file = std::fs::File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut zip = zip::ZipArchive::new(file).unwrap();
    let mut out = HashMap::new();
    for i in 0..zip.len() {
        let mut entry = zip.by_index(i).unwrap();
        let name = entry.name().trim_end_matches(".npy").to_owned();
        let mut bytes = Vec::new();
        entry.read_to_end(&mut bytes).unwrap();
        out.insert(name, parse_npy(&bytes));
    }
    out
}

pub fn fixture_dir() -> PathBuf {
    repo().join("crates/env/tests/fixtures/parity")
}

/// The fixture files, sorted by name.
pub fn fixtures() -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(fixture_dir())
        .expect("fixture dir")
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "npz"))
        .collect();
    v.sort();
    v
}

struct Play<'a>(&'a mut ZCore);

impl movie::Playback for Play<'_> {
    fn run_frame(&mut self, buttons: u8) {
        self.0.run_frame(buttons, false);
    }
    fn poke(&mut self, addr: u16, value: u8) {
        self.0.poke(addr, value);
    }
}

/// zcore state at a fixture's room entry: its corpus movie (`movie`, after the base) replayed for `entry_frame`
/// inputs, with the pokes tagged before `entry_frame` applied (how `record_env_parity.py` reached the entry on
/// the oracle; the fixture's `ram0` check proves the two states agree).
pub fn entry_state(f: &HashMap<String, Npy>) -> MachineState {
    let movies = repo().join("corpus/movies");
    let name = f["movie"].string();
    let entry = f["entry_frame"].scalar_i64() as usize;
    let m = movie::Movie::load(&movies.join(format!("{name}.tfmovie"))).expect("corpus movie");
    let mut core = open_core().expect("ROM");
    movie::play_base(&mut Play(&mut core), &m, &movies).expect("base movie");
    let mut pokes = m.pokes.iter().peekable();
    for (k, &buttons) in m.inputs[..entry].iter().enumerate() {
        while let Some(q) = pokes.next_if(|q| q.frame as usize <= k) {
            core.poke(q.addr, q.value);
        }
        core.run_frame(buttons, false);
    }
    core.snapshot()
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

/// A decoded `TFZSTATE` file as a session snapshot (executor flags from the file, no frames or steps).
pub fn read_state_file(path: &Path) -> Result<zelda::SessionSnapshot<MachineState>, env::EnvError> {
    let err = |e: &dyn std::fmt::Display| env::EnvError::State(format!("{}: {e}", path.display()));
    let rom = rom().ok_or_else(|| env::EnvError::State("ROM not found".into()))?;
    let bytes = std::fs::read(path).map_err(|e| err(&e))?;
    let file = zcore::state_file::decode(&rom, &bytes).map_err(|e| err(&e))?;
    Ok(zelda::SessionSnapshot {
        machine: file.machine,
        was_in_cave: file.flags & 1 != 0,
        just_exited_cave: file.flags & 2 != 0,
        total_frames: 0,
        steps: 0,
    })
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
        let rom = rom().ok_or_else(|| env::EnvError::State("ROM not found".into()))?;
        let flags = snapshot.was_in_cave as u8 | (snapshot.just_exited_cave as u8) << 1;
        let bytes = zcore::state_file::encode(&rom, &snapshot.machine, flags);
        let tmp = PathBuf::from(format!("{}.{}.tmp", path.display(), std::process::id()));
        std::fs::write(&tmp, bytes)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
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
