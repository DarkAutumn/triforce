//! Process-wide inputs: the repo root (the directory holding the `triforce` package), the validated ROM, the
//! game data, and the `TFZSTATE` start-state store.

use crate::errors::{ConfigError, RomError, StateFormatError};
use env::config::{GameData, STATES_DIR};
use env::{EnvError, StateStore};
use parking_lot::Mutex;
use pyo3::exceptions::PyOSError;
use pyo3::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use zcore::{MachineState, Rom, state_file};
use zelda::SessionSnapshot;

struct Loaded {
    root: PathBuf,
    rom: Option<Arc<Rom>>,
    data: Option<Arc<GameData>>,
}

static LOADED: Mutex<Option<Loaded>> = Mutex::new(None);

/// The directory holding the `triforce` package (the repo root of an editable install): the parent of the
/// directory this extension module was loaded from.
fn find_root(py: Python<'_>) -> PyResult<PathBuf> {
    let file: String = py
        .import("triforce._native")?
        .getattr("__file__")?
        .extract()?;
    let package = Path::new(&file)
        .parent()
        .ok_or_else(|| ConfigError::new_err(format!("extension path {file:?} has no parent")))?;
    Ok(package.parent().unwrap_or(Path::new(".")).to_path_buf())
}

fn with_loaded<T>(py: Python<'_>, f: impl FnOnce(&mut Loaded) -> PyResult<T>) -> PyResult<T> {
    let mut guard = LOADED.lock();
    if guard.is_none() {
        *guard = Some(Loaded {
            root: find_root(py)?,
            rom: None,
            data: None,
        });
    }
    f(guard.as_mut().expect("loaded"))
}

/// The repo root.
pub fn root(py: Python<'_>) -> PyResult<PathBuf> {
    with_loaded(py, |l| Ok(l.root.clone()))
}

/// `$TRIFORCE_ROM`, else `<root>/roms/zelda.nes`.
pub fn rom_path(root: &Path) -> PathBuf {
    match std::env::var_os("TRIFORCE_ROM") {
        Some(p) => PathBuf::from(p),
        None => root.join("roms/zelda.nes"),
    }
}

/// The validated ROM, read once per process.
pub fn rom(py: Python<'_>) -> PyResult<Arc<Rom>> {
    with_loaded(py, |l| {
        if let Some(rom) = &l.rom {
            return Ok(rom.clone());
        }
        let path = rom_path(&l.root);
        if !path.exists() {
            return Err(RomError::new_err(format!(
                "ROM not found: set TRIFORCE_ROM or create roms/zelda.nes ({})",
                path.display()
            )));
        }
        let bytes = std::fs::read(&path)
            .map_err(|e| RomError::new_err(format!("{}: {e}", path.display())))?;
        let rom = Rom::from_ines(&bytes)
            .map_err(|e| RomError::new_err(format!("{}: {e}", path.display())))?;
        l.rom = Some(rom.clone());
        Ok(rom)
    })
}

/// The default state directory: `$TRIFORCE_STATES_DIR`, else `<root>/triforce/states`.
fn default_states_dir(root: &Path) -> PathBuf {
    std::env::var_os("TRIFORCE_STATES_DIR").map_or_else(|| root.join(STATES_DIR), PathBuf::from)
}

/// The game data (configs, map, routes, chain hp), loaded once; `states_dir` overrides where start states live.
pub fn data(py: Python<'_>, states_dir: Option<&Path>) -> PyResult<Arc<GameData>> {
    let base = with_loaded(py, |l| {
        if let Some(data) = &l.data {
            return Ok(data.clone());
        }
        let mut data = GameData::load(&l.root).map_err(|e| ConfigError::new_err(e.0))?;
        Arc::get_mut(&mut data).expect("fresh game data").states_dir = default_states_dir(&l.root);
        l.data = Some(data.clone());
        Ok(data)
    })?;
    Ok(match states_dir {
        Some(dir) if dir != base.states_dir => {
            let mut data = (*base).clone();
            data.states_dir = dir.to_path_buf();
            Arc::new(data)
        }
        _ => base,
    })
}

/// Start states as `TFZSTATE` files: bare names (`'1_44w_0'`, `'start'`) resolve under `dir` (`.state`
/// appended); absolute or existing paths are used as given.
#[derive(Clone)]
pub struct TfzStore {
    pub dir: PathBuf,
    pub rom: Arc<Rom>,
}

impl TfzStore {
    pub fn new(data: &GameData, rom: Arc<Rom>) -> TfzStore {
        TfzStore {
            dir: data.states_dir.clone(),
            rom,
        }
    }

    /// `emulator.state_path`.
    pub fn path(&self, name: &str) -> PathBuf {
        let p = Path::new(name);
        if p.is_absolute() || p.exists() {
            return p.to_path_buf();
        }
        if name.ends_with(".state") {
            self.dir.join(name)
        } else {
            self.dir.join(format!("{name}.state"))
        }
    }

    pub fn read(&self, path: &Path) -> Result<state_file::StateFile, StoreError> {
        let bytes =
            std::fs::read(path).map_err(|e| StoreError::Io(format!("{}: {e}", path.display())))?;
        state_file::decode(&self.rom, &bytes)
            .map_err(|e| StoreError::Format(format!("{}: {e}", path.display())))
    }

    pub fn encode(&self, snapshot: &SessionSnapshot<MachineState>) -> Vec<u8> {
        state_file::encode(&self.rom, &snapshot.machine, flags(snapshot))
    }

    /// Writes atomically (parallel workers may race on the same file).
    pub fn write(&self, path: &Path, bytes: &[u8]) -> std::io::Result<()> {
        if let Some(dir) = path.parent()
            && !dir.as_os_str().is_empty()
        {
            std::fs::create_dir_all(dir)?;
        }
        let tmp = PathBuf::from(format!(
            "{}.{}.{:?}.tmp",
            path.display(),
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::write(&tmp, bytes)?;
        std::fs::rename(&tmp, path)
    }
}

/// Executor flags of a state file (bit 0 `was_in_cave`, bit 1 `just_exited_cave`).
pub fn flags<S>(snapshot: &SessionSnapshot<S>) -> u8 {
    snapshot.was_in_cave as u8 | (snapshot.just_exited_cave as u8) << 1
}

/// Why a state file couldn't be read.
pub enum StoreError {
    Io(String),
    Format(String),
}

impl From<StoreError> for PyErr {
    fn from(e: StoreError) -> PyErr {
        match e {
            StoreError::Io(m) => PyOSError::new_err(m),
            StoreError::Format(m) => StateFormatError::new_err(m),
        }
    }
}

impl StateStore<MachineState> for TfzStore {
    fn load(&self, name: &str) -> Result<MachineState, EnvError> {
        match self.read(&self.path(name)) {
            Ok(file) => Ok(file.machine),
            Err(StoreError::Io(m) | StoreError::Format(m)) => Err(EnvError::State(m)),
        }
    }

    fn save(&self, path: &Path, snapshot: &SessionSnapshot<MachineState>) -> Result<(), EnvError> {
        self.write(path, &self.encode(snapshot))
            .map_err(|e| EnvError::State(format!("{}: {e}", path.display())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zcore::ZCore;

    /// The ROM; missing panics (`test_rom`), or None with `TRIFORCE_ALLOW_NO_ROM=1`.
    fn rom() -> Option<Arc<Rom>> {
        test_rom::bytes().map(|bytes| Rom::from_ines(&bytes).expect("valid ROM"))
    }

    #[test]
    fn store_resolves_names_and_round_trips_states() {
        let Some(rom) = rom() else { return };
        let dir = std::env::temp_dir().join(format!("tfz-store-{}", std::process::id()));
        let store = TfzStore {
            dir: dir.clone(),
            rom: rom.clone(),
        };
        assert_eq!(store.path("1_44w_0"), dir.join("1_44w_0.state"));
        assert_eq!(store.path("start.state"), dir.join("start.state"));
        assert_eq!(store.path("/abs/x.state"), PathBuf::from("/abs/x.state"));

        let mut core = ZCore::power_on(rom);
        for _ in 0..10 {
            core.run_frame(0, false);
        }
        let snapshot = SessionSnapshot {
            machine: core.snapshot(),
            was_in_cave: true,
            just_exited_cave: false,
            total_frames: 5,
            steps: 1,
        };
        store.save(&dir.join("x.state"), &snapshot).unwrap();
        let file = store.read(&store.path("x")).ok().expect("readable");
        assert_eq!(file.flags, 1);
        let mut other = ZCore::power_on(store.rom.clone());
        other.restore(&store.load("x").unwrap());
        assert_eq!(other.ram(), core.ram());

        std::fs::write(dir.join("old.state"), [0x1f, 0x8b, 8, 0]).unwrap();
        let Err(EnvError::State(message)) = store.load("old") else {
            panic!("legacy state loaded")
        };
        assert!(message.contains("legacy stable-retro state"), "{message}");
        let Err(EnvError::State(message)) = store.load("missing") else {
            panic!("missing state loaded")
        };
        assert!(message.contains("missing.state"), "{message}");
        std::fs::remove_dir_all(dir).unwrap();
    }
}
