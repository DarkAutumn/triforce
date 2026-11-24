//! The ROM for the workspace's tests and benches: [`paths::rom_path`] (`$TRIFORCE_ROM`, else the shared store's
//! `rom/zelda.nes`, else `<repo>/roms/zelda.nes`).
//!
//! A missing ROM is a failure: [`bytes`] and [`path`] panic with the ROM message, so no ROM test can pass by
//! returning early. `TRIFORCE_ALLOW_NO_ROM=1` turns that into `None` (the caller returns early, as a skip).

use std::path::{Path, PathBuf};

/// Set to `1` to let ROM tests skip (return early) instead of failing.
pub const ALLOW_NO_ROM_ENV: &str = "TRIFORCE_ALLOW_NO_ROM";

/// The repository root.
pub fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Whether a missing ROM may skip (`TRIFORCE_ALLOW_NO_ROM=1`).
pub fn missing_allowed() -> bool {
    std::env::var_os(ALLOW_NO_ROM_ENV).is_some_and(|v| v == "1")
}

/// The ROM's path when it exists. Missing: panics with the ROM message (every place looked in), or (with
/// `TRIFORCE_ALLOW_NO_ROM=1`) prints it and returns None.
pub fn path() -> Option<PathBuf> {
    match paths::rom_path(&repo()) {
        Ok(path) => Some(path),
        Err(missing) => {
            let message = format!("{missing}; {ALLOW_NO_ROM_ENV}=1 skips the ROM tests");
            assert!(missing_allowed(), "{message}");
            eprintln!("{message}");
            None
        }
    }
}

/// The ROM file's bytes; missing as in [`path`].
pub fn bytes() -> Option<Vec<u8>> {
    let path = path()?;
    Some(std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display())))
}
