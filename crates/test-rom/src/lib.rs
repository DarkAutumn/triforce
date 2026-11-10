//! The ROM for the workspace's tests and benches: `$TRIFORCE_ROM`, else `<repo>/roms/zelda.nes`.
//!
//! A missing ROM is a failure: [`bytes`] and [`path`] panic with the ROM message, so no ROM test can pass by
//! returning early. `TRIFORCE_ALLOW_NO_ROM=1` turns that into `None` (the caller returns early, as a skip).

use std::path::{Path, PathBuf};

/// The message of a missing ROM.
pub const ROM_MISSING: &str = "ROM not found: set TRIFORCE_ROM or create roms/zelda.nes";

/// Set to `1` to let ROM tests skip (return early) instead of failing.
pub const ALLOW_NO_ROM_ENV: &str = "TRIFORCE_ALLOW_NO_ROM";

/// The repository root.
pub fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Where the ROM should be: `$TRIFORCE_ROM`, else `<repo>/roms/zelda.nes`.
pub fn rom_path() -> PathBuf {
    std::env::var_os("TRIFORCE_ROM").map_or_else(|| repo().join("roms/zelda.nes"), PathBuf::from)
}

/// Whether a missing ROM may skip (`TRIFORCE_ALLOW_NO_ROM=1`).
pub fn missing_allowed() -> bool {
    std::env::var_os(ALLOW_NO_ROM_ENV).is_some_and(|v| v == "1")
}

/// The ROM's path when it exists. Missing: panics with the ROM message, or (with `TRIFORCE_ALLOW_NO_ROM=1`)
/// prints it and returns None.
pub fn path() -> Option<PathBuf> {
    let path = rom_path();
    if path.is_file() {
        return Some(path);
    }
    let message = format!("{ROM_MISSING} ({}); {ALLOW_NO_ROM_ENV}=1 skips the ROM tests", path.display());
    assert!(missing_allowed(), "{message}");
    eprintln!("{message}");
    None
}

/// The ROM file's bytes; missing as in [`path`].
pub fn bytes() -> Option<Vec<u8>> {
    let path = path()?;
    Some(std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display())))
}
