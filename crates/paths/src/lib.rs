//! Where the shared things are, for Rust: the twin of `triforce/paths.py` (`docs/data-layout.md`).
//!
//! The shared store is `$TRIFORCE_SHARED`, else `~/triforce-shared` (`triforce.store` writes it; `layout.json` there
//! names every location for tools that are not Python). Rust reads only the ROM from it: `$TRIFORCE_ROM` when set
//! (nothing else is tried), else the store's `rom/zelda.nes`, else the checkout's `roms/zelda.nes` (the layout before
//! the store, kept until the lead's reconciliation). A missing ROM is an error naming every place looked in.

use std::path::{Path, PathBuf};

/// The shared store's root (`$TRIFORCE_SHARED`).
pub const SHARED_ENV: &str = "TRIFORCE_SHARED";
/// An explicit ROM file (`$TRIFORCE_ROM`): the only place looked in when set.
pub const ROM_ENV: &str = "TRIFORCE_ROM";
/// The store's ROM, under its root.
pub const STORE_ROM: &str = "rom/zelda.nes";
/// The checkout's ROM before the store (a worktree's `roms` link).
pub const CHECKOUT_ROM: &str = "roms/zelda.nes";

/// The shared store's root: `$TRIFORCE_SHARED`, else `$HOME/triforce-shared`.
pub fn shared_root() -> PathBuf {
    shared_root_from(std::env::var_os(SHARED_ENV), std::env::var_os("HOME"))
}

fn shared_root_from(shared: Option<std::ffi::OsString>, home: Option<std::ffi::OsString>) -> PathBuf {
    match shared.filter(|value| !value.is_empty()) {
        Some(root) => PathBuf::from(root),
        None => PathBuf::from(home.unwrap_or_default()).join("triforce-shared"),
    }
}

/// The places the ROM is looked for, in order, for the checkout `checkout` (the folder holding `roms/`).
pub fn rom_candidates(checkout: &Path) -> Vec<PathBuf> {
    rom_candidates_from(std::env::var_os(ROM_ENV), &shared_root(), checkout)
}

fn rom_candidates_from(explicit: Option<std::ffi::OsString>, shared: &Path, checkout: &Path) -> Vec<PathBuf> {
    match explicit.filter(|value| !value.is_empty()) {
        Some(path) => vec![PathBuf::from(path)],
        None => vec![shared.join(STORE_ROM), checkout.join(CHECKOUT_ROM)],
    }
}

/// The ROM file: the first of [`rom_candidates`] that exists, else `Err` with [`rom_missing`]'s message.
pub fn rom_path(checkout: &Path) -> Result<PathBuf, String> {
    let candidates = rom_candidates(checkout);
    candidates.iter().find(|path| path.is_file()).cloned().ok_or_else(|| missing_message(&candidates))
}

/// The message of a missing ROM: every place looked in.
pub fn rom_missing(checkout: &Path) -> String {
    missing_message(&rom_candidates(checkout))
}

fn missing_message(candidates: &[PathBuf]) -> String {
    let looked: Vec<String> = candidates.iter().map(|path| path.display().to_string()).collect();
    format!(
        "ROM not found: looked in {} (set {ROM_ENV}, or ask the lead to install the store's {STORE_ROM})",
        looked.join(", ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_root_is_the_variable_else_home() {
        assert_eq!(shared_root_from(Some("/s".into()), Some("/h".into())), PathBuf::from("/s"));
        assert_eq!(shared_root_from(Some("".into()), Some("/h".into())), PathBuf::from("/h/triforce-shared"));
        assert_eq!(shared_root_from(None, Some("/h".into())), PathBuf::from("/h/triforce-shared"));
    }

    #[test]
    fn an_explicit_rom_is_the_only_candidate() {
        let explicit = rom_candidates_from(Some("/x/z.nes".into()), Path::new("/s"), Path::new("/c"));
        assert_eq!(explicit, vec![PathBuf::from("/x/z.nes")]);
        let default = rom_candidates_from(None, Path::new("/s"), Path::new("/c"));
        assert_eq!(default, vec![PathBuf::from("/s/rom/zelda.nes"), PathBuf::from("/c/roms/zelda.nes")]);
    }

    #[test]
    fn the_message_names_every_place() {
        let message = missing_message(&[PathBuf::from("/s/rom/zelda.nes"), PathBuf::from("/c/roms/zelda.nes")]);
        assert!(message.starts_with("ROM not found: looked in /s/rom/zelda.nes, /c/roms/zelda.nes"), "{message}");
    }

    #[test]
    fn the_store_comes_before_the_checkout() {
        let dir = tempfile::tempdir().unwrap();
        let shared = dir.path().join("shared");
        let checkout = dir.path().join("checkout");
        std::fs::create_dir_all(shared.join("rom")).unwrap();
        std::fs::create_dir_all(checkout.join("roms")).unwrap();
        std::fs::write(checkout.join(CHECKOUT_ROM), b"c").unwrap();
        let found = |shared: &Path| {
            rom_candidates_from(None, shared, &checkout).into_iter().find(|path| path.is_file())
        };
        assert_eq!(found(&shared), Some(checkout.join(CHECKOUT_ROM)));
        std::fs::write(shared.join(STORE_ROM), b"s").unwrap();
        assert_eq!(found(&shared), Some(shared.join(STORE_ROM)));
    }
}
