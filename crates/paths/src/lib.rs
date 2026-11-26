//! Where the shared things are, for Rust: the twin of `triforce/paths.py` (`docs/data-layout.md`).
//!
//! The shared store is `$TRIFORCE_SHARED`, else `~/triforce-shared` (`triforce.store` writes it; `layout.json` there
//! names every location for tools that are not Python). The ROM: `$TRIFORCE_ROM` when set (nothing else is tried),
//! else the store's `rom/zelda.nes`, else the checkout's `roms/zelda.nes` (the layout before the store, kept until the
//! lead's reconciliation). A missing ROM is an error naming every place looked in. The data (harvested states, the
//! bench's harvested rooms, curricula, demos): [`data_dir`], the same rule as `paths.data_source`, with the pins of
//! the checkout this crate was built from ([`pins`], `triforce/data_pins.yaml`, as `paths.pins` reads them from the
//! checkout `triforce/paths.py` lives in).

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

/// The pins file, under a checkout: `kind: snapshot` lines (`#` comments, blank lines); no file, no pins.
pub const PINS_FILE: &str = "triforce/data_pins.yaml";
/// The data kinds (`paths.DATA_KINDS`).
pub const DATA_KINDS: [&str; 4] = ["harvested", "harvested-rooms", "trajectories", "demos"];

/// The checkout this crate was built from (the folder holding `crates/paths`): its pins are the ones read.
pub fn checkout() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// kind -> snapshot name the checkout `checkout` pins (`<checkout>/triforce/data_pins.yaml`; empty without the
/// file), or the line that is not a pin (`paths.pins`' rule: `kind: name` of a known kind, once each).
pub fn pins(checkout: &Path) -> Result<Vec<(String, String)>, String> {
    let path = checkout.join(PINS_FILE);
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Ok(Vec::new());
    };
    let mut found: Vec<(String, String)> = Vec::new();
    for (number, line) in text.lines().enumerate() {
        let body = line.split('#').next().unwrap_or("").trim();
        if body.is_empty() {
            continue;
        }
        let pin = body.split_once(':').map(|(kind, name)| (kind.trim(), name.trim()));
        match pin {
            Some((kind, name))
                if DATA_KINDS.contains(&kind)
                    && !name.is_empty()
                    && !name.contains(' ')
                    && name != "current"
                    && !found.iter().any(|(k, _)| k == kind) =>
            {
                found.push((kind.to_owned(), name.to_owned()));
            }
            _ => {
                return Err(format!(
                    "{}:{}: {:?}: a pin is 'KIND: SNAPSHOT' with KIND one of {}, once each, and a snapshot name",
                    path.display(),
                    number + 1,
                    line.trim(),
                    DATA_KINDS.join(", ")
                ))
            }
        }
    }
    Ok(found)
}

/// The snapshot `checkout` pins for `kind`; a malformed pins file is a panic naming its line (a committed file:
/// `paths.pins` refuses it the same way and the Python tests read it).
fn pinned(kind: &str, checkout: &Path) -> Option<String> {
    let pins = pins(checkout).unwrap_or_else(|error| panic!("{error}"));
    pins.into_iter().find(|(k, _)| k == kind).map(|(_, name)| name)
}

/// The folder of `kind` a reader uses, `legacy` being where the caller's layout keeps it (`<states>/harvested`,
/// `<checkout>/trajectories`): `legacy` while it exists (the transition: today's command lines read what they read),
/// else the snapshot this crate's checkout pins ([`pins`]), else the store's current snapshot of `kind`
/// (`data/<kind>/current` names it), else `legacy` again, for the reader's refusal ([`data_missing`] says where it
/// looked).
pub fn data_dir(kind: &str, legacy: &Path) -> PathBuf {
    data_dir_in(kind, legacy, &checkout(), &shared_root())
}

fn data_dir_in(kind: &str, legacy: &Path, checkout: &Path, shared: &Path) -> PathBuf {
    if legacy.exists() {
        return legacy.to_path_buf();
    }
    let found = match pinned(kind, checkout) {
        Some(name) => snapshot_in(kind, &name, shared),
        None => current_snapshot_in(kind, shared),
    };
    found.unwrap_or_else(|| legacy.to_path_buf())
}

/// The store's current snapshot folder of `kind`, None when the store has none.
pub fn current_snapshot(kind: &str) -> Option<PathBuf> {
    current_snapshot_in(kind, &shared_root())
}

fn current_snapshot_in(kind: &str, shared: &Path) -> Option<PathBuf> {
    let name = std::fs::read_to_string(shared.join("data").join(kind).join("current")).ok()?;
    snapshot_in(kind, name.trim(), shared)
}

fn snapshot_in(kind: &str, name: &str, shared: &Path) -> Option<PathBuf> {
    let folder = shared.join("data").join(kind).join(name);
    (!name.is_empty() && folder.is_dir()).then_some(folder)
}

/// What a reader of `kind` says when [`data_dir`] found nothing: both places it looked.
pub fn data_missing(kind: &str, legacy: &Path) -> String {
    let shared = shared_root();
    let checkout = checkout();
    match pinned(kind, &checkout) {
        Some(name) => format!(
            "{} is missing and the store at {} has no {kind} snapshot {name:?} (pinned by {})",
            legacy.display(),
            shared.display(),
            checkout.join(PINS_FILE).display()
        ),
        None => format!(
            "{} is missing and the store at {} has no current {kind} snapshot ({})",
            legacy.display(),
            shared.display(),
            shared.join("data").join(kind).join("current").display()
        ),
    }
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
    fn data_is_the_callers_folder_else_the_stores_current_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let shared = dir.path().join("shared");
        let checkout = dir.path().join("checkout");
        let legacy = checkout.join("triforce/states/harvested");
        assert_eq!(data_dir_in("harvested", &legacy, &checkout, &shared), legacy);
        std::fs::create_dir_all(shared.join("data/harvested/s1")).unwrap();
        std::fs::write(shared.join("data/harvested/current"), "s1\n").unwrap();
        assert_eq!(data_dir_in("harvested", &legacy, &checkout, &shared), shared.join("data/harvested/s1"));
        std::fs::create_dir_all(&legacy).unwrap();
        assert_eq!(data_dir_in("harvested", &legacy, &checkout, &shared), legacy);
        assert_eq!(current_snapshot_in("trajectories", &shared), None);
    }

    #[test]
    fn a_pin_wins_over_current_and_the_legacy_folder_over_both() {
        let dir = tempfile::tempdir().unwrap();
        let shared = dir.path().join("shared");
        let checkout = dir.path().join("checkout");
        let legacy = checkout.join("trajectories");
        for name in ["old", "new"] {
            std::fs::create_dir_all(shared.join("data/trajectories").join(name)).unwrap();
        }
        std::fs::write(shared.join("data/trajectories/current"), "new\n").unwrap();
        assert_eq!(data_dir_in("trajectories", &legacy, &checkout, &shared), shared.join("data/trajectories/new"));
        std::fs::create_dir_all(checkout.join("triforce")).unwrap();
        std::fs::write(checkout.join(PINS_FILE), "# main's pins\n\ntrajectories: old  # the d3first tracks\n").unwrap();
        assert_eq!(data_dir_in("trajectories", &legacy, &checkout, &shared), shared.join("data/trajectories/old"));
        // Another kind is not pinned: its current.
        assert_eq!(pinned("harvested", &checkout), None);
        // A pin the store does not hold: the reader's refusal, never the current one.
        std::fs::write(checkout.join(PINS_FILE), "trajectories: gone\n").unwrap();
        assert_eq!(data_dir_in("trajectories", &legacy, &checkout, &shared), legacy);
        std::fs::create_dir_all(&legacy).unwrap();
        assert_eq!(data_dir_in("trajectories", &legacy, &checkout, &shared), legacy);
    }

    #[test]
    fn a_malformed_pins_file_names_its_line() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("triforce")).unwrap();
        for bad in ["models: x", "trajectories x", "trajectories:", "trajectories: a\ntrajectories: b", "demos: current"] {
            std::fs::write(dir.path().join(PINS_FILE), bad).unwrap();
            let error = pins(dir.path()).unwrap_err();
            assert!(error.contains("a pin is 'KIND: SNAPSHOT'"), "{bad}: {error}");
        }
        assert_eq!(pins(&dir.path().join("nowhere")).unwrap(), Vec::new());
    }

    #[test]
    fn this_checkouts_pins_parse() {
        // The file `paths.pins` reads too (tests/test_store.py checks the Python side and that the store holds them).
        assert!(pins(&checkout()).is_ok());
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
