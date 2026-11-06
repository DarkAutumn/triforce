//! `GameData` load errors users see when configs, game.yaml tags and routes.yaml disagree, and specialist
//! lookup by name or path.  Each case edits a temporary copy of the repo's data.

mod common;

use env::GameData;
use std::path::Path;

/// A copy of the game data (`triforce/game.yaml`, `triforce/routes.yaml`, `specialists/`) under a temp dir.
fn copy_data() -> tempfile::TempDir {
    let tmp = tempfile::tempdir().unwrap();
    let repo = common::repo();
    std::fs::create_dir_all(tmp.path().join("triforce")).unwrap();
    std::fs::create_dir_all(tmp.path().join("specialists")).unwrap();
    for rel in ["triforce/game.yaml", "triforce/routes.yaml"] {
        std::fs::copy(repo.join(rel), tmp.path().join(rel)).unwrap();
    }
    for entry in std::fs::read_dir(repo.join("specialists")).unwrap() {
        let path = entry.unwrap().path();
        // The families are the top-level YAML files; `specialists/arms/` holds experiment variants.
        if path.extension().is_none_or(|e| e != "yaml") {
            continue;
        }
        std::fs::copy(
            &path,
            tmp.path()
                .join("specialists")
                .join(path.file_name().unwrap()),
        )
        .unwrap();
    }
    tmp
}

fn edit(root: &Path, rel: &str, from: &str, to: &str) {
    let path = root.join(rel);
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.contains(from), "{rel} lacks {from:?}");
    std::fs::write(&path, text.replacen(from, to, 1)).unwrap();
}

fn load_error(root: &Path) -> String {
    GameData::load(root).unwrap_err().to_string()
}

#[test]
fn an_unmodified_copy_loads() {
    let tmp = copy_data();
    let data = GameData::load(tmp.path()).unwrap();
    assert!(data.chain_hp.is_empty(), "chain_hp.json is optional");
    assert_eq!(data.states_dir, tmp.path().join("triforce/states"));
}

#[test]
fn a_family_without_a_config_is_rejected() {
    let tmp = copy_data();
    std::fs::remove_file(tmp.path().join("specialists/wallmaster.yaml")).unwrap();
    let err = load_error(tmp.path());
    assert!(err.starts_with("routes.yaml: families: ["), "{err}");
    assert!(err.contains("!= specialists/*.yaml"), "{err}");
}

#[test]
fn a_config_must_declare_its_file_name() {
    let tmp = copy_data();
    edit(
        tmp.path(),
        "specialists/boss-aquamentus.yaml",
        "name: boss-aquamentus",
        "name: aqua",
    );
    assert_eq!(
        load_error(tmp.path()),
        "specialists/boss-aquamentus.yaml: name: declares name \"aqua\""
    );
}

#[test]
fn config_rooms_must_match_the_game_yaml_tags() {
    let tmp = copy_data();
    edit(
        tmp.path(),
        "specialists/boss-aquamentus.yaml",
        "rooms: ['(1, 0x35)']",
        "rooms: ['(1, 0x35)', '(1, 0x36)']",
    );
    assert_eq!(
        load_error(tmp.path()),
        "specialists/boss-aquamentus.yaml: rooms: [(1, 0x35), (1, 0x36)] != game.yaml rooms tagged \
         `specialist: boss-aquamentus` [(1, 0x35)]"
    );
}

#[test]
fn bad_config_values_name_the_file_and_key() {
    let tmp = copy_data();
    edit(
        tmp.path(),
        "specialists/boss-aquamentus.yaml",
        "max_steps: 400",
        "max_steps: 0",
    );
    let err = load_error(tmp.path());
    assert!(
        err.starts_with("specialists/boss-aquamentus.yaml: max_steps:"),
        "{err}"
    );
}

#[test]
fn specialists_resolve_by_name_or_yaml_path() {
    let data = common::game_data();
    assert_eq!(data.specialist("goriya").unwrap().name, "goriya");
    let err = data.specialist("nope").unwrap_err().to_string();
    assert!(
        err.starts_with("No specialist config specialists/nope.yaml; known: ["),
        "{err}"
    );
    assert!(err.contains("\"goriya\""), "{err}");

    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("custom.yaml");
    std::fs::write(
        &path,
        "name: custom\nrooms: ['(1, 0x44)']\nsuccess: [exit-correct]\n",
    )
    .unwrap();
    let custom = data.specialist(path.to_str().unwrap()).unwrap();
    assert_eq!(custom.name, "custom");
    // A path config is validated against the map like any other.
    std::fs::write(
        &path,
        "name: custom\nrooms: ['(1, 0x44)']\nsuccess: [exit-correct]\nroute: nowhere\n",
    )
    .unwrap();
    let err = data
        .specialist(path.to_str().unwrap())
        .unwrap_err()
        .to_string();
    assert!(err.contains("route: unknown route \"nowhere\""), "{err}");
    // A missing .yaml path is an unknown specialist, not an I/O error.
    let missing = tmp.path().join("missing.yaml");
    let err = data
        .specialist(missing.to_str().unwrap())
        .unwrap_err()
        .to_string();
    assert!(err.starts_with("No specialist config"), "{err}");
}
