//! `GameData` load errors users see when configs, game.yaml tags and routes.yaml disagree, and specialist
//! lookup by name or path.  Each case edits a temporary copy of the repo's data.

mod common;

use env::GameData;
use std::path::Path;

/// A copy of the game data (`triforce/game.yaml`, `triforce/routes.yaml`, `triforce/heart_value.json`,
/// `triforce/chain_hp.json`, `specialists/`) under a temp dir.
fn copy_data() -> tempfile::TempDir {
    let tmp = tempfile::tempdir().unwrap();
    let repo = common::repo();
    std::fs::create_dir_all(tmp.path().join("triforce")).unwrap();
    std::fs::create_dir_all(tmp.path().join("specialists")).unwrap();
    for rel in ["triforce/game.yaml", "triforce/routes.yaml", "triforce/heart_value.json", "triforce/chain_hp.json"] {
        std::fs::copy(repo.join(rel), tmp.path().join(rel)).unwrap();
    }
    for entry in std::fs::read_dir(repo.join("specialists")).unwrap() {
        let path = entry.unwrap().path();
        // The families are the top-level YAML files.
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
    assert!(data.chain_hp.contains_key("9:0x10"), "the shipped profile (round 29's level 9) loads");
    assert_eq!(data.states_dir, tmp.path().join("triforce/states"));
}

/// Round 26: a heart-value table fitted for the route before it changed (here: another exit at one step) does not
/// stop the data from loading (the fitter reads the route through it to refit), but a family that prices hearts by
/// it is refused when it is handed out; a family that reads no table is not.
#[test]
fn a_stale_heart_value_table_stops_only_the_families_that_read_it() {
    let tmp = copy_data();
    edit(tmp.path(), "triforce/heart_value.json", "\"exit\": \"triforce\"", "\"exit\": \"N\"");
    let data = GameData::load(tmp.path()).unwrap();
    let error = data.specialist("goriya").unwrap_err().to_string();
    assert!(error.contains("does not match route") && error.contains("refit"), "{error}");
    assert!(!data.specialists["ow-start"].reward.heart_value.uses_table());
    data.specialist("ow-start").unwrap();
}

/// Phase X: a route's start states are built before its chains can fit its tables, so a family holding states in a
/// route's namespace (here like-like's `d3first_4_32w_0`, with `chain_hp` and `heart_value` but no
/// `chain_hp.d3first.json` or `heart_value.d3first.json`) still loads and is handed out (the state builder, the
/// runner, d2first's chains); only an env that draws its start states (`check_draws`) asks for the namespace's tables.
#[test]
fn a_routes_states_without_their_tables_stop_only_the_draws() {
    let tmp = copy_data();
    let states = tmp.path().join("triforce/states");
    std::fs::create_dir_all(&states).unwrap();
    std::fs::copy(common::repo().join("triforce/states/4_32w_0.state"), states.join("d3first_4_32w_0.state")).unwrap();
    let data = GameData::load(tmp.path()).unwrap();
    let like_like = data.specialist("like-like").unwrap();
    data.check_config(&like_like).unwrap();
    let error = data.check_draws(&like_like).unwrap_err().to_string();
    assert!(error.contains("chain_hp.d3first.json"), "{error}");
    std::fs::remove_file(states.join("d3first_4_32w_0.state")).unwrap();
    let data = GameData::load(tmp.path()).unwrap();
    data.check_draws(&data.specialist("like-like").unwrap()).unwrap();
}

/// Round 27: a table prices its own route and the routes it extends (V at a step depends only on the steps up to the
/// next triforce).  With a scouting route `{extends: d2first, steps: [...]}` beside it (as `d6` was before it merged),
/// `d2first`'s table prices `d2first` only: a family on the scouting route is refused and an episode there gets no
/// table (the plain health terms).  Fitted on the scouting route, it prices both.  Since 1.9.0.0 `d2first` ends the
/// game (`exit: end`, which nothing may follow), so the scouting route here takes `d2first`'s last step, Zelda's
/// room, and `d2first`'s table its row.
#[test]
fn a_table_prices_the_routes_its_own_extends() {
    let tmp = copy_data();
    let end = "- {level: 9, room: '0x32', specialist: ganon, entry: S, exit: end}\n";
    edit(tmp.path(), "triforce/routes.yaml", end, "");
    let scout = format!("\nscout:\n  extends: d2first\n  steps:\n  {end}clips-gauntlet:\n");
    edit(tmp.path(), "triforce/routes.yaml", "\nclips-gauntlet:\n", &scout);
    let path = tmp.path().join("triforce/heart_value.json");
    let mut table: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let last = table["steps"].as_array_mut().unwrap().pop().unwrap();
    assert_eq!((last["room"].as_str(), last["exit"].as_str()), (Some("9:0x32"), Some("end")));
    std::fs::write(&path, serde_json::to_string(&table).unwrap()).unwrap();
    let data = GameData::load(tmp.path()).unwrap();
    assert_eq!(data.heart_values.as_deref().unwrap().routes, ["d2first"]);
    let goriya = data.specialist("goriya").unwrap();
    assert!(data.heart_values_for(&goriya, "d2first").is_some() && data.heart_values_for(&goriya, "scout").is_none());
    let mut on_scout = (*goriya).clone();
    on_scout.route = Some("scout".into());
    let error = data.check_heart_values(&on_scout).unwrap_err().to_string();
    assert!(error.contains("neither is nor extends"), "{error}");
    // The table fitted on the scouting route: d2first's rows and one more.
    assert_eq!(last["step"], data.map.routes["d2first"].len());
    table["route"] = "scout".into();
    table["steps"].as_array_mut().unwrap().push(last);
    std::fs::write(&path, serde_json::to_string(&table).unwrap()).unwrap();
    let data = GameData::load(tmp.path()).unwrap();
    assert_eq!(data.heart_values.as_deref().unwrap().routes, ["d2first", "scout"]);
    assert!(data.heart_values_for(&goriya, "d2first").is_some() && data.heart_values_for(&goriya, "scout").is_some());
    data.check_heart_values(&on_scout).unwrap();
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
        "rooms: ['(1, 0x35)', '(7, 0x2a)']",
        "rooms: ['(1, 0x35)', '(7, 0x2a)', '(1, 0x36)']",
    );
    assert_eq!(
        load_error(tmp.path()),
        "specialists/boss-aquamentus.yaml: rooms: [(1, 0x35), (1, 0x36), (7, 0x2a)] != game.yaml rooms tagged \
         `specialist: boss-aquamentus` [(1, 0x35), (7, 0x2a)]"
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
