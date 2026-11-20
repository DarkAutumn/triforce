//! Round 32: a route's own runs play up to the step its `own_models_until` marks (`GameMap::own_models_until`,
//! `GameMap::step_policy`); a route without the key, `d2first`, plays every step by the family's key.

use zelda::MapLocation;
use zelda::game_map::{GAME_YAML, GameMap, ROUTES_YAML, route_policy, split_policy};

fn with_routes(routes: &str) -> Result<GameMap, String> {
    GameMap::from_yaml(GAME_YAML, &ROUTES_YAML.replacen("\nalternates:", &format!("\n{routes}alternates:"), 1))
        .map_err(|e| e.0)
}

#[test]
fn d3first_plays_its_own_runs_until_the_magic_sword() {
    let map = GameMap::builtin();
    let until = map.own_models_until["d3first"];
    let step = &map.routes["d3first"][until];
    assert_eq!((step.location(), step.waypoint.as_deref()), (MapLocation::new(0, 0x21, true), Some("sword")));
    assert_eq!(map.own_models_until.len(), 1, "d3first's alone: d2first plays its families' keys everywhere");
    assert_eq!(map.step_policy("d3first", 0, "ow-start"), "ow-start@d3first");
    assert_eq!(map.step_policy("d3first", until - 1, "ow-ghini"), "ow-ghini@d3first");
    assert_eq!(map.step_policy("d3first", until, "ow-ghini"), "ow-ghini", "the grave's cave plays the default run");
    let last = map.routes["d3first"].len() - 1;
    assert_eq!(map.step_policy("d3first", last, "ganon"), "ganon");
    for k in [0, 100, map.routes["d2first"].len() - 1] {
        assert_eq!(map.step_policy("d2first", k, "darknut"), "darknut");
    }
    assert_eq!(route_policy("darknut", "d3first"), "darknut@d3first");
    assert_eq!(split_policy("darknut@d3first"), ("darknut", Some("d3first")));
    assert_eq!(split_policy("darknut"), ("darknut", None));
}

#[test]
fn an_extending_route_keeps_the_cutoff_and_the_marker_names_one_step() {
    let routes = "r32base:\n  extends: d2first\n  states: r32base\n  own_models_until: {waypoint: raft}\n  steps: []\n\
                  r32scout:\n  extends: r32base\n  steps: []\n";
    let map = with_routes(routes).unwrap();
    let raft = map.own_models_until["r32base"];
    assert_eq!(map.routes["r32base"][raft].waypoint.as_deref(), Some("raft"));
    assert_eq!(map.own_models_until["r32scout"], raft, "inherited through extends");
    assert_eq!(map.step_policy("r32scout", raft - 1, "darknut"), "darknut@r32scout");
    // d2first takes several heart containers from caves: the marker must name one step.
    let e = with_routes("r32two:\n  extends: d2first\n  states: r32two\n  own_models_until: {waypoint: heart}\n  steps: []\n")
        .unwrap_err();
    assert!(e.contains("waypoint \"heart\" marks") && e.contains("not one"), "{e}");
    let e = with_routes("r32none:\n  extends: d2first\n  states: r32none\n  own_models_until: {waypoint: grail}\n  steps: []\n")
        .unwrap_err();
    assert!(e.contains("no step of the route has waypoint \"grail\""), "{e}");
    let e = with_routes("r32bad:\n  extends: d3first\n  own_models_until: {step: 3}\n  steps: []\n").unwrap_err();
    assert!(e.contains("own_models_until"), "{e}");
}
