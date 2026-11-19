//! Phase X: a route's start-state namespace (routes.yaml `states:`), the names in it, and bare names staying the
//! routes' without one (`d2first`'s, unchanged).

use zelda::game_map::{
    GAME_YAML, GameMap, ROUTES_YAML, bare_state, namespaced_state, state_namespace, valid_namespace,
};

/// d2first's steps again as a route with its own states (`r32test`), and one extending it that keeps them.
const TEST_ROUTES: &str = "\nr32test:\n  extends: d2first\n  states: r32test\n  steps: []\nr32scout:\n  extends: r32test\n  steps: []\n";

fn map() -> GameMap {
    GameMap::from_yaml(GAME_YAML, &ROUTES_YAML.replacen("\nalternates:", &format!("{TEST_ROUTES}alternates:"), 1))
        .expect("test routes load")
}

#[test]
fn a_route_names_its_states_namespace_and_extensions_keep_it() {
    let map = map();
    assert_eq!(map.namespace("d2first"), None, "d2first keeps its bare names");
    assert_eq!(map.namespace("r32test"), Some("r32test"));
    assert_eq!(map.namespace("r32scout"), Some("r32test"), "inherited through extends");
    assert_eq!(map.namespace_routes(Some("r32test")), ["r32test", "r32scout"]);
    assert_eq!(map.namespace_routes(None), ["d2first", "clips-gauntlet"]);
    assert!(GameMap::builtin().namespaces.is_empty(), "the shipped routes.yaml has no namespace yet");
}

#[test]
fn state_names_in_a_namespace() {
    assert_eq!(state_namespace("d3first_3_7cs_0"), Some("d3first"));
    assert_eq!(state_namespace("/x/d3first_0_1be_v2_h4_7.state"), Some("d3first"));
    for bare in ["3_7cs_0", "start", "0001.state", "/x/1_44w_0.state", "entry-step-3"] {
        assert_eq!(state_namespace(bare), None, "{bare}");
        assert_eq!(bare_state(bare), bare.rsplit('/').next().unwrap(), "{bare}");
    }
    assert_eq!(bare_state("d3first_3_7cs_0"), "3_7cs_0");
    assert_eq!(namespaced_state(Some("d3first"), "3_7cs_0"), "d3first_3_7cs_0");
    assert_eq!(namespaced_state(Some("d3first"), "start"), "start", "the game start is every route's");
    assert_eq!(namespaced_state(None, "3_7cs_0"), "3_7cs_0");
    assert!(valid_namespace("d3first") && valid_namespace("r32-test"));
    assert!(!valid_namespace("3first") && !valid_namespace("d3_first") && !valid_namespace("D3"));
}

#[test]
fn a_bare_start_on_a_namespaced_route_is_the_routes_state() {
    let map = map();
    assert_eq!(map.route_state("r32test", "3_7cs_0").unwrap(), "r32test_3_7cs_0");
    assert_eq!(map.route_state("r32test", "r32test_3_7cs_0").unwrap(), "r32test_3_7cs_0");
    assert_eq!(map.route_state("r32test", "start").unwrap(), "start");
    assert_eq!(map.route_state("d2first", "3_7cs_0").unwrap(), "3_7cs_0", "d2first unchanged");
    assert!(map.route_state("d2first", "r32test_3_7cs_0").is_err(), "another route's state");
    assert!(map.route_state("r32test", "d3first_3_7cs_0").is_err());
}

#[test]
fn a_mapping_route_needs_extends_or_states() {
    let bad = ROUTES_YAML.replacen("\nalternates:", "\nbad:\n  steps: []\nalternates:", 1);
    let e = GameMap::from_yaml(GAME_YAML, &bad).unwrap_err().0;
    assert!(e.contains("extends") && e.contains("states"), "{e}");
    let bad = ROUTES_YAML.replacen("\nalternates:", "\nbad:\n  extends: d2first\n  states: D3_x\n  steps: []\nalternates:", 1);
    assert!(GameMap::from_yaml(GAME_YAML, &bad).unwrap_err().0.contains("lower-case"));
}
