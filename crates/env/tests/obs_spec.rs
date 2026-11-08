//! Observation schema v2 against Python's (`fixtures/observation_spec.json`, dumped from
//! `triforce.observation`: feature-name tuples, vocabularies and the gymnasium space of every key).

use env::observation::{ENTITY_VOCAB, OBS_SPEC, ROOM_VOCAB, feature_names};
use serde_json::Value;

#[test]
fn feature_names_and_spaces_match_python() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/observation_spec.json");
    let spec: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let names = |key: &str| -> Vec<String> {
        spec[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_owned())
            .collect()
    };
    let n = feature_names();
    assert_eq!(n.room_state, names("ROOM_STATE_FEATURES"));
    assert_eq!(n.link, names("LINK_FEATURES"));
    assert_eq!(n.entity, names("ENTITY_FEATURES"));
    assert_eq!(n.history, names("HISTORY_FEATURES"));
    assert_eq!(n.objective, names("OBJECTIVE_FEATURES"));
    assert_eq!(n.target, names("TARGET_FEATURES"));
    assert_eq!(spec["ROOM_VOCAB"], ROOM_VOCAB);
    assert_eq!(spec["ENTITY_VOCAB"], ENTITY_VOCAB);
    let spaces = spec["spaces"].as_object().unwrap();
    assert_eq!(spaces.len(), OBS_SPEC.len());
    for key in OBS_SPEC {
        let space = &spaces[key.key];
        let shape: Vec<usize> = space["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(shape, key.shape, "{}", key.key);
        assert_eq!(space["dtype"], key.dtype, "{}", key.key);
        assert_eq!(space["low"].as_f64().unwrap(), key.low, "{}", key.key);
        assert_eq!(space["high"].as_f64().unwrap(), key.high, "{}", key.key);
    }
}
