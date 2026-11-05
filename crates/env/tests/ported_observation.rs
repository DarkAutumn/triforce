//! `ObservationBuilder` on the named start states (zcore): contract conformance and decoded values
//! against known savestate facts.  Ported from tests/test_observation.py before the Python game stack is deleted
//! (plan step 16).  Every state comes from `common::load_named_state`.
//!
//! | Python test | Rust |
//! |---|---|
//! | test_space_matches_contract | here `space_matches_contract`; obs_spec.rs `feature_names_and_spaces_match_python` checks every key's shape, dtype and bounds against Python's dump |
//! | test_every_savestate_builds_a_valid_observation | here `every_savestate_builds_a_valid_observation` |
//! | test_goriya_room_entities_and_grid | here `goriya_room_entities_and_grid` |
//! | test_push_block_in_grid_and_room_state | here `push_block_in_grid_and_room_state` |
//! | test_start_state_link_and_room | here `start_state_link_and_room` |
//! | test_key_room_treasure | here `key_room_treasure` |
//! | test_sword_cave_exposes_ware_and_fires | here `sword_cave_exposes_ware_and_fires` |
//! | test_velocity_is_position_delta | here `velocity_is_position_delta` |
//! | test_room_change_resets_velocity_and_records_entry | here `room_change_resets_velocity_and_records_entry` |
//! | test_objective_block | here `objective_block` (the objective is passed to the builder instead of patched onto the state) |
//!
//! Features are looked up by their Python names (`feature_names()`), as the Python tests' R/L/E/H/O dicts did.

mod common;

use common::{ZCore, load_named_state, named_states, obs_values, obs_violations, open_core};
use env::observation::{
    ITEM_TYPE_BASE, NUM_ENTITY_SLOTS, OBS_SCHEMA_VERSION, OBS_SPEC, Observation,
    ObservationBuilder, TREASURE_SLOT, feature_names, room_index,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::BTreeSet;
use std::sync::Arc;
use zelda::objectives::{Objective, ObjectiveKind};
use zelda::tiles::TileClass;
use zelda::{Action, Change, GameMap, GameView, LoadOptions, MapLocation, Session, Var};

type S = Session<ZCore>;

fn index(names: &[String], name: &str) -> usize {
    names
        .iter()
        .position(|n| n == name)
        .unwrap_or_else(|| panic!("no feature {name}"))
}

fn r(name: &str) -> usize {
    index(&feature_names().room_state, name)
}

fn l(name: &str) -> usize {
    index(&feature_names().link, name)
}

fn e(name: &str) -> usize {
    index(&feature_names().entity, name)
}

fn h(name: &str) -> usize {
    index(&feature_names().history, name)
}

fn o(name: &str) -> usize {
    index(&feature_names().objective, name)
}

/// `pytest.approx` (relative 1e-6, absolute 1e-12).
fn approx(a: f32, b: f64) -> bool {
    (a as f64 - b).abs() <= (1e-6 * b.abs()).max(1e-12)
}

fn no_objective() -> Objective {
    Objective {
        kind: ObjectiveKind::None,
        targets: BTreeSet::new(),
        next_rooms: BTreeSet::new(),
        pbrs_targets: BTreeSet::new(),
    }
}

fn session() -> Option<S> {
    open_core().map(|nes| Session::new(nes, Arc::new(GameMap::builtin())))
}

/// `session.load(name, randomize_rng=False)`.
fn load(s: &mut S, name: &str) -> Arc<GameView> {
    let machine = load_named_state(name).expect("state");
    s.load(
        &machine,
        &LoadOptions::default(),
        &mut ChaCha8Rng::seed_from_u64(0),
    )
    .unwrap_or_else(|e| panic!("{name}: {e}"))
    .clone()
}

/// `_reset`: a fresh builder (objective block off) and its first observation of `name`.
fn reset(s: &mut S, name: &str) -> (ObservationBuilder, Observation) {
    let mut builder = ObservationBuilder::new(false);
    let state = load(s, name);
    let obs = builder.reset(&state, &no_objective(), s.mask());
    (builder, obs)
}

fn step(s: &mut S, builder: &mut ObservationBuilder, text: &str) -> (Change, Observation) {
    let change = s.act(Action::parse(text).unwrap()).unwrap();
    let obs = builder.build(&change, &no_objective(), s.mask());
    (change, obs)
}

/// Pre-entry states load with the room's monsters in their spawn clouds: WAIT until the first `slots` entities
/// are out.
fn wait_until_spawned(
    s: &mut S,
    builder: &mut ObservationBuilder,
    slots: usize,
) -> (Change, Observation) {
    for _ in 0..10 {
        let (change, obs) = step(s, builder, "WAIT 8");
        if obs.entity_features[..slots]
            .iter()
            .all(|f| f[e("spawning")] == 0.0)
        {
            return (change, obs);
        }
    }
    panic!("monsters still spawning after 10 WAITs");
}

fn grid_all(
    obs: &Observation,
    rows: std::ops::Range<usize>,
    cols: std::ops::Range<usize>,
    class: TileClass,
) -> bool {
    rows.into_iter()
        .all(|row| cols.clone().all(|col| obs.grid[row][col] == class as u8))
}

fn velocities(obs: &Observation) -> Vec<[f32; 2]> {
    obs.entity_features
        .iter()
        .map(|f| [f[e("vx")], f[e("vy")]])
        .collect()
}

#[test]
fn space_matches_contract() {
    const { assert!(OBS_SCHEMA_VERSION >= 1 && NUM_ENTITY_SLOTS == 12) };
    let keys: BTreeSet<&str> = OBS_SPEC.iter().map(|k| k.key).collect();
    let want: BTreeSet<&str> = [
        "room_id",
        "room_state",
        "link",
        "entity_types",
        "entity_features",
        "grid",
        "history",
        "objective",
        "action_mask",
    ]
    .into_iter()
    .collect();
    assert_eq!(keys, want);
    let n = feature_names();
    assert_eq!(n.entity[0], "presence");
    for names in [&n.room_state, &n.link, &n.entity, &n.history, &n.objective] {
        assert_eq!(names.iter().collect::<BTreeSet<_>>().len(), names.len());
    }
}

/// Acceptance: every savestate in the states directory produces a contract-conforming observation.
#[test]
fn every_savestate_builds_a_valid_observation() {
    let Some(mut s) = session() else { return };
    let names = named_states();
    assert!(names.len() > 100, "{}", names.len());
    let mut builder = ObservationBuilder::new(false);
    for name in &names {
        let state = load(&mut s, name);
        let obs = builder.reset(&state, &no_objective(), s.mask());
        let bad = obs_violations(&obs);
        assert!(bad.is_empty(), "{name}: {bad:?}");
        for (t, f) in obs.entity_types.iter().zip(&obs.entity_features) {
            if *t == 0 {
                assert!(
                    f.iter().all(|&x| x == 0.0),
                    "{name}: empty slot with features"
                );
            } else {
                assert_eq!(f[0], 1.0, "{name}: occupied slot without presence");
            }
        }
        assert_eq!(
            obs.room_id,
            room_index(state.level(), state.in_cave(), state.location()),
            "{name}"
        );
        assert_eq!(obs.action_mask, s.mask().to_flat(), "{name}");
    }
}

#[test]
fn goriya_room_entities_and_grid() {
    let Some(mut s) = session() else { return };
    let (mut builder, obs) = reset(&mut s, "1_44w_0");
    for f in &obs.entity_features[..3] {
        assert_eq!(f[e("spawning")], 1.0, "just created, in their spawn clouds");
        assert_eq!(f[e("hurts_on_touch")], 0.0);
    }
    assert_eq!(obs.history[h("entered_from_W")], 1.0);
    assert_eq!(obs.history[h("has_last_action")], 0.0);
    let (_, obs) = wait_until_spawned(&mut s, &mut builder, 3);
    let ram = *s.state().ram();
    for i in 0..3 {
        // NES slots 1-3 hold the three RedGoriyas.
        assert_eq!(obs.entity_types[i], 0x06);
        let f = &obs.entity_features[i];
        let (x, y) = (ram[0x70 + i + 1] as f64, ram[0x84 + i + 1] as f64);
        assert!(approx(f[e("x")], x / 255.0) && approx(f[e("y")], (y - 64.0) / 176.0));
        assert!(approx(f[e("dx")], (x - ram[0x70] as f64) / 255.0));
        assert!(
            f[e("hurts_on_touch")] == 1.0 && f[e("killable")] == 1.0 && f[e("projectile")] == 0.0
        );
        assert!(f[e("hp")] > 0.0);
    }
    assert!(obs.entity_types[3..TREASURE_SLOT].iter().all(|&t| t == 0));

    // The boomerang appears when the room is cleared: the treasure slot shows kind + position, flagged hidden.
    let treasure = &obs.entity_features[TREASURE_SLOT];
    assert_eq!(obs.entity_types[TREASURE_SLOT], ITEM_TYPE_BASE + 0x1D);
    assert!(
        treasure[e("hidden")] == 1.0
            && treasure[e("is_treasure")] == 1.0
            && treasure[e("active")] == 0.0
    );
    assert!(approx(treasure[e("x")], 0x80 as f64 / 255.0));
    assert!(approx(treasure[e("y")], (0x70 - 64) as f64 / 176.0));

    let rs = &obs.room_state;
    assert!(rs[r("door_S_bomb")] == 1.0 && rs[r("door_S_is_open")] == 0.0);
    assert!(rs[r("door_W_key")] == 1.0 && rs[r("door_W_is_open")] == 1.0);
    assert!(rs[r("door_N_none")] == 1.0 && rs[r("door_E_open")] == 1.0);
    assert!(rs[r("trigger_all_dead_item")] == 1.0 && rs[r("item_hidden_until_clear")] == 1.0);
    assert!(approx(rs[r("enemies_remaining")], 3.0 / 8.0));

    assert!(grid_all(&obs, 18..22, 15..17, TileClass::Bombable));
    assert!(
        grid_all(&obs, 10..12, 0..4, TileClass::DoorW)
            && grid_all(&obs, 10..12, 28..32, TileClass::DoorE)
    );
    assert!(grid_all(&obs, 0..4, 15..17, TileClass::Wall));
    assert!(
        obs.grid[8][8] == TileClass::Block as u8 && obs.grid[4][4] == TileClass::Walkable as u8
    );
    assert_eq!(obs.history[h("entered_from_W")], 1.0);
    assert_eq!(
        s.state().location(),
        0x44,
        "the builder's room state is 0x44"
    );
}

#[test]
fn push_block_in_grid_and_room_state() {
    let Some(mut s) = session() else { return };
    // conftest.load_and_place: settle the room, then put Link just N of the push block.
    load(&mut s, "1_22e_0");
    for _ in 0..30 {
        if s.act(Action::parse("WAIT 8").unwrap())
            .unwrap()
            .interrupt
            .is_none()
        {
            break;
        }
    }
    let state = s.apply_ram(&[(Var::LinkX, 96), (Var::LinkY, 125)]).clone();
    let mut builder = ObservationBuilder::new(false);
    let obs = builder.reset(&state, &no_objective(), s.mask());
    let mut cells = BTreeSet::new();
    for (row, cols) in obs.grid.iter().enumerate() {
        for (col, &c) in cols.iter().enumerate() {
            if c == TileClass::PushBlock as u8 {
                cells.insert((row, col));
            }
        }
    }
    assert_eq!(
        cells,
        [(10, 12), (10, 13), (11, 12), (11, 13)]
            .into_iter()
            .collect()
    );
    assert_eq!(obs.grid[10][16], TileClass::Stairs as u8);
    assert!(
        obs.room_state[r("push_block_pending")] == 1.0
            && obs.room_state[r("push_block_present")] == 1.0
    );
    assert!(obs.entity_types[10] == 0x68 && obs.entity_features[10][e("is_tile_object")] == 1.0);

    step(&mut s, &mut builder, "MOVE S 32");
    let (_, obs) = step(&mut s, &mut builder, "WAIT 32");
    assert!(
        !obs.grid
            .iter()
            .flatten()
            .any(|&c| c == TileClass::PushBlock as u8)
    );
    assert!(
        obs.room_state[r("push_block_done")] == 1.0
            && obs.room_state[r("push_block_pending")] == 0.0
    );
}

#[test]
fn start_state_link_and_room() {
    let Some(mut s) = session() else { return };
    let (_, obs) = reset(&mut s, "start");
    assert_eq!(obs.room_id, room_index(0, false, 0x77));
    let link = &obs.link;
    assert!(link[l("sword_0")] == 1.0 && link[l("beams_ready")] == 0.0);
    assert!(approx(link[l("x")], 120.0 / 255.0) && approx(link[l("y")], (141.0 - 64.0) / 176.0));
    let state = s.state();
    assert!(approx(
        link[l("hp_fraction")],
        state.health() / state.max_health() as f64
    ));
    assert!(approx(link[l("max_hp")], 3.0 / 16.0));
    assert!(obs.room_state[r("overworld")] == 1.0 && obs.entity_types.iter().all(|&t| t == 0));
    let caves = obs
        .grid
        .iter()
        .flatten()
        .filter(|&&c| c == TileClass::Cave as u8)
        .count();
    assert_eq!(caves, 4, "$F3 pair over $24 pair at the sword cave mouth");
}

#[test]
fn key_room_treasure() {
    let Some(mut s) = session() else { return };
    let (_, obs) = reset(&mut s, "1_72e_0");
    assert_eq!(obs.entity_types[TREASURE_SLOT], ITEM_TYPE_BASE + 0x19);
    assert!(obs.room_state[r("item_is_key")] == 1.0 && obs.room_state[r("item_visible")] == 0.0);
    assert!(approx(obs.room_state[r("enemies_unspawned")], 3.0 / 8.0));
    for f in &obs.entity_features[..3] {
        assert!(f[e("spawning")] == 1.0 && f[e("hurts_on_touch")] == 0.0);
    }
}

#[test]
fn sword_cave_exposes_ware_and_fires() {
    let Some(mut s) = session() else { return };
    let (_, obs) = reset(&mut s, "0_77cave_0");
    let (types, feats) = (&obs.entity_types, &obs.entity_features);
    assert_eq!(types[..3], [0x6A, 0x40, 0x40]);
    assert_eq!(feats[0][e("hurts_on_touch")], 0.0);
    assert!(feats[1..3].iter().all(|f| f[e("hurts_on_touch")] == 1.0));
    assert!(feats[..3].iter().all(|f| f[e("killable")] == 0.0));
    assert_eq!(types[TREASURE_SLOT], ITEM_TYPE_BASE + 0x01);
    assert!(approx(feats[TREASURE_SLOT][e("x")], 0x78 as f64 / 255.0));
    assert!(approx(
        feats[TREASURE_SLOT][e("y")],
        (0x98 - 64) as f64 / 176.0
    ));
    assert!(obs.room_state[r("in_cave")] == 1.0 && obs.room_state[r("cave_ware_1")] == 1.0);
}

#[test]
fn velocity_is_position_delta() {
    let Some(mut s) = session() else { return };
    let (mut builder, obs) = reset(&mut s, "1_44w_0");
    assert!(velocities(&obs).iter().flatten().all(|&v| v == 0.0));
    // Monsters don't move inside their spawn clouds.
    let (_, obs) = wait_until_spawned(&mut s, &mut builder, 3);
    let before: Vec<[f32; 2]> = obs.entity_features[..3]
        .iter()
        .map(|f| [f[e("x")], f[e("y")]])
        .collect();
    let (change, obs) = step(&mut s, &mut builder, "WAIT 16");
    let velocity = &velocities(&obs)[..3];
    assert!(velocity.iter().flatten().map(|v| v.abs()).sum::<f32>() > 0.0);
    for (i, f) in obs.entity_features[..3].iter().enumerate() {
        for (axis, (key, scale)) in [("x", 255.0f64), ("y", 176.0)].into_iter().enumerate() {
            let moved_px = (f[e(key)] - before[i][axis]) as f64 * scale;
            let want = (moved_px / (change.frames as f64 * 4.0)).clamp(-1.0, 1.0);
            assert!(
                (velocity[i][axis] as f64 - want).abs() <= 1e-5,
                "slot {i} {key}: {} vs {want}",
                velocity[i][axis]
            );
        }
    }

    let hist = &obs.history;
    assert!(
        hist[h("kind_WAIT")] == 1.0
            && hist[h("duration_16")] == 1.0
            && hist[h("has_last_action")] == 1.0
    );
    for d in ["N", "S", "W", "E", "NW", "NE", "SW", "SE"] {
        assert_eq!(hist[h(&format!("dir_{d}"))], 0.0);
    }
    assert!(approx(
        hist[h("frames")],
        (change.frames as f64 / 64.0).min(1.0)
    ));
}

#[test]
fn room_change_resets_velocity_and_records_entry() {
    let Some(mut s) = session() else { return };
    let (mut builder, mut obs) = reset(&mut s, "start");
    for _ in 0..30 {
        let (change, next) = step(&mut s, &mut builder, "MOVE E 32");
        obs = next;
        if change.changed_location() {
            break;
        }
    }
    assert_eq!(s.state().location(), 0x78);
    assert!(obs.history[h("room_changed")] == 1.0 && obs.history[h("entered_from_W")] == 1.0);
    assert!(velocities(&obs).iter().flatten().all(|&v| v == 0.0));
    assert!(obs.history[h("dir_E")] == 1.0 && obs.history[h("kind_MOVE")] == 1.0);
}

#[test]
fn objective_block() {
    let Some(mut s) = session() else { return };
    let state = load(&mut s, "1_44w_0");
    let objective = Objective {
        kind: ObjectiveKind::Move,
        next_rooms: [MapLocation::new(1, 0x45, false)].into_iter().collect(),
        ..no_objective()
    };

    let off = ObservationBuilder::new(false).reset(&state, &objective, s.mask());
    assert!(off.objective.iter().all(|&x| x == 0.0));

    let on = ObservationBuilder::new(true).reset(&state, &objective, s.mask());
    assert!(on.objective[o("next_room_E")] == 1.0 && on.objective[o("objective_MOVE")] == 1.0);
    assert_eq!(on.objective.iter().sum::<f32>(), 2.0);
    for ((key, a), (_, b)) in obs_values(&off).into_iter().zip(obs_values(&on)) {
        if key != "objective" {
            assert_eq!(a, b, "{key}");
        }
    }
}
