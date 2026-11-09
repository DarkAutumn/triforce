//! `Policy::evaluate` behaviour on a small network with deterministic pseudo-random weights: the masked heads'
//! rules (forced kind, heads that don't apply), scoring vs sampling, and that a row's bits don't depend on what the
//! scratch memory evaluated before. Parity with torch is tested from Python (tests/test_native_policy.py).

use policy::noise::keyed_noise;
use policy::{Input, ModelConfig, Pick, Policy, Tensor};
use std::collections::HashMap;

const SLOTS: usize = 3;
const FEATURES: usize = 5;
const ROWS: usize = 6;
const COLS: usize = 8;
const KINDS: usize = 3;
const DIRS: usize = 4;
const DURS: usize = 2;
const MASK: usize = KINDS * (1 + DIRS + DURS);

fn config() -> ModelConfig {
    ModelConfig::from_json(&format!(
        r#"{{"room_embed_dim": 4, "entity_embed_dim": 4, "entity_d_model": 8, "entity_heads": 2,
            "entity_layers": 1, "entity_ff_dim": 16, "entity_out_dim": 8, "tile_embed_dim": 3,
            "grid_encoder": "conv", "grid_channels": [4, 6, 6], "grid_out_dim": 8, "hidden_dim": 16,
            "obs_spec": {{"room_id": {{"shape": [1], "vocab": 10}},
                          "entity_types": {{"shape": [{SLOTS}], "vocab": 7}},
                          "entity_features": {{"shape": [{SLOTS}, {FEATURES}], "vocab": null}},
                          "grid": {{"shape": [{ROWS}, {COLS}], "vocab": 5}},
                          "link": {{"shape": [3], "vocab": null}},
                          "history": {{"shape": [2], "vocab": null}},
                          "action_mask": {{"shape": [{MASK}], "vocab": null}}}}}}"#
    ))
    .unwrap()
}

/// Every weight the config needs, with values from a fixed hash of (name, index).
fn weights(seed: u64) -> HashMap<String, Tensor> {
    let mut w = HashMap::new();
    let mut add = |name: &str, shape: &[usize]| {
        let n: usize = shape.iter().product();
        let data = (0..n)
            .map(|i| {
                let h = policy::noise::mix(seed ^ (i as u64) ^ name.bytes().fold(7u64, |a, b| a * 31 + b as u64));
                ((h >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.9
            })
            .collect();
        w.insert(name.to_owned(), Tensor { shape: shape.to_vec(), data });
    };
    add("room_embedding.weight", &[10, 4]);
    add("entity_encoder.type_embedding.weight", &[7, 4]);
    add("entity_encoder.input_proj.weight", &[8, FEATURES + 4]);
    add("entity_encoder.input_proj.bias", &[8]);
    let l = "entity_encoder.transformer.layers.0";
    add(&format!("{l}.self_attn.in_proj_weight"), &[24, 8]);
    add(&format!("{l}.self_attn.in_proj_bias"), &[24]);
    add(&format!("{l}.self_attn.out_proj.weight"), &[8, 8]);
    add(&format!("{l}.self_attn.out_proj.bias"), &[8]);
    add(&format!("{l}.linear1.weight"), &[16, 8]);
    add(&format!("{l}.linear1.bias"), &[16]);
    add(&format!("{l}.linear2.weight"), &[8, 16]);
    add(&format!("{l}.linear2.bias"), &[8]);
    for n in ["norm1", "norm2"] {
        add(&format!("{l}.{n}.weight"), &[8]);
        add(&format!("{l}.{n}.bias"), &[8]);
    }
    add("entity_encoder.output_proj.weight", &[8, 8]);
    add("entity_encoder.output_proj.bias", &[8]);
    add("grid_encoder.tile_embedding.weight", &[5, 3]);
    for (i, (cout, cin)) in [(4, 3), (6, 4), (6, 6)].into_iter().enumerate() {
        add(&format!("grid_encoder.body.{}.weight", 2 * i), &[cout, cin, 3, 3]);
        add(&format!("grid_encoder.body.{}.bias", 2 * i), &[cout]);
    }
    // 6×8 → 6×8 → 3×4 → 2×2.
    add("grid_encoder.head.0.weight", &[8, 6 * 2 * 2]);
    add("grid_encoder.head.0.bias", &[8]);
    let features = 4 + 8 + 8 + 2 + 3;
    for tower in ["policy_net", "value_net"] {
        for (i, inp) in [(0, features), (3, 16), (6, 16)] {
            add(&format!("mlp_extractor.{tower}.{i}.weight"), &[16, inp]);
            add(&format!("mlp_extractor.{tower}.{i}.bias"), &[16]);
            add(&format!("mlp_extractor.{tower}.{}.weight", i + 1), &[16]);
            add(&format!("mlp_extractor.{tower}.{}.bias", i + 1), &[16]);
        }
    }
    for (name, out) in [("kind_head", KINDS), ("kind_condition", 16), ("direction_head", DIRS), ("duration_head", DURS)]
    {
        add(&format!("{name}.weight"), &[out, 16]);
        add(&format!("{name}.bias"), &[out]);
    }
    add("kind_embedding.weight", &[KINDS, 16]);
    add("value_head.weight", &[1, 16]);
    add("value_head.bias", &[1]);
    w
}

fn policy() -> Policy {
    Policy::new(config(), weights(1)).unwrap()
}

struct Obs {
    types: Vec<i64>,
    features: Vec<f32>,
    grid: Vec<i64>,
    history: Vec<f32>,
    link: Vec<f32>,
    mask: Vec<bool>,
}

fn obs(seed: u64) -> Obs {
    let f = |i: u64| ((policy::noise::mix(seed * 1000 + i) >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0;
    let mut features: Vec<f32> = (0..(SLOTS * FEATURES) as u64).map(f).collect();
    features[FEATURES * 2] = -1.0; // slot 2 empty
    Obs {
        types: (0..SLOTS as i64).map(|i| (i + seed as i64) % 7).collect(),
        features,
        grid: (0..(ROWS * COLS) as i64).map(|i| (i * 3 + seed as i64) % 5).collect(),
        history: vec![f(900), f(901)],
        link: vec![f(902), f(903), f(904)],
        mask: vec![true; MASK],
    }
}

fn input<'a>(o: &'a Obs, vectors: &'a [&'a [f32]]) -> Input<'a> {
    Input {
        room_id: 3,
        entity_types: &o.types,
        entity_features: &o.features,
        grid: &o.grid,
        vectors,
        mask: &o.mask,
    }
}

#[test]
fn vector_keys_follow_sorted_names() {
    assert_eq!(policy().vector_keys(), ["history", "link"]);
}

#[test]
fn scratch_history_does_not_change_a_row() {
    let p = policy();
    let (a, b) = (obs(1), obs(2));
    let mut noise = vec![0f32; p.noise_size()];
    keyed_noise(5, 0, &mut noise);
    let (va, vb) = ([&a.history[..], &a.link[..]], [&b.history[..], &b.link[..]]);
    let mut used = p.scratch();
    p.evaluate(&input(&a, &va), Pick::Sample(&noise), &mut used).unwrap();
    let after = p.evaluate(&input(&b, &vb), Pick::Sample(&noise), &mut used).unwrap();
    let fresh = p.evaluate(&input(&b, &vb), Pick::Sample(&noise), &mut p.scratch()).unwrap();
    assert_eq!(after, fresh);
    assert_eq!(after.value.to_bits(), fresh.value.to_bits());
}

#[test]
fn scoring_a_sampled_action_gives_its_log_prob() {
    let p = policy();
    let o = obs(3);
    let v = [&o.history[..], &o.link[..]];
    let mut s = p.scratch();
    for step in 0..20 {
        let mut noise = vec![0f32; p.noise_size()];
        keyed_noise(11, step, &mut noise);
        let sampled = p.evaluate(&input(&o, &v), Pick::Sample(&noise), &mut s).unwrap();
        let scored = p.evaluate(&input(&o, &v), Pick::Given(sampled.action), &mut s).unwrap();
        assert_eq!(scored.log_prob.to_bits(), sampled.log_prob.to_bits());
        let total: f32 = sampled.head_log_prob.iter().sum();
        assert!((total - sampled.log_prob).abs() < 1e-6);
    }
}

#[test]
fn one_legal_kind_is_forced_and_an_empty_row_does_not_apply() {
    let p = policy();
    let mut o = obs(4);
    o.mask = vec![false; MASK];
    o.mask[1] = true; // only kind 1
    // kind 1: two legal directions, no legal duration.
    o.mask[KINDS + DIRS + 2] = true;
    o.mask[KINDS + DIRS + 3] = true;
    let v = [&o.history[..], &o.link[..]];
    let out = p.evaluate(&input(&o, &v), Pick::Argmax, &mut p.scratch()).unwrap();
    assert_eq!(out.action[0], 1);
    assert!(!out.head_applies[0] && out.head_log_prob[0] == 0.0 && out.head_entropy[0] == 0.0);
    assert!(out.head_applies[1] && [2, 3].contains(&out.action[1]));
    assert!(!out.head_applies[2] && out.action[2] == 0 && out.head_log_prob[2] == 0.0);
    // Masked directions get no probability.
    let p_dir: f32 = out.head_log_probs[1][..DIRS].iter().map(|v| v.exp()).sum();
    assert!((p_dir - 1.0).abs() < 1e-5);
    assert_eq!(out.head_log_probs[1][0].exp(), 0.0);
}

#[test]
fn bad_inputs_are_errors() {
    let p = policy();
    let mut o = obs(5);
    let v = [&o.history[..], &o.link[..]];
    assert!(p.evaluate(&input(&o, &v[..1]), Pick::Argmax, &mut p.scratch()).is_err());
    o.mask = vec![false; MASK];
    let v = [&o.history[..], &o.link[..]];
    let e = p.evaluate(&input(&o, &v), Pick::Argmax, &mut p.scratch()).unwrap_err();
    assert!(e.0.contains("empty kind mask"));
    let mut w = weights(1);
    w.remove("value_head.bias");
    assert!(Policy::new(config(), w).is_err());
}
