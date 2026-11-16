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

const CONFIG: &str = r#"{"room_embed_dim": 4, "entity_embed_dim": 4, "entity_d_model": 8, "entity_heads": 2,
    "entity_layers": 1, "entity_ff_dim": 16, "entity_out_dim": 8, "tile_embed_dim": 3,
    "grid_channels": [4, 6, 6], "grid_out_dim": 8, "hidden_dim": 16,
    "obs_spec": {"room_id": {"shape": [1], "vocab": 10},
                 "entity_types": {"shape": [SLOTS], "vocab": 7},
                 "entity_features": {"shape": [SLOTS, FEATURES], "vocab": null},
                 "grid": {"shape": [ROWS, COLS], "vocab": 5},
                 "link": {"shape": [3], "vocab": null},
                 "history": {"shape": [2], "vocab": null},
                 "action_mask": {"shape": [MASK], "vocab": null}}}"#;

/// The test config's JSON with `edits` (from, to) applied.
fn config_json(edits: &[(&str, &str)]) -> String {
    let mut json = CONFIG
        .replace("SLOTS", &SLOTS.to_string())
        .replace("FEATURES", &FEATURES.to_string())
        .replace("ROWS", &ROWS.to_string())
        .replace("COLS", &COLS.to_string())
        .replace("MASK", &MASK.to_string());
    for (from, to) in edits {
        assert!(json.contains(from), "{from}");
        json = json.replace(from, to);
    }
    json
}

fn config() -> ModelConfig {
    ModelConfig::from_json(&config_json(&[])).unwrap()
}

/// Every weight the config needs, with values from a fixed hash of (name, index).
fn weights(seed: u64) -> HashMap<String, Tensor> {
    weights_for(seed, KINDS, 3)
}

/// [`weights`] for `kinds` action kinds and a `link` vector of `link` values.
fn weights_for(seed: u64, kinds: usize, link: usize) -> HashMap<String, Tensor> {
    weights_with(seed, kinds, link, FEATURES)
}

/// [`weights_for`] with `features` entity features per slot.
fn weights_with(seed: u64, kinds: usize, link: usize, features: usize) -> HashMap<String, Tensor> {
    let mut w = HashMap::new();
    let mut add = |name: &str, shape: &[usize]| {
        let n: usize = shape.iter().product();
        let data = (0..n)
            .map(|i| {
                let h = policy::noise::mix(
                    seed ^ (i as u64) ^ name.bytes().fold(7u64, |a, b| a.wrapping_mul(31).wrapping_add(b as u64)),
                );
                ((h >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.9
            })
            .collect();
        w.insert(name.to_owned(), Tensor { shape: shape.to_vec(), data });
    };
    add("room_embedding.weight", &[10, 4]);
    add("entity_encoder.type_embedding.weight", &[7, 4]);
    add("entity_encoder.input_proj.weight", &[8, features + 4]);
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
    let features = 4 + 8 + 8 + 2 + link;
    for tower in ["policy_net", "value_net"] {
        for (i, inp) in [(0, features), (3, 16), (6, 16)] {
            add(&format!("mlp_extractor.{tower}.{i}.weight"), &[16, inp]);
            add(&format!("mlp_extractor.{tower}.{i}.bias"), &[16]);
            add(&format!("mlp_extractor.{tower}.{}.weight", i + 1), &[16]);
            add(&format!("mlp_extractor.{tower}.{}.bias", i + 1), &[16]);
        }
    }
    for (name, out) in [("kind_head", kinds), ("kind_condition", 16), ("direction_head", DIRS), ("duration_head", DURS)]
    {
        add(&format!("{name}.weight"), &[out, 16]);
        add(&format!("{name}.bias"), &[out]);
    }
    add("kind_embedding.weight", &[kinds, 16]);
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

#[test]
fn out_of_range_given_actions_are_errors() {
    let p = policy();
    let o = obs(6);
    let v = [&o.history[..], &o.link[..]];
    for action in [[KINDS, 0, 0], [0, DIRS, 0], [0, 0, DURS], [usize::MAX, 0, 0]] {
        let e = p.evaluate(&input(&o, &v), Pick::Given(action), &mut p.scratch()).unwrap_err();
        assert!(e.0.contains("is outside the"), "{action:?}: {}", e.0);
    }
    assert!(p.evaluate(&input(&o, &v), Pick::Given([KINDS - 1, DIRS - 1, DURS - 1]), &mut p.scratch()).is_ok());
}

#[test]
fn non_finite_weights_and_inputs_are_errors() {
    let mut w = weights(1);
    w.get_mut("kind_head.weight").unwrap().data[3] = f32::NAN;
    let e = Policy::new(config(), w).err().expect("a NaN weight is refused");
    assert!(e.0.contains("kind_head.weight") && e.0.contains("non-finite"), "{}", e.0);

    let p = policy();
    let mut o = obs(7);
    o.link[1] = f32::NAN;
    let v = [&o.history[..], &o.link[..]];
    let e = p.evaluate(&input(&o, &v), Pick::Argmax, &mut p.scratch()).unwrap_err();
    assert!(e.0.contains("link") && e.0.contains("non-finite"), "{}", e.0);
    assert!(p.value(&input(&o, &v), &mut p.scratch()).is_err());
    let mut joint = vec![0f32; p.joint_size()];
    assert!(p.joint_log_probs(&input(&o, &v), &mut p.scratch(), &mut joint).is_err());
    let mut o = obs(7);
    o.features[1] = f32::INFINITY;
    let v = [&o.history[..], &o.link[..]];
    assert!(p.evaluate(&input(&o, &v), Pick::Argmax, &mut p.scratch()).is_err());
}

#[test]
fn degenerate_configs_are_errors_not_panics() {
    let no_heads = ModelConfig::from_json(&config_json(&[("\"entity_heads\": 2", "\"entity_heads\": 0")])).unwrap();
    let e = Policy::new(no_heads, weights(1)).err().expect("entity_heads 0 is refused");
    assert!(e.0.contains("unsupported model config"), "{}", e.0);

    let no_convs = ModelConfig::from_json(&config_json(&[("[4, 6, 6]", "[]")])).unwrap();
    let mut w = weights(1);
    w.retain(|name, _| !name.starts_with("grid_encoder.body."));
    let head = Tensor { shape: vec![8, 3 * ROWS * COLS], data: vec![0.1; 8 * 3 * ROWS * COLS] };
    w.insert("grid_encoder.head.0.weight".into(), head);
    let e = Policy::new(no_convs, w).err().expect("an empty grid_channels is refused");
    assert!(e.0.contains("unsupported model config"), "{}", e.0);
}

#[test]
fn joint_log_probs_match_scoring_each_action() {
    let p = policy();
    assert_eq!(p.joint_size(), KINDS * DIRS * DURS);
    for (seed, masked) in [(6, false), (7, true)] {
        let mut o = obs(seed);
        if masked {
            // Kind 0: directions 1 and 3, no duration; kind 2: every direction, duration 1; kind 1 illegal.
            o.mask = vec![false; MASK];
            o.mask[0] = true;
            o.mask[2] = true;
            o.mask[KINDS + 1] = true;
            o.mask[KINDS + 3] = true;
            for d in 0..DIRS {
                o.mask[KINDS + 2 * DIRS + d] = true;
            }
            o.mask[KINDS + KINDS * DIRS + 2 * DURS + 1] = true;
        }
        let v = [&o.history[..], &o.link[..]];
        let mut s = p.scratch();
        let mut joint = vec![0f32; p.joint_size()];
        let value = p.joint_log_probs(&input(&o, &v), &mut s, &mut joint).unwrap();
        let mut total = 0f64;
        for (i, &lp) in joint.iter().enumerate() {
            let action = p.joint_action(i);
            assert_eq!(p.joint_index(action), i);
            if lp == f32::NEG_INFINITY {
                continue;
            }
            total += (lp as f64).exp();
            let scored = p.evaluate(&input(&o, &v), Pick::Given(action), &mut s).unwrap();
            assert_eq!(scored.log_prob.to_bits(), lp.to_bits(), "{action:?}");
            assert_eq!(scored.value.to_bits(), value.to_bits());
        }
        assert!((total - 1.0).abs() < 1e-5, "{total}");
        let legal = joint.iter().filter(|v| v.is_finite()).count();
        assert_eq!(legal, if masked { 2 + DIRS } else { KINDS * DIRS * DURS });
    }
}

/// `evaluate_with_distribution` is `evaluate` plus the whole factored distribution: the output's bits are the
/// same, and each joint action's three entries add up to its `joint_log_probs` value, bit for bit (`-inf` when
/// the action is illegal).
#[test]
fn the_distribution_holds_every_joint_log_prob() {
    let p = policy();
    assert_eq!(p.distribution_size(), MASK);
    let (mut s, mut plain) = (p.scratch(), p.scratch());
    let mut noise = vec![0f32; p.noise_size()];
    let mut dist = vec![0f32; p.distribution_size()];
    let mut joint = vec![0f32; p.joint_size()];
    let mut illegal = 0;
    for seed in 0..50u64 {
        let mut o = obs(seed);
        // A random mask (every other seed keeps all legal) with at least one legal kind.
        if seed % 2 == 1 {
            for (i, m) in o.mask.iter_mut().enumerate() {
                *m = policy::noise::mix(seed * 7919 + i as u64) % 5 < 3;
            }
            o.mask[(seed % KINDS as u64) as usize] = true;
        }
        let v = [&o.history[..], &o.link[..]];
        keyed_noise(seed as i64, 3, &mut noise);
        let pick = match seed % 3 {
            0 => Pick::Argmax,
            1 => Pick::Sample(&noise),
            _ => Pick::Given(p.evaluate(&input(&o, &v), Pick::Argmax, &mut plain).unwrap().action),
        };
        let expected = p.evaluate(&input(&o, &v), pick, &mut plain).unwrap();
        let out = p.evaluate_with_distribution(&input(&o, &v), pick, &mut s, &mut dist).unwrap();
        assert_eq!(format!("{out:?}"), format!("{expected:?}"), "seed {seed}");
        p.joint_log_probs(&input(&o, &v), &mut plain, &mut joint).unwrap();
        for (i, &lp) in joint.iter().enumerate() {
            let [k, d, u] = p.joint_action(i);
            let sum = (dist[k] + dist[KINDS + k * DIRS + d]) + dist[KINDS + KINDS * DIRS + k * DURS + u];
            assert_eq!(sum.to_bits(), lp.to_bits(), "seed {seed} action {:?}: {sum} vs {lp}", [k, d, u]);
            illegal += (lp == f32::NEG_INFINITY) as usize;
        }
        // The taken action's entries are its head log-probs.
        let [k, d, u] = out.action;
        let taken = (dist[k] + dist[KINDS + k * DIRS + d]) + dist[KINDS + KINDS * DIRS + k * DURS + u];
        assert_eq!(taken.to_bits(), out.log_prob.to_bits(), "seed {seed}");
        assert!(dist.iter().all(|v| !v.is_nan()));
    }
    assert!(illegal > 200, "the masks made illegal actions: {illegal}");
    let mut short = vec![0f32; MASK - 1];
    let o = obs(1);
    let v = [&o.history[..], &o.link[..]];
    assert!(p.evaluate_with_distribution(&input(&o, &v), Pick::Argmax, &mut s, &mut short).is_err());
}

/// `Policy::act`: argmax and a plain draw are `evaluate`'s, bit for bit; a rule draws only legal actions the rule
/// keeps, scores them as `Pick::Given` does, and draws the same for the same (job, step).
#[test]
fn act_modes_follow_evaluate_and_their_rules() {
    use policy::{ActMode, Choose};
    let p = policy();
    let mut o = obs(6);
    // kind 0 illegal; kind 2: direction 1 illegal.
    o.mask[0] = false;
    o.mask[KINDS + 2 * DIRS + 1] = false;
    let v = [&o.history[..], &o.link[..]];
    let inp = input(&o, &v);
    let mut s = p.scratch();
    let mut joint = vec![0f32; p.joint_size()];
    p.joint_log_probs(&inp, &mut s, &mut joint).unwrap();
    let argmax = p.evaluate(&inp, Pick::Argmax, &mut s).unwrap();
    let (out, acting) = p.act(&inp, Choose::Argmax, 1, 0, None, &mut s, None).unwrap();
    assert_eq!(out, argmax);
    assert_eq!((acting.log_prob, acting.entropy), (0.0, 0.0));
    let entropy: f32 = joint.iter().filter(|v| v.is_finite()).map(|&v| -v.exp() * v).sum();
    for step in 0..30 {
        let mut noise = vec![0f32; p.noise_size()];
        keyed_noise(9, step, &mut noise);
        let sampled = p.evaluate(&inp, Pick::Sample(&noise), &mut s).unwrap();
        let (out, acting) = p.act(&inp, ActMode::SAMPLE.choose(0, None), 9, step, None, &mut s, None).unwrap();
        assert_eq!(out, sampled);
        assert_eq!(acting.log_prob.to_bits(), out.log_prob.to_bits());
        assert_eq!(acting.entropy.to_bits(), entropy.to_bits());
    }
    let mut dist = vec![0f32; p.distribution_size()];
    for mode in ["floor:0.05", "temperature:0.5+minp:0.3", "topp:0.6", "escape:1+floor:0.1"] {
        let mode: ActMode = mode.parse().unwrap();
        let Choose::Draw(rule) = mode.choose(5, None) else { panic!("{mode} draws when stalled") };
        let mut q = vec![0f64; p.joint_size()];
        rule.probabilities(&joint, &mut q, &mut vec![0; p.joint_size()]);
        let mut seen = vec![false; p.joint_size()];
        for step in 0..200 {
            let (out, acting) = p.act(&inp, mode.choose(5, None), 4, step, None, &mut s, Some(&mut dist)).unwrap();
            let again = p.act(&inp, mode.choose(5, None), 4, step, None, &mut p.scratch(), None).unwrap();
            assert_eq!((out, acting), again, "{mode} step {step}");
            // Replaying the action scores it the same.
            let scored = p.act(&inp, mode.choose(5, None), 4, step, Some(out.action), &mut s, None).unwrap();
            assert_eq!(scored, (out, acting), "{mode} step {step} replayed");
            let index = p.joint_index(out.action);
            assert!(joint[index].is_finite() && q[index] > 0.0, "{mode} drew {:?}", out.action);
            assert_eq!(out.log_prob.to_bits(), joint[index].to_bits());
            assert!((acting.log_prob - q[index].ln() as f32).abs() < 1e-6);
            seen[index] = true;
        }
        // Every kept action with a fair share turns up; nothing else does.
        for (i, &kept) in q.iter().enumerate() {
            if kept > 0.05 {
                assert!(seen[i], "{mode}: joint action {i} (q {kept}) never drawn");
            }
        }
    }
}

/// The bits of everything an evaluation reports.
fn bits(o: &policy::Output) -> Vec<u32> {
    let mut v = vec![o.value.to_bits(), o.log_prob.to_bits(), o.entropy.to_bits()];
    v.extend(o.action.iter().map(|&a| a as u32));
    v.extend(o.head_log_prob.iter().chain(&o.head_entropy).map(|x| x.to_bits()));
    v
}

/// A first-layer input column whose weights are all zero is dropped (`Policy::live`): a model widened with zero
/// columns in the middle of its features computes bit for bit what the narrow one does, whatever the new inputs
/// hold (without the drop, the later columns would sum in other lanes).
#[test]
fn zero_input_columns_change_no_bit() {
    let narrow = policy();
    // Every other weight is the same hash of (name, index): only the first layers' columns move. The narrow
    // features: room 4, entities 8, grid 8, history 2, link 3; the wide link has two columns after its first, at
    // features 23 and 24.
    let mut w = weights_for(1, KINDS, 5);
    let source = weights(1);
    for tower in ["policy_net", "value_net"] {
        let name = format!("mlp_extractor.{tower}.0.weight");
        let old = &source[&name].data;
        let wide = w.get_mut(&name).unwrap();
        for (r, row) in wide.data.chunks_exact_mut(27).enumerate() {
            let src = &old[r * 25..(r + 1) * 25];
            row[..23].copy_from_slice(&src[..23]);
            row[23..25].fill(0.0);
            row[25..].copy_from_slice(&src[23..]);
        }
    }
    let wide = Policy::new(ModelConfig::from_json(&config_json(&[(r#""shape": [3]"#, r#""shape": [5]"#)])).unwrap(), w)
        .unwrap();
    for seed in 0..20 {
        let o = obs(seed);
        let link = [o.link[0], 0.75, -2.5, o.link[1], o.link[2]];
        let (vn, vw) = ([&o.history[..], &o.link[..]], [&o.history[..], &link[..]]);
        let mut noise = vec![0f32; narrow.noise_size()];
        keyed_noise(seed as i64, 2, &mut noise);
        for pick in [Pick::Argmax, Pick::Sample(&noise)] {
            let a = narrow.evaluate(&input(&o, &vn), pick, &mut narrow.scratch()).unwrap();
            let b = wide.evaluate(&input(&o, &vw), pick, &mut wide.scratch()).unwrap();
            assert_eq!(bits(&a), bits(&b), "seed {seed} {pick:?}");
        }
    }
}

/// Entity features added with zero `input_proj` columns (schema 6, `triforce convert-model`) are dropped as the
/// towers' columns are (`Policy::entity_live`): the widened model computes bit for bit what the narrow one does,
/// whatever the new columns hold, for 1, 3 or 8 features added after a slot's features (before the type embedding,
/// where the converter puts them) or among them. Without the drop the later columns sum in other lanes of
/// `kernels::dot` and the last bits move (round 26: one added column changed 224 of 240 decisions of goriya).
#[test]
fn zero_entity_columns_change_no_bit() {
    let narrow = policy();
    assert_eq!(narrow.dropped_inputs(), [0, 0], "a model without zero columns drops none");
    let source = weights(1);
    let old = &source["entity_encoder.input_proj.weight"].data;
    for (k, at) in [(1, FEATURES), (3, FEATURES), (8, FEATURES), (1, 2), (3, 1)] {
        let features = FEATURES + k;
        let (n_old, n_new) = (FEATURES + 4, features + 4);
        let mut w = weights_with(1, KINDS, 3, features);
        let proj = w.get_mut("entity_encoder.input_proj.weight").unwrap();
        for (r, row) in proj.data.chunks_exact_mut(n_new).enumerate() {
            let src = &old[r * n_old..(r + 1) * n_old];
            row[..at].copy_from_slice(&src[..at]);
            row[at..at + k].fill(0.0);
            row[at + k..].copy_from_slice(&src[at..]);
        }
        let shape = |n: usize| format!(r#""shape": [{SLOTS}, {n}]"#);
        let config = config_json(&[(&shape(FEATURES), &shape(features))]);
        let wide = Policy::new(ModelConfig::from_json(&config).unwrap(), w).unwrap();
        assert_eq!(wide.dropped_inputs(), [0, k], "{k} at {at}");
        for seed in 0..20 {
            let o = obs(seed);
            let new = |i: usize| {
                ((policy::noise::mix(seed * 7919 + i as u64) >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            };
            let mut wide_features = Vec::with_capacity(SLOTS * features);
            for (slot, row) in o.features.chunks_exact(FEATURES).enumerate() {
                wide_features.extend_from_slice(&row[..at]);
                wide_features.extend((0..k).map(|j| new(slot * 16 + j)));
                wide_features.extend_from_slice(&row[at..]);
            }
            let ow = Obs {
                types: o.types.clone(),
                features: wide_features,
                grid: o.grid.clone(),
                history: o.history.clone(),
                link: o.link.clone(),
                mask: o.mask.clone(),
            };
            let v = [&o.history[..], &o.link[..]];
            let mut noise = vec![0f32; narrow.noise_size()];
            keyed_noise(seed as i64, 2, &mut noise);
            for pick in [Pick::Argmax, Pick::Sample(&noise)] {
                let a = narrow.evaluate(&input(&o, &v), pick, &mut narrow.scratch()).unwrap();
                let b = wide.evaluate(&input(&ow, &v), pick, &mut wide.scratch()).unwrap();
                assert_eq!(bits(&a), bits(&b), "{k} at {at}, seed {seed} {pick:?}");
            }
            let (mut ja, mut jb) = (vec![0f32; narrow.joint_size()], vec![0f32; wide.joint_size()]);
            let va = narrow.joint_log_probs(&input(&o, &v), &mut narrow.scratch(), &mut ja).unwrap();
            let vb = wide.joint_log_probs(&input(&ow, &v), &mut wide.scratch(), &mut jb).unwrap();
            assert_eq!(va.to_bits(), vb.to_bits(), "{k} at {at}, seed {seed}: value");
            let joint_bits = |j: &[f32]| j.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(joint_bits(&ja), joint_bits(&jb), "{k} at {at}, seed {seed}: joint log-probs");
        }
    }
}

/// Kinds added after the eighth (`LEAD_KINDS`) take their plain-draw noise after the direction and duration heads',
/// so a model widened with new kinds, masked, draws the same actions from the same `keyed_noise` (job, step) as the
/// one it was widened from.
#[test]
fn added_kinds_draw_as_before_while_masked() {
    let mask = |kinds: usize| kinds * (1 + DIRS + DURS);
    let config = |kinds: usize| {
        let edit = (format!(r#""action_mask": {{"shape": [{MASK}]"#), format!(r#""action_mask": {{"shape": [{}]"#, mask(kinds)));
        ModelConfig::from_json(&config_json(&[(&edit.0, &edit.1)])).unwrap()
    };
    let old = weights_for(3, 8, 3);
    let mut new = weights_for(3, 11, 3);
    for (name, t) in &old {
        if !name.starts_with("kind_head") && name != "kind_embedding.weight" {
            new.insert(name.clone(), t.clone());
        }
    }
    // The old kinds' rows kept; the new rows are whatever the hash gave.
    for name in ["kind_head.weight", "kind_head.bias", "kind_embedding.weight"] {
        let n = old[name].data.len();
        new.get_mut(name).unwrap().data[..n].copy_from_slice(&old[name].data);
    }
    let (p8, p11) = (Policy::new(config(8), old).unwrap(), Policy::new(config(11), new).unwrap());
    for seed in 0..40u64 {
        let mut o = obs(seed);
        let m8: Vec<bool> = (0..mask(8)).map(|i| (policy::noise::mix(seed * 77 + i as u64) & 3) != 0).collect();
        let mut m11 = vec![false; mask(11)];
        m11[..8].copy_from_slice(&m8[..8]);
        m11[11..11 + 8 * DIRS].copy_from_slice(&m8[8..8 + 8 * DIRS]);
        m11[11 + 11 * DIRS..11 + 11 * DIRS + 8 * DURS].copy_from_slice(&m8[8 + 8 * DIRS..]);
        if !m8[..8].contains(&true) {
            continue;
        }
        let v = [&o.history[..], &o.link[..]];
        o.mask = m8;
        let a = p8.act(&input(&o, &v), policy::ActMode::SAMPLE.choose(0, None), seed as i64, 4, None, &mut p8.scratch(), None);
        let rule: policy::ActMode = "temperature:0.5".parse().unwrap();
        let ra = p8.act(&input(&o, &v), rule.choose(0, None), seed as i64, 4, None, &mut p8.scratch(), None).unwrap();
        o.mask = m11;
        let b = p11.act(&input(&o, &v), policy::ActMode::SAMPLE.choose(0, None), seed as i64, 4, None, &mut p11.scratch(), None);
        let rb = p11.act(&input(&o, &v), rule.choose(0, None), seed as i64, 4, None, &mut p11.scratch(), None).unwrap();
        let (a, b) = (a.unwrap(), b.unwrap());
        assert_eq!(bits(&a.0), bits(&b.0), "seed {seed}: sample");
        assert_eq!((a.1.log_prob.to_bits(), a.1.entropy.to_bits()), (b.1.log_prob.to_bits(), b.1.entropy.to_bits()));
        assert_eq!(bits(&ra.0), bits(&rb.0), "seed {seed}: temperature");
        assert_eq!(ra.1.log_prob.to_bits(), rb.1.log_prob.to_bits());
    }
}
