//! The specialist policy network (`triforce/model.py` `ZeldaPolicy`) evaluated on the CPU in Rust.
//!
//! Python trains the network; Rust acts with it (docs/rust-actor.md). [`Policy::new`] takes the checkpoint's
//! `model_config` (with its `obs_spec`) and the torch `state_dict` as named float32 arrays, and only supports the
//! current architecture. [`Policy::evaluate`] runs one observation: the policy and value towers, the factored
//! masked heads (kind → kind-conditioned direction and duration), and picks the action by Gumbel-argmax with the
//! given noise, by argmax, or scores a given action.
//!
//! Determinism: one observation's outputs are a pure function of the weights and that observation (and noise),
//! bit for bit, on the same binary (`kernels`). Python's torch forward agrees within float tolerance.

pub mod kernels;
pub mod noise;

use kernels::{argmax, dot, layer_norm, linear, masked_log_softmax, relu, tanh};
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap};
use std::fmt;

/// `model.py` `MASKED_LOGIT`.
pub const MASKED_LOGIT: f32 = -1e9;
/// Largest head size the fixed output arrays hold.
pub const MAX_HEAD: usize = 16;

/// The observation keys with their own encoders; every other key but the mask is a float vector.
const ROOM_KEY: &str = "room_id";
const ENTITY_TYPES_KEY: &str = "entity_types";
const ENTITY_FEATURES_KEY: &str = "entity_features";
const GRID_KEY: &str = "grid";
const MASK_KEY: &str = "action_mask";
const SPECIAL_KEYS: [&str; 5] = [ROOM_KEY, ENTITY_TYPES_KEY, ENTITY_FEATURES_KEY, GRID_KEY, MASK_KEY];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyError(pub String);

impl fmt::Display for PolicyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for PolicyError {}

fn err<T>(msg: impl Into<String>) -> Result<T, PolicyError> {
    Err(PolicyError(msg.into()))
}

/// One observation key of `model_config.obs_spec`.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct KeySpec {
    pub shape: Vec<usize>,
    pub vocab: Option<usize>,
}

/// `model.py` `ModelConfig` (a checkpoint's `model_config`).
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    pub room_embed_dim: usize,
    pub entity_embed_dim: usize,
    pub entity_d_model: usize,
    pub entity_heads: usize,
    pub entity_layers: usize,
    pub entity_ff_dim: usize,
    pub entity_out_dim: usize,
    pub tile_embed_dim: usize,
    pub grid_encoder: String,
    pub grid_channels: Vec<usize>,
    pub grid_out_dim: usize,
    pub hidden_dim: usize,
    pub obs_spec: BTreeMap<String, KeySpec>,
}

impl ModelConfig {
    pub fn from_json(text: &str) -> Result<ModelConfig, PolicyError> {
        serde_json::from_str(text).map_err(|e| PolicyError(format!("model_config: {e}")))
    }

    fn key(&self, name: &str) -> Result<&KeySpec, PolicyError> {
        self.obs_spec
            .get(name)
            .ok_or_else(|| PolicyError(format!("model_config.obs_spec has no '{name}'")))
    }

    /// The float vector keys in the order `model.py` concatenates them (sorted names).
    pub fn vector_keys(&self) -> Vec<String> {
        self.obs_spec
            .keys()
            .filter(|k| !SPECIAL_KEYS.contains(&k.as_str()))
            .cloned()
            .collect()
    }
}

/// A float32 tensor from torch's `state_dict`.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

struct Weights(HashMap<String, Tensor>);

impl Weights {
    fn take(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>, PolicyError> {
        let t = self.0.remove(name).ok_or_else(|| PolicyError(format!("missing weight '{name}'")))?;
        if t.shape != shape {
            return err(format!("weight '{name}' has shape {:?}, expected {shape:?}", t.shape));
        }
        if t.data.len() != shape.iter().product::<usize>() {
            return err(format!("weight '{name}' has {} values for shape {shape:?}", t.data.len()));
        }
        Ok(t.data)
    }

    fn linear(&mut self, prefix: &str, out: usize, inp: usize) -> Result<Linear, PolicyError> {
        Ok(Linear {
            w: self.take(&format!("{prefix}.weight"), &[out, inp])?,
            b: self.take(&format!("{prefix}.bias"), &[out])?,
            out,
            inp,
        })
    }

    fn norm(&mut self, prefix: &str, n: usize) -> Result<Norm, PolicyError> {
        Ok(Norm {
            gamma: self.take(&format!("{prefix}.weight"), &[n])?,
            beta: self.take(&format!("{prefix}.bias"), &[n])?,
        })
    }
}

struct Linear {
    w: Vec<f32>,
    b: Vec<f32>,
    out: usize,
    inp: usize,
}

impl Linear {
    #[inline]
    fn apply(&self, x: &[f32], y: &mut [f32]) {
        debug_assert_eq!(x.len(), self.inp);
        linear(&self.w, &self.b, x, &mut y[..self.out]);
    }
}

struct Norm {
    gamma: Vec<f32>,
    beta: Vec<f32>,
}

struct Embedding {
    w: Vec<f32>,
    dim: usize,
    vocab: usize,
}

impl Embedding {
    fn row(&self, id: i64) -> &[f32] {
        let i = (id.max(0) as usize).min(self.vocab - 1);
        &self.w[i * self.dim..(i + 1) * self.dim]
    }
}

struct EncoderLayer {
    in_proj: Linear,
    out_proj: Linear,
    linear1: Linear,
    linear2: Linear,
    norm1: Norm,
    norm2: Norm,
}

struct Conv {
    w: Vec<f32>,
    b: Vec<f32>,
    cin: usize,
    cout: usize,
    stride: usize,
    h_in: usize,
    w_in: usize,
    h_out: usize,
    w_out: usize,
}

struct Tower {
    layers: [Linear; 3],
    norms: [Norm; 3],
    last_tanh: bool,
}

impl Tower {
    fn apply(&self, x: &[f32], a: &mut [f32], b: &mut [f32]) {
        self.layers[0].apply(x, a);
        layer_norm(a, &self.norms[0].gamma, &self.norms[0].beta);
        relu(a);
        self.layers[1].apply(a, b);
        layer_norm(b, &self.norms[1].gamma, &self.norms[1].beta);
        relu(b);
        self.layers[2].apply(b, a);
        layer_norm(a, &self.norms[2].gamma, &self.norms[2].beta);
        if self.last_tanh {
            tanh(a);
        } else {
            relu(a);
        }
    }
}

/// How [`Policy::evaluate`] picks the action.
#[derive(Clone, Copy, Debug)]
pub enum Pick<'a> {
    /// Each head's argmax of log-probs plus this Gumbel noise (kind, direction, duration columns;
    /// `Policy::noise_size` values, e.g. from [`noise::keyed_noise`]).
    Sample(&'a [f32]),
    /// Each head's most likely entry.
    Argmax,
    /// Score this (kind, direction, duration).
    Given([usize; 3]),
}

/// One observation, as slices of the pool's row (shapes from `model_config.obs_spec`).
#[derive(Clone, Copy, Debug)]
pub struct Input<'a> {
    pub room_id: i64,
    /// `[slots]`.
    pub entity_types: &'a [i64],
    /// `[slots, features]` row-major.
    pub entity_features: &'a [f32],
    /// `[rows, cols]` row-major tile classes.
    pub grid: &'a [i64],
    /// The float vectors in [`Policy::vector_keys`] order.
    pub vectors: &'a [&'a [f32]],
    /// The flat action mask (kinds, then each kind's directions, then each kind's durations).
    pub mask: &'a [bool],
}

/// What one evaluation produces.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Output {
    /// (kind, direction, duration); a head that doesn't apply picks 0.
    pub action: [usize; 3],
    /// Joint log-prob over the heads that apply.
    pub log_prob: f32,
    /// Summed entropy over the heads that apply.
    pub entropy: f32,
    pub value: f32,
    /// Per head (kind, direction, duration): log-probs (direction/duration given the chosen kind; masked entries
    /// are `MASKED_LOGIT`-small), whether the head applies, its entropy and its chosen log-prob (0 where it
    /// doesn't apply).
    pub head_log_probs: [[f32; MAX_HEAD]; 3],
    pub head_applies: [bool; 3],
    pub head_entropy: [f32; 3],
    pub head_log_prob: [f32; 3],
}

/// Per-thread working memory (no allocation per call).
pub struct Scratch {
    tokens: Vec<f32>,
    qkv: Vec<f32>,
    attn: Vec<f32>,
    heads: Vec<f32>,
    ff: Vec<f32>,
    scores: Vec<f32>,
    token_in: Vec<f32>,
    pooled: Vec<f32>,
    grid_a: Vec<f32>,
    grid_b: Vec<f32>,
    patch: Vec<f32>,
    features: Vec<f32>,
    tower_a: Vec<f32>,
    tower_b: Vec<f32>,
    value_a: Vec<f32>,
    value_b: Vec<f32>,
    conditioned: Vec<f32>,
    logits: Vec<f32>,
    empty: Vec<bool>,
    masked: Vec<bool>,
}

/// The network.
pub struct Policy {
    config: ModelConfig,
    vector_keys: Vec<String>,
    vector_sizes: Vec<usize>,
    hash: u64,
    slots: usize,
    entity_features: usize,
    grid_rows: usize,
    grid_cols: usize,
    room_embedding: Embedding,
    type_embedding: Embedding,
    input_proj: Linear,
    layers: Vec<EncoderLayer>,
    output_proj: Linear,
    tile_embedding: Embedding,
    convs: Vec<Conv>,
    grid_head: Linear,
    policy_tower: Tower,
    value_tower: Tower,
    kind_head: Linear,
    kind_condition: Linear,
    kind_embedding: Embedding,
    direction_head: Linear,
    duration_head: Linear,
    value_head: Linear,
    kinds: usize,
    directions: usize,
    durations: usize,
}

impl Policy {
    /// Builds the network from `model_config` and the `state_dict` (every weight must be used exactly once).
    pub fn new(config: ModelConfig, weights: HashMap<String, Tensor>) -> Result<Policy, PolicyError> {
        if config.grid_encoder != "conv" {
            return err(format!("grid_encoder '{}' is not supported (only 'conv')", config.grid_encoder));
        }
        let hash = weights_hash(&config, &weights);
        let mut w = Weights(weights);
        let room = config.key(ROOM_KEY)?;
        let types = config.key(ENTITY_TYPES_KEY)?;
        let feats = config.key(ENTITY_FEATURES_KEY)?;
        let grid = config.key(GRID_KEY)?;
        let (Some(room_vocab), Some(type_vocab), Some(tile_vocab)) = (room.vocab, types.vocab, grid.vocab) else {
            return err("room_id, entity_types and grid need a vocab");
        };
        let [slots, entity_features] = feats.shape[..] else {
            return err("entity_features must be [slots, features]");
        };
        let [grid_rows, grid_cols] = grid.shape[..] else {
            return err("grid must be [rows, cols]");
        };
        let vector_keys = config.vector_keys();
        let mut vector_sizes = Vec::new();
        for key in &vector_keys {
            let spec = config.key(key)?;
            if spec.vocab.is_some() || spec.shape.len() != 1 {
                return err(format!("observation key '{key}' must be a 1-D float vector"));
            }
            vector_sizes.push(spec.shape[0]);
        }

        let room_embedding = Embedding {
            w: w.take("room_embedding.weight", &[room_vocab, config.room_embed_dim])?,
            dim: config.room_embed_dim,
            vocab: room_vocab,
        };
        let d = config.entity_d_model;
        if d % config.entity_heads != 0 {
            return err("entity_d_model must be a multiple of entity_heads");
        }
        let type_embedding = Embedding {
            w: w.take("entity_encoder.type_embedding.weight", &[type_vocab, config.entity_embed_dim])?,
            dim: config.entity_embed_dim,
            vocab: type_vocab,
        };
        let input_proj = w.linear("entity_encoder.input_proj", d, entity_features + config.entity_embed_dim)?;
        let mut layers = Vec::new();
        for i in 0..config.entity_layers {
            let p = format!("entity_encoder.transformer.layers.{i}");
            layers.push(EncoderLayer {
                in_proj: Linear {
                    w: w.take(&format!("{p}.self_attn.in_proj_weight"), &[3 * d, d])?,
                    b: w.take(&format!("{p}.self_attn.in_proj_bias"), &[3 * d])?,
                    out: 3 * d,
                    inp: d,
                },
                out_proj: w.linear(&format!("{p}.self_attn.out_proj"), d, d)?,
                linear1: w.linear(&format!("{p}.linear1"), config.entity_ff_dim, d)?,
                linear2: w.linear(&format!("{p}.linear2"), d, config.entity_ff_dim)?,
                norm1: w.norm(&format!("{p}.norm1"), d)?,
                norm2: w.norm(&format!("{p}.norm2"), d)?,
            });
        }
        let output_proj = w.linear("entity_encoder.output_proj", config.entity_out_dim, d)?;

        let tile_embedding = Embedding {
            w: w.take("grid_encoder.tile_embedding.weight", &[tile_vocab, config.tile_embed_dim])?,
            dim: config.tile_embed_dim,
            vocab: tile_vocab,
        };
        let mut convs = Vec::new();
        let (mut cin, mut h, mut wd) = (config.tile_embed_dim, grid_rows, grid_cols);
        for (i, &cout) in config.grid_channels.iter().enumerate() {
            let stride = if i == 0 { 1 } else { 2 };
            let (h_out, w_out) = ((h + 2 - 3) / stride + 1, (wd + 2 - 3) / stride + 1);
            let name = format!("grid_encoder.body.{}", 2 * i);
            convs.push(Conv {
                w: w.take(&format!("{name}.weight"), &[cout, cin, 3, 3])?,
                b: w.take(&format!("{name}.bias"), &[cout])?,
                cin,
                cout,
                stride,
                h_in: h,
                w_in: wd,
                h_out,
                w_out,
            });
            (cin, h, wd) = (cout, h_out, w_out);
        }
        let flat = cin * h * wd;
        let grid_head = w.linear("grid_encoder.head.0", config.grid_out_dim, flat)?;

        let features = config.room_embed_dim
            + config.entity_out_dim
            + config.grid_out_dim
            + vector_sizes.iter().sum::<usize>();
        let hidden = config.hidden_dim;
        let tower = |w: &mut Weights, name: &str, last_tanh: bool| -> Result<Tower, PolicyError> {
            let p = format!("mlp_extractor.{name}");
            Ok(Tower {
                layers: [
                    w.linear(&format!("{p}.0"), hidden, features)?,
                    w.linear(&format!("{p}.3"), hidden, hidden)?,
                    w.linear(&format!("{p}.6"), hidden, hidden)?,
                ],
                norms: [
                    w.norm(&format!("{p}.1"), hidden)?,
                    w.norm(&format!("{p}.4"), hidden)?,
                    w.norm(&format!("{p}.7"), hidden)?,
                ],
                last_tanh,
            })
        };
        let policy_tower = tower(&mut w, "policy_net", true)?;
        let value_tower = tower(&mut w, "value_net", false)?;

        let kinds = w.0.get("kind_head.weight").map(|t| t.shape[0]).unwrap_or(0);
        let directions = w.0.get("direction_head.weight").map(|t| t.shape[0]).unwrap_or(0);
        let durations = w.0.get("duration_head.weight").map(|t| t.shape[0]).unwrap_or(0);
        if [kinds, directions, durations].iter().any(|&n| n == 0 || n > MAX_HEAD) {
            return err(format!("head sizes {kinds}/{directions}/{durations} must be 1..={MAX_HEAD}"));
        }
        let mask_size = config.key(MASK_KEY)?.shape.first().copied().unwrap_or(0);
        if mask_size != kinds * (1 + directions + durations) {
            return err(format!("action_mask has {mask_size} entries for {kinds}/{directions}/{durations} heads"));
        }
        let kind_head = w.linear("kind_head", kinds, hidden)?;
        let kind_condition = w.linear("kind_condition", hidden, hidden)?;
        let kind_embedding = Embedding {
            w: w.take("kind_embedding.weight", &[kinds, hidden])?,
            dim: hidden,
            vocab: kinds,
        };
        let direction_head = w.linear("direction_head", directions, hidden)?;
        let duration_head = w.linear("duration_head", durations, hidden)?;
        let value_head = w.linear("value_head", 1, hidden)?;
        if let Some(extra) = w.0.keys().next() {
            return err(format!("unused weight '{extra}' (architecture mismatch)"));
        }
        Ok(Policy {
            config,
            vector_keys,
            vector_sizes,
            hash,
            slots,
            entity_features,
            grid_rows,
            grid_cols,
            room_embedding,
            type_embedding,
            input_proj,
            layers,
            output_proj,
            tile_embedding,
            convs,
            grid_head,
            policy_tower,
            value_tower,
            kind_head,
            kind_condition,
            kind_embedding,
            direction_head,
            duration_head,
            value_head,
            kinds,
            directions,
            durations,
        })
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// The float vector keys [`Input::vectors`] follows.
    pub fn vector_keys(&self) -> &[String] {
        &self.vector_keys
    }

    /// A hash of the config and every weight's name, shape and bits: identifies the model in replay records.
    pub fn hash(&self) -> u64 {
        self.hash
    }

    /// Gumbel noise values per sample: kinds + directions + durations.
    pub fn noise_size(&self) -> usize {
        self.kinds + self.directions + self.durations
    }

    pub fn head_sizes(&self) -> [usize; 3] {
        [self.kinds, self.directions, self.durations]
    }

    pub fn scratch(&self) -> Scratch {
        let d = self.config.entity_d_model;
        let hidden = self.config.hidden_dim;
        let grid_max = self
            .convs
            .iter()
            .map(|c| (c.cin * c.h_in * c.w_in).max(c.cout * c.h_out * c.w_out))
            .max()
            .unwrap_or(0);
        let patch = self.convs.iter().map(|c| c.cin * 9).max().unwrap_or(0);
        Scratch {
            tokens: vec![0.0; self.slots * d],
            qkv: vec![0.0; self.slots * 3 * d],
            attn: vec![0.0; d],
            heads: vec![0.0; d],
            ff: vec![0.0; self.config.entity_ff_dim],
            scores: vec![0.0; self.slots],
            token_in: vec![0.0; self.entity_features + self.config.entity_embed_dim],
            pooled: vec![0.0; d],
            grid_a: vec![0.0; grid_max],
            grid_b: vec![0.0; grid_max],
            patch: vec![0.0; patch],
            features: vec![0.0; self.policy_tower.layers[0].inp],
            tower_a: vec![0.0; hidden],
            tower_b: vec![0.0; hidden],
            value_a: vec![0.0; hidden],
            value_b: vec![0.0; hidden],
            conditioned: vec![0.0; hidden],
            logits: vec![0.0; MAX_HEAD],
            empty: vec![false; self.slots],
            masked: vec![false; self.slots],
        }
    }

    fn check(&self, input: &Input) -> Result<(), PolicyError> {
        let expect = |what: &str, got: usize, want: usize| {
            if got == want { Ok(()) } else { err(format!("{what} has {got} values, expected {want}")) }
        };
        expect("entity_types", input.entity_types.len(), self.slots)?;
        expect("entity_features", input.entity_features.len(), self.slots * self.entity_features)?;
        expect("grid", input.grid.len(), self.grid_rows * self.grid_cols)?;
        expect("vectors", input.vectors.len(), self.vector_sizes.len())?;
        for ((v, &n), key) in input.vectors.iter().zip(&self.vector_sizes).zip(&self.vector_keys) {
            expect(key, v.len(), n)?;
        }
        expect("action_mask", input.mask.len(), self.kinds * (1 + self.directions + self.durations))
    }

    /// The concatenated features (room, entities, grid, vectors) into `s.features`.
    fn features(&self, input: &Input, s: &mut Scratch) {
        let mut at = 0;
        let room = self.room_embedding.row(input.room_id);
        s.features[at..at + room.len()].copy_from_slice(room);
        at += room.len();
        let out = self.config.entity_out_dim;
        self.entities(input, s);
        at += out;
        let grid_out = self.config.grid_out_dim;
        self.grid(input, s, at);
        at += grid_out;
        for v in input.vectors {
            s.features[at..at + v.len()].copy_from_slice(v);
            at += v.len();
        }
        debug_assert_eq!(at, s.features.len());
    }

    /// The entity encoder's output into `s.features[room_embed_dim..]`.
    fn entities(&self, input: &Input, s: &mut Scratch) {
        let d = self.config.entity_d_model;
        let nf = self.entity_features;
        let slots = self.slots;
        for i in 0..slots {
            s.empty[i] = input.entity_features[i * nf] <= 0.0;
        }
        s.masked.copy_from_slice(&s.empty);
        if s.empty.iter().all(|&e| e) {
            s.masked[0] = false;
        }
        for i in 0..slots {
            s.token_in[..nf].copy_from_slice(&input.entity_features[i * nf..(i + 1) * nf]);
            s.token_in[nf..].copy_from_slice(self.type_embedding.row(input.entity_types[i]));
            self.input_proj.apply(&s.token_in, &mut s.tokens[i * d..(i + 1) * d]);
        }
        let heads = self.config.entity_heads;
        let hd = d / heads;
        let scale = 1.0 / (hd as f32).sqrt();
        for layer in &self.layers {
            for i in 0..slots {
                layer.in_proj.apply(&s.tokens[i * d..(i + 1) * d], &mut s.qkv[i * 3 * d..(i + 1) * 3 * d]);
            }
            // Attention for each query slot; the new token values go to `tokens` only after every query has used
            // the old ones (they live in qkv, so updating `tokens` in place per slot is safe).
            for i in 0..slots {
                for h in 0..heads {
                    let q = &s.qkv[i * 3 * d + h * hd..i * 3 * d + (h + 1) * hd];
                    let mut max = f32::NEG_INFINITY;
                    for j in 0..slots {
                        s.scores[j] = if s.masked[j] {
                            f32::NEG_INFINITY
                        } else {
                            let k = &s.qkv[j * 3 * d + d + h * hd..j * 3 * d + d + (h + 1) * hd];
                            dot(q, k) * scale
                        };
                        max = max.max(s.scores[j]);
                    }
                    let mut total = 0f32;
                    for j in 0..slots {
                        let e = if s.masked[j] { 0.0 } else { (s.scores[j] - max).exp() };
                        s.scores[j] = e;
                        total += e;
                    }
                    let out = &mut s.heads[h * hd..(h + 1) * hd];
                    out.fill(0.0);
                    for j in 0..slots {
                        if s.masked[j] {
                            continue;
                        }
                        let p = s.scores[j] / total;
                        let v = &s.qkv[j * 3 * d + 2 * d + h * hd..j * 3 * d + 2 * d + (h + 1) * hd];
                        for (o, x) in out.iter_mut().zip(v) {
                            *o += p * x;
                        }
                    }
                }
                layer.out_proj.apply(&s.heads, &mut s.attn);
                let token = &mut s.tokens[i * d..(i + 1) * d];
                for (t, a) in token.iter_mut().zip(&s.attn) {
                    *t += a;
                }
                layer_norm(token, &layer.norm1.gamma, &layer.norm1.beta);
                layer.linear1.apply(token, &mut s.ff);
                relu(&mut s.ff);
                layer.linear2.apply(&s.ff, &mut s.attn);
                for (t, a) in token.iter_mut().zip(&s.attn) {
                    *t += a;
                }
                layer_norm(token, &layer.norm2.gamma, &layer.norm2.beta);
            }
        }
        s.pooled.fill(0.0);
        let mut count = 0f32;
        for i in 0..slots {
            if !s.empty[i] {
                count += 1.0;
                for (p, t) in s.pooled.iter_mut().zip(&s.tokens[i * d..(i + 1) * d]) {
                    *p += t;
                }
            }
        }
        let count = count.max(1.0);
        for p in s.pooled.iter_mut() {
            *p /= count;
        }
        let at = self.config.room_embed_dim;
        let out = &mut s.features[at..at + self.config.entity_out_dim];
        self.output_proj.apply(&s.pooled, out);
        relu(out);
    }

    /// The grid encoder's output into `s.features[at..]`.
    fn grid(&self, input: &Input, s: &mut Scratch, at: usize) {
        let (rows, cols) = (self.grid_rows, self.grid_cols);
        let c0 = self.config.tile_embed_dim;
        // Embedded grid as [C][H][W].
        for (cell, &id) in input.grid.iter().enumerate() {
            let e = self.tile_embedding.row(id);
            for c in 0..c0 {
                s.grid_a[c * rows * cols + cell] = e[c];
            }
        }
        let mut src_is_a = true;
        for conv in &self.convs {
            let (src, dst) = if src_is_a {
                (&s.grid_a[..], &mut s.grid_b[..])
            } else {
                (&s.grid_b[..], &mut s.grid_a[..])
            };
            let plane = conv.h_in * conv.w_in;
            let patch = &mut s.patch[..conv.cin * 9];
            for oy in 0..conv.h_out {
                for ox in 0..conv.w_out {
                    // The 3×3 patch in the weight's [cin][kh][kw] order, zero outside the padded border.
                    for ci in 0..conv.cin {
                        for ky in 0..3 {
                            for kx in 0..3 {
                                let y = (oy * conv.stride + ky) as isize - 1;
                                let x = (ox * conv.stride + kx) as isize - 1;
                                patch[ci * 9 + ky * 3 + kx] = if y < 0
                                    || x < 0
                                    || y >= conv.h_in as isize
                                    || x >= conv.w_in as isize
                                {
                                    0.0
                                } else {
                                    src[ci * plane + y as usize * conv.w_in + x as usize]
                                };
                            }
                        }
                    }
                    let out_plane = conv.h_out * conv.w_out;
                    for co in 0..conv.cout {
                        let v = dot(&conv.w[co * conv.cin * 9..(co + 1) * conv.cin * 9], patch) + conv.b[co];
                        dst[co * out_plane + oy * conv.w_out + ox] = v.max(0.0);
                    }
                }
            }
            src_is_a = !src_is_a;
        }
        let last = self.convs.last().expect("conv grid encoder has layers");
        let flat = last.cout * last.h_out * last.w_out;
        let src = if src_is_a { &s.grid_a[..flat] } else { &s.grid_b[..flat] };
        let out = &mut s.features[at..at + self.config.grid_out_dim];
        self.grid_head.apply(src, out);
        relu(out);
    }

    /// The value estimate alone (bootstrap values).
    pub fn value(&self, input: &Input, s: &mut Scratch) -> Result<f32, PolicyError> {
        self.check(input)?;
        self.features(input, s);
        self.value_tower.apply(&s.features, &mut s.value_a, &mut s.value_b);
        let mut v = [0f32];
        self.value_head.apply(&s.value_a, &mut v);
        Ok(v[0])
    }

    /// Evaluates one observation and picks an action (`model.py` `ZeldaPolicy.evaluate`).
    pub fn evaluate(&self, input: &Input, pick: Pick, s: &mut Scratch) -> Result<Output, PolicyError> {
        self.check(input)?;
        let (kinds, dirs, durs) = (self.kinds, self.directions, self.durations);
        if let Pick::Sample(noise) = pick
            && noise.len() != kinds + dirs + durs
        {
            return err(format!("noise has {} values, expected {}", noise.len(), kinds + dirs + durs));
        }
        let kind_mask = &input.mask[..kinds];
        if !kind_mask.iter().any(|&m| m) {
            return err("empty kind mask");
        }
        self.features(input, s);
        self.policy_tower.apply(&s.features, &mut s.tower_a, &mut s.tower_b);
        self.value_tower.apply(&s.features, &mut s.value_a, &mut s.value_b);
        let mut value = [0f32];
        self.value_head.apply(&s.value_a, &mut value);

        let mut out = Output {
            action: [0; 3],
            log_prob: 0.0,
            entropy: 0.0,
            value: value[0],
            head_log_probs: [[0.0; MAX_HEAD]; 3],
            head_applies: [false; 3],
            head_entropy: [0.0; 3],
            head_log_prob: [0.0; 3],
        };
        // Kind head.
        self.kind_head.apply(&s.tower_a, &mut s.logits);
        let kind = head(&s.logits[..kinds], kind_mask, pick, 0, &mut out, 0);
        out.head_applies[0] = kind_mask.iter().filter(|&&m| m).count() > 1;
        if !out.head_applies[0] {
            out.head_entropy[0] = 0.0;
            out.head_log_prob[0] = 0.0;
        }
        // Kind-conditioned direction and duration heads.
        self.kind_condition.apply(&s.tower_a, &mut s.conditioned);
        for (c, e) in s.conditioned.iter_mut().zip(self.kind_embedding.row(kind as i64)) {
            *c = (*c + e).max(0.0);
        }
        let dir_mask = &input.mask[kinds + kind * dirs..kinds + (kind + 1) * dirs];
        self.direction_head.apply(&s.conditioned, &mut s.logits);
        sub_head(&s.logits[..dirs], dir_mask, pick, kinds, &mut out, 1);
        let base = kinds + kinds * dirs;
        let dur_mask = &input.mask[base + kind * durs..base + (kind + 1) * durs];
        self.duration_head.apply(&s.conditioned, &mut s.logits);
        sub_head(&s.logits[..durs], dur_mask, pick, kinds + dirs, &mut out, 2);

        out.action[0] = kind;
        out.log_prob = (out.head_log_prob[0] + out.head_log_prob[1]) + out.head_log_prob[2];
        out.entropy = (out.head_entropy[0] + out.head_entropy[1]) + out.head_entropy[2];
        Ok(out)
    }
}

/// A masked categorical head: log-probs, entropy, choice; writes them into `out` at head `index`.
fn head(logits: &[f32], mask: &[bool], pick: Pick, noise_at: usize, out: &mut Output, index: usize) -> usize {
    let n = logits.len();
    let lp = &mut out.head_log_probs[index][..n];
    masked_log_softmax(logits, mask, MASKED_LOGIT, lp);
    let mut entropy = 0f32;
    for &v in lp.iter() {
        entropy -= v.exp() * v;
    }
    let choice = match pick {
        Pick::Given(a) => a[index],
        Pick::Argmax => argmax(lp),
        Pick::Sample(noise) => {
            let mut perturbed = [0f32; MAX_HEAD];
            for (p, (v, g)) in perturbed.iter_mut().zip(lp.iter().zip(&noise[noise_at..noise_at + n])) {
                *p = v + g;
            }
            argmax(&perturbed[..n])
        }
    };
    out.head_entropy[index] = entropy;
    out.head_log_prob[index] = lp[choice.min(n - 1)];
    choice
}

/// A kind-conditioned head: zero log-prob/entropy and choice 0 when the kind's row is all masked.
fn sub_head(logits: &[f32], row_mask: &[bool], pick: Pick, noise_at: usize, out: &mut Output, index: usize) {
    let applies = row_mask.iter().any(|&m| m);
    let mut safe = [true; MAX_HEAD];
    if applies {
        safe[..row_mask.len()].copy_from_slice(row_mask);
    }
    let choice = head(logits, &safe[..row_mask.len()], pick, noise_at, out, index);
    out.head_applies[index] = applies;
    if applies {
        out.action[index] = choice;
    } else {
        out.action[index] = 0;
        out.head_entropy[index] = 0.0;
        out.head_log_prob[index] = 0.0;
    }
}

/// FNV-1a over the config JSON and every weight (sorted by name): name, shape, bits.
fn weights_hash(config: &ModelConfig, weights: &HashMap<String, Tensor>) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |bytes: &[u8]| {
        for b in bytes {
            h ^= *b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    eat(format!("{config:?}").as_bytes());
    let mut names: Vec<&String> = weights.keys().collect();
    names.sort();
    for name in names {
        let t = &weights[name];
        eat(name.as_bytes());
        for d in &t.shape {
            eat(&(*d as u64).to_le_bytes());
        }
        for v in &t.data {
            eat(&v.to_bits().to_le_bytes());
        }
    }
    h
}
