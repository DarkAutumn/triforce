//! `SpecialistEnv` parity with the Python `ZeldaEnv` on the FCEUmm oracle (env parity fixtures, plan step 10).
//!
//! For every `tests/fixtures/parity/*.npz`: load the recorded room-entry oracle state as the recorder did
//! (`randomize_rng=False, fresh_room=True`), build the fixture's specialist env with RAM randomization cleared,
//! set its route and route step, `restore` a snapshot at every episode start and replay the recorded actions.
//! Observations, masks, rewards and reward terms, terminated/truncated/ending/outcome and the terminal info must
//! match exactly (floats within 1e-6).  A difference is allowed only when `tests/parity_exceptions.toml` lists
//! it (and docs/decisions.md logs it).

mod common;

use common::{MemStore, Npy, OracleNes, entry_state, fixtures, open_oracle, read_npz, repo};
use env::config::{GameData, RamRandomization, SpecialistConfig};
use env::observation::Observation;
use env::specialist_env::{EnvOptions, SpecialistEnv};
use rand::SeedableRng;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use zelda::{LoadOptions, MapLocation};

#[derive(serde::Deserialize)]
struct Exceptions {
    #[serde(default)]
    exception: Vec<Exception>,
}

#[derive(serde::Deserialize)]
#[allow(dead_code)]
struct Exception {
    room: String,
    step: i64,
    field: String,
    reason: String,
}

struct Checker<'a> {
    room: String,
    exceptions: &'a [Exception],
    failures: Vec<String>,
    /// Values compared (obs elements, terms, info keys), so a vacuous pass shows.
    compared: usize,
}

impl Checker<'_> {
    fn check<T: PartialEq + std::fmt::Debug>(&mut self, step: i64, field: &str, got: T, want: T) {
        self.compared += 1;
        if got != want && !self.excepted(step, field) {
            self.failures.push(format!(
                "{} step {step} {field}: rust {got:?} != python {want:?}",
                self.room
            ));
        }
    }

    fn close(&mut self, step: i64, field: &str, got: &[f64], want: &[f64]) {
        self.compared += want.len();
        let bad =
            got.len() != want.len() || got.iter().zip(want).any(|(a, b)| (a - b).abs() > 1e-6);
        if bad && !self.excepted(step, field) {
            let detail: Vec<String> = got
                .iter()
                .zip(want)
                .enumerate()
                .filter(|(_, (a, b))| (*a - *b).abs() > 1e-6)
                .take(6)
                .map(|(i, (a, b))| format!("[{i}] {a} vs {b}"))
                .collect();
            self.failures.push(format!(
                "{} step {step} {field}: rust/python differ (len {} vs {}): {}",
                self.room,
                got.len(),
                want.len(),
                detail.join(", ")
            ));
        }
    }

    /// Every key of Python's info JSON must match Rust's (numbers within 1e-6).
    fn json(&mut self, step: i64, field: &str, got: &Value, want: &Value) {
        self.compared += want.as_object().map_or(1, |o| o.len());
        if let Some(diff) = json_diff(got, want, field)
            && !self.excepted(step, field)
        {
            self.failures
                .push(format!("{} step {step} {diff}", self.room));
        }
    }

    fn excepted(&self, step: i64, field: &str) -> bool {
        self.exceptions
            .iter()
            .any(|e| e.room == self.room && (e.step == step || e.step == -1) && e.field == field)
    }
}

fn json_diff(got: &Value, want: &Value, path: &str) -> Option<String> {
    match (got, want) {
        (Value::Object(g), Value::Object(w)) => w.iter().find_map(|(k, wv)| {
            let p = format!("{path}.{k}");
            match g.get(k) {
                Some(gv) => json_diff(gv, wv, &p),
                None => Some(format!("{p}: missing in rust (python {wv})")),
            }
        }),
        (Value::Array(g), Value::Array(w)) if g.len() == w.len() => g
            .iter()
            .zip(w)
            .enumerate()
            .find_map(|(i, (gv, wv))| json_diff(gv, wv, &format!("{path}[{i}]"))),
        (Value::Number(g), Value::Number(w)) => ((g.as_f64().unwrap() - w.as_f64().unwrap()).abs()
            > 1e-6)
            .then(|| format!("{path}: rust {g} != python {w}")),
        _ => (got != want).then(|| format!("{path}: rust {got} != python {want}")),
    }
}

fn row<T: Clone>(v: &[T], n: usize, i: usize) -> Vec<T> {
    v[i * n..(i + 1) * n].to_vec()
}

/// (key, flattened values) of an observation in fixture order.
fn obs_fields(o: &Observation) -> Vec<(&'static str, Vec<f64>)> {
    let f = |v: &[f32]| v.iter().map(|&x| x as f64).collect::<Vec<_>>();
    vec![
        ("room_id", vec![o.room_id as f64]),
        ("room_state", f(&o.room_state)),
        ("link", f(&o.link)),
        (
            "entity_types",
            o.entity_types.iter().map(|&x| x as f64).collect(),
        ),
        (
            "entity_features",
            o.entity_features.iter().flat_map(|r| f(r)).collect(),
        ),
        ("grid", o.grid.iter().flatten().map(|&x| x as f64).collect()),
        ("history", f(&o.history)),
        ("objective", f(&o.objective)),
        (
            "action_mask",
            o.action_mask.iter().map(|&x| x as u8 as f64).collect(),
        ),
    ]
}

fn check_obs(
    c: &mut Checker,
    step: i64,
    obs: &Observation,
    f: &HashMap<String, Npy>,
    prefix: &str,
    i: Option<usize>,
) {
    for (key, got) in obs_fields(obs) {
        let arr = &f[&format!("{prefix}{key}")];
        let all = arr.f64();
        let want = match i {
            Some(i) => row(&all, arr.row(), i),
            None => all,
        };
        let exact = matches!(key, "room_id" | "entity_types" | "grid" | "action_mask");
        if exact {
            c.check(step, &format!("obs.{key}"), got, want);
        } else {
            c.close(step, &format!("obs.{key}"), &got, &want);
        }
    }
}

fn run_fixture(
    nes: OracleNes,
    f: &HashMap<String, Npy>,
    data: &Arc<GameData>,
    c: &mut Checker,
) -> OracleNes {
    let specialist = f["specialist"].string();
    let config = SpecialistConfig {
        ram_randomization: RamRandomization::default(),
        ..(*data.specialists[&specialist]).clone()
    };
    let route = f["route"].string();
    let route_step = f["route_step"].scalar_i64();
    let route_step = (route_step >= 0).then_some(route_step as usize);
    let label = f["label"].string();
    let opts = EnvOptions {
        explicit_starts: true,
        ..Default::default()
    };
    let mut env = SpecialistEnv::new(
        Arc::new(config),
        data.clone(),
        nes,
        MemStore::default(),
        opts,
    )
    .unwrap();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
    let load = LoadOptions {
        randomize_rng: false,
        fresh_room: true,
        ..Default::default()
    };
    env.session_mut()
        .load(&entry_state(f), &load, &mut rng)
        .expect("load");
    let loc = MapLocation::new(
        f["level"].scalar_i64() as u8,
        f["room"].scalar_i64() as u8,
        f["in_cave"].bools()[0],
    );
    c.check(-1, "location", env.session().state().full_location(), loc);
    env.set_route(&route, route_step);
    let snap = env.session().snapshot();
    let (obs0, info0) = env.restore(&snap, &label).unwrap();
    check_obs(c, -1, &obs0, f, "obs0_", None);
    c.check(-1, "mask", obs0.action_mask.to_vec(), f["mask0"].bools());
    c.check(
        -1,
        "start_state",
        info0.start_state.clone(),
        Some(label.clone()),
    );

    let steps = f["frames"].len();
    let actions = f["actions"].i64();
    let episode_start = f["episode_start"].bools();
    let (frames, interrupt) = (f["frames"].i64(), f["interrupt"].i64());
    let (rewards, terminated, truncated) = (
        f["reward"].f64(),
        f["terminated"].bools(),
        f["truncated"].bools(),
    );
    let (endings, outcomes, infos) = (
        f["ending"].strings(),
        f["outcome"].strings(),
        f["info_json"].strings(),
    );
    let (term_step, term_name, term_value) = (
        f["reward_term_step"].i64(),
        f["reward_term_name"].strings(),
        f["reward_term_value"].f64(),
    );
    let (tail, max) = (f["linger_tail_steps"].i64(), f["linger_max_steps"].i64());
    for i in 0..steps {
        let step = i as i64;
        if episode_start[i] && i > 0 {
            let snap = env.session().snapshot();
            env.restore(&snap, &label).unwrap();
        }
        let a = row(&actions, 3, i);
        let result = env
            .step_indices([a[0] as usize, a[1] as usize, a[2] as usize])
            .unwrap_or_else(|e| panic!("{} step {i}: {e}", c.room));
        c.check(step, "frames", result.info.frames as i64, frames[i]);
        c.check(
            step,
            "interrupt",
            result.info.interrupt.map_or(-1, |x| x.index() as i64),
            interrupt[i],
        );
        check_obs(c, step, &result.obs, f, "obs_", Some(i));
        c.close(step, "reward", &[result.reward], &[rewards[i]]);
        c.check(step, "terminated", result.terminated, terminated[i]);
        c.check(step, "truncated", result.truncated, truncated[i]);
        c.check(
            step,
            "ending",
            result.info.ending().unwrap_or("").to_owned(),
            endings[i].clone(),
        );
        c.check(
            step,
            "outcome",
            result.info.outcome().unwrap_or("").to_owned(),
            outcomes[i].clone(),
        );
        let want_terms: Vec<(String, f64)> = (0..term_step.len())
            .filter(|&k| term_step[k] == step)
            .map(|k| (term_name[k].clone(), term_value[k]))
            .collect();
        let got_terms: Vec<(String, f64)> = result
            .info
            .rewards
            .iter()
            .map(|(n, v)| (n.to_string(), *v))
            .collect();
        c.check(
            step,
            "reward_terms",
            got_terms.iter().map(|t| t.0.clone()).collect::<Vec<_>>(),
            want_terms.iter().map(|t| t.0.clone()).collect::<Vec<_>>(),
        );
        if got_terms.len() == want_terms.len() {
            let g: Vec<f64> = got_terms.iter().map(|t| t.1).collect();
            let w: Vec<f64> = want_terms.iter().map(|t| t.1).collect();
            c.close(step, "reward_values", &g, &w);
        }
        c.check(
            step,
            "linger_tail_steps",
            env.linger.tail_steps() as i64,
            tail[i],
        );
        c.check(
            step,
            "linger_max_steps",
            env.linger.summary().max_steps as i64,
            max[i],
        );
        if !infos[i].is_empty() {
            let want: Value = serde_json::from_str(&infos[i]).unwrap();
            c.json(step, "info", &result.info.to_json(), &want);
        } else {
            c.check(step, "terminal", result.info.terminal.is_some(), false);
        }
        if c.failures.len() > 30 {
            break;
        }
        if result.info.terminal.is_none() && env.session().state().game_over() {
            break;
        }
    }
    let (session, _) = env.into_parts();
    session.into_nes()
}

#[test]
fn env_matches_python_fixtures() {
    let _lock = oracle::lock();
    let Some(mut nes) = open_oracle() else { return };
    let data = common::game_data();
    let text =
        std::fs::read_to_string(repo().join("crates/env/tests/parity_exceptions.toml")).unwrap();
    let exceptions: Exceptions = toml::from_str(&text).unwrap();
    let files = fixtures();
    assert!(
        files.len() >= 14,
        "expected the 14 parity fixtures, found {}",
        files.len()
    );
    let mut failures = Vec::new();
    for path in files {
        let room = path.file_stem().unwrap().to_string_lossy().into_owned();
        let f = read_npz(&path);
        let mut c = Checker {
            room: room.clone(),
            exceptions: &exceptions.exception,
            failures: Vec::new(),
            compared: 0,
        };
        nes = run_fixture(nes, &f, &data, &mut c);
        eprintln!(
            "{room}: {} steps, {} values compared, {} differences",
            f["frames"].len(),
            c.compared,
            c.failures.len()
        );
        assert!(c.compared > 10_000, "{room}: too few comparisons");
        failures.extend(c.failures);
    }
    assert!(
        failures.is_empty(),
        "{} differences:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
