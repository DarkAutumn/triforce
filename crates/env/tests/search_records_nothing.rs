//! F021: simulated (search) steps leave no trace outside the episode. An env that searches before every real step
//! (simulations that run into episode ends) and an env that never searches, same seed and actions, must agree on
//! everything the shared records see: the start-state pool's draws (its failure memory), the curriculum's
//! progress file, and the room-visit statistics. Skipped without the ROM (`TRIFORCE_ALLOW_NO_ROM=1`).

mod common;

use common::{TfzStore, game_data, open_core, state_path};
use env::config::RamRandomization;
use env::curriculum::{Curriculum, CurriculumSpec, TRAJECTORY_FILE};
use env::explore::random_action;
use env::specialist_env::{EnvOptions, SpecialistEnv};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::path::Path;
use std::sync::Arc;
use zelda::Action;
use zelda::visits::RoomVisit;

const SPECIALIST: &str = "stalfos-keese-gel";
const STEPS: usize = 80;

/// A one-track trajectory directory (three copies of a real seed state) for a curriculum.
fn trajectories(root: &Path) {
    let track = root.join("1_73s_0");
    std::fs::create_dir_all(&track).unwrap();
    for i in 0..3 {
        std::fs::copy(state_path("1_73s_0"), track.join(format!("{i:04}.state"))).unwrap();
    }
    let meta = serde_json::json!({"start_state": "1_73s_0", "success": true});
    std::fs::write(track.join(TRAJECTORY_FILE), meta.to_string()).unwrap();
}

/// What the shared records saw: each episode's start state and room visits, and the curriculum progress file.
type Trace = (Vec<(String, Vec<RoomVisit>)>, String);

fn run(search: bool, tracks: &Path, run_dir: &Path) -> Option<Trace> {
    let nes = open_core()?;
    let data = game_data();
    let mut config = (*data.specialists[SPECIALIST]).clone();
    config.ram_randomization = RamRandomization::default();
    config.max_steps = 4;
    assert!(config.start_states.len() >= 2, "the start-state pool has a choice to make");
    let mut spec = CurriculumSpec::new(tracks.to_path_buf(), run_dir.join("curriculum.json"), 0.5, 0.6);
    spec.stages = 2;
    spec.window = 2;
    let opts = EnvOptions {
        curriculum: Some(Curriculum::new(spec).unwrap().shared()),
        track_visits: true,
        seed: 5,
        ..Default::default()
    };
    let mut env = SpecialistEnv::new(Arc::new(config), data, nes, TfzStore::new(), opts).unwrap();
    let mut actions = ChaCha8Rng::seed_from_u64(1);
    let mut sim_actions = ChaCha8Rng::seed_from_u64(2);
    let mut episodes = Vec::new();
    let mut ends_simulated = 0;
    env.reset(None, None).unwrap();
    for _ in 0..STEPS {
        if search {
            let snapshot = env.save();
            env.set_simulating(true);
            for _ in 0..8 {
                let action: Action = random_action(env.session().mask(), &mut sim_actions, None);
                let r = env.simulate(action).unwrap();
                if r.terminated || r.truncated {
                    ends_simulated += 1;
                    break;
                }
            }
            env.load(&snapshot);
            env.set_simulating(false);
        }
        let action = random_action(env.session().mask(), &mut actions, None);
        let r = env.step(action).unwrap();
        if r.terminated || r.truncated {
            let start = r.info.terminal.as_ref().unwrap().start_state.clone();
            episodes.push((start, env.take_visits()));
            env.reset(None, None).unwrap();
        }
    }
    if search {
        assert!(ends_simulated > 10, "the simulations reach episode ends ({ends_simulated})");
    }
    let progress = std::fs::read_to_string(run_dir.join("curriculum.json")).unwrap();
    Some((episodes, progress))
}

#[test]
fn simulated_steps_record_nothing() {
    let tmp = tempfile::tempdir().unwrap();
    let tracks = tmp.path().join("trajectories");
    trajectories(&tracks);
    let (plain_dir, searched_dir) = (tmp.path().join("plain"), tmp.path().join("searched"));
    std::fs::create_dir_all(&plain_dir).unwrap();
    std::fs::create_dir_all(&searched_dir).unwrap();
    let Some(plain) = run(false, &tracks, &plain_dir) else { return };
    let searched = run(true, &tracks, &searched_dir).unwrap();
    assert!(plain.0.len() >= 10, "{} episodes", plain.0.len());
    let starts: std::collections::BTreeSet<_> = plain.0.iter().map(|(s, _)| s.as_str()).collect();
    assert!(starts.len() >= 2, "curriculum and pool starts both happen: {starts:?}");
    assert!(plain.0.iter().any(|(_, v)| !v.is_empty()), "visits are tracked");
    assert_eq!(searched.0, plain.0, "start states and room visits");
    assert_eq!(searched.1, plain.1, "curriculum progress");
}
