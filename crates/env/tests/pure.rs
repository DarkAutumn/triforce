//! Emulator-free ports of `tests/test_rewards.py` (StepRewards, terminal rewards, ending classification, the
//! projectile-block reward), `tests/test_curriculum.py`, the config tests of `tests/test_specialist.py` and
//! `tests/test_env.py::test_measured_ending_credits_threshold_action_before_check`.

mod common;

use env::config::{GameData, SpecialistConfig, format_room, parse_room};
use env::critic::{Critique, GameplayCritic};
use env::curriculum::{Curriculum, CurriculumSpec, TRAJECTORY_FILE, Track};
use env::end_conditions::{
    DEFAULT_FAILURES, EndCondition, EndConditions, StepView, classify_ending, enter_room_name,
    parse_enter_room, validate_success_kind,
};
use env::rewards::{
    HARD_FAILURE, Outcome, REWARD_MAXIMUM, REWARD_MEDIUM, SOFT_FAILURE, SUCCESS, StepRewards,
    apply_terminal_rewards, success_reward_value,
};
use indexmap::IndexMap;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::{BTreeSet, HashSet};
use std::path::Path;
use std::sync::Arc;
use zelda::blocks::ProjectileBlock;
use zelda::linger::{LingerState, LingerStep, LingerTracker};
use zelda::objectives::{Objective, ObjectiveKind};
use zelda::room::Wavefront;
use zelda::tables::RAM_SIZE;
use zelda::{Action, ActionMask, Change, GameMap, GameView, MapLocation};

fn approx(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * b.abs().max(1.0)
}

fn terms(rewards: &StepRewards) -> Vec<(&'static str, f64)> {
    rewards.terms().collect()
}

// --- StepRewards (test_rewards.py) ---

#[test]
fn step_value_is_clamped_to_one() {
    let mut rewards = StepRewards::default();
    rewards.add(Outcome::reward("reward-a", 0.8));
    rewards.add(Outcome::reward("reward-b", 0.8));
    assert_eq!(rewards.value(), REWARD_MAXIMUM);
    let mut rewards = StepRewards::default();
    rewards.add(Outcome::penalty("penalty-a", -0.9));
    rewards.add(Outcome::penalty("penalty-b", -0.9));
    assert_eq!(rewards.value(), -REWARD_MAXIMUM);
}

#[test]
#[should_panic(expected = "Duplicate outcome")]
fn duplicate_outcome_is_rejected() {
    let mut rewards = StepRewards::default();
    rewards.add(Outcome::reward("reward-a", 0.1));
    rewards.add(Outcome::reward("reward-a", 0.1));
}

#[test]
fn health_halving_exempts_equipment() {
    let mut rewards = StepRewards::default();
    rewards.add(Outcome::reward("reward-hit", 0.5));
    rewards.add(Outcome::reward("reward-gained-keys", 1.0));
    rewards.add(Outcome::penalty("penalty-lost-health", -0.25));
    rewards.scale_rewards(0.5, &["reward-gained-"]);
    assert_eq!(
        terms(&rewards),
        [
            ("reward-hit", 0.25),
            ("reward-gained-keys", 1.0),
            ("penalty-lost-health", -0.25)
        ]
    );
}

#[test]
fn success_reward_scales_with_health() {
    for (health, max_health, expected) in [
        (3.0, 3.0, 20.0),
        (1.5, 3.0, 15.0),
        (0.5, 16.0, 10.3125),
        (0.0, 3.0, 10.0),
    ] {
        let value = success_reward_value(health, max_health);
        assert!(
            approx(value, expected),
            "{health}/{max_health}: {value} != {expected}"
        );
    }
}

#[test]
fn success_terminal_escapes_step_clamp() {
    let mut rewards = StepRewards::default();
    rewards.add(Outcome::reward("reward-new-location", 1.0));
    apply_terminal_rewards(&mut rewards, "exit-correct", SUCCESS, true, 3.0, 3.0);
    assert_eq!(rewards.value(), 20.0); // 21 clamped to the terminal maximum
    assert_eq!(rewards.ending.as_deref(), Some("exit-correct"));
    assert_eq!(rewards.outcome, Some(SUCCESS));
}

#[test]
fn hard_failure_drops_rewards_soft_keeps_them() {
    let mut hard = StepRewards::default();
    hard.add(Outcome::reward("reward-hit", 0.25));
    hard.add(Outcome::penalty("penalty-lost-health", -0.5));
    apply_terminal_rewards(&mut hard, "death", HARD_FAILURE, true, 0.0, 3.0);
    assert_eq!(
        terms(&hard),
        [
            ("penalty-lost-health", -0.5),
            ("penalty-terminal-failure", -20.0)
        ]
    );
    assert_eq!(hard.value(), -20.0); // -20.5 clamped to the terminal minimum

    let mut soft = StepRewards::default();
    soft.add(Outcome::reward("reward-hit", 0.25));
    apply_terminal_rewards(&mut soft, "exit-wrong", SOFT_FAILURE, true, 3.0, 3.0);
    assert!(approx(soft.value(), -19.75), "{}", soft.value());
}

#[test]
fn truncation_records_but_adds_no_terminal() {
    let mut rewards = StepRewards::default();
    rewards.add(Outcome::penalty("penalty-room-stalling", -0.01));
    apply_terminal_rewards(&mut rewards, "max-steps", SOFT_FAILURE, false, 3.0, 3.0);
    assert_eq!(rewards.outcome, Some(SOFT_FAILURE));
    assert!(approx(rewards.value(), -0.01), "{}", rewards.value());
}

// --- classification ---

#[test]
fn classification_is_deny_by_default() {
    let success = vec!["exit-correct".to_owned()];
    let none = IndexMap::new();
    assert_eq!(
        classify_ending("exit-correct", &success, &none),
        Ok(SUCCESS)
    );
    assert_eq!(classify_ending("death", &success, &none), Ok(HARD_FAILURE));
    assert_eq!(
        classify_ending("exit-wrong", &success, &none),
        Ok(SOFT_FAILURE)
    );
    // A success kind the specialist didn't list.
    assert!(classify_ending("triforce-collected", &success, &none).is_err());
    assert!(classify_ending("mystery", &success, &none).is_err());
    let overrides: IndexMap<String, String> = [("exit-wrong".to_owned(), HARD_FAILURE.to_owned())]
        .into_iter()
        .collect();
    assert_eq!(
        classify_ending("exit-wrong", &success, &overrides),
        Ok(HARD_FAILURE)
    );
    let failures: HashSet<&str> = DEFAULT_FAILURES.iter().map(|(e, _)| *e).collect();
    for kind in ["exit-correct", "triforce-collected", "boss-dead-and-heart"] {
        assert!(!failures.contains(kind), "{kind}");
    }
}

#[test]
fn enter_room_kinds_round_trip() {
    assert_eq!(
        parse_enter_room("enter-room:2:0x0e"),
        Ok(MapLocation::new(2, 0x0e, false))
    );
    assert_eq!(
        enter_room_name(MapLocation::new(2, 0x0e, false)),
        "enter-room:2:0x0e"
    );
    assert_eq!(
        validate_success_kind("enter-room:2:0xE").as_deref(),
        Ok("enter-room:2:0x0e")
    );
    assert_eq!(
        enter_room_name(parse_enter_room("enter-room:0:0x77:cave").unwrap()),
        "enter-room:0:0x77:cave"
    );
    for bad in ["exit-sideways", "enter-room:2", "enter-room:x:0x0e"] {
        assert!(validate_success_kind(bad).is_err(), "{bad}");
    }
}

// --- projectile-block reward ---

/// A playable (mode 5) view of `location` from a blank RAM image.
fn view(location: MapLocation, map: &Arc<GameMap>) -> Arc<GameView> {
    let mut ram = [0u8; RAM_SIZE];
    ram[0x10] = location.level;
    ram[0xEB] = location.value;
    ram[0x12] = 5;
    Arc::new(GameView::new(&ram, 0, false, map.clone()))
}

/// `_block_change`: a MOVE step that stays in `room` and confirmed `events`.
fn block_change(events: Vec<ProjectileBlock>, room: MapLocation, map: &Arc<GameMap>) -> Change {
    let state = view(room, map);
    Change {
        action: Action::parse("MOVE E 8").unwrap(),
        frames: 8,
        interrupt: None,
        control_frames: 8,
        projectile_blocks: events,
        previous: state.clone(),
        state,
        action_mask: ActionMask::default(),
        health_lost: 0.0,
        health_gained: 0.0,
        enemies_hit: Vec::new(),
        enemies_stunned: Vec::new(),
        items_gained: Vec::new(),
    }
}

fn critique(critic: &mut GameplayCritic, change: &Change, map: &GameMap) -> StepRewards {
    let objective = Objective {
        kind: ObjectiveKind::Move,
        targets: BTreeSet::new(),
        next_rooms: BTreeSet::new(),
        pbrs_targets: BTreeSet::new(),
    };
    let c = Critique {
        change,
        map,
        prev_objective: &objective,
        objective: &objective,
        prev_wavefront: None,
        quiet_steps: 0,
    };
    let mut rewards = StepRewards::default();
    critic.critique(&c, &mut rewards);
    rewards
}

fn block(frame: u64, slot: u8, projectile_id: u8, full_location: MapLocation) -> ProjectileBlock {
    ProjectileBlock {
        frame,
        slot,
        projectile_id,
        full_location,
    }
}

#[test]
fn block_reward_multiplier_scales_unchanged_base_without_sound_onset() {
    let map = Arc::new(GameMap::builtin());
    let room = MapLocation::new(1, 0x44, false);
    for multiplier in [0.0, 3.0] {
        // create_critic('default', pbrs=False, block_reward=multiplier)
        let text = format!(
            "name: t\nrooms: ['(1, 0x44)']\nsuccess: [exit-correct]\nreward: {{pbrs: false, block_reward: {multiplier}}}\n"
        );
        let config = SpecialistConfig::from_yaml(&text, "t.yaml", None).unwrap();
        let mut critic = GameplayCritic::for_config(&config).unwrap();
        let change = block_change(vec![block(10, 11, 0x5C, room)], room, &map);
        let rewards = critique(&mut critic, &change, &map);
        let value = rewards
            .get("reward-block-projectile")
            .map_or(0.0, |o| o.value);
        assert_eq!(value, REWARD_MEDIUM * multiplier, "multiplier {multiplier}");
        if multiplier != 0.0 {
            assert_eq!(rewards.get("reward-block-projectile").unwrap().count, 1);
        }
        // Nothing else rewards this step: the block reward is the only term.
        assert_eq!(
            rewards.len(),
            usize::from(multiplier != 0.0),
            "{:?}",
            terms(&rewards)
        );
    }
}

#[test]
fn each_block_advances_decay_in_its_collision_room_not_the_option_end_room() {
    let map = Arc::new(GameMap::builtin());
    let (room, next_room) = (
        MapLocation::new(1, 0x44, false),
        MapLocation::new(1, 0x43, false),
    );
    let mut critic = GameplayCritic::new(false, 2.0, 0.0);
    // Python sets `_room_combat_counts[room] = 9` (private here): nine earlier blocks in `room` count the same.
    let earlier: Vec<_> = (0..9).map(|f| block(f, 11, 0x5C, room)).collect();
    critique(&mut critic, &block_change(earlier, room, &map), &map);
    // Next two decay factors in `room` are 0.5 and 0.25; the new room remains 1.0.
    let events = vec![
        block(10, 10, 0x5C, room),
        block(10, 11, 0x5C, room),
        block(20, 11, 0x53, next_room),
    ];
    let change = block_change(events, next_room, &map);
    let rewards = critique(&mut critic, &change, &map);
    let outcome = rewards.get("reward-block-projectile").unwrap();
    assert_eq!(outcome.value, REWARD_MEDIUM * 2.0 * (0.5 + 0.25 + 1.0));
    assert_eq!(outcome.count, 3);
    assert_eq!(rewards.value(), 1.0); // normal per-step clamping is unchanged
}

#[test]
fn grouping_blocks_into_longer_options_does_not_change_decay_or_total_reward() {
    let map = Arc::new(GameMap::builtin());
    let room = MapLocation::new(1, 0x44, false);
    let events: Vec<_> = (0..12).map(|frame| block(frame, 11, 0x5C, room)).collect();
    let total = |groups: Vec<Vec<ProjectileBlock>>| {
        let mut critic = GameplayCritic::new(false, 1.0, 0.0);
        let (mut reward, mut count) = (0.0, 0);
        for group in groups {
            let rewards = critique(&mut critic, &block_change(group, room, &map), &map);
            let outcome = rewards.get("reward-block-projectile").unwrap();
            reward += outcome.value;
            count += outcome.count;
        }
        (reward, count)
    };
    let together = total(vec![events.clone()]);
    let apart = total(events.into_iter().map(|e| vec![e]).collect());
    assert_eq!(together, apart);
    assert_eq!(together.1, 12);
}

// --- curriculum (test_curriculum.py) ---

fn trajectory(root: &Path, name: &str, length: usize, success: bool) {
    let dir = root.join(name);
    std::fs::create_dir_all(&dir).unwrap();
    for index in 0..length {
        std::fs::write(dir.join(format!("{index:04}.state")), b"").unwrap();
    }
    let meta = serde_json::json!({"start_state": name, "success": success});
    std::fs::write(dir.join(TRAJECTORY_FILE), meta.to_string()).unwrap();
}

fn curriculum(tmp: &Path) -> Curriculum {
    trajectory(&tmp.join("traj"), "a", 20, true);
    trajectory(&tmp.join("traj"), "failed", 5, false);
    let mut spec = CurriculumSpec::new(
        tmp.join("traj"),
        tmp.join("run").join("curriculum.json"),
        1.0,
        0.5,
    );
    spec.stages = 4;
    spec.window = 4;
    Curriculum::new(spec).unwrap()
}

fn tag(index: usize) -> (String, usize) {
    ("a".to_owned(), index)
}

#[test]
fn starts_near_the_end_and_moves_back_on_success() {
    let tmp = tempfile::tempdir().unwrap();
    let mut curriculum = curriculum(tmp.path());
    // Unsuccessful explorations are skipped.
    assert_eq!(curriculum.tracks.keys().collect::<Vec<_>>(), ["a"]);
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let (path, (name, index)) = curriculum.next(&mut rng).unwrap();
    assert!(name == "a" && (15..20).contains(&index), "{name} {index}");
    assert!(
        path.ends_with(format!("{index:04}.state")),
        "{}",
        path.display()
    );

    for _ in 0..3 {
        assert_eq!(curriculum.record(&tag(15), true).unwrap().point, 15);
    }
    // 3/4 >= 0.5: one stage (stride 5) earlier.
    assert_eq!(curriculum.record(&tag(16), false).unwrap().point, 10);
    for _ in 0..20 {
        let index = curriculum.next(&mut rng).unwrap().1.1;
        assert!((10..15).contains(&index), "{index}");
    }
}

#[test]
fn failed_window_restarts_the_stage_and_stale_episodes_do_not_count() {
    let tmp = tempfile::tempdir().unwrap();
    let mut curriculum = curriculum(tmp.path());
    for _ in 0..3 {
        curriculum.record(&tag(15), false).unwrap();
    }
    curriculum.record(&tag(15), true).unwrap(); // 1/4 < 0.5: stay, counters restart
    assert_eq!(
        curriculum.progress()["a"],
        Track {
            length: 20,
            point: 15,
            attempts: 0,
            successes: 0,
            stages_passed: 0
        }
    );
    curriculum.record(&tag(19), true).unwrap();
    curriculum.record(&tag(3), true).unwrap(); // older/later stage index: ignored
    assert_eq!(curriculum.progress()["a"].attempts, 1);
}

#[test]
fn complete_track_is_no_longer_sampled_and_progress_is_shared() {
    let tmp = tempfile::tempdir().unwrap();
    let shared = curriculum(tmp.path()).shared();
    let other = shared.clone(); // a second env of the same run
    for point in [15, 10, 5] {
        for _ in 0..4 {
            other.lock().record(&tag(point), true).unwrap();
        }
    }
    assert_eq!(shared.lock().progress()["a"].point, 0);
    assert!(
        shared
            .lock()
            .next(&mut ChaCha8Rng::seed_from_u64(1))
            .is_none()
    );
    // A worker that (re)opens the same progress file sees the same progress.
    let spec = shared.lock().spec.clone();
    let reopened = Curriculum::new(spec).unwrap();
    assert_eq!(reopened.progress()["a"].point, 0);
    assert_eq!(reopened.progress()["a"].stages_passed, 3);
    assert!(reopened.next(&mut ChaCha8Rng::seed_from_u64(2)).is_none());
}

// --- specialist configs (test_specialist.py) ---

fn routes() -> serde_yaml_ng::Value {
    let text = std::fs::read_to_string(common::repo().join("triforce/routes.yaml")).unwrap();
    serde_yaml_ng::from_str(&text).unwrap()
}

fn routes_families(routes: &serde_yaml_ng::Value) -> &serde_yaml_ng::Mapping {
    routes["families"]
        .as_mapping()
        .expect("routes.yaml families")
}

#[test]
fn every_family_has_a_config() {
    let data = GameData::load(&common::repo()).unwrap();
    let mut configs: Vec<String> = std::fs::read_dir(common::repo().join("specialists"))
        .unwrap()
        .filter_map(|e| {
            e.unwrap()
                .file_name()
                .to_string_lossy()
                .strip_suffix(".yaml")
                .map(str::to_owned)
        })
        .collect();
    configs.sort();
    let routes = routes();
    let mut families: Vec<String> = routes_families(&routes)
        .keys()
        .map(|k| k.as_str().unwrap().to_owned())
        .collect();
    families.sort();
    assert_eq!(configs, families);
    assert_eq!(
        data.specialists.keys().cloned().collect::<Vec<_>>(),
        families
    );
}

#[test]
fn config_rooms_match_game_yaml_tags_and_routes() {
    let data = GameData::load(&common::repo()).unwrap();
    let routes = routes();
    let families = routes_families(&routes);
    assert!(!data.specialists.is_empty());
    for (name, config) in &data.specialists {
        assert_eq!(&config.name, name);
        let rooms: HashSet<MapLocation> = config.rooms.iter().copied().collect();
        let tagged: HashSet<MapLocation> =
            data.map.rooms_for_specialist(name).into_iter().collect();
        assert_eq!(rooms, tagged, "{name}: rooms must be tagged in game.yaml");
        let formatted: HashSet<String> = config.rooms.iter().map(|r| format_room(*r)).collect();
        let family: HashSet<String> = families[name.as_str()]
            .as_sequence()
            .unwrap()
            .iter()
            .map(|r| format_room(parse_room(r.as_str().unwrap()).unwrap()))
            .collect();
        assert_eq!(formatted, family, "{name}");
        for room in &config.rooms {
            assert_eq!(
                data.map.get(*room).and_then(|r| r.specialist.as_deref()),
                Some(name.as_str())
            );
        }
    }
}

#[test]
fn every_route_room_belongs_to_its_family() {
    let data = GameData::load(&common::repo()).unwrap();
    let routes = routes();
    let steps = routes["milestone1"]
        .as_sequence()
        .expect("milestone1 route");
    assert!(!steps.is_empty());
    for step in steps {
        let level = step["level"].as_u64().unwrap() as u8;
        let room = step["room"].as_str().unwrap();
        let room = u8::from_str_radix(room.trim_start_matches("0x"), 16).unwrap();
        let in_cave = step
            .get("in_cave")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let location = MapLocation::new(level, room, in_cave);
        let specialist = data
            .specialist(step["specialist"].as_str().unwrap())
            .unwrap();
        assert!(
            specialist.rooms.contains(&location),
            "{location:?} not in {}",
            specialist.name
        );
    }
}

#[test]
fn trainable_specialists_have_start_states() {
    let data = GameData::load(&common::repo()).unwrap();
    for (name, config) in &data.specialists {
        if config.trainable {
            assert!(
                !config.resolve_start_states(&data.states_dir).is_empty(),
                "{name}"
            );
        }
    }
}

/// `test_yaml_on_off_and_max_parse`: PyYAML (YAML 1.1) reads `on`/`off` as booleans and so does the Rust
/// config (serde_yaml_ng alone is YAML 1.2, where they are strings).
#[test]
fn yaml_on_off_and_max_parse() {
    let config = SpecialistConfig::from_yaml(
        "name: t\nrooms: ['(1, 0x73)']\nsuccess: exit-correct\nreward: {pbrs: off}\nobs: {objective: on}\n\
         ram_randomization: {hp: [0.5, max], bombs: [0, 4]}\n",
        "t.yaml",
        None,
    )
    .unwrap();
    assert!(!config.reward.pbrs && config.obs.objective);
    assert_eq!(config.success, ["exit-correct"]);
}

// --- measured ending (test_env.py) ---

/// `test_measured_ending_credits_threshold_action_before_check`, every `progress` case: the shared linger
/// tracker credits the step's progress before NoRoomProgress checks its tail.
#[test]
fn measured_ending_credits_threshold_action_before_check() {
    let map = Arc::new(GameMap::builtin());
    let objective = Objective {
        kind: ObjectiveKind::Move,
        targets: BTreeSet::new(),
        next_rooms: BTreeSet::new(),
        pbrs_targets: BTreeSet::new(),
    };
    // Python's pbrs_wavefront {t: t}: a tile's distance is its column.
    let wavefront = Arc::new(Wavefront::from_distances((0..10).map(|x| ((x, 0), x))));
    let state = |tile: i32, room: u8| LingerState {
        full_location: MapLocation::new(1, room, false),
        game_over: false,
        tile: (tile, 0),
        keys: 0,
        max_health: 3,
        room_kills: 0,
        boss_alive: false,
        objective_kind: ObjectiveKind::Move,
        next_rooms: [MapLocation::new(1, 0x63, false)].into_iter().collect(),
        has_targets: true,
    };

    // build_end_conditions: no_progress_steps swaps Timeout for NoRoomProgress.
    let conditions = EndConditions::new(&[], &[], false, Some(3), None);
    assert!(matches!(
        conditions.0.last(),
        Some(EndCondition::NoRoomProgress {
            no_progress_steps: 3
        })
    ));
    assert!(
        !conditions
            .0
            .iter()
            .any(|c| matches!(c, EndCondition::Timeout { .. }))
    );
    assert!(matches!(
        EndConditions::new(&[], &[], false, None, None).0.last(),
        Some(EndCondition::Timeout { .. })
    ));
    // Python's ValueErrors ('shared LingerTracker', 'positive') can't happen here: the env always passes its own
    // tracker's tail, and config validation rejects a non-positive no_progress_steps.
    for bad in [
        "no_progress_steps: 0",
        "no_progress_steps: -1",
        "no_damage_steps: 0",
    ] {
        let text = format!("name: t\nrooms: ['(1, 0x73)']\nsuccess: [exit-correct]\n{bad}\n");
        let err = SpecialistConfig::from_yaml(&text, "t.yaml", None).unwrap_err();
        let key = bad.split(':').next().unwrap();
        assert!(err.0.contains(key), "{err}");
    }
    // no_damage_steps: on by default (DEFAULT_NO_DAMAGE_STEPS), `null` turns it off.
    let base = "name: t\nrooms: ['(1, 0x73)']\nsuccess: [exit-correct]\n";
    let config = SpecialistConfig::from_yaml(base, "t.yaml", None).unwrap();
    assert_eq!(config.no_damage_steps, Some(env::config::DEFAULT_NO_DAMAGE_STEPS));
    let off = SpecialistConfig::from_yaml(&format!("{base}no_damage_steps: null\n"), "t.yaml", None).unwrap();
    assert_eq!(off.no_damage_steps, None);

    for progress in ["hit", "pickup", "heart", "healing", "approach", "exit"] {
        // (shared tracker, the specialist's NoRoomProgress condition)
        let mut env = (
            LingerTracker::default(),
            conditions.0.last().unwrap().clone(),
        );
        // One step prev -> curr: the tracker credits it, then NoRoomProgress checks the tail.
        let update = |env: &mut (LingerTracker, EndCondition),
                      prev: &LingerState,
                      curr: &LingerState,
                      step: LingerStep,
                      hearts_taken: u32| {
            env.0
                .update(prev, curr, &step, hearts_taken, || wavefront.clone());
            let mut change = block_change(Vec::new(), curr.full_location, &map);
            change.previous = view(prev.full_location, &map);
            let step_view = StepView {
                change: &change,
                prev_objective: &objective,
                map: &map,
                linger_tail_steps: env.0.tail_steps(),
            };
            env.1.check(&step_view)
        };
        let quiet = LingerStep {
            frames: 8,
            ..Default::default()
        };

        let prev = state(5, 0x73);
        env.0.reset(&prev, || wavefront.clone());
        assert_eq!(update(&mut env, &prev, &prev, quiet, 0), None, "{progress}");
        assert_eq!(update(&mut env, &prev, &prev, quiet, 0), None, "{progress}");
        assert_eq!(env.0.tail_steps(), 2, "{progress}");

        // The threshold step makes progress: it must be credited before the check.
        let curr = state(
            if progress == "approach" { 4 } else { 5 },
            if progress == "exit" { 0x63 } else { 0x73 },
        );
        let step = LingerStep {
            enemies_hit: progress == "hit",
            items_gained: progress == "pickup",
            health_gained: if progress == "healing" { 1.0 } else { 0.0 },
            ..quiet
        };
        let hearts_taken = u32::from(progress == "heart");
        assert_eq!(
            update(&mut env, &prev, &curr, step, hearts_taken),
            None,
            "{progress}"
        );
        assert_eq!(env.0.tail_steps(), 0, "{progress}");

        let prev = curr;
        assert_eq!(update(&mut env, &prev, &prev, quiet, 0), None, "{progress}");
        assert_eq!(update(&mut env, &prev, &prev, quiet, 0), None, "{progress}");
        let ending = update(&mut env, &prev, &prev, quiet, 0);
        assert_eq!(ending.as_deref(), Some("no-room-progress"), "{progress}");
        assert_eq!(
            classify_ending(&ending.unwrap(), &[], &IndexMap::new()),
            Ok(SOFT_FAILURE)
        );

        env.1.clear();
        env.0.reset(&prev, || wavefront.clone());
        assert_eq!(update(&mut env, &prev, &prev, quiet, 0), None, "{progress}");
        assert_eq!(env.0.tail_steps(), 1, "{progress}");
    }
}
