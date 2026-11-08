//! `SpecialistEnv` end to end on real savestates (zcore): episodes, endings, outcomes, terminal rewards,
//! the info contract, objectives and routes, critic behaviour, RAM randomization, start states and capture.
//!
//! Ports of the emulator parts of tests/test_env.py, tests/test_nearby_items.py, tests/test_rewards.py and
//! tests/test_specialist.py.  States are the regenerated `TFZSTATE` start states in `triforce/states`
//! (`TfzStore`).  Python's `random` streams are not reproduced (the env draws from ChaCha8), so tests assert
//! properties, not particular draws.  Every test skips when the ROM is missing.

mod common;

use common::{MachineState, TfzStore, ZCore, assert_obs_valid, open_core};
use env::config::{
    GameData, HpBound, ItemsConfig, RamRandomization, RewardConfig, SpecialistConfig,
};
use env::critic::{Critique, GameplayCritic};
use env::curriculum::{Curriculum, CurriculumSpec};
use env::rewards::{
    Outcome, PENALTY_TERMINAL_FAILURE, REWARD_MEDIUM, REWARD_TERMINAL_SUCCESS, StepRewards,
    TERMINAL_FAILURE_PENALTY_VALUE, success_reward_value,
};
use env::specialist_env::{EnvOptions, SpecialistEnv, StepInfo, StepResult};
use env::start_states::{SavestateCapture, ram_randomization_values};
use env::{EnvError, StateStore};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::BTreeSet;
use std::sync::{Arc, LazyLock};
use zelda::actions::{MASK_SIZE, NUM_DIRECTIONS, NUM_DURATIONS, NUM_KINDS};
use zelda::enums::{SwordKind, item};
use zelda::objectives::{Objective, ObjectiveKind};
use zelda::room::Target;
use zelda::room_entry::is_pre_entry;
use zelda::{
    Action, ActionKind, ActionMask, Change, Direction, GameView, LoadOptions, MapLocation, Nes,
    Session, SessionSnapshot, Var,
};

type Env = SpecialistEnv<ZCore, TfzStore>;
type S = Session<ZCore>;

const STEP_INFO_KEYS: [&str; 11] = [
    "frames",
    "interrupt",
    "action",
    "rewards",
    "room",
    "health",
    "max_health",
    "steps",
    "total_frames",
    "specialist",
    "route",
];

static DATA: LazyLock<Arc<GameData>> =
    LazyLock::new(common::game_data);

fn data() -> &'static Arc<GameData> {
    &DATA
}

// --- helpers (tests/test_env.py, tests/conftest.py) ---

/// An env on zcore.  `update` overrides config fields; RAM randomization is cleared first so tests are
/// deterministic (`make_env`).
fn make_env_with(
    nes: ZCore,
    specialist: &str,
    opts: EnvOptions,
    update: impl FnOnce(&mut SpecialistConfig),
) -> Env {
    let mut config = (*data().specialists[specialist]).clone();
    config.ram_randomization = RamRandomization::default();
    update(&mut config);
    SpecialistEnv::new(
        Arc::new(config),
        data().clone(),
        nes,
        TfzStore::new(),
        opts,
    )
    .expect("env")
}

fn make_env(
    nes: ZCore,
    specialist: &str,
    start_state: Option<&str>,
    update: impl FnOnce(&mut SpecialistConfig),
) -> Env {
    let opts = EnvOptions {
        start_state: start_state.map(str::to_owned),
        ..Default::default()
    };
    make_env_with(nes, specialist, opts, update)
}

/// The emulator back from an env.
fn release(env: Env) -> ZCore {
    env.into_parts().0.into_nes()
}

fn hp(lo: f64, hi: f64) -> RamRandomization {
    RamRandomization {
        hp: Some((HpBound::Fraction(lo), HpBound::Fraction(hi))),
        ..Default::default()
    }
}

fn act(env: &mut Env, text: &str) -> StepResult {
    env.step(Action::parse(text).unwrap()).unwrap()
}

fn term(info: &StepInfo, name: &str) -> Option<f64> {
    info.rewards.iter().find(|(k, _)| *k == name).map(|t| t.1)
}

fn has_term(info: &StepInfo, name: &str) -> bool {
    term(info, name).is_some()
}

fn term_sum(info: &StepInfo) -> f64 {
    info.rewards.iter().map(|t| t.1).sum()
}

fn pbrs_total(info: &StepInfo) -> f64 {
    term(info, "reward-pbrs-movement").unwrap_or(0.0)
        + term(info, "penalty-pbrs-movement").unwrap_or(0.0)
}

fn objective(env: &Env) -> &Objective {
    &env.objective().expect("episode started").objective
}

fn next_rooms(env: &Env) -> BTreeSet<MapLocation> {
    objective(env).next_rooms.clone()
}

fn rooms(list: &[(u8, u8)]) -> BTreeSet<MapLocation> {
    list.iter()
        .map(|&(l, r)| MapLocation::new(l, r, false))
        .collect()
}

fn state(env: &Env) -> &Arc<GameView> {
    env.session().state()
}

fn approx(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-6 * b.abs().max(1.0)
}

fn assert_info_contract(info: &StepInfo) {
    let json = info.to_json();
    for key in STEP_INFO_KEYS {
        assert!(json.get(key).is_some(), "info lacks {key}: {json}");
    }
}

/// Uniform over legal kinds, then legal direction/duration for that kind.
fn random_legal_action(flat: &[bool; MASK_SIZE], rng: &mut ChaCha8Rng) -> [usize; 3] {
    let mask = ActionMask::from_flat(flat);
    let kinds: Vec<usize> = (0..NUM_KINDS).filter(|&i| mask.kind[i]).collect();
    let kind = kinds[rng.random_range(0..kinds.len())];
    let dirs: Vec<usize> = (0..NUM_DIRECTIONS)
        .filter(|&i| mask.direction[kind][i])
        .collect();
    let durs: Vec<usize> = (0..NUM_DURATIONS)
        .filter(|&i| mask.duration[kind][i])
        .collect();
    let mut pick = |v: &[usize]| {
        if v.is_empty() {
            0
        } else {
            v[rng.random_range(0..v.len())]
        }
    };
    let dir = pick(&dirs);
    let dur = pick(&durs);
    [kind, dir, dur]
}

/// One env step of a 1-ply lookahead walker: tries every legal MOVE 8 on a session snapshot and takes the one that
/// leaves the room through a wanted exit or gets closest on the PBRS wavefront (objective exits/targets).
fn greedy_step(env: &mut Env) -> Result<StepResult, EnvError> {
    let attached = env.objective().expect("episode started").clone();
    let wanted = attached.objective.next_rooms;
    let wavefront = attached.wavefront;
    let snap = env.session().snapshot();
    let mut best: Option<(i32, Action)> = None;
    for direction in env.session().mask().directions(ActionKind::Move) {
        let action = Action::new(ActionKind::Move, direction, 8).unwrap();
        let change = env.session_mut().act(action)?;
        let score = if change.gained_triforce() {
            -1
        } else if change.changed_location() {
            if wanted.contains(&change.state.full_location()) {
                -1
            } else {
                9999
            }
        } else {
            wavefront
                .as_ref()
                .and_then(|w| w.get(change.state.link.tile()))
                .unwrap_or(999)
        };
        env.session_mut().restore(&snap);
        if best.is_none_or(|(s, _)| score < s) {
            best = Some((score, action));
        }
    }
    env.step(best.expect("a legal MOVE").1)
}

/// Steps until the episode ends; returns every step's result.
fn run_until_done(
    env: &mut Env,
    mut policy: impl FnMut(&mut Env) -> Result<StepResult, EnvError>,
    max_steps: usize,
) -> Result<Vec<StepResult>, EnvError> {
    let mut results = Vec::new();
    for _ in 0..max_steps {
        let result = policy(env)?;
        let done = result.terminated || result.truncated;
        results.push(result);
        if done {
            return Ok(results);
        }
    }
    panic!(
        "episode did not end in {max_steps} steps (room {:?})",
        state(env).full_location()
    );
}

fn last_info(results: Vec<StepResult>) -> (f64, StepInfo) {
    let last = results.into_iter().last().expect("a step");
    (last.reward, last.info)
}

/// MOVE 8 toward the nearest active enemy along the larger axis (to get hit).
fn chase_action(state: &GameView) -> &'static str {
    let Some(enemy) = state
        .active_enemies()
        .min_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap())
    else {
        return "WAIT 8";
    };
    let dx = enemy.position.x - state.link.position.x;
    let dy = enemy.position.y - state.link.position.y;
    if dx.abs() > dy.abs() {
        if dx > 0 { "MOVE E 8" } else { "MOVE W 8" }
    } else if dy > 0 {
        "MOVE S 8"
    } else {
        "MOVE N 8"
    }
}

/// `chase_action`, or WAIT 8 when that move is masked.
fn chase(session: &S) -> Action {
    let action = Action::parse(chase_action(session.state())).unwrap();
    if session.mask().is_allowed(&action) {
        action
    } else {
        Action::parse("WAIT 8").unwrap()
    }
}

/// One env step toward the nearest enemy (WAIT when that move is masked).
fn chase_step(env: &mut Env) -> Result<StepResult, EnvError> {
    let action = chase(env.session());
    env.step(action)
}

/// `GameSession.load(name, randomize_rng=..., fresh_room=..., ram_overrides=...)` on a bare session.
fn load(s: &mut S, name: &str, opts: LoadOptions) -> Arc<GameView> {
    let machine = TfzStore::new().load(name).expect("state");
    s.load(&machine, &opts, &mut ChaCha8Rng::seed_from_u64(0))
        .expect("load")
        .clone()
}

fn session(nes: ZCore) -> S {
    Session::new(nes, data().map.clone())
}

fn act_until(
    s: &mut S,
    text: &str,
    pred: impl Fn(&Change) -> bool,
    max_steps: usize,
) -> Vec<Change> {
    let mut changes = Vec::new();
    for _ in 0..max_steps {
        let change = s.act(Action::parse(text).unwrap()).unwrap();
        let done = pred(&change);
        changes.push(change);
        if done {
            return changes;
        }
    }
    panic!(
        "'{text}' never satisfied the predicate in {max_steps} steps (at {:?}, pos {:?})",
        s.state().full_location(),
        s.state().link.position
    );
}

/// A boomerang shot already in Link's shield box in D1 0x73 (`test_core._place_shield_projectile`).
fn place_shield_projectile(s: &mut S, slots: &[u16]) {
    load(s, "1_73s_0", LoadOptions::default());
    s.act(Action::parse("MOVE N 16").unwrap()).unwrap(); // off the entry door, in playable mode before RAM placement
    s.apply_ram(&[
        (Var::LinkX, 128),
        (Var::LinkY, 144),
        (Var::LinkDirection, 1),
        (Var::HeartsAndContainers, 0x22),
        (Var::PartialHearts, 255),
    ]);
    let nes = s.nes_mut();
    for address in [0xAC, 0x3D, 0x4F0, 0x66C, 0x512] {
        nes.poke(address, 0);
    }
    for &slot in slots {
        for (base, value) in [
            (0x34F, 0x5C),
            (0xAC, 0x17),
            (0x70, 140),
            (0x84, 144),
            (0x98, 2),
            (0x4BF, 0xE1),
            (0x3D0, 2),
            (0x3BC, 0xA0),
            (0x3A8, 0),
            (0x380, 0x51),
            (0x394, 0),
            (0x28, 0),
            (0x3D, 0),
            (0x405, 0),
            (0x492, 0),
        ] {
            nes.poke(base + slot, value);
        }
    }
    nes.poke(0x34C, slots.len() as u8);
    s.apply_ram(&[]);
}

fn no_objective() -> Objective {
    Objective {
        kind: ObjectiveKind::None,
        targets: BTreeSet::new(),
        next_rooms: BTreeSet::new(),
        pbrs_targets: BTreeSet::new(),
    }
}

/// `critique_gameplay` of one change outside an env (no objective, PBRS off unless the critic has it).
fn critique(critic: &mut GameplayCritic, change: &Change, rewards: &mut StepRewards) {
    let none = no_objective();
    critic.critique(
        &Critique {
            change,
            map: &data().map,
            prev_objective: &none,
            objective: &none,
            prev_wavefront: None,
            quiet_steps: 0,
        },
        rewards,
    );
}

// --- tests/test_env.py ---

#[test]
fn spaces_and_reset_contract() {
    let Some(nes) = open_core() else { return };
    let mut env = make_env(nes, "stalfos-keese-gel", Some("1_73s_0"), |_| {});
    let (obs, info) = env.reset(Some(1), None).unwrap();
    assert_obs_valid(&obs, env.session().mask());
    assert_eq!([NUM_KINDS, NUM_DIRECTIONS, NUM_DURATIONS], [7, 8, 4]);
    assert_info_contract(&info);
    assert_eq!(&*info.specialist, "stalfos-keese-gel");
    assert_eq!(info.room, (1, 0x73, false));
}

/// stalfos-keese-gel from a start state with random legal actions: every episode ends with a classified outcome,
/// every step returns valid observations and the info contract.
#[test]
fn random_legal_episode_ends_classified() {
    let Some(nes) = open_core() else { return };
    let mut env = make_env(nes, "stalfos-keese-gel", Some("1_73s_0"), |c| {
        c.max_steps = 150
    });
    let mut rng = ChaCha8Rng::seed_from_u64(3);
    let (mut obs, _) = env.reset(Some(3), None).unwrap();
    let mut outcomes = Vec::new();
    for _ in 0..600 {
        let r = env
            .step_indices(random_legal_action(&obs.action_mask, &mut rng))
            .unwrap();
        assert_info_contract(&r.info);
        assert!(r.reward.is_finite() && r.info.frames > 0);
        assert_obs_valid(&r.obs, env.session().mask());
        obs = r.obs;
        if r.terminated || r.truncated {
            let t = r.info.terminal.as_ref().expect("terminal info");
            assert!(["success", "hard-failure", "soft-failure"].contains(&t.outcome));
            assert_eq!(r.truncated, t.ending == "max-steps");
            let h = &t.hearts;
            assert_eq!(h.dropped, h.taken + h.left + h.expired, "{h:?}");
            outcomes.push(t.outcome);
            if outcomes.len() == 2 {
                break;
            }
            obs = env.reset(None, None).unwrap().0;
        }
    }
    assert!(!outcomes.is_empty(), "no episode ended");
}

/// An ow-start env in 0x77 with the sword (so N is the correct exit, 0x78 E the alternate): the regenerated
/// 0_77c_0 (just out of the sword cave; no state enters 0x77 from N or W as the legacy 0_77n/0_77w did) with
/// Link moved to the start-screen spawn, clear of the cave mouth's rocks.
fn sworded_ow_start(nes: ZCore, seed: u64, update: impl FnOnce(&mut SpecialistConfig)) -> Env {
    let mut env = make_env(nes, "ow-start", Some("0_77c_0"), update);
    env.reset(Some(seed), None).unwrap();
    env.session_mut()
        .apply_ram(&[(Var::LinkX, 0x78), (Var::LinkY, 0x8D)]);
    let snapshot = env.session().snapshot();
    env.restore(&snapshot, "0_77c_0").unwrap();
    env
}

#[test]
fn correct_exit_is_success_with_health_scaled_reward() {
    let Some(nes) = open_core() else { return };
    let mut env = sworded_ow_start(nes, 0, |c| c.ram_randomization = hp(0.5, 0.5));
    assert_eq!(state(&env).health(), 1.5);
    let (reward, info) = last_info(run_until_done(&mut env, greedy_step, 400).unwrap());
    assert_eq!(
        (info.ending(), info.outcome()),
        (Some("exit-correct"), Some("success"))
    );
    let terminal = term(&info, REWARD_TERMINAL_SUCCESS).unwrap();
    assert!(
        approx(terminal, success_reward_value(1.5, 3.0)) && approx(terminal, 15.0),
        "{terminal}"
    );
    assert!(
        approx(reward, term_sum(&info).min(20.0)),
        "{reward} vs {:?}",
        info.rewards
    );
}

#[test]
fn full_health_success_pays_full_terminal() {
    let Some(nes) = open_core() else { return };
    let mut env = sworded_ow_start(nes, 0, |_| {});
    let (_, info) = last_info(run_until_done(&mut env, greedy_step, 400).unwrap());
    assert!(
        approx(term(&info, REWARD_TERMINAL_SUCCESS).unwrap(), 20.0),
        "{:?}",
        info.rewards
    );
}

#[test]
fn wrong_exit_is_soft_failure() {
    let Some(nes) = open_core() else { return };
    let mut env = sworded_ow_start(nes, 0, |_| {});
    assert!(!next_rooms(&env).contains(&MapLocation::new(0, 0x76, false)));
    let results = run_until_done(
        &mut env,
        |e| e.step(Action::parse("MOVE W 16").unwrap()),
        20,
    )
    .unwrap();
    let (reward, info) = last_info(results);
    assert_eq!(info.room, (0, 0x76, false));
    assert_eq!(
        (info.ending(), info.outcome()),
        (Some("exit-wrong"), Some("soft-failure"))
    );
    assert_eq!(
        term(&info, PENALTY_TERMINAL_FAILURE),
        Some(-TERMINAL_FAILURE_PENALTY_VALUE)
    );
    assert!(reward <= -19.0, "{reward}");
}

#[test]
fn death_is_hard_failure() {
    let Some(nes) = open_core() else { return };
    // 0.01 rounds up to half a heart.
    let mut env = make_env(nes, "goriya", Some("1_44w_0"), |c| {
        c.ram_randomization = hp(0.01, 0.01)
    });
    env.reset(Some(2), None).unwrap();
    assert_eq!(state(&env).health(), 0.5);
    act(&mut env, "MOVE E 16");
    let (reward, info) = last_info(run_until_done(&mut env, chase_step, 400).unwrap());
    assert_eq!(
        (info.ending(), info.outcome()),
        (Some("death"), Some("hard-failure"))
    );
    let others: f64 = info
        .rewards
        .iter()
        .filter(|(k, _)| *k != PENALTY_TERMINAL_FAILURE)
        .map(|t| t.1)
        .sum();
    assert!(
        (reward - (-20.0 + others)).abs() <= 1.0,
        "{reward} vs {:?}",
        info.rewards
    );
    assert!(
        info.rewards.iter().all(|(_, v)| *v <= 0.0),
        "hard failures drop positive rewards: {:?}",
        info.rewards
    );
}

/// Deny-by-default: an ow-start variant that only lists triforce-collected can't classify exit-correct.
#[test]
fn unlisted_ending_raises() {
    let Some(nes) = open_core() else { return };
    let mut env = sworded_ow_start(nes, 0, |c| {
        c.success = vec!["triforce-collected".into()]
    });
    match run_until_done(&mut env, greedy_step, 400) {
        Err(EnvError::Unclassified(e)) => assert!(e.0.contains("exit-correct"), "{e}"),
        Err(e) => panic!("expected an unclassified ending, got {e}"),
        Ok(results) => panic!(
            "episode ended classified: {:?}",
            results.last().unwrap().info.ending()
        ),
    }
}

#[test]
fn max_steps_truncates_without_terminal_reward() {
    let Some(nes) = open_core() else { return };
    let mut env = make_env(nes, "stalfos-keese-gel", Some("1_73s_0"), |c| {
        c.max_steps = 3
    });
    env.reset(Some(0), None).unwrap();
    let mut last = None;
    for _ in 0..3 {
        last = Some(act(&mut env, "WAIT 8"));
    }
    let r = last.unwrap();
    assert!(r.truncated && !r.terminated);
    assert_eq!(
        (r.info.ending(), r.info.outcome()),
        (Some("max-steps"), Some("soft-failure"))
    );
    assert!(
        !r.info
            .rewards
            .iter()
            .any(|(k, _)| k.starts_with("reward-terminal") || k.starts_with("penalty-terminal")),
        "{:?}",
        r.info.rewards
    );
    let err = env.step(Action::parse("WAIT 8").unwrap()).unwrap_err();
    assert!(matches!(err, EnvError::EpisodeFinished));
    // The message Python's EpisodeFinishedError carries.
    assert_eq!(
        err.to_string(),
        "step() called on a finished episode; call reset()"
    );
}

#[test]
fn no_room_progress_truncates_at_threshold_and_resets_shared_tracker() {
    let Some(nes) = open_core() else { return };
    let mut env = make_env(nes, "stalfos-keese-gel", Some("1_73s_0"), |c| {
        c.no_progress_steps = Some(3);
        c.max_steps = 100;
    });
    env.reset(Some(0), None).unwrap();
    let mut info = None;
    for step in 1..=3u32 {
        let r = act(&mut env, "WAIT 8");
        assert!(!r.terminated);
        assert_eq!(r.truncated, step == 3);
        assert_eq!(env.linger.tail_steps(), step);
        info = Some(r.info);
    }
    let info = info.unwrap();
    let t = info.terminal.as_ref().unwrap();
    assert_eq!(
        (t.ending.as_str(), t.outcome),
        ("no-room-progress", "soft-failure")
    );
    assert!(
        t.linger.tail_steps == 3 && t.linger.steps == 3,
        "{:?}",
        t.linger
    );
    assert!(t.bootstrap_truncation);
    assert_eq!(
        info.to_json()["bootstrap_truncation"],
        serde_json::json!(true)
    );
    assert!(
        !info
            .rewards
            .iter()
            .any(|(k, _)| k.starts_with("reward-terminal") || k.starts_with("penalty-terminal"))
    );
    assert!(matches!(
        env.step(Action::parse("WAIT 8").unwrap()),
        Err(EnvError::EpisodeFinished)
    ));
    env.reset(Some(0), None).unwrap();
    assert_eq!(env.linger.tail_steps(), 0);
    let r = act(&mut env, "WAIT 8");
    assert!(!(r.terminated || r.truncated));
}

#[test]
fn no_room_progress_disabled_preserves_position_timeout() {
    let Some(nes) = open_core() else { return };
    let mut env = make_env(nes, "stalfos-keese-gel", Some("1_73s_0"), |c| {
        c.max_steps = 100
    });
    env.reset(Some(0), None).unwrap();
    let mut info = None;
    for step in 1..=50 {
        let r = act(&mut env, "WAIT 8");
        assert!(!r.truncated);
        assert_eq!(
            r.terminated,
            step == 50,
            "step {step}: {:?}",
            r.info.ending()
        );
        info = Some(r.info);
    }
    let info = info.unwrap();
    assert_eq!(
        (info.ending(), info.outcome()),
        (Some("stuck"), Some("hard-failure"))
    );
    assert!(
        info.terminal
            .as_ref()
            .is_some_and(|t| !t.bootstrap_truncation)
    );
}

#[test]
fn no_damage_truncates_and_bootstraps() {
    let Some(nes) = open_core() else { return };
    let mut env = make_env(nes, "stalfos-keese-gel", Some("1_73s_0"), |c| {
        c.no_damage_steps = Some(3);
        c.max_steps = 100;
    });
    env.reset(Some(0), None).unwrap();
    for step in 1..=3u32 {
        let r = act(&mut env, "WAIT 8");
        assert!(!r.terminated);
        assert_eq!(r.truncated, step == 3);
        if step == 3 {
            let t = r.info.terminal.as_ref().unwrap();
            assert_eq!((t.ending.as_str(), t.outcome), ("no-damage", "soft-failure"));
            assert!(t.bootstrap_truncation);
        }
    }
}

#[test]
fn no_room_progress_keeps_episode_cap() {
    let Some(nes) = open_core() else { return };
    let mut env = make_env(nes, "stalfos-keese-gel", Some("1_73s_0"), |c| {
        c.no_progress_steps = Some(10);
        c.max_steps = 2;
    });
    env.reset(Some(0), None).unwrap();
    act(&mut env, "WAIT 8");
    let r = act(&mut env, "WAIT 8");
    assert!(r.truncated && !r.terminated && r.info.ending() == Some("max-steps"));
}

#[test]
fn masked_action_raises() {
    let Some(nes) = open_core() else { return };
    let mut env = make_env(nes, "stalfos-keese-gel", Some("1_73s_0"), |_| {});
    env.reset(Some(0), None).unwrap();
    act(&mut env, "MOVE N 16"); // off the S door tile, where weapons are masked anyway
    assert!(!env.session().mask().kind[ActionKind::Sword.index()]); // empty room: no weapons
    let sword = Action::new(ActionKind::Sword, Direction::N, 0).unwrap();
    assert!(matches!(
        env.step_indices(sword.to_indices()),
        Err(EnvError::InvalidAction(_))
    ));
}

#[test]
fn start_state_outside_rooms_is_rejected() {
    let Some(nes) = open_core() else { return };
    let mut env = make_env(nes, "stalfos-keese-gel", Some("1_44w_0"), |_| {});
    match env.reset(Some(0), None) {
        Err(EnvError::StartState(m)) => assert!(m.contains("0x44"), "{m}"),
        Err(e) => panic!("expected a start-state error, got {e}"),
        Ok(_) => panic!("1_44w loaded into stalfos-keese-gel"),
    }
}

/// In the sword cave (no sword) the objective is the treasure; after taking it, S out of the cave is a correct exit
/// that stays inside ow-start, and N from the start screen ends the episode with success.
#[test]
fn sword_cave_objective_and_exit() {
    let Some(nes) = open_core() else { return };
    let mut env = make_env(nes, "ow-start", Some("0_77cave_0"), |_| {});
    env.reset(Some(0), None).unwrap();
    assert_eq!(objective(&env).kind, ObjectiveKind::Treasure);
    assert!(next_rooms(&env).is_empty());
    let mut gained = None;
    for _ in 0..40 {
        let r = greedy_step(&mut env).unwrap();
        assert!(!r.terminated);
        if state(&env).sword() != SwordKind::None {
            gained = Some(r.info);
            break;
        }
    }
    let gained = gained.expect("Link never took the sword");
    assert!(
        has_term(&gained, "reward-gained-sword"),
        "{:?}",
        gained.rewards
    );
    assert_eq!(next_rooms(&env), rooms(&[(0, 0x77)]));

    let mut last = None;
    for _ in 0..60 {
        let r = act(&mut env, "MOVE S 16");
        let arrived = r.info.room == (0, 0x77, false);
        last = Some(r);
        if arrived {
            break;
        }
    }
    let r = last.unwrap();
    assert_eq!(r.info.room, (0, 0x77, false));
    assert!(
        !r.terminated && has_term(&r.info, "reward-new-location"),
        "{:?}",
        r.info.rewards
    );
    let (_, info) = last_info(run_until_done(&mut env, greedy_step, 400).unwrap());
    assert_eq!(
        (info.ending(), info.outcome()),
        (Some("exit-correct"), Some("success"))
    );
}

#[test]
fn triforce_pickup_is_success_in_the_triforce_room() {
    let Some(nes) = open_core() else { return };
    let mut env = make_env(nes, "stalfos-keese-gel", Some("1_36w_0"), |_| {});
    env.reset(Some(0), None).unwrap();
    assert_eq!(objective(&env).kind, ObjectiveKind::Treasure);
    let (_, info) = last_info(run_until_done(&mut env, greedy_step, 100).unwrap());
    assert_eq!(
        (info.ending(), info.outcome()),
        (Some("triforce-collected"), Some("success"))
    );
}

#[test]
fn items_config_masks_bombs_even_when_link_has_them() {
    let Some(mut nes) = open_core() else { return };
    for bombs_on in [true, false] {
        let items = ItemsConfig {
            bombs: bombs_on,
            ..data().specialists["goriya"].items.clone()
        };
        let mut env = make_env(nes, "goriya", Some("1_44w_0"), |c| {
            c.ram_randomization = RamRandomization {
                bombs: Some((4, 4)),
                ..Default::default()
            };
            c.items = items;
        });
        env.reset(Some(0), None).unwrap();
        // Off the door tile (items can't be used there) and the Goriyas out of their clouds.
        let mut obs = None;
        for _ in 0..10 {
            obs = Some(act(&mut env, "MOVE E 16").obs);
            if state(&env).link.position.x >= 32 && state(&env).has_active_enemies() {
                break;
            }
        }
        let obs = obs.unwrap();
        assert!(state(&env).bombs() == 4 && state(&env).has_active_enemies());
        assert_eq!(
            env.session().mask().kind[ActionKind::Bomb.index()],
            bombs_on
        );
        assert_eq!(
            ActionMask::from_flat(&obs.action_mask).kind[ActionKind::Bomb.index()],
            bombs_on
        );
        nes = release(env);
    }
}

#[test]
fn route_shutter_requires_combat_without_a_room_treasure() {
    let Some(nes) = open_core() else { return };
    let mut env = make_env(nes, "d2-rope", Some("2_2es_0"), |_| {});
    env.reset(Some(0), None).unwrap();
    assert_eq!(objective(&env).kind, ObjectiveKind::Fight);
    assert!(next_rooms(&env).is_empty());
}

#[test]
fn open_route_ignores_optional_combat() {
    let Some(mut nes) = open_core() else { return };
    for (start_state, next_room) in [("2_6es_0", 0x5e), ("2_6ds_0", 0x6e)] {
        let mut env = make_env(nes, "d2-rope", Some(start_state), |_| {});
        env.reset(Some(0), None).unwrap();
        assert_eq!(objective(&env).kind, ObjectiveKind::Move, "{start_state}");
        assert_eq!(next_rooms(&env), rooms(&[(2, next_room)]), "{start_state}");
        nes = release(env);
    }
}

/// 0x43's route exit is the locked E door to 0x44.  Without a key the shortest path to the triforce goes N for the
/// Stalfos room's key (0x33, off the route); the objective still only accepts 0x44.
#[test]
fn route_room_only_offers_its_route_exit() {
    let Some(mut nes) = open_core() else { return };
    for keys in [0, 1] {
        let mut env = make_env(nes, "stalfos-keese-gel", Some("1_43w_0"), |c| {
            c.ram_randomization = RamRandomization {
                keys: Some((keys, keys)),
                ..Default::default()
            };
        });
        env.reset(Some(0), None).unwrap();
        assert_eq!(state(&env).keys() as i64, keys);
        assert_eq!(next_rooms(&env), rooms(&[(1, 0x44)]), "keys {keys}");
        nes = release(env);
    }
}

/// Entering Dungeon 1 with no keys, 0x72 (W) and 0x74 (E) are equally short key detours; the route takes 0x72 first.
#[test]
fn first_visit_to_0x73_follows_the_route_order() {
    let Some(nes) = open_core() else { return };
    let mut env = make_env(nes, "stalfos-keese-gel", Some("1_73s_0"), |_| {});
    env.reset(Some(0), None).unwrap();
    assert_eq!(state(&env).keys(), 0);
    assert_eq!(next_rooms(&env), rooms(&[(1, 0x72)]));
}

/// ow-octorok lists enter-room:1:0x73 (the D1 stairs in 0x37) as success.  After Dungeon 1 (0_37c_2: Link just came
/// out of those stairs with the triforce) the objective wants E, so walking back down is exit-wrong.
#[test]
fn enter_room_success_needs_the_objective() {
    let Some(nes) = open_core() else { return };
    let mut env = make_env(nes, "ow-octorok", Some("0_37c_2"), |_| {});
    env.reset(Some(0), None).unwrap();
    assert_eq!(next_rooms(&env), rooms(&[(0, 0x38)]));
    let mut info = None;
    for action in ["MOVE S 16", "MOVE S 16", "MOVE N 16", "MOVE N 16"] {
        let r = act(&mut env, action);
        let done = r.terminated || r.truncated;
        info = Some(r.info);
        if done {
            break;
        }
    }
    let info = info.unwrap();
    assert_eq!(info.room, (1, 0x73, false));
    assert_eq!(
        (info.ending(), info.outcome()),
        (Some("exit-wrong"), Some("soft-failure"))
    );
}

/// 0x38 from W, just out of Dungeon 1: milestone1 goes N toward D2, milestone1-fairy S toward the fairy pond.
#[test]
fn after_dungeon_1_each_route_leaves_0x38_its_own_way() {
    let Some(mut nes) = open_core() else { return };
    for (route, wanted) in [("milestone1", 0x28), ("milestone1-fairy", 0x48)] {
        let mut env = make_env(nes, "ow-zora", Some("0_38w_2"), |c| {
            c.route = Some(route.into())
        });
        let (_, info) = env.reset(Some(0), None).unwrap();
        assert_eq!(info.route, route);
        assert_eq!(next_rooms(&env), rooms(&[(0, wanted)]), "{route}");
        nes = release(env);
    }
}

/// Without a pinned route, 0x38-from-W resets play either route; rooms where the routes agree play milestone1.
#[test]
fn a_room_the_routes_leave_differently_trains_both() {
    let Some(nes) = open_core() else { return };
    let mut env = make_env(nes, "ow-zora", Some("0_38w_2"), |_| {});
    let mut seen = BTreeSet::new();
    for seed in 0..16 {
        let (_, info) = env.reset(Some(seed), None).unwrap();
        seen.insert((info.route.clone(), next_rooms(&env)));
    }
    let want: BTreeSet<_> = [
        ("milestone1".to_owned(), rooms(&[(0, 0x28)])),
        ("milestone1-fairy".to_owned(), rooms(&[(0, 0x48)])),
    ]
    .into_iter()
    .collect();
    assert_eq!(seen, want);
    let mut env = make_env(release(env), "ow-octorok", Some("0_37c_2"), |_| {});
    let (_, info) = env.reset(Some(0), None).unwrap();
    assert_eq!(info.route, "milestone1");
}

/// 0x39 (milestone1-fairy's waypoint): below full hearts no exit is correct and the pond edge is the target; walking
/// N to it fills every heart (Z_04 UpdatePondFairy), and then S is the exit.
#[test]
fn fairy_pond_holds_the_exit_until_the_hearts_are_full() {
    let Some(nes) = open_core() else { return };
    let mut env = make_env(nes, "ow-start", Some("0_39s_2"), |c| {
        c.ram_randomization = hp(0.25, 0.25)
    });
    let (_, info) = env.reset(Some(0), None).unwrap();
    assert_eq!(info.route, "milestone1-fairy");
    assert!(state(&env).health() < state(&env).max_health() as f64);
    assert_eq!(objective(&env).kind, ObjectiveKind::Treasure);
    assert!(next_rooms(&env).is_empty());
    for _ in 0..12 {
        let r = act(&mut env, "MOVE N 8");
        assert!(!(r.terminated || r.truncated), "{:?}", r.info.ending());
        if state(&env).is_health_full() {
            break;
        }
    }
    assert_eq!(state(&env).health(), state(&env).max_health() as f64);
    assert_eq!(next_rooms(&env), rooms(&[(0, 0x49)]));
}

/// 0x49 from S goes N to the pond; back from the pond (from N) the waypoint is done and the route goes S/E.
#[test]
fn fairy_route_heads_to_the_pond_then_east() {
    let Some(mut nes) = open_core() else { return };
    for (start_state, wanted) in [("0_49s_2", 0x39), ("0_49n_2", 0x59)] {
        let mut env = make_env(nes, "ow-octorok", Some(start_state), |_| {});
        let (_, info) = env.reset(Some(0), None).unwrap();
        assert_eq!(info.route, "milestone1-fairy", "{start_state}");
        assert_eq!(next_rooms(&env), rooms(&[(0, wanted)]), "{start_state}");
        nes = release(env);
    }
}

/// 0x5c's lower-left pocket only opens S.  Link is 2 tiles tall, so the wavefront from the E exit must not reach it
/// along the off-map row under the bottom trees.
#[test]
fn wavefront_has_no_lane_below_the_bottom_tree_line() {
    let Some(nes) = open_core() else { return };
    let mut s = session(nes);
    let opts = LoadOptions {
        randomize_rng: true,
        fresh_room: true,
        ..Default::default()
    };
    let state = load(&mut s, "0_5cw_0", opts);
    let exits: Vec<Target> = state
        .room()
        .exits
        .get(Direction::E)
        .expect("an E exit")
        .iter()
        .map(|&t| Target::Tile(t))
        .collect();
    let wavefront = state.room().wavefront(&exits, &[], &[]);
    assert!(wavefront.get(state.link.tile()).is_some()); // the upper corridor reaches the E exit
    assert!(wavefront.get((2, 16)).is_none()); // the pocket doesn't
}

/// 1_73e (2 keys) exits N through the locked door into 0x63, another stalfos-keese-gel room: one_room ends the
/// episode there as exit-correct, otherwise it continues.
#[test]
fn one_room_episode_ends_at_the_first_correct_exit() {
    let Some(mut nes) = open_core() else { return };
    for one_room in [true, false] {
        let mut env = make_env(nes, "stalfos-keese-gel", Some("1_73e_0"), |c| {
            c.one_room = one_room
        });
        env.reset(Some(0), None).unwrap();
        let mut infos: Vec<StepInfo> = Vec::new();
        while !infos.iter().any(|i| i.room == (1, 0x63, false)) {
            assert!(infos.len() < 60 && !infos.last().is_some_and(|i| i.terminal.is_some()));
            infos.push(greedy_step(&mut env).unwrap().info);
        }
        let want = if one_room { Some("exit-correct") } else { None };
        assert_eq!(infos.last().unwrap().ending(), want, "one_room {one_room}");
        nes = release(env);
    }
}

/// A saved half-heart frame is survivable only if loading leaves the first action to the policy.
#[test]
fn curriculum_reset_preserves_the_first_shield_turn() {
    let Some(nes) = open_core() else { return };
    let dir = tempfile::tempdir().unwrap();
    let track = dir.path().join("trajectories").join("1_73s_0");
    std::fs::create_dir_all(&track).unwrap();
    let store = TfzStore::new();
    let mut s = session(nes);
    load(&mut s, "1_73s_0", LoadOptions::default());
    s.act(Action::parse("MOVE N 16").unwrap()).unwrap();
    store
        .save(&track.join("0000.state"), &s.snapshot())
        .unwrap();
    s.apply_ram(&[
        (Var::LinkX, 96),
        (Var::LinkY, 144),
        (Var::LinkDirection, 4),
        (Var::HeartsAndContainers, 0x20),
        (Var::PartialHearts, 0x7F),
    ]);
    let nes = s.nes_mut();
    for address in [
        0xAC, 0x3D, 0x4F0, 0x66C, 0x512, 0x394, 0x3A8, 0x3F8, 0x56, 0x57,
    ] {
        nes.poke(address, 0);
    }
    for (base, value) in [
        (0x34F, 0x5C),
        (0xAC, 0x17),
        (0x70, 110),
        (0x84, 144),
        (0x98, 2),
        (0x4BF, 0xE1),
        (0x3D0, 2),
        (0x3BC, 0xA0),
        (0x3A8, 0),
        (0x380, 0x51),
        (0x394, 0),
        (0x28, 0),
        (0x3D, 0),
        (0x405, 0),
        (0x492, 0),
    ] {
        nes.poke(base + 11, value);
    }
    nes.poke(0x34C, 1);
    s.apply_ram(&[]);
    store
        .save(&track.join("0001.state"), &s.snapshot())
        .unwrap();
    std::fs::write(
        track.join("trajectory.json"),
        r#"{"start_state": "1_73s_0", "success": true}"#,
    )
    .unwrap();
    let spec = CurriculumSpec::new(
        track.parent().unwrap().to_path_buf(),
        dir.path().join("progress.json"),
        1.0,
        0.8,
    );
    let curriculum = Curriculum::new(spec).unwrap().shared();
    let opts = EnvOptions {
        curriculum: Some(curriculum),
        ..Default::default()
    };
    let mut env = make_env_with(s.into_nes(), "stalfos-keese-gel", opts, |_| {});
    let (_, info) = env.reset(Some(0), None).unwrap();
    assert!(
        info.start_state
            .as_deref()
            .is_some_and(|n| n.ends_with("0001.state")),
        "{:?}",
        info.start_state
    );
    assert!(
        info.health == 0.5 && !state(&env).game_over(),
        "{}",
        info.health
    );
    let r = act(&mut env, "MOVE E 4");
    assert!(!r.terminated && !r.truncated, "{:?}", r.info.ending());
    assert!(
        r.info.health == 0.5 && state(&env).link.position.x > 96,
        "health {} x {}",
        r.info.health,
        state(&env).link.position.x
    );
}

// --- tests/test_nearby_items.py ---

/// stalfos-keese-gel at 1_73e (one room) with Link hurt at (120, y) and `drops` (kind, x, y) placed in slots 3+;
/// the env restarts from that snapshot.
fn placed_env(
    nes: ZCore,
    drops: &[(i16, u8, u8)],
    y: u8,
    nearby: bool,
) -> (Env, SessionSnapshot<MachineState>) {
    let reward = RewardConfig {
        nearby_items: nearby,
        ..Default::default()
    };
    placed_env_with(nes, drops, y, reward, 0x21)
}

/// [`placed_env`] with any reward block and `$66F` (hearts and containers: 0x21 two of three, 0x22 full).
fn placed_env_with(
    nes: ZCore,
    drops: &[(i16, u8, u8)],
    y: u8,
    reward: RewardConfig,
    hearts_and_containers: u8,
) -> (Env, SessionSnapshot<MachineState>) {
    let mut env = make_env(nes, "stalfos-keese-gel", Some("1_73e_0"), |c| {
        c.one_room = true;
        c.reward = reward;
    });
    env.reset(Some(1), None).unwrap();
    let s = env.session_mut();
    s.apply_ram(&[
        (Var::LinkX, 120),
        (Var::LinkY, y),
        (Var::HeartsAndContainers, hearts_and_containers),
        (Var::PartialHearts, 0xFF),
    ]);
    let nes = s.nes_mut();
    nes.poke(0x53, 0); // placed off the doorway
    nes.poke(0x394, 0); // grid-aligned
    for (i, &(kind, x, drop_y)) in drops.iter().enumerate() {
        let slot = 3 + i as u16;
        for (address, value) in [
            (0x34F, 0x60),
            (0xAC, kind as u8),
            (0x70, x),
            (0x84, drop_y),
            (0x405, 0),
            (0x3A8, 0xC0),
        ] {
            nes.poke(address + slot, value);
        }
    }
    s.apply_ram(&[]);
    let snapshot = env.session().snapshot();
    env.restore(&snapshot, "1_73e_0").unwrap();
    (env, snapshot)
}

fn movement_reward(env: &mut Env, direction: &str) -> f64 {
    act(env, &format!("MOVE {direction} 4")); // seeds the critic's tile baseline
    pbrs_total(&act(env, &format!("MOVE {direction} 8")).info)
}

fn finish(env: &mut Env) -> StepInfo {
    for _ in 0..100 {
        let r = greedy_step(env).unwrap();
        if r.terminated || r.truncated {
            return r.info;
        }
    }
    panic!("greedy nearest-target walker did not leave the room");
}

#[test]
fn nearest_clock_then_heart_then_exit_without_changing_task() {
    let Some(nes) = open_core() else { return };
    let (mut env, snapshot) = placed_env(
        nes,
        &[(item::HEART, 72, 157), (item::CLOCK, 152, 157)],
        157,
        true,
    );
    let shaped = movement_reward(&mut env, "E");
    assert!(
        shaped > 0.0,
        "nearer clock wins over heart west and exit north: {shaped}"
    );
    assert_eq!(objective(&env).kind, ObjectiveKind::Move);
    let info = finish(&mut env);
    let t = info.terminal.as_ref().unwrap();
    assert_eq!(t.ending, "exit-correct");
    assert!(
        t.hearts.taken == 1 && t.hearts.healed == 1.0,
        "{:?}",
        t.hearts
    );
    assert_eq!(info.health, 3.0);

    let (mut legacy, _) = placed_env(release(env), &[], 157, false);
    legacy.restore(&snapshot, "1_73e_0").unwrap();
    let shaped = movement_reward(&mut legacy, "E");
    assert!(shaped < 0.0, "{shaped}");
}

#[test]
fn nearer_exit_leaves_distant_helpful_drops() {
    let Some(nes) = open_core() else { return };
    let (mut env, _) = placed_env(
        nes,
        &[(item::HEART, 48, 189), (item::CLOCK, 200, 189)],
        125,
        true,
    );
    let shaped = movement_reward(&mut env, "N");
    assert!(shaped > 0.0, "{shaped}");
    let info = finish(&mut env);
    let t = info.terminal.as_ref().unwrap();
    assert_eq!(t.ending, "exit-correct");
    assert!(t.hearts.taken == 0 && t.hearts.left == 1, "{:?}", t.hearts);
    assert_eq!(info.health, 2.0);
}

fn item_targets() -> RewardConfig {
    RewardConfig {
        item_targets: true,
        ..Default::default()
    }
}

#[test]
fn item_targets_pull_toward_any_dropped_item() {
    let Some(nes) = open_core() else { return };
    // A rupee east of Link: no target without the option (the exit is north), the nearest target with it.
    let drops = [(item::RUPEE, 152, 157)];
    let (mut env, _) = placed_env_with(nes, &drops, 157, RewardConfig::default(), 0x21);
    let off = movement_reward(&mut env, "E");
    let (mut env, _) = placed_env_with(release(env), &drops, 157, item_targets(), 0x21);
    let on = movement_reward(&mut env, "E");
    assert!(off < 0.0 && on > 0.0, "off {off}, on {on}");
}

#[test]
fn item_targets_skip_healing_drops_at_full_health() {
    let Some(nes) = open_core() else { return };
    let drops = [(item::HEART, 152, 157)];
    let (mut env, _) = placed_env_with(nes, &drops, 157, item_targets(), 0x21);
    let hurt = movement_reward(&mut env, "E");
    let (mut env, _) = placed_env_with(release(env), &drops, 157, item_targets(), 0x22);
    let full = movement_reward(&mut env, "E");
    assert!(hurt > 0.0 && full < 0.0, "hurt {hurt}, full {full}");
}

// --- tests/test_rewards.py (emulator parts) ---

#[test]
fn gameplay_rewards_a_real_shield_collision_in_weapon_settling() {
    let Some(nes) = open_core() else { return };
    let mut s = session(nes);
    place_shield_projectile(&mut s, &[11]);
    s.nes_mut().poke(0x70 + 11, 180);
    s.apply_ram(&[]);
    let start = s.frame();
    let change = s.act(Action::parse("BEAMS E").unwrap()).unwrap();
    assert!(
        change.blocks() == 1 && change.health_lost == 0.0,
        "{} {}",
        change.blocks(),
        change.health_lost
    );
    let event = change.projectile_blocks[0];
    assert!(
        start + (change.control_frames as u64) < event.frame
            && event.frame <= start + change.frames as u64,
        "start {start} control {} frames {} event {}",
        change.control_frames,
        change.frames,
        event.frame
    );
    let mut rewards = StepRewards::default();
    critique(
        &mut GameplayCritic::new(false, 1.0, 0.0),
        &change,
        &mut rewards,
    );
    let block = rewards
        .get("reward-block-projectile")
        .expect("block reward");
    assert!(
        block.value == REWARD_MEDIUM && block.count == 1,
        "{block:?}"
    );
}

#[test]
fn real_simultaneous_projectiles_each_earn_one_block_reward_and_not_again() {
    let Some(nes) = open_core() else { return };
    let mut s = session(nes);
    place_shield_projectile(&mut s, &[10, 11]);
    let mut critic = GameplayCritic::new(false, 1.0, 0.0);
    let change = s.act(Action::parse("WAIT 16").unwrap()).unwrap();
    let mut rewards = StepRewards::default();
    critique(&mut critic, &change, &mut rewards);
    assert_eq!(change.blocks(), 2);
    let block = rewards
        .get("reward-block-projectile")
        .expect("block reward");
    assert!(
        block.value == 2.0 * REWARD_MEDIUM && block.count == 2,
        "{block:?}"
    );
    let follow = s.act(Action::parse("WAIT 4").unwrap()).unwrap();
    let mut rewards = StepRewards::default();
    critique(&mut critic, &follow, &mut rewards);
    assert_eq!(follow.blocks(), 0);
    assert!(
        rewards.get("reward-block-projectile").is_none(),
        "{rewards:?}"
    );
}

fn walk(env: &mut Env, steps: usize) -> Vec<StepInfo> {
    let mut infos = Vec::new();
    for _ in 0..steps {
        let r = greedy_step(env).unwrap();
        let done = r.terminated || r.truncated;
        infos.push(r.info);
        if done {
            break;
        }
    }
    infos
}

/// 1_73e has 2 keys and the N door is locked: walking to it opens the door, rewarded once, then Link walks through
/// into 0x63.
#[test]
fn locked_door_reward_fires_once() {
    let Some(nes) = open_core() else { return };
    let mut env = make_env(nes, "stalfos-keese-gel", Some("1_73e_0"), |_| {});
    env.reset(Some(0), None).unwrap();
    let infos = walk(&mut env, 40);
    let hits: Vec<usize> = (0..infos.len())
        .filter(|&i| has_term(&infos[i], "reward-opened-locked-door"))
        .collect();
    assert_eq!(hits.len(), 1, "{hits:?}");
    assert_eq!(
        term(&infos[hits[0]], "reward-opened-locked-door"),
        Some(REWARD_MEDIUM)
    );
    assert!(infos[hits[0]..].iter().any(|i| i.room == (1, 0x63, false)));
}

/// Same start, same RNG, same actions: PBRS on adds pbrs-movement outcomes, PBRS off has none, and every other
/// outcome is identical.
#[test]
fn pbrs_switch_changes_only_movement_shaping() {
    let Some(nes) = open_core() else { return };
    let script = [
        "MOVE N 16",
        "MOVE N 16",
        "MOVE W 16",
        "MOVE S 8",
        "MOVE W 16",
        "WAIT 8",
    ];
    let run = |nes: ZCore, pbrs: bool| {
        let mut env = make_env(nes, "stalfos-keese-gel", Some("1_73s_0"), |c| {
            c.reward = RewardConfig {
                pbrs,
                ..Default::default()
            };
        });
        env.reset(Some(5), None).unwrap();
        let steps: Vec<Vec<(&'static str, f64)>> = script
            .iter()
            .map(|a| act(&mut env, a).info.rewards)
            .collect();
        (release(env), steps)
    };
    let (nes, on) = run(nes, true);
    let (_, off) = run(nes, false);
    let is_pbrs = |k: &str| k == "reward-pbrs-movement" || k == "penalty-pbrs-movement";
    assert!(
        on.iter().any(|step| step.iter().any(|(k, _)| is_pbrs(k))),
        "{on:?}"
    );
    assert!(
        !off.iter().any(|step| step.iter().any(|(k, _)| is_pbrs(k))),
        "{off:?}"
    );
    let strip: Vec<Vec<(&str, f64)>> = on
        .iter()
        .map(|step| step.iter().filter(|(k, _)| !is_pbrs(k)).copied().collect())
        .collect();
    assert_eq!(strip, off);
}

#[test]
fn wait_gets_no_movement_shaping() {
    let Some(nes) = open_core() else { return };
    let mut env = make_env(nes, "stalfos-keese-gel", Some("1_73s_0"), |_| {});
    env.reset(Some(0), None).unwrap();
    act(&mut env, "MOVE N 8"); // seeds the PBRS baseline
    for _ in 0..3 {
        let info = act(&mut env, "WAIT 16").info;
        assert!(
            !info.rewards.iter().any(|(k, _)| k.contains("pbrs")),
            "{:?}",
            info.rewards
        );
        assert!(!has_term(&info, "penalty-attack-miss"));
    }
}

/// From the 0x73 S entry (no keys) the objective is W or E; walking toward an exit earns positive PBRS.
#[test]
fn pbrs_rewards_walking_toward_objective() {
    let Some(nes) = open_core() else { return };
    let mut env = make_env(nes, "stalfos-keese-gel", Some("1_73s_0"), |_| {});
    env.reset(Some(0), None).unwrap();
    let total: f64 = walk(&mut env, 8).iter().map(pbrs_total).sum();
    assert!(total > 0.0, "{total}");
}

#[test]
fn sword_miss_is_penalized() {
    let Some(nes) = open_core() else { return };
    // Half health: the sword, not beams.
    let mut env = make_env(nes, "goriya", Some("1_44w_0"), |c| {
        c.ram_randomization = hp(0.5, 0.5)
    });
    env.reset(Some(0), None).unwrap();
    for _ in 0..10 {
        // Off the door tile and the Goriyas out of their spawn clouds.
        act(&mut env, "MOVE E 16");
        if state(&env).link.position.x >= 32 && state(&env).has_active_enemies() {
            break;
        }
    }
    assert!(env.session().mask().kind[ActionKind::Sword.index()]);
    let info = act(&mut env, "SWORD S").info; // the Goriyas are E of Link
    assert!(
        term(&info, "penalty-attack-miss") == Some(-0.01) || has_term(&info, "reward-hit"),
        "{:?}",
        info.rewards
    );
}

/// Taking damage costs 0.25 per half heart and halves the step's other positive rewards (not pickups).
///
/// Python monkeypatches the env critic's `critique_triforce` to add `reward-hit` 0.5 every step.  The Rust env's
/// critic is private, so this runs the same chase on the env's session after its reset (same config, hp and start
/// state) through a `GameplayCritic` whose rewards are pre-seeded with a 0.5 stand-in reward (`reward-pbrs-movement`,
/// which this PBRS-off critic never adds itself) and an exempt pickup (`reward-gained-triforce` 1.0, impossible in
/// 0x44).
#[test]
fn damage_penalty_and_reward_halving() {
    let Some(nes) = open_core() else { return };
    let mut env = make_env(nes, "goriya", Some("1_44w_0"), |c| {
        c.ram_randomization = hp(0.5, 0.5)
    });
    env.reset(Some(1), None).unwrap();
    let (mut s, _) = env.into_parts();
    let mut critic = GameplayCritic::new(false, 1.0, 0.0);
    let first = s.act(Action::parse("MOVE E 16").unwrap()).unwrap();
    critique(&mut critic, &first, &mut StepRewards::default());
    for _ in 0..120 {
        let before = s.state().health();
        let action = chase(&s);
        let change = s.act(action).unwrap();
        let mut rewards = StepRewards::default();
        rewards.add(Outcome::reward("reward-pbrs-movement", 0.5));
        rewards.add(Outcome::reward("reward-gained-triforce", 1.0));
        critique(&mut critic, &change, &mut rewards);
        let lost = before - s.state().health();
        let seeded = rewards.get("reward-pbrs-movement").unwrap().value;
        assert_eq!(
            rewards.get("reward-gained-triforce").unwrap().value,
            1.0,
            "pickups are exempt"
        );
        let terminated = s.state().game_over();
        if lost > 0.0 && !terminated {
            let penalty = rewards
                .get("penalty-lost-health")
                .expect("damage penalty")
                .value;
            assert!(
                approx(penalty, -(0.25 * lost * 2.0).min(1.0)),
                "{penalty} for {lost}"
            );
            assert!(approx(seeded, 0.25), "{seeded}");
            return;
        }
        assert!(approx(seeded, 0.5), "{seeded}");
        if terminated {
            break;
        }
    }
    eprintln!("Link never took non-fatal damage: skipping");
}

/// The Zora surfaces and submerges every few steps; that must not reset the room-stalling counter (only kills and
/// pickups do), so dithering at the 0x38 entrance is penalized after the grace period.
#[test]
fn stalling_penalty_fires_in_zora_room() {
    let Some(nes) = open_core() else { return };
    let mut env = make_env(nes, "ow-zora", Some("0_38s_0"), |_| {});
    env.reset(Some(0), None).unwrap();
    let script = ["WAIT 8", "WAIT 8", "MOVE E 4", "MOVE W 4"];
    for i in 0..300 {
        let r = act(&mut env, script[i % script.len()]);
        assert!(
            !(r.terminated || r.truncated),
            "step {i}: {:?}",
            r.info.ending()
        );
        if has_term(&r.info, "penalty-room-stalling") {
            return;
        }
    }
    panic!("no room-stalling penalty in 300 steps of dithering");
}

// --- tests/test_specialist.py (session-backed) ---

/// RAM decides: every seed state a config resolves must load into one of its rooms.
#[test]
fn start_states_load_into_their_rooms() {
    let Some(nes) = open_core() else { return };
    let mut s = session(nes);
    let mut loaded = 0;
    for (name, config) in &data().specialists {
        let paths = config.resolve_start_states(&data().states_dir);
        if config.trainable {
            assert!(!paths.is_empty(), "{name} has no start states");
        }
        for path in paths {
            let state = load(&mut s, path.to_str().unwrap(), LoadOptions::default());
            assert!(
                config.rooms.contains(&state.full_location()),
                "{name}: {} -> {:?}",
                path.file_name().unwrap().to_string_lossy(),
                state.full_location()
            );
            loaded += 1;
        }
    }
    assert!(loaded > 0);
}

#[test]
fn hp_randomization_bounds() {
    let Some(nes) = open_core() else { return };
    let mut s = session(nes);
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let cases = [
        ((HpBound::Fraction(0.5), HpBound::Max), 3),
        ((HpBound::Fraction(1.0), HpBound::Max), 16),
        ((HpBound::Fraction(0.01), HpBound::Fraction(0.01)), 3),
    ];
    for (bounds, max_health) in cases {
        let spec = RamRandomization {
            hp: Some(bounds),
            bombs: Some((1, 3)),
            keys: Some((2, 2)),
            ..Default::default()
        };
        let lo = bounds.0.value();
        for _ in 0..10 {
            let values = ram_randomization_values(&spec, max_health, &mut rng, None);
            let opts = LoadOptions {
                overrides: vec![(Var::HeartsAndContainers, ((max_health - 1) << 4) as u8)],
                ..Default::default()
            };
            load(&mut s, "start", opts);
            let state = s.apply_ram(&values).clone();
            let health = state.health();
            assert_eq!(state.max_health(), max_health);
            let floor = ((lo * 2.0 * max_health as f64).round_ties_even() / 2.0).max(0.5);
            assert!(
                floor <= health && health <= max_health as f64,
                "{bounds:?}: {health}"
            );
            assert!((1..=3).contains(&state.bombs()) && state.keys() == 2);
            if health == max_health as f64 {
                assert!(
                    state.is_health_full(),
                    "beams need the NES full-health encoding"
                );
            }
        }
    }
}

/// full_hp: half the resets at full health (the sword fires beams), the rest strictly below full.
#[test]
fn full_hp_share_splits_resets_between_beams_and_no_beams() {
    let Some(nes) = open_core() else { return };
    let mut s = session(nes);
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let spec = RamRandomization {
        hp: Some((HpBound::Fraction(0.5), HpBound::Max)),
        full_hp: Some(0.5),
        ..Default::default()
    };
    let draws: Vec<Vec<(Var, u8)>> = (0..400)
        .map(|_| ram_randomization_values(&spec, 3, &mut rng, None))
        .collect();
    let is_full = |v: &[(Var, u8)]| v.contains(&(Var::PartialHearts, 0xFF));
    let full: Vec<&Vec<(Var, u8)>> = draws.iter().filter(|v| is_full(v)).collect();
    let share = full.len() as f64 / draws.len() as f64;
    assert!(0.4 < share && share < 0.6, "{share}");
    let partial = draws.iter().find(|v| !is_full(v)).expect("a partial draw");
    for values in [full[0], partial] {
        let opts = LoadOptions {
            overrides: vec![(Var::HeartsAndContainers, 0x20), (Var::Sword, 1)],
            ..Default::default()
        };
        load(&mut s, "start", opts);
        let state = s.apply_ram(values).clone();
        assert_eq!(state.is_health_full(), is_full(values));
        assert!((1.5..=3.0).contains(&state.health()), "{}", state.health());
        assert_eq!(state.are_beams_available(), state.is_health_full());
    }
}

#[test]
fn capture_saves_first_entry_once_before_monsters_spawn() {
    let Some(nes) = open_core() else { return };
    let dir = tempfile::tempdir().unwrap();
    let store = TfzStore::new();
    let mut capture = SavestateCapture::new(dir.path().to_path_buf());
    let mut s = session(nes);
    load(&mut s, "start", LoadOptions::default());
    s.set_capture_pre_entry(true);
    let changes = act_until(&mut s, "MOVE E 32", |c| c.changed_location(), 30);
    let change = changes.last().unwrap();
    assert!(s.last_pre_entry().is_some(), "the room entry was captured");
    let path = capture
        .on_step(change, s.last_pre_entry(), || s.snapshot(), &store)
        .unwrap()
        .expect("a captured state");
    s.set_capture_pre_entry(false);
    assert_eq!(path.file_name().unwrap(), "0_78w.state");
    assert!(
        capture
            .on_step(change, s.last_pre_entry(), || s.snapshot(), &store)
            .unwrap()
            .is_none()
    ); // exists
    let machine = store.load(path.to_str().unwrap()).unwrap();
    s.nes_mut().restore(&machine);
    assert!(
        is_pre_entry(|a| s.nes().peek(a)),
        "saved before the room's monsters exist"
    );
    let state = load(&mut s, path.to_str().unwrap(), LoadOptions::default());
    assert_eq!(state.full_location(), MapLocation::new(0, 0x78, false));
    let change = s.act(Action::parse("MOVE E 8").unwrap()).unwrap();
    assert!(
        capture
            .on_step(&change, s.last_pre_entry(), || s.snapshot(), &store)
            .unwrap()
            .is_none()
    ); // no room change

    // The env path (`capture_dir`): the step that enters a new room reports the written pre-entry state.
    let env_dir = tempfile::tempdir().unwrap();
    let opts = EnvOptions {
        start_state: Some("start".into()),
        capture_dir: Some(env_dir.path().to_path_buf()),
        lenient_actions: true,
        ..Default::default()
    };
    let mut env = make_env_with(s.into_nes(), "ow-start", opts, |_| {});
    env.reset(Some(0), None).unwrap();
    let mut captured = None;
    for _ in 0..30 {
        let r = act(&mut env, "MOVE E 32");
        if r.info.room != (0, 0x77, false) {
            captured = r.info.captured_state;
            break;
        }
        assert!(r.info.captured_state.is_none());
    }
    let captured = captured.expect("entering 0x78 captured a state");
    assert_eq!(captured, env_dir.path().join("0_78w.state"));
    let machine = TfzStore::new().load(captured.to_str().unwrap()).unwrap();
    let mut s = env.into_parts().0;
    s.nes_mut().restore(&machine);
    assert!(is_pre_entry(|a| s.nes().peek(a)));
}
