//! `SpecialistEnv` end to end on real savestates (zcore): episodes, endings, outcomes, terminal rewards,
//! the info contract, objectives and routes, critic behaviour, RAM randomization, start states and capture.
//!
//! Ported from the emulator parts of the removed Python tests tests/test_env.py, tests/test_nearby_items.py,
//! tests/test_rewards.py and tests/test_specialist.py.  States are the regenerated `TFZSTATE` start states in
//! `triforce/states` (`TfzStore`).  Python's `random` streams are not reproduced (the env draws from ChaCha8), so
//! tests assert properties, not particular draws.  Without the ROM every test fails (skips with
//! `TRIFORCE_ALLOW_NO_ROM=1`).

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
use env::observation::{Observation, ObservationBuilder, feature_names};
use env::specialist_env::{EnvOptions, ShapingDial, SpecialistEnv, StepInfo, StepResult};
use env::start_states::{StartWeights, ram_randomization_values, state_label};
use env::{EnvError, StateStore};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::BTreeSet;
use std::sync::{Arc, LazyLock};
use zelda::actions::{MASK_SIZE, NUM_DIRECTIONS, NUM_DURATIONS, NUM_KINDS};
use zelda::cheats::CheatItem;
use zelda::enums::{SwordKind, enemy, item};
use zelda::tables;
use zelda::objectives::{FLAG_ITEM_TAKEN, Objective, ObjectiveKind};
use zelda::room::Target;
use zelda::room_state::room_flags_addr;
use zelda::{
    Action, ActionKind, ActionMask, Change, Direction, GameView, LoadOptions, MapLocation,
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
        retreats: BTreeSet::new(),
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

/// F109: an underworld cellar is a sub-room in the info's room, like the visit records (`1:0x7f:cave`).
#[test]
fn cellar_info_room_is_in_cave() {
    let Some(nes) = open_core() else { return };
    let mut env = make_env(nes, "equipment-room", Some("1_7fcave_0"), |_| {});
    let (_, info) = env.reset(Some(0), None).unwrap();
    assert_eq!(info.room, (1, 0x7f, true));
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
        approx(terminal, success_reward_value(1.5, 3.0, 0.5)) && approx(terminal, 15.0),
        "{terminal}"
    );
    assert!(
        approx(reward, term_sum(&info).min(20.0)),
        "{reward} vs {:?}",
        info.rewards
    );
}

/// `reward.exit_health_weight` reaches the terminal: at 1.0 a half-health exit pays 20 × 1.5/3 = 10.
#[test]
fn exit_health_weight_scales_success_terminal() {
    let Some(nes) = open_core() else { return };
    let mut env = sworded_ow_start(nes, 0, |c| {
        c.ram_randomization = hp(0.5, 0.5);
        c.reward.exit_health_weight = 1.0;
    });
    let (_, info) = last_info(run_until_done(&mut env, greedy_step, 400).unwrap());
    assert_eq!(info.outcome(), Some("success"));
    let terminal = term(&info, REWARD_TERMINAL_SUCCESS).unwrap();
    assert!(approx(terminal, 10.0), "{terminal}");
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
        // F016: 1:0x44 is no `requires_bombs` room, so BOMB is masked even with `items.bombs` on.
        assert!(!env.session().mask().kind[ActionKind::Bomb.index()]);
        assert!(!ActionMask::from_flat(&obs.action_mask).kind[ActionKind::Bomb.index()]);
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
    let Some(nes) = open_core() else { return };
    let mut env = make_env(nes, "d2-rope", Some("2_6es_0"), |_| {});
    env.reset(Some(0), None).unwrap();
    assert_eq!(objective(&env).kind, ObjectiveKind::Move);
    assert_eq!(next_rooms(&env), rooms(&[(2, 0x5e)]));
}

/// 0x43 is crossed twice (the bow detour, Lee, Nov 9): from W the route goes N to the Stalfos key room 0x33, back
/// from N (with the bow) through the locked E door to 0x44; with or without a key the objective offers only that exit.
/// Without one, the locked route exit grants a cheated key (`zelda::cheats`).
#[test]
fn route_room_only_offers_its_route_exit() {
    let Some(mut nes) = open_core() else { return };
    for (start_state, wanted) in [("1_43w_0", 0x33), ("1_43n_0", 0x44)] {
        for keys in [0, 1] {
            let mut env = make_env(nes, "stalfos-keese-gel", Some(start_state), |c| {
                c.ram_randomization = RamRandomization {
                    keys: Some((keys, keys)),
                    ..Default::default()
                };
            });
            env.reset(Some(0), None).unwrap();
            let cheated_key = wanted == 0x44 && keys == 0;
            assert_eq!(state(&env).keys() as i64, keys + cheated_key as i64, "{start_state} keys {keys}");
            assert_eq!(env.cheats.granted(CheatItem::Keys), cheated_key as i32);
            assert_eq!(state(&env).bow(), wanted == 0x44, "{start_state}");
            assert_eq!(next_rooms(&env), rooms(&[(1, wanted)]), "{start_state} keys {keys}");
            nes = release(env);
        }
    }
}

/// Dungeon 1's entrance 0x73 follows the route's steps (Lee, Nov 8): entered from S (with Dungeon 2's keys: the
/// shortest way is N) the key room 0x72 W comes first, back from it (from W) the key room 0x74 E, back from that (from
/// E) N.  Each step has its own start states, so one-room training resets play every one of them.
#[test]
fn the_d1_entrance_follows_the_route_steps() {
    let Some(mut nes) = open_core() else { return };
    for (start_state, step, wanted) in [("1_73s_0", 46, 0x72), ("1_73w_0", 48, 0x74), ("1_73e_0", 50, 0x63)] {
        let mut env = make_env(nes, "stalfos-keese-gel", Some(start_state), |_| {});
        let (_, info) = env.reset(Some(0), None).unwrap();
        assert!(state(&env).keys() > 0, "{start_state}: Link carries keys from Dungeon 2");
        assert_eq!(env.route(), ("d2first", Some(step)), "{start_state}");
        assert_eq!(info.route, "d2first");
        assert_eq!(next_rooms(&env), rooms(&[(1, wanted)]), "{start_state}");
        nes = release(env);
    }
}

/// ow-octorok lists enter-room:2:0x7d (the D2 stairs in 0x3c) as success.  After Dungeon 2 (0_3cc_0: the triforce
/// put Link next to those stairs) the objective wants S, so walking back down is exit-wrong.
#[test]
fn enter_room_success_needs_the_objective() {
    let Some(nes) = open_core() else { return };
    let mut env = make_env(nes, "ow-octorok", Some("0_3cc_0"), |_| {});
    env.reset(Some(0), None).unwrap();
    assert_eq!(next_rooms(&env), rooms(&[(0, 0x4c)]));
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
    assert_eq!(info.room, (2, 0x7d, false));
    assert_eq!(
        (info.ending(), info.outcome()),
        (Some("exit-wrong"), Some("soft-failure"))
    );
}

/// Rooms the route crosses twice train both crossings: the entry side picks the step (0x4c from E goes N to the D2
/// stairs room, from N E; 0x3c from S takes the stairs, out of them after the triforce S).
#[test]
fn a_room_the_route_crosses_twice_trains_both_steps() {
    let Some(mut nes) = open_core() else { return };
    for (start_state, step, wanted) in [
        ("0_4ce_0", 12, (0, 0x3c)),
        ("0_4cn_0", 25, (0, 0x4d)),
        ("0_3cs_0", 13, (2, 0x7d)),
        ("0_3cc_0", 24, (0, 0x4c)),
    ] {
        let mut env = make_env(nes, "ow-octorok", Some(start_state), |_| {});
        env.reset(Some(0), None).unwrap();
        assert_eq!(env.route(), ("d2first", Some(step)), "{start_state}");
        assert_eq!(next_rooms(&env), rooms(&[wanted]), "{start_state}");
        nes = release(env);
    }
}

/// 0x0a before the white sword: leaving S (back to 0x1a, to re-enter for a better Lynel spawn; Lee, Nov 8) ends the
/// one-room episode as `retreat`.  ow-lynel ships it as a soft failure (like a wrong exit, `endings: {retreat:
/// soft-failure}`); without the override it is neutral: no terminal reward or penalty, no wrong-exit penalty.
#[test]
fn leaving_the_lynel_room_south_is_a_retreat() {
    let Some(mut nes) = open_core() else { return };
    for neutral in [false, true] {
        let mut env = make_env(nes, "ow-lynel", Some("0_0as_0"), |c| {
            if neutral {
                c.endings.shift_remove("retreat");
            }
        });
        env.reset(Some(0), None).unwrap();
        assert_eq!(objective(&env).retreats, rooms(&[(0, 0x1a)]));
        let mut info = None;
        for _ in 0..20 {
            let r = act(&mut env, "MOVE S 16");
            let done = r.terminated || r.truncated;
            let terminated = r.terminated;
            info = Some(r.info);
            if done {
                assert!(terminated);
                break;
            }
        }
        let info = info.unwrap();
        assert_eq!(info.room, (0, 0x1a, false));
        let outcome = if neutral { "neutral" } else { "soft-failure" };
        assert_eq!((info.ending(), info.outcome()), (Some("retreat"), Some(outcome)));
        assert_eq!(has_term(&info, PENALTY_TERMINAL_FAILURE), !neutral);
        assert_eq!(has_term(&info, "penalty-wrong-location"), !neutral);
        assert!(!has_term(&info, "reward-terminal-success"));
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

/// F151: a curriculum start (a trajectory's `NNNN.state`) gets the route step of the seed state its track was
/// explored from, as an episode from that seed state does.
#[test]
fn curriculum_start_takes_the_seed_states_route_step() {
    let Some(nes) = open_core() else { return };
    let src = common::repo().join("trajectories/goriya/1_23w_0");
    if !src.is_dir() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let track = dir.path().join("goriya/1_23w_0");
    std::fs::create_dir_all(&track).unwrap();
    for entry in std::fs::read_dir(&src).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), track.join(entry.file_name())).unwrap();
    }
    let spec = CurriculumSpec::new(track.parent().unwrap().to_path_buf(), dir.path().join("p.json"), 1.0, 0.8);
    let opts = EnvOptions { curriculum: Some(Curriculum::new(spec).unwrap().shared()), ..Default::default() };
    let mut env = make_env_with(nes, "goriya", opts, |_| {});
    let (_, info) = env.reset(Some(0), None).unwrap();
    let start = info.start_state.clone().unwrap();
    assert!(start.rsplit('/').next().unwrap().starts_with(|c: char| c.is_ascii_digit()), "{start}");
    let curriculum = (env.route().0.to_owned(), env.route().1);
    let mut seed = make_env(release(env), "goriya", Some("1_23w_0"), |_| {});
    seed.reset(Some(0), None).unwrap();
    assert!(seed.route().1.is_some());
    assert_eq!(curriculum, (seed.route().0.to_owned(), seed.route().1));
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
    let (env, snapshot, _) = placed_env_configured(nes, drops, y, hearts_and_containers, |c| {
        c.reward = reward
    });
    (env, snapshot)
}

/// [`placed_env_with`] with any config change; also returns the restored episode's first observation.
fn placed_env_configured(
    nes: ZCore,
    drops: &[(i16, u8, u8)],
    y: u8,
    hearts_and_containers: u8,
    update: impl FnOnce(&mut SpecialistConfig),
) -> (Env, SessionSnapshot<MachineState>, Observation) {
    placed_env_opts(nes, drops, y, hearts_and_containers, EnvOptions::default(), update)
}

/// [`placed_env_configured`] with env options (start state 1_73e_0 is filled in).
fn placed_env_opts(
    nes: ZCore,
    drops: &[(i16, u8, u8)],
    y: u8,
    hearts_and_containers: u8,
    opts: EnvOptions,
    update: impl FnOnce(&mut SpecialistConfig),
) -> (Env, SessionSnapshot<MachineState>, Observation) {
    let opts = EnvOptions {
        start_state: Some("1_73e_0".to_owned()),
        ..opts
    };
    let mut env = make_env_with(nes, "stalfos-keese-gel", opts, |c| {
        c.one_room = true;
        update(c);
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
    let (obs, _) = env.restore(&snapshot, "1_73e_0").unwrap();
    (env, snapshot, obs)
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

/// `obs.target` points the observation at the nearest wavefront target: with `reward.item_targets` a rupee east of
/// Link (the exit is north) → target_E, target_is_item and a positive dx; off → all zeros.
#[test]
fn target_observation_points_at_the_nearest_item() {
    let Some(nes) = open_core() else { return };
    let drops = [(item::RUPEE, 152, 157)];
    let names = &feature_names().target;
    let at = |obs: &Observation, name: &str| obs.target[names.iter().position(|n| n == name).unwrap()];
    let (env, _, obs) = placed_env_configured(nes, &drops, 157, 0x21, |c| {
        c.reward = item_targets();
        c.obs.target = true;
    });
    assert_eq!(at(&obs, "target_E"), 1.0, "{:?}", obs.target);
    assert_eq!(obs.target[..4].iter().sum::<f32>(), 1.0);
    assert_eq!(at(&obs, "target_is_item"), 1.0);
    // Same tile row: dy is only Link's offset inside his tile.
    let (dx, dy) = (at(&obs, "target_dx"), at(&obs, "target_dy"));
    assert!(dx > 0.0 && dy.abs() < 8.0 / 176.0, "{:?}", obs.target);
    let distance = at(&obs, "target_distance");
    assert!(distance > 0.0 && distance < 0.25, "{distance}");

    // Explicit: the canonical stalfos-keese-gel yaml turns both options on.
    let (env, _, obs) = placed_env_configured(release(env), &drops, 157, 0x21, |c| {
        c.reward = item_targets();
        c.obs.target = false;
    });
    assert!(obs.target.iter().all(|&x| x == 0.0), "{:?}", obs.target);

    // Without item targets the exit north is the target: the walk heads N, not toward the rupee.
    let (_, _, obs) = placed_env_configured(release(env), &drops, 157, 0x21, |c| {
        c.reward = RewardConfig::default();
        c.obs.target = true;
    });
    assert_eq!(at(&obs, "target_is_item"), 0.0);
    assert_eq!(at(&obs, "target_E"), 0.0, "{:?}", obs.target);
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
    panic!("Link never took non-fatal damage in 120 chase steps: the test checked nothing");
}

/// The Zora surfaces and submerges every few steps; that must not reset the room-stalling counter (only kills and
/// pickups do), so dithering at the 0x38 entrance is penalized after the grace period.
#[test]
fn stalling_penalty_fires_in_zora_room() {
    let Some(nes) = open_core() else { return };
    let mut env = make_env(nes, "ow-zora", Some("0_38n_0"), |_| {});
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

/// game.yaml `requires_bombs` (Dodongo's room): with no bombs Link gets a cheated one; placing it counts one used and
/// one cheated, a natural last bomb only one used, and either way the floor re-arms to 1. Outside such rooms the
/// count is untouched. (Leaving the room with an unused cheated bomb: `zelda::cheats` unit tests.)
#[test]
fn requires_bombs_room_cheats_one_bomb_and_counts_it_when_placed() {
    let Some(mut nes) = open_core() else { return };
    let bomb = ActionKind::Bomb.index();
    for (start, cheated) in [(0, 1), (1, 0)] {
        let mut env = make_env(nes, "boss-dodongo", Some("2_0es_0"), |c| {
            c.curriculum.enabled = false;
            c.ram_randomization = RamRandomization {
                bombs: Some((start, start)),
                ..Default::default()
            };
        });
        env.reset(Some(1), None).unwrap();
        assert_eq!(state(&env).bombs(), 1, "start {start}");
        assert_eq!((env.cheats.used.bombs, env.cheats.cheated.bombs), (0, 0), "granting doesn't count");
        // Off the door tile, where items can't be used.
        for _ in 0..4 {
            if env.session().mask().kind[bomb] {
                break;
            }
            act(&mut env, "MOVE N 16");
        }
        let place = ["BOMB N", "BOMB E", "BOMB W", "BOMB S"]
            .into_iter()
            .find(|t| env.session().mask().is_allowed(&Action::parse(t).unwrap()))
            .expect("a bomb direction");
        let r = act(&mut env, place);
        assert_eq!((env.cheats.used.bombs, env.cheats.cheated.bombs), (1, cheated), "start {start}");
        assert_eq!(r.info.cheated.bombs, cheated);
        assert_eq!(state(&env).bombs(), 1, "the floor re-arms");
        assert_eq!(r.obs.link, {
            let mut b = ObservationBuilder::new(true, true);
            b.reset(state(&env), objective(&env), None, env.session().mask()).link
        }, "the observation sees the re-armed count");
        nes = release(env);
    }

    let mut env = make_env(nes, "d2-rope", Some("2_2es_0"), |c| {
        c.ram_randomization = RamRandomization {
            bombs: Some((0, 0)),
            ..Default::default()
        };
    });
    env.reset(Some(0), None).unwrap();
    act(&mut env, "WAIT 8");
    assert_eq!(state(&env).bombs(), 0);
    assert_eq!((env.cheats.used.bombs, env.cheats.cheated.bombs), (0, 0));
}

/// Dodongo walking -> stunned (ObjState 2, a bomb's smoke) and walking -> bloated (1, a swallowed bomb) count once
/// each in the terminal info's `boss` and in the room visit. The transitions are poked into RAM behind the session's
/// cached state, so the action's `Change` sees the boss leave state 0.
#[test]
fn dodongo_stuns_and_swallows_count_per_episode_and_visit() {
    let Some(nes) = open_core() else { return };
    let opts = EnvOptions {
        start_state: Some("2_0es_0".into()),
        track_visits: true,
        ..Default::default()
    };
    let mut env = make_env_with(nes, "boss-dodongo", opts, |c| {
        c.curriculum.enabled = false;
        c.max_steps = 40;
    });
    env.reset(Some(1), None).unwrap();
    let walking = |env: &Env| {
        state(env)
            .enemies
            .iter()
            .find(|e| matches!(e.id, enemy::DODONGO | enemy::DODONGO2) && e.status == 0)
            .map(|e| e.index as usize)
    };
    let mut slot = walking(&env);
    while slot.is_none() {
        act(&mut env, "WAIT 8");
        slot = walking(&env);
    }
    let address = (tables::OBJ_STATUS.0 + slot.unwrap()) as u16;
    let mut terminal = None;
    for status in [2, 1] {
        while walking(&env).is_none() {
            act(&mut env, "WAIT 8");
        }
        env.session_mut().nes_mut().poke(address, status);
        let r = act(&mut env, "WAIT 4");
        assert_eq!(state(&env).enemy_by_index(slot.unwrap() as u8).map(|e| e.status), Some(status));
        terminal = r.info.terminal;
    }
    while terminal.is_none() {
        terminal = act(&mut env, "WAIT 8").info.terminal;
    }
    let t = terminal.unwrap();
    assert_eq!((t.boss.stuns, t.boss.eats), (1, 1));
    let visits = env.take_visits();
    let total = visits.iter().fold((0, 0), |(s, e), v| (s + v.boss_stuns, e + v.boss_eats));
    assert_eq!(total, (1, 1));
}

/// Cheated keys (`zelda::cheats`): Dungeon 1's 0x33 entered from S leaves N through a locked door and holds a key
/// (the room's treasure). With no keys Link gets none while that key is left; once it is taken (the room's
/// item-taken flag) a cheated key appears without counting as a key gained. Opening the N door with it counts one
/// cheated key used and no second key is granted while the door opens; leaving S with it unused takes it back.
#[test]
fn locked_route_exit_cheats_a_key_once_the_rooms_key_is_taken() {
    let Some(mut nes) = open_core() else { return };
    for through_door in [true, false] {
        let opts = EnvOptions {
            start_state: Some("1_33s_0".into()),
            track_visits: true,
            ..Default::default()
        };
        let mut env = make_env_with(nes, "stalfos-keese-gel", opts, |c| {
            c.curriculum.enabled = false;
            c.ram_randomization = RamRandomization {
                keys: Some((0, 0)),
                ..Default::default()
            };
        });
        env.reset(Some(1), None).unwrap();
        assert!(state(&env).is_door_locked(Direction::N), "the route exit is locked");
        assert_eq!(state(&env).keys(), 0, "the room's key treasure comes first");
        act(&mut env, "WAIT 8");
        assert_eq!((state(&env).keys(), env.cheats.granted(CheatItem::Keys)), (0, 0));

        // The treasure key taken (and spent elsewhere: Link has none).
        let ram = state(&env).ram();
        let flags = room_flags_addr(ram, 0x33).expect("level flags");
        let taken = ram[flags] | FLAG_ITEM_TAKEN;
        env.session_mut().nes_mut().poke(flags as u16, taken);
        act(&mut env, "WAIT 8");
        assert_eq!(state(&env).keys(), 1, "a cheated key");
        assert_eq!((env.cheats.granted(CheatItem::Keys), env.cheats.cheated.keys), (1, 0), "granting doesn't count");

        let terminal = if through_door {
            // Past the room's middle blocks: Link stands in the N doorway's column, below the door.
            env.session_mut().apply_ram(&[(Var::LinkX, 0x78), (Var::LinkY, 0x5D)]);
            for _ in 0..10 {
                if state(&env).keys() == 0 {
                    break;
                }
                act(&mut env, "MOVE N 4");
            }
            assert_eq!(env.cheats.cheated.keys, 1, "the door took the cheated key");
            let mut terminal = None;
            while terminal.is_none() {
                assert_eq!((state(&env).keys(), env.cheats.granted(CheatItem::Keys)), (0, 0), "no second key");
                terminal = act(&mut env, "MOVE N 16").info.terminal;
            }
            assert_eq!(state(&env).location(), 0x23, "through the opened door");
            terminal
        } else {
            let mut terminal = None;
            while terminal.is_none() {
                terminal = act(&mut env, "MOVE S 16").info.terminal;
            }
            assert_eq!(state(&env).location(), 0x43);
            assert_eq!(state(&env).keys(), 0, "the unused cheated key is taken back");
            terminal
        };
        let cheated = through_door as u32;
        assert_eq!(terminal.unwrap().cheated.keys, cheated);
        let visits = env.take_visits();
        assert_eq!(visits.iter().map(|v| (v.keys_gained, v.cheated.keys)).collect::<Vec<_>>(), [(0, cheated)]);
        nes = release(env);
    }
}

/// One step's result without its rewards: what must not depend on the look-ahead.
type Unrewarded = (Observation, bool, bool, Option<env::specialist_env::TerminalInfo>);

/// Plays ten episodes of the Goriya room at full health, firing beams whenever they're legal (else mostly
/// attacking); returns every step without its rewards, each episode's room visits and the total reward.
fn goriya_episodes(nes: ZCore, skip_look_ahead: bool) -> (Vec<Unrewarded>, Vec<Vec<zelda::visits::RoomVisit>>, f64) {
    let opts = EnvOptions {
        start_state: Some("1_44w_0".into()),
        seed: 3,
        track_visits: true,
        skip_look_ahead,
        ..Default::default()
    };
    let mut env = make_env_with(nes, "goriya", opts, |c| {
        c.ram_randomization = RamRandomization {
            hp: Some((HpBound::Max, HpBound::Max)),
            full_hp: Some(1.0),
            ..Default::default()
        };
        c.max_steps = 120;
    });
    let mut rng = ChaCha8Rng::seed_from_u64(4);
    let (mut steps, mut visits, mut total) = (Vec::new(), Vec::new(), 0.0);
    for _ in 0..10 {
        let (mut obs, _) = env.reset(None, None).unwrap();
        loop {
            let mut mask = ActionMask::from_flat(&obs.action_mask);
            if mask.kind[ActionKind::Beams as usize] {
                for kind in [ActionKind::Move, ActionKind::Wait, ActionKind::Sword] {
                    mask.disallow_kind(kind);
                }
            } else if rng.random::<f32>() < 0.7 && mask.kind[ActionKind::Sword as usize] {
                mask.disallow_kind(ActionKind::Move);
                mask.disallow_kind(ActionKind::Wait);
            }
            let action = random_legal_action(&mask.to_flat(), &mut rng);
            let r = env.step_indices(action).unwrap();
            total += r.reward;
            obs = r.obs.clone();
            let mut terminal = r.info.terminal.clone();
            if let Some(t) = &mut terminal {
                (t.reward_total, t.episode_rewards) = (0.0, Default::default());
            }
            let done = r.terminated || r.truncated;
            steps.push((r.obs, r.terminated, r.truncated, terminal));
            if done {
                visits.push(env.take_visits());
                break;
            }
        }
    }
    (steps, visits, total)
}

#[test]
fn endings_linger_and_visits_ignore_the_look_ahead() {
    let (Some(a), Some(b)) = (open_core(), open_core()) else { return };
    let (with, with_visits, with_reward) = goriya_episodes(a, false);
    let (without, without_visits, without_reward) = goriya_episodes(b, true);
    assert!(with.len() > 50);
    assert!(with == without, "the same game, endings and linger with the look-ahead on or off");
    assert!(with_visits == without_visits);
    assert_ne!(with_reward, without_reward, "the look-ahead credited beams on their firing steps");
}

/// Outcome fine-tuning: `reward.shaping_scale` (or the pool's dial) scales every non-terminal term;
/// `frame_penalty` adds an unscaled `penalty-frames` per emulated frame.
#[test]
fn shaping_scale_and_frame_penalty() {
    let Some(nes) = open_core() else { return };
    let mut nes = Some(nes);
    let mut run = |scale: f64, frame_penalty: f64| {
        let reward = RewardConfig {
            shaping_scale: scale,
            frame_penalty,
            ..Default::default()
        };
        let (mut env, _) = placed_env_with(nes.take().unwrap(), &[], 157, reward, 0x21);
        act(&mut env, "MOVE E 4");
        let r = act(&mut env, "MOVE E 8");
        nes = Some(release(env));
        r
    };
    let full = run(1.0, 0.0);
    let half = run(0.5, 0.0);
    assert!(pbrs_total(&full.info) != 0.0, "{:?}", full.info.rewards);
    assert!((half.reward - full.reward * 0.5).abs() < 1e-9, "{} vs {}", half.reward, full.reward);
    let zero = run(0.0, 0.001);
    let frames = term(&zero.info, "penalty-frames").expect("penalty-frames");
    assert!((frames + 0.001 * zero.info.frames as f64).abs() < 1e-12, "{frames}");
    assert!((zero.reward - frames).abs() < 1e-12, "{:?}", zero.info.rewards);
    for (name, value) in &zero.info.rewards {
        assert!(*name == "penalty-frames" || *value == 0.0, "{name} = {value}");
    }

    // A dial set after construction applies to the next step.
    let dial = ShapingDial::new(1.0);
    let opts = EnvOptions {
        shaping: Some(dial.clone()),
        ..Default::default()
    };
    let (mut env, _, _) = placed_env_opts(nes.unwrap(), &[], 157, 0x21, opts, |_| {});
    act(&mut env, "MOVE E 4");
    dial.set(0.5);
    let r = act(&mut env, "MOVE E 8");
    assert!((r.reward - full.reward * 0.5).abs() < 1e-9, "{} vs {}", r.reward, full.reward);
}

/// Prioritized level replay: a pool's shared start weights decide every non-curriculum reset's seed state.
#[test]
fn shared_start_weights_pick_the_start_state() {
    let Some(nes) = open_core() else { return };
    let specialist = "stalfos-keese-gel";
    let seeds = data().specialists[specialist].resolve_start_states(&data().states_dir);
    assert!(seeds.len() > 2, "{seeds:?}");
    let target = state_label(&seeds[seeds.len() / 2].to_string_lossy()).to_owned();
    let weights = StartWeights::default();
    weights.set(Some(std::collections::HashMap::from([(target.clone(), 1.0)])));
    let opts = EnvOptions {
        start_weights: Some(weights),
        ..Default::default()
    };
    let mut env = make_env_with(nes, specialist, opts, |c| c.harvested_fraction = 0.0);
    for _ in 0..6 {
        let (_, info) = env.reset(None, None).unwrap();
        let start = info.start_state.expect("start state");
        assert_eq!(state_label(&start), target);
    }
}

/// F132: the target block's direction is a move Link can make: at the W doorway entries of 1:0x43 and 1:0x73 (whose
/// targets lie N/S, which the doorway forbids) and beside 1:0x42's idle push block.
#[test]
fn target_direction_is_a_legal_move() {
    let Some(mut nes) = open_core() else { return };
    for (name, place) in [("1_43w_0", None), ("1_73w_0", None), ("1_42s_0", Some((120u8, 149u8)))] {
        let level = name[..1].parse::<u8>().unwrap();
        let room = u8::from_str_radix(&name[2..4], 16).unwrap();
        let specialist = data().specialist_for(MapLocation::new(level, room, false)).unwrap().to_owned();
        let mut env = make_env(nes, &specialist, Some(name), |c| c.obs.target = true);
        let (mut obs, _) = env.reset(Some(0), None).unwrap();
        if let Some((x, y)) = place {
            env.session_mut().apply_ram(&[(Var::LinkX, x), (Var::LinkY, y), (Var::LinkGridOffset, 0)]);
            obs = env.restore(&env.session().snapshot(), name).unwrap().0;
        }
        let dir = obs.target[..4].iter().position(|&v| v == 1.0);
        if let Some(i) = dir {
            let d = [Direction::N, Direction::S, Direction::W, Direction::E][i];
            let action = Action::new(ActionKind::Move, d, 8).unwrap();
            assert!(env.session().mask().is_allowed(&action), "{name}: target {d:?} is masked");
        }
        nes = release(env);
    }
}

/// F016: bombs are legal only in a family's `requires_bombs` rooms: trap-room in 3:0x5a has none, at 3:0x4c's wall
/// it does; ow-tektite in 0:0x1a has none, at 0x2c's rock wall it does.
#[test]
fn bombs_only_in_requires_bombs_rooms() {
    let Some(mut nes) = open_core() else { return };
    for (specialist, name, legal) in [
        ("trap-room", "3_5ae_0", false),
        ("trap-room", "3_4cw_0", true),
        ("ow-tektite", "0_1an_0", false),
        ("ow-tektite", "0_2ce_0", true),
    ] {
        let mut env = make_env(nes, specialist, Some(name), |_| {});
        env.reset(Some(0), None).unwrap();
        env.session_mut().apply_ram(&[(Var::Bombs, 4)]);
        // Walk into the room, away from the entry doorway (no attack crosses a doorway).
        let walk = match name.as_bytes()[4] {
            b'w' => "MOVE E 8",
            b'e' => "MOVE W 8",
            b'n' => "MOVE S 8",
            _ => "MOVE N 8",
        };
        let mut any = false;
        for _ in 0..40 {
            any |= env.session().mask().direction[ActionKind::Bomb.index()].iter().any(|&b| b);
            if any {
                break;
            }
            let a = Action::parse(walk).unwrap();
            let a = if env.session().mask().is_allowed(&a) { walk } else { "WAIT 8" };
            let r = act(&mut env, a);
            if r.terminated || r.truncated {
                break;
            }
        }
        assert_eq!(any, legal, "{specialist} {name}: {}", env.session().mask());
        nes = release(env);
    }
}
