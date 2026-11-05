//! `SpecialistEnv` tests on the named start states (zcore) that oracle_env.rs did not port yet,
//! ported from the Python tests before the Python game stack is deleted (plan step 16).  Every state comes from
//! `common::load_named_state` (through `TfzStore`).
//!
//! Mapping of the game parts of tests/test_env.py, tests/test_nearby_items.py and
//! tests/test_quiet_enemy_penalty.py:
//!
//! | Python test | Rust |
//! |---|---|
//! | env::test_spaces_and_reset_contract | oracle_env.rs `spaces_and_reset_contract` |
//! | env::test_random_legal_episode_ends_classified | oracle_env.rs (same name) |
//! | env::test_correct_exit_is_success_with_health_scaled_reward | oracle_env.rs (same name) |
//! | env::test_full_health_success_pays_full_terminal | oracle_env.rs (same name) |
//! | env::test_wrong_exit_is_soft_failure | oracle_env.rs (same name) |
//! | env::test_death_is_hard_failure | oracle_env.rs (same name) |
//! | env::test_unlisted_ending_raises | oracle_env.rs (same name) |
//! | env::test_max_steps_truncates_without_terminal_reward | oracle_env.rs (same name) |
//! | env::test_no_room_progress_truncates_at_threshold_and_resets_shared_tracker | oracle_env.rs (same name) |
//! | env::test_no_room_progress_disabled_preserves_position_timeout | oracle_env.rs (same name) |
//! | env::test_no_room_progress_keeps_episode_cap | oracle_env.rs (same name) |
//! | env::test_measured_ending_credits_threshold_action_before_check (6 params) | pure.rs `measured_ending_credits_threshold_action_before_check` |
//! | env::test_masked_action_raises | oracle_env.rs (same name) |
//! | env::test_close_keeps_borrowed_session | here `close_keeps_borrowed_session` |
//! | env::test_start_state_outside_rooms_is_rejected | oracle_env.rs (same name) |
//! | env::test_sword_cave_objective_and_exit | oracle_env.rs (same name) |
//! | env::test_triforce_pickup_is_success_in_the_triforce_room | oracle_env.rs (same name) |
//! | env::test_items_config_masks_bombs_even_when_link_has_them (2 params) | oracle_env.rs (same name) |
//! | env::test_route_shutter_requires_combat_without_a_room_treasure | oracle_env.rs (same name) |
//! | env::test_open_route_ignores_optional_combat (2 params) | oracle_env.rs (same name) |
//! | env::test_route_room_only_offers_its_route_exit (2 params) | oracle_env.rs (same name) |
//! | env::test_first_visit_to_0x73_follows_the_route_order | oracle_env.rs (same name) |
//! | env::test_enter_room_success_needs_the_objective | oracle_env.rs (same name) |
//! | env::test_after_dungeon_1_each_route_leaves_0x38_its_own_way (2 params) | oracle_env.rs (same name) |
//! | env::test_a_room_the_routes_leave_differently_trains_both | oracle_env.rs (same name) |
//! | env::test_fairy_pond_holds_the_exit_until_the_hearts_are_full | oracle_env.rs (same name) |
//! | env::test_fairy_route_heads_to_the_pond_then_east (2 params) | oracle_env.rs (same name) |
//! | env::test_wavefront_has_no_lane_below_the_bottom_tree_line | oracle_env.rs (same name) |
//! | env::test_one_room_episode_ends_at_the_first_correct_exit (2 params) | oracle_env.rs (same name) |
//! | env::test_curriculum_reset_preserves_the_first_shield_turn | oracle_env.rs (same name) |
//! | nearby_items::test_nearest_clock_then_heart_then_exit_without_changing_task | oracle_env.rs (same name) |
//! | nearby_items::test_nearer_exit_leaves_distant_helpful_drops | oracle_env.rs (same name) |
//! | quiet_enemy_penalty::test_quiet_fight_penalty_respects_grace_and_stops_without_targetable_enemies | here (same name) |

mod common;

use common::{TfzStore, ZCore, open_core};
use env::StateStore;
use env::config::{GameData, RamRandomization, RewardConfig, SpecialistConfig};
use env::critic::ROOM_STEP_GRACE;
use env::specialist_env::{EnvOptions, SpecialistEnv, StepResult};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::sync::{Arc, LazyLock};
use zelda::{Action, LoadOptions, Session};

type Env = SpecialistEnv<ZCore, TfzStore>;

static DATA: LazyLock<Arc<GameData>> =
    LazyLock::new(common::game_data);

/// `make_env`: an env with RAM randomization cleared, then `update` applied to the config.
fn make_env(
    nes: ZCore,
    specialist: &str,
    start_state: &str,
    update: impl FnOnce(&mut SpecialistConfig),
) -> Env {
    let mut config = (*DATA.specialists[specialist]).clone();
    config.ram_randomization = RamRandomization::default();
    update(&mut config);
    let opts = EnvOptions {
        start_state: Some(start_state.to_owned()),
        ..Default::default()
    };
    SpecialistEnv::new(Arc::new(config), DATA.clone(), nes, TfzStore::new(), opts).expect("env")
}

fn act(env: &mut Env, text: &str) -> StepResult {
    env.step(Action::parse(text).unwrap()).unwrap()
}

fn term(r: &StepResult, name: &str) -> Option<f64> {
    r.info.rewards.iter().find(|(k, _)| *k == name).map(|t| t.1)
}

/// The env borrows the emulator; giving it back leaves a working session (Python: `close` keeps a borrowed
/// session).
#[test]
fn close_keeps_borrowed_session() {
    let Some(nes) = open_core() else { return };
    let mut env = make_env(nes, "stalfos-keese-gel", "1_73s_0", |_| {});
    env.reset(Some(0), None).unwrap();
    let mut session = env.into_parts().0;
    let machine = TfzStore::new().load("1_73s_0").unwrap();
    session
        .load(
            &machine,
            &LoadOptions::default(),
            &mut ChaCha8Rng::seed_from_u64(0),
        )
        .unwrap();
    assert_eq!(session.state().location(), 0x73);
    let _: Session<ZCore> = session;
}

/// A stronger quiet-fight cost uses real shared progress and stops when the fight is over.
#[test]
fn quiet_fight_penalty_respects_grace_and_stops_without_targetable_enemies() {
    let Some(nes) = open_core() else { return };
    let mut env = make_env(nes, "goriya", "1_44w_0", |c| {
        c.no_progress_steps = Some(1000);
        c.max_steps = 1000;
        c.reward = RewardConfig {
            pbrs: false,
            quiet_enemy_penalty: 0.05,
            ..Default::default()
        };
    });
    env.reset(Some(0), None).unwrap();
    let all_active = |env: &Env| {
        let state = env.session().state();
        !state.enemies.is_empty() && state.enemies.iter().all(|e| e.is_active())
    };
    for _ in 0..100 {
        if all_active(&env) {
            break;
        }
        act(&mut env, "WAIT 4");
    }
    assert!(all_active(&env));
    // Freeze the room's monsters, without inventing hits or progress.
    let s = env.session_mut();
    s.nes_mut().poke(0x66C, 1);
    s.apply_ram(&[]);
    let snapshot = s.snapshot();
    env.restore(&snapshot, "1_44w_0").unwrap();

    for _ in 0..ROOM_STEP_GRACE + 50 {
        let r = act(&mut env, "WAIT 8");
        assert!(!(r.terminated || r.truncated));
        assert_eq!(term(&r, "penalty-quiet-enemies"), None);
        if env.linger.tail_steps() == ROOM_STEP_GRACE {
            break;
        }
    }
    assert_eq!(env.linger.tail_steps(), ROOM_STEP_GRACE);
    let r = act(&mut env, "WAIT 8");
    assert_eq!(env.linger.tail_steps(), ROOM_STEP_GRACE + 1);
    assert_eq!(term(&r, "penalty-quiet-enemies"), Some(-0.05));

    let s = env.session_mut();
    let slots: Vec<u8> = s.state().enemies.iter().map(|e| e.index).collect();
    let nes = s.nes_mut();
    for index in slots {
        nes.poke(0x34F + index as u16, 0);
    }
    nes.poke(0x34C, 0);
    s.apply_ram(&[]);
    let r = act(&mut env, "WAIT 8");
    assert!(env.linger.tail_steps() > ROOM_STEP_GRACE);
    assert_eq!(term(&r, "penalty-quiet-enemies"), None);
}
