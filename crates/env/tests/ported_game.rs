//! Session and game-model tests on the named start states (zcore), ported from the Python tests
//! before the Python game stack is deleted (plan step 16).  Every state comes from `common::load_named_state`.
//!
//! Mapping of tests/test_core.py, tests/test_room_state.py and tests/test_room_entry.py (zelda = crates/zelda):
//!
//! | Python test | Rust |
//! |---|---|
//! | core::test_action_indices_round_trip | zelda src/actions.rs `action_indices_round_trip` |
//! | core::test_from_indices_ignores_heads_that_do_not_apply | zelda src/actions.rs (same name) |
//! | core::test_invalid_actions_are_rejected | zelda src/actions.rs (same name) |
//! | core::test_mask_flat_round_trip | zelda src/actions.rs (same name) |
//! | core::test_decodes_dungeon_savestate | here `decodes_dungeon_and_overworld_savestates` (game.rs covers it on the chain state without the bomb count) |
//! | core::test_decodes_overworld_start | here `decodes_dungeon_and_overworld_savestates`; zelda game.rs `decodes_dungeon_and_overworld_states` |
//! | core::test_north_shutter_is_barred_not_a_wall | here `north_shutter_is_barred_not_a_wall` |
//! | core::test_boomerang_stays_active_on_return_until_caught | zelda game.rs `boomerang_stays_active_until_caught` |
//! | core::test_doorway_restricts_moves_and_weapons | zelda game.rs `masks` |
//! | core::test_sword_and_beams_are_exclusive | zelda game.rs `masks` |
//! | core::test_no_weapons_in_empty_room | zelda game.rs `masks` |
//! | core::test_push_block_move_is_legal | here `push_block_move_is_legal` |
//! | core::test_wall_move_costs_at_most_two_frames | zelda game.rs `executor_timing` |
//! | core::test_timed_move_runs_requested_duration | zelda game.rs `executor_timing` |
//! | core::test_wait_is_interrupted_in_combat | zelda game.rs `executor_timing` (`Interrupt` is a closed enum, so "reasons ⊆ Interrupt.ALL" holds by type) |
//! | core::test_shield_collision_* (8 tests), core::test_magic_*, core::test_boomerang_disappearance_*, core::test_immune_clink_*, core::test_fireball_disappearance_*, core::test_unrelated_clink_*, core::test_real_before_trace_* | zelda src/blocks.rs unit tests (same names) |
//! | core::test_executor_counts_every_collision_without_interrupting_or_replacing_callback | zelda src/executor.rs (same name, scripted `Nes` double) |
//! | core::test_executor_counts_blocks_in_attack_settling_and_uncontrollable_frames | zelda src/executor.rs (same name) |
//! | core::test_real_shield_block_can_coincide_with_a_heart_pickup | zelda game.rs `real_shield_blocks` |
//! | core::test_real_projectile_block_counts_are_independent_of_option_duration | zelda game.rs `real_shield_blocks` |
//! | core::test_block_collection_preserves_real_seeded_timing_state_and_user_callback | here `block_collection_preserves_timing_state_and_frame_sink` |
//! | core::test_push_block_opens | here `push_block_opens` |
//! | core::test_scripted_walk_through_three_rooms | zelda game.rs `walk_three_rooms` |
//! | core::test_beam_look_ahead_credits_firing_step | zelda game.rs `look_ahead` |
//! | core::test_without_look_ahead_a_beam_emulates_only_real_frames | zelda game.rs `look_ahead` |
//! | core::test_snapshot_restore_is_deterministic | zelda game.rs `snapshot_restore_is_deterministic` |
//! | core::test_cave_person_line_blocks_north | zelda game.rs `caves` |
//! | room_state::test_goriya_room_doors_trigger_and_hidden_item | zelda game.rs `room_states` |
//! | room_state::test_key_rooms_hidden_and_carried | zelda game.rs `room_states` |
//! | room_state::test_cleared_boss_room_and_persisted_flags | here `cleared_boss_room_and_persisted_flags` |
//! | room_state::test_push_block_progress | here `push_block_progress` |
//! | room_state::test_bombing_the_south_wall_opens_it | here `bombing_the_south_wall_opens_it` |
//! | room_state::test_sword_cave_ware | zelda game.rs `caves` |
//! | room_state::test_overworld_rooms | here `overworld_rooms` (zelda game.rs covers the start room) |
//! | room_state::test_map_and_compass_use_the_level_bit | here `map_and_compass_use_the_level_bit` |
//! | room_entry::test_pre_entry_state_draws_a_new_layout_per_load_and_reproduces_per_seed | zelda game.rs `pre_entry_loads` (ChaCha8 seeds replace `random.seed`) |
//! | room_entry::test_fresh_room_refills_a_room_the_game_remembers_as_cleared | here `fresh_room_refills_a_room_the_game_remembers_as_cleared` (zelda game.rs `pre_entry_loads` checks the same rule on poked flags) |
//!
//! Every test skips when the ROM is missing.

mod common;

use common::{ZCore, load_named_state, open_core};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use zelda::enums::{SwordKind, enemy};
use zelda::room_entry::{is_pre_entry, pending_room};
use zelda::room_state::{DoorType, PushBlockState, SecretTrigger, decode_room_state, room_flags};
use zelda::{
    Action, ActionKind, Change, Direction, GameMap, GameView, LoadOptions, MapLocation, Nes,
    Session, Var, compute_action_mask,
};

type S = Session<ZCore>;

fn session() -> Option<S> {
    open_core().map(|nes| Session::new(nes, Arc::new(GameMap::builtin())))
}

fn rng() -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(0)
}

/// `session.load(name, randomize_rng=False, fresh_room=..., ram_overrides=...)`.
fn load(s: &mut S, name: &str, fresh_room: bool, overrides: &[(Var, u8)]) -> Arc<GameView> {
    let opts = LoadOptions {
        fresh_room,
        overrides: overrides.to_vec(),
        ..Default::default()
    };
    let machine = load_named_state(name).expect("state");
    s.load(&machine, &opts, &mut rng()).expect("load").clone()
}

fn act(s: &mut S, text: &str) -> Change {
    s.act(Action::parse(text).unwrap()).unwrap()
}

/// Repeats an action until `pred(change)` holds (conftest.act_until).
fn act_until(
    s: &mut S,
    text: &str,
    pred: impl Fn(&Change) -> bool,
    max_steps: usize,
) -> Vec<Change> {
    let mut changes = Vec::new();
    for _ in 0..max_steps {
        let change = act(s, text);
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

/// Loads a pre-entry state, WAITs until a step runs uninterrupted (the room's objects exist), then writes `ram`
/// (conftest.load_and_place: load-time overrides of Link's position don't stick on pre-entry states).
fn load_and_place(s: &mut S, name: &str, ram: &[(Var, u8)]) -> Arc<GameView> {
    load(s, name, false, &[]);
    act_until(s, "WAIT 8", |c| c.interrupt.is_none(), 30);
    s.apply_ram(ram).clone()
}

/// Link standing just N of 1_22e's push block, after the room has created it.
fn place_at_push_block(s: &mut S) -> Arc<GameView> {
    load_and_place(s, "1_22e_0", &[(Var::LinkX, 96), (Var::LinkY, 125)])
}

fn door(rs: &zelda::room_state::RoomState, d: Direction) -> zelda::room_state::DoorState {
    rs.door(d).unwrap_or_else(|| panic!("no {d:?} door"))
}

// --- tests/test_core.py ---

#[test]
fn decodes_dungeon_and_overworld_savestates() {
    let Some(mut s) = session() else { return };
    let state = load(&mut s, "1_44w_0", false, &[]);
    assert_eq!(state.full_location(), MapLocation::new(1, 0x44, false));
    // On d2first Link reaches D1 0x44 with the white sword and what is left of Dungeon 2's bombs (decoded from
    // $0658): one in chain 0.
    assert_eq!(state.bombs(), 1);
    assert_eq!(state.sword(), SwordKind::White);
    let goriyas = state
        .enemies
        .iter()
        .filter(|e| e.id == enemy::RED_GORIYA)
        .count();
    assert_eq!(goriyas, 3);
    assert_eq!(
        state.bomb_walls().first().map(|w| w.wall_direction),
        Some(Direction::S)
    );
    let dodongo = load(&mut s, "2_0es_0", false, &[]);
    assert!(dodongo.bombs() > 0);
    assert_eq!(i64::from(dodongo.bombs()), i64::from(s.nes().peek(0x658)));

    let state = load(&mut s, "start", false, &[]);
    assert_eq!((state.level(), state.location()), (0, 0x77));
    assert_eq!(state.sword(), SwordKind::None);
    assert_eq!(state.max_health(), 3);
}

#[test]
fn north_shutter_is_barred_not_a_wall() {
    let Some(mut s) = session() else { return };
    let state = load(&mut s, "2_2es_0", true, &[]);
    assert!(state.is_door_barred(Direction::N));
    assert!(!state.is_door_open(Direction::N));
    assert!(
        !state
            .room()
            .is_wall_intact(Direction::N, state.current_tiles())
    );
}

#[test]
fn push_block_move_is_legal() {
    let Some(mut s) = session() else { return };
    let state = place_at_push_block(&mut s);
    assert!(state.push_block_pending());
    assert!(state.is_push_block_direction(Direction::S));
    assert!(!state.can_link_move(Direction::S));
    assert!(
        compute_action_mask(&state)
            .is_allowed(&Action::new(ActionKind::Move, Direction::S, 32).unwrap())
    );
}

/// A boomerang shot already in Link's shield box in D1 0x73 (`_place_shield_projectile`).
fn place_shield_projectile(s: &mut S, slots: &[u16]) {
    load(s, "1_73s_0", false, &[]);
    act(s, "MOVE N 16"); // off the entry door, in playable mode before RAM placement
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

fn pixels(s: &S) -> Box<[u8; 256 * 240]> {
    Box::new(*s.nes().pixels().expect("the last frame was rendered"))
}

/// Block collection only reads RAM: an option that collects a real block emulates exactly the frames a bare
/// no-input run of the same length does (RAM, PRG-RAM, pixels), and the frame sink sees every real frame and
/// stays installed.  Python compared against a run with `ShieldCollisionWatch.collisions` monkeypatched out;
/// Rust can't patch the watch, so the comparison run is the emulator stepped directly with no buttons (a WAIT
/// presses none: zelda src/executor.rs), which is a stricter reference.
#[test]
fn block_collection_preserves_timing_state_and_frame_sink() {
    let Some(mut s) = session() else { return };
    place_shield_projectile(&mut s, &[11]);
    s.set_render(true);
    let snapshot = s.snapshot();
    let observed = Arc::new(AtomicU64::new(0));
    let seen = observed.clone();
    s.set_frame_sink(Some(Box::new(move |_: &ZCore| {
        seen.fetch_add(1, Ordering::Relaxed);
    })));
    let active = act(&mut s, "WAIT 16");
    let (active_ram, active_prg) = (*s.nes().ram(), *s.nes().prg_ram());
    let active_screen = pixels(&s);
    assert_eq!(observed.load(Ordering::Relaxed), active.frames as u64);
    assert_eq!(active.blocks(), 1);

    s.restore(&snapshot);
    for _ in 0..active.frames {
        s.nes_mut().run_frame(0, true);
    }
    assert_eq!(*s.nes().ram(), active_ram);
    assert_eq!(*s.nes().prg_ram(), active_prg);
    assert_eq!(pixels(&s), active_screen);

    s.restore(&snapshot);
    observed.store(0, Ordering::Relaxed);
    let again = act(&mut s, "WAIT 16");
    assert_eq!(
        observed.load(Ordering::Relaxed),
        again.frames as u64,
        "the sink stays installed"
    );
    assert_eq!(
        (
            again.frames,
            again.control_frames,
            again.interrupt,
            again.blocks()
        ),
        (active.frames, active.control_frames, active.interrupt, 1)
    );
    s.set_frame_sink(None);
}

#[test]
fn push_block_opens() {
    let Some(mut s) = session() else { return };
    place_at_push_block(&mut s);
    act(&mut s, "MOVE S 32");
    act(&mut s, "WAIT 32");
    assert!(s.nes().peek(0x4CF) >= 1, "BlockPushComplete");
    assert!(!s.state().push_block_pending());
}

// --- tests/test_room_state.py ---

#[test]
fn cleared_boss_room_and_persisted_flags() {
    let Some(mut s) = session() else { return };
    // No savestate starts in a cleared 0x35: walk back W from the triforce room (Aquamentus dead, heart taken).
    load(&mut s, "1_36w_2", false, &[]);
    act_until(&mut s, "MOVE W 16", |c| c.state.location() == 0x35, 30);
    let state = s.state().clone();
    let rs = decode_room_state(state.ram());
    assert_eq!((rs.level, rs.room), (1, 0x35));
    assert!(rs.is_boss_room && !rs.is_triforce_room);
    let south = door(&rs, Direction::S);
    assert!(south.door_type == DoorType::Key && south.is_open);
    let east = door(&rs, Direction::E);
    assert!(
        east.door_type == DoorType::Shutter && east.is_open,
        "CurOpenedDoors only"
    );
    assert!(rs.item_taken && !rs.has_item && rs.kill_state == 3 && rs.all_dead());
    // Flag bytes ($6FF+room): $53:F0 and $35:F4 match ram-catalog R.1 (the boss room's kill/item bits); the
    // other rooms on the walk depend on what the chain killed on its way, so only their visited bit is pinned.
    assert_eq!(room_flags(state.ram(), 0x53), 0xF0);
    assert_eq!(room_flags(state.ram(), 0x35), 0xF4);
    for room in [0x74, 0x52] {
        assert_ne!(room_flags(state.ram(), room) & 0x80, 0, "room {room:#x} visited");
    }
    // In a dungeon the room flags are the level block at $6FF.
    for room in [0x74, 0x53, 0x35, 0x52, 0x43] {
        assert_eq!(room_flags(state.ram(), room), state.ram()[0x6FF + room as usize]);
    }
    assert_eq!(room_flags(state.ram(), state.location()), rs.flags);
}

#[test]
fn push_block_progress() {
    let Some(mut s) = session() else { return };
    place_at_push_block(&mut s);
    let rs = decode_room_state(s.state().ram());
    assert!(
        rs.has_push_block
            && rs.push_block_state == PushBlockState::Pending
            && rs.block_push_complete == 0
    );
    let east = door(&rs, Direction::E);
    assert!(east.door_type == DoorType::Key && east.is_open);
    act(&mut s, "MOVE S 32");
    let moving = decode_room_state(s.state().ram()).push_block_state;
    assert!(
        matches!(moving, PushBlockState::Moving | PushBlockState::Done),
        "{moving:?}"
    );
    act(&mut s, "WAIT 32");
    let rs = decode_room_state(s.state().ram());
    assert!(rs.push_block_state == PushBlockState::Done && rs.block_push_complete >= 1);
}

#[test]
fn bombing_the_south_wall_opens_it() {
    let Some(mut s) = session() else { return };
    load_and_place(
        &mut s,
        "1_44w_0",
        &[
            (Var::Bombs, 4), // regenerated 1_44w_0 has no bombs: give the precondition the legacy state had
            (Var::HeartsAndContainers, 0xFF),
            (Var::PartialHearts, 0xFF),
            (Var::LinkX, 120),
            (Var::LinkY, 189),
            (Var::LinkDirection, 4),
        ],
    );
    act(&mut s, "BOMB S");
    let mut opened = false;
    for _ in 0..10 {
        act(&mut s, "WAIT 16");
        let south = door(&decode_room_state(s.state().ram()), Direction::S);
        assert_eq!(south.door_type, DoorType::Bomb);
        if south.is_open {
            opened = true;
            break;
        }
    }
    assert!(opened);
}

#[test]
fn overworld_rooms() {
    let Some(mut s) = session() else { return };
    let rs = decode_room_state(load(&mut s, "start", false, &[]).ram());
    assert!(
        (rs.level, rs.room) == (0, 0x77) && rs.doors.is_empty() && !rs.has_item && !rs.item_taken
    );
    assert!(!rs.has_push_block && rs.secret_trigger == SecretTrigger::None);
    // No regenerated state enters an overworld room with a floor item outdoors (the legacy 0_5fs heart container
    // at 0x5f is off every route); the sword cave is the overworld-level room with one.
    let rs = decode_room_state(load(&mut s, "0_77cave_0", false, &[]).ram());
    assert!(rs.is_overworld() && rs.in_cave);
    assert!(rs.item_id == Some(0x01) && rs.item_visible && rs.item_position == Some((0x78, 0x98)));
}

#[test]
fn map_and_compass_use_the_level_bit() {
    let Some(mut s) = session() else { return };
    let state = load(
        &mut s,
        "2_7ds_0",
        false,
        &[(Var::Map, 0x01), (Var::Compass, 0x02)],
    );
    let rs = decode_room_state(state.ram());
    assert_eq!(rs.level, 2);
    assert!(
        !rs.has_map && rs.has_compass,
        "D1's map bit does not count in D2"
    );
    assert!(!state.has_map() && state.has_compass());
    // D1 with the map and no compass: 1_44w_0 (after the 0x43 map pickup; the legacy used 1_42e).
    let rs = decode_room_state(load(&mut s, "1_44w_0", false, &[]).ram());
    assert!(rs.has_map && !rs.has_compass);
}

// --- tests/test_room_entry.py ---

/// From 0x59 (entered from 0x58, where chain 0 killed 1 of the 4 Octoroks), walking W back into 0x58: the game
/// remembers the kills and spawns only the 3 left unless the room's kill history is cleared before entry (zelda
/// game.rs `pre_entry_loads` covers a fully cleared room).
#[test]
fn fresh_room_refills_a_room_the_game_remembers_as_cleared() {
    let Some(mut s) = session() else { return };
    load(&mut s, "0_59w_0", false, &[]);
    s.set_capture_pre_entry(true);
    let changes = act_until(&mut s, "MOVE W 16", |c| c.changed_location(), 12);
    s.set_capture_pre_entry(false);
    let last = changes.last().unwrap();
    assert_eq!(last.state.full_location(), MapLocation::new(0, 0x58, false));
    assert_eq!(last.state.enemies.len(), 3);
    let pre_entry = s.last_pre_entry().expect("captured").machine.clone();

    s.nes_mut().restore(&pre_entry);
    assert!(is_pre_entry(|a| s.nes().peek(a)));
    assert_eq!(pending_room(|a| s.nes().peek(a)), Some((0, 0x58)));
    // Python's load defaults to randomize_rng=True.
    let mut opts = LoadOptions {
        randomize_rng: true,
        ..Default::default()
    };
    let state = s.load(&pre_entry, &opts, &mut rng()).unwrap();
    assert_eq!(state.enemies.len(), 3);
    opts.fresh_room = true;
    let state = s.load(&pre_entry, &opts, &mut rng()).unwrap();
    assert_eq!(state.enemies.len(), 4);
    assert_eq!(
        act(&mut s, "WAIT 8").state.full_location(),
        MapLocation::new(0, 0x58, false)
    );
}
