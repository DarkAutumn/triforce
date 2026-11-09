//! Game-model and session tests on zcore, from the regenerated start states (`triforce/states`).
//!
//! Ports of the emulator tests in tests/test_core.py, tests/test_room_state.py and tests/test_room_entry.py,
//! on their chain-0 states (`1_44w_0`, `1_72e_0`, `1_74w_0`, `1_73s_0`, `0_77cave_0`, `start`, `1_35s_0`).
//! Every case runs inside one test on one emulator.
#![cfg(feature = "zcore")]

mod common;

use common::{Counted, load_named_state, open_core};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::BTreeSet;
use std::sync::Arc;
use zelda::enums::{AnimationKind, AnimationState, SwordKind, enemy};
use zelda::room_entry::{
    LEVEL_KILL_COUNTS, ROOM_FLAGS_UW, ROOM_HISTORY, is_pre_entry, pending_room,
};
use zelda::room_state::{DoorType, PushBlockState, SecretTrigger, decode_room_state};
use zelda::{
    Action, ActionKind, Change, Direction, GameMap, Interrupt, LoadOptions, MapLocation, Nes,
    Session, Var,
};

type S = Session<Counted>;

fn rng() -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(0)
}

fn load(s: &mut S, name: &str, overrides: &[(Var, u8)]) {
    let opts = LoadOptions {
        overrides: overrides.to_vec(),
        ..Default::default()
    };
    s.load(&load_named_state(name), &opts, &mut rng()).unwrap();
}

fn act(s: &mut S, text: &str) -> Change {
    s.act(Action::parse(text).unwrap()).unwrap()
}

fn act_until(s: &mut S, text: &str, pred: impl Fn(&S, &Change) -> bool, max: usize) -> Vec<Change> {
    let mut out = Vec::new();
    for _ in 0..max {
        let c = act(s, text);
        let done = pred(s, &c);
        out.push(c);
        if done {
            return out;
        }
    }
    panic!(
        "'{text}' never satisfied the predicate in {max} steps (at {:?})",
        s.state().full_location()
    );
}

const HURT: [(Var, u8); 2] = [(Var::HeartsAndContainers, 0x21), (Var::PartialHearts, 0xFF)];

fn full_health(containers: u8) -> [(Var, u8); 2] {
    [
        (
            Var::HeartsAndContainers,
            ((containers - 1) << 4) | (containers - 1),
        ),
        (Var::PartialHearts, 0xFF),
    ]
}

/// From the 0x44 entry (Link on the W door): MOVE E until he is off the door and the Goriyas are out.
fn leave_west_door(s: &mut S) {
    act_until(
        s,
        "MOVE E 16",
        |s, _| s.state().link.position.x >= 32 && s.state().has_active_enemies(),
        30,
    );
}

fn decodes_dungeon_and_overworld_states(s: &mut S) {
    load(s, "1_44w_0", &[]);
    let st = s.state().clone();
    assert_eq!(st.full_location(), MapLocation::new(1, 0x44, false));
    assert_eq!(st.sword(), SwordKind::White); // d2first: the white sword before Dungeon 1
    assert_eq!(
        st.enemies
            .iter()
            .filter(|e| e.id == enemy::RED_GORIYA)
            .count(),
        3
    );
    assert_eq!(
        st.bomb_walls().first().map(|w| w.wall_direction),
        Some(Direction::S)
    );

    load(s, "start", &[]);
    let st = s.state().clone();
    assert_eq!((st.level(), st.location()), (0, 0x77));
    assert_eq!(st.sword(), SwordKind::None);
    assert_eq!(st.max_health(), 3);
    let rs = decode_room_state(st.ram());
    assert!(rs.doors.is_empty() && !rs.has_item && !rs.item_taken);
    assert!(!rs.has_push_block && rs.secret_trigger == SecretTrigger::None);
}

fn masks(s: &mut S) {
    let bombs = (Var::Bombs, 4);
    load(s, "1_44w_0", &[HURT[0], HURT[1], bombs]);
    assert_eq!(
        s.mask().directions(ActionKind::Move),
        vec![Direction::W, Direction::E]
    );
    assert!(
        !s.mask().kind[ActionKind::Sword.index()],
        "next to the wall a weapon press would leave the room"
    );
    leave_west_door(s);
    assert!(s.mask().kind[ActionKind::Sword.index()] && s.mask().kind[ActionKind::Bomb.index()]);
    assert!(!s.mask().kind[ActionKind::Beams.index()]);

    let f = full_health(16);
    load(s, "1_44w_0", &f);
    leave_west_door(s);
    assert!(s.mask().kind[ActionKind::Beams.index()] && !s.mask().kind[ActionKind::Sword.index()]);

    load(s, "1_73s_0", &[]);
    act(s, "MOVE N 16");
    let m = *s.mask();
    assert!(m.kind[ActionKind::Move.index()] && m.kind[ActionKind::Wait.index()]);
    assert!(!m.kind[ActionKind::Sword.index()] && !m.kind[ActionKind::Bomb.index()]);
}

fn executor_timing(s: &mut S) {
    load(s, "1_73s_0", &[]);
    let start_y = s.state().link.position.y;
    let c = act(s, "MOVE N 16");
    assert_eq!(c.interrupt, None);
    assert!((16..=18).contains(&c.frames), "{}", c.frames);
    assert!(c.state.link.position.y < start_y);
    act_until(
        s,
        "MOVE W 32",
        |_, c| c.interrupt == Some(Interrupt::Wall),
        30,
    );
    let c = act(s, "MOVE W 32");
    assert_eq!(c.interrupt, Some(Interrupt::Wall));
    assert!(c.frames <= 2);

    load(s, "1_44w_0", &[]);
    act(s, "MOVE E 16");
    let mut reasons = BTreeSet::new();
    for _ in 0..40 {
        let c = act(s, "WAIT 32");
        reasons.insert(c.interrupt.map(|i| i.name()));
        if c.state.game_over() {
            break;
        }
    }
    assert!(
        reasons.iter().any(Option::is_some),
        "Goriyas never interrupted a WAIT"
    );
}

fn look_ahead(s: &mut S) {
    load(s, "1_44w_0", &full_health(16));
    leave_west_door(s);
    let mut credited = 0;
    let mut lined_up = false;
    for _ in 0..40 {
        let st = s.state().clone();
        let Some(target) = st
            .active_enemies()
            .min_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap())
        else {
            break;
        };
        let (dx, dy) = (
            target.position.x - st.link.position.x,
            target.position.y - st.link.position.y,
        );
        if dy.abs() < 8 && s.mask().kind[ActionKind::Beams.index()] {
            let d = if dx > 0 { Direction::E } else { Direction::W };
            let c = s
                .act(Action::new(ActionKind::Beams, d, 0).unwrap())
                .unwrap();
            credited += c.damage_dealt();
            let follow = act(s, "WAIT 32");
            assert!(
                follow.enemies_hit.is_empty() || credited == 0,
                "beam damage counted twice"
            );
            if credited > 0 {
                lined_up = true;
                break;
            }
        } else {
            act(s, if dy > 0 { "MOVE S 8" } else { "MOVE N 8" });
        }
    }
    assert!(lined_up, "never landed a beam shot from 1_44");

    // look_ahead=false emulates only real frames; with it the beam runs hidden frames.
    load(s, "1_44w_0", &full_health(16));
    leave_west_door(s);
    assert!(s.mask().kind[ActionKind::Beams.index()]);
    let snap = s.snapshot();
    let mut hidden = Vec::new();
    for on in [true, false] {
        s.restore(&snap);
        s.look_ahead = on;
        let (runs, frame) = (s.nes().1, s.frame());
        s.act(Action::new(ActionKind::Beams, Direction::E, 0).unwrap())
            .unwrap();
        hidden.push((s.nes().1 - runs) - (s.frame() - frame));
    }
    s.look_ahead = true;
    assert!(hidden[0] > 0 && hidden[1] == 0, "{hidden:?}");
}

fn snapshot_restore_is_deterministic(s: &mut S) {
    load(s, "1_44w_0", &[]);
    let snap = s.snapshot();
    let script = ["MOVE E 16", "WAIT 8", "MOVE S 8", "SWORD E"];
    let first: Vec<_> = script
        .iter()
        .map(|a| act(s, a).state.link.position)
        .collect();
    s.restore(&snap);
    let second: Vec<_> = script
        .iter()
        .map(|a| act(s, a).state.link.position)
        .collect();
    assert_eq!(first, second);
    assert_eq!(s.steps, 4, "restore takes the snapshot's step count");
}

fn caves(s: &mut S) {
    load(s, "0_77cave_0", &[]);
    assert!(s.state().in_cave());
    act_until(
        s,
        "MOVE N 16",
        |s, _| !s.mask().direction[ActionKind::Move.index()][0],
        30,
    );
    assert!(s.state().link.position.y < 0x8E);
    assert_eq!(s.state().sword(), SwordKind::None);

    load(s, "0_77cave_0", &[]);
    let rs = decode_room_state(s.state().ram());
    assert!(rs.in_cave && rs.is_overworld() && rs.doors.is_empty());
    assert!(
        rs.has_item
            && rs.item_id == Some(0x01)
            && rs.item_visible
            && rs.item_position == Some((0x78, 0x98))
    );
    assert_eq!(
        rs.cave_wares.iter().map(|w| w.item_id).collect::<Vec<_>>(),
        vec![0x01]
    );
    assert_eq!(rs.cave_wares[0].price, 0);
    assert_eq!(rs.enemies_remaining, 0, "the old man's fires are not foes");
    let mut rs = rs;
    for _ in 0..30 {
        let cmd = if s.state().link.position.x < 0x78 - 4 {
            "MOVE E 4"
        } else {
            "MOVE N 8"
        };
        act(s, cmd);
        rs = decode_room_state(s.state().ram());
        if !rs.has_item {
            break;
        }
    }
    assert!(!rs.has_item && rs.item_taken && s.state().sword() == SwordKind::Wood);
}

fn walk_three_rooms(s: &mut S) {
    load(s, "start", &[]);
    let mut visited = vec![s.state().location()];
    for d in ["E", "W", "W"] {
        let changes = act_until(s, &format!("MOVE {d} 32"), |_, c| c.changed_location(), 30);
        visited.push(changes.last().unwrap().state.location());
    }
    assert_eq!(visited, vec![0x77, 0x78, 0x77, 0x76]);
}

fn room_states(s: &mut S) {
    load(s, "1_44w_0", &[]);
    let rs = decode_room_state(s.state().ram());
    assert_eq!((rs.level, rs.room, rs.in_cave), (1, 0x44, false));
    let doors: Vec<(Direction, DoorType, bool)> = rs
        .doors
        .iter()
        .map(|(d, x)| (*d, x.door_type, x.is_open))
        .collect();
    assert_eq!(
        doors,
        vec![
            (Direction::N, DoorType::Wall, false),
            (Direction::S, DoorType::Bomb, false),
            (Direction::W, DoorType::Key, true),
            (Direction::E, DoorType::Open, true),
        ]
    );
    assert_eq!(rs.door(Direction::S).unwrap().kind(), "bomb");
    assert_eq!(rs.open_doors(), vec![Direction::W, Direction::E]);
    assert_eq!(rs.secret_trigger, SecretTrigger::AllDeadItem);
    assert!(
        rs.has_item && rs.item_id == Some(0x1D) && !rs.item_visible && rs.item_hidden_until_clear
    );
    assert_eq!(rs.item_position, Some((0x80, 0x70)));
    assert!(rs.enemies_remaining == 3 && rs.enemies_unspawned == 3 && !rs.all_dead());
    assert!(rs.visited && !rs.item_taken && rs.kill_state == 0);
    act_until(
        s,
        "WAIT 8",
        |s, _| decode_room_state(s.state().ram()).enemies_unspawned == 0,
        30,
    );
    let rs = decode_room_state(s.state().ram());
    assert!(rs.enemies_remaining == 3 && rs.enemies_unspawned == 0 && !rs.all_dead());
    assert!(rs.has_item && !rs.item_visible && rs.item_hidden_until_clear);

    load(s, "1_72e_0", &[]);
    let rs = decode_room_state(s.state().ram());
    assert!(
        rs.item_id == Some(0x19)
            && rs.item_position == Some((0xA0, 0xC0))
            && rs.item_hidden_until_clear
    );
    assert!(rs.enemies_remaining == 3 && rs.enemies_unspawned == 3);

    load(s, "1_74w_0", &[]);
    for _ in 0..4 {
        let st = act(s, "WAIT 8").state;
        let rs = decode_room_state(st.ram());
        assert!(
            rs.item_id == Some(0x19)
                && rs.item_visible
                && rs.item_carried
                && !rs.item_hidden_until_clear
        );
        assert_eq!(
            rs.item_position,
            Some((st.ram()[0x71] as i32, st.ram()[0x85] as i32))
        );
    }
    assert_eq!(
        decode_room_state(s.state().ram()).push_block_state,
        PushBlockState::None
    );
}

fn boomerang_stays_active_until_caught(s: &mut S) {
    load(s, "1_35s_0", &[]);
    act_until(s, "MOVE N 32", |s, _| s.nes().peek(0x53) == 0, 30);
    act(s, "BOOMERANG E");
    let c = act_until(
        s,
        "WAIT 4",
        |s, _| (0x40..=0x57).contains(&s.nes().peek(0xBB)),
        30,
    )
    .pop()
    .unwrap();
    assert_eq!(
        c.state.animation_state(AnimationKind::Boomerang),
        AnimationState::Active
    );
    let c = act_until(s, "WAIT 4", |s, _| s.nes().peek(0xBB) == 0, 60)
        .pop()
        .unwrap();
    assert_eq!(
        c.state.animation_state(AnimationKind::Boomerang),
        AnimationState::Inactive
    );
    let bait = s
        .apply_ram(&[(Var::BaitOrBoomerangAnimation, 0x80)])
        .clone();
    assert_eq!(
        bait.animation_state(AnimationKind::Boomerang),
        AnimationState::Inactive
    );
}

fn layout(s: &S) -> Vec<(i32, i32)> {
    let mut v: Vec<(i32, i32)> = s
        .state()
        .enemies
        .iter()
        .filter(|e| !e.is_dying())
        .map(|e| (e.position.x, e.position.y))
        .collect();
    v.sort();
    v
}

fn pre_entry_loads(s: &mut S) {
    // 1_72 is saved before D1 0x72's EnterRoom (3 Keese): the spawn cycle decides their layout.
    let state = load_named_state("1_72e_0");
    let opts = LoadOptions {
        randomize_rng: true,
        ..Default::default()
    };
    let mut layouts = BTreeSet::new();
    for seed in 0..9 {
        s.load(&state, &opts, &mut ChaCha8Rng::seed_from_u64(seed))
            .unwrap();
        layouts.insert(layout(s));
    }
    assert!(layouts.len() >= 3, "{layouts:?}");
    s.load(&state, &opts, &mut ChaCha8Rng::seed_from_u64(4))
        .unwrap();
    let first = layout(s);
    s.load(&state, &opts, &mut ChaCha8Rng::seed_from_u64(4))
        .unwrap();
    assert_eq!(layout(s), first);

    // A room the game remembers as cleared (kill state 3, in the room history) spawns nobody unless fresh_room
    // clears its history before entry.
    s.nes_mut().restore(&state);
    assert!(
        is_pre_entry(|a| s.nes().peek(a)) && pending_room(|a| s.nes().peek(a)) == Some((1, 0x72))
    );
    let nes = s.nes_mut();
    let flags = nes.peek(ROOM_FLAGS_UW + 0x72);
    nes.poke(ROOM_FLAGS_UW + 0x72, flags | 0xC0);
    nes.poke(LEVEL_KILL_COUNTS + 0x72, 3);
    nes.poke(ROOM_HISTORY, 0x72);
    let cleared = nes.snapshot();
    s.load(&cleared, &LoadOptions::default(), &mut rng())
        .unwrap();
    assert_eq!(s.state().enemies.len(), 0);
    s.load(
        &cleared,
        &LoadOptions {
            fresh_room: true,
            ..Default::default()
        },
        &mut rng(),
    )
    .unwrap();
    assert_eq!(s.state().enemies.len(), 3);
    assert_eq!(
        act(s, "WAIT 8").state.full_location(),
        MapLocation::new(1, 0x72, false)
    );
}

/// A boomerang shot already in Link's shield box in D1 0x73 (`_place_shield_projectile`).
fn place_shield_projectile(s: &mut S, slots: &[u16]) {
    load(s, "1_73s_0", &[]);
    act(s, "MOVE N 16");
    s.apply_ram(&[
        (Var::LinkX, 128),
        (Var::LinkY, 144),
        (Var::LinkDirection, 1),
        (Var::HeartsAndContainers, 0x22),
        (Var::PartialHearts, 255),
    ]);
    let nes = s.nes_mut();
    for a in [0xAC, 0x3D, 0x4F0, 0x66C, 0x512] {
        nes.poke(a, 0);
    }
    for &slot in slots {
        for (base, v) in [
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
            nes.poke(base + slot, v);
        }
    }
    nes.poke(0x34C, slots.len() as u8);
    s.apply_ram(&[]);
}

fn real_shield_blocks(s: &mut S) {
    place_shield_projectile(s, &[11]);
    s.apply_ram(&[(Var::HeartsAndContainers, 0x21), (Var::PartialHearts, 255)]);
    for (base, v) in [
        (0x34F, 0x60),
        (0xAC, 0x22),
        (0x70, 128),
        (0x84, 144),
        (0x405, 0),
        (0x3A8, 0xC0),
    ] {
        s.nes_mut().poke(base + 1, v);
    }
    s.apply_ram(&[]);
    let c = act(s, "WAIT 4");
    assert!(
        c.health_gained == 1.0 && c.health_lost == 0.0,
        "{} {}",
        c.health_gained,
        c.health_lost
    );
    assert_eq!(c.blocks(), 1, "a block can coincide with a heart pickup");

    for duration in [4, 32] {
        place_shield_projectile(s, &[10, 11]);
        let start = s.frame();
        let c = act(s, &format!("WAIT {duration}"));
        assert_eq!(c.blocks(), 2);
        assert_eq!(
            c.projectile_blocks
                .iter()
                .map(|b| b.slot)
                .collect::<BTreeSet<_>>(),
            [10, 11].into_iter().collect()
        );
        assert!(
            c.projectile_blocks
                .iter()
                .all(|b| start < b.frame && b.frame <= start + c.frames as u64)
        );
        assert!(c.health_lost == 0.0 && c.interrupt.is_none());
        assert_eq!(act(s, "WAIT 4").blocks(), 0);
    }
}

#[test]
fn game_model_on_zcore() {
    let Some(nes) = open_core() else { return };
    let nes = Counted(nes, 0);
    let mut s = Session::new(nes, Arc::new(GameMap::builtin()));
    s.set_disabled_kinds(&[]);
    decodes_dungeon_and_overworld_states(&mut s);
    masks(&mut s);
    executor_timing(&mut s);
    look_ahead(&mut s);
    snapshot_restore_is_deterministic(&mut s);
    caves(&mut s);
    walk_three_rooms(&mut s);
    room_states(&mut s);
    boomerang_stays_active_until_caught(&mut s);
    pre_entry_loads(&mut s);
    real_shield_blocks(&mut s);
}
