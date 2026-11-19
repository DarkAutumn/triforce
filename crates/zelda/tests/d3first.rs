//! Phase X: the route `d3first` (routes.yaml; docs/routes/d3first.md) as the engine reads it: its length and end,
//! its triforces in order, what Link holds at each dungeon's entrance (sword, heart containers, items, pieces), the
//! whirlwind's four warps (plays and facings from the game's index, `zelda::whirlwind::plays`), its stretches (the
//! shared blocks'), and every step's room, entry and exit a door of game.yaml leading into the next step's room. Its
//! game.yaml rooms are its own map's (`GameMap::route_map`): `d2first` plays 0:0x2d without the rock wall d3first
//! bombs and 0:0x3d without the Armos stairs d3first takes (both in the real game, from d2first's own start states).
#![cfg(feature = "zcore")]

mod common;

use common::{Counted, load_named_state, open_core};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::BTreeSet;
use std::sync::Arc;
use zelda::cheats::{CheatItem, needs};
use zelda::enums::SwordKind;
use zelda::game_map::{RouteStep, whirlwind_level};
use zelda::objectives::{GameCompletion, ObjectiveKind, StepProgress, route_entry, route_progress};
use zelda::whirlwind::{FACING_DOWN, FACING_UP, plays, step_index};
use zelda::{Action, ActionKind, Direction, GameMap, LoadOptions, MapLocation, Session, Var};

const ROUTE: &str = "d3first";

/// The map d3first plays (its own game.yaml rooms over the shared ones).
fn map() -> Arc<GameMap> {
    Arc::new(GameMap::builtin()).route_map(ROUTE)
}

fn steps(map: &GameMap) -> &[RouteStep] {
    &map.routes[ROUTE]
}

/// The route's step `room` (level, room id) entered from `entry` and left by `exit` (the first such).
fn index(steps: &[RouteStep], level: u8, room: u8, entry: &str, exit: &str) -> usize {
    steps
        .iter()
        .position(|s| (s.level, s.room, s.in_cave, s.entry.as_str(), s.exit.as_str()) == (level, room, false, entry, exit))
        .unwrap_or_else(|| panic!("{ROUTE} has no step {level}:{room:#04x} {entry} -> {exit}"))
}

/// Max hearts at step `i` as the audit counts them (`triforce/entry_states.py` RouteIndex.hearts): 3, and one per
/// heart-container room (game.yaml treasure) the route went through before it.
fn hearts(map: &GameMap, steps: &[RouteStep], i: usize) -> usize {
    let rooms: BTreeSet<MapLocation> = steps[..i]
        .iter()
        .filter(|s| s.way.is_none_or(|w| w.clips()))
        .map(RouteStep::location)
        .filter(|&l| map.get(l).and_then(|r| r.treasure.as_deref()) == Some("heart-container"))
        .collect();
    3 + rooms.len()
}

fn pieces(levels: &[u8]) -> u8 {
    levels.iter().fold(0, |bits, level| bits | 1 << (level - 1))
}

#[test]
fn the_route_runs_from_the_start_to_the_end_of_the_game() {
    let map = map();
    let steps = steps(&map);
    assert_eq!(steps.len(), 308);
    let (first, last) = (&steps[0], steps.last().unwrap());
    assert_eq!((first.level, first.room, first.entry.as_str(), first.exit.as_str()), (0, 0x77, "start", "cave"));
    assert_eq!((last.level, last.room, last.exit.as_str()), (9, 0x32, "end"));
    assert_eq!(map.namespace(ROUTE), Some("d3first"));
    let triforces: Vec<u8> = steps.iter().filter(|s| s.exit == "triforce").map(|s| s.level).collect();
    assert_eq!(triforces, [3, 4, 1, 5, 2, 7, 6, 8]);
    let stretches: Vec<&str> = map.stretches[ROUTE].iter().map(|s| s.name.as_str()).collect();
    assert_eq!(stretches, ["d4-island", "dock-44", "woods", "ring-07", "arrow-20", "arrow-20n"], "the blocks' only");
}

/// The wood sword until the grave's magic sword, the items and pieces in the route's order, and the heart containers
/// the owner's list gives: 3, +1 for each of Dungeons 3, 4, 1, the heart rock, Dungeon 5, the ladder heart, the raft
/// heart, Dungeons 2, 7, 6, 8 and 0x47's.
#[test]
fn link_holds_the_routes_things_at_each_dungeons_entrance() {
    let map = map();
    let steps = steps(&map);
    let progress = route_progress(&map, steps);
    let entrance = |level: u8| steps.iter().position(|s| s.level == level).unwrap();
    let held = |p: &StepProgress| {
        [("bow", p.bow), ("raft", p.raft), ("ladder", p.ladder), ("recorder", p.recorder), ("ring", p.ring)]
            .iter()
            .filter(|(_, held)| *held)
            .map(|(name, _)| *name)
            .collect::<Vec<_>>()
    };
    use SwordKind::*;
    let want: [(u8, SwordKind, usize, &[u8], &[&str]); 9] = [
        (3, Wood, 3, &[], &[]),
        (4, Wood, 4, &[3], &["raft"]),
        (1, Wood, 5, &[3, 4], &["raft", "ladder"]),
        (5, Wood, 7, &[1, 3, 4], &["bow", "raft", "ladder"]),
        (2, Wood, 10, &[1, 3, 4, 5], &["bow", "raft", "ladder", "recorder"]),
        (7, Wood, 11, &[1, 2, 3, 4, 5], &["bow", "raft", "ladder", "recorder"]),
        (6, Magic, 12, &[1, 2, 3, 4, 5, 7], &["bow", "raft", "ladder", "recorder"]),
        (8, Magic, 13, &[1, 2, 3, 4, 5, 6, 7], &["bow", "raft", "ladder", "recorder"]),
        (9, Magic, 15, &[1, 2, 3, 4, 5, 6, 7, 8], &["bow", "raft", "ladder", "recorder"]),
    ];
    for (level, sword, containers, levels, items) in want {
        let i = entrance(level);
        let p = &progress[i];
        assert_eq!(
            (p.sword, hearts(&map, steps, i), p.triforces, held(p)),
            (sword, containers, pieces(levels), items.to_vec()),
            "Dungeon {level}'s entrance, step {i}"
        );
    }
    assert_eq!(hearts(&map, steps, steps.len()), 15);
    let rock = index(steps, 0, 0x2c, "cave", "E");
    let raft_heart = index(steps, 0, 0x2f, "cave", "S");
    let ladder_heart = index(steps, 0, 0x5f, "E", "N");
    let tree_heart = index(steps, 0, 0x47, "cave", "E");
    assert_eq!(
        [rock, ladder_heart + 1, raft_heart, tree_heart].map(|i| hearts(&map, steps, i)),
        [7, 9, 10, 15],
        "the heart rock, the ladder heart, the raft heart, 0x47's"
    );
    assert!(!progress[entrance(9)].silver_arrow && progress[steps.len() - 1].ring && progress[steps.len() - 1].silver_arrow);
}

/// The four warps: out of Level 5 (index 0, pieces 1, 3, 4, 5) one play facing right or up lands in 0x74 (Level
/// 3); out of Level 2 one play facing right or up on 0x45's pier (Level 4); out of Level 6 two plays facing down in
/// 0x3c (Level 2); out of Level 8 one play facing down in 0x37 (Level 1). Link comes out of a level facing down: on
/// the first two the family turns E before the play (RECORDER is offered only facing so). Each warp leaves the index
/// on the level it went to (the game's own step, `step_index`).
#[test]
fn the_whirlwind_warps_follow_the_index() {
    let map = map();
    let steps = steps(&map);
    let progress = route_progress(&map, steps);
    let mut index = 0u8;
    let mut warps = Vec::new();
    for (i, step) in steps.iter().enumerate().filter(|(_, s)| s.exit == "recorder") {
        let next = &steps[i + 1];
        let level = whirlwind_level(next.location()).expect("a landing room");
        let triforce = progress[i].triforces;
        let (count, facing) = plays(index, triforce, false, level).expect("the level's piece is held");
        let up = facing & FACING_UP != 0;
        index = (0..count).fold(index, |at, _| step_index(at, triforce, up));
        assert_eq!(index & 7, level - 1, "step {i}: the plays leave the index on the level");
        warps.push((step.room, next.room, level, count, facing));
    }
    assert_eq!(
        warps,
        [
            (0x0b, 0x74, 3, 1, FACING_UP),
            (0x3c, 0x45, 4, 1, FACING_UP),
            (0x22, 0x3c, 2, 2, FACING_DOWN),
            (0x6d, 0x37, 1, 1, FACING_DOWN)
        ]
    );
    assert!(["ow-leever", "ow-octorok"].iter().all(|f| {
        let text = std::fs::read_to_string(common::repo().join(format!("specialists/{f}.yaml"))).unwrap();
        let v: serde_yaml_ng::Value = serde_yaml_ng::from_str(&text).unwrap();
        v["items"]["recorder"].as_bool() == Some(true)
    }));
}

/// Every step's exit is a game.yaml exit of its room (but the triforce, the whirlwind, a maze's sequence and the
/// end) leading into the next step's room on the step's way, and the next step's entry is the side Link comes in by.
#[test]
fn every_step_is_a_door_of_the_game_data() {
    let map = map();
    let steps = steps(&map);
    for clips in [true, false] {
        let way: Vec<(usize, &RouteStep)> = steps.iter().enumerate().filter(|(_, s)| s.on_way(clips)).collect();
        for pair in way.windows(2) {
            let ((i, step), (_, next)) = (pair[0], pair[1]);
            let at = format!("{ROUTE} step {i} {} {} -> {} (clips {clips})", step.location(), step.entry, step.exit);
            let room = map.get(step.location()).unwrap_or_else(|| panic!("{at}: no game.yaml room"));
            match step.exit.as_str() {
                "triforce" => {
                    let mouth = map.get(next.location()).and_then(|r| r.exits.get("cave")).map(|e| e.destination.level);
                    assert_eq!((next.level, next.entry.as_str(), mouth), (0, "cave", Some(step.level)), "{at}");
                }
                "recorder" => assert_eq!(next.entry, "recorder", "{at}"),
                "maze" => assert!(room.exits.contains_key("maze"), "{at}"),
                exit => {
                    let door = room.exits.get(exit).unwrap_or_else(|| panic!("{at}: no exit {exit}"));
                    assert_eq!(door.destination, next.location(), "{at}");
                    assert_eq!(route_entry(step.location(), next.location()), Some(next.entry.as_str()), "{at}");
                }
            }
        }
    }
}

/// d2first's pre-entry state `state`, played on (MOVE N) until Link stands in `room` in play (mode 5).
fn session(state: &str, room: u8) -> Option<Session<Counted>> {
    let core = open_core()?;
    let mut s = Session::new(Counted(core, 0), Arc::new(GameMap::builtin()));
    s.load(&load_named_state(state), &LoadOptions::default(), &mut ChaCha8Rng::seed_from_u64(0)).unwrap();
    for _ in 0..100 {
        if s.state().full_location() == MapLocation::new(0, room, false) && s.state().mode() == 5 {
            return Some(s);
        }
        let north = Action::parse("MOVE N 8").unwrap();
        let action = if s.mask().is_allowed(&north) { north } else { Action::parse("WAIT 4").unwrap() };
        s.act(action).unwrap();
    }
    panic!("{state}: never in 0:{room:#04x}: {}", s.state().full_location());
}

/// 0:0x2d's rock wall (d3first's 30 secret, its own game.yaml entry for the room): d2first plays the room without it
/// (no wall to bomb, no bomb granted on its step crossing S -> W into the heart rock's room, from its own pre-entry
/// state); in d3first's map the wall is shut and a Link with no bomb gets one where the step leaves by the cave.
#[test]
fn the_rock_wall_of_0x2d_is_d3firsts_only() {
    let Some(mut s) = session("0_2ds_0", 0x2d) else { return };
    s.apply_ram(&[(Var::Bombs, 0)]);
    let map = s.game_map().clone();
    let state = s.state().clone();
    assert!(!state.ow_bomb_secret_closed() && !map.requires_bombs(state.full_location()));
    let d2 = &map.routes["d2first"];
    let crossing = d2.iter().position(|st| (st.level, st.room, st.entry.as_str(), st.exit.as_str()) == (0, 0x2d, "S", "W"));
    let d2first = GameCompletion::new(map.clone(), "d2first", crossing).unwrap();
    assert_eq!(needs(&map, &d2first, None, &state)[CheatItem::Bombs], 0, "d2first crosses 0x2d");
    s.set_map(map.route_map(ROUTE));
    let (own, state) = (s.game_map().clone(), s.state().clone());
    assert!(state.ow_bomb_secret_closed());
    let wall = index(steps(&own), 0, 0x2d, "W", "cave");
    let d3first = GameCompletion::new(own.clone(), ROUTE, Some(wall)).unwrap();
    assert_eq!(needs(&own, &d3first, None, &state)[CheatItem::Bombs], 1, "d3first bombs the wall");
}

/// 0:0x3d's secret Armos (the statue at X $90, Y $80; d3first's 30 secret, woken from its E side): only d3first's
/// map gives the room its `cave` exit, so only there the game state offers MOVE W into it from (`$A0`, `$7D`);
/// d2first's crossing (from its own pre-entry state) meets a statue as before.
#[test]
fn only_d3first_walks_into_the_armos_of_0x3d() {
    let Some(mut s) = session("0_3ds_0", 0x3d) else { return };
    s.apply_ram(&[(Var::LinkX, 0xA0), (Var::LinkY, 0x7D), (Var::LinkDirection, Direction::W.value()), (Var::LinkGridOffset, 0)]);
    let push = Action::new(ActionKind::Move, Direction::W, 16).unwrap();
    assert!(!s.state().is_secret_armos_direction(Direction::W), "d2first: no stairs under the statue");
    assert!(!s.mask().is_allowed(&push), "d2first: {}", s.mask());
    let map = s.game_map().clone();
    s.set_map(map.route_map(ROUTE));
    assert!(s.state().is_secret_armos_direction(Direction::W), "the statue is the room's secret Armos");
    assert!(s.mask().is_allowed(&push), "d3first: {}", s.mask());
    let stairs = index(steps(&map), 0, 0x3d, "E", "cave");
    let mut d3first = GameCompletion::new(s.game_map().clone(), ROUTE, Some(stairs)).unwrap();
    let objective = d3first.objectives(None, s.state());
    assert_eq!(objective.kind, ObjectiveKind::Cave, "d3first heads for the Armos's stairs");
}
