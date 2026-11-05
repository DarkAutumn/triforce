//! `GameCompletion` objectives on real rooms with poked RAM: key collection (observed and seeded from room
//! flags), treasure/fight precedence, caves, and exit targets agreeing with exit correctness.
#![cfg(feature = "zcore")]

mod common;

use common::{Counted, load_named_state, open_core};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;
use zelda::enums::SwordKind;
use zelda::game::GameView;
use zelda::objectives::{
    CAVE_TREASURE_TILE, GameCompletion, Objective, ObjectiveKind, exit_is_correct_at, tiles_in_span,
};
use zelda::room::Target;
use zelda::room_entry::ROOM_FLAGS_UW;
use zelda::room_state::room_flags;
use zelda::tables::{self, RAM_SIZE};
use zelda::{Direction, GameMap, LoadOptions, MapLocation, Session, Var};

const ROUTE: &str = "milestone1";

fn loc(level: u8, room: u8) -> MapLocation {
    MapLocation::new(level, room, false)
}

/// RAM images of named states, after load (Link controllable).
fn images(names: &[&str]) -> Option<Vec<Box<[u8; RAM_SIZE]>>> {
    let nes = open_core()?;
    let mut s = Session::new(Counted(nes, 0), Arc::new(GameMap::builtin()));
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    Some(
        names
            .iter()
            .map(|name| {
                s.load(&load_named_state(name), &LoadOptions::default(), &mut rng)
                    .unwrap();
                Box::new(*s.state().ram())
            })
            .collect(),
    )
}

fn view(map: &Arc<GameMap>, ram: &[u8; RAM_SIZE], frame: u64) -> GameView {
    GameView::new(ram, frame, false, map.clone())
}

fn set(ram: &mut [u8; RAM_SIZE], var: Var, value: u8) {
    ram[var.addr() as usize] = value;
}

/// No monsters (or anything else) in object slots 1-11.
fn clear_objects(ram: &mut [u8; RAM_SIZE]) {
    ram[tables::OBJ_ID.0 + 1..tables::OBJ_ID.0 + 12].fill(0);
}

fn show_treasure(ram: &mut [u8; RAM_SIZE], x: u8, y: u8) {
    set(ram, Var::TreasureFlag, 0);
    set(ram, Var::TreasureX, x);
    set(ram, Var::TreasureY, y);
}

fn completion(map: &Arc<GameMap>) -> GameCompletion {
    GameCompletion::new(map.clone(), ROUTE, None).unwrap()
}

/// Every exit target pays out as a correct exit into one of the next rooms (PBRS and exit rewards agree).
fn assert_targets_are_correct_exits(map: &GameMap, state: &GameView, objective: &Objective) {
    assert!(!objective.next_rooms.is_empty());
    for target in &objective.pbrs_targets {
        let Target::Tile(tile) = *target else {
            continue;
        };
        assert!(
            objective.next_rooms.iter().any(|&next| exit_is_correct_at(
                map,
                state.full_location(),
                tile,
                objective,
                next
            )),
            "target {tile:?} in {} is not a correct exit into {:?}",
            state.full_location(),
            objective.next_rooms
        );
    }
}

#[test]
fn unknown_route_names_the_known_ones() {
    let map = Arc::new(GameMap::builtin());
    let err = GameCompletion::new(map, "nope", None).unwrap_err();
    assert!(err.starts_with("Unknown route 'nope'"), "{err}");
    assert!(
        err.contains("\"milestone1\"") && err.contains("\"milestone1-fairy\""),
        "{err}"
    );
}

#[test]
fn exit_spans_bound_targets_and_correct_exits() {
    let tiles = [(0, 9), (0, 10), (0, 11), (0, 12)];
    assert_eq!(
        tiles_in_span(&tiles, Some((10, 11)), Direction::W),
        [(0, 10), (0, 11)]
    );
    assert_eq!(tiles_in_span(&tiles, None, Direction::W), tiles);
    let top = [(14, 0), (15, 0), (16, 0), (17, 0)];
    assert_eq!(
        tiles_in_span(&top, Some((15, 16)), Direction::N),
        [(15, 0), (16, 0)]
    );
    assert_eq!(tiles_in_span(&top, Some((0, 3)), Direction::N), []);

    let map = GameMap::builtin();
    let from = loc(0, 0x5b);
    let to = loc(0, 0x5c);
    let objective = Objective {
        kind: ObjectiveKind::Move,
        targets: Default::default(),
        next_rooms: [to].into_iter().collect(),
        pbrs_targets: Default::default(),
    };
    assert!(
        !exit_is_correct_at(&map, from, (31, 10), &objective, loc(0, 0x4b)),
        "not a next room"
    );
    // Entering a cave (no cardinal direction) is correct whenever the cave is a next room.
    let cave = MapLocation::new(0, 0x5b, true);
    let into_cave = Objective {
        next_rooms: [cave].into_iter().collect(),
        ..objective.clone()
    };
    assert!(exit_is_correct_at(&map, from, (0, 0), &into_cave, cave));
}

#[test]
fn dungeon_key_rooms_fight_then_treasure_then_leave_and_stay_collected() {
    let Some(images) = images(&["1_72e_0", "1_73s_0"]) else {
        return;
    };
    let map = Arc::new(GameMap::builtin());
    let (key_room, start) = (&images[0], &images[1]);
    let key_view = view(&map, key_room, 0);
    assert_eq!(key_view.full_location(), loc(1, 0x72));
    assert_eq!(
        map.get(loc(1, 0x72)).unwrap().treasure.as_deref(),
        Some("key")
    );

    // From the D1 entrance without keys, a key room comes next.
    let mut no_keys = start.clone();
    set(&mut no_keys, Var::Keys, 0);
    let entrance = view(&map, &no_keys, 0);
    let first = completion(&map).objectives(None, &entrance);
    assert_eq!(first.kind, ObjectiveKind::Move);
    assert!(
        first
            .next_rooms
            .iter()
            .all(|r| [loc(1, 0x72), loc(1, 0x74)].contains(r)),
        "{:?}",
        first.next_rooms
    );
    assert_targets_are_correct_exits(&map, &entrance, &first);

    // In the key room: fight while monsters are up (the key drops when they're dead).
    let mut gc = completion(&map);
    let fight = gc.objectives(None, &key_view);
    assert!(!key_view.enemies.is_empty());
    assert_eq!(fight.kind, ObjectiveKind::Fight);
    assert!(
        fight.next_rooms.is_empty(),
        "no exit is correct before the key"
    );
    let enemy_tiles: Vec<Target> = if key_view.has_active_enemies() {
        key_view
            .active_enemies()
            .flat_map(|e| e.link_overlap_tiles())
            .map(Target::Tile)
            .collect()
    } else {
        // Still spawning: aim at where they are.
        key_view
            .enemies
            .iter()
            .flat_map(|e| e.link_overlap_tiles())
            .map(Target::Tile)
            .collect()
    };
    assert!(!enemy_tiles.is_empty() && enemy_tiles.iter().all(|t| fight.targets.contains(t)));
    // The key on the floor: go get it.
    let mut dropped = key_room.clone();
    clear_objects(&mut dropped);
    show_treasure(&mut dropped, 0x78, 0x98);
    let dropped = view(&map, &dropped, 10);
    let treasure = gc.objectives(Some(&key_view), &dropped);
    assert_eq!(treasure.kind, ObjectiveKind::Treasure);
    let tiles: Vec<Target> = dropped
        .treasure()
        .unwrap()
        .link_overlap_tiles()
        .into_iter()
        .map(Target::Tile)
        .collect();
    assert!(tiles.iter().all(|t| treasure.targets.contains(t)));
    assert!(treasure.next_rooms.is_empty());
    // Picked up: the room is done, leave toward the route.
    let mut taken = *dropped.ram();
    set(&mut taken, Var::TreasureFlag, 0xFF);
    set(&mut taken, Var::Keys, 1);
    let taken = view(&map, &taken, 20);
    let leave = gc.objectives(Some(&dropped), &taken);
    assert_eq!(leave.kind, ObjectiveKind::Move);
    assert!(!leave.next_rooms.contains(&loc(1, 0x72)));
    assert_targets_are_correct_exits(&map, &taken, &leave);
    // Back at the entrance with the key spent: the collected room is never the next room again.
    let again = gc.objectives(Some(&taken), &entrance);
    assert!(
        !again.next_rooms.contains(&loc(1, 0x72)),
        "{:?}",
        again.next_rooms
    );
    assert!(!again.next_rooms.is_empty());

    // The same knowledge comes from the persistent room flags of a fresh episode.
    let mut flagged = no_keys.clone();
    flagged[(ROOM_FLAGS_UW + 0x72) as usize] |= 0x10;
    assert_eq!(
        room_flags(&flagged, 0x72) & 0x10,
        0x10,
        "level 1 flags live at ROOM_FLAGS_UW"
    );
    let seeded = completion(&map).objectives(None, &view(&map, &flagged, 0));
    assert_eq!(seeded.next_rooms, again.next_rooms);
}

#[test]
fn map_and_compass_treasure_is_not_an_objective() {
    let Some(images) = images(&["1_72e_0"]) else {
        return;
    };
    let mut map = GameMap::builtin();
    map.get_mut(loc(1, 0x72)).unwrap().treasure = Some("compass".into());
    let map = Arc::new(map);
    let mut ram = images[0].clone();
    clear_objects(&mut ram);
    show_treasure(&mut ram, 0x78, 0x98);
    let state = view(&map, &ram, 0);
    let objective = completion(&map).objectives(None, &state);
    assert_eq!(objective.kind, ObjectiveKind::Move);
    assert_targets_are_correct_exits(&map, &state, &objective);
}

#[test]
fn sword_cave_wants_the_sword_then_the_exit() {
    let Some(images) = images(&["0_77cave_0", "start"]) else {
        return;
    };
    let map = Arc::new(GameMap::builtin());
    let (cave, start) = (&images[0], &images[1]);
    let st = view(&map, cave, 0);
    assert!(st.in_cave() && st.sword() == SwordKind::None);

    // The sword on its stand.
    let mut on_stand = cave.clone();
    show_treasure(&mut on_stand, 0x78, 0x98);
    let on_stand = view(&map, &on_stand, 0);
    let objective = completion(&map).objectives(None, &on_stand);
    assert_eq!(objective.kind, ObjectiveKind::Treasure);
    let tiles: Vec<Target> = on_stand
        .treasure()
        .unwrap()
        .link_overlap_tiles()
        .into_iter()
        .map(Target::Tile)
        .collect();
    assert!(tiles.iter().all(|t| objective.targets.contains(t)));
    // Not on screen (yet): the fixed cave treasure tile.
    let mut hidden = cave.clone();
    set(&mut hidden, Var::TreasureFlag, 0xFF);
    let objective = completion(&map).objectives(None, &view(&map, &hidden, 0));
    assert_eq!(objective.kind, ObjectiveKind::Treasure);
    assert!(
        objective
            .targets
            .contains(&Target::Tile(CAVE_TREASURE_TILE))
    );
    assert!(objective.next_rooms.is_empty());

    // With the sword: walk out south, back to the overworld room.
    let mut armed = cave.clone();
    set(&mut armed, Var::Sword, 1);
    let objective = completion(&map).objectives(None, &view(&map, &armed, 0));
    assert_eq!(objective.kind, ObjectiveKind::Move);
    assert_eq!(
        objective.next_rooms.iter().copied().collect::<Vec<_>>(),
        [loc(0, 0x77)]
    );

    // Outside: the cave is the objective without a sword, never with one.
    let outside = view(&map, start, 0);
    let objective = completion(&map).objectives(None, &outside);
    assert_eq!(objective.kind, ObjectiveKind::Cave);
    assert_eq!(
        objective.next_rooms.iter().copied().collect::<Vec<_>>(),
        [MapLocation::new(0, 0x77, true)]
    );
    let mut armed = start.clone();
    set(&mut armed, Var::Sword, 1);
    let armed = view(&map, &armed, 0);
    let objective = completion(&map).objectives(None, &armed);
    assert_eq!(objective.kind, ObjectiveKind::Move);
    assert!(!objective.next_rooms.iter().any(|r| r.in_cave));
    assert_targets_are_correct_exits(&map, &armed, &objective);
}
