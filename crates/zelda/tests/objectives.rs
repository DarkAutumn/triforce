//! `GameCompletion` objectives on real rooms with poked RAM: key collection (observed and seeded from room
//! flags), treasure/fight precedence, caves, exit targets agreeing with exit correctness, the route's step order, the
//! bow detour's push block, stairs and cellar played in the emulator, and Dungeon 3's treasure rules and bomb wall.
#![cfg(feature = "zcore")]

mod common;

use common::{Counted, load_named_state, open_core};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;
use zelda::enums::SwordKind;
use zelda::game::GameView;
use zelda::objectives::{
    CAVE_TREASURE_TILE, GameCompletion, Objective, ObjectiveKind, bomb_spot, exit_is_correct_at, start_route,
    tiles_in_span,
};
use zelda::room::Target;
use zelda::room_entry::{ROOM_FLAGS_OW, ROOM_FLAGS_UW};
use zelda::room_state::room_flags;
use zelda::tables::{self, RAM_SIZE};
use zelda::{Action, ActionKind, Direction, GameMap, LoadOptions, MapLocation, Session, Var};

const ROUTE: &str = "d2first";

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
    assert!(err.contains("\"d2first\""), "{err}");
}

/// F048: a step index past the route's end is an error, not a later panic.
#[test]
fn an_out_of_range_step_is_an_error() {
    let map = Arc::new(GameMap::builtin());
    let n = map.routes[ROUTE].len();
    let err = GameCompletion::new(map.clone(), ROUTE, Some(n)).unwrap_err();
    assert!(err.contains("out of range"), "{err}");
    assert!(GameCompletion::new(map, ROUTE, Some(n - 1)).is_ok());
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
        retreats: Default::default(),
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

/// Link holds only the Dungeon 2 triforce, with `hearts` heart containers (full).
fn after_d2(ram: &mut [u8; RAM_SIZE], hearts: u8) {
    set(ram, Var::Triforce, 0b10);
    set(ram, Var::HeartsAndContainers, ((hearts - 1) << 4) | (hearts - 1));
}

/// The route's triforces in route order: Dungeon 2's first, though Dungeon 1's is closer.
#[test]
fn the_dungeon_2_triforce_comes_first() {
    let Some(images) = images(&["0_3cs_0"]) else {
        return;
    };
    let map = Arc::new(GameMap::builtin());
    let mut ram = images[0].clone();
    set(&mut ram, Var::Triforce, 0);
    let state = view(&map, &ram, 0);
    let objective = completion(&map).objectives(None, &state);
    assert_eq!(objective.kind, ObjectiveKind::Cave);
    assert_eq!(
        objective.next_rooms.iter().copied().collect::<Vec<_>>(),
        [loc(2, 0x7d)]
    );
}

/// After Dungeon 2: the bomb_secret rock wall of 0x2c is the cave entrance until the heart container is taken; the
/// take-any cave points at the heart (right ware); then the white sword (5 hearts) leads north, never back into 0x2c's
/// cave.
#[test]
fn heart_cave_then_white_sword() {
    let Some(images) = images(&["0_2ce_0"]) else {
        return;
    };
    let map = Arc::new(GameMap::builtin());
    let cave = MapLocation::new(0, 0x2c, true);
    let mut ram = images[0].clone();
    clear_objects(&mut ram);
    after_d2(&mut ram, 4);
    let state = view(&map, &ram, 0);
    assert_eq!(state.room().cave_tile, None, "the cave is hidden behind the rock wall");
    let mut gc = completion(&map);
    let objective = gc.objectives(None, &state);
    assert_eq!(objective.kind, ObjectiveKind::Cave);
    assert_eq!(objective.next_rooms.iter().copied().collect::<Vec<_>>(), [cave]);
    let secret: Vec<Target> = [(18, 12), (19, 12), (18, 13), (19, 13)]
        .into_iter()
        .map(Target::Tile)
        .collect();
    assert!(secret.iter().all(|t| objective.targets.contains(t)), "{:?}", objective.targets);

    // Inside the cave (game mode $0B): the heart container on the right, not the middle ware.
    let mut inside = ram.clone();
    inside[0x12] = 0x0B;
    set(&mut inside, Var::TreasureFlag, 0xFF);
    let inside = view(&map, &inside, 0);
    assert!(inside.in_cave());
    let objective = gc.objectives(Some(&state), &inside);
    assert_eq!(objective.kind, ObjectiveKind::Treasure);
    assert_eq!(
        objective.targets.iter().copied().collect::<Vec<_>>(),
        [Target::Tile((19, 11))]
    );

    // Heart taken (item flag set, 5 containers): the white sword is next, north through 0x1c.
    let mut taken = ram.clone();
    after_d2(&mut taken, 5);
    taken[ROOM_FLAGS_OW as usize + 0x2c] |= 0x10;
    let taken = view(&map, &taken, 0);
    let objective = gc.objectives(Some(&inside), &taken);
    assert_eq!(objective.kind, ObjectiveKind::Move);
    assert_eq!(
        objective.next_rooms.iter().copied().collect::<Vec<_>>(),
        [loc(0, 0x1c)]
    );
    assert_targets_are_correct_exits(&map, &taken, &objective);
}

/// Leaving 0x2c (not into its cave) before the heart container, or taking the potion instead, misses the heart: no
/// exit is correct any more (the chain ends the run on `missed`).
#[test]
fn missed_heart_has_no_correct_exit() {
    let Some(images) = images(&["0_2ce_0"]) else {
        return;
    };
    let map = Arc::new(GameMap::builtin());
    let mut ram = images[0].clone();
    clear_objects(&mut ram);
    after_d2(&mut ram, 4);
    let at_2c = view(&map, &ram, 0);
    let mut north = ram.clone();
    north[0xEB] = 0x1c;
    let north = view(&map, &north, 0);
    assert_eq!(north.full_location(), loc(0, 0x1c));

    let mut gc = completion(&map);
    gc.objectives(None, &at_2c);
    assert_eq!(gc.missed(), None);
    let objective = gc.objectives(Some(&at_2c), &north);
    assert!(gc.missed().is_some_and(|m| m.contains("without the heart container")));
    assert!(objective.next_rooms.is_empty());

    // The potion: the cave's item flag is set but the heart containers stayed at 4.
    let mut inside = ram.clone();
    inside[0x12] = 0x0B;
    let inside = view(&map, &inside, 0);
    let mut potion = inside.ram().clone();
    potion[ROOM_FLAGS_OW as usize + 0x2c] |= 0x10;
    let potion = view(&map, &potion, 0);
    let mut gc = completion(&map);
    gc.objectives(None, &inside);
    let objective = gc.objectives(Some(&inside), &potion);
    assert!(gc.missed().is_some_and(|m| m.contains("not the heart container")));
    assert!(objective.next_rooms.is_empty());
}

/// `ram` as if Link were in overworld/level room `room` (the room byte poked).
fn in_room(ram: &[u8; RAM_SIZE], room: u8) -> Box<[u8; RAM_SIZE]> {
    let mut ram = Box::new(*ram);
    ram[0xEB] = room;
    ram
}

/// Dungeon 1 follows the route's steps through the entrance 0x73 (Lee, Nov 8): W to the key room 0x72, back in from
/// W then E to the key room 0x74, back in from E then N, though Link carries Dungeon 2's keys and the shortest way
/// to the triforce is N from the start.
#[test]
fn dungeon_1_entrance_follows_the_route_steps() {
    let Some(images) = images(&["1_73s_0", "1_73w_0", "1_73e_0"]) else {
        return;
    };
    let map = Arc::new(GameMap::builtin());
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let from_s = view(&map, &images[0], 0);
    assert!(from_s.keys() > 0 && from_s.has_triforce(2), "Link brings Dungeon 2's keys and triforce");
    // Off the route (no step), the key-aware shortest way: N.
    let shortest = completion(&map).objectives(None, &from_s);
    assert_eq!(shortest.next_rooms.iter().copied().collect::<Vec<_>>(), [loc(1, 0x63)]);
    // From S (a training reset): the first visit's step, W.
    let (route, step) = start_route(&map, loc(1, 0x73), Some("S"), &mut rng, None, &from_s);
    assert_eq!(route, ROUTE);
    let mut gc = GameCompletion::new(map.clone(), &route, step).unwrap();
    let objective = gc.objectives(None, &from_s);
    assert_eq!(objective.next_rooms.iter().copied().collect::<Vec<_>>(), [loc(1, 0x72)]);
    assert_targets_are_correct_exits(&map, &from_s, &objective);
    // Back from 0x72 (entered from W): E.  Back from 0x74 (entered from E): N.
    for (image, came_from, next) in [(&images[1], 0x72, 0x74), (&images[2], 0x74, 0x63)] {
        let state = view(&map, image, 0);
        let prev = in_room(image, came_from);
        let prev = view(&map, &prev, 0);
        let mut gc = completion(&map);
        let objective = gc.objectives(Some(&prev), &state);
        assert_eq!(
            objective.next_rooms.iter().copied().collect::<Vec<_>>(),
            [loc(1, next)],
            "0x73 from 0x{came_from:02x}"
        );
        assert_targets_are_correct_exits(&map, &state, &objective);
        let (_, start) = start_route(&map, loc(1, 0x73), state_entry_of(came_from), &mut rng, None, &state);
        assert_eq!(start, gc.step(), "a reset in the same room plays the same step");
    }
}

/// The entry of 1:0x73 from its key-room neighbour.
fn state_entry_of(neighbour: u8) -> Option<&'static str> {
    Some(if neighbour == 0x72 { "W" } else { "E" })
}

/// 0x0a before the white sword: the cave is the objective and S (back to 0x1a) a retreat.  In 0x1a from N without
/// the white sword Link is off the route (the step leaving W needs it), so the objective sends him back N; with it,
/// W.
#[test]
fn lynel_room_retreat_and_return() {
    let Some(images) = images(&["0_0as_0"]) else {
        return;
    };
    let map = Arc::new(GameMap::builtin());
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let at_0a = view(&map, &images[0], 0);
    assert_eq!(at_0a.sword(), SwordKind::Wood);
    let (route, step) = start_route(&map, loc(0, 0x0a), Some("S"), &mut rng, None, &at_0a);
    let mut gc = GameCompletion::new(map.clone(), &route, step).unwrap();
    let objective = gc.objectives(None, &at_0a);
    assert_eq!(objective.kind, ObjectiveKind::Cave);
    assert_eq!(
        objective.next_rooms.iter().copied().collect::<Vec<_>>(),
        [MapLocation::new(0, 0x0a, true)]
    );
    assert_eq!(objective.retreats.iter().copied().collect::<Vec<_>>(), [loc(0, 0x1a)]);

    let back = in_room(&images[0], 0x1a);
    let back = view(&map, &back, 0);
    let objective = gc.objectives(Some(&at_0a), &back);
    assert_eq!(gc.step(), None);
    assert_eq!(objective.next_rooms.iter().copied().collect::<Vec<_>>(), [loc(0, 0x0a)]);
    assert!(objective.retreats.is_empty());

    let mut armed = in_room(&images[0], 0x1a);
    set(&mut armed, Var::Sword, 2);
    let armed = view(&map, &armed, 0);
    let objective = completion(&map).objectives(Some(&at_0a), &armed);
    assert_eq!(objective.next_rooms.iter().copied().collect::<Vec<_>>(), [loc(0, 0x19)]);
}

/// Off the route, the objective prefers only the route exits of steps Link plays with his progress.  Without the
/// sword (he left 0x77 N instead of taking it: seed 1 of the sampled headless runs, Nov 11) every step but the
/// sword cave's needs it, so the way leads back to the cave rather than along the route; it used to lead him along
/// the route to 0x4d, whose route exit W and 0x4c's route exit E (taken there after Dungeon 2) then sent him back
/// and forth between the two, every crossing a correct exit, for 3 million frames.  With the sword, 0x4d entered
/// from W (off the route before Dungeon 2) still sends him W, onto the route.
#[test]
fn off_route_objectives_follow_only_the_route_steps_with_links_progress() {
    let Some(images) = images(&["0_4ds_0"]) else {
        return;
    };
    let map = Arc::new(GameMap::builtin());
    let room = |id: u8, sword: u8| {
        let mut ram = in_room(&images[0], id);
        clear_objects(&mut ram);
        set(&mut ram, Var::Sword, sword);
        ram
    };
    let next = |from: u8, to: u8, sword: u8| {
        let (prev, here) = (room(from, sword), room(to, sword));
        let objective = completion(&map).objectives(Some(&view(&map, &prev, 0)), &view(&map, &here, 0));
        objective.next_rooms.iter().copied().collect::<Vec<_>>()
    };
    assert_eq!(view(&map, &images[0], 0).sword(), SwordKind::Wood);
    assert_eq!(next(0x77, 0x67, 0), [loc(0, 0x77)], "0x67 from S without the sword: back to the cave");
    let (at_4c, at_4d) = (next(0x4d, 0x4c, 0), next(0x4c, 0x4d, 0));
    assert!(!(at_4c.contains(&loc(0, 0x4d)) && at_4d.contains(&loc(0, 0x4c))), "{at_4c:?} {at_4d:?}");
    assert!(!at_4d.contains(&loc(0, 0x4c)), "0x4d from W without the sword: {at_4d:?}");
    assert_eq!(next(0x4c, 0x4d, 1), [loc(0, 0x4c)], "0x4d from W with the sword: onto the route");
}

fn act<N: zelda::Nes>(s: &mut Session<N>, text: &str) -> zelda::Change {
    s.act(Action::parse(text).unwrap()).unwrap()
}

fn tiles(targets: &[(i32, i32)]) -> std::collections::BTreeSet<Target> {
    targets.iter().map(|&t| Target::Tile(t)).collect()
}

/// The bow detour in the emulator (Lee, Nov 9).  In 1:0x22 from E the objective points at the west block's push
/// spots while it is pending (pushing N Link stands half inside its lower row, pushing S above it), then at the
/// stairs; down them the bow cellar (1:0x7f, an in_cave room under its own id) is a Treasure objective with MOVE N/S
/// legal at the top of its left ladder; with the bow the way out is up that ladder (its N exit, the top row) to 0x22,
/// where the step out of the cellar leaves E.
#[test]
fn bow_detour_block_stairs_cellar_and_back() {
    let Some(nes) = open_core() else { return };
    let map = Arc::new(GameMap::builtin());
    let mut s = Session::new(Counted(nes, 0), map.clone());
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    s.load(&load_named_state("1_22e_0"), &LoadOptions::default(), &mut rng)
        .unwrap();
    for _ in 0..30 {
        if act(&mut s, "WAIT 8").interrupt.is_none() {
            break;
        }
    }
    let entered = s.state().clone();
    assert!(entered.push_block_pending() && !entered.bow());
    let (route, step) = start_route(&map, loc(1, 0x22), Some("E"), &mut rng, None, &entered);
    let step = step.expect("0x22 from E is a route step");
    assert_eq!(map.routes[&route][step].exit, "cave");
    let mut gc = GameCompletion::new(map.clone(), &route, Some(step)).unwrap();
    let cellar = MapLocation::new(1, 0x7f, true);
    let objective = gc.objectives(None, &entered);
    assert_eq!(objective.next_rooms, [cellar].into_iter().collect());
    assert_eq!(objective.pbrs_targets, tiles(&[(12, 11), (12, 8)]));

    // Pushed S from above, the block opens the way: the stairs (top-left tile) are the target.
    let above = s
        .apply_ram(&[(Var::LinkX, 96), (Var::LinkY, 125), (Var::DoorwayDir, 0)])
        .clone();
    act(&mut s, "MOVE S 32");
    act(&mut s, "WAIT 16");
    assert!(!s.state().push_block_pending());
    let pushed = s.state().clone();
    let objective = gc.objectives(Some(&above), &pushed);
    assert_eq!(objective.pbrs_targets, tiles(&[(16, 10)]));

    // Down the stairs: the cellar is controllable once laid out, Link at the top of its left ladder.
    let beside = s.apply_ram(&[(Var::LinkX, 112), (Var::LinkY, 141)]).clone();
    act(&mut s, "MOVE E 16");
    let below = s.state().clone();
    assert_eq!(below.full_location(), cellar);
    assert!(below.in_cellar() && !below.in_cave() && below.peek(0x11) == 1);
    assert_eq!((below.link.position.x, below.link.position.y), (48, 93));
    let objective = gc.objectives(Some(&beside), &below);
    assert_eq!(gc.step(), Some(step + 1));
    assert_eq!(objective.kind, ObjectiveKind::Treasure);
    assert!(objective.next_rooms.is_empty());
    let bow = below.treasure().expect("the bow");
    assert!(bow.link_overlap_tiles().into_iter().all(|t| objective.targets.contains(&Target::Tile(t))));
    assert_eq!(s.mask().directions(ActionKind::Move), [Direction::N, Direction::S]);
    assert_eq!(below.room().exits.get(Direction::N), Some(&[(6, 0)][..]));

    // On the bow's walkway: taken, the ladder out is the target.
    s.apply_ram(&[(Var::LinkX, 128), (Var::LinkY, 141)]);
    act(&mut s, "WAIT 4");
    let armed = s.state().clone();
    assert!(armed.bow());
    let objective = gc.objectives(Some(&below), &armed);
    assert_eq!(objective.kind, ObjectiveKind::Move);
    assert_eq!(objective.next_rooms, [loc(1, 0x22)].into_iter().collect());
    assert_eq!(objective.pbrs_targets, tiles(&[(6, 0)]));
    assert!(exit_is_correct_at(&map, cellar, (6, 0), &objective, loc(1, 0x22)));

    // Up the ladder: back in 0x22 below the reset block, the step out of the cellar leaves E.
    s.apply_ram(&[(Var::LinkX, 48), (Var::LinkY, 93)]);
    let top = s.state().clone();
    for _ in 0..10 {
        if act(&mut s, "MOVE N 16").changed_location() {
            break;
        }
    }
    let out = s.state().clone();
    assert_eq!(out.full_location(), loc(1, 0x22));
    assert_eq!((out.link.position.x, out.link.position.y), (96, 173));
    assert!(out.push_block_pending());
    let objective = gc.objectives(Some(&top), &out);
    assert_eq!(gc.step(), Some(step + 2));
    assert_eq!(objective.next_rooms, [loc(1, 0x23)].into_iter().collect());
}

/// The GameCompletion of the route step entering 3:`room` from `entry` (as one-room training picks it).
fn d3_step(map: &Arc<GameMap>, state: &GameView, room: u8, entry: &str) -> GameCompletion {
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let (route, step) = start_route(map, loc(3, room), Some(entry), &mut rng, None, state);
    let step = step.unwrap_or_else(|| panic!("3:{room:#04x} from {entry} is a route step"));
    assert_eq!(map.routes[&route][step].location(), loc(3, room));
    GameCompletion::new(map.clone(), &route, Some(step)).unwrap()
}

/// Dungeon 3's route-step treasure rules (Lee, Nov 9).  0x69 from N (`treasure: skip`) heads for the stairs down to
/// the raft cellar past its Darknuts, where the default rule (off the route) fights them for the bombs they guard;
/// 0x5b from S (`skip`) leaves W at once; 0x5b from W (`if_bombs_below: 4`) fights for its bombs and takes them only
/// while Link holds fewer than 4, else leaves N.
#[test]
fn d3_treasure_rules_skip_and_take_below_four_bombs() {
    let Some(images) = images(&["3_69n_0", "3_5bs_0", "3_5bw_0"]) else {
        return;
    };
    let map = Arc::new(GameMap::builtin());
    let darknuts = view(&map, &images[0], 0);
    assert!(!darknuts.enemies.is_empty() && darknuts.treasure().is_none());
    let objective = d3_step(&map, &darknuts, 0x69, "N").objectives(None, &darknuts);
    assert_eq!(objective.kind, ObjectiveKind::Move);
    assert_eq!(objective.next_rooms, [MapLocation::new(3, 0x0f, true)].into_iter().collect());
    assert_eq!(objective.pbrs_targets, tiles(&[(26, 10)]));
    assert_eq!(completion(&map).objectives(None, &darknuts).kind, ObjectiveKind::Fight);

    let central = view(&map, &images[1], 0);
    assert!(!central.enemies.is_empty());
    let objective = d3_step(&map, &central, 0x5b, "S").objectives(None, &central);
    assert_eq!(objective.kind, ObjectiveKind::Move);
    assert_eq!(objective.next_rooms, [loc(3, 0x5a)].into_iter().collect());

    for (bombs, kind) in [
        (0, ObjectiveKind::Fight),
        (3, ObjectiveKind::Fight),
        (4, ObjectiveKind::Move),
        (8, ObjectiveKind::Move),
    ] {
        let mut ram = images[2].clone();
        set(&mut ram, Var::Bombs, bombs);
        let state = view(&map, &ram, 0);
        assert!(!state.enemies.is_empty());
        let objective = d3_step(&map, &state, 0x5b, "W").objectives(None, &state);
        assert_eq!(objective.kind, kind, "{bombs} bombs");
        let next = if kind == ObjectiveKind::Move { vec![loc(3, 0x4b)] } else { vec![] };
        assert_eq!(objective.next_rooms, next.into_iter().collect(), "{bombs} bombs");
    }
    // The Darknuts dead and their bombs on the floor: taken below 4 bombs, left from 4 on.
    for (bombs, kind) in [(3, ObjectiveKind::Treasure), (4, ObjectiveKind::Move)] {
        let mut ram = images[2].clone();
        clear_objects(&mut ram);
        show_treasure(&mut ram, 120, 141);
        set(&mut ram, Var::Bombs, bombs);
        let state = view(&map, &ram, 0);
        let objective = d3_step(&map, &state, 0x5b, "W").objectives(None, &state);
        assert_eq!(objective.kind, kind, "{bombs} bombs, treasure showing");
    }
}

/// Dungeon 3's 0x4c from W leaves E through an intact bomb wall into Manhandla's room: the objective's target is
/// the wall's bomb spot (16 px short of its hotspot), a bomb placed from it facing E opens the wall, and the target
/// becomes the open doorway, a correct exit into 0x4d that Link walks through.
#[test]
fn d3_bomb_wall_exit_targets_its_bomb_spot_then_the_doorway() {
    let Some(nes) = open_core() else { return };
    let map = Arc::new(GameMap::builtin());
    let mut s = Session::new(Counted(nes, 0), map.clone());
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    s.load(&load_named_state("3_4cw_0"), &LoadOptions::default(), &mut rng)
        .unwrap();
    for _ in 0..30 {
        if act(&mut s, "WAIT 8").interrupt.is_none() {
            break;
        }
    }
    let entered = s.state().clone();
    let walls: Vec<Direction> = entered.bomb_walls().iter().map(|w| w.wall_direction).collect();
    assert_eq!(walls, [Direction::E]);
    let mut gc = d3_step(&map, &entered, 0x4c, "W");
    let boss = loc(3, 0x4d);
    let objective = gc.objectives(None, &entered);
    assert_eq!(objective.kind, ObjectiveKind::Move);
    assert_eq!(objective.next_rooms, [boss].into_iter().collect());
    assert_eq!(objective.pbrs_targets, tiles(&[(24, 10)]));
    assert_eq!(bomb_spot(Direction::E), (24, 10));

    // On the bomb spot (16 px short of the hotspot (0xD0, 0x8D)) facing E with a bomb: the wall opens.
    s.apply_ram(&[
        (Var::LinkX, 0xC0),
        (Var::LinkY, 0x8D),
        (Var::LinkDirection, 1),
        (Var::Bombs, 1),
        (Var::HeartsAndContainers, 0x55),
        (Var::PartialHearts, 0xFF),
    ]);
    assert_eq!(s.state().link.tile(), (24, 10));
    act(&mut s, "BOMB E");
    for _ in 0..10 {
        if s.state().bomb_walls().is_empty() {
            break;
        }
        act(&mut s, "WAIT 16");
    }
    let open = s.state().clone();
    assert!(open.bomb_walls().is_empty(), "the bomb opened the E wall");
    let objective = gc.objectives(Some(&entered), &open);
    assert_eq!(objective.next_rooms, [boss].into_iter().collect());
    assert_eq!(objective.pbrs_targets, tiles(&[(31, 10)]));
    assert!(exit_is_correct_at(&map, loc(3, 0x4c), (31, 10), &objective, boss));
    // Walk through (back on the doorway row: the Zols may have knocked him off it while the bomb ticked).
    s.apply_ram(&[(Var::LinkX, 0xC0), (Var::LinkY, 0x8D)]);
    for _ in 0..10 {
        if act(&mut s, "MOVE E 16").changed_location() {
            break;
        }
    }
    assert_eq!(s.state().full_location(), boss);
}

/// F189: a completion without a route step drops every finished waypoint before its first objective (the 0x2c heart
/// and the bow are done in 1_23w_0): the first objective already equals the one with the state's own step.
#[test]
fn a_stepless_completion_drops_every_finished_waypoint_at_once() {
    let Some(images) = images(&["1_23w_0"]) else { return };
    let map = Arc::new(GameMap::builtin());
    let state = view(&map, &images[0], 0);
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let (route, step) = start_route(&map, loc(1, 0x23), Some("W"), &mut rng, None, &state);
    assert!(step.is_some());
    let with_step = GameCompletion::new(map.clone(), &route, step).unwrap().objectives(None, &state);
    let stepless = GameCompletion::new(map.clone(), &route, None).unwrap().objectives(None, &state);
    assert_eq!(with_step.next_rooms, [loc(1, 0x33)].into());
    assert_eq!(stepless.next_rooms, with_step.next_rooms);
}

/// F026: leaving a room whose treasure (an untaken map) still lies there for a room whose treasure is pending does
/// not mark the new room's treasure as taken: the chain's objective equals a fresh episode's (the boomerang fight).
#[test]
fn a_treasure_left_behind_does_not_take_the_next_rooms() {
    let Some(images) = images(&["1_43n_0", "1_44w_0"]) else { return };
    let map = Arc::new(GameMap::builtin());
    let (a, b) = (view(&map, &images[0], 0), view(&map, &images[1], 100));
    assert!(a.treasure().is_some(), "the map lies in 0x43");
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let (route, step_a) = start_route(&map, loc(1, 0x43), Some("N"), &mut rng, None, &a);
    let (_, step_b) = start_route(&map, loc(1, 0x44), Some("W"), &mut rng, None, &b);
    let mut chain = GameCompletion::new(map.clone(), &route, step_a).unwrap();
    chain.objectives(None, &a);
    let walked = chain.objectives(Some(&a), &b);
    let fresh = GameCompletion::new(map.clone(), &route, step_b).unwrap().objectives(None, &b);
    assert_eq!(fresh.kind, ObjectiveKind::Fight);
    assert_eq!((walked.kind, walked.next_rooms), (fresh.kind, fresh.next_rooms));
}
