//! d3first's bombs in a chain run through Dungeon 5 (the owner, Nov 21: no bag handed over at the entrance): from
//! `d3first_5_76s_0` (Link brings 8 bombs, the median a chain run arrives with on the corrected route) to the Olive
//! Garden's door.  The route grants nothing; 5:0x66's and 5:0x65's walls are opened with his own bombs, so he enters
//! 5:0x05 with 6 and nothing cheated.  Skipped without the ROM.
//!
//! SETUP (RAM writes, the walk is about the items): the rooms' monsters are removed and Link's hearts refilled before
//! every decision; a key door is lined up a frame at a time (walker aid, controller input).

mod common;

use common::{TfzStore, ZCore, open_core};
use env::chain::{ChainEnv, ChainStatus, Route, RunResult};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use zelda::clips::skip_available;
use zelda::executor::direction_buttons;
use zelda::objectives::Objective;
use zelda::room::Target;
use zelda::{Action, ActionKind, Direction, MapLocation, Var};

type C = ChainEnv<ZCore, TfzStore>;

const OBJ_TYPE: u16 = 0x34F;
/// Object types the walk keeps (as `zelda/tests/d8.rs`): the dock, the ladder, the rock wall, the whirlwind, a push
/// block, the pushed gravestone, the burned tree, the flute secret.
const KEEP: [u8; 9] = [0x61, 0x5F, 0x63, 0x2E, 0x68, 0x65, 0x64, 0x5E, 0x33];

fn uw(room: u8) -> MapLocation {
    MapLocation::new(5, room, false)
}

/// Removes the room's monsters and refills Link's hearts (SETUP).
fn tidy(c: &mut C) {
    let st = c.session().state().clone();
    if st.mode() != 5 {
        return;
    }
    let s = c.session_mut();
    for slot in 1..=11u16 {
        let t = st.ram()[(OBJ_TYPE + slot) as usize];
        if t != 0 && !KEEP.contains(&t) {
            zelda::Nes::poke(s.nes_mut(), OBJ_TYPE + slot, 0);
        }
    }
    let hc = st.hearts_and_containers();
    s.apply_ram(&[(Var::HeartsAndContainers, (hc & 0xF0) | (hc >> 4))]);
}

/// Walker aid: presses a frame at a time until Link stands on (`tx`, `ty`), the row first (`row_first`) or the
/// column first.
fn line_up(c: &mut C, (tx, ty): (i32, i32), row_first: bool) {
    for _ in 0..24 {
        let p = c.session().state().link.position;
        let vertical = (p.y != ty).then_some(if p.y < ty { Direction::S } else { Direction::N });
        let horizontal = (p.x != tx).then_some(if p.x < tx { Direction::E } else { Direction::W });
        let Some(d) = (if row_first { horizontal.or(vertical) } else { vertical.or(horizontal) }) else {
            break;
        };
        c.session_mut().press(direction_buttons(d), 1);
    }
}

/// The walker's next action down the objective's path search (as `zelda/tests/d8.rs`): a bomb at a bomb wall's spot,
/// a skip that ends nearer, a push block, the move into the nearer neighbour, off the edge from an exit tile.
fn next_action(c: &C, objective: &Objective, nudge: usize) -> Action {
    let state = c.session().state();
    let mask = c.session().mask();
    let wait = Action::parse("WAIT 4").unwrap();
    if let Some(shove) = zelda::actions::shove_move(state) {
        return shove;
    }
    if !matches!(state.mode(), 5 | 9 | 11) || state.link.status != 0 {
        return wait;
    }
    let targets: Vec<Target> = objective.targets.iter().copied().collect();
    let room = state.room();
    let locked: Vec<Direction> = [Direction::N, Direction::S, Direction::E, Direction::W]
        .into_iter()
        .filter(|&d| state.keys() > 0 && !state.full_location().in_cave && state.is_door_locked(d))
        .collect();
    let wavefront = room.wavefront(&targets, &[], &locked, objective.clips);
    let here = state.link.tile();
    let (x, y) = (state.link.position.x, state.link.position.y);
    let moving = |d: Direction| {
        let along = match d {
            Direction::W | Direction::E => x.rem_euclid(8),
            _ => (y - 0x3D).rem_euclid(8),
        };
        let pixels = match (along, matches!(d, Direction::W | Direction::N)) {
            (0, _) => 8,
            (a, true) => a,
            (a, false) => 8 - a,
        };
        Action::parse(&format!("MOVE {} {}", d.name(), if pixels <= 6 { 4 } else { 8 })).unwrap()
    };
    let Some(distance) = wavefront.get(here) else {
        return wait;
    };
    if distance == 0 {
        for d in [Direction::N, Direction::S, Direction::W, Direction::E] {
            let bomb = Action::new(ActionKind::Bomb, d, 0).unwrap();
            if state.bomb_walls().iter().any(|w| w.wall_direction == d) && mask.is_allowed(&bomb) {
                return bomb;
            }
        }
        for d in [Direction::N, Direction::S, Direction::W, Direction::E] {
            if state.is_push_block_direction(d) && mask.is_allowed(&moving(d)) {
                return Action::parse(&format!("MOVE {} 16", d.name())).unwrap();
            }
        }
        return moving(room.exits.direction_of(here).unwrap_or(Direction::N));
    }
    let mut skip_ahead = false;
    for jump in wavefront.jumps_from(here) {
        if wavefront.get(jump.to).is_some_and(|to| to < distance) {
            skip_ahead = true;
            let clip = Action::parse(&format!("CLIP {}", jump.direction.name())).unwrap();
            if skip_available(state, jump.direction) && mask.is_allowed(&clip) {
                return clip;
            }
        }
    }
    // A skip from this tile that isn't offered off its grid point: onto the grid point first.
    if skip_ahead && x.rem_euclid(8) != 0 {
        return moving(if x.rem_euclid(8) <= 4 { Direction::W } else { Direction::E });
    }
    if skip_ahead && (y - 0x3D).rem_euclid(8) != 0 {
        return moving(if (y - 0x3D).rem_euclid(8) <= 4 { Direction::N } else { Direction::S });
    }
    let order = [Direction::N, Direction::S, Direction::W, Direction::E];
    let mut best: Option<(i32, Action)> = None;
    for i in 0..4 {
        let d = order[(i + nudge) % 4];
        let next = match d {
            Direction::N => (here.0, here.1 - 1),
            Direction::S => (here.0, here.1 + 1),
            Direction::W => (here.0 - 1, here.1),
            _ => (here.0 + 1, here.1),
        };
        let walked = room.can_move(here.0, here.1, d) && mask.is_allowed(&moving(d));
        if let Some(to) = wavefront.get(next).filter(|_| walked)
            && to < distance
            && best.as_ref().is_none_or(|(b, _)| to < *b)
        {
            best = Some((to, moving(d)));
        }
    }
    if let Some((_, action)) = best {
        return action;
    }
    match room.exits.direction_of(here) {
        Some(d) if distance == 1 => moving(d),
        _ => wait,
    }
}

/// Plays the chain by the walker until the run ends; a key door the objective leaves through is lined up first.
fn walk(c: &mut C, mut status: ChainStatus, limit: usize) -> RunResult {
    let (mut last, mut still) = (None, 0usize);
    for _ in 0..limit {
        if let ChainStatus::Done(r) = status {
            return r;
        }
        tidy(c);
        let objective = c.objective().expect("run started").clone();
        let state = c.session().state().clone();
        let tile = (state.full_location(), state.link.position.x, state.link.position.y);
        still = if last == Some(tile) { still + 1 } else { 0 };
        last = Some(tile);
        let here = state.link.tile();
        let key_door = [Direction::N, Direction::S, Direction::W, Direction::E].into_iter().find_map(|d| {
            let (mx, my) = match d {
                Direction::N => (0x78, 0x5D),
                Direction::S => (0x78, 0xBD),
                Direction::W => (0x20, 0x8D),
                _ => (0xD0, 0x8D),
            };
            let mouth = zelda::Position::new(mx, my).tile();
            let near = (mouth.0 - here.0).abs() <= 1 && (mouth.1 - here.1).abs() <= 1;
            let wanted = objective.targets.contains(&Target::Dir(d)) && state.is_door_locked(d);
            (state.mode() == 5 && state.keys() > 0 && wanted && near).then_some((d, (mx, my)))
        });
        let action = match key_door {
            Some((d, mouth)) => {
                line_up(c, mouth, matches!(d, Direction::N | Direction::S));
                Action::new(ActionKind::Move, d, 8).unwrap()
            }
            None => next_action(c, &objective, still / 3),
        };
        status = c.step(action).expect("step");
    }
    panic!("the walk didn't end: {}", c.session().state().full_location());
}

/// Dungeon 5's way in from its entrance to the Olive Garden's door: Link's own bombs, no route grant.
#[test]
fn no_bag_is_handed_over_at_dungeon_5() {
    let Some(nes) = open_core() else { return };
    let data = common::game_data();
    let route = Route::load(&data.map, &data.states_dir, "d3first", Some(uw(0x05)), Some("5_76s_0")).unwrap();
    let mut c = ChainEnv::new(data.clone(), nes, TfzStore::new(), route);
    let status = c.reset(Some(0), &mut ChaCha8Rng::seed_from_u64(0)).unwrap();
    let (_, grants) = c.context();
    assert_eq!((c.session().state().bombs(), grants.route.bombs, grants.room.bombs), (8, 0, 0), "his own eight");
    let run = walk(&mut c, status, 20_000);
    assert_eq!(run.result, "success", "{run:?}");
    let rooms: Vec<_> = run.rooms.iter().map(|r| (r.location(), r.bombs_in, r.route_bombs_in, r.cheated.bombs)).collect();
    let wall = |room: u8| rooms.iter().find(|r| r.0 == uw(room)).copied().unwrap();
    // The two wall rooms take one of his bombs each; nothing is granted by the route anywhere.
    assert!(rooms.iter().all(|r| r.2 == 0), "{rooms:?}");
    assert_eq!((wall(0x66).1, wall(0x65).1), (8, 7), "{rooms:?}");
    assert_eq!(rooms.last().unwrap().0, uw(0x05), "{rooms:?}");
    assert_eq!(c.context().1.route.bombs, 0);
    assert_eq!((c.session().state().bombs(), c.cheated().bombs), (6, 0), "both walls with his own bombs");
}
