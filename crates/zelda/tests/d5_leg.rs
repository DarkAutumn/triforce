//! Round 24: the overworld leg from the Dungeon 4 triforce to Dungeon 5 in the real game (`triforce/routes.yaml`
//! `d2first` steps 130-181 since round 31, its stretches `coast-heart`, `lake-3e` and `mountain`; `docs/routes/d5-leg.md`,
//! `zelda-asm/mechanics.md` "Overworld: room `$5F`").
//!
//! - The whole clip way played by the route's objective (`GameCompletion`): from `4_03s_0` a walker takes, at every
//!   decision, the move or the skip (`CLIP`) down the objective's path search (the PBRS wavefront, clips on), from
//!   the D4 triforce through the raft back to 0x55, the wrap over 0x55's river, 0x51's and 0x60's wrap exits, the
//!   overworld ladder clip to 0x5f's heart container (8 -> 9 hearts), the raft to 0x2f's take-any cave (9 -> 10, the
//!   potion never taken), 0x3e's wrap exit, 0x1f's wall gap into 0x0f's secret cave (+100 rupees) and back, 0x0c's
//!   wrap exit onto the mountain top and into Dungeon 5 (level 5, room 0x76). Every room change is an exit the
//!   objective counts correct.
//! - Each wrap exit (0x51, 0x60, 0x3e, 0x0c) and 0x55's wrap onto its W part offered on its rows only, played on
//!   each of them, never on a softlock row.
//! - The plain overworld ladder: 0x5f from S, E over its two single squares of water to the heart container and back
//!   to land, by MOVE alone.
//! - The run-time clips switch turned off on 0x5f's water waits until Link can walk on.
//!
//! SETUP (RAM writes): monsters are removed on entry to every overworld room (object slots 1-11, caves left alone);
//! rooms without a state of their own are reached as `wrap_exit.rs` does, by setting `RoomId` before a room change.
#![cfg(feature = "zcore")]

mod common;

use common::{Counted, load_named_state, open_core};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;
use zelda::clips::{SkipKind, skip_available, skip_target};
use zelda::executor::direction_buttons;
use zelda::objectives::{GameCompletion, Objective, ObjectiveKind, exit_is_correct, start_route, stranded};
use zelda::room::Target;
use zelda::{Action, ActionKind, Direction, GameMap, GameView, LoadOptions, MapLocation, Session, Var};

type S = Session<Counted>;

const ROUTE: &str = "d2first";
const OBJ_TYPE: u16 = 0x34F;
const ROOM_ID: u16 = 0xEB;
/// The D4 triforce step of `d2first`.
const TRIFORCE_STEP: usize = 129;

fn ow(room: u8) -> MapLocation {
    MapLocation::new(0, room, false)
}

fn map() -> Arc<GameMap> {
    Arc::new(GameMap::builtin())
}

/// The start state `name` loaded (no randomization) to its first playable frame, clips on.
fn start(name: &str) -> Option<S> {
    let core = open_core()?;
    let mut s = Session::new(Counted(core, 0), map());
    s.set_clips(true);
    s.load(&load_named_state(name), &LoadOptions::default(), &mut ChaCha8Rng::seed_from_u64(0)).unwrap();
    assert_eq!(s.state().mode(), 5, "{name}: {}", describe(&s));
    Some(s)
}

fn act(s: &mut S, text: &str) -> zelda::Change {
    s.act(Action::parse(text).unwrap()).unwrap()
}

/// (x, y, grid offset) of Link.
fn link(s: &S) -> (i32, i32, i32) {
    let st = s.state();
    (st.link.position.x, st.link.position.y, st.var(Var::LinkGridOffset))
}

fn describe(s: &S) -> String {
    let st = s.state();
    format!(
        "{} mode {} at ({:#04x}, {:#04x}) offset {} hearts {}/{} frame {}",
        st.full_location(),
        st.mode(),
        st.link.position.x,
        st.link.position.y,
        st.var(Var::LinkGridOffset),
        st.health(),
        st.max_health(),
        s.frame()
    )
}

/// Object types `clear_room` keeps: the dock (`$61`, asm:Room_SetUpTileObjOW), the ladder (`$5F`), the rock wall's
/// tile object (`$63`).
const KEEP: [u8; 3] = [0x61, 0x5F, 0x63];

/// Removes an overworld room's monsters (SETUP; caves keep their people, `KEEP` stays).
fn clear_room(s: &mut S) {
    if s.state().full_location().in_cave || s.state().mode() != 5 {
        return;
    }
    let ram = s.state().ram().to_vec();
    let gone: Vec<u16> =
        (1..=11u16).filter(|&slot| ram[(OBJ_TYPE + slot) as usize] != 0 && !KEEP.contains(&ram[(OBJ_TYPE + slot) as usize])).collect();
    if gone.is_empty() {
        return;
    }
    for slot in gone {
        zelda::Nes::poke(s.nes_mut(), OBJ_TYPE + slot, 0);
    }
    s.apply_ram(&[]);
}

/// Holds `buttons` a frame at a time until `until` holds (at most 900 frames).
fn press_until(s: &mut S, buttons: u8, until: impl Fn(&S) -> bool) {
    for _ in 0..900 {
        if until(s) {
            return;
        }
        clear_room(s);
        s.press(buttons, 1);
    }
    panic!("never got there: {}", describe(s));
}

/// Walks `d` until Link's coordinate on that axis is `to`, on a grid point.
fn walk(s: &mut S, d: Direction, to: i32) {
    let horizontal = matches!(d, Direction::W | Direction::E);
    press_until(s, direction_buttons(d), |s| {
        let (x, y, offset) = link(s);
        (if horizontal { x } else { y }) == to && offset == 0
    });
}

/// Walks `d` with `RoomId` set to the room behind `room` (SETUP) until Link plays in `room`.
fn walk_into(s: &mut S, d: Direction, room: u8) {
    let behind = match d {
        Direction::N => room.wrapping_add(16),
        Direction::S => room.wrapping_sub(16),
        Direction::W => room.wrapping_add(1),
        _ => room.wrapping_sub(1),
    };
    zelda::Nes::poke(s.nes_mut(), ROOM_ID, behind);
    s.apply_ram(&[]);
    press_until(s, direction_buttons(d), |s| s.state().full_location() == ow(room) && s.state().mode() == 5);
}

/// The walker's next action: down the objective's path search from Link's tile (the move into the neighbour with
/// the lowest distance, or the skip whose end plus its cost is lowest), off the edge from an exit tile, up into a
/// cave or a ware at distance 0; WAIT while Link isn't controllable.
fn next_action(s: &S, objective: &Objective, nudge: u32) -> Action {
    let state = s.state();
    let mask = s.mask();
    let wait = Action::parse("WAIT 4").unwrap();
    // Play (5) or a cave (11).
    if !matches!(state.mode(), 5 | 11) || state.link.status != 0 {
        return wait;
    }
    let targets: Vec<Target> = objective.targets.iter().copied().collect();
    let room = state.room();
    let wavefront = room.wavefront(&targets, &[], &[], objective.clips);
    let here = state.link.tile();
    // MOVE 4 (about 6 pixels) when the next grid point that way is that near, else 8 (about 12), so the walk keeps
    // near the tiles.
    let (x, y) = (state.link.position.x, state.link.position.y);
    let moving = |d: Direction| {
        let (along, back) = match d {
            Direction::W => (x.rem_euclid(8), true),
            Direction::E => (x.rem_euclid(8), false),
            Direction::N => ((y - 0x3D).rem_euclid(8), true),
            _ => ((y - 0x3D).rem_euclid(8), false),
        };
        let pixels = match (along, back) {
            (0, _) => 8,
            (a, true) => a,
            (a, false) => 8 - a,
        };
        Action::parse(&format!("MOVE {} {}", d.name(), if pixels <= 6 { 4 } else { 8 })).unwrap()
    };
    // No way from here (the frame the raft is boarded at 0x55's top edge, before it moves): wait.
    let Some(distance) = wavefront.get(here) else {
        return wait;
    };
    if distance == 0 {
        // On a cave mouth or a ware: up into it. On an exit tile reached at distance 0 the edge is ahead.
        let d = room.exits.direction_of(here).unwrap_or(Direction::N);
        return moving(d);
    }
    // A skip the search knows from here that ends nearer the targets, when the game offers it: taken (its macro
    // walks the run-up itself; a MOVE of 8 frames overshoots the cell and walks back and forth around its zone).
    for jump in wavefront.jumps_from(here) {
        if wavefront.get(jump.to).is_some_and(|to| to < distance) && skip_available(state, jump.direction) {
            return Action::parse(&format!("CLIP {}", jump.direction.name())).unwrap();
        }
    }
    let mut best: Option<(i32, Action)> = None;
    let order = [Direction::N, Direction::S, Direction::W, Direction::E];
    for i in 0..4 {
        let d = order[(i + nudge as usize) % 4];
        let (dx, dy) = match d {
            Direction::N => (0, -1),
            Direction::S => (0, 1),
            Direction::W => (-1, 0),
            _ => (1, 0),
        };
        let next = (here.0 + dx, here.1 + dy);
        let walked = room.can_move(here.0, here.1, d) && mask.is_allowed(&moving(d));
        let to = wavefront.get(next).filter(|_| walked);
        if let Some(to) = to
            && to < distance
            && best.as_ref().is_none_or(|(b, _)| to < *b)
        {
            best = Some((to, moving(d)));
        }
    }
    if let Some((_, action)) = best {
        return action;
    }
    // At distance 1 from an exit's start beyond the edge: off the edge.
    if distance == 1
        && let Some(d) = room.exits.direction_of(here)
    {
        return moving(d);
    }
    let _ = ActionKind::Move;
    wait
}

/// A room visit of the walk: the room, the route step the objective plays there, frames spent, and Link's max
/// hearts and rupees on leaving it.
#[derive(Debug)]
struct Visit {
    room: MapLocation,
    step: Option<usize>,
    frames: u64,
    max_health: i32,
    rupees: i32,
}

/// Plays `s` by the objective of `gc` until `done`, checking every room change is a correct exit; the visits.
fn walk_by_objective(s: &mut S, gc: &mut GameCompletion, done: impl Fn(&S) -> bool, limit: usize) -> Vec<Visit> {
    let map = s.game_map().clone();
    let mut prev: Option<Arc<GameView>> = None;
    let mut objective = gc.objectives(None, s.state());
    let mut visits = vec![Visit {
        room: s.state().full_location(),
        step: gc.step(),
        frames: 0,
        max_health: 0,
        rupees: 0,
    }];
    let mut since = s.frame();
    let (mut last_tile, mut still) = (None, 0u32);
    for _ in 0..limit {
        clear_room(s);
        let state = s.state().clone();
        let before = objective.clone();
        let entered = prev.as_ref().is_some_and(|p| p.full_location() != state.full_location());
        objective = gc.objectives(prev.as_deref(), &state);
        if entered {
            let p = prev.as_ref().unwrap();
            // The triforce's warp out of the dungeon is the step's own end, not an exit.
            let warp = p.level() > 0 && state.level() == 0;
            assert!(
                warp || exit_is_correct(&map, p, &before, state.full_location()),
                "{} -> {}: not a correct exit ({before:?})",
                p.full_location(),
                state.full_location()
            );
            let visit = visits.last_mut().unwrap();
            (visit.frames, visit.max_health, visit.rupees) = (s.frame() - since, p.max_health(), p.rupees());
            since = s.frame();
            visits.push(Visit { room: state.full_location(), step: gc.step(), frames: 0, max_health: 0, rupees: 0 });
        }
        assert!(gc.missed().is_none(), "{:?} at {}", gc.missed(), describe(s));
        // Done once the objective has seen the room (its visit records the step Link plays there).
        if done(s) {
            break;
        }
        let tile = (state.full_location(), state.link.position.x, state.link.position.y);
        still = if last_tile == Some(tile) { still + 1 } else { 0 };
        last_tile = Some(tile);
        // Walker aid (controller input): a cave mouth takes Link only at an X multiple of $10
        // (asm:Warp_CheckUnderLink), which MOVEs of 4 or 8 frames rarely hit; next to one, line him up a frame at a time.
        let here = state.link.tile();
        let near_mouth = objective.kind == ObjectiveKind::Cave
            && state.mode() == 5
            && objective.targets.iter().any(|t| matches!(*t, Target::Tile((c, r)) if (c - here.0).abs() <= 1 && (0..=2).contains(&(here.1 - r))));
        if near_mouth && state.link.position.x % 16 != 0 {
            let buttons = direction_buttons(if state.link.position.x % 16 < 8 { Direction::W } else { Direction::E });
            for _ in 0..16 {
                if s.state().link.position.x % 16 == 0 && s.state().var(Var::LinkGridOffset) == 0 {
                    break;
                }
                s.press(buttons, 1);
            }
        }
        let action = next_action(s, &objective, still / 3);
        prev = Some(state);
        s.act(action).unwrap();
    }
    assert!(done(s), "the walk ended at {} ({} rooms)", describe(s), visits.len());
    let last = visits.last_mut().unwrap();
    if last.room != s.state().full_location() {
        (last.frames, last.max_health, last.rupees) = (s.frame() - since, s.state().max_health(), s.state().rupees());
        visits.push(Visit { room: s.state().full_location(), step: gc.step(), frames: 0, max_health: 0, rupees: 0 });
    }
    visits
}

/// The clip way of `d2first`'s leg from the D4 triforce to Dungeon 5, played by the route's objective. Every room
/// change is a correct exit into the clip way's next room; the heart containers of 0x5f (the ladder clip) and 0x2f
/// (its cave's right ware) take Link from 8 to 10 hearts, 0x0f's cave pays 100 rupees, and the walk ends in level 5
/// room 0x76, where Dungeon 5's first step (N) takes over.
#[test]
fn the_clip_way_is_played_by_the_objective_from_the_d4_triforce_to_dungeon_5() {
    let Some(mut s) = start("4_03s_0") else { return };
    let map = s.game_map().clone();
    let mut gc = GameCompletion::new(map.clone(), ROUTE, Some(TRIFORCE_STEP)).unwrap();
    gc.set_clips(true);
    let potion = s.state().var(Var::Potion);
    assert_eq!((s.state().max_health(), s.state().ladder(), s.state().raft()), (8, true, true));
    let visits = walk_by_objective(&mut s, &mut gc, |s| s.state().level() == 5 && s.state().mode() == 5, 20_000);
    for v in &visits {
        println!("{} step {:?}: {} frames, out with {} hearts, {} rupees", v.room, v.step, v.frames, v.max_health, v.rupees);
    }
    let d5 = MapLocation::new(5, 0x76, false);
    let way: Vec<(usize, MapLocation)> = map
        .way_steps(ROUTE, true)
        .map(|(i, st)| (i, st.location()))
        .filter(|&(i, _)| i >= TRIFORCE_STEP)
        .take_while(|&(_, l)| l != d5)
        .collect();
    let want: Vec<MapLocation> = way.iter().map(|&(_, l)| l).chain([d5]).collect();
    let got: Vec<MapLocation> = visits.iter().map(|v| v.room).collect();
    assert_eq!(got, want, "the rooms of the clip way");
    let route_steps: Vec<Option<usize>> = visits[..visits.len() - 1].iter().map(|v| v.step).collect();
    let want_steps: Vec<Option<usize>> = way.iter().map(|&(i, _)| Some(i)).collect();
    assert_eq!(route_steps, want_steps, "each room is the route's step");
    // Where the slices meet: the clip way's last step (0x0b E -> cave) into Dungeon 5's first step (0x76 S -> N).
    let steps = &map.routes[ROUTE];
    let last = way.last().unwrap().0;
    assert_eq!((steps[last].location(), steps[last].exit.as_str()), (ow(0x0b), "cave"));
    let (route, first) = start_route(&map, d5, Some("S"), None, &mut ChaCha8Rng::seed_from_u64(0), Some(ROUTE), s.state(), true);
    let first = first.expect("Dungeon 5's first step");
    assert_eq!((route.as_str(), steps[first].exit.as_str(), steps[first].way), (ROUTE, "N", None));
    assert_eq!(steps[first - 1].location(), ow(0x0b), "the mountain stretch rejoins on 0x76");
    let objective = GameCompletion::new(map.clone(), ROUTE, Some(first)).unwrap().objectives(None, s.state());
    assert_eq!(objective.next_rooms, [MapLocation::new(5, 0x66, false)].into());
    let out = |room: MapLocation| visits.iter().find(|v| v.room == room).unwrap();
    assert_eq!(out(ow(0x60)).max_health, 8, "8 hearts out of Dungeon 4");
    assert_eq!(out(ow(0x5f)).max_health, 9, "0x5f's heart container by the ladder clip");
    assert_eq!(out(MapLocation::new(0, 0x2f, true)).max_health, 10, "0x2f's heart container");
    assert_eq!(s.state().var(Var::Potion), potion, "the potion never taken");
    let before_0f = out(ow(0x1f)).rupees;
    assert_eq!(s.state().rupees(), before_0f + 100, "0x0f's cave: 100 rupees (counted up by the end)");
    assert_eq!((s.state().full_location(), s.state().max_health()), (MapLocation::new(5, 0x76, false), 10));
    println!("total {} frames", visits.iter().map(|v| v.frames).sum::<u64>());
}

/// From `4_03s_0`: the triforce, the warp to 0x45, the raft down to 0x55 (Link halted on it from Y `$3D` to `$7F`
/// at X `$80`), the wrap over 0x55's river (CLIP E on Y `$8D`), W into 0x54 at (`$F0`, `$8D`), monsters removed.
fn into_0x54(s: &mut S) {
    press_until(s, direction_buttons(Direction::N), |s| s.state().level() == 0);
    press_until(s, 0, |s| s.state().full_location() == ow(0x45) && s.state().mode() == 5);
    assert_eq!((link(s).0, link(s).1), (0x80, 0x7D), "0x45: out of the dungeon's mouth: {}", describe(s));
    press_until(s, direction_buttons(Direction::S), |s| s.state().full_location() == ow(0x55) && s.state().mode() == 5);
    // On the top edge at the dock's X with Down held the raft takes him down (asm:Dock_Update).
    press_until(s, direction_buttons(Direction::S), |s| {
        let st = s.state();
        st.mode() == 5 && st.link.status == 0 && st.link.position.y >= 0x7F
    });
    assert_eq!((link(s).0, link(s).1), (0x80, 0x7F), "the raft lands on the dock: {}", describe(s));
    walk(s, Direction::S, 0x8D);
    walk(s, Direction::E, 0xE0);
    let target = skip_target(s.state(), Direction::E).expect("the wrap over the river");
    assert_eq!(target.kind, SkipKind::Wrap);
    let c = act(s, "CLIP E");
    assert!(c.clip.is_some_and(|c| c.completed()), "{}", describe(s));
    assert_eq!((s.state().full_location(), link(s).0, link(s).1), (ow(0x55), 0x00, 0x8D), "W part: {}", describe(s));
    act(s, "MOVE W 8");
    assert_eq!((s.state().full_location(), link(s).0), (ow(0x54), 0xF0), "{}", describe(s));
    clear_room(s);
}

/// The raft rides of the leg: 0x45's pier S into 0x55 (the ride down to the dock), 0x3f's dock N to 0x2f's pier and
/// 0x2f's pier S back to 0x3f's dock.
#[test]
fn the_rafts_ride_down_from_0x45_and_up_and_down_between_0x3f_and_0x2f() {
    let Some(mut s) = start("4_03s_0") else { return };
    into_0x54(&mut s);
    // To a top edge of 0x54 (columns 14-17 are open to it) and N into 0x3f from S (SETUP).
    walk(&mut s, Direction::W, 0x70);
    walk(&mut s, Direction::N, 0x45);
    walk_into(&mut s, Direction::N, 0x3f);
    // The dock (X $60, Y $7D): the ride N into 0x2f.
    walk(&mut s, Direction::N, 0x8D);
    walk(&mut s, Direction::W, 0x60);
    press_until(&mut s, direction_buttons(Direction::N), |s| s.state().full_location() == ow(0x2f));
    press_until(&mut s, 0, |s| s.state().mode() == 5 && s.state().link.status == 0);
    assert_eq!((link(&s).0, link(&s).2), (0x60, 0), "0x2f's pier: {}", describe(&s));
    // Back S off the pier at X $60: the ride down to 0x3f's dock.
    press_until(&mut s, direction_buttons(Direction::S), |s| s.state().full_location() == ow(0x3f) && s.state().mode() == 5);
    press_until(&mut s, direction_buttons(Direction::S), |s| {
        let st = s.state();
        st.mode() == 5 && st.link.status == 0 && st.link.position.y >= 0x7F
    });
    assert_eq!((link(&s).0, link(&s).1), (0x60, 0x7F), "0x3f's dock: {}", describe(&s));
}

/// Into 0x60 from N (SETUP from 0x54's S edge) and to rest on (`$E0`, `y`) of its E side.
fn into_0x60_at(s: &mut S, y: i32) {
    walk(s, Direction::W, 0x90);
    walk(s, Direction::S, 0xDD);
    walk_into(s, Direction::S, 0x60);
    walk(s, Direction::S, y);
    walk(s, Direction::E, 0xE0);
}

/// 0x60's wrap exit round the world into 0x5f (room id - 1, entered from E) and the overworld ladder clip from the
/// arrival on its water to the heart container, with the agent's actions: `CLIP E`, then `CLIP W` (offered there,
/// and only W); 8 -> 9 hearts; then MOVE W twice over the single squares by the plain ladder to land, and N.
#[test]
fn round_the_world_into_0x5f_and_the_ladder_clip_take_the_heart_container() {
    let Some(mut s) = start("4_03s_0") else { return };
    into_0x54(&mut s);
    into_0x60_at(&mut s, 0x8D);
    assert_eq!(skip_target(s.state(), Direction::E).map(|t| t.kind), Some(SkipKind::WrapExit), "{}", describe(&s));
    let from = s.frame();
    let c = act(&mut s, "CLIP E");
    assert!(c.clip.is_some_and(|c| c.completed()), "{}", describe(&s));
    assert_eq!((s.state().full_location(), link(&s).0, link(&s).1), (ow(0x5f), 0xF0, 0x8D), "{}", describe(&s));
    let state = s.state().clone();
    assert!(stranded(&state), "on the water only a skip leads on");
    let usable: Vec<Direction> = [Direction::N, Direction::S, Direction::W, Direction::E]
        .into_iter()
        .filter(|&d| skip_available(&state, d))
        .collect();
    assert_eq!(usable, [Direction::W]);
    let target = skip_target(&state, Direction::W).unwrap();
    assert_eq!((target.kind, target.t, target.end), (SkipKind::Ladder, 0xE0, -32));
    assert!(s.mask().is_allowed(&Action::parse("CLIP W").unwrap()) && s.mask().is_allowed(&Action::parse("MOVE W 8").unwrap()));
    let c = act(&mut s, "CLIP W");
    assert!(c.clip.is_some_and(|c| c.completed()), "{}", describe(&s));
    assert_eq!((link(&s).0, s.state().max_health()), (0xC0, 9), "the heart container: {}", describe(&s));
    println!("0x60 rest to the heart container: {} frames", s.frame() - from);
    assert!(!stranded(s.state()), "the plain ladder leads on from the platform");
    for _ in 0..12 {
        if link(&s).0 <= 0x88 {
            break;
        }
        act(&mut s, "MOVE W 8");
    }
    assert!(link(&s).0 <= 0x88, "on land: {}", describe(&s));
    walk(&mut s, Direction::W, 0x80);
    press_until(&mut s, direction_buttons(Direction::N), |s| s.state().full_location() == ow(0x4f));
}

/// The plain way's 0x5f (from S, out of 0x6f; SETUP from 0x54's top edge): E along Y `$8D` over the two single
/// squares of water by the plain ladder (MOVE alone, no CLIP offered), the heart container (8 -> 9), back W to land.
/// The room model crosses them with the ladder and not without it.
#[test]
fn the_plain_ladder_takes_0x5f_s_heart_container_from_the_s_side() {
    let Some(mut s) = start("4_03s_0") else { return };
    into_0x54(&mut s);
    walk(&mut s, Direction::W, 0x70);
    walk(&mut s, Direction::N, 0x45);
    walk_into(&mut s, Direction::N, 0x5f);
    clear_room(&mut s);
    walk(&mut s, Direction::N, 0x8D);
    walk(&mut s, Direction::E, 0x80);
    let room = s.state().room().clone();
    assert!(room.ladder(), "0x5f is a ladder room");
    let plain = zelda::room::Room::new(ow(0x5f), room.tiles(), false);
    let heart = [Target::Tile((24, 10))];
    assert!(room.wavefront(&heart, &[], &[], false).get(s.state().link.tile()).is_some());
    assert!(plain.wavefront(&heart, &[], &[], false).get(s.state().link.tile()).is_none());
    let mut gc = GameCompletion::new(s.game_map().clone(), ROUTE, None).unwrap();
    gc.set_clips(true);
    let objective = gc.objectives(None, s.state());
    assert_eq!(objective.kind, ObjectiveKind::Treasure, "{objective:?}");
    for _ in 0..16 {
        if s.state().max_health() == 9 {
            break;
        }
        assert!(!skip_available(s.state(), Direction::E), "no clip on the plain way: {}", describe(&s));
        act(&mut s, "MOVE E 8");
    }
    assert_eq!(s.state().max_health(), 9, "the heart container: {}", describe(&s));
    println!("taken at X {:#04x}", link(&s).0);
    let objective = gc.objectives(None, s.state());
    assert_ne!(objective.kind, ObjectiveKind::Treasure, "taken: {objective:?}");
    for _ in 0..16 {
        if link(&s).0 <= 0x88 {
            break;
        }
        act(&mut s, "MOVE W 8");
    }
    assert!(link(&s).0 <= 0x88, "back on land: {}", describe(&s));
}

/// The run-time switch turned off on 0x5f's water waits (the clip way holds: CLIP W stays the objective's way,
/// `clips_allowed` stays on), and it still waits once the ladder clip has put Link on the platform: past 0x55's
/// river the coast stretch's clip way holds until its rejoin (`past_wrap_in`), so Link goes N to 0x4f by the clip
/// way's step and the switch is off there.
#[test]
fn the_clips_switch_holds_on_0x5f_s_water() {
    let Some(mut s) = start("4_03s_0") else { return };
    into_0x54(&mut s);
    into_0x60_at(&mut s, 0x8D);
    let map = s.game_map().clone();
    let step = map.routes[ROUTE].iter().position(|st| st.location() == ow(0x60)).unwrap();
    let mut gc = GameCompletion::new(map.clone(), ROUTE, Some(step)).unwrap();
    gc.set_clips(true);
    gc.objectives(None, s.state());
    let prev = s.state().clone();
    act(&mut s, "CLIP E");
    assert_eq!(s.state().full_location(), ow(0x5f));
    gc.set_clips_allowed(false);
    let objective = gc.objectives(Some(&prev), s.state());
    assert!(gc.clips_held() && gc.clips_allowed() && objective.clips, "held on the water: {objective:?}");
    assert_eq!(objective.kind, ObjectiveKind::Treasure);
    let prev = s.state().clone();
    act(&mut s, "CLIP W");
    assert_eq!(s.state().max_health(), 9);
    let objective = gc.objectives(Some(&prev), s.state());
    assert!(gc.clips_held() && gc.clips_allowed() && objective.clips, "held on the platform: {objective:?}");
    assert_eq!(gc.step(), Some(step + 1), "the clip way's 0x5f");
    assert_eq!(objective.next_rooms, [ow(0x4f)].into_iter().collect(), "on N to the rejoin");
    // W over the single squares by the plain ladder to land (as `round_the_world_...`): still held there.
    for _ in 0..12 {
        if link(&s).0 <= 0x88 {
            break;
        }
        act(&mut s, "MOVE W 8");
    }
    walk(&mut s, Direction::W, 0x80);
    let objective = gc.objectives(None, s.state());
    assert!(gc.clips_held() && objective.clips && !stranded(s.state()), "held on 0x5f's land: {}", describe(&s));
    let visits = walk_by_objective(&mut s, &mut gc, in_room(ow(0x4f)), 2_000);
    let rejoin = map.stretches[ROUTE].iter().find(|st| st.clips.contains(&step)).unwrap().rejoin;
    assert_eq!(visits.last().unwrap().step, Some(rejoin), "{visits:?}");
    assert!(!gc.clips_held() && !gc.clips_allowed(), "off at 0x4f");
    // Off the route's stretches (no step), where Link can walk: at once.
    let mut gc = GameCompletion::new(s.game_map().clone(), ROUTE, None).unwrap();
    gc.set_clips(true);
    gc.objectives(None, s.state());
    gc.set_clips_allowed(false);
    let objective = gc.objectives(None, s.state());
    assert!(!gc.clips_held() && !objective.clips);
}

/// The rows (Link's Y) of a room's E side: Y `$3D`-`$DD`.
fn rows() -> impl Iterator<Item = i32> {
    (0x3D..=0xDD).step_by(8)
}

/// Link put at rest on (`x`, `y`) of his room (SETUP), facing E.
fn place(s: &mut S, x: i32, y: i32) {
    s.apply_ram(&[
        (Var::LinkX, x as u8),
        (Var::LinkY, y as u8),
        (Var::LinkGridOffset, 0),
        (Var::LinkPosFrac, 0),
        (Var::LinkDirection, 1),
    ]);
}

/// The wrap exits of the leg (0x51 into 0x50, 0x3e into 0x3d, 0x0c into 0x0b) and 0x55's wrap over its river:
/// CLIP E is offered at X `$E0` exactly on the game.yaml rows (`span`, `wrap_in`) and nowhere else on the E side; on
/// each of those rows it plays into the room to the W (0x55: onto its W part at X `$00`), where Link walks on (MOVE W
/// moves him: no softlock row). 0x60's (rows 9-10) is played in the 0x5f tests: its arrival needs the ladder.
#[test]
fn each_wrap_exit_of_the_leg_is_offered_and_played_on_its_rows_only() {
    let Some(mut s) = start("0_38w_0") else { return };
    let base = s.snapshot();
    for (room, west, rows_ok) in [(0x51u8, 0x50u8, (7, 12)), (0x3e, 0x3d, (9, 16)), (0x0c, 0x0b, (5, 14)), (0x55, 0x55, (9, 10))] {
        s.restore(&base);
        walk_into(&mut s, Direction::W, room);
        clear_room(&mut s);
        let entered = s.snapshot();
        let mut played = 0;
        for y in rows() {
            s.restore(&entered);
            place(&mut s, 0xE0, y);
            let row = zelda::Position::new(0xE0, y).tile().1;
            let kind = skip_target(s.state(), Direction::E).map(|t| t.kind);
            let want = if room == 0x55 { SkipKind::Wrap } else { SkipKind::WrapExit };
            let on_rows = (rows_ok.0..=rows_ok.1).contains(&row);
            if !on_rows {
                assert_ne!(kind, Some(SkipKind::WrapExit), "{room:#04x} row {row}: a wrap exit off its rows");
                if room == 0x55 {
                    assert_eq!(kind, None, "0x55 row {row}");
                }
                continue;
            }
            assert_eq!(kind, Some(want), "{room:#04x} row {row}");
            let c = act(&mut s, "CLIP E");
            assert!(c.clip.is_some_and(|c| c.completed()), "{room:#04x} row {row}: {}", describe(&s));
            let (x, ..) = link(&s);
            assert_eq!(s.state().full_location(), ow(west), "{room:#04x} row {row}: {}", describe(&s));
            assert_eq!(x, if room == 0x55 { 0x00 } else { 0xF0 }, "{room:#04x} row {row}");
            if room == 0x55 {
                act(&mut s, "MOVE W 8");
                assert_eq!(s.state().full_location(), ow(0x54), "0x55 row {row}: W from the W part");
            } else {
                clear_room(&mut s);
                act(&mut s, "MOVE W 8");
                assert!(link(&s).0 < 0xF0, "{room:#04x} row {row}: stuck after the wrap exit: {}", describe(&s));
            }
            played += 1;
        }
        assert_eq!(played, rows_ok.1 - rows_ok.0 + 1, "{room:#04x}: every row played");
    }
}

/// 0x60's wrap exit is offered on rows 9-10 only (the rows whose arrival on 0x5f's water leads on with the ladder
/// clip); on the other open rows of its E edge (8, 11-13) the arrival would be a softlock.
#[test]
fn round_the_world_is_offered_only_on_the_rows_that_lead_on() {
    let Some(mut s) = start("4_03s_0") else { return };
    into_0x54(&mut s);
    into_0x60_at(&mut s, 0x8D);
    let entered = s.snapshot();
    for y in rows() {
        s.restore(&entered);
        place(&mut s, 0xE0, y);
        let row = zelda::Position::new(0xE0, y).tile().1;
        let kind = skip_target(s.state(), Direction::E).map(|t| t.kind);
        assert_eq!(kind == Some(SkipKind::WrapExit), (9..=10).contains(&row), "row {row}: {kind:?}");
        if (9..=10).contains(&row) {
            act(&mut s, "CLIP E");
            assert_eq!(s.state().full_location(), ow(0x5f), "row {row}");
            assert!(skip_available(s.state(), Direction::W), "row {row}: the ladder clip from the arrival");
        }
    }
}

/// 0x1f's wall gap: N at X `$80` only (into 0x0f) and back S; MOVE N is offered there and nowhere else along the
/// wall, and the room model's N exit is that column.
#[test]
fn the_wall_gap_of_0x1f_leads_n_at_x_80_only() {
    let Some(mut s) = start("0_38w_0") else { return };
    walk_into(&mut s, Direction::W, 0x1f);
    clear_room(&mut s);
    let entered = s.snapshot();
    let room = s.state().room().clone();
    assert_eq!(room.exits.get(Direction::N).unwrap(), [(16, 0)], "the gap is the N exit");
    for x in (0x10..=0xE0).step_by(8) {
        s.restore(&entered);
        place(&mut s, x, 0x55);
        let up = s.mask().is_allowed(&Action::parse("MOVE N 8").unwrap());
        assert_eq!(up, x == 0x80, "X {x:#04x}");
    }
    s.restore(&entered);
    place(&mut s, 0x80, 0x5D);
    press_until(&mut s, direction_buttons(Direction::N), |s| s.state().full_location() == ow(0x0f) && s.state().mode() == 5);
    assert_eq!(link(&s).0, 0x80);
    press_until(&mut s, direction_buttons(Direction::S), |s| s.state().full_location() == ow(0x1f) && s.state().mode() == 5);
    walk(&mut s, Direction::S, 0x6D);
}

const D5: MapLocation = MapLocation::new(5, 0x76, false);

/// `4_03s_0` with a `d2first` completion at the D4 triforce, the run-time clips switch `on` (the session's CLIP mask
/// follows it).
fn leg(on: bool) -> Option<(S, GameCompletion)> {
    let mut s = start("4_03s_0")?;
    let mut gc = GameCompletion::new(s.game_map().clone(), ROUTE, Some(TRIFORCE_STEP)).unwrap();
    gc.set_clips(true);
    if !on {
        s.set_clips(false);
        gc.set_clips_allowed(false);
    }
    Some((s, gc))
}

/// The steps of `d2first`'s `clips` way from the D4 triforce up to (not including) Dungeon 5's first room.
fn way(map: &GameMap, clips: bool) -> Vec<(usize, MapLocation)> {
    map.way_steps(ROUTE, clips)
        .map(|(i, st)| (i, st.location()))
        .filter(|&(i, _)| i >= TRIFORCE_STEP)
        .take_while(|&(_, l)| l != D5)
        .collect()
}

fn in_room(room: MapLocation) -> impl Fn(&S) -> bool {
    move |s| s.state().full_location() == room && s.state().mode() == 5
}

/// The plain way of the leg (clips off from the D4 triforce on), played by the route's objective: E along row 5, S
/// at 0x5d, row 6, N into 0x5f and its heart container by the plain ladder (9 hearts), 0x2f's (10), 0x3e N, 0x2e,
/// 0x2d, 0x0f's rupees, then the plain mountain way: 0x1d S, 0x2d W, 0x2c N, 0x1c W, the Lost Hills by their
/// sequence (four Ns, three loops back into 0x1b) into 0x0b from S and down into Dungeon 5, whose first step takes
/// over. Every room is the plain way's step; each stretch's rejoin (0x4f, 0x1d, 5:0x76) is its shared step.
#[test]
fn the_plain_way_is_played_by_the_objective_from_the_d4_triforce_to_dungeon_5() {
    let Some((mut s, mut gc)) = leg(false) else { return };
    let map = s.game_map().clone();
    let visits = walk_by_objective(&mut s, &mut gc, in_room(D5), 40_000);
    for v in &visits {
        println!("{} step {:?}: {} frames, out with {} hearts, {} rupees", v.room, v.step, v.frames, v.max_health, v.rupees);
    }
    let way = way(&map, false);
    let got: Vec<MapLocation> = visits.iter().map(|v| v.room).collect();
    let want: Vec<MapLocation> = way.iter().map(|&(_, l)| l).chain([D5]).collect();
    assert_eq!(got, want, "the rooms of the plain way");
    let steps: Vec<Option<usize>> = visits.iter().map(|v| v.step).collect();
    let d5_first = map.routes[ROUTE].iter().position(|st| st.location() == D5).unwrap();
    let want_steps: Vec<Option<usize>> = way.iter().map(|&(i, _)| Some(i)).chain([Some(d5_first)]).collect();
    assert_eq!(steps, want_steps, "each room is the plain way's step, the rejoins the shared ones");
    let route = &map.routes[ROUTE];
    for stretch in map.stretches[ROUTE].iter().filter(|st| st.rejoin > TRIFORCE_STEP && st.rejoin <= d5_first) {
        let rejoin = visits.iter().position(|v| v.step == Some(stretch.rejoin)).expect("the rejoin");
        assert_eq!(route[visits[rejoin - 1].step.unwrap()].way, Some(zelda::game_map::Way::Plain));
    }
    let maze = visits.iter().position(|v| v.room == ow(0x1b)).unwrap();
    assert_eq!(route[visits[maze].step.unwrap()].exit, "maze");
    assert_eq!((visits[maze + 1].room, route[visits[maze + 1].step.unwrap()].entry.as_str()), (ow(0x0b), "S"));
    let out = |room: MapLocation| visits.iter().find(|v| v.room == room).unwrap();
    assert_eq!(out(ow(0x5f)).max_health, 9, "0x5f's heart container by the plain ladder");
    assert_eq!(out(MapLocation::new(0, 0x2f, true)).max_health, 10, "0x2f's heart container");
    assert_eq!(s.state().max_health(), 10);
}

/// The run-time switch turned off inside a stretch's clip way (at `off`, on entry; the walker's skips follow the
/// objective's switch): the walk reaches the stretch's rejoin as its shared step, every room change a correct exit.
fn switched_off_inside(off: MapLocation, rejoin: MapLocation) -> Option<(Arc<GameMap>, Vec<Visit>)> {
    let (mut s, mut gc) = leg(true)?;
    walk_by_objective(&mut s, &mut gc, in_room(off), 20_000);
    assert!(gc.step().is_some_and(|i| s.game_map().routes[ROUTE][i].way == Some(zelda::game_map::Way::Clips)));
    gc.set_clips_allowed(false);
    let visits = walk_by_objective(&mut s, &mut gc, in_room(rejoin), 20_000);
    for v in &visits {
        println!("{} step {:?}: {} frames", v.room, v.step, v.frames);
    }
    let map = s.game_map().clone();
    let last = visits.last().unwrap().step.expect("the rejoin is a step");
    assert!(map.stretches[ROUTE].iter().any(|st| st.rejoin == last), "{:?}", visits.last());
    Some((map, visits))
}

/// coast-heart: the switch turned off anywhere on the clip way past 0x55's river: in 0x55's W part (right after the
/// wrap, X < `$50`: tile columns 10-13 are the river), on entering 0x54, 0x53, 0x52, 0x51, 0x50, 0x60 and 0x5f (on
/// its water). No walk leads back to the plain way from there (E is 0x55's W part, cut off by the river), so the
/// switch off waits for the rejoin (`past_wrap_in`; on 0x5f's water `stranded` holds it too): the clip way holds
/// through 0x51's and 0x60's wrap exits and the ladder clip, Link never turns back, and from 0x4f (the rejoin) on
/// the switch is off and he goes on by the shared steps (0x3f, 0x2f's heart cave, 0x3f) into the lake stretch's
/// plain way at 0x3e.  Turned off on entering 0x55 (on the dock side, before the wrap), the plain way takes over at
/// once (0x55 from N leaves E).
#[test]
fn the_switch_off_in_the_coast_stretch_holds_the_clip_way_past_the_river() {
    let Some((s, _)) = leg(true) else { return };
    let map = s.game_map().clone();
    let stretch = map.stretches[ROUTE].iter().find(|st| st.name == "coast-heart").unwrap().clone();
    let route = &map.routes[ROUTE];
    let lake = map.stretches[ROUTE].iter().find(|st| st.name == "lake-3e").unwrap();
    let shared: Vec<usize> = (stretch.rejoin..lake.clips.start).chain([lake.plain.start]).collect();
    assert_eq!(
        shared.iter().map(|&i| route[i].location()).collect::<Vec<_>>(),
        [ow(0x4f), ow(0x3f), ow(0x2f), MapLocation::new(0, 0x2f, true), ow(0x2f), ow(0x3f), ow(0x3e)]
    );
    let w_part = |s: &S| in_room(ow(0x55))(s) && s.state().link.position.x < 0x50;
    let rooms = [0x54u8, 0x53, 0x52, 0x51, 0x50, 0x60, 0x5f];
    let offs: Vec<(String, Box<dyn Fn(&S) -> bool>)> = std::iter::once(("0x55's W part".to_owned(), Box::new(w_part) as Box<dyn Fn(&S) -> bool>))
        .chain(rooms.iter().map(|&r| (format!("{}", ow(r)), Box::new(in_room(ow(r))) as Box<dyn Fn(&S) -> bool>)))
        .collect();
    for (name, off) in &offs {
        let (mut s, mut gc) = leg(true).unwrap();
        walk_by_objective(&mut s, &mut gc, off, 20_000);
        let at = gc.step().unwrap_or_else(|| panic!("{name}: no step"));
        assert!(stretch.clips.contains(&at), "{name}: step {at}");
        gc.set_clips_allowed(false);
        let visits = walk_by_objective(&mut s, &mut gc, in_room(ow(0x3e)), 20_000);
        let steps: Vec<Option<usize>> = visits.iter().map(|v| v.step).collect();
        let want: Vec<Option<usize>> = (at..stretch.clips.end).chain(shared.iter().copied()).map(Some).collect();
        assert_eq!(steps, want, "{name}: the clip way to 0x4f, then the shared steps ({visits:?})");
        assert!(!gc.clips_allowed() && !gc.clips_held(), "{name}: off from the rejoin on");
    }
    // Before the wrap (0x55 on the dock side, its parts step): the plain way at once, E along row 5.
    let (mut s, mut gc) = leg(true).unwrap();
    walk_by_objective(&mut s, &mut gc, in_room(ow(0x55)), 20_000);
    assert_eq!(gc.step(), Some(stretch.clips.start));
    gc.set_clips_allowed(false);
    let visits = walk_by_objective(&mut s, &mut gc, in_room(ow(0x4f)), 20_000);
    let steps: Vec<Option<usize>> = visits.iter().map(|v| v.step).collect();
    let want: Vec<Option<usize>> = stretch.plain.clone().chain([stretch.rejoin]).map(Some).collect();
    assert_eq!(steps, want, "the plain way from 0x55's dock side");
}

/// lake-3e: off in 0x3d, after 0x3e's wrap exit.
#[test]
fn the_switch_off_in_the_lake_stretch_leads_to_its_plain_way() {
    switched_off_inside(ow(0x3d), ow(0x1d));
}

/// mountain: off in 0x0d, before 0x0c's wrap exit.
#[test]
fn the_switch_off_in_the_mountain_stretch_leads_to_its_plain_way() {
    switched_off_inside(ow(0x0d), D5);
}
