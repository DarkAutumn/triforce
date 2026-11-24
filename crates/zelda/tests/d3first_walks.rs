//! Round 40: the fresh generation's `d3first` (decisions 2026-11-23, "The fresh generation's Dungeon 3", "continued",
//! "3:0x69 is crossed by a line"; docs/experiments/round40.md "Route40") walked in the real game by the route's
//! objective: the agent's own actions (MOVE, CLIP, BOMB) through the session's masks with the env's per-step rules
//! mirrored (each family's `items: {bombs}` and `clips:`, a step's `bombs:`, the route's bomb reserve, vetoed wraps),
//! the cheat ledger settling after every action, every room change judged by `exit_is_correct`.
//! - The opening, from the sword cave's mouth to Level 3: the E screen wrap in 0:0x76 (game.yaml's d3first entry,
//!   `wrap_in`) and 0x75's wrap exit, with clips on; with clips off 0x76 is walked across, as before.
//! - Dungeon 3 from its entrance to the triforce on the new plan: 3:0x7b's key taken with its Zols alive, 3:0x6b
//!   crossed with its five Zols alive and its key left, 3:0x5b's first pass a crossing entered with the drop cycle
//!   and the kill streak at 0, the bomb reserve 4 from 3:0x59 to 0x5a, the keys of the route's ledger.
//! - 2:0x3e: the Moldorm fought for its key; 9:0x56: its key taken (both needed: `r40_route.py ledger`).
//! - 0:0x1e left E on Link's tile row 4 (Y $5D) and 0x1f crossed along it to the gap (round40.md "0x1e's row").
//!
//! SETUP (RAM writes): Link's hearts are filled before every action (the walk proves the way, not the fights); in a
//! room whose objective is a fight the monsters are removed every action (object slots 1-11 but the kept types), so
//! the fights are skipped and nothing dies: every other room keeps its monsters alive and the walk kills nothing (it
//! never swings the sword), and a crossing's bomb carriers are left alive: the walk aims at the exit alone. Walker
//! aid (controller input): a cave mouth is lined up a frame at a time.
#![cfg(feature = "zcore")]

mod common;

use common::{Counted, load_named_state, open_core};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::BTreeMap;
use std::sync::Arc;
use zelda::cheats::{CheatItem, CheatLedger};
use zelda::clips::skip_available;
use zelda::executor::direction_buttons;
use zelda::objectives::{
    GameCompletion, Objective, ObjectiveKind, Role, exit_is_correct, route_bomb_targets, wraps_into_wrong_exits,
};
use zelda::room::Target;
use zelda::{Action, ActionKind, Direction, GameMap, GameView, LoadOptions, MapLocation, Session, Var};

type S = Session<Counted>;

const ROUTE: &str = "d3first";
const OBJ_TYPE: u16 = 0x34F;
/// Object types the walk never removes: the dock, the ladder, the rock wall, the whirlwind, a dungeon push block.
const KEEP: [u8; 5] = [0x61, 0x5F, 0x63, 0x2E, 0x68];

/// The map d3first plays (its own game.yaml rooms over the shared ones).
fn map() -> Arc<GameMap> {
    Arc::new(GameMap::builtin()).route_map(ROUTE)
}

fn ow(room: u8) -> MapLocation {
    MapLocation::new(0, room, false)
}

fn uw(level: u8, room: u8) -> MapLocation {
    MapLocation::new(level, room, false)
}

fn describe(s: &S) -> String {
    let st = s.state();
    format!(
        "{} mode {} at ({:#04x}, {:#04x}) dir {:?} hearts {}/{} bombs {} keys {} frame {}",
        st.full_location(),
        st.mode(),
        st.link.position.x,
        st.link.position.y,
        st.link.direction,
        st.health(),
        st.max_health(),
        st.bombs(),
        st.keys(),
        s.frame()
    )
}

/// A family's `items: {bombs}` and `clips:` (its d3first task file where it has one, else its own), as the env
/// applies them.
#[derive(Clone, Debug)]
struct Family {
    clips: bool,
    bombs: String,
}

fn family(name: &str) -> Family {
    let own = common::repo().join("specialists").join(ROUTE).join(format!("{name}.yaml"));
    let path = if own.exists() { own } else { common::repo().join("specialists").join(format!("{name}.yaml")) };
    let v: serde_yaml_ng::Value = serde_yaml_ng::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let clips = match &v["clips"] {
        serde_yaml_ng::Value::Bool(b) => *b,
        serde_yaml_ng::Value::Mapping(m) => m.get("enabled").and_then(|e| e.as_bool()).unwrap_or(false),
        _ => false,
    };
    Family { clips, bombs: v["items"]["bombs"].as_str().unwrap_or("tenth").to_owned() }
}

/// One room visit: what Link held coming in and going out, the actions played (with Link's position before each).
#[derive(Clone, Debug)]
struct Visit {
    room: MapLocation,
    step: Option<usize>,
    keys_in: i32,
    bombs_in: i32,
    cycle_in: u8,
    streak_in: i32,
    reserve: u8,
    kinds: Vec<ObjectiveKind>,
    actions: Vec<(Action, i32, i32)>,
    keys_out: i32,
    bombs_out: i32,
    /// Link's tile row on the visit's last decision.
    row_out: i32,
}

struct Walk {
    s: S,
    gc: GameCompletion,
    ledger: CheatLedger,
    families: BTreeMap<String, Family>,
    clips: bool,
}

/// Fills Link's hearts (the walk proves the way, not the fights).
fn refill(s: &mut S) {
    let containers = (s.state().max_health() - 1).clamp(0, 15) as u8;
    s.apply_ram(&[(Var::HeartsAndContainers, (containers << 4) | containers), (Var::PartialHearts, 0xFF)]);
}

/// Removes the monsters of Link's room (object slots 1-11 but the kept types), in play only.
fn clear_room(s: &mut S) {
    let st = s.state();
    if (st.level() == 0 && st.full_location().in_cave) || st.mode() != 5 {
        return;
    }
    let ram = st.ram().to_vec();
    let gone: Vec<u16> = (1..=11u16)
        .filter(|&slot| {
            let t = ram[(OBJ_TYPE + slot) as usize];
            t != 0 && !KEEP.contains(&t)
        })
        .collect();
    if gone.is_empty() {
        return;
    }
    for slot in gone {
        zelda::Nes::poke(s.nes_mut(), OBJ_TYPE + slot, 0);
    }
    s.apply_ram(&[]);
}

impl Walk {
    /// The named d3first state, with a completion at the route step `step` (its first step in the state's room
    /// entered from `entry`) and the cheat ledger begun; `clips` the run-time switch.
    fn start(state: &str, room: MapLocation, entry: &str, clips: bool) -> Option<Walk> {
        Walk::start_with(state, room, entry, clips, &[])
    }

    /// `start` with RAM written after the load, before the route and the cheat ledger see the state.
    fn start_with(state: &str, room: MapLocation, entry: &str, clips: bool, ram: &[(Var, u8)]) -> Option<Walk> {
        let core = open_core()?;
        let map = map();
        let mut s = Session::new(Counted(core, 0), map.clone());
        s.load(&load_named_state(state), &LoadOptions::default(), &mut ChaCha8Rng::seed_from_u64(0)).unwrap();
        s.apply_ram(ram);
        let step = map.routes[ROUTE]
            .iter()
            .position(|st| st.location() == room && st.entry == entry)
            .unwrap_or_else(|| panic!("no {ROUTE} step in {room} from {entry}"));
        let mut gc = GameCompletion::new(map.clone(), ROUTE, Some(step)).unwrap();
        gc.set_clips(clips);
        let mut w = Walk { s, gc, ledger: CheatLedger::default(), families: BTreeMap::new(), clips };
        w.ledger.begin(&mut w.s, &w.gc);
        Some(w)
    }

    fn family_of(&mut self) -> Family {
        let name = self.gc.step().and_then(|i| self.s.game_map().routes[ROUTE][i].specialist.clone()).unwrap();
        self.families.entry(name.clone()).or_insert_with(|| family(&name)).clone()
    }

    /// The env's per-step masks: BOMB by the family's mode (a step's own `bombs:` first) and the route's reserve,
    /// BOOMERANG and the items off, CLIP by the family and the run-time switch, the wraps the objective vetoes, and in
    /// a `requires_bombs` room BOMB kept to the objective's wall while no bomb of Link's reaches it yet (the env's
    /// `ItemConfig::bomb_targets`, round 23's one bomb a wall: the walker bombs nothing but walls, and without it laid
    /// a second bomb in the first one's fuse).
    fn masks(&mut self, objective: &Objective) {
        let f = self.family_of();
        let state = self.s.state().clone();
        let map = self.s.game_map().clone();
        let step = self.gc.step().map(|i| map.routes[ROUTE][i].clone());
        let mode = step.and_then(|s| s.bombs).unwrap_or(f.bombs);
        let reserve = self.gc.bomb_reserve();
        let here = state.full_location();
        let bombs = match mode.as_str() {
            "never" => false,
            _ if state.next_kill_arms_drop() => true,
            "tenth" => false,
            "required" => map.requires_bombs(here),
            _ => map.requires_bombs(here) || state.bombs() > i32::from(reserve),
        };
        let off: Vec<ActionKind> = [
            (bombs, ActionKind::Bomb),
            (false, ActionKind::Boomerang),
            (false, ActionKind::Arrow),
            (false, ActionKind::Recorder),
            (false, ActionKind::Food),
            (false, ActionKind::Candle),
        ]
        .into_iter()
        .filter(|(on, _)| !on)
        .map(|(_, k)| k)
        .collect();
        self.s.set_disabled_kinds(&off);
        self.s.set_clips(f.clips && self.clips);
        let wraps = wraps_into_wrong_exits(&map, self.s.state(), objective);
        self.s.set_vetoed_wraps(&wraps);
        let targets = map.requires_bombs(here).then(|| route_bomb_targets(&map, self.s.state(), objective));
        self.s.set_bomb_targets(targets.as_deref());
    }

    /// The objective of `state` as the env makes it (round 40, Carrier40b, after `r40-gen` met `r40-carrier`): the
    /// route's bag tests told the cheated bombs in the bag, and the wall's cheated bomb following the objective
    /// (`wall_grant: objective`, `CheatLedger::follow_objective`; the env's `wall_is_next` without its drops, which
    /// the walk never takes). Without it the walk's ledger never handed a wall its bomb.
    fn objective(&mut self, prev: Option<&GameView>, state: &GameView) -> Objective {
        self.gc.set_cheated_bombs(self.ledger.outstanding(CheatItem::Bombs));
        let objective = self.gc.objectives(prev, state);
        let map = self.s.game_map().clone();
        let wall_next = !route_bomb_targets(&map, state, &objective).is_empty()
            && objective.labels.iter().any(|(_, role)| matches!(role, Role::BombWall { .. } | Role::Cave(_)));
        self.ledger.follow_objective(&mut self.s, &self.gc, wall_next);
        objective
    }

    /// Plays the route's objective until `done`, judging every room change; the visits in order.
    fn run(&mut self, done: impl Fn(&S) -> bool, limit: usize) -> Vec<Visit> {
        let map = self.s.game_map().clone();
        let mut prev: Option<Arc<GameView>> = None;
        let start = self.s.state().clone();
        let mut objective = self.objective(None, &start);
        let mut visits = vec![self.visit()];
        let mut last = None;
        let mut still = 0u32;
        for _ in 0..limit {
            refill(&mut self.s);
            let state = self.s.state().clone();
            let before = objective.clone();
            let entered = prev.as_ref().is_some_and(|p| p.full_location() != state.full_location());
            objective = self.objective(prev.as_deref(), &state);
            if entered {
                let p = prev.as_ref().unwrap();
                // The triforce's warp out of the dungeon is the step's own end, not an exit.
                let warp = p.level() > 0 && state.level() == 0;
                let ok = warp || exit_is_correct(&map, p, &before, state.full_location());
                assert!(ok, "{} -> {}: not a correct exit ({before:?})", p.full_location(), state.full_location());
                let v = visits.last_mut().unwrap();
                (v.keys_out, v.bombs_out) = (p.keys(), p.bombs());
                visits.push(self.visit());
            }
            assert!(self.gc.missed().is_none(), "{:?} at {}", self.gc.missed(), describe(&self.s));
            if done(&self.s) {
                break;
            }
            if objective.kind == ObjectiveKind::Fight && state.level() > 0 {
                clear_room(&mut self.s);
            }
            self.masks(&objective);
            let state = self.s.state().clone();
            let here = (state.full_location(), state.link.position.x, state.link.position.y);
            still = if last == Some(here) { still + 1 } else { 0 };
            last = Some(here);
            // Walker aid: an overworld cave mouth is entered from its tile's grid point.
            let tile = state.link.tile();
            let near_mouth = objective.kind == ObjectiveKind::Cave
                && state.mode() == 5
                && state.level() == 0
                && objective.targets.iter().any(|t| matches!(*t, Target::Tile((c, r)) if (c - tile.0).abs() <= 1 && (0..=2).contains(&(tile.1 - r))));
            if near_mouth && state.link.position.x % 16 != 0 {
                let buttons = direction_buttons(if state.link.position.x % 16 < 8 { Direction::W } else { Direction::E });
                for _ in 0..16 {
                    if self.s.state().link.position.x % 16 == 0 && self.s.state().var(Var::LinkGridOffset) == 0 {
                        break;
                    }
                    self.s.press(buttons, 1);
                }
            }
            let action = next_action(&self.s, &objective, still / 3);
            assert!(still < 2000, "stuck at {}: {:?} {:?}, mask {}, last {:?}", describe(&self.s), objective.kind,
                objective.targets, self.s.mask(), visits.last().unwrap().actions.iter().rev().take(8).collect::<Vec<_>>());
            let v = visits.last_mut().unwrap();
            v.kinds.push(objective.kind);
            v.row_out = state.link.tile().1;
            v.actions.push((action, state.link.position.x, state.link.position.y));
            let from = state.full_location();
            prev = Some(state);
            self.s.act(action).unwrap();
            self.ledger.after_action(&mut self.s, &self.gc, from);
        }
        assert!(done(&self.s), "the walk ended at {} ({} rooms)", describe(&self.s), visits.len());
        let st = self.s.state();
        let v = visits.last_mut().unwrap();
        (v.keys_out, v.bombs_out) = (st.keys(), st.bombs());
        visits
    }

    fn visit(&self) -> Visit {
        let st = self.s.state();
        Visit {
            room: st.full_location(),
            step: self.gc.step(),
            keys_in: st.keys(),
            bombs_in: st.bombs(),
            cycle_in: st.world_kill_cycle(),
            streak_in: st.kill_streak(),
            reserve: self.gc.bomb_reserve(),
            kinds: Vec::new(),
            actions: Vec::new(),
            keys_out: 0,
            bombs_out: 0,
            row_out: 0,
        }
    }
}

/// The objective's next action: a BOMB at a bomb wall's spot, a CLIP where a jump of the path search shortens the
/// way, else a MOVE down the wavefront (d7.rs's walker without the items).
fn next_action(s: &S, objective: &Objective, nudge: u32) -> Action {
    let state = s.state();
    let mask = s.mask();
    let wait = Action::parse("WAIT 4").unwrap();
    if !matches!(state.mode(), 5 | 9 | 11) || state.link.status != 0 {
        return wait;
    }
    // The walk kills nothing: a crossing's bomb carriers (the route's `carrier_targets`, `Objective::carriers`) stay
    // alive and the exit is its way.
    let carriers: Vec<Target> = objective
        .labels
        .iter()
        .filter(|(_, role)| objective.kind == ObjectiveKind::Move && matches!(role, Role::Monster { .. }))
        .map(|a| a.0)
        .collect();
    let targets: Vec<Target> = objective.targets.iter().copied().filter(|t| !carriers.contains(t)).collect();
    let room = state.room();
    // A locked door is a way through while Link holds a key in a dungeon, as the objective's own search counts it.
    let locked: Vec<Direction> = if state.keys() > 0 && state.level() > 0 && !state.full_location().in_cave {
        vec![Direction::N, Direction::S, Direction::E, Direction::W]
    } else {
        Vec::new()
    };
    let wavefront = room.wavefront(&targets, &[], &locked, objective.clips);
    let here = state.link.tile();
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
        let d = room.exits.direction_of(here).unwrap_or(Direction::N);
        return moving(d);
    }
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
        // A locked door the wavefront passes (Link holds a key) is walked into: the game opens it on the push.
        let door = !locked.is_empty() && state.is_door_locked(d);
        let walked = (room.can_move(here.0, here.1, d) || door) && mask.is_allowed(&moving(d));
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
    if distance == 1
        && let Some(d) = room.exits.direction_of(here)
    {
        return moving(d);
    }
    wait
}

fn visit_of(visits: &[Visit], room: MapLocation, nth: usize) -> &Visit {
    visits.iter().filter(|v| v.room == room).nth(nth).unwrap_or_else(|| panic!("no visit {nth} of {room}: {visits:#?}"))
}

fn clips_in(v: &Visit) -> Vec<(Action, i32, i32)> {
    v.actions.iter().filter(|(a, _, _)| a.kind == ActionKind::Clip).copied().collect()
}

/// From the sword cave's mouth (step 2) to the D3 triforce with clips on: the opening wraps the screen in 0x76 and
/// 0x75 and kills nothing; Dungeon 3 on the new plan, its fights skipped.
#[test]
fn the_opening_wraps_and_dungeon_3_runs_on_the_new_plan() {
    let Some(mut w) = Walk::start("d3first_0_77c_0", ow(0x77), "cave", true) else { return };
    let map = w.s.game_map().clone();
    let route = map.routes[ROUTE].clone();
    let step = |room: MapLocation, entry: &str| route.iter().position(|s| s.location() == room && s.entry == entry).unwrap();
    let visits = w.run(|s| s.state().has_triforce(3), 40_000);
    let rooms: Vec<MapLocation> = visits.iter().map(|v| v.room).collect();
    assert_eq!(rooms[..5], [ow(0x77), ow(0x76), ow(0x75), ow(0x74), uw(3, 0x7c)], "{rooms:?}");

    // 0x76 (ow-tektite, its four Tektites alive): CLIP E at the E edge Link came in by, the wrap, then one MOVE W.
    let v76 = visit_of(&visits, ow(0x76), 0);
    let clips = clips_in(v76);
    assert_eq!(clips.len(), 1, "one wrap in 0x76: {:?}", v76.actions);
    let (clip, x, _) = clips[0];
    assert_eq!(clip.direction, Direction::E);
    assert!(x >= 0xD0, "the wrap from the E edge, X {x:#04x}");
    let after = v76.actions.iter().position(|a| a.0 == clip).unwrap();
    assert!(v76.actions[..after].iter().all(|&(_, x, _)| x >= 0xD0), "no walk across before it: {:?}", v76.actions);
    assert!(v76.actions[after + 1..].iter().all(|&(a, x, _)| a.kind == ActionKind::Move && a.direction == Direction::W && x < 0x20));
    // 0x75: its wrap exit (CLIP E) into 0x74.
    let v75 = visit_of(&visits, ow(0x75), 0);
    assert_eq!(clips_in(v75).iter().map(|c| c.0.direction).collect::<Vec<_>>(), [Direction::E], "{:?}", v75.actions);

    // No kill before 3:0x5b: the drop cycle and the kill streak at 0 on every entry up to its first pass.
    let first_pass = visits.iter().position(|v| v.room == uw(3, 0x5b)).unwrap();
    for v in &visits[..=first_pass] {
        assert_eq!((v.cycle_in, v.streak_in), (0, 0), "{} entered after a kill", v.room);
    }
    // 3:0x7b: its key taken in the open, no fight; 3:0x6b crossed, its key left, no fight.
    let (v7b, v6b) = (visit_of(&visits, uw(3, 0x7b), 0), visit_of(&visits, uw(3, 0x6b), 0));
    assert_eq!((v7b.keys_in, v7b.keys_out), (0, 1));
    assert_eq!((v6b.keys_in, v6b.keys_out), (1, 1));
    for v in [v7b, v6b] {
        assert!(!v.kinds.contains(&ObjectiveKind::Fight), "{}: {:?}", v.room, v.kinds);
    }
    // 3:0x5b's first pass (S -> N): a crossing, its bombs left, no bomb laid. Link brings none of his own (the bag
    // shows the room's wall grant, `zelda::cheats`: a requires_bombs room's bomb is topped up on entering it).
    let v5b = &visits[first_pass];
    assert_eq!(v5b.step, Some(step(uw(3, 0x5b), "S")));
    assert!(v5b.bombs_in <= 1 && v5b.bombs_out == v5b.bombs_in, "{v5b:?}");
    assert!(v5b.actions.iter().all(|(a, _, _)| a.kind != ActionKind::Bomb));
    assert!(!v5b.kinds.contains(&ObjectiveKind::Fight), "{:?}", v5b.kinds);

    // The bomb reserve: 6 from the entrance to 0x49, 4 from 0x59's fight through 0x5a, 0 on 0x5b's last pass, 6 at
    // 0x5c, 0 at Manhandla, 6 in the triforce room.
    let reserves: Vec<(MapLocation, u8)> = visits.iter().filter(|v| v.room.level == 3).map(|v| (v.room, v.reserve)).collect();
    let want: Vec<(MapLocation, u8)> = [
        (0x7c, 6), (0x7b, 6), (0x6b, 6), (0x5b, 6), (0x4b, 6), (0x4a, 6), (0x49, 6), (0x59, 4), (0x69, 4),
        (0x0f, 4), (0x69, 4), (0x59, 4), (0x5a, 4), (0x5b, 0), (0x5c, 6), (0x4c, 6), (0x4d, 0), (0x3d, 6),
    ]
    .into_iter()
    .map(|(r, n)| (MapLocation::new(3, r, r == 0x0f), n))
    .collect();
    assert_eq!(reserves, want);
    // 0x4a's four bombs are Link's in 0x59 (no Darknut drop on this walk): the reserve keeps them all through 0x5a.
    let v59 = visit_of(&visits, uw(3, 0x59), 0);
    assert_eq!((v59.bombs_in, v59.bombs_out), (4, 4));
    assert_eq!(visit_of(&visits, uw(3, 0x5a), 0).bombs_out, 4);

    // The keys (r40_route.py ledger): 1 after 0x7b, 0x6b leaves its key, 0x4b +1 and its W door, 0x49 +1, 0x59's E
    // door; out of Dungeon 3 with 1.
    let keys: Vec<(MapLocation, i32)> = visits.iter().filter(|v| v.room.level == 3).map(|v| (v.room, v.keys_out)).collect();
    assert_eq!(keys.iter().find(|k| k.0 == uw(3, 0x4b)).unwrap().1, 1);
    assert_eq!(keys.iter().find(|k| k.0 == uw(3, 0x49)).unwrap().1, 2);
    assert_eq!(w.s.state().keys(), 1, "{keys:?}");
    assert_eq!(w.ledger.cheated.keys, 0);
}

/// With clips off the opening walks 0x76 across (no CLIP): the wrap is the way only with the switch on.
#[test]
fn with_clips_off_0x76_is_walked() {
    let Some(mut w) = Walk::start("d3first_0_76e_0", ow(0x76), "E", false) else { return };
    let visits = w.run(|s| s.state().full_location() == ow(0x75), 4_000);
    let v76 = visit_of(&visits, ow(0x76), 0);
    assert!(clips_in(v76).is_empty(), "{:?}", v76.actions);
    assert!(v76.actions.iter().any(|&(_, x, _)| x < 0x80), "walked across: {:?}", v76.actions);
}

/// 2:0x3e (step after 0x4e): the Moldorm is fought (the room's objective a fight) and its key taken.
#[test]
fn the_moldorm_is_fought_for_its_key() {
    let Some(mut w) = Walk::start("d3first_2_3es_0", uw(2, 0x3e), "S", true) else { return };
    let keys = w.s.state().keys();
    let visits = w.run(|s| s.state().full_location() == uw(2, 0x2e), 4_000);
    let v = visit_of(&visits, uw(2, 0x3e), 0);
    assert_eq!(v.kinds.first(), Some(&ObjectiveKind::Fight), "{:?}", v.kinds);
    assert_eq!(v.keys_out, keys + 1, "the Moldorm's key");
}

/// 9:0x56 (S -> W): its key taken, then out by its W bomb wall.
#[test]
fn nine_0x56_gives_its_key() {
    let Some(mut w) = Walk::start("d3first_9_56s_0", uw(9, 0x56), "S", true) else { return };
    let keys = w.s.state().keys();
    let visits = w.run(|s| s.state().full_location() == uw(9, 0x55), 4_000);
    assert_eq!(visit_of(&visits, uw(9, 0x56), 0).keys_out, keys + 1);
}

/// 0x1e's row (round40.md "0x1e's row and 0x1f's line"): d3first leaves 0:0x1e E on Link's tile row 4 (Y $5D), and
/// in 0x1f the objective's way runs along it: from the W edge E on row 4, every row-3 tile W of column 15 farther
/// from the exit than the row-4 tile below it (no step N pays before X $78), then up the gap. The walk goes on
/// through the gap into 0x0f with the Peahats alive. (The line's timing is play: the walker is not frame-exact.)
#[test]
fn the_peahat_row_leads_through_0x1f() {
    let Some(mut w) = Walk::start("d3first_0_1ew_0", ow(0x1e), "W", true) else { return };
    let visits = w.run(|s| s.state().full_location() == ow(0x1f) && s.state().mode() == 5, 4_000);
    assert_eq!(visit_of(&visits, ow(0x1e), 0).row_out, 4, "out of 0x1e on tile row 4: {:?}", visits[0].actions);
    let state = w.s.state().clone();
    assert_eq!((state.link.position.y, state.link.tile().1), (0x5D, 4), "in 0x1f on Y $5D");
    let objective = w.gc.objectives(None, &state);
    let targets: Vec<Target> = objective.targets.iter().copied().collect();
    let wavefront = state.room().wavefront(&targets, &[], &[], objective.clips);
    let at = |c: i32, r: i32| wavefront.get((c, r)).unwrap_or_else(|| panic!("({c}, {r}) off the way"));
    for c in 0..16 {
        assert_eq!(at(c, 4), at(c + 1, 4) + 1, "row 4 leads E at column {c}");
    }
    for c in 0..15 {
        assert!(wavefront.get((c, 3)).is_none_or(|d| d > at(c, 4)), "row 3 at column {c} against row 4's {}", at(c, 4));
    }
    let visits = w.run(|s| s.state().full_location() == ow(0x0f), 4_000);
    assert_eq!(visits.last().unwrap().room, ow(0x0f));
}

/// 3:0x5b's last pass (W -> E), entered with a full bag and with the reserve's 4 (States40, round40.md "The start
/// states, 0x5b's last pass and the first Darknut's drop"): the walk from `d3first_3_5bw_0` with the bag set, the
/// fights skipped as everywhere in the walk, to Manhandla's room. States40 recorded 7 out of 0x5b and 6 at
/// Manhandla's door either way (the room's bombs judged on entering, 8 a crossing). Carrier40b, the owner's ordering
/// (`bomb_rules: {wall_first_rooms}`, decisions 2026-11-24 "Dungeon 3's bombs"): with 8 the wall first (8 + 4
/// overflows the bag), then the fight for the four at 7, the four (8): 8 out, 7 at Manhandla's door. With 4 the
/// order as before, the fight, the four (8), the wall: 7 out, 6 at Manhandla, which the owner calls fine ("Leave that
/// room with 7 isn't a hard rule. 6 is fine.").
#[test]
fn the_last_pass_of_0x5b_with_8_and_with_4_bombs() {
    for (bombs, kinds_want, out, at_door) in [
        (8u8, vec![ObjectiveKind::Move, ObjectiveKind::Fight, ObjectiveKind::Treasure, ObjectiveKind::Move], 8, 7),
        (4, vec![ObjectiveKind::Fight, ObjectiveKind::Treasure, ObjectiveKind::Move], 7, 6),
    ] {
        let Some(mut w) = Walk::start_with("d3first_3_5bw_0", uw(3, 0x5b), "W", true, &[(Var::Bombs, bombs)]) else {
            return;
        };
        let visits = w.run(|s| s.state().full_location() == uw(3, 0x4d), 6_000);
        let v5b = visit_of(&visits, uw(3, 0x5b), 0);
        let kinds: Vec<ObjectiveKind> = v5b.kinds.iter().fold(Vec::new(), |mut k, &x| {
            if k.last() != Some(&x) {
                k.push(x);
            }
            k
        });
        let laid = v5b.actions.iter().filter(|a| a.0.kind == ActionKind::Bomb).count();
        assert_eq!(kinds, kinds_want, "in with {bombs}");
        assert_eq!((v5b.bombs_in, laid, v5b.bombs_out), (i32::from(bombs), 1, out), "0x5b in with {bombs}");
        assert_eq!(w.s.state().bombs(), at_door, "at Manhandla's door, 0x5b in with {bombs}: {visits:#?}");
    }
}
