//! Round 26: the route's leg to Dungeon 7 and Dungeon 7 (steps 205-243 of `d2first` since round 31, scouted as `d7` and
//! merged) played in the real game by the route's objective, from `5_14s_0` (the D5
//! triforce step) to the D7 triforce (the route's end): the agent's own actions (MOVE, BOMB, RECORDER, FOOD) through
//! the session's masks with the env's per-step rules mirrored (family `items:`/`clips:`, the route's recorder facings,
//! vetoed wraps), the cheat ledger settling after every action, every room change judged by `exit_is_correct`, and
//! per step what the step means (the walk's [`Tick`]s): the whirlwind's plays and facings at 0x0b and the arrival
//! read as the route's `entry: recorder` in 0x45, the shops' wares as targets and the purchases by inventory, the
//! Armos woken by MOVE from its side at 0x34, the pond drained at 0x42, the Grumble fed in 7:0x28, the recorder at
//! 7:0x1c (and never at 7:0x39), the candle of the cellar 7:0x4a and the triforce of 7:0x2b. Beside the walk: a wrong
//! ware is `missed`, a start with too few rupees is topped up and counted, the whirlwind's mask at 0x0b follows the
//! game's counter and the pieces held, and the Armos wakes to MOVE from either side.
//!
//! SETUP (RAM writes): monsters are removed every action (object slots 1-11) but the kept types (`KEEP`); nothing
//! else is written (the state's own 171 rupees, 0 bombs, 5 keys; bombs come from the ledger and the rooms) but where a
//! test says so. Walker aids (controller input): cave mouths, the Armos stand tile and a push block's spot are lined
//! up a frame at a time.
#![cfg(feature = "zcore")]

mod common;

use common::{Counted, load_named_state, open_core};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use zelda::actions::{compute_action_mask, food_lies, food_wanted, recorder_wanted};
use zelda::cheats::CheatLedger;
use zelda::clips::skip_available;
use zelda::enums::enemy::{DIGDOGGER_CHILD, DIGDOGGER2, DIGDOGGER3, STALFOS};
use zelda::executor::direction_buttons;
use zelda::game_map::Secret;
use zelda::objectives::{GameCompletion, Objective, ObjectiveKind, exit_is_correct, pond_shore, wraps_into_wrong_exits};
use zelda::enums::Tile;
use zelda::room::Target;
use zelda::whirlwind::{FACING_DOWN, FACING_UP, LEVEL_INDEX, SUMMONED};
use zelda::{Action, ActionKind, Direction, GameMap, GameView, LoadOptions, MapLocation, Session, Var, tables};

type S = Session<Counted>;

const ROUTE: &str = zelda::game_map::DEFAULT_ROUTE;
const OBJ_TYPE: u16 = 0x34F;

fn map() -> Arc<GameMap> {
    Arc::new(GameMap::builtin())
}

fn describe(s: &S) -> String {
    let st = s.state();
    format!(
        "{} mode {} at ({:#04x}, {:#04x}) offset {} dir {:?} hearts {}/{} rupees {} bombs {} keys {} frame {}",
        st.full_location(),
        st.mode(),
        st.link.position.x,
        st.link.position.y,
        st.var(Var::LinkGridOffset),
        st.link.direction,
        st.health(),
        st.max_health(),
        st.rupees(),
        st.bombs(),
        st.keys(),
        s.frame()
    )
}

/// A family's `items:` and `clips:` switches (specialists/<name>.yaml), as the env applies them.
#[derive(Clone, Debug, Default)]
struct Family {
    clips: bool,
    bombs: String,
    recorder: bool,
    food: bool,
    candle: bool,
}

fn family(name: &str) -> Family {
    let path = common::repo().join("specialists").join(format!("{name}.yaml"));
    let text = std::fs::read_to_string(&path).unwrap();
    let v: serde_yaml_ng::Value = serde_yaml_ng::from_str(&text).unwrap();
    let clips = match &v["clips"] {
        serde_yaml_ng::Value::Bool(b) => *b,
        serde_yaml_ng::Value::Mapping(m) => m.get("enabled").and_then(|e| e.as_bool()).unwrap_or(false),
        _ => false,
    };
    let items = &v["items"];
    let flag = |k: &str| items[k].as_bool().unwrap_or(false);
    Family {
        clips,
        bombs: items["bombs"].as_str().unwrap_or("tenth").to_owned(),
        recorder: flag("recorder"),
        food: flag("food"),
        candle: flag("candle"),
    }
}

/// Object types the walk keeps: the dock, the ladder, the rock wall, the whirlwind, a dungeon push block, the
/// Grumble, the woken Armos (until its stairs show), the big Digdogger (until the recorder), the pond's flute secret.
const KEEP: [u8; 9] = [0x61, 0x5F, 0x63, 0x2E, 0x68, 0x36, 0x1E, 0x38, 0x5E];
/// The Grumble Goriya's object type (a person, slot 1).
const OBJ_GRUMBLE: u8 = 0x36;

/// The room's walkable tiles (`.`), walls (`#`), the objective's targets (`T`) and Link (`L`).
fn grid(s: &S, objective: &Objective) -> String {
    let st = s.state();
    let room = st.room();
    let here = st.link.tile();
    let mut out = String::new();
    for r in 0..zelda::room::ROWS {
        for c in 0..zelda::room::COLS {
            let ch = if (c, r) == here {
                'L'
            } else if objective.targets.contains(&Target::Tile((c, r))) {
                'T'
            } else if room.is_tile_walkable(c, r) {
                '.'
            } else {
                '#'
            };
            out.push(ch);
        }
        out.push('\n');
    }
    out
}

/// Link's room has a game.yaml secret Armos (0x34's).
fn armos_room(s: &S) -> bool {
    let st = s.state();
    st.level() == 0
        && !st.full_location().in_cave
        && matches!(s.game_map().get(st.full_location()).and_then(|r| r.secret), Some(zelda::game_map::Secret::Armos(_)))
}

fn clear_room(s: &mut S, keep_extra: &[u8]) {
    let st = s.state();
    if (st.level() == 0 && st.full_location().in_cave) || !matches!(st.mode(), 5) {
        return;
    }
    let ram = st.ram().to_vec();
    let gone: Vec<u16> = (1..=11u16)
        .filter(|&slot| {
            let t = ram[(OBJ_TYPE + slot) as usize];
            t != 0 && !KEEP.contains(&t) && !keep_extra.contains(&t)
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

struct Walk {
    s: S,
    gc: GameCompletion,
    ledger: CheatLedger,
    families: BTreeMap<String, Family>,
    ticks: Vec<Tick>,
}

#[derive(Debug)]
struct Visit {
    room: MapLocation,
    frames: u64,
}

/// What the walk saw before one action: the objective, the mask, and the bytes the route's rules read.
#[derive(Clone, Debug)]
struct Tick {
    room: MapLocation,
    step: Option<usize>,
    kind: ObjectiveKind,
    targets: BTreeSet<Target>,
    facing: Direction,
    /// The session offers RECORDER; the route's facings for it (`GameCompletion::recorder_facings`).
    recorder: bool,
    recorder_facings: u8,
    /// The game state's own rule for the recorder in a dungeon (a big Digdogger that hasn't heard it).
    recorder_wanted: bool,
    /// The session offers FOOD (any facing); a Grumble waits for it (its text typed); placed food lies.
    food: bool,
    food_wanted: bool,
    food_lies: bool,
    /// In a Grumble's room (slot 1 type `$36`) or while food lies: the game state's mask on a copy of the RAM with a
    /// Stalfos poked into slot 5 offers BOOMERANG (with food lying in slot $0F, with the slot empty).
    boomerang_probe: Option<(bool, bool)>,
    /// The action is a MOVE into the room's secret Armos (`GameView::is_secret_armos_direction`).
    armos_move: Option<Direction>,
    action: Action,
    /// Object types in slots 1-11 (before the walk removes monsters).
    types: Vec<u8>,
    /// `TeleportingLevelIndex` `$523` and `IsWhirlwindSummoned` `$508`.
    index: u8,
    summoned: bool,
    /// The room's stairs tiles (an overworld secret's once it shows).
    stairs: Vec<Tile>,
    rupees: i32,
    /// The pond's shore tiles where the room has a recorder secret (`pond_shore`).
    shore: Vec<Tile>,
    /// Link holds the arrows, the food; the candle (`$65B`: 2 red); his health.
    arrows: bool,
    food_held: bool,
    candle: i32,
    hearts: f64,
}

impl Tick {
    fn tiles(&self) -> BTreeSet<Target> {
        self.stairs.iter().map(|&t| Target::Tile(t)).collect()
    }
}

fn ow(room: u8) -> MapLocation {
    MapLocation::new(0, room, false)
}

fn cave(level: u8, room: u8) -> MapLocation {
    MapLocation::new(level, room, true)
}

fn uw(room: u8) -> MapLocation {
    MapLocation::new(7, room, false)
}

/// The ticks of `room` on route step `step`.
fn at(ticks: &[Tick], room: MapLocation, step: usize) -> Vec<&Tick> {
    ticks.iter().filter(|t| t.room == room && t.step == Some(step)).collect()
}

impl Walk {
    fn family_of(&mut self) -> Family {
        let name = self
            .gc
            .step()
            .and_then(|i| self.s.game_map().routes[ROUTE][i].specialist.clone())
            .unwrap_or_else(|| "goriya".to_owned());
        self.families.entry(name.clone()).or_insert_with(|| family(&name)).clone()
    }

    /// The env's per-step masks: the family's items and clips, the route's recorder facings and wraps; the facings.
    fn masks(&mut self, objective: &Objective) -> u8 {
        let f = self.family_of();
        let state = self.s.state().clone();
        let map = self.s.game_map().clone();
        let reserve = self.gc.bomb_reserve();
        let bombs = match f.bombs.as_str() {
            "never" => false,
            _ if state.next_kill_arms_drop() => true,
            "tenth" => false,
            "required" => map.requires_bombs(state.full_location()),
            _ => map.requires_bombs(state.full_location()) || state.bombs() > i32::from(reserve),
        };
        let off: Vec<ActionKind> = [
            (bombs, ActionKind::Bomb),
            (true, ActionKind::Boomerang),
            (false, ActionKind::Arrow),
            (f.recorder, ActionKind::Recorder),
            (f.food, ActionKind::Food),
            (f.candle, ActionKind::Candle),
        ]
        .into_iter()
        .filter(|(on, _)| !on)
        .map(|(_, k)| k)
        .collect();
        self.s.set_disabled_kinds(&off);
        self.s.set_clips(f.clips);
        self.gc.set_clips(f.clips);
        let facings = self.gc.recorder_facings(self.s.state());
        self.s.set_recorder_facings(Some(facings));
        let wraps = wraps_into_wrong_exits(&map, self.s.state(), objective);
        self.s.set_vetoed_wraps(&wraps);
        facings
    }
}

fn next_action(s: &S, objective: &Objective, nudge: u32) -> Action {
    let state = s.state();
    let mask = s.mask();
    let wait = Action::parse("WAIT 4").unwrap();
    if !matches!(state.mode(), 5 | 9 | 11) || state.link.status != 0 {
        return wait;
    }
    let recorder = Action::new(ActionKind::Recorder, Direction::None, 0).unwrap();
    if mask.is_allowed(&recorder) {
        return recorder;
    }
    for d in [Direction::N, Direction::W, Direction::E, Direction::S] {
        let food = Action::new(ActionKind::Food, d, 0).unwrap();
        if mask.is_allowed(&food) {
            return food;
        }
    }
    let targets: Vec<Target> = objective.targets.iter().copied().collect();
    let room = state.room();
    // A locked door Link holds a key for is a way through (as the objective's own path search counts it).
    let locked: Vec<Direction> = [Direction::N, Direction::S, Direction::E, Direction::W]
        .into_iter()
        .filter(|&d| state.keys() > 0 && !state.full_location().in_cave && state.is_door_locked(d))
        .collect();
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
        // A bomb wall's spot: the bomb.
        for d in [Direction::N, Direction::S, Direction::W, Direction::E] {
            let bomb = Action::new(ActionKind::Bomb, d, 0).unwrap();
            if state.bomb_walls().iter().any(|w| w.wall_direction == d) && mask.is_allowed(&bomb) {
                return bomb;
            }
        }
        // A push block, or the secret Armos from its side (once: its Armos fading in on the square, the walk waits
        // for the stairs).
        if (1..=11u16).any(|slot| state.peek(OBJ_TYPE + slot) == zelda::enums::enemy::ARMOS) {
            return wait;
        }
        for d in [Direction::N, Direction::S, Direction::W, Direction::E] {
            if (state.is_push_block_direction(d) || state.is_secret_armos_direction(d)) && mask.is_allowed(&moving(d)) {
                return Action::parse(&format!("MOVE {} 16", d.name())).unwrap();
            }
        }
        if objective.kind == ObjectiveKind::Move && objective.targets.contains(&Target::Tile(here)) && room.exits.direction_of(here).is_none() {
            // Standing where the objective wants him (the whirlwind's row): wait.
            if objective.next_rooms.iter().all(|r| r.level == 0 && !r.in_cave) && state.level() == 0 && !state.full_location().in_cave {
                return wait;
            }
        }
        // On the pond's shore before its stairs show: the recorder was played (it is offered first); wait.
        let pond = matches!(s.game_map().get(state.full_location()).and_then(|r| r.secret), Some(zelda::game_map::Secret::Recorder(_)));
        if pond && !state.full_location().in_cave && state.room().stairs_tiles().is_empty() {
            return wait;
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
    if distance == 1
        && let Some(d) = room.exits.direction_of(here)
    {
        return moving(d);
    }
    wait
}

impl Walk {
    /// `5_14s_0` with a completion on the D5 triforce step (204) and the ledger begun;
    /// `rupees` overrides Link's 171 rupees first.
    fn start(rupees: Option<u8>) -> Option<Walk> {
        let core = open_core()?;
        let map = map();
        let mut s = Session::new(Counted(core, 0), map.clone());
        s.load(&load_named_state("5_14s_0"), &LoadOptions::default(), &mut ChaCha8Rng::seed_from_u64(0)).unwrap();
        if let Some(r) = rupees {
            s.apply_ram(&[(Var::Rupees, r)]);
        }
        let first = map.routes[ROUTE].iter().position(|s| s.level == 5 && s.exit == "triforce").unwrap();
        let gc = GameCompletion::new(map.clone(), ROUTE, Some(first)).unwrap();
        let mut w = Walk { s, gc, ledger: CheatLedger::default(), families: BTreeMap::new(), ticks: Vec::new() };
        w.ledger.begin(&mut w.s, &w.gc);
        Some(w)
    }

    /// The tick before `action` (`types`: the object types before the walk removed monsters).
    fn tick(&self, objective: &Objective, facings: u8, types: Vec<u8>, action: Action) -> Tick {
        let state = self.s.state();
        let map = self.s.game_map();
        let mask = self.s.mask();
        let offered = |m: &zelda::ActionMask, kind: ActionKind, dirs: &[Direction]| {
            dirs.iter().any(|&d| Action::new(kind, d, 0).is_ok_and(|a| m.is_allowed(&a)))
        };
        use Direction::*;
        let all = [N, S, W, E, NE, NW, SE, SW];
        let lies = food_lies(state);
        let grumble = state.peek(OBJ_TYPE + 1) == OBJ_GRUMBLE;
        let boomerang_probe = (grumble || lies).then(|| {
            let probe = |food: bool| {
                let mut ram = *state.ram();
                let slot = 5;
                ram[tables::OBJ_ID.0 + slot] = STALFOS;
                ram[tables::OBJ_POS_X.0 + slot] = 0x40;
                ram[tables::OBJ_POS_Y.0 + slot] = 0x9D;
                ram[tables::OBJ_SPAWN_STATE.0 + slot] = 0;
                ram[tables::OBJ_STATUS.0 + 0x0F] = if food { 0x80 } else { 0 };
                let view = GameView::new(&ram, 0, false, map.clone());
                offered(&compute_action_mask(&view), ActionKind::Boomerang, &all)
            };
            (probe(true), probe(false))
        });
        let armos_move = (action.kind == ActionKind::Move && state.is_secret_armos_direction(action.direction))
            .then_some(action.direction);
        let shore = match map.get(state.full_location()).and_then(|r| r.secret) {
            Some(Secret::Recorder(tile)) if !state.full_location().in_cave => pond_shore(state.room(), tile),
            _ => Vec::new(),
        };
        Tick {
            room: state.full_location(),
            step: self.gc.step(),
            kind: objective.kind,
            targets: objective.targets.clone(),
            facing: state.link.direction,
            recorder: mask.is_allowed(&recorder()),
            recorder_facings: facings,
            recorder_wanted: recorder_wanted(state),
            food: offered(mask, ActionKind::Food, &[N, S, W, E]),
            food_wanted: food_wanted(state),
            food_lies: lies,
            boomerang_probe,
            armos_move,
            action,
            types,
            index: state.peek(LEVEL_INDEX),
            summoned: state.peek(SUMMONED) != 0,
            stairs: if state.full_location().in_cave { Vec::new() } else { state.room().stairs_tiles() },
            shore,
            rupees: state.rupees(),
            arrows: state.arrows() != zelda::enums::ArrowKind::None,
            food_held: state.var(Var::Food) != 0,
            candle: state.var(Var::Candle),
            hearts: state.health(),
        }
    }

    /// Plays the route's objective until `done`, judging every room change, recording a [`Tick`] per action.
    fn run(&mut self, done: impl Fn(&S) -> bool, limit: usize) -> Vec<Visit> {
        let map = self.s.game_map().clone();
        let mut prev: Option<Arc<GameView>> = None;
        let mut objective = self.gc.objectives(None, self.s.state());
        let mut visits = vec![Visit { room: self.s.state().full_location(), frames: 0 }];
        let mut since = self.s.frame();
        let (mut last_tile, mut still) = (None, 0u32);
        for _ in 0..limit {
            let types: Vec<u8> = (1..=11u16).map(|slot| self.s.state().peek(OBJ_TYPE + slot)).collect();
            clear_room(&mut self.s, &[]);
            let state = self.s.state().clone();
            let before = objective.clone();
            let entered = prev.as_ref().is_some_and(|p| p.full_location() != state.full_location());
            objective = self.gc.objectives(prev.as_deref(), &state);
            if entered {
                let p = prev.as_ref().unwrap();
                // The triforce's warp out of the dungeon is the step's own end, not an exit.
                let warp = p.level() > 0 && state.level() == 0;
                let ok = warp || exit_is_correct(&map, p, &before, state.full_location());
                assert!(ok, "{} -> {}: not a correct exit ({before:?})", p.full_location(), state.full_location());
                visits.last_mut().unwrap().frames = self.s.frame() - since;
                since = self.s.frame();
                visits.push(Visit { room: state.full_location(), frames: 0 });
            }
            assert!(self.gc.missed().is_none(), "{:?} at {}", self.gc.missed(), describe(&self.s));
            if done(&self.s) {
                break;
            }
            let facings = self.masks(&objective);
            let tile = (state.full_location(), state.link.position.x, state.link.position.y);
            still = if last_tile == Some(tile) { still + 1 } else { 0 };
            last_tile = Some(tile);
            let here = state.link.tile();
            let near_mouth = objective.kind == ObjectiveKind::Cave
                && state.mode() == 5
                && state.level() == 0
                && !armos_room(&self.s)
                && objective.targets.iter().any(|t| matches!(*t, Target::Tile((c, r)) if (c - here.0).abs() <= 1 && (0..=2).contains(&(here.1 - r))));
            if near_mouth && state.link.position.x % 16 != 0 {
                let buttons = direction_buttons(if state.link.position.x % 16 < 8 { Direction::W } else { Direction::E });
                for _ in 0..16 {
                    if self.s.state().link.position.x % 16 == 0 && self.s.state().var(Var::LinkGridOffset) == 0 {
                        break;
                    }
                    self.s.press(buttons, 1);
                }
            }
            // Walker aid (controller input): the Armos wakes, and a block moves, only with Link pressing from the
            // stand tile's grid point; MOVEs of 4 or 8 frames (6 or 12 pixels) rarely end there, so within a tile of
            // it he is lined up a frame at a time.
            let armos = objective.kind == ObjectiveKind::Cave && armos_room(&self.s) && state.room().stairs_tiles().is_empty();
            let stand = state.mode() == 5 && (armos || (state.level() > 0 && state.push_block_pending() && state.peek(0x34D) != 0));
            if stand && let Some((tx, ty)) = objective.targets.iter().find_map(|t| match *t {
                Target::Tile((c, r)) if (c - here.0).abs() <= 1 && (r - here.1).abs() <= 1 => Some((c * 8, r * 8 + 0x3D)),
                _ => None,
            }) {
                // Beside the Armos the row first would press into the statues above and below the stand tile.
                for _ in 0..24 {
                    let p = self.s.state().link.position;
                    let vertical = (p.y != ty).then_some(if p.y < ty { Direction::S } else { Direction::N });
                    let horizontal = (p.x != tx).then_some(if p.x < tx { Direction::E } else { Direction::W });
                    let Some(d) = (if armos { horizontal.or(vertical) } else { vertical.or(horizontal) }) else {
                        break;
                    };
                    self.s.press(direction_buttons(d), 1);
                }
            }
            let action = next_action(&self.s, &objective, still / 3);
            assert!(still < 2000, "stuck at {}\n{}", describe(&self.s), grid(&self.s, &objective));
            self.ticks.push(self.tick(&objective, facings, types, action));
            let from = state.full_location();
            prev = Some(state);
            self.s.act(action).unwrap();
            self.ledger.after_action(&mut self.s, &self.gc, from);
        }
        assert!(done(&self.s), "the walk ended at {} ({} rooms)", describe(&self.s), visits.len());
        visits
    }
}

fn recorder() -> Action {
    Action::new(ActionKind::Recorder, Direction::None, 0).unwrap()
}

fn played(ticks: &[&Tick], kind: ActionKind) -> usize {
    ticks.iter().filter(|k| k.action.kind == kind).count()
}

/// The walk from the D5 triforce to the D7 triforce, every step asserted for what it means.
#[test]
fn the_d7_route_is_walked_from_the_d5_triforce_to_the_d7_triforce() {
    let Some(mut w) = Walk::start(None) else { return };
    let map = w.s.game_map().clone();
    let route = &map.routes[ROUTE];
    let visits = w.run(|s| s.state().has_triforce(7), 40_000);
    let t = &w.ticks;

    // 0x0b (step 205, `exit: recorder`): Link comes out of Dungeon 5 facing down with the counter at 0; the mask
    // offers RECORDER only facing down (or left) and the walk plays twice: 0 -> $FC (L5) -> $FB (L4, 0x45).
    assert_eq!((route[205].location(), route[205].exit.as_str()), (ow(0x0b), "recorder"));
    let whirl = at(t, ow(0x0b), 205);
    for k in &whirl {
        assert!(!k.recorder || k.recorder_facings & k.facing.value() != 0, "RECORDER offered facing {:?}: {k:?}", k.facing);
    }
    let plays: Vec<(u8, Direction, u8)> = whirl
        .iter()
        .filter(|k| k.action.kind == ActionKind::Recorder)
        .map(|k| (k.index, k.facing, k.recorder_facings))
        .collect();
    assert_eq!(plays, [(0x00, Direction::S, FACING_DOWN), (0xFC, Direction::S, FACING_DOWN)]);
    let waiting: Vec<_> = whirl.iter().filter(|k| k.index == 0xFB).collect();
    assert!(!waiting.is_empty() && waiting.iter().all(|k| k.summoned && k.recorder_facings == 0 && !k.recorder));
    // Set down in 0x45: the route's `entry: recorder` step, the room's first tick.
    let landing = t.iter().find(|k| k.room == ow(0x45)).unwrap();
    assert_eq!((landing.step, route[206].entry.as_str()), (Some(206), "recorder"));

    // The shops: the route's ware is the target until it is held (the purchase is read from the inventory), then
    // the cave's exit; Link pays its price from his own rupees (171 -> 91 -> 31), nothing cheated.
    for (room, step, price, rupees) in [(0x44, 214, 80, 91), (0x34, 217, 60, 31)] {
        let shop = at(t, cave(0, room), step);
        let ware = map.get(cave(0, room)).unwrap().treasure_tile.unwrap();
        assert_eq!(map.requires_rupees(cave(0, room)), price);
        let held = |k: &Tick| if room == 0x44 { k.arrows } else { k.food_held };
        let buying: Vec<_> = shop.iter().filter(|k| !held(k)).collect();
        assert!(!buying.is_empty() && shop.iter().any(|k| held(k)), "{room:#04x}: bought in the cave");
        for k in &buying {
            assert_eq!((k.kind, &k.targets), (ObjectiveKind::Treasure, &[Target::Tile(ware)].into()), "{room:#04x}");
        }
        assert!(shop.iter().filter(|k| held(k)).all(|k| k.kind == ObjectiveKind::Move), "{room:#04x}: then out");
        let out = t.iter().find(|k| k.room == ow(room) && k.step == Some(step + 1)).unwrap();
        assert!(held(out) && out.rupees == rupees, "{room:#04x}: {out:?}");
    }

    // 0x34 (step 216, `exit: cave` by the Armos): the targets are the statue's side tiles until the stairs show; a
    // MOVE E or W into the statue (offered by the mask: the statue is a wall) wakes it; then the stairs.
    let Some(Secret::Armos((c, r))) = map.get(ow(0x34)).unwrap().secret else { panic!("0x34's Armos") };
    let armos = at(t, ow(0x34), 216);
    let sides: BTreeSet<Target> = [Target::Tile((c - 2, r)), Target::Tile((c + 2, r))].into();
    let hidden: Vec<_> = armos.iter().filter(|k| k.stairs.is_empty()).collect();
    assert!(!hidden.is_empty() && hidden.iter().all(|k| k.targets == sides));
    let woke: Vec<Direction> = hidden.iter().filter_map(|k| k.armos_move).collect();
    assert!(!woke.is_empty() && woke.iter().all(|d| matches!(d, Direction::E | Direction::W)), "{woke:?}");
    let shown: Vec<_> = armos.iter().filter(|k| !k.stairs.is_empty()).collect();
    assert!(!shown.is_empty() && shown.iter().all(|k| k.stairs == [(c, r)] && k.targets == k.tiles()));
    assert_eq!(hidden[0].hearts, shown[0].hearts, "woken from the side: no heart lost");

    // 0x42 (step 223, `exit: cave` by the recorder): RECORDER offered while the pond is full, played once; the
    // targets are the pond's shore before, its stairs after the drain.
    let pond = at(t, ow(0x42), 223);
    assert_eq!(played(&pond, ActionKind::Recorder), 1);
    let play = pond.iter().position(|k| k.action.kind == ActionKind::Recorder).unwrap();
    assert!(pond[..=play].iter().all(|k| !k.shore.is_empty() && k.targets == k.shore.iter().map(|&s| Target::Tile(s)).collect()));
    let drained: Vec<_> = pond.iter().filter(|k| !k.stairs.is_empty()).collect();
    assert!(!drained.is_empty() && drained.iter().all(|k| !k.recorder && k.targets == k.tiles()));

    // 7:0x39 (step 228, boss-digdogger, crossed S to W): the game's rule wants the recorder (the big Digdogger), the
    // route never offers it.
    let crossed = at(t, uw(0x39), 228);
    assert!(crossed.iter().any(|k| k.recorder_wanted && k.types.iter().any(|&x| matches!(x, DIGDOGGER2 | DIGDOGGER3))));
    assert!(crossed.iter().all(|k| !k.recorder));

    // 7:0x28 (step 230): FOOD only once the Grumble's text has typed (its state 1), placed once; the Grumble takes it
    // within the FOOD action (no decision ever sees it lying) and is gone. BOOMERANG while food lies is withheld: the
    // room's own RAM with a monster poked in offers it with slot $0F empty, not with food lying there.
    let grumble = at(t, uw(0x28), 230);
    assert!(grumble.iter().all(|k| !k.food || k.food_wanted));
    let first = grumble.iter().position(|k| k.food).expect("FOOD offered in 7:0x28");
    assert!(grumble[..first].iter().any(|k| k.types.contains(&OBJ_GRUMBLE) && !k.food_wanted), "the text types first");
    assert_eq!(played(&grumble, ActionKind::Food), 1);
    let fed = grumble.iter().position(|k| k.action.kind == ActionKind::Food).unwrap();
    assert!(grumble.iter().all(|k| !k.food_lies));
    assert!(grumble[fed + 1..].iter().all(|k| !k.types.contains(&OBJ_GRUMBLE) && !k.food), "the Grumble is gone");
    assert!(grumble[first..=fed].iter().all(|k| k.boomerang_probe == Some((false, true))), "{:?}", grumble[first..=fed].to_vec());

    // 7:0x4a (step 234, `waypoint: candle`): the red candle taken in the cellar under 0x1a.
    assert!(at(t, cave(7, 0x4a), 234).iter().any(|k| k.candle == 0));
    let up = t.iter().find(|k| k.room == uw(0x1a) && k.step == Some(235)).unwrap();
    assert_eq!(up.candle, 2, "the red candle");

    // 7:0x1c (step 237, boss-digdogger fought): RECORDER offered while the big Digdogger hasn't heard it, played
    // once; the small Digdoggers after it.
    let dig = at(t, uw(0x1c), 237);
    assert_eq!(played(&dig, ActionKind::Recorder), 1);
    let play = dig.iter().position(|k| k.action.kind == ActionKind::Recorder).unwrap();
    assert!(dig[play].recorder && dig[play].types.iter().any(|&x| matches!(x, DIGDOGGER2 | DIGDOGGER3)));
    assert!(dig[play + 1..].iter().any(|k| k.types.contains(&DIGDOGGER_CHILD)), "the small Digdoggers");

    // 7:0x2b's triforce (step 243; the route runs on to Dungeon 6 since round 27, `d6.rs`): the last decision is the
    // treasure objective of step 243, whose pickup sets the level 7 piece (the walk's end) before the warp out; no
    // rupee or key cheated.
    let last = t.last().unwrap();
    assert_eq!((last.room, last.step, last.kind), (uw(0x2b), Some(243), ObjectiveKind::Treasure));
    assert_eq!(route[244].location(), ow(0x42), "step 244 starts in 0:0x42, out of Dungeon 7");
    assert!(t.iter().all(|k| k.room != ow(0x42) || k.step != Some(243)));
    assert!(w.s.state().has_triforce(7) && visits.iter().any(|v| v.room == uw(0x2b)));
    assert_eq!((w.ledger.cheated.rupees, w.ledger.cheated.keys), (0, 0));
}

/// Presses `buttons` a frame at a time until `until` (at most `frames`), following the objective every frame.
fn press_following(w: &mut Walk, buttons: u8, frames: u32, until: impl Fn(&GameView) -> bool) {
    for _ in 0..frames {
        let prev = w.s.state().clone();
        if until(&prev) {
            return;
        }
        w.s.press(buttons, 1);
        let state = w.s.state().clone();
        w.gc.objectives(Some(&prev), &state);
    }
    assert!(until(w.s.state()), "not reached in {frames} frames: {}", describe(&w.s));
}

/// 0x44's shop sells the bombs at X $78: bought instead of the arrow, the route is `missed`.
#[test]
fn a_wrong_ware_in_the_arrow_shop_is_missed() {
    let Some(mut w) = Walk::start(None) else { return };
    w.run(|s| s.state().full_location() == cave(0, 0x44), 40_000);
    let (rupees, bombs) = (w.s.state().rupees(), w.s.state().bombs());
    press_following(&mut w, direction_buttons(Direction::E), 300, |s| s.link.position.x == 0x78);
    press_following(&mut w, direction_buttons(Direction::N), 300, |s| s.rupees() < rupees);
    press_following(&mut w, 0, 300, |s| s.rupees() == rupees - 20);
    assert!(w.s.state().bombs() > bombs && w.s.state().arrows() == zelda::enums::ArrowKind::None);
    assert_eq!(w.gc.missed(), Some("bought another ware in (0, 0x44, cave), not the arrows"));
}

/// From 30 rupees: the arrow shop's cave tops them up to its price (80, 50 cheated), the food shop's from 0 to 60
/// (60 cheated); Link leaves the second with none, holding both wares.
#[test]
fn too_few_rupees_for_the_shops_are_topped_up_and_counted() {
    let Some(mut w) = Walk::start(Some(30)) else { return };
    w.run(|s| s.state().full_location() == ow(0x34) && s.state().var(Var::Food) != 0, 40_000);
    let state = w.s.state();
    assert!(state.arrows() != zelda::enums::ArrowKind::None && state.rupees() == 0, "{}", describe(&w.s));
    assert_eq!((w.ledger.used.rupees, w.ledger.cheated.rupees), (140, 110));
    let first = w.ticks.iter().find(|k| k.room == cave(0, 0x44) && k.kind == ObjectiveKind::Treasure).unwrap();
    assert_eq!(first.rupees, 80, "topped up in the cave");
}

/// 0x0b (the whirlwind step) out of Dungeon 5: for a counter (`$523`) and the pieces held (`$671`), RECORDER is
/// offered exactly in the facings whose play starts the shortest sequence to L4 (0x45), never without the L4 piece;
/// a play facing up from counter 1 leaves it at 2, one play (up) still to make.
#[test]
fn the_whirlwind_mask_follows_the_counter_and_the_pieces() {
    let Some(mut w) = Walk::start(None) else { return };
    w.run(|s| s.state().full_location() == ow(0x0b) && s.state().mode() == 5, 40_000);
    let set = |w: &mut Walk, index: u8, pieces: u8, facing: Direction| {
        zelda::Nes::poke(w.s.nes_mut(), LEVEL_INDEX, index);
        w.s.apply_ram(&[(Var::Triforce, pieces), (Var::LinkDirection, facing.value())]);
        let state = w.s.state().clone();
        let objective = w.gc.objectives(Some(&state), &state);
        w.masks(&objective)
    };
    let both = FACING_UP | FACING_DOWN;
    for (index, pieces, want) in [
        (0x00, 0x1F, FACING_DOWN), // pieces 1-5: 0 -> $FC (L5) -> $FB (L4)
        (0x01, 0x1F, FACING_UP),   // 2, 3 up (down: 0, 4, 3)
        (0x03, 0x1F, both),        // at L4 without a whirlwind: away and back
        (0x00, 0x0F, FACING_DOWN), // pieces 1-4: one play down (7, 6, 5, 4 skipped)
        (0x02, 0x0F, FACING_UP),   // one up; down 1, 0, 3
        (0x00, 0xFF, FACING_UP),   // all pieces: 1, 2, 3 up; down five
        (0x00, 0x17, 0),           // no L4 piece: never
    ] {
        for facing in [Direction::N, Direction::S, Direction::W, Direction::E] {
            let facings = set(&mut w, index, pieces, facing);
            assert_eq!(facings, want, "index {index:#04x} pieces {pieces:#04x}");
            let offered = w.s.mask().is_allowed(&recorder());
            assert_eq!(offered, want & facing.value() != 0, "index {index:#04x} pieces {pieces:#04x} facing {facing:?}");
        }
    }
    set(&mut w, 0x01, 0x1F, Direction::E);
    w.s.act(recorder()).unwrap();
    let state = w.s.state().clone();
    assert_eq!((state.peek(LEVEL_INDEX), state.peek(SUMMONED) != 0), (0x02, true));
    let objective = w.gc.objectives(Some(&state), &state);
    assert_eq!(w.masks(&objective), FACING_UP, "one more play, up");
}

/// 0x34's secret Armos: from its W side (`$30`, `$7D`) MOVE E, from its E side (`$50`, `$7D`) MOVE W is offered
/// though the statue is a wall, and wakes it: the stairs show on the statue's square, no heart lost; from above
/// (`$40`, `$6D`, where Down wakes it and it costs a heart) the mask offers no MOVE S into it.
#[test]
fn the_secret_armos_wakes_to_move_from_its_side() {
    for (x, facing) in [(0x30, Direction::E), (0x50, Direction::W)] {
        let Some(mut w) = Walk::start(None) else { return };
        w.run(|s| s.state().full_location() == ow(0x34) && s.state().mode() == 5, 40_000);
        let Some(Secret::Armos((c, r))) = w.s.game_map().get(ow(0x34)).unwrap().secret else { panic!() };
        clear_room(&mut w.s, &[]);
        let place = |w: &mut Walk, x: u8, y: u8, d: Direction| {
            w.s.apply_ram(&[(Var::LinkX, x), (Var::LinkY, y), (Var::LinkDirection, d.value()), (Var::LinkGridOffset, 0)]);
        };
        place(&mut w, 0x40, 0x6D, Direction::S);
        let above = w.s.state().clone();
        assert!(!above.can_link_move(Direction::S) && !above.is_secret_armos_direction(Direction::S));
        assert!(!w.s.mask().is_allowed(&Action::parse("MOVE S 16").unwrap()));
        place(&mut w, x, 0x7D, facing);
        let side = w.s.state().clone();
        let push = Action::new(ActionKind::Move, facing, 16).unwrap();
        assert!(!side.can_link_move(facing) && side.is_secret_armos_direction(facing), "{facing:?}");
        assert!(w.s.mask().is_allowed(&push), "{facing:?}: {}", w.s.mask());
        let hearts = side.health();
        w.s.act(push).unwrap();
        let mut woke = false;
        for _ in 0..40 {
            woke |= (1..=11u16).any(|slot| w.s.state().peek(OBJ_TYPE + slot) == 0x1E);
            if !w.s.state().room().stairs_tiles().is_empty() {
                break;
            }
            w.s.act(Action::parse("WAIT 4").unwrap()).unwrap();
        }
        assert!(woke, "{facing:?}: the Armos woke");
        assert_eq!(w.s.state().room().stairs_tiles(), [(c, r)], "{facing:?}: the stairs");
        assert_eq!(w.s.state().health(), hearts, "{facing:?}: no heart lost");
    }
}
