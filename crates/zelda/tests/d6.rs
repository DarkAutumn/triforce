//! Round 27: the route's magic sword and Dungeon 6 (steps 248-280 of `d2first`, scouted as the route `d6` and merged)
//! played in the real game by the route's
//! objective, from `7_2bw_0` (the D7 triforce step) through the leg (0x42, 0x52, the stretch `woods`, the graveyard),
//! the gravestone of 0x21 and the magic sword's cave, 0x31, 0x32, 0x22 and Dungeon 6 to its triforce: the agent's own
//! actions (MOVE, CLIP, BOMB, ARROW) through the session's masks with the env's per-step rules mirrored (family
//! `items:`/`clips:`, the route's bomb and rupee reserves, ARROW's facings, the doorways kept out of, vetoed wraps),
//! the cheat ledger settling after every action, every room change judged by `exit_is_correct`, and per step what the
//! step means (the walk's [`Tick`]s): the wrap out of 0x51 (clips) or the Lost Woods (plain), the stand square under
//! the stone, the push by MOVE N that ends on the stairs, no flying Ghini anywhere, the sword as the cave's target
//! and `waypoint: sword`, the way out by column X `$20`, the key doors and keys of Dungeon 6, 0x28's one bomb, the
//! masked doorways, the tunnel's right ladder, Gohma by ARROW facing N, its heart container and the triforce. Beside
//! the walk: the cave's refusal at 11 containers is `missed`; a MOVE toward a gravestone ends before it; 0x29's key
//! door, 0x28 N and 0x2c S are not offered and a MOVE toward them ends at the mouth.
//!
//! SETUP (RAM writes): monsters are removed every action (object slots 1-11) but the kept types (`KEEP`: tile
//! objects, the ladder, Gohma); nothing else is written (the state's own 12 hearts, 37 rupees, 2 bombs, 2 keys) but
//! where a test says so. Walker aids (controller input): cave mouths, the stand square of the stone and of a push
//! block and Link's X under Gohma are lined up a frame at a time.
#![cfg(feature = "zcore")]

mod common;

use common::{Counted, load_named_state, open_core};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use zelda::cheats::CheatLedger;
use zelda::clips::skip_available;
use zelda::enums::SwordKind;
use zelda::enums::Tile;
use zelda::enums::enemy::{BLUE_WIZZROBE, FLYING_GHINI, RED_GOHMA, RED_WIZZROBE};
use zelda::executor::direction_buttons;
use zelda::game_map::Secret;
use zelda::objectives::{
    GameCompletion, Objective, ObjectiveKind, doorways_kept_out, exit_is_correct, gohma, gohma_stand,
    wraps_into_wrong_exits,
};
use zelda::room::Target;
use zelda::{Action, ActionKind, Direction, GameMap, GameView, LoadOptions, MapLocation, Session, Var};

type S = Session<Counted>;

const ROUTE: &str = zelda::game_map::DEFAULT_ROUTE;
const OBJ_TYPE: u16 = 0x34F;
/// `Gohma_ObjEyeState` (`$46B` + slot): 3 while the eye is fully open.
const GOHMA_EYE: u16 = 0x46B;
/// The first step of the stretch (0x42 out of Dungeon 7) and the route's length.
const FIRST: usize = 248;
const STEPS: usize = 281;

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
    arrows: bool,
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
        arrows: flag("arrows"),
        recorder: flag("recorder"),
        food: flag("food"),
        candle: flag("candle"),
    }
}

/// Object types the walk keeps: the dock, the ladder, the rock wall, the whirlwind, a dungeon push block, the pushed
/// gravestone, the flute secret, and Gohma (killed by the walk's own arrow). A flying Ghini is kept too: the walk
/// must never make one.
const KEEP: [u8; 10] = [0x61, 0x5F, 0x63, 0x2E, 0x68, 0x65, 0x5E, 0x33, RED_GOHMA, FLYING_GHINI];

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

fn clear_room(s: &mut S) {
    let st = s.state();
    if (st.level() == 0 && st.full_location().in_cave) || !matches!(st.mode(), 5) {
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
}

/// What the walk saw before one action: the objective, the mask, and what the route's rules read.
#[derive(Clone, Debug)]
struct Tick {
    room: MapLocation,
    step: Option<usize>,
    kind: ObjectiveKind,
    targets: BTreeSet<Target>,
    action: Action,
    /// Link's position, facing and tile.
    x: i32,
    y: i32,
    facing: Direction,
    /// The game mode (5: play).
    mode: u8,
    /// A Gohma lives in the room (`zelda::objectives::gohma`: made, not dying).
    gohma: bool,
    /// Object types in slots 1-11 (before the walk removes monsters).
    types: Vec<u8>,
    /// The room's stairs tiles (an overworld secret's once it shows).
    stairs: Vec<Tile>,
    /// The directions the session's mask offers MOVE, ARROW and BOMB in; whether it offers CANDLE and CLIP.
    moves: Vec<Direction>,
    arrows: Vec<Direction>,
    bombs: Vec<Direction>,
    candle: bool,
    /// The route's rules for the state: ARROW's facings, the doorways MOVE keeps out of, the reserves.
    arrow_facings: Option<u8>,
    kept_out: u8,
    bomb_reserve: u8,
    rupee_reserve: u8,
    /// The action is a MOVE N that pushes the secret gravestone.
    stone_push: bool,
    hearts: f64,
    max_hearts: i32,
    rupees: i32,
    bomb_count: i32,
    keys: i32,
    sword: SwordKind,
}

fn ow(room: u8) -> MapLocation {
    MapLocation::new(0, room, false)
}

fn cave(level: u8, room: u8) -> MapLocation {
    MapLocation::new(level, room, true)
}

fn uw(room: u8) -> MapLocation {
    MapLocation::new(6, room, false)
}

/// The ticks of `room` on route step `step`.
fn at(ticks: &[Tick], room: MapLocation, step: usize) -> Vec<&Tick> {
    ticks.iter().filter(|t| t.room == room && t.step == Some(step)).collect()
}

fn played(ticks: &[&Tick], kind: ActionKind) -> usize {
    ticks.iter().filter(|k| k.action.kind == kind).count()
}

impl Walk {
    fn family_of(&mut self) -> Family {
        let name = self
            .gc
            .step()
            .and_then(|i| self.s.game_map().routes[ROUTE][i].specialist.clone())
            .unwrap_or_else(|| "wizzrobe".to_owned());
        self.families.entry(name.clone()).or_insert_with(|| family(&name)).clone()
    }

    /// The env's per-step masks (`env::specialist_env::update_disabled_kinds`): the family's items and clips with
    /// the route's reserves, the bomb targets of a `required` family, ARROW's facings, the doorways kept out of and
    /// the vetoed wraps.
    fn masks(&mut self, objective: &Objective) {
        let f = self.family_of();
        let state = self.s.state().clone();
        let map = self.s.game_map().clone();
        let reserve = self.gc.bomb_reserve();
        let required = map.requires_bombs(state.full_location());
        let bombs = match f.bombs.as_str() {
            "never" => false,
            _ if state.next_kill_arms_drop() => true,
            "tenth" => false,
            "required" => required,
            _ => required || state.bombs() > i32::from(reserve),
        };
        let arrows = f.arrows && state.rupees() > i32::from(self.gc.rupee_reserve());
        let off: Vec<ActionKind> = [
            (bombs, ActionKind::Bomb),
            (true, ActionKind::Boomerang),
            (arrows, ActionKind::Arrow),
            (f.recorder, ActionKind::Recorder),
            (f.food, ActionKind::Food),
            (f.candle, ActionKind::Candle),
        ]
        .into_iter()
        .filter(|(on, _)| !on)
        .map(|(_, k)| k)
        .collect();
        self.s.set_disabled_kinds(&off);
        let limited = f.bombs == "required" && required && !state.next_kill_arms_drop();
        let targets = limited.then(|| zelda::objectives::route_bomb_targets(&map, &state, objective));
        self.s.set_bomb_targets(targets.as_deref());
        self.s.set_clips(f.clips);
        self.gc.set_clips(f.clips);
        let facings = self.gc.recorder_facings(self.s.state());
        self.s.set_recorder_facings(Some(facings));
        self.s.set_arrow_facings(self.gc.arrow_facings(&state));
        // Round 28: CANDLE's facings by the route (ow-moblin has `candle: true` for 0x6d's tree: nowhere here).
        self.s.set_candle_facings(self.gc.candle_facings(&state));
        self.s.set_vetoed_doorways(doorways_kept_out(&map, &state, objective));
        let wraps = if self.s.clips_effective() { wraps_into_wrong_exits(&map, self.s.state(), objective) } else { Vec::new() };
        self.s.set_vetoed_wraps(&wraps);
    }
}

fn next_action(s: &S, objective: &Objective, nudge: u32) -> Action {
    let state = s.state();
    let mask = s.mask();
    let wait = Action::parse("WAIT 4").unwrap();
    if !matches!(state.mode(), 5 | 9 | 11) || state.link.status != 0 {
        return wait;
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
        // The secret gravestone from below: one MOVE N of 32 frames, which ends on the stairs it uncovers.
        if state.is_secret_push_direction(Direction::N) {
            return Action::parse("MOVE N 32").unwrap();
        }
        // On the stone's stairs, or under it while it slides: the cave opens, the walk waits.
        if state.secret_push().is_some() && objective.kind == ObjectiveKind::Cave {
            return wait;
        }
        // A push block.
        for d in [Direction::N, Direction::S, Direction::W, Direction::E] {
            if state.is_push_block_direction(d) && mask.is_allowed(&moving(d)) {
                return Action::parse(&format!("MOVE {} 16", d.name())).unwrap();
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
    /// `7_2bw_0` (Link entering Dungeon 7's triforce room: 12 hearts, the white sword, 37 rupees, 2 bombs, 2 keys)
    /// with a completion of `d6` on the D7 triforce step (247) and the ledger begun.
    fn start() -> Option<Walk> {
        let core = open_core()?;
        let map = map();
        let mut s = Session::new(Counted(core, 0), map.clone());
        s.load(&load_named_state("7_2bw_0"), &LoadOptions::default(), &mut ChaCha8Rng::seed_from_u64(0)).unwrap();
        let first = map.routes[ROUTE].iter().position(|s| s.level == 7 && s.exit == "triforce").unwrap();
        assert_eq!(first, FIRST - 1);
        let gc = GameCompletion::new(map.clone(), ROUTE, Some(first)).unwrap();
        let mut w = Walk { s, gc, ledger: CheatLedger::default(), families: BTreeMap::new(), ticks: Vec::new() };
        w.ledger.begin(&mut w.s, &w.gc);
        Some(w)
    }

    /// The run-time clips switch off: the plain way through the stretch.
    fn clips_off(&mut self) {
        self.s.set_clips_allowed(false);
        self.gc.set_clips_allowed(false);
    }

    /// The tick before `action` (`types`: the object types before the walk removed monsters).
    fn tick(&self, objective: &Objective, types: Vec<u8>, action: Action) -> Tick {
        let state = self.s.state();
        let map = self.s.game_map();
        let mask = self.s.mask();
        use Direction::*;
        let offered = |kind: ActionKind| -> Vec<Direction> {
            [N, S, W, E, NE, NW, SE, SW]
                .into_iter()
                .filter(|&d| {
                    let duration = if kind == ActionKind::Move { 8 } else { 0 };
                    Action::new(kind, d, duration).is_ok_and(|a| mask.is_allowed(&a))
                })
                .collect()
        };
        Tick {
            room: state.full_location(),
            step: self.gc.step(),
            kind: objective.kind,
            targets: objective.targets.clone(),
            action,
            x: state.link.position.x,
            y: state.link.position.y,
            facing: state.link.direction,
            mode: state.mode(),
            gohma: gohma(state).is_some(),
            types,
            stairs: if state.full_location().in_cave { Vec::new() } else { state.room().stairs_tiles() },
            moves: offered(ActionKind::Move),
            arrows: offered(ActionKind::Arrow),
            bombs: offered(ActionKind::Bomb),
            candle: !offered(ActionKind::Candle).is_empty(),
            arrow_facings: self.gc.arrow_facings(state),
            kept_out: doorways_kept_out(map, state, objective),
            bomb_reserve: self.gc.bomb_reserve(),
            rupee_reserve: self.gc.rupee_reserve(),
            stone_push: action.kind == ActionKind::Move && state.is_secret_push_direction(action.direction),
            hearts: state.health(),
            max_hearts: state.max_health(),
            rupees: state.rupees(),
            bomb_count: state.bombs(),
            keys: state.keys(),
            sword: state.sword(),
        }
    }

    /// Walker aid (controller input): presses a frame at a time until Link stands on (`tx`, `ty`), the row first
    /// (`row_first`) or the column first.
    fn line_up(&mut self, (tx, ty): (i32, i32), row_first: bool) {
        for _ in 0..24 {
            let p = self.s.state().link.position;
            let vertical = (p.y != ty).then_some(if p.y < ty { Direction::S } else { Direction::N });
            let horizontal = (p.x != tx).then_some(if p.x < tx { Direction::E } else { Direction::W });
            let Some(d) = (if row_first { horizontal.or(vertical) } else { vertical.or(horizontal) }) else {
                break;
            };
            self.s.press(direction_buttons(d), 1);
        }
    }

    /// Gohma's script (d6-walks.md §6), by the mask: on the objective's stand tile Link's X is kept 2-4 pixels left
    /// of Gohma's a frame at a time (walker aid), and ARROW N is played when the mask offers it and the eye is fully
    /// open; else the walk goes to the stand tile. None when no Gohma lives here.
    fn gohma_action(&mut self, objective: &Objective) -> Option<Action> {
        let state = self.s.state().clone();
        let g = gohma(&state)?;
        let stand = gohma_stand(&state)?;
        assert_eq!(objective.targets, [Target::Tile(stand)].into(), "Gohma's stand tile is the target");
        if state.link.tile().1 != stand.1 || (state.link.tile().0 - stand.0).abs() > 2 || state.mode() != 5 {
            return None;
        }
        let slot = u16::from(g.index);
        for _ in 0..16 {
            let st = self.s.state();
            let Some(g) = gohma(st) else { break };
            let dx = st.link.position.x - g.position.x;
            if (-4..=-2).contains(&dx) {
                break;
            }
            self.s.press(direction_buttons(if dx < -4 { Direction::E } else { Direction::W }), 1);
        }
        let arrow = Action::new(ActionKind::Arrow, Direction::N, 0).unwrap();
        let open = self.s.state().peek(GOHMA_EYE + slot) == 3;
        Some(if open && self.s.mask().is_allowed(&arrow) { arrow } else { Action::parse("WAIT 4").unwrap() })
    }

    /// Plays the route's objective until `done`, judging every room change, recording a [`Tick`] per action.
    fn run(&mut self, done: impl Fn(&S) -> bool, limit: usize) -> Vec<Visit> {
        let map = self.s.game_map().clone();
        let mut prev: Option<Arc<GameView>> = None;
        let mut objective = self.gc.objectives(None, self.s.state());
        let mut visits = vec![Visit { room: self.s.state().full_location() }];
        let (mut last_tile, mut still) = (None, 0u32);
        for _ in 0..limit {
            let types: Vec<u8> = (1..=11u16).map(|slot| self.s.state().peek(OBJ_TYPE + slot)).collect();
            clear_room(&mut self.s);
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
                visits.push(Visit { room: state.full_location() });
            }
            assert!(self.gc.missed().is_none(), "{:?} at {}", self.gc.missed(), describe(&self.s));
            if done(&self.s) {
                break;
            }
            self.masks(&objective);
            let tile = (state.full_location(), state.link.position.x, state.link.position.y);
            still = if last_tile == Some(tile) { still + 1 } else { 0 };
            last_tile = Some(tile);
            let here = state.link.tile();
            let stone = state.secret_push().is_some();
            let near_mouth = objective.kind == ObjectiveKind::Cave
                && state.mode() == 5
                && state.level() == 0
                && !stone
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
            // Walker aid (controller input): the stone and a block move only with Link pressing from the stand tile's
            // grid point; MOVEs of 4 or 8 frames (6 or 12 pixels) rarely end there, so within a tile of it he is
            // lined up a frame at a time (the stone's row first: its column holds the stones above and below).
            let under_stone = stone && objective.kind == ObjectiveKind::Cave && state.room().stairs_tiles().is_empty();
            let block = state.level() > 0 && state.push_block_pending() && state.peek(0x34D) != 0;
            if state.mode() == 5
                && (under_stone || block)
                && !state.secret_push_sliding()
                && let Some(spot) = objective.targets.iter().find_map(|t| match *t {
                    Target::Tile((c, r)) if (c - here.0).abs() <= 1 && (r - here.1).abs() <= 1 => Some((c * 8, r * 8 + 0x3D)),
                    _ => None,
                })
            {
                self.line_up(spot, under_stone);
            }
            // A key door the objective leaves through (its room has no exit tiles until it opens): within a tile of
            // its mouth Link is lined up (walker aid) and walks into it, which spends the key.
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
            let action = if let Some((d, mouth)) = key_door {
                self.line_up(mouth, matches!(d, Direction::N | Direction::S));
                Action::new(ActionKind::Move, d, 8).unwrap()
            } else if let Some(action) = self.gohma_action(&objective) {
                action
            } else {
                let mut action = next_action(&self.s, &objective, still / 3);
                // Walker aid (controller input), the graveyard only: its free rows and columns are one square wide
                // between the stones, and a turn off the 8-pixel grid snaps Link to a grid point of the game's
                // choosing. Before a turn he is lined up on the grid point of the tile the walk decided on, a frame
                // at a time along the line he walks, and the walk decides again there.
                let graveyard = map.get(state.full_location()).and_then(|r| r.specialist.as_deref()) == Some("ow-ghini");
                let facing = state.link.direction;
                let turn = action.kind == ActionKind::Move
                    && facing.is_cardinal()
                    && action.direction != facing
                    && action.direction != facing.opposite();
                if graveyard && state.mode() == 5 && turn && state.var(Var::LinkGridOffset) != 0 {
                    self.line_up((here.0 * 8, here.1 * 8 + 0x3D), matches!(facing, Direction::W | Direction::E));
                    action = next_action(&self.s, &objective, still / 3);
                }
                action
            };
            assert!(
                still < 2000,
                "stuck at {} playing {action} ({:?} {:?}) mask {}\n{}",
                describe(&self.s),
                objective.kind,
                objective.targets,
                self.s.mask(),
                grid(&self.s, &objective)
            );
            self.ticks.push(self.tick(&objective, types, action));
            let from = state.full_location();
            prev = Some(state);
            self.s.act(action).unwrap();
            self.ledger.after_action(&mut self.s, &self.gc, from);
        }
        assert!(done(&self.s), "the walk ended at {} ({} rooms)", describe(&self.s), visits.len());
        visits
    }
}

/// The rooms of `visits` from the first visit of `from` on, as (level, room id, in a sub-room).
fn rooms_from(visits: &[Visit], from: MapLocation) -> Vec<(u8, u8, bool)> {
    let start = visits.iter().position(|v| v.room == from).expect("the room is visited");
    visits[start..].iter().map(|v| (v.room.level, v.room.value, v.room.in_cave)).collect()
}

/// The walk from the D7 triforce to the D6 triforce, clips on, every step asserted for what it means.
#[test]
fn the_d6_route_is_walked_from_the_d7_triforce_to_the_d6_triforce() {
    let Some(mut w) = Walk::start() else { return };
    let map = w.s.game_map().clone();
    let route = &map.routes[ROUTE];
    assert_eq!(route.len(), STEPS);
    let visits = w.run(|s| s.state().has_triforce(6), 60_000);
    let t = &w.ticks;

    // The rooms in order, clips on: the wrap out of 0x51 into 0x50, the graveyard, 0x21 and its cave, Dungeon 6 with
    // its tunnel; every change was a correct exit (`run`).
    let o = |r: u8| (0, r, false);
    let u = |r: u8| (6, r, false);
    let rooms = rooms_from(&visits, ow(0x42));
    let expected = vec![
        o(0x42), o(0x52), o(0x51), o(0x50), o(0x40), o(0x41), o(0x31), o(0x21), (0, 0x21, true), o(0x21), o(0x31),
        o(0x32), o(0x22), u(0x79), u(0x78), u(0x68), u(0x58), u(0x48), u(0x38), u(0x28), u(0x29), u(0x39), u(0x3a),
        (6, 0x08, true), u(0x1d), u(0x2d), u(0x2c), u(0x1c), u(0x0c),
        // The piece ends the level: the game sets Link down at Dungeon 6's entrance.
        o(0x22),
    ];
    assert_eq!(rooms, expected);

    // 0x42 (step 248, `entry: cave`): the game sets Link down at Dungeon 7's entrance with its piece and 12 full
    // hearts; the bomb reserve 2 from here.
    let out = t.iter().find(|k| k.room == ow(0x42)).unwrap();
    assert_eq!((out.step, route[FIRST].entry.as_str(), out.bomb_reserve), (Some(FIRST), "cave", 2));
    assert_eq!((out.hearts, out.max_hearts, out.sword), (12.0, 12, SwordKind::White));

    // The stretch `woods`, clip way: one CLIP E out of 0x51's W edge (step 250), 0x50 from E (step 251).
    let wrap = at(t, ow(0x51), 250);
    assert_eq!(played(&wrap, ActionKind::Clip), 1, "{:?}", wrap.iter().map(|k| k.action).collect::<Vec<_>>());
    assert!(t.iter().any(|k| k.room == ow(0x50) && k.step == Some(251)));
    assert!(t.iter().all(|k| !matches!(k.step, Some(252..=255))), "no plain-way step");

    // The graveyard (steps 256-262): no flying Ghini is ever made, in any of its rooms.
    for k in t.iter().filter(|k| [ow(0x40), ow(0x41), ow(0x31), ow(0x21)].contains(&k.room)) {
        assert!(!k.types.contains(&FLYING_GHINI), "a flying Ghini in {}: {k:?}", k.room);
    }

    // 0x21 (step 259, `exit: cave` under the gravestone): the target is the square under the stone until the stairs
    // show; one MOVE N from there pushes it; then the stairs are the target and Link stands on them (the MOVE ended
    // there); no heart is lost.
    let Some(Secret::Push((c, r))) = map.get(ow(0x21)).unwrap().secret else { panic!("0x21's gravestone") };
    let grave = at(t, ow(0x21), 259);
    let hidden: Vec<_> = grave.iter().filter(|k| k.stairs.is_empty()).collect();
    assert!(!hidden.is_empty());
    for k in &hidden {
        assert_eq!((k.kind, &k.targets), (ObjectiveKind::Cave, &[Target::Tile((c, r + 2))].into()));
    }
    let pushes: Vec<_> = grave.iter().filter(|k| k.stone_push).collect();
    assert_eq!(pushes.len(), 1, "one push");
    assert_eq!((pushes[0].x, pushes[0].y, pushes[0].action), (c * 8, (r + 2) * 8 + 0x3D, Action::parse("MOVE N 32").unwrap()));
    let shown: Vec<_> = grave.iter().filter(|k| !k.stairs.is_empty()).collect();
    for k in &shown {
        assert_eq!((k.stairs.as_slice(), &k.targets), (&[(c, r)][..], &[Target::Tile((c, r))].into()));
        assert_eq!((k.x, k.y), (c * 8, r * 8 + 0x3D), "on the stairs square");
    }
    assert_eq!(grave.last().unwrap().hearts, 12.0);

    // The cave (step 260, `waypoint: sword`): the middle ware is the target until Link holds the magic sword, then
    // the way out.
    let sword = at(t, cave(0, 0x21), 260);
    let before: Vec<_> = sword.iter().filter(|k| k.sword == SwordKind::White).collect();
    assert!(!before.is_empty() && before.iter().all(|k| k.kind == ObjectiveKind::Treasure));
    let after: Vec<_> = sword.iter().filter(|k| k.sword == SwordKind::Magic).collect();
    assert!(!after.is_empty() && after.iter().all(|k| k.kind == ObjectiveKind::Move));

    // Out of the cave (step 261): the stone is back (no stairs), and the way S keeps off the column under the exit
    // where the room's Ghini is made (game.yaml `path_costs`): W along the row Y $7D, and below it Link is at X
    // $20-$28.
    let back = at(t, ow(0x21), 261);
    assert!(!back.is_empty() && back.iter().all(|k| k.stairs.is_empty() && k.sword == SwordKind::Magic));
    assert!(back.iter().all(|k| !k.stone_push), "no second push");
    assert_eq!((back[0].x, back[0].y), (0x50, 0x6D), "the cave's exit spot");
    let below: Vec<_> = back.iter().filter(|k| k.mode == 5 && k.y > 0x8D).collect();
    assert!(!below.is_empty() && below.iter().all(|k| k.x <= 0x28), "{:?}", back.iter().map(|k| (k.x, k.y)).collect::<Vec<_>>());

    // 0x22 (step 264): into Dungeon 6 by its open entrance.
    assert!(t.iter().any(|k| k.room == ow(0x22) && k.step == Some(264) && k.kind == ObjectiveKind::Cave));

    // Dungeon 6. Keys: 2 in, 0x79 W -1, 0x58 +1, 0x2d +1, 0x2c N -1; bombs: 2 in, one at 0x28's wall; nothing
    // cheated.
    let first = |room: u8| t.iter().find(|k| k.room == uw(room)).unwrap_or_else(|| panic!("6:{room:#04x}"));
    let ledger: Vec<(i32, i32)> = [0x79, 0x78, 0x48, 0x28, 0x29, 0x2c, 0x1c].into_iter().map(|r| (first(r).keys, first(r).bomb_count)).collect();
    assert_eq!(ledger, [(2, 2), (1, 2), (2, 2), (2, 2), (2, 1), (3, 1), (2, 1)]);
    assert_eq!(w.ledger.cheated.bombs + w.ledger.cheated.keys + w.ledger.cheated.rupees, 0);
    // Reserves: bombs 2 through 0x28, 1 from 0x29; rupees 2, and 1 from Gohma's room.
    assert_eq!((first(0x28).bomb_reserve, first(0x29).bomb_reserve, first(0x2c).rupee_reserve, first(0x1c).rupee_reserve), (2, 1, 2, 1));

    // CANDLE is never offered (ow-moblin's is kept to the tree of 0x6d, round 28), nor ARROW outside Gohma's room.
    assert!(t.iter().all(|k| !k.candle));
    assert!(t.iter().all(|k| k.arrows.is_empty() || k.room == uw(0x1c)));

    // The Wizzrobe rooms were full when Link came in (the walk removes monsters; the fights are the models').
    for room in [0x78, 0x38, 0x28, 0x29, 0x3a, 0x2c] {
        let robes = t.iter().filter(|k| k.room == uw(room)).any(|k| k.types.iter().any(|x| matches!(*x, BLUE_WIZZROBE | RED_WIZZROBE)));
        assert!(robes, "6:{room:#04x}: its Wizzrobes");
    }

    // 0x28 (step 271, wizzrobe, `required`): BOMB is offered only where the bomb lands in reach of the E wall (the
    // squares before it, X $B0 on; the round 22 mask), played once from the wall's spot facing E, and one opens it.
    let wall = at(t, uw(0x28), 271);
    assert_eq!(played(&wall, ActionKind::Bomb), 1);
    let offered: Vec<_> = wall.iter().filter(|k| !k.bombs.is_empty()).map(|k| (k.x, k.y, k.bombs.clone())).collect();
    assert!(!offered.is_empty() && offered.iter().all(|(x, _, _)| *x >= 0xB0), "BOMB only by the E wall: {offered:?}");
    let bomb = wall.iter().find(|k| k.action.kind == ActionKind::Bomb).unwrap();
    assert_eq!((bomb.action.direction, bomb.x), (Direction::E, 0xD0));
    assert!((bomb.y - 0x8D).abs() <= 4, "on the door row's tile: Y {:#04x}", bomb.y);
    // Its N doorway (Gleeok's room) is kept out of; 0x29's N key door and 0x2c's S doorway too, 0x2c's N key door
    // (the route's) is not.
    assert!(wall.iter().all(|k| k.kept_out == Direction::N.value()));
    assert!(at(t, uw(0x29), 272).iter().all(|k| k.kept_out == Direction::N.value()));
    assert!(at(t, uw(0x2c), 278).iter().all(|k| k.kept_out == Direction::S.value()));
    assert!(t.iter().filter(|k| ![uw(0x28), uw(0x29), uw(0x2c)].contains(&k.room)).all(|k| k.kept_out == 0));

    // 0x38 (step 270) and 0x3a (step 274): the block's push spots are the targets once the room is empty, the push
    // opens 0x38's N shutter and shows 0x3a's stairs.
    for (room, step) in [(0x38, 270), (0x3a, 274)] {
        let ticks = at(t, uw(room), step);
        assert!(ticks.iter().any(|k| k.action == Action::parse("MOVE N 16").unwrap() || k.action.to_string().ends_with(" 16")), "6:{room:#04x}: the push");
    }
    assert!(at(t, uw(0x3a), 274).iter().any(|k| !k.stairs.is_empty()), "0x3a's stairs");

    // The tunnel (step 275): in by the left ladder, out by the right one into 0x1d (the room after it: `rooms`).
    let tunnel = at(t, cave(6, 0x08), 275);
    assert!(tunnel.first().unwrap().x < 0x80 && tunnel.last().unwrap().x >= 0x80);

    // Gohma (step 279, boss-gohma): ARROW is offered only facing N, while Gohma lives and the route's rule says so;
    // the stand tile under it is the target; the kill by arrows paid from Link's rupees above the reserve; then the
    // heart container (13 hearts) and the N shutter.
    let boss = at(t, uw(0x1c), 279);
    let alive: Vec<_> = boss.iter().filter(|k| k.gohma).collect();
    assert!(!alive.is_empty() && alive.iter().all(|k| k.types.contains(&RED_GOHMA)));
    for k in &alive {
        assert_eq!(k.arrow_facings, Some(Direction::N.value()));
        assert!(k.arrows.is_empty() || k.arrows == [Direction::N], "ARROW only N: {:?}", k.arrows);
        assert_eq!(k.kind, ObjectiveKind::Fight, "{k:?}");
    }
    assert!(alive.iter().any(|k| k.arrows == [Direction::N] && k.facing != Direction::N), "offered before the turn up");
    let shots = played(&boss, ActionKind::Arrow);
    assert!((1..=4).contains(&shots), "{shots} arrows");
    let rupees_in = boss.first().unwrap().rupees;
    let dead: Vec<_> = boss.iter().filter(|k| !k.gohma && k.max_hearts == 12).collect();
    assert!(dead.iter().any(|k| k.kind == ObjectiveKind::Treasure), "the heart container is the target");
    assert!(boss.iter().filter(|k| !k.gohma).all(|k| k.arrows.is_empty() && k.arrow_facings.is_none()), "no ARROW once it is dead");
    let last = boss.last().unwrap();
    assert_eq!((last.max_hearts, last.rupees), (13, rupees_in - shots as i32));

    // The route's end: 0x0c's triforce on step 280.
    let end = t.last().unwrap();
    assert_eq!((end.room, end.step, end.kind), (uw(0x0c), Some(STEPS - 1), ObjectiveKind::Treasure));
    assert!(w.s.state().has_triforce(6));
    // (`--nocapture`: the walk's numbers for the round document.)
    eprintln!(
        "d6 walk: {} actions, {} frames, {shots} arrows, {} hearts of {} at the end, {} rupees, {} bombs, {} keys",
        t.len(),
        w.s.frame(),
        w.s.state().health(),
        w.s.state().max_health(),
        w.s.state().rupees(),
        w.s.state().bombs(),
        w.s.state().keys()
    );
}

/// Clips off: the plain way of the stretch `woods` (0x51 S, the Lost Woods four times, 0x60 N, 0x50 from S) to the
/// graveyard, steps 252-255.
#[test]
fn the_woods_stretch_is_walked_the_plain_way() {
    let Some(mut w) = Walk::start() else { return };
    w.clips_off();
    let visits = w.run(|s| s.state().full_location() == ow(0x40) && s.state().mode() == 5, 60_000);
    let rooms: Vec<u8> = rooms_from(&visits, ow(0x42)).into_iter().map(|r| r.1).collect();
    // The Lost Woods come back three times (the same room: no room change is seen) before the fourth exit leads on.
    assert_eq!(rooms, [0x42, 0x52, 0x51, 0x61, 0x60, 0x50, 0x40]);
    let mut turns: Vec<&BTreeSet<Target>> = at(&w.ticks, ow(0x61), 253).iter().map(|k| &k.targets).collect();
    turns.dedup();
    assert_eq!(turns.len(), 4, "the maze's four exits in turn (N, W, S, W): {turns:?}");
    let t = &w.ticks;
    assert!(t.iter().all(|k| k.action.kind != ActionKind::Clip));
    for (room, step) in [(0x51, 252), (0x61, 253), (0x60, 254), (0x50, 255)] {
        assert!(t.iter().any(|k| k.room == ow(room) && k.step == Some(step)), "0x{room:02x} on step {step}");
    }
    assert!(t.iter().all(|k| !matches!(k.step, Some(250 | 251))), "no clip-way step");
}

/// Walks to `room` (Link playing in it) and returns the walk.
fn walk_to(room: MapLocation) -> Option<Walk> {
    let mut w = Walk::start()?;
    w.run(|s| s.state().full_location() == room && matches!(s.state().mode(), 5 | 11), 60_000);
    Some(w)
}

fn place(w: &mut Walk, x: u8, y: u8, d: Direction) {
    let ram = [(Var::LinkX, x), (Var::LinkY, y), (Var::LinkDirection, d.value()), (Var::LinkGridOffset, 0), (Var::DoorwayDir, 0)];
    w.s.apply_ram(&ram);
}

fn flying_ghinis(s: &S) -> usize {
    (1..=11u16).filter(|&slot| s.state().peek(OBJ_TYPE + slot) == FLYING_GHINI).count()
}

/// The cave under 0x21 refuses a Link with 11 heart containers (the old man asks 12): the route is `missed`.
#[test]
fn the_sword_cave_refuses_eleven_containers_and_the_route_is_missed() {
    let Some(mut w) = walk_to(cave(0, 0x21)) else { return };
    assert!(w.gc.missed().is_none());
    w.s.apply_ram(&[(Var::HeartsAndContainers, 0xAA)]);
    let state = w.s.state().clone();
    let objective = w.gc.objectives(Some(&state), &state);
    assert_eq!(w.gc.missed(), Some("(0, 0x21, cave) refuses the sword: 11 heart containers of 12"));
    assert!(objective.next_rooms.is_empty() && objective.targets.is_empty());
}

/// A gravestone square is never pressed: beside a scenery stone the mask offers no MOVE into it, a MOVE played
/// anyway ends without a frame of its direction, and a MOVE along the stone's column from two squares off ends on the
/// grid point before it. The same frames by the controller (one frame of Down on that grid point) release a flying
/// Ghini.
#[test]
fn a_move_never_presses_a_gravestone_square() {
    let Some(mut w) = walk_to(ow(0x31)) else { return };
    clear_room(&mut w.s);
    // The scenery stone at X $30, Y $90 (Link's Y $8D): from the free square above it, (X $30, Y $7D), Down is into it.
    place(&mut w, 0x30, 0x7D, Direction::S);
    let beside = w.s.state().clone();
    assert!(!beside.can_link_move(Direction::S) && !beside.is_secret_push_direction(Direction::S));
    let into = Action::parse("MOVE S 16").unwrap();
    assert!(!w.s.mask().is_allowed(&into), "{}", w.s.mask());
    w.s.act(into).unwrap();
    for _ in 0..20 {
        w.s.act(Action::parse("WAIT 4").unwrap()).unwrap();
    }
    assert_eq!(flying_ghinis(&w.s), 0, "MOVE S beside the stone made a flying Ghini");
    // Along the column from above: X $30 has stones at Y $6D-$7C too, so come down the free column X $40 is not the
    // case; from (X $30, Y $7D) up is the stone of the top row. A MOVE N of 32 frames from there presses nothing.
    place(&mut w, 0x30, 0x7D, Direction::N);
    w.s.act(Action::parse("MOVE N 32").unwrap()).unwrap();
    for _ in 0..20 {
        w.s.act(Action::parse("WAIT 4").unwrap()).unwrap();
    }
    assert_eq!(flying_ghinis(&w.s), 0, "MOVE N beside the stone made a flying Ghini");
    // A MOVE that reaches the stone's grid point mid-way: from (X $20, Y $7D) E for 32 frames passes X $30 (free
    // row); from (X $30, Y $5D)... the row above the top stones: S for 32 frames ends at once (the stone is next).
    place(&mut w, 0x30, 0x5D, Direction::S);
    assert!(!w.s.state().can_link_move(Direction::S));
    w.s.act(Action::parse("MOVE S 32").unwrap()).unwrap();
    assert_eq!((w.s.state().link.position.x, w.s.state().link.position.y), (0x30, 0x5D));
    for _ in 0..20 {
        w.s.act(Action::parse("WAIT 4").unwrap()).unwrap();
    }
    assert_eq!(flying_ghinis(&w.s), 0);
    // The controller's own frame of Down there: the game makes one (asm:Spawn_CheckTouchedArmosOrGravestone).
    place(&mut w, 0x30, 0x7D, Direction::S);
    w.s.press(direction_buttons(Direction::S), 2);
    assert_eq!(flying_ghinis(&w.s), 1, "the game's rule: {}", describe(&w.s));
}

/// The doorways the route keeps out of: at the mouth of 0x29's N key door MOVE N is not offered, a MOVE N from two
/// squares below it ends at the mouth and no key is spent; by the controller one frame of Up there spends one.
/// 0x28's N doorway and 0x2c's S doorway are masked the same way; 0x2c's N key door, the route's own, is offered.
#[test]
fn the_doorways_off_the_route_are_masked_and_a_move_ends_at_their_mouth() {
    let Some(mut w) = walk_to(uw(0x29)) else { return };
    let masks = |w: &mut Walk| {
        clear_room(&mut w.s);
        let state = w.s.state().clone();
        let objective = w.gc.objectives(Some(&state), &state);
        w.masks(&objective);
    };
    // 0x29 (a ring of blocks: the N door's mouth is on the top row). At the mouth MOVE N is not offered; played
    // anyway it presses nothing and no key is spent.
    let keys = w.s.state().keys();
    assert_eq!(keys, 2);
    place(&mut w, 0x78, 0x5D, Direction::N);
    masks(&mut w);
    let up = Action::parse("MOVE N 32").unwrap();
    assert!(w.s.state().at_doorway_mouth(Direction::N) && w.s.state().can_link_move(Direction::N));
    assert!(!w.s.mask().is_allowed(&up), "{}", w.s.mask());
    assert!(w.s.mask().is_allowed(&Action::parse("MOVE E 8").unwrap()), "{}", w.s.mask());
    w.s.act(up).unwrap();
    assert_eq!((w.s.state().keys(), w.s.state().full_location(), w.s.state().link.position.y), (keys, uw(0x29), 0x5D));
    // The game's rule, by the controller: one frame of Up at the mouth spends a key.
    w.s.press(direction_buttons(Direction::N), 1);
    assert_eq!(w.s.state().keys(), keys - 1, "{}", describe(&w.s));

    // 0x2c: S (five Wizzrobes shut Link in) is masked at its mouth, N (the route's key door) is not.
    let Some(mut w) = walk_to(uw(0x2c)) else { return };
    place(&mut w, 0x78, 0xBD, Direction::S);
    masks(&mut w);
    assert!(w.s.state().at_doorway_mouth(Direction::S) && w.s.state().can_link_move(Direction::S));
    assert!(!w.s.mask().is_allowed(&Action::parse("MOVE S 8").unwrap()), "{}", w.s.mask());
    // A MOVE S of 32 frames down the middle from two squares above ends at the mouth, in the room.
    place(&mut w, 0x78, 0x9D, Direction::S);
    masks(&mut w);
    let down = Action::parse("MOVE S 32").unwrap();
    let objective = w.gc.objectives(None, w.s.state());
    assert!(w.s.mask().is_allowed(&down), "{}\n{}", w.s.mask(), grid(&w.s, &objective));
    w.s.act(down).unwrap();
    assert_eq!((w.s.state().full_location(), w.s.state().link.position.x, w.s.state().link.position.y), (uw(0x2c), 0x78, 0xBD));
    w.s.act(Action::parse("WAIT 8").unwrap()).unwrap();
    assert_eq!((w.s.state().full_location(), w.s.state().var(Var::DoorwayDir)), (uw(0x2c), 0), "{}", describe(&w.s));
    place(&mut w, 0x78, 0x5D, Direction::N);
    masks(&mut w);
    assert!(w.s.mask().is_allowed(&Action::parse("MOVE N 8").unwrap()), "{}", w.s.mask());

    // 0x28: N (Gleeok's room) is masked at its mouth.
    let Some(mut w) = walk_to(uw(0x28)) else { return };
    place(&mut w, 0x78, 0x5D, Direction::N);
    masks(&mut w);
    assert!(w.s.state().at_doorway_mouth(Direction::N) && w.s.state().can_link_move(Direction::N));
    assert!(!w.s.mask().is_allowed(&Action::parse("MOVE N 8").unwrap()), "{}", w.s.mask());
}
