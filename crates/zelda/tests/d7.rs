//! Round 26: the scouting route `d7` played in the real game by the route's objective, from `5_14s_0` (the D5
//! triforce step) into 7:0x2b (the D7 triforce room): the agent's own actions (MOVE, BOMB, RECORDER, FOOD) through the
//! session's masks with the env's per-step rules mirrored (family `items:`/`clips:`, the route's recorder facings,
//! vetoed wraps), the cheat ledger settling after every action, every room change judged by `exit_is_correct`.
//!
//! SETUP (RAM writes): monsters are removed every action (object slots 1-11) but the kept types (`KEEP`); nothing
//! else is written (the state's own 171 rupees, 0 bombs, 5 keys; bombs come from the ledger and the rooms).
//! Walker aids (controller input): cave mouths, the Armos stand tile and a push block's spot are lined up a frame
//! at a time. Not yet asserted per step (round 26, unfinished): the per-room checks listed in the round's brief.
#![cfg(feature = "zcore")]

mod common;

use common::{Counted, load_named_state, open_core};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::BTreeMap;
use std::sync::Arc;
use zelda::cheats::CheatLedger;
use zelda::clips::skip_available;
use zelda::executor::direction_buttons;
use zelda::objectives::{GameCompletion, Objective, ObjectiveKind, exit_is_correct, wraps_into_wrong_exits};
use zelda::room::Target;
use zelda::{Action, ActionKind, Direction, GameMap, GameView, LoadOptions, MapLocation, Session, Var};

type S = Session<Counted>;

const ROUTE: &str = "d7";
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
    log: Vec<String>,
}

#[derive(Debug)]
struct Visit {
    room: MapLocation,
    step: Option<usize>,
    frames: u64,
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

    /// The env's per-step masks: the family's items and clips, the route's recorder facings and wraps.
    fn masks(&mut self, objective: &Objective) {
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
        // A push block, or the secret Armos from its side.
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
    fn run(&mut self, done: impl Fn(&S) -> bool, limit: usize) -> Vec<Visit> {
        let map = self.s.game_map().clone();
        let mut prev: Option<Arc<GameView>> = None;
        let mut objective = self.gc.objectives(None, self.s.state());
        let mut visits = vec![Visit { room: self.s.state().full_location(), step: self.gc.step(), frames: 0 }];
        let mut since = self.s.frame();
        let (mut last_tile, mut still) = (None, 0u32);
        for _ in 0..limit {
            let keep: &[u8] = &[];
            clear_room(&mut self.s, keep);
            let state = self.s.state().clone();
            let before = objective.clone();
            let entered = prev.as_ref().is_some_and(|p| p.full_location() != state.full_location());
            objective = self.gc.objectives(prev.as_deref(), &state);
            if entered {
                let p = prev.as_ref().unwrap();
                let warp = p.level() > 0 && state.level() == 0;
                let ok = warp || exit_is_correct(&map, p, &before, state.full_location());
                self.log.push(format!(
                    "{} -> {} step {:?} kind {:?} ok {ok} {} 34A {:#04x}",
                    p.full_location(),
                    state.full_location(),
                    self.gc.step(),
                    objective.kind,
                    describe(&self.s),
                    self.s.state().peek(0x34A)
                ));
                println!("{}", self.log.last().unwrap());
                if std::env::var("D7_GRID").is_ok_and(|r| state.full_location().value == u8::from_str_radix(&r, 16).unwrap()) {
                    println!("{}", grid(&self.s, &objective));
                }
                if state.level() > 0 && !state.full_location().in_cave {
                    let room = self.s.state().room();
                    let t = |c, r| format!("{:02x}", room.tile(c, r));
                    println!(
                        "  doors N {} {} S {} {} W {} {} E {} {}",
                        t(15, 2),
                        t(15, 1),
                        t(15, 19),
                        t(15, 20),
                        t(2, 10),
                        t(1, 10),
                        t(29, 10),
                        t(30, 10)
                    );
                }
                assert!(ok, "{} -> {}: not a correct exit ({before:?})", p.full_location(), state.full_location());
                visits.last_mut().unwrap().frames = self.s.frame() - since;
                since = self.s.frame();
                visits.push(Visit { room: state.full_location(), step: self.gc.step(), frames: 0 });
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
            let near_mouth = objective.kind == ObjectiveKind::Cave
                && state.mode() == 5
                && state.level() == 0
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
            let stand = state.mode() == 5
                && ((objective.kind == ObjectiveKind::Cave && armos_room(&self.s) && state.room().stairs_tiles().is_empty())
                    || (state.level() > 0 && state.push_block_pending() && state.peek(0x34D) != 0));
            if stand && let Some((tx, ty)) = objective.targets.iter().find_map(|t| match *t {
                Target::Tile((c, r)) if (c - here.0).abs() <= 1 && (r - here.1).abs() <= 1 => Some((c * 8, r * 8 + 0x3D)),
                _ => None,
            }) {
                for _ in 0..24 {
                    let p = self.s.state().link.position;
                    let d = if p.y != ty {
                        if p.y < ty { Direction::S } else { Direction::N }
                    } else if p.x != tx {
                        if p.x < tx { Direction::E } else { Direction::W }
                    } else {
                        break;
                    };
                    self.s.press(direction_buttons(d), 1);
                }
            }
            let action = next_action(&self.s, &objective, still / 3);
            if std::env::var("D7_TRACE").is_ok_and(|r| state.full_location().value == u8::from_str_radix(&r, 16).unwrap()) {
                let t: Vec<_> = objective.targets.iter().take(4).collect();
                let tv: Vec<Target> = objective.targets.iter().copied().collect();
                let room = self.s.state().room();
                let here = self.s.state().link.tile();
                let d0 = room.wavefront(&tv, &[], &[], objective.clips).get(here);
                println!(
                    "  {} obj {:?} {t:?}+{} d {d0:?} -> {action} | {} | 51A {} 523 {}",
                    describe(&self.s),
                    objective.kind,
                    objective.targets.len(),
                    self.s.mask(),
                    self.s.state().peek(0x51A),
                    self.s.state().peek(0x523)
                );
            }
            if still == 50 {
                let room = self.s.state().room();
                let raw: Vec<String> = (0..4).map(|r| format!("{:02x} {:02x}", room.tile(15, r), room.tile(16, r))).collect();
                println!("stuck grid (N door rows 0-3: {raw:?}):\n{}", grid(&self.s, &objective));
            }
            if still > 0 && still % 200 == 0 {
                println!("stuck {still}: {} obj {:?} action {action} mask {}", describe(&self.s), objective.kind, self.s.mask());
            }
            assert!(still < 2000, "stuck at {}", describe(&self.s));
            let from = state.full_location();
            prev = Some(state);
            self.s.act(action).unwrap();
            self.ledger.after_action(&mut self.s, &self.gc, from);
        }
        assert!(done(&self.s), "the walk ended at {} ({} rooms)", describe(&self.s), visits.len());
        visits
    }
}

#[test]
fn the_d7_route_is_walked_from_the_d5_triforce_into_the_d7_triforce_room() {
    let Some(core) = open_core() else { return };
    let map = map();
    let mut s = Session::new(Counted(core, 0), map.clone());
    s.load(&load_named_state("5_14s_0"), &LoadOptions::default(), &mut ChaCha8Rng::seed_from_u64(0)).unwrap();
    let first = map.routes["d2first"].len() - 1;
    let gc = GameCompletion::new(map.clone(), ROUTE, Some(first)).unwrap();
    let mut w = Walk { s, gc, ledger: CheatLedger::default(), families: BTreeMap::new(), log: Vec::new() };
    w.ledger.begin(&mut w.s, &w.gc);
    println!("start {}", describe(&w.s));
    let target = MapLocation::new(7, 0x2b, false);
    let visits = w.run(|s| s.state().full_location() == target, 40_000);
    for l in &w.log {
        println!("{l}");
    }
    println!("{visits:?}");
}
