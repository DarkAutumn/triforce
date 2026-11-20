//! Round 28: `d2first`'s steps 277-292 since round 31 (scouted as `d8`) played in the real game by its objective,
//! from Dungeon 6's triforce through the leg (the two recorder plays out of 0x22, 0x3c, 0x4c, 0x4d, 0x5d), the burned
//! tree of 0x6d and Dungeon 8 to its triforce: the agent's own actions (MOVE, RECORDER, CANDLE, BOMB, SWORD) through
//! the session's masks with the env's per-step rules mirrored (family `items:`/`clips:`, a step's own bomb mode, the
//! route's bomb and rupee reserves, RECORDER's, ARROW's and CANDLE's facings, the doorways kept out of), the cheat
//! ledger settling after every action, every room change judged by `exit_is_correct`, and per step what the step
//! means (the walk's [`Tick`]s). Beside the walk: 0x3f is crossed, not fought, at 5 bombs; the two key doors off the
//! route are masked and a MOVE ends at their mouth; CANDLE is offered at neither tree's wrong side nor at 0x4d's
//! shop tree and from both windows of 0x6d's; in 0x6e a `free` family on a `bombs: required` step gets BOMB only by
//! the wall.
//!
//! SETUP. The start is staged by the route's own walk: `7_2bw_0` (the D7 triforce step; its whirlwind index `$FB` was
//! left by a chain's two plays at 0x0b) is walked by round 27's rules to Dungeon 6's triforce (`crates/zelda/tests/
//! d6.rs` asserts that stretch), so Link stands in 6:0x0c with what that walk brings: 13 hearts, the magic sword, 35
//! rupees, 1 bomb, 3 keys (2 before round 31: 6:0x29's key is taken since). RAM writes: monsters are removed every
//! action (object slots 1-11) but the kept types
//! (`KEEP`: tile objects, the whirlwind, the ladder, Gohma, Gleeok's body); Gleeok's fireballs and flying heads are
//! removed with the rest (dodging them is the model's work), his four heads are cut by the walk's own SWORD. Nothing
//! else is written but where a test says so. Walker aids (controller input, a frame at a time): cave mouths, key
//! doors' mouths, push spots, Link's X under Gohma, and Link's place under Gleeok's lowest head on row Y `$9D`.
#![cfg(feature = "zcore")]

mod common;

use common::{Counted, load_named_state, open_core};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use zelda::cheats::CheatLedger;
use zelda::clips::skip_available;
use zelda::enums::Tile;
use zelda::enums::enemy::{BLUE_DARKNUT, FLYING_GHINI, GLEEOK4, MANHANDLA, POLS_VOICE, RED_GOHMA};
use zelda::executor::direction_buttons;
use zelda::game::gleeok_heads;
use zelda::game_map::Secret;
use zelda::objectives::{GameCompletion, Objective, ObjectiveKind, doorways_kept_out, exit_is_correct, gohma, gohma_stand};
use zelda::room::Target;
use zelda::{Action, ActionKind, Direction, GameMap, GameView, LoadOptions, MapLocation, Session, Var};

type S = Session<Counted>;

const ROUTE: &str = zelda::game_map::DEFAULT_ROUTE;
const OBJ_TYPE: u16 = 0x34F;
/// `Gohma_ObjEyeState` (`$46B` + slot): 3 while the eye is fully open.
const GOHMA_EYE: u16 = 0x46B;
/// The invincibility timer of the slot Gleeok's attached heads share (`$4F0` + 5): a hit lands only at 0.
const GLEEOK_HEAD_TIMER: u16 = 0x4F0 + 5;
/// Link's row under Gleeok (docs/routes/d8-walks.md §6.1: no attached head reaches it, the blade reaches them).
const GLEEOK_ROW: i32 = 0x9D;
/// The first step of the stretch (0x22 out of Dungeon 6) and the step after its triforce (0x6d out of Dungeon 8; the
/// route runs on to the end of the game, `d9.rs`).
const FIRST: usize = 277;
const NEXT: usize = 293;

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
/// gravestone, the burned tree, the flute secret, Gohma (killed by the walk's own arrow) and Gleeok's body (his heads
/// are cut by the walk's own sword). A flying Ghini is kept too: the walk must never make one.
const KEEP: [u8; 12] = [0x61, 0x5F, 0x63, 0x2E, 0x68, 0x65, 0x64, 0x5E, 0x33, RED_GOHMA, FLYING_GHINI, GLEEOK4];

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
    /// A room whose monsters the walk leaves alone ([`walk_into`]: the room as the game makes it).
    spare: Option<MapLocation>,
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
    /// Link's position and facing.
    x: i32,
    y: i32,
    facing: Direction,
    /// The game mode (5: play).
    mode: u8,
    /// Gleeok's attached heads.
    heads: usize,
    /// Object types in slots 1-11 (before the walk removes monsters).
    types: Vec<u8>,
    /// The room's stairs tiles (an overworld secret's once it shows).
    stairs: Vec<Tile>,
    /// The directions the session's mask offers ARROW, BOMB and CANDLE in; whether it offers RECORDER.
    arrows: Vec<Direction>,
    bombs: Vec<Direction>,
    candles: Vec<Direction>,
    recorder: bool,
    /// The route's rules for the state: the facings of RECORDER and CANDLE, the doorways MOVE keeps out of, the
    /// step's own bomb mode, the reserves.
    recorder_facings: u8,
    candle_facings: Option<u8>,
    kept_out: u8,
    step_bombs: Option<String>,
    bomb_reserve: u8,
    rupee_reserve: u8,
    /// One of Link's candle fires burns.
    fire: bool,
    hearts: f64,
    max_hearts: i32,
    rupees: i32,
    bomb_count: i32,
    keys: i32,
}

fn ow(room: u8) -> MapLocation {
    MapLocation::new(0, room, false)
}

fn uw(room: u8) -> MapLocation {
    MapLocation::new(8, room, false)
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
            .unwrap_or_else(|| "darknut".to_owned());
        self.families.entry(name.clone()).or_insert_with(|| family(&name)).clone()
    }

    /// The env's per-step masks (`env::specialist_env::update_disabled_kinds`): the family's items and clips with
    /// the step's own bomb mode and the route's reserves, the bomb targets of `required`, the facings of RECORDER,
    /// ARROW and CANDLE, the doorways kept out of.
    fn masks(&mut self, objective: &Objective) {
        let f = self.family_of();
        let state = self.s.state().clone();
        let map = self.s.game_map().clone();
        let reserve = self.gc.bomb_reserve();
        let required = map.requires_bombs(state.full_location());
        let mode = self.gc.step_bombs(&state).map_or(f.bombs.clone(), str::to_owned);
        let bombs = match mode.as_str() {
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
        let limited = mode == "required" && required && !state.next_kill_arms_drop();
        let targets = limited.then(|| zelda::objectives::route_bomb_targets(&map, &state, objective));
        self.s.set_bomb_targets(targets.as_deref());
        self.s.set_clips(f.clips);
        self.gc.set_clips(f.clips);
        let facings = self.gc.recorder_facings(self.s.state());
        self.s.set_recorder_facings(Some(facings));
        self.s.set_arrow_facings(self.gc.arrow_facings(&state));
        self.s.set_candle_facings(self.gc.candle_facings(&state));
        self.s.set_vetoed_doorways(doorways_kept_out(&map, &state, objective));
        self.s.set_vetoed_wraps(&[]);
    }
}

fn next_action(s: &S, objective: &Objective, nudge: u32) -> Action {
    let state = s.state();
    let mask = s.mask();
    let wait = Action::parse("WAIT 4").unwrap();
    if !matches!(state.mode(), 5 | 9 | 11) || state.link.status != 0 {
        return wait;
    }
    // The route's item plays, where the mask offers them: the recorder (the whirlwind's two plays), the candle (the
    // tree).
    let recorder = Action::new(ActionKind::Recorder, Direction::None, 0).unwrap();
    if mask.is_allowed(&recorder) {
        return recorder;
    }
    for d in [Direction::W, Direction::E, Direction::N, Direction::S] {
        let candle = Action::new(ActionKind::Candle, d, 0).unwrap();
        if mask.is_allowed(&candle) {
            return candle;
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
        // The secret gravestone from below: one MOVE N of 32 frames, which ends on the stairs it uncovers.
        if state.is_secret_push_direction(Direction::N) {
            return Action::parse("MOVE N 32").unwrap();
        }
        // On the stone's stairs, or under it while it slides: the cave opens, the walk waits.
        if state.secret_push().is_some() && objective.kind == ObjectiveKind::Cave {
            return wait;
        }
        // On a stand tile of the tree while its fire burns (CANDLE is offered first when it can be played): wait.
        if state.secret_burn().is_some() && objective.kind == ObjectiveKind::Cave {
            return wait;
        }
        // Standing where the objective wants him on the whirlwind's row: wait for it.
        let overworld = state.level() == 0 && !state.full_location().in_cave;
        if overworld
            && objective.kind == ObjectiveKind::Move
            && objective.targets.contains(&Target::Tile(here))
            && room.exits.direction_of(here).is_none()
        {
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
    /// `7_2bw_0` (Link entering Dungeon 7's triforce room) with a completion of `d8` on the D7 triforce step (243)
    /// and the ledger begun.
    fn start() -> Option<Walk> {
        let core = open_core()?;
        let map = map();
        let mut s = Session::new(Counted(core, 0), map.clone());
        s.load(&load_named_state("7_2bw_0"), &LoadOptions::default(), &mut ChaCha8Rng::seed_from_u64(0)).unwrap();
        let first = map.routes[ROUTE].iter().position(|s| s.level == 7 && s.exit == "triforce").unwrap();
        let gc = GameCompletion::new(map.clone(), ROUTE, Some(first)).unwrap();
        let (ledger, families) = (CheatLedger::default(), BTreeMap::new());
        let mut w = Walk { s, gc, ledger, families, ticks: Vec::new(), spare: None };
        w.ledger.begin(&mut w.s, &w.gc);
        Some(w)
    }

    /// The staged start of the stretch: the walk of round 27's steps (244-276, `d6.rs`) until Link holds Dungeon
    /// 6's piece (the action that takes it ends with the game's own warp: he stands at its entrance, 0x22); the
    /// ticks begin anew.
    fn at_the_d6_triforce() -> Option<Walk> {
        let mut w = Walk::start()?;
        w.run(|s| s.state().has_triforce(6), 60_000);
        assert_eq!(w.s.state().full_location(), ow(0x22));
        assert_eq!(w.ledger.cheated.bombs + w.ledger.cheated.keys + w.ledger.cheated.rupees, 0);
        w.ticks.clear();
        Some(w)
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
            heads: gleeok_heads(state.ram()).len(),
            types,
            stairs: if state.full_location().in_cave { Vec::new() } else { state.room().stairs_tiles() },
            arrows: offered(ActionKind::Arrow),
            bombs: offered(ActionKind::Bomb),
            candles: offered(ActionKind::Candle),
            recorder: mask.is_allowed(&Action::new(ActionKind::Recorder, Direction::None, 0).unwrap()),
            recorder_facings: self.gc.recorder_facings(state),
            candle_facings: self.gc.candle_facings(state),
            kept_out: doorways_kept_out(map, state, objective),
            step_bombs: self.gc.step_bombs(state).map(str::to_owned),
            bomb_reserve: self.gc.bomb_reserve(),
            rupee_reserve: self.gc.rupee_reserve(),
            fire: state.link_fire_burning(),
            hearts: state.health(),
            max_hearts: state.max_health(),
            rupees: state.rupees(),
            bomb_count: state.bombs(),
            keys: state.keys(),
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

    /// Gohma's script (round 27, `d6.rs`): Link's X kept 2-4 pixels left of Gohma's on the stand tile, ARROW N when
    /// the mask offers it and the eye is fully open. None when no Gohma lives here.
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

    /// Gleeok's script (docs/routes/d8-walks.md §6.4, without its dodging: the walk removes fireballs and flying
    /// heads): Link is kept on row Y `$9D` under the head that hangs lowest, a frame at a time (walker aid), and
    /// SWORD N (BEAMS N at full hearts) is played when the mask offers it, that head is within 8 pixels of Link's X
    /// and 10-27 above him and the heads' timer is 0. None when no head is left.
    fn gleeok_action(&mut self) -> Option<Action> {
        let state = self.s.state().clone();
        if state.mode() != 5 || gleeok_heads(state.ram()).is_empty() {
            return None;
        }
        for _ in 0..12 {
            let st = self.s.state();
            let Some(low) = gleeok_heads(st.ram()).into_iter().map(|(_, p)| p).max_by_key(|p| p.y) else { break };
            let p = st.link.position;
            let d = if p.y != GLEEOK_ROW && (p.x - 0x78).abs() <= 0x38 {
                if p.y > GLEEOK_ROW { Direction::N } else { Direction::S }
            } else if (p.x - low.x).abs() > 3 {
                if p.x < low.x { Direction::E } else { Direction::W }
            } else {
                break;
            };
            self.s.press(direction_buttons(d), 1);
        }
        let st = self.s.state();
        let low = gleeok_heads(st.ram()).into_iter().map(|(_, p)| p).max_by_key(|p| p.y)?;
        let p = st.link.position;
        let reach = (p.x - low.x).abs() <= 8 && (10..=27).contains(&(p.y - low.y));
        let ready = st.peek(GLEEOK_HEAD_TIMER) == 0 && p.y == GLEEOK_ROW;
        let swing = [ActionKind::Sword, ActionKind::Beams]
            .into_iter()
            .map(|kind| Action::new(kind, Direction::N, 0).unwrap())
            .find(|a| self.s.mask().is_allowed(a));
        Some(match swing {
            Some(a) if reach && ready => a,
            _ => Action::parse("WAIT 4").unwrap(),
        })
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
            if self.spare != Some(self.s.state().full_location()) {
                clear_room(&mut self.s);
            }
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
            let secret = state.secret_push().is_some() || state.secret_burn().is_some();
            let near_mouth = objective.kind == ObjectiveKind::Cave
                && state.mode() == 5
                && state.level() == 0
                && !secret
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
            // grid point.
            let under_stone =
                state.secret_push().is_some() && objective.kind == ObjectiveKind::Cave && state.room().stairs_tiles().is_empty();
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
            // A key door the objective leaves through: within a tile of its mouth Link is lined up (walker aid) and
            // walks into it, which spends the key.
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
            } else if let Some(action) = self.gleeok_action() {
                action
            } else {
                let mut action = next_action(&self.s, &objective, still / 3);
                // Walker aid (controller input), the graveyard only (round 27): before a turn Link is lined up on
                // the grid point of the tile the walk decided on.
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

/// The walk from the D6 triforce to the D8 triforce, every step asserted for what it means.
#[test]
fn the_d8_route_is_walked_from_the_d6_triforce_to_the_d8_triforce() {
    let Some(mut w) = Walk::at_the_d6_triforce() else { return };
    let map = w.s.game_map().clone();
    let route = &map.routes[ROUTE];
    assert_eq!(route[NEXT].location(), ow(0x6d), "step 293 starts in 0:0x6d, out of Dungeon 8");
    let frames_in = w.s.frame();
    let visits = w.run(|s| s.state().has_triforce(8), 60_000);
    let t = &w.ticks;

    // The rooms in order (one way: the stretch has no clip); every change was a correct exit (`run`), the whirlwind's
    // landing included.
    let o = |r: u8| (0, r, false);
    let u = |r: u8| (8, r, false);
    let rooms = rooms_from(&visits, ow(0x22));
    let expected = vec![
        o(0x22), o(0x3c), o(0x4c), o(0x4d), o(0x5d), o(0x6d), u(0x7e), u(0x6e), u(0x5e), u(0x4e), u(0x3e), u(0x3f),
        (8, 0x2f, true), u(0x4c), u(0x3c), u(0x2c),
        // The piece ends the level: the game sets Link down at Dungeon 8's entrance.
        o(0x6d),
    ];
    assert_eq!(rooms, expected);
    assert!(t.iter().all(|k| k.action.kind != ActionKind::Clip), "no clip on the stretch");

    // 0x22 (step 277, `entry: cave`, `exit: recorder`, ow-lynel with `recorder: true`): the game sets Link down at
    // Dungeon 6's entrance with its piece and 13 full hearts, facing down; RECORDER is offered from the first tick,
    // only while the route's facings hold his, and the walk plays twice; after the second it is not offered again;
    // nothing else is played but waiting; the whirlwind sets him down in 0x3c (a correct exit).
    let out = at(t, ow(0x22), FIRST);
    let first = out[0];
    assert_eq!((route[FIRST].entry.as_str(), route[FIRST].exit.as_str(), first.bomb_reserve), ("cave", "recorder", 1));
    assert_eq!((first.x, first.y, first.facing, first.hearts, first.max_hearts), (0x70, 0x7D, Direction::S, 13.0, 13));
    assert!(first.recorder && first.recorder_facings & Direction::S.value() != 0, "{first:?}");
    for k in &out {
        assert!(!k.recorder || k.recorder_facings & k.facing.value() != 0, "RECORDER offered facing {:?}", k.facing);
        assert!(matches!(k.action.kind, ActionKind::Recorder | ActionKind::Wait), "{}", k.action);
        assert!(k.candles.is_empty());
    }
    let plays: Vec<usize> = out.iter().enumerate().filter(|(_, k)| k.action.kind == ActionKind::Recorder).map(|(i, _)| i).collect();
    assert_eq!(plays.len(), 2, "two plays");
    assert!(out[plays[1] + 1..].iter().all(|k| !k.recorder), "no third play");
    assert_eq!(out.last().unwrap().hearts, 13.0);
    let landed = t.iter().find(|k| k.room == ow(0x3c) && k.action.kind == ActionKind::Move).unwrap();
    assert_eq!((landed.step, landed.x, landed.y), (Some(278), 0x80, 0xAD), "set down at Dungeon 2's entrance");

    // The leg (steps 278-281): 0x3c S, 0x4c E, 0x4d S, 0x5d S. ow-moblin has `candle: true` from 0x4d on: CANDLE is
    // offered in neither 0x4d (its shop tree) nor 0x5d, and RECORDER nowhere after 0x22.
    for (room, step) in [(0x3c, 278), (0x4c, 279), (0x4d, 280), (0x5d, 281)] {
        let ticks = at(t, ow(room), step);
        assert!(!ticks.is_empty(), "0x{room:02x} on step {step}");
        assert!(ticks.iter().all(|k| k.candles.is_empty() && !k.recorder), "0x{room:02x}");
        assert!(ticks.iter().all(|k| k.candle_facings == Some(0)), "0x{room:02x}");
    }

    // 0x6d (step 282, `exit: cave` under the tree): while the tree stands the targets are the stand tiles of its row
    // (three E of it, three W); CANDLE is offered only inside the window (X $B8-$CF from the N entry), only facing W,
    // only while no fire of Link's burns, and is played once; the walk waits while the fire burns; then the stairs
    // are the target; no heart is lost.
    let Some(Secret::Burn((c, r))) = map.get(ow(0x6d)).unwrap().secret else { panic!("0x6d's tree") };
    let stand: BTreeSet<Target> = [3, 4, 5, -5, -4, -3].into_iter().map(|d| Target::Tile((c + d, r))).collect();
    let tree = at(t, ow(0x6d), 282);
    let standing: Vec<_> = tree.iter().filter(|k| k.stairs.is_empty() && k.mode == 5).collect();
    assert!(!standing.is_empty());
    for k in &standing {
        assert_eq!((k.kind, &k.targets), (ObjectiveKind::Cave, &stand), "{k:?}");
        let window = (0xB8..=0xCF).contains(&k.x) && (k.y - 0x5D).abs() <= 8;
        assert_eq!(k.candles, if window && !k.fire { vec![Direction::W] } else { vec![] }, "at ({:#04x}, {:#04x})", k.x, k.y);
    }
    let burns: Vec<_> = tree.iter().filter(|k| k.action.kind == ActionKind::Candle).collect();
    assert_eq!(burns.len(), 1, "one fire");
    assert_eq!(burns[0].action.direction, Direction::W);
    assert!((0xB8..=0xCF).contains(&burns[0].x) && (burns[0].y - 0x5D).abs() <= 8, "from the window: {:?}", burns[0]);
    assert!(standing.iter().any(|k| k.fire && k.action.kind == ActionKind::Wait), "waits while it burns");
    let shown: Vec<_> = tree.iter().filter(|k| !k.stairs.is_empty()).collect();
    assert!(!shown.is_empty());
    for k in &shown {
        assert_eq!((k.stairs.as_slice(), &k.targets), (&[(c, r)][..], &[Target::Tile((c, r))].into()));
        assert!(k.candles.is_empty(), "no CANDLE once the stairs show");
    }
    assert_eq!(tree.last().unwrap().hearts, 13.0);

    // Dungeon 8. Keys: 3 in (round 31: 6:0x29's), 0x5e +1, 0x4e N -1, 0x4c's left (round 31); bombs: 1 in, 0x6e's
    // wall, 0x3f's four (fought for at 0), 0x4c's wall; nothing cheated.
    let first = |room: u8| t.iter().find(|k| k.room == uw(room)).unwrap_or_else(|| panic!("8:{room:#04x}"));
    let ledger: Vec<(i32, i32)> =
        [0x7e, 0x6e, 0x5e, 0x4e, 0x3e, 0x3f, 0x4c, 0x3c, 0x2c].into_iter().map(|r| (first(r).keys, first(r).bomb_count)).collect();
    assert_eq!(ledger, [(3, 1), (3, 1), (3, 0), (4, 0), (3, 0), (3, 0), (3, 4), (3, 3), (3, 3)]);
    assert_eq!(w.ledger.cheated.bombs + w.ledger.cheated.keys + w.ledger.cheated.rupees, 0);
    // Reserves: bombs 1 to 0x6e's wall, 2 from 0x5e, 1 from Gleeok's room; rupees 1 (Gohma's step), 8 from 0x4c.
    let reserves: Vec<(u8, u8)> = [0x6e, 0x5e, 0x3f, 0x4c, 0x3c].into_iter().map(|r| (first(r).bomb_reserve, first(r).rupee_reserve)).collect();
    assert_eq!(reserves, [(1, 1), (2, 1), (2, 1), (2, 8), (1, 8)]);

    // The rooms were full when Link came in (the walk removes monsters; the fights are the models').
    for (room, monster) in [(0x6e, MANHANDLA), (0x5e, BLUE_DARKNUT), (0x4e, BLUE_DARKNUT), (0x3e, BLUE_DARKNUT), (0x3f, BLUE_DARKNUT), (0x4c, POLS_VOICE), (0x3c, GLEEOK4)] {
        assert!(t.iter().filter(|k| k.room == uw(room)).any(|k| k.types.contains(&monster)), "8:{room:#04x}: its monsters");
    }

    // 0x6e (step 284, boss-manhandla `free`, the step `bombs: required`, `treasure: skip`): BOMB is offered only
    // where the bomb lands in reach of the N wall, played once from its spot facing N, and the one bomb opens it.
    let wall = at(t, uw(0x6e), 284);
    assert!(wall.iter().all(|k| k.step_bombs.as_deref() == Some("required")));
    assert_eq!(played(&wall, ActionKind::Bomb), 1);
    let offered: Vec<_> = wall.iter().filter(|k| !k.bombs.is_empty()).map(|k| (k.x, k.y)).collect();
    assert!(!offered.is_empty() && offered.iter().all(|(_, y)| *y <= 0x7D), "BOMB only by the N wall: {offered:?}");
    let bomb = wall.iter().find(|k| k.action.kind == ActionKind::Bomb).unwrap();
    // (The objective's bomb spots are the tiles a bomb reaches the wall from: the first one on the walk's way.)
    assert_eq!(bomb.action.direction, Direction::N);
    assert!((bomb.x - 0x78).abs() <= 4 && bomb.y <= 0x6D, "before the N wall: ({:#04x}, {:#04x})", bomb.x, bomb.y);
    assert!(t.iter().filter(|k| k.room != uw(0x6e)).all(|k| k.step_bombs.is_none()), "no other step names a bomb mode");

    // 0x5e (step 285) and 0x4e (step 286): the key doors off the route are kept out of (0x5e E, 0x4e W); 0x4e's N
    // key door, the route's own, is not; no other room of the stretch keeps a doorway out.
    assert!(at(t, uw(0x5e), 285).iter().all(|k| k.kept_out == Direction::E.value()));
    assert!(at(t, uw(0x4e), 286).iter().all(|k| k.kept_out == Direction::W.value()));
    assert!(t.iter().filter(|k| ![uw(0x5e), uw(0x4e)].contains(&k.room)).all(|k| k.kept_out == 0));
    // 0x5e's key is the target once its five are gone, before the N exit.
    assert!(at(t, uw(0x5e), 285).iter().any(|k| k.kind == ObjectiveKind::Treasure && k.keys == 3));

    // 0x3f (step 288, `treasure: {if_bombs_below: 5}`): Link holds no bomb, so the room is fought for its four
    // (the treasure is the target once its six are gone), then the stairs.
    let oneway = at(t, uw(0x3f), 288);
    assert!(oneway.iter().any(|k| k.kind == ObjectiveKind::Treasure && k.bomb_count == 0), "the bombs are a target");
    assert_eq!(oneway.last().unwrap().bomb_count, 4);
    assert_eq!((oneway.last().unwrap().kind, &oneway.last().unwrap().targets), (ObjectiveKind::Move, &[Target::Tile((26, 10))].into()));

    // The tunnel (step 289): in by the right ladder, out by the left one into 0x4c (the room after it: `rooms`).
    let tunnel = at(t, MapLocation::new(8, 0x2f, true), 289);
    assert!(tunnel.first().unwrap().x >= 0x80 && tunnel.last().unwrap().x < 0x80);

    // 0x4c (step 290, pols-voice: `required`, `arrows: true`, `treasure: skip` since round 31): out of the ring
    // straight to the N wall, its corner key left (the owner, Nov 19); BOMB is offered only by the N wall and played
    // once; ARROW is offered nowhere else on the stretch (and not here once the walk has removed the Pols Voices:
    // `arrows_are_offered_among_the_pols_voices_above_the_rupee_reserve`).
    let ring = at(t, uw(0x4c), 290);
    assert_eq!((ring[0].x, ring[0].y), (0x70, 0x7D), "up the stairs inside the ring");
    assert!(ring.iter().all(|k| k.kind != ObjectiveKind::Treasure && k.keys == 3), "the key is left");
    assert_eq!(played(&ring, ActionKind::Bomb), 1);
    let offered: Vec<_> = ring.iter().filter(|k| !k.bombs.is_empty()).map(|k| (k.x, k.y)).collect();
    assert!(!offered.is_empty() && offered.iter().all(|(_, y)| *y <= 0x7D), "BOMB only by the N wall: {offered:?}");
    assert!(ring.iter().all(|k| k.arrows.is_empty() || k.rupees > 8));
    assert!(t.iter().all(|k| k.arrows.is_empty() || k.room == uw(0x4c)), "ARROW only in 0x4c");
    assert!(t.iter().filter(|k| k.room.level == 8).all(|k| k.candles.is_empty() && !k.recorder));

    // Gleeok (step 291, boss-gleeok `never`): four heads on entry; the walk's swings from row Y $9D cut them (3 + 2
    // + 2 + 2 hits that land); BOMB is never offered; then the heart container (14) and the N shutter.
    let boss = at(t, uw(0x3c), 291);
    assert_eq!(boss.iter().map(|k| k.heads).max(), Some(4));
    let alive: Vec<_> = boss.iter().filter(|k| k.heads > 0).collect();
    assert!(alive.iter().all(|k| k.kind == ObjectiveKind::Fight), "the fight is the objective while a head is left");
    assert!(boss.iter().all(|k| k.bombs.is_empty()));
    let swings = boss.iter().filter(|k| matches!(k.action.kind, ActionKind::Sword | ActionKind::Beams)).count();
    assert!(boss.iter().filter(|k| matches!(k.action.kind, ActionKind::Sword | ActionKind::Beams)).all(|k| k.y == GLEEOK_ROW));
    assert!((9..=40).contains(&swings), "{swings} swings");
    let dead: Vec<_> = boss.iter().filter(|k| k.heads == 0 && k.max_hearts == 13 && k.mode == 5).collect();
    assert!(dead.iter().any(|k| k.kind == ObjectiveKind::Treasure), "the heart container is the target");
    assert_eq!(boss.last().unwrap().max_hearts, 14);

    // 0x2c's triforce on step 292, the walk's end.
    let end = t.last().unwrap();
    assert_eq!((end.room, end.step, end.kind), (uw(0x2c), Some(NEXT - 1), ObjectiveKind::Treasure));
    assert!(w.s.state().has_triforce(8));
    // (`--nocapture`: the walk's numbers for the round document.)
    eprintln!(
        "d8 walk: {} actions, {} frames from the D6 triforce, {swings} swings at Gleeok, {} hearts of {} at the end, {} rupees, {} bombs, {} keys",
        t.len(),
        w.s.frame() - frames_in,
        w.s.state().health(),
        w.s.state().max_health(),
        w.s.state().rupees(),
        w.s.state().bombs(),
        w.s.state().keys()
    );
}

/// Walks from the D6 triforce to `room` (Link playing in it) and returns the walk.
fn walk_to(room: MapLocation) -> Option<Walk> {
    let mut w = Walk::at_the_d6_triforce()?;
    w.run(|s| s.state().full_location() == room && matches!(s.state().mode(), 5 | 11), 60_000);
    Some(w)
}

/// Walks from the D6 triforce into `room` and on (no input, nothing removed) until Link can play there and the room
/// holds a `monster`: the room as the game makes it.
fn walk_into(room: MapLocation, monster: u8) -> Option<Walk> {
    let mut w = Walk::at_the_d6_triforce()?;
    w.spare = Some(room);
    w.run(|s| s.state().full_location() == room, 60_000);
    let made = |s: &S| s.state().mode() == 5 && (1..=11u16).any(|slot| s.state().peek(OBJ_TYPE + slot) == monster);
    for _ in 0..400 {
        if made(&w.s) {
            break;
        }
        w.s.press(0, 1);
    }
    assert!(made(&w.s), "{room}: no monster {monster:#04x}: {}", describe(&w.s));
    Some(w)
}

fn place(w: &mut Walk, x: u8, y: u8, d: Direction) {
    let ram = [(Var::LinkX, x), (Var::LinkY, y), (Var::LinkDirection, d.value()), (Var::LinkGridOffset, 0), (Var::DoorwayDir, 0)];
    w.s.apply_ram(&ram);
}

/// The objective and the masks for the state as it stands (monsters removed first unless `monsters`).
fn look(w: &mut Walk, monsters: bool) -> Objective {
    if !monsters {
        clear_room(&mut w.s);
    }
    let state = w.s.state().clone();
    let objective = w.gc.objectives(Some(&state), &state);
    w.masks(&objective);
    objective
}

fn candles(w: &Walk) -> Vec<Direction> {
    [Direction::N, Direction::S, Direction::W, Direction::E]
        .into_iter()
        .filter(|&d| w.s.mask().is_allowed(&Action::new(ActionKind::Candle, d, 0).unwrap()))
        .collect()
}

/// CANDLE on the overworld only where a fire burns the route's tree: from 0x6d's W window facing E as from its E
/// window facing W (X `$78`-`$8F` and `$B8`-`$CF` on row Y `$5D`), at neither window's far side, not a square off the
/// row's reach, and never beside 0x4d's shop tree, where the same family plays two rooms before. The fire from the W
/// window shows the stairs too.
#[test]
fn the_candle_is_offered_only_where_it_burns_the_routes_tree() {
    let Some(mut w) = walk_to(ow(0x4d)) else { return };
    // 0x4d's shop tree at tile (26, 12): X $D0, row Y $9D. On both sides of it, facing it: no CANDLE.
    for x in [0xB0u8, 0xB8, 0xC0, 0xE8] {
        place(&mut w, x, 0x9D, if x < 0xD0 { Direction::E } else { Direction::W });
        look(&mut w, false);
        assert_eq!((w.gc.candle_facings(w.s.state()), candles(&w)), (Some(0), vec![]), "0x4d at X {x:#04x}");
    }
    let Some(mut w) = walk_to(ow(0x6d)) else { return };
    assert_eq!(w.gc.step(), Some(282));
    let offered = |w: &mut Walk, x: u8, y: u8| {
        place(w, x, y, Direction::S);
        look(w, false);
        candles(w)
    };
    for x in (0x70u8..=0xD8).step_by(8) {
        let expected = match x {
            0x78..=0x8F => vec![Direction::E],
            0xB8..=0xCF => vec![Direction::W],
            _ => vec![],
        };
        // The tree itself and the squares beside it are not walkable; the mask is asked where Link can stand.
        if (0x90..=0xB0).contains(&x) {
            continue;
        }
        assert_eq!(offered(&mut w, x, 0x5D), expected, "0x6d at X {x:#04x}");
    }
    assert_eq!(offered(&mut w, 0xC0, 0x7D), vec![], "two squares under the row");
    // From the W window: CANDLE E, and the stairs show (walked: 92 frames after B).
    place(&mut w, 0x80, 0x5D, Direction::S);
    let objective = look(&mut w, false);
    assert_eq!(objective.kind, ObjectiveKind::Cave);
    w.s.act(Action::new(ActionKind::Candle, Direction::E, 0).unwrap()).unwrap();
    look(&mut w, false);
    assert!(w.s.state().link_fire_burning() && candles(&w).is_empty(), "no second fire while one burns");
    for _ in 0..40 {
        w.s.act(Action::parse("WAIT 4").unwrap()).unwrap();
    }
    assert_eq!(w.s.state().room().stairs_tiles(), [(20, 4)], "{}", describe(&w.s));
    let objective = look(&mut w, false);
    assert_eq!((objective.targets, candles(&w)), ([Target::Tile((20, 4))].into(), vec![]));
}

/// 0x3f by the owner's rule (a room is fought for its bombs only at 4 bombs or fewer): with 5 bombs the objective
/// crosses to the stairs with the room's monsters alive; with 4 it fights.
#[test]
fn the_one_way_room_is_fought_for_its_bombs_only_at_four_or_fewer() {
    let Some(mut w) = walk_into(uw(0x3f), BLUE_DARKNUT) else { return };
    assert_eq!(w.gc.step(), Some(288));
    assert!((1..=11u16).any(|slot| w.s.state().peek(OBJ_TYPE + slot) == BLUE_DARKNUT), "the room is full");
    w.s.apply_ram(&[(Var::Bombs, 5)]);
    let objective = look(&mut w, true);
    assert_eq!((objective.kind, &objective.targets), (ObjectiveKind::Move, &[Target::Tile((26, 10))].into()), "the stairs");
    assert_eq!(objective.next_rooms, [MapLocation::new(8, 0x2f, true)].into());
    w.s.apply_ram(&[(Var::Bombs, 4)]);
    let objective = look(&mut w, true);
    assert_eq!(objective.kind, ObjectiveKind::Fight, "{objective:?}");
}

/// The key doors off the route: at the mouth of 0x5e's E door and of 0x4e's W door MOVE into it is not offered and a
/// MOVE played anyway spends no key; by the controller one frame there spends one. 0x4e's N key door, the route's
/// own, is offered.
#[test]
fn the_key_doors_off_the_route_are_masked_and_a_move_ends_at_their_mouth() {
    for (room, x, y, d, own) in [(0x5e, 0xD0u8, 0x8Du8, Direction::E, None), (0x4e, 0x20, 0x8D, Direction::W, Some(Direction::N))] {
        let Some(mut w) = walk_to(uw(room)) else { return };
        let keys = w.s.state().keys();
        assert!(keys >= 2);
        place(&mut w, x, y, d);
        look(&mut w, false);
        let into = Action::new(ActionKind::Move, d, 32).unwrap();
        assert!(w.s.state().at_doorway_mouth(d), "8:{room:#04x}: {}", describe(&w.s));
        assert!(!w.s.mask().is_allowed(&into), "8:{room:#04x}: {}", w.s.mask());
        w.s.act(into).unwrap();
        assert_eq!((w.s.state().keys(), w.s.state().full_location()), (keys, uw(room)));
        if let Some(own) = own {
            place(&mut w, 0x78, 0x5D, own);
            look(&mut w, false);
            assert!(w.s.mask().is_allowed(&Action::new(ActionKind::Move, own, 8).unwrap()), "{}", w.s.mask());
            place(&mut w, x, y, d);
            look(&mut w, false);
        }
        // The game's rule, by the controller: one frame of the direction at the mouth spends a key.
        w.s.press(direction_buttons(d), 1);
        assert_eq!(w.s.state().keys(), keys - 1, "8:{room:#04x}: {}", describe(&w.s));
    }
}

/// 0x6e: boss-manhandla is `free`, and the room is `requires_bombs`; on the route's step (`bombs: required`) BOMB
/// is not offered in the middle of the room with the Manhandla alive and bombs in the bag, and is offered at the
/// wall's spot. With the step's mode taken away (the family's own `free`) it is offered in the middle too.
#[test]
fn the_manhandla_room_is_crossed_by_the_bomb_mask_of_required() {
    let Some(mut w) = walk_into(uw(0x6e), MANHANDLA) else { return };
    assert_eq!(w.gc.step_bombs(w.s.state()), Some("required"));
    w.s.apply_ram(&[(Var::Bombs, 4)]);
    let bombs = |w: &Walk| {
        [Direction::N, Direction::S, Direction::W, Direction::E]
            .into_iter()
            .filter(|&d| w.s.mask().is_allowed(&Action::new(ActionKind::Bomb, d, 0).unwrap()))
            .collect::<Vec<_>>()
    };
    assert!((1..=11u16).any(|slot| w.s.state().peek(OBJ_TYPE + slot) == MANHANDLA), "the Manhandla lives");
    place(&mut w, 0x78, 0x9D, Direction::N);
    let objective = look(&mut w, true);
    assert_eq!(objective.kind, ObjectiveKind::Move, "crossed, not fought: {objective:?}");
    assert_eq!(bombs(&w), vec![], "{}", w.s.mask());
    // The family's own mode there: BOMB wherever the game places one.
    w.s.set_bomb_targets(None);
    w.s.set_disabled_kinds(&[]);
    assert!(!bombs(&w).is_empty(), "{}", w.s.mask());
    place(&mut w, 0x78, 0x5D, Direction::N);
    look(&mut w, true);
    assert!(bombs(&w).contains(&Direction::N), "{}", w.s.mask());
}

/// 0x4c (pols-voice, `arrows: true`): with its Pols Voices alive ARROW is offered while Link holds more rupees than
/// the route's reserve there (8), not at 8, and a shot costs one rupee; in a Darknut room of the same dungeon (0x5e,
/// darknut: no arrows) it is never offered.
#[test]
fn arrows_are_offered_among_the_pols_voices_above_the_rupee_reserve() {
    let arrows = |w: &Walk| {
        [Direction::N, Direction::S, Direction::W, Direction::E]
            .into_iter()
            .filter(|&d| w.s.mask().is_allowed(&Action::new(ActionKind::Arrow, d, 0).unwrap()))
            .collect::<Vec<_>>()
    };
    // Frames with no input until the room's monsters are out of their clouds (the mask offers the sword).
    let armed = |w: &mut Walk| {
        for _ in 0..200 {
            look(w, true);
            let sword = [ActionKind::Sword, ActionKind::Beams].into_iter().any(|kind| {
                [Direction::N, Direction::S, Direction::W, Direction::E]
                    .into_iter()
                    .any(|d| w.s.mask().is_allowed(&Action::new(kind, d, 0).unwrap()))
            });
            if sword {
                return;
            }
            w.s.press(0, 1);
        }
        panic!("no sword offered: {} {}", describe(&w.s), w.s.mask());
    };
    let Some(mut w) = walk_into(uw(0x5e), BLUE_DARKNUT) else { return };
    armed(&mut w);
    assert!(w.s.state().rupees() > 8 && arrows(&w).is_empty(), "{}", w.s.mask());
    let Some(mut w) = walk_into(uw(0x4c), POLS_VOICE) else { return };
    armed(&mut w);
    assert_eq!((w.gc.step(), w.gc.rupee_reserve()), (Some(290), 8));
    let rupees = w.s.state().rupees();
    let offered = arrows(&w);
    assert!(rupees > 8 && !offered.is_empty(), "{} rupees: {}", rupees, w.s.mask());
    w.s.apply_ram(&[(Var::Rupees, 8)]);
    look(&mut w, true);
    assert!(arrows(&w).is_empty(), "at the reserve: {}", w.s.mask());
    w.s.apply_ram(&[(Var::Rupees, 9)]);
    look(&mut w, true);
    let offered = arrows(&w);
    assert!(!offered.is_empty(), "one above the reserve: {}", w.s.mask());
    w.s.act(Action::new(ActionKind::Arrow, offered[0], 0).unwrap()).unwrap();
    for _ in 0..4 {
        w.s.act(Action::parse("WAIT 4").unwrap()).unwrap();
    }
    assert_eq!(w.s.state().rupees(), 8, "a rupee a shot");
    look(&mut w, true);
    assert!(arrows(&w).is_empty(), "back at the reserve: {}", w.s.mask());
}

/// Round 32: 0x4c needs the route's reserve and a shot (game.yaml `requires_rupees: 9`) while a Pols Voice lives
/// (`zelda::cheats`, as Gohma's room): a purse of 3 (d3first's runs bring 1-10) is topped up to 9, six granted;
/// ARROW is offered (one above the reserve), the shot is one cheated rupee and the purse is topped up again; with
/// the Pols Voices gone the room needs none and grants nothing more.  A purse above 9 (d2first's walk) is untouched.
#[test]
fn the_pols_voices_arrows_are_topped_up_to_the_reserve_and_a_shot() {
    use zelda::cheats::{CheatItem, needs};
    let Some(mut w) = walk_into(uw(0x4c), POLS_VOICE) else { return };
    // Frames with no input until the Pols Voices are out of their clouds (the mask offers the sword).
    for _ in 0..200 {
        look(&mut w, true);
        let sword = [ActionKind::Sword, ActionKind::Beams].into_iter().any(|kind| {
            [Direction::N, Direction::S, Direction::W, Direction::E]
                .into_iter()
                .any(|d| w.s.mask().is_allowed(&Action::new(kind, d, 0).unwrap()))
        });
        if sword {
            break;
        }
        w.s.press(0, 1);
    }
    let here = uw(0x4c);
    let need = |w: &Walk| needs(w.s.game_map(), &w.gc, Some(here), w.s.state())[CheatItem::Rupees];
    assert_eq!(need(&w), 9, "a Pols Voice lives");
    let (rupees, cheated) = (w.s.state().rupees(), w.ledger.cheated.rupees);
    assert!(rupees > 9, "{rupees}");
    w.ledger.after_action(&mut w.s, &w.gc, here);
    assert_eq!((w.s.state().rupees(), w.ledger.granted(CheatItem::Rupees)), (rupees, 0), "d2first's purse: untouched");
    w.s.apply_ram(&[(Var::Rupees, 3)]);
    w.ledger.after_action(&mut w.s, &w.gc, here);
    assert_eq!((w.s.state().rupees(), w.ledger.granted(CheatItem::Rupees)), (9, 6), "topped up to 9");
    look(&mut w, true);
    let arrow = [Direction::N, Direction::S, Direction::W, Direction::E]
        .into_iter()
        .map(|d| Action::new(ActionKind::Arrow, d, 0).unwrap())
        .find(|a| w.s.mask().is_allowed(a))
        .expect("ARROW offered one above the reserve");
    w.s.act(arrow).unwrap();
    w.ledger.after_action(&mut w.s, &w.gc, here);
    for _ in 0..4 {
        w.s.act(Action::parse("WAIT 4").unwrap()).unwrap();
        w.ledger.after_action(&mut w.s, &w.gc, here);
    }
    let alive = (1..=11u16).any(|slot| w.s.state().peek(OBJ_TYPE + slot) == POLS_VOICE);
    assert_eq!(w.ledger.cheated.rupees - cheated, 1, "the shot's rupee was granted");
    if alive {
        assert_eq!((w.s.state().rupees(), w.ledger.granted(CheatItem::Rupees)), (9, 6), "topped up again");
    }
    clear_room(&mut w.s);
    assert_eq!(need(&w), 0, "no Pols Voice: no need");
    let granted = w.ledger.granted(CheatItem::Rupees);
    w.ledger.after_action(&mut w.s, &w.gc, here);
    assert_eq!(w.ledger.granted(CheatItem::Rupees), granted, "nothing more granted; leaving takes it back");
}
