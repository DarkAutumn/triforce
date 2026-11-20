//! Round 29: `d2first`'s steps 293-353 since round 31 (scouted as `d9`) played in the real game by its objective,
//! from Dungeon 8's triforce through the leg (the whirlwind out of 0x6d, 0x47's burned tree and heart container, the
//! ladder over 0x17's river, the rock wall of 0x05), Dungeon 9 with the red ring and the silver arrow, Ganon, the
//! Triforce of Power and Zelda's room to the first frame of game mode `$13`, on both ways of the block clips (the
//! stretches `ring-07`, `arrow-20`, `arrow-20n`: clips on, 0x55, 0x07, 0x20 and 0x03 left by CLIP with their monsters
//! alive; clips off, emptied and their blocks pushed): the agent's own actions (MOVE, RECORDER, CANDLE, BOMB, SWORD,
//! ARROW, CLIP) through the session's masks with the env's per-step rules mirrored (family `items:`/`clips:`, a
//! step's own bomb mode and `sword: false`, the route's bomb and rupee reserves, RECORDER's, ARROW's and CANDLE's
//! facings, the doorways kept out of), the cheat ledger settling after every action, every room change judged by
//! `exit_is_correct`, and per step what the step means (the walk's [`Tick`]s). Beside the walk: a start in a room the
//! route comes back to enters it with the kills the game keeps on the way it is played; the potion of 0x47 is
//! `missed`; 0x66 with seven pieces is `missed`; Dungeon 9's key doors off the route are masked; ARROW in Ganon's
//! room.
//!
//! SETUP. The start is staged by the route's own walk: `7_2bw_0` (the D7 triforce step) is walked by round 27's and
//! round 28's rules to Dungeon 8's triforce (`crates/zelda/tests/d6.rs` and `d8.rs` assert those stretches; their
//! walker is repeated here), so Link stands in 0:0x6d with what that walk brings: 14 hearts, the magic sword, 45
//! rupees, 3 bombs, 3 keys, the whirlwind index `$F9`. RAM writes from there on, every action:
//! - monster shots, Bubbles, Traps and dropped items are removed (object slots 1-11);
//! - counting monsters are **pinned** inside the NW wall (their position rewritten: a room whose slots are emptied
//!   is "all dead" to the game, which shows its room item and opens its shutters), but while the objective is a
//!   FIGHT they are put into the death state (`ObjMetastate` `$10`, what a kill sets), so the game banks the kills as
//!   a played run's and the route's return visits find what it keeps;
//! - Ganon, Zelda and her guard fires are left alone: Ganon is hit by the walk's own SWORD and ARROW, the fires by
//!   its SWORD (his fireballs are removed with the shots: dodging them is the model's work);
//! - the damage-free kill streak (`HelpDropCount`) is kept at 0: the staged kills would arm the tenth-kill bomb.
//!
//! Nothing else is written but where a test says so. Walker aids (controller input, a frame at a time): cave mouths,
//! key doors' mouths, push spots, Link's place beside the hidden Ganon, Link's square under Zelda's fires, and the
//! grid point a shorter skip leaves from (where a MOVE stops depends on Link's movement fraction: a MOVE 4 goes 4 or 5
//! pixels, so 1-3 pixels off the grid point no MOVE lands on it).
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
use zelda::enums::enemy::{
    BLUE_WIZZROBE, FLYING_GHINI, GANON, GLEEOK4, LIKE_LIKE, PATRA, PATRA2, PATRA_CHILD1, PATRA_CHILD2, RED_GOHMA,
    RED_LAMNOLA, RED_WIZZROBE, ZOL,
};
use zelda::executor::direction_buttons;
use zelda::game::gleeok_heads;
use zelda::game_map::Secret;
use zelda::objectives::{
    GANON_JUMP_SPOTS, GameCompletion, GanonPhase, Objective, ObjectiveKind, doorways_kept_out, exit_is_correct,
    banked_kills_follow_clips, game_ended, ganon, gohma, gohma_stand, start_banked_kills, zelda_tiles,
};
use zelda::room::{Target, Wavefront};
use zelda::{Action, ActionKind, Direction, GameMap, GameView, LoadOptions, MapLocation, Session, SessionSnapshot, Var};

type S = Session<Counted>;
type Snapshot = SessionSnapshot<zcore::MachineState>;

const ROUTE: &str = zelda::game_map::DEFAULT_ROUTE;
const OBJ_TYPE: u16 = 0x34F;
const OBJ_X: u16 = 0x70;
const OBJ_Y: u16 = 0x84;
/// `ObjMetastate` (`$405` + slot): `$10` starts a death sparkle (what a kill sets).
const OBJ_METASTATE: u16 = 0x405;
/// `LevelKillCounts` (`$560` + room): a dungeon room's banked kills, `$0F` = all dead.
const LEVEL_KILL_COUNTS: u16 = 0x560;
/// `IsRoomAllDead` (`$34D`).
const ROOM_ALL_DEAD: u16 = 0x34D;
/// `HelpDropCount` (`$50`).
const HELP_DROP_COUNT: u16 = 0x50;
/// `Gohma_ObjEyeState` (`$46B` + slot): 3 while the eye is fully open.
const GOHMA_EYE: u16 = 0x46B;
/// The invincibility timer of the slot Gleeok's attached heads share (`$4F0` + 5): a hit lands only at 0.
const GLEEOK_HEAD_TIMER: u16 = 0x4F0 + 5;
/// Link's row under Gleeok (docs/routes/d8-walks.md §6.1).
const GLEEOK_ROW: i32 = 0x9D;
/// Object types: a Bubble's three kinds, the Traps, the guard fire, the standing fire, Zelda, the old men.
const BUBBLES: std::ops::RangeInclusive<u8> = 0x2B..=0x2D;
const TRAPS: [u8; 2] = [0x49, 0x4A];
const GUARD_FIRE: u8 = 0x3F;
const STANDING_FIRE: u8 = 0x40;
const ZELDA: u8 = 0x37;
/// The first step of the stretch (0x6d out of Dungeon 8) and the route's length (both ways of its clip stretches).
const FIRST: usize = 293;
const STEPS: usize = 354;

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

/// Object types the staging walk (to the D8 triforce, as `d8.rs`) keeps.
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

/// The staging walk's room rule (`d8.rs`): every monster slot emptied but the kept types.
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

/// A monster that keeps a room from being "all dead" (asm:Secret_UpdateRoomAllDead): types `$01`-`$48` but the
/// Bubbles; of those the walk leaves Ganon, Zelda and the fires alone.
fn counting(t: u8) -> bool {
    (1..0x49).contains(&t) && !BUBBLES.contains(&t) && ![GANON, ZELDA, GUARD_FIRE, STANDING_FIRE, 0x2E, 0x2F].contains(&t)
}

/// This stretch's room rule (the module doc): shots, Bubbles, Traps and drops removed; counting monsters pinned, or
/// (`kill`) put into the death state; the kill streak kept at 0.
fn settle_room(s: &mut S, kill: bool) {
    let st = s.state();
    if !matches!(st.mode(), 5 | 9) {
        return;
    }
    let ram = st.ram().to_vec();
    let nes = s.nes_mut();
    zelda::Nes::poke(nes, HELP_DROP_COUNT, 0);
    for slot in 1..=11u16 {
        let t = ram[(OBJ_TYPE + slot) as usize];
        if (0x53..=0x5C).contains(&t) || BUBBLES.contains(&t) || TRAPS.contains(&t) || t == 0x60 {
            zelda::Nes::poke(nes, OBJ_TYPE + slot, 0);
        } else if counting(t) {
            if kill {
                if ram[(OBJ_METASTATE + slot) as usize] == 0 {
                    zelda::Nes::poke(nes, OBJ_METASTATE + slot, 0x10);
                }
            } else {
                zelda::Nes::poke(nes, OBJ_X + slot, 0x00);
                zelda::Nes::poke(nes, OBJ_Y + slot, 0x3D);
            }
        }
    }
    s.apply_ram(&[]);
}

struct Walk {
    s: S,
    gc: GameCompletion,
    ledger: CheatLedger,
    families: BTreeMap<String, Family>,
    ticks: Vec<Tick>,
    /// Round 29's stretch: [`settle_room`] instead of the staging walk's [`clear_room`].
    stretch: bool,
    /// A room whose monsters the walk leaves alone (the room as the game makes it).
    spare: Option<MapLocation>,
    /// The pre-entry frame of Link's room (the scroll into it): what a start state of the step is.
    entry: Option<Snapshot>,
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
    x: i32,
    y: i32,
    facing: Direction,
    mode: u8,
    /// Object types in slots 1-11 (before the walk's room rule).
    types: Vec<u8>,
    /// The room's stairs tiles (an overworld secret's once it shows).
    stairs: Vec<Tile>,
    /// The directions the session's mask offers ARROW, BOMB, CANDLE and MOVE in; whether it offers SWORD or BEAMS,
    /// and RECORDER.
    arrows: Vec<Direction>,
    bombs: Vec<Direction>,
    candles: Vec<Direction>,
    sword: bool,
    recorder: bool,
    /// The route's rules for the state.
    kept_out: u8,
    step_bombs: Option<String>,
    bomb_reserve: u8,
    rupee_reserve: u8,
    /// Ganon's phase, when he is in the room.
    ganon: Option<GanonPhase>,
    /// `LevelKillCounts` of Link's room and `IsRoomAllDead`.
    banked: u8,
    all_dead: bool,
    fire: bool,
    hearts: f64,
    max_hearts: i32,
    rupees: i32,
    bomb_count: i32,
    keys: i32,
    ring: bool,
    silver: bool,
    /// Link holds the Triforce of Power.
    power: bool,
    /// Where the offered BOMBs land (`zelda::bombs::landing`).
    landings: Vec<(i32, i32)>,
}

fn ow(room: u8) -> MapLocation {
    MapLocation::new(0, room, false)
}

fn uw(room: u8) -> MapLocation {
    MapLocation::new(9, room, false)
}

fn cellar(room: u8) -> MapLocation {
    MapLocation::new(9, room, true)
}

/// The ticks of `room` on route step `step`.
fn at(ticks: &[Tick], room: MapLocation, step: usize) -> Vec<&Tick> {
    ticks.iter().filter(|t| t.room == room && t.step == Some(step)).collect()
}

const CARDINALS: [Direction; 4] = [Direction::N, Direction::S, Direction::W, Direction::E];

fn offered(s: &S, kind: ActionKind) -> Vec<Direction> {
    let duration = if kind == ActionKind::Move { 8 } else { 0 };
    CARDINALS.into_iter().filter(|&d| Action::new(kind, d, duration).is_ok_and(|a| s.mask().is_allowed(&a))).collect()
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
    /// the step's own bomb mode and sword switch and the route's reserves, the bomb targets of `required`, the
    /// facings of RECORDER, ARROW and CANDLE, the doorways kept out of.
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
        let sword = self.gc.step_sword(&state);
        let off: Vec<ActionKind> = [
            (bombs, ActionKind::Bomb),
            (true, ActionKind::Boomerang),
            (arrows, ActionKind::Arrow),
            (f.recorder, ActionKind::Recorder),
            (f.food, ActionKind::Food),
            (f.candle, ActionKind::Candle),
            (sword, ActionKind::Sword),
            (sword, ActionKind::Beams),
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

/// The path search the walk moves by: the objective's targets, and a locked door Link holds a key for as a way
/// through (as the objective's own path search counts it).
fn walk_wavefront(state: &GameView, objective: &Objective) -> Wavefront {
    let targets: Vec<Target> = objective.targets.iter().copied().collect();
    let locked: Vec<Direction> = CARDINALS
        .into_iter()
        .filter(|&d| state.keys() > 0 && !state.full_location().in_cave && state.is_door_locked(d))
        .collect();
    state.room().wavefront(&targets, &[], &locked, objective.clips)
}

fn next_action(s: &S, objective: &Objective, nudge: u32) -> Action {
    let state = s.state();
    let mask = s.mask();
    let wait = Action::parse("WAIT 4").unwrap();
    if !matches!(state.mode(), 5 | 9 | 11) || state.link.status != 0 {
        return wait;
    }
    // The route's item plays, where the mask offers them: the recorder (the whirlwind), the candle (a tree), the
    // arrow (the brown Ganon), and on the overworld the bomb of a `required` step (the rock wall: offered only in
    // reach of it).
    let recorder = Action::new(ActionKind::Recorder, Direction::None, 0).unwrap();
    if mask.is_allowed(&recorder) {
        return recorder;
    }
    for kind in [ActionKind::Candle, ActionKind::Arrow] {
        for d in [Direction::W, Direction::E, Direction::N, Direction::S] {
            let item = Action::new(kind, d, 0).unwrap();
            if mask.is_allowed(&item) {
                return item;
            }
        }
    }
    if state.level() == 0 && objective.kind == ObjectiveKind::Cave && state.ow_bomb_secret_closed() {
        let bomb = Action::new(ActionKind::Bomb, Direction::N, 0).unwrap();
        if mask.is_allowed(&bomb) {
            return bomb;
        }
    }
    let room = state.room();
    let wavefront = walk_wavefront(state, objective);
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
        for d in CARDINALS {
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
        for d in CARDINALS {
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
    // Off the grid point of a tile a shorter jump leaves from (the skip is offered only on the grid point): onto it.
    if wavefront.jumps_from(here).any(|jump| wavefront.get(jump.to).is_some_and(|to| to < distance)) {
        let (gx, gy) = (here.0 * 8, here.1 * 8 + 0x3D);
        let toward = if x != gx {
            Some(if x > gx { Direction::W } else { Direction::E })
        } else if y != gy {
            Some(if y > gy { Direction::N } else { Direction::S })
        } else {
            None
        };
        if let Some(d) = toward.filter(|&d| mask.is_allowed(&moving(d))) {
            return moving(d);
        }
    }
    let mut best: Option<(i32, Action)> = None;
    for i in 0..4 {
        let d = CARDINALS[(i + nudge as usize) % 4];
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
    /// `7_2bw_0` (Link entering Dungeon 7's triforce room) with a completion of `d9` on the D7 triforce step (247)
    /// and the ledger begun.
    fn start() -> Option<Walk> {
        let core = open_core()?;
        let map = map();
        let mut s = Session::new(Counted(core, 0), map.clone());
        s.load(&load_named_state("7_2bw_0"), &LoadOptions::default(), &mut ChaCha8Rng::seed_from_u64(0)).unwrap();
        s.set_capture_pre_entry(true);
        let first = map.routes[ROUTE].iter().position(|s| s.level == 7 && s.exit == "triforce").unwrap();
        let gc = GameCompletion::new(map.clone(), ROUTE, Some(first)).unwrap();
        let (ledger, families) = (CheatLedger::default(), BTreeMap::new());
        let mut w = Walk { s, gc, ledger, families, ticks: Vec::new(), stretch: false, spare: None, entry: None };
        w.ledger.begin(&mut w.s, &w.gc);
        Some(w)
    }

    /// The staged start of the stretch: the walk of round 27's and round 28's steps (248-296, `d6.rs` and `d8.rs`)
    /// until Link holds Dungeon 8's piece (the action that takes it ends with the game's own warp: he stands at its
    /// entrance, 0x6d); the ticks begin anew and the stretch's room rule takes over.
    fn at_the_d8_triforce() -> Option<Walk> {
        let mut w = Walk::start()?;
        w.run(|s| s.state().has_triforce(8), 120_000);
        assert_eq!(w.s.state().full_location(), ow(0x6d));
        assert_eq!(w.ledger.cheated.bombs + w.ledger.cheated.keys + w.ledger.cheated.rupees, 0);
        w.ticks.clear();
        w.stretch = true;
        Some(w)
    }

    /// The run-time clips switch off from here: the plain way through the stretch's block rooms.
    fn clips_off(&mut self) {
        self.s.set_clips_allowed(false);
        self.gc.set_clips_allowed(false);
    }

    /// The tick before `action` (`types`: the object types before the walk's room rule).
    fn tick(&self, objective: &Objective, types: Vec<u8>, action: Action) -> Tick {
        let state = self.s.state();
        let map = self.s.game_map();
        let sword = [ActionKind::Sword, ActionKind::Beams].into_iter().any(|kind| !offered(&self.s, kind).is_empty());
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
            types,
            stairs: if state.full_location().in_cave { Vec::new() } else { state.room().stairs_tiles() },
            arrows: offered(&self.s, ActionKind::Arrow),
            bombs: offered(&self.s, ActionKind::Bomb),
            candles: offered(&self.s, ActionKind::Candle),
            sword,
            recorder: self.s.mask().is_allowed(&Action::new(ActionKind::Recorder, Direction::None, 0).unwrap()),
            kept_out: doorways_kept_out(map, state, objective),
            step_bombs: self.gc.step_bombs(state).map(str::to_owned),
            bomb_reserve: self.gc.bomb_reserve(),
            rupee_reserve: self.gc.rupee_reserve(),
            ganon: ganon(state).map(|(_, phase)| phase),
            banked: state.peek(LEVEL_KILL_COUNTS + u16::from(state.location())),
            all_dead: state.peek(ROOM_ALL_DEAD) != 0,
            fire: state.link_fire_burning(),
            hearts: state.health(),
            max_hearts: state.max_health(),
            rupees: state.rupees(),
            bomb_count: state.bombs(),
            keys: state.keys(),
            ring: state.red_ring(),
            silver: state.silver_arrow(),
            power: state.triforce_of_power(),
            landings: offered(&self.s, ActionKind::Bomb)
                .into_iter()
                .filter_map(|d| zelda::bombs::landing(state, d).map(|p| (p.x, p.y)))
                .collect(),
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

    /// Walker aid (controller input): Link stands 1-3 pixels off the grid point of a tile a shorter jump leaves from
    /// (the skip is offered only on the grid point, and no MOVE stops there: the module doc): onto it.
    fn skip_grid_aid(&mut self, objective: &Objective) {
        let state = self.s.state();
        if state.mode() != 5 || state.level() == 0 || !objective.clips {
            return;
        }
        let wavefront = walk_wavefront(state, objective);
        let here = state.link.tile();
        let Some(distance) = wavefront.get(here) else { return };
        if !wavefront.jumps_from(here).any(|jump| wavefront.get(jump.to).is_some_and(|to| to < distance)) {
            return;
        }
        let (x, y) = (state.link.position.x, state.link.position.y);
        let (gx, gy) = (here.0 * 8, here.1 * 8 + 0x3D);
        let off = (x - gx).abs() + (y - gy).abs();
        if (1..4).contains(&off) && (x == gx || y == gy) {
            self.line_up((gx, gy), x != gx);
        }
    }

    /// Gohma's script (round 27, `d6.rs`).
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

    /// Gleeok's script (round 28, `d8.rs`).
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

    /// Ganon's script (docs/routes/d9-fights-walks.md §4, the rule with his position from RAM, without its dodging:
    /// the walk removes his fireballs). Hidden: Link is kept, a frame at a time (walker aid), on the square a stab
    /// reaches him from without his touch (his left: Link's X − his in −16..−1; by the W wall his right, +17..+39;
    /// Link's Y − his in −4..+16), and SWORD toward him is played when the mask offers it. Shown: the walk waits
    /// where it stands (nothing hits him; he jumps away). Brown and dying are the objective's (the stand tiles and
    /// ARROW as the mask offers it; his ashes). None when no Ganon is here or the objective's own play follows.
    fn ganon_action(&mut self, objective: &Objective) -> Option<Action> {
        let state = self.s.state().clone();
        let (_, phase) = ganon(&state)?;
        let wait = Action::parse("WAIT 4").unwrap();
        if state.mode() != 5 || state.link.status != 0 {
            return Some(wait);
        }
        match phase {
            GanonPhase::Scene | GanonPhase::Shown => Some(wait),
            GanonPhase::Brown => None,
            GanonPhase::Dying => objective.targets.contains(&Target::Tile(state.link.tile())).then_some(wait),
            GanonPhase::Hidden => {
                let stand = |gx: i32, gy: i32| {
                    let left = gx - 12 >= 0x20;
                    ((if left { gx - 12 } else { gx + 28 }).clamp(0x20, 0xD0), (gy + 6).clamp(0x5D, 0xBD), left)
                };
                for _ in 0..12 {
                    let st = self.s.state();
                    let Some((g, GanonPhase::Hidden)) = ganon(st) else { break };
                    let (tx, ty, _) = stand(g.position.x, g.position.y);
                    let p = st.link.position;
                    let d = if (p.y - ty).abs() > 4 {
                        if p.y < ty { Direction::S } else { Direction::N }
                    } else if (p.x - tx).abs() > 3 {
                        if p.x < tx { Direction::E } else { Direction::W }
                    } else {
                        break;
                    };
                    self.s.press(direction_buttons(d), 1);
                }
                let st = self.s.state();
                let Some((g, GanonPhase::Hidden)) = ganon(st) else { return Some(wait) };
                let (_, _, left) = stand(g.position.x, g.position.y);
                let (dx, dy) = (st.link.position.x - g.position.x, st.link.position.y - g.position.y);
                let reach = (-4..=16).contains(&dy) && if left { (-16..=-1).contains(&dx) } else { (17..=39).contains(&dx) };
                let facing = if left { Direction::E } else { Direction::W };
                let swing = [ActionKind::Sword, ActionKind::Beams]
                    .into_iter()
                    .map(|kind| Action::new(kind, facing, 0).unwrap())
                    .find(|a| self.s.mask().is_allowed(a));
                Some(match swing {
                    Some(a) if reach && st.link.status == 0 => a,
                    _ => wait,
                })
            }
        }
    }

    /// Zelda's room (d9-fights-walks.md §5, "stab"): while a guard fire stands on the way, Link is lined up on the
    /// square under the two fires of the path ((`$78`, `$AD`), walker aid) and one SWORD N is played. None once
    /// they are out, or in any other room.
    fn zelda_action(&mut self) -> Option<Action> {
        let state = self.s.state().clone();
        if !self.gc.ends_game(&state) || state.mode() != 5 || state.link.status != 0 {
            return None;
        }
        let on_path = (1..=11u16).any(|slot| state.peek(OBJ_TYPE + slot) == GUARD_FIRE && state.peek(OBJ_Y + slot) == 0x9D);
        if !on_path || state.link.position.y < 0xAD {
            return None;
        }
        self.line_up((0x78, 0xAD), false);
        let swing = Action::new(ActionKind::Sword, Direction::N, 0).unwrap();
        Some(if self.s.mask().is_allowed(&swing) { swing } else { Action::parse("WAIT 4").unwrap() })
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
                if self.stretch {
                    // The fight of the tick before, in this room: its monsters die the game's way.
                    let same = prev.as_ref().is_some_and(|p| p.full_location() == self.s.state().full_location());
                    settle_room(&mut self.s, same && objective.kind == ObjectiveKind::Fight);
                } else {
                    clear_room(&mut self.s);
                }
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
                self.entry = self.s.last_pre_entry().cloned();
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
            let block = state.level() > 0 && state.push_block_pending() && state.peek(ROOM_ALL_DEAD) != 0;
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
            let key_door = CARDINALS.into_iter().find_map(|d| {
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
            } else if let Some(action) = self.ganon_action(&objective) {
                action
            } else if let Some(action) = self.zelda_action() {
                action
            } else if self.stretch && objective.kind == ObjectiveKind::Fight && ganon(&state).is_none() {
                // The room's monsters are dying by the walk's staging (`settle_room`): the walk stands and waits.
                Action::parse("WAIT 4").unwrap()
            } else {
                self.skip_grid_aid(&objective);
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
                "stuck at {} on step {:?} playing {action} ({:?} {:?}) mask {}; BOMB N: {:?}, lands {:?}, targets {:?}\n{}",
                describe(&self.s),
                self.gc.step(),
                objective.kind,
                objective.targets,
                self.s.mask(),
                zelda::attack::explain(&state, ActionKind::Bomb, Direction::N),
                zelda::bombs::landing(&state, Direction::N),
                zelda::objectives::route_bomb_targets(&map, &state, &objective),
                grid(&self.s, &objective)
            );
            self.ticks.push(self.tick(&objective, types, action));
            let from = state.full_location();
            prev = Some(state);
            self.s.act(action).unwrap();
            self.ledger.after_action(&mut self.s, &self.gc, from);
            assert!(!self.s.state().game_over(), "Link died: {}", describe(&self.s));
        }
        assert!(done(&self.s), "the walk ended at {} ({} rooms)", describe(&self.s), visits.len());
        visits
    }
}

/// The walk from the D8 triforce to the first frame of game mode `$13` with the clips switch off (the plain way:
/// every block room emptied and its block pushed), every step asserted for what it means.
#[test]
fn the_d9_route_is_walked_the_plain_way_from_the_d8_triforce_to_the_end_of_the_game() {
    let Some(mut w) = Walk::at_the_d8_triforce() else { return };
    w.clips_off();
    let map = w.s.game_map().clone();
    let route = &map.routes[ROUTE];
    assert_eq!(route.len(), STEPS);
    let frames_in = w.s.frame();
    let start = (w.s.state().health(), w.s.state().max_health(), w.s.state().rupees(), w.s.state().bombs(), w.s.state().keys());
    let visits = w.run(|s| game_ended(s.state()), 200_000);
    let t = &w.ticks;

    // The rooms in order: the plain way's steps, each entered once; every change was a correct exit (`run`), the
    // whirlwind's landing and the stairs into level 9 included.
    let rooms: Vec<MapLocation> = visits.iter().map(|v| v.room).collect();
    let way: Vec<usize> = (FIRST..STEPS).filter(|&i| route[i].on_way(false)).collect();
    let expected: Vec<MapLocation> = way.iter().map(|&i| route[i].location()).collect();
    assert_eq!(rooms, expected);
    assert!(t.iter().all(|k| k.action.kind != ActionKind::Clip), "no clip on the plain way");
    let steps: Vec<usize> = {
        let mut seen: Vec<usize> = t.iter().filter_map(|k| k.step).collect();
        seen.dedup();
        seen
    };
    assert_eq!(steps, way, "every step of the plain way, in order, and never off the route");
    for (clip, plain) in [(321, 322), (334, 335), (339, 340)] {
        assert!(!way.contains(&clip) && way.contains(&plain), "the stretches' plain-way steps: {plain}, not {clip}");
    }
    assert!(t.iter().all(|k| k.step.is_some() || !matches!(k.mode, 5 | 9 | 11)), "on the route in every playable tick");

    // 0x6d (step 293, `entry: cave`, `exit: recorder`, ow-moblin): Link is set down at Dungeon 8's entrance with
    // its piece and 14 full hearts, facing down; RECORDER is offered at once, played once, then never again; the
    // whirlwind sets him down in 0x37.
    let out = at(t, ow(0x6d), FIRST);
    let first = out[0];
    assert_eq!((route[FIRST].entry.as_str(), route[FIRST].exit.as_str(), first.bomb_reserve), ("cave", "recorder", 5));
    assert_eq!((first.x, first.y, first.facing, first.hearts, first.max_hearts), (0x60, 0x5D, Direction::S, 14.0, 14));
    assert!(first.recorder, "{first:?}");
    assert!(out.iter().all(|k| matches!(k.action.kind, ActionKind::Recorder | ActionKind::Wait) && k.candles.is_empty()));
    let plays: Vec<usize> = out.iter().enumerate().filter(|(_, k)| k.action.kind == ActionKind::Recorder).map(|(i, _)| i).collect();
    assert_eq!(plays.len(), 1, "one play");
    assert!(out[plays[0] + 1..].iter().all(|k| !k.recorder), "no second play");
    let landed = t.iter().find(|k| k.room == ow(0x37) && k.action.kind == ActionKind::Move).unwrap();
    assert_eq!((landed.step, landed.x, landed.y), (Some(294), 0x80, 0x8D), "set down at Dungeon 1's entrance");

    // The leg's crossings (steps 294-296, 299-306): no RECORDER, no CANDLE, no BOMB, no ARROW.
    for (room, step) in [(0x37, 294), (0x38, 295), (0x48, 296), (0x47, 299), (0x48, 300), (0x38, 301), (0x28, 302), (0x27, 303), (0x17, 304), (0x07, 305), (0x06, 306)] {
        let ticks = at(t, ow(room), step);
        assert!(!ticks.is_empty(), "0x{room:02x} on step {step}");
        assert!(ticks.iter().all(|k| k.candles.is_empty() && !k.recorder && k.bombs.is_empty() && k.arrows.is_empty()), "0x{room:02x}");
    }

    // 0x47 (step 297, `exit: cave` under the tree, ow-zora with `candle: true`): while the tree stands the targets
    // hold the tiles of its column above it; CANDLE is offered only there, only facing S, only while no fire of
    // Link's burns, and is played once; then the stairs are the target; no heart is lost.
    let Some(Secret::Burn((c, r))) = map.get(ow(0x47)).unwrap().secret else { panic!("0x47's tree") };
    assert_eq!((c, r), (22, 14));
    let above: BTreeSet<Target> = [(21, 10), (21, 11), (21, 12), (22, 10), (22, 11), (22, 12)].into_iter().map(Target::Tile).collect();
    let tree = at(t, ow(0x47), 297);
    let standing: Vec<_> = tree.iter().filter(|k| k.stairs.is_empty() && k.mode == 5).collect();
    assert!(!standing.is_empty());
    for k in &standing {
        assert!(k.kind == ObjectiveKind::Cave && above.is_subset(&k.targets), "{k:?}");
        let window = (0xA8..=0xB8).contains(&k.x) && (0x8D..=0x9D).contains(&k.y);
        assert_eq!(k.candles, if window && !k.fire { vec![Direction::S] } else { vec![] }, "at ({:#04x}, {:#04x})", k.x, k.y);
    }
    let burns: Vec<_> = tree.iter().filter(|k| k.action.kind == ActionKind::Candle).collect();
    assert_eq!((burns.len(), burns[0].action.direction), (1, Direction::S), "one fire, facing S");
    let shown: Vec<_> = tree.iter().filter(|k| !k.stairs.is_empty()).collect();
    assert!(!shown.is_empty());
    for k in &shown {
        assert_eq!((k.stairs.as_slice(), &k.targets), (&[(c, r)][..], &[Target::Tile((c, r))].into()));
        assert!(k.candles.is_empty(), "no CANDLE once the stairs show");
    }
    assert_eq!(tree.last().unwrap().hearts, 14.0);
    // Its cave (step 298, `waypoint: heart`): the right ware is the target; 15 containers, and (Link was full) 15
    // hearts; then out S.
    let cave = at(t, MapLocation::new(0, 0x47, true), 298);
    assert_eq!((cave[0].kind, &cave[0].targets), (ObjectiveKind::Treasure, &[Target::Tile((19, 11))].into()));
    assert_eq!((cave.last().unwrap().max_hearts, cave.last().unwrap().hearts), (15, 15.0));
    assert_eq!(cave.last().unwrap().kind, ObjectiveKind::Move);

    // 0x17 (step 304): from S, E of the river, to the N exit W of it: the ladder (no heart lost with the Boulders
    // pinned). 0x07 (step 305, `bombs: never`) and 0x06: nothing but MOVE.
    let river = at(t, ow(0x17), 304);
    assert!(river.first().unwrap().x >= 0x70 && river.last().unwrap().x <= 0x48, "over the river");
    assert!(at(t, ow(0x07), 305).iter().all(|k| k.step_bombs.as_deref() == Some("never")));

    // 0x05 (step 307, ow-lynel `tenth`, the step `bombs: required`): the rock wall's four tiles are the target
    // while it stands; BOMB is offered only where it lands in the game's reach of the wall's tile object (the blast
    // within $10 px of (8c, 8r + $40) of its game.yaml tile on both axes: asm:RockWall_Update) - under it facing N,
    // or (round 31: the walk now comes W along the wall's lower row, Y $A5) beside it facing W -, played once (the
    // walker plays N, from under it); then its mouth.
    let wall = at(t, ow(0x05), 307);
    let closed: Vec<_> = wall.iter().filter(|k| k.targets.contains(&Target::Tile((10, 12))) && k.mode == 5).collect();
    assert!(!closed.is_empty() && closed.iter().all(|k| k.kind == ObjectiveKind::Cave && k.step_bombs.as_deref() == Some("required")));
    let rock = (10 * 8, 12 * 8 + 0x40);
    for k in &closed {
        assert_eq!(k.landings.len(), k.bombs.len(), "{k:?}");
        for &(lx, ly) in &k.landings {
            assert!((lx - rock.0).abs() < 0x10 && (ly - rock.1).abs() < 0x10, "BOMB at ({:#04x}, {:#04x}) lands at ({lx:#04x}, {ly:#04x})", k.x, k.y);
        }
    }
    assert!(closed.iter().any(|k| !k.bombs.is_empty()), "BOMB offered at the wall");
    let bombs: Vec<_> = wall.iter().filter(|k| k.action.kind == ActionKind::Bomb).collect();
    assert_eq!((bombs.len(), bombs[0].action.direction), (1, Direction::N));
    assert!((0x48..=0x58).contains(&bombs[0].x), "from under the wall: X {:#04x}", bombs[0].x);
    assert_eq!(wall.last().unwrap().bomb_count, bombs[0].bomb_count - 1, "one bomb");

    // Dungeon 9. What Link holds on his first tick of each room (keys, bombs): 3 keys in; 0x14 E, 0x16 N and 0x63 W
    // take one each, 0x61 gives one; the bombs he brings pay the first walls, the rest are cheated and counted,
    // 0x16's four pay the last four.
    let first = |room: MapLocation, step: usize| *at(t, room, step).first().unwrap_or_else(|| panic!("{room} on step {step}"));
    let keys: Vec<i32> = [(0x76, 308), (0x14, 313), (0x15, 314), (0x16, 326), (0x06, 327), (0x63, 330), (0x62, 331), (0x61, 332), (0x20, 335), (0x42, 352)]
        .into_iter()
        .map(|(r, s)| first(uw(r), s).keys)
        .collect();
    assert_eq!(keys, [3, 3, 2, 2, 1, 1, 0, 0, 1, 1]);
    let reserves: Vec<u8> = [(0x76, 308), (0x56, 310), (0x55, 311), (0x27, 317), (0x17, 318), (0x07, 319), (0x16, 326), (0x06, 327), (0x05, 328), (0x10, 336), (0x30, 346), (0x03, 349)]
        .into_iter()
        .map(|(r, s)| first(uw(r), s).bomb_reserve)
        .collect();
    assert_eq!(reserves, [4, 4, 3, 2, 1, 0, 0, 4, 3, 2, 1, 0]);
    assert!(t.iter().filter(|k| k.room.level == 9 && k.room != uw(0x42) && k.room != uw(0x32)).all(|k| k.rupee_reserve == 1));

    // 0x66 (step 309): with eight pieces the old man ($4B) is there on Link's first frame only; then only the two
    // fires stand, and the N shutter is the way (round 31, phase IX: N into 0x56, not W into 0x65).
    let hall = at(t, uw(0x66), 309);
    let men: Vec<Vec<u8>> = hall.iter().map(|k| k.types.iter().copied().filter(|ty| *ty != 0).collect()).collect();
    assert!(men[1..].iter().all(|types| types == &[STANDING_FIRE, STANDING_FIRE]), "the old man is gone: {men:02x?}");

    // The eight bomb walls of the way: BOMB is played once in each wall room on the visit that opens it, facing
    // the wall, and offered nowhere else in level 9 but there.
    let walls = [(0x56, 310, Direction::W), (0x26, 316, Direction::E), (0x27, 317, Direction::N), (0x17, 318, Direction::N), (0x06, 327, Direction::W), (0x20, 335, Direction::N), (0x31, 345, Direction::W), (0x04, 348, Direction::W)];
    for (room, step, d) in walls {
        let ticks = at(t, uw(room), step);
        let bombs: Vec<_> = ticks.iter().filter(|k| k.action.kind == ActionKind::Bomb).collect();
        assert_eq!(bombs.iter().map(|k| k.action.direction).collect::<Vec<_>>(), [d], "9:{room:#04x}");
    }
    let bombed: usize = t.iter().filter(|k| k.room.level == 9 && k.action.kind == ActionKind::Bomb).count();
    assert_eq!(bombed, 8);
    let wall_steps: Vec<usize> = walls.iter().map(|w| w.1).collect();
    assert!(t.iter().filter(|k| k.room.level == 9 && !k.bombs.is_empty()).all(|k| wall_steps.contains(&k.step.unwrap())), "BOMB only at the route's walls");

    // The rooms emptied for their blocks, shutters and bombs: full when Link comes in, FIGHT is the objective while
    // a counting monster lives, and (but for 0x14 and 0x20's first visit, which leave by a door) the stairs or the
    // shutter follow.
    let fights: [(u8, usize, &[u8]); 11] = [
        (0x55, 311, &[RED_LAMNOLA]),
        (0x14, 313, &[LIKE_LIKE]),
        (0x07, 319, &[BLUE_WIZZROBE, RED_WIZZROBE]),
        (0x16, 326, &[PATRA2, PATRA_CHILD2]),
        (0x05, 328, &[BLUE_WIZZROBE, RED_WIZZROBE]),
        (0x61, 332, &[PATRA, PATRA_CHILD1]),
        (0x20, 335, &[BLUE_WIZZROBE, RED_WIZZROBE]),
        (0x10, 336, &[BLUE_WIZZROBE, RED_WIZZROBE]),
        (0x30, 346, &[BLUE_WIZZROBE, RED_WIZZROBE]),
        (0x03, 349, &[ZOL, LIKE_LIKE]),
        (0x52, 351, &[PATRA, PATRA_CHILD1]),
    ];
    for (room, step, monsters) in fights {
        let ticks = at(t, uw(room), step);
        for m in monsters {
            assert!(ticks.iter().any(|k| k.types.contains(m)), "9:{room:#04x}: no {m:#04x}");
        }
        assert!(ticks.iter().any(|k| k.kind == ObjectiveKind::Fight), "9:{room:#04x}: fought");
        assert!(ticks.iter().filter(|k| k.kind == ObjectiveKind::Fight).all(|k| k.bombs.is_empty()), "9:{room:#04x}: no BOMB in the fight");
    }
    // A Patra's body is no target while a child lives: every FIGHT target lies on a child then.
    for (room, step) in [(0x16, 326), (0x61, 332), (0x52, 351)] {
        let with_children: Vec<_> = at(t, uw(room), step)
            .into_iter()
            .filter(|k| k.kind == ObjectiveKind::Fight && k.types.iter().any(|ty| matches!(*ty, PATRA_CHILD1 | PATRA_CHILD2)))
            .collect();
        assert!(!with_children.is_empty(), "9:{room:#04x}");
    }
    // The crossings: never a FIGHT (their monsters pinned, alive); 0x56 (round 31) leaves its key.
    for (room, step) in [(0x56, 310), (0x15, 314), (0x16, 315), (0x26, 316), (0x27, 317), (0x17, 318), (0x17, 323), (0x27, 324), (0x26, 325), (0x63, 330), (0x62, 331), (0x51, 343), (0x41, 344), (0x31, 345), (0x04, 348)] {
        let ticks = at(t, uw(room), step);
        assert!(!ticks.is_empty() && ticks.iter().all(|k| k.kind != ObjectiveKind::Fight), "9:{room:#04x} on step {step}");
    }
    // 0x16 from S (step 326): the four bombs are the target once the Patra is gone, then the N key door.
    let patra = at(t, uw(0x16), 326);
    assert!(patra.iter().any(|k| k.kind == ObjectiveKind::Treasure), "the bombs are a target");
    assert_eq!(patra.last().unwrap().bomb_count - patra.first().unwrap().bomb_count, 4);
    // 0x61 (step 332): its key first (it lies in the open), then the fight.
    let keyroom = at(t, uw(0x61), 332);
    assert_eq!(keyroom[0].kind, ObjectiveKind::Treasure);
    assert!(keyroom.iter().filter(|k| k.keys == 0 && k.mode == 5).all(|k| k.kind == ObjectiveKind::Treasure), "the key first");

    // The red ring (step 320) and the silver arrow (step 337): the item is the cellar's target, taken, and every
    // later tick holds it.
    for (room, step, ring) in [(0x00, 320, true), (0x4f, 337, false)] {
        let ticks = at(t, cellar(room), step);
        assert_eq!(ticks[0].kind, ObjectiveKind::Treasure, "9:{room:#04x}");
        let held = |k: &Tick| if ring { k.ring } else { k.silver };
        assert!(!held(ticks[0]) && held(ticks.last().unwrap()) && ticks.last().unwrap().kind == ObjectiveKind::Move, "9:{room:#04x}");
        assert!(t.iter().filter(|k| k.step.is_some_and(|s| s > step)).all(held));
        assert!(t.iter().filter(|k| k.step.is_some_and(|s| s < step)).all(|k| !held(k)));
    }

    // The tunnels: in by one ladder, out by the other (0x60, 0x70, 0x77 right to left; 0x67 left to right; 0x75
    // right to left before the arrow, left to right with it).
    for (room, step, in_right) in [(0x60, 312, true), (0x70, 329, true), (0x75, 333, true), (0x75, 341, false), (0x67, 347, false), (0x77, 350, true)] {
        let ticks = at(t, cellar(room), step);
        let (a, b) = (ticks.first().unwrap().x, ticks.last().unwrap().x);
        assert_eq!((a >= 0x80, b >= 0x80), (in_right, !in_right), "9:{room:#04x} on step {step}: X {a:#04x} to {b:#04x}");
    }

    // The kills the game keeps: the plain way's `banked_kills` are what `LevelKillCounts` holds when Link comes back,
    // and the rooms hold what the game leaves in them (0x07 and 0x10 their three Bubbles, 0x20 and 0x61 nothing),
    // with nothing of the walk's written in between: no FIGHT there, and 0x20's block moves at once.
    for (room, step) in [(0x07, 322), (0x10, 338), (0x20, 340), (0x61, 342)] {
        let ticks = at(t, uw(room), step);
        let banked = route[step].banked_kills.unwrap();
        assert!(ticks.iter().all(|k| k.banked == banked), "9:{room:#04x}: banked {:?}, the route says {banked}", ticks[0].banked);
        assert!(ticks.iter().all(|k| !k.types.iter().any(|ty| counting(*ty))), "9:{room:#04x}: a monster is back");
        assert!(ticks.iter().all(|k| k.kind != ObjectiveKind::Fight), "9:{room:#04x}");
        let bubbles = ticks.iter().map(|k| k.types.iter().filter(|ty| BUBBLES.contains(ty)).count()).max().unwrap();
        assert_eq!(bubbles, if matches!(room, 0x07 | 0x10) { 3 } else { 0 }, "9:{room:#04x}");
    }
    assert!(at(t, uw(0x20), 340).iter().any(|k| k.all_dead && k.action.kind == ActionKind::Move && k.action.duration == 16), "the block is pushed");

    // 0x06 (step 327, `sword: false`): the old man is there, Link is halted by his text, SWORD and BEAMS are never
    // offered, and the W wall is bombed.
    let man = at(t, uw(0x06), 327);
    assert!(man.iter().any(|k| k.types.iter().any(|ty| (0x4B..=0x52).contains(ty))), "the old man");
    assert!(man.iter().all(|k| !k.sword), "no SWORD at the old man");
    assert!(t.iter().filter(|k| k.room != uw(0x06) && k.mode == 5 && k.room.level == 9 && !k.room.in_cave).any(|k| k.sword));

    // The key doors off the route: kept out of in 0x26 (S and W) on both visits and in 0x63 (N and S); the route's
    // own (0x63 W) is not; no other room of the stretch keeps a doorway out.
    let (s_w, n_s) = (Direction::S.value() | Direction::W.value(), Direction::N.value() | Direction::S.value());
    for (room, step, bits) in [(0x26, 316, s_w), (0x26, 325, s_w), (0x63, 330, n_s)] {
        assert!(at(t, uw(room), step).iter().all(|k| k.kept_out == bits), "9:{room:#04x}");
    }
    assert!(t.iter().filter(|k| ![uw(0x26), uw(0x63)].contains(&k.room)).all(|k| k.kept_out == 0));

    // Ganon (step 352, the family `ganon`: `bombs: never`, `arrows: true`; rupee reserve 0). ARROW is never offered
    // while he is hidden or shown, and BOMB never at all; four SWORD hits land while he is hidden (three shown
    // phases, then brown); brown, the targets are tiles of his row or column and ARROW is offered only toward him
    // from them, played once: a rupee, and he dies; the Triforce of Power is the target on his ashes and is taken;
    // then N.
    let boss = at(t, uw(0x42), 352);
    assert!(boss.iter().all(|k| k.bombs.is_empty() && k.rupee_reserve == 0));
    let blue: Vec<_> = boss.iter().filter(|k| matches!(k.ganon, Some(GanonPhase::Scene | GanonPhase::Hidden | GanonPhase::Shown))).collect();
    assert!(!blue.is_empty() && blue.iter().all(|k| k.arrows.is_empty()), "no ARROW while he is blue");
    assert!(blue.iter().filter(|k| k.mode == 5).all(|k| k.kind == ObjectiveKind::Fight));
    let shown = boss.windows(2).filter(|p| p[0].ganon != Some(GanonPhase::Shown) && p[1].ganon == Some(GanonPhase::Shown)).count();
    assert_eq!(shown, 3, "three hits show him");
    let jump: BTreeSet<Target> = GANON_JUMP_SPOTS
        .iter()
        .flat_map(|&(x, y)| zelda::game::overlap_tiles(zelda::Position::new(x + 8, y + 8), (2, 2)))
        .map(Target::Tile)
        .collect();
    assert!(boss.iter().filter(|k| k.ganon == Some(GanonPhase::Shown)).all(|k| k.targets == jump), "shown: where he jumps to");
    let brown: Vec<_> = boss.iter().filter(|k| k.ganon == Some(GanonPhase::Brown)).collect();
    assert!(!brown.is_empty() && brown.iter().all(|k| k.kind == ObjectiveKind::Fight && !k.targets.is_empty()));
    let shots: Vec<_> = boss.iter().filter(|k| k.action.kind == ActionKind::Arrow).collect();
    assert_eq!(shots.len(), 1, "one arrow");
    assert_eq!(shots[0].ganon, Some(GanonPhase::Brown));
    assert!(shots[0].arrows == [shots[0].action.direction], "ARROW only toward him: {:?}", shots[0].arrows);
    let swings = boss.iter().filter(|k| matches!(k.action.kind, ActionKind::Sword | ActionKind::Beams)).count();
    let dying: Vec<_> = boss.iter().filter(|k| k.ganon == Some(GanonPhase::Dying) && !k.power && k.mode == 5).collect();
    let kinds: Vec<_> = dying.iter().map(|k| (k.kind, k.x, k.y)).collect();
    assert!(!dying.is_empty() && dying.iter().all(|k| k.kind == ObjectiveKind::Treasure && k.arrows.is_empty()), "his ashes are the target: {kinds:?}");
    assert_eq!(boss.iter().map(|k| k.rupees).min().unwrap(), shots[0].rupees - 1, "a rupee a shot");
    assert!(w.s.state().triforce_of_power());
    assert_eq!(boss.last().unwrap().kind, ObjectiveKind::Move, "N, the doors open");

    // Zelda's room (step 353, `exit: end`): the target is the row under her (X $70-$7F at Y $95; on Link's first
    // tick she still stands on the room's spawn spot, her init not run: the target is under that); ARROW is never
    // offered; one SWORD N from ($78, $AD) puts out both fires of the path; the game ends.
    let end = at(t, uw(0x32), 353);
    assert!(end.iter().all(|k| k.kind == ObjectiveKind::Move && k.arrows.is_empty()));
    let under_her: BTreeSet<Target> = [Target::Tile((14, 11)), Target::Tile((15, 11))].into();
    let off: Vec<_> = end.iter().filter(|k| k.mode == 5 && k.targets != under_her).map(|k| &k.targets).collect();
    assert!(off.len() <= 1, "{off:?}");
    let stabs: Vec<_> = end.iter().filter(|k| k.action.kind == ActionKind::Sword).collect();
    assert_eq!(stabs.iter().map(|k| (k.x, k.y, k.action.direction)).collect::<Vec<_>>(), [(0x78, 0xAD, Direction::N)]);
    let fires = |k: &Tick| k.types.iter().filter(|ty| **ty == GUARD_FIRE).count();
    assert_eq!((fires(end[0]).max(fires(end[1])), fires(end.last().unwrap())), (4, 2), "both fires of the path are out");
    assert!(game_ended(w.s.state()) && w.s.state().full_location() == uw(0x32));
    assert_eq!(t.last().unwrap().step, Some(STEPS - 1));

    // The ledgers of this walk (it brings 3 bombs out of Dungeon 8, a run with no drops brings none).
    let end_state = w.s.state();
    assert_eq!((start.1, end_state.max_health()), (14, 15));
    assert_eq!(w.ledger.cheated.keys, 0);
    assert_eq!(w.ledger.cheated.rupees, 0);
    // (`--nocapture`: the walk's numbers for the round document.)
    eprintln!(
        "d9 walk: {} actions, {} frames from the D8 triforce; start {:?} (hearts, containers, rupees, bombs, keys); end {} hearts of {}, {} rupees, {} bombs, {} keys; cheated bombs {}; {swings} swings and {} arrow at Ganon",
        t.len(),
        w.s.frame() - frames_in,
        start,
        end_state.health(),
        end_state.max_health(),
        end_state.rupees(),
        end_state.bombs(),
        end_state.keys(),
        w.ledger.cheated.bombs,
        shots.len(),
    );
}

/// The walk with the clips switch on (the clip way, the owner's primary way): 0x55, 0x07, 0x20 from N and 0x03 are
/// left by `CLIP` to their stairs with their monsters alive, 0x20's first visit leaves the ring Link comes up in by
/// `CLIP` (its blocks do not move while a monster lives), and the return visits find 0x07 and 0x20 full. Every other
/// room as the plain way, but for the clip families' crossings where a skip is shorter.
#[test]
fn the_d9_route_is_walked_the_clip_way_from_the_d8_triforce_to_the_end_of_the_game() {
    let Some(mut w) = Walk::at_the_d8_triforce() else { return };
    let map = w.s.game_map().clone();
    let route = &map.routes[ROUTE];
    let frames_in = w.s.frame();
    let start = (w.s.state().health(), w.s.state().max_health(), w.s.state().rupees(), w.s.state().bombs(), w.s.state().keys());
    let visits = w.run(|s| game_ended(s.state()), 200_000);
    let t = &w.ticks;

    // The rooms and steps in order: the clip way's.
    let rooms: Vec<MapLocation> = visits.iter().map(|v| v.room).collect();
    let way: Vec<usize> = (FIRST..STEPS).filter(|&i| route[i].on_way(true)).collect();
    let expected: Vec<MapLocation> = way.iter().map(|&i| route[i].location()).collect();
    assert_eq!(rooms, expected);
    let steps: Vec<usize> = {
        let mut seen: Vec<usize> = t.iter().filter_map(|k| k.step).collect();
        seen.dedup();
        seen
    };
    assert_eq!(steps, way, "every step of the clip way, in order, and never off the route");
    for (clip, plain) in [(321, 322), (334, 335), (339, 340)] {
        assert!(way.contains(&clip) && !way.contains(&plain), "the stretches' clip-way steps: {clip}, not {plain}");
    }
    assert!(t.iter().all(|k| k.step.is_some() || !matches!(k.mode, 5 | 9 | 11)), "on the route in every playable tick");

    // The rooms whose fight the clip way skips: their monsters live through every visit.
    let clipped: BTreeSet<(MapLocation, usize)> =
        t.iter().filter(|k| k.action.kind == ActionKind::Clip).map(|k| (k.room, k.step.unwrap())).collect();
    let wizzrobes: &[u8] = &[BLUE_WIZZROBE, RED_WIZZROBE];
    let rings: [(u8, usize, &[u8]); 6] = [
        (0x55, 311, &[RED_LAMNOLA]),
        (0x07, 319, wizzrobes),
        (0x07, 321, wizzrobes),
        (0x20, 334, wizzrobes),
        (0x20, 339, wizzrobes),
        (0x03, 349, &[ZOL, LIKE_LIKE]),
    ];
    // CLIP is played where the clip way skips a fight (0x07's return walks out of the ring past the pinned
    // Wizzrobes without one), and on two crossings of the clip families where a skip is the shorter way: 0x51
    // (like-like, through its rows of blocks) and 0x04 (wizzrobe, out of the tunnel's ring).
    let skips: BTreeSet<(MapLocation, usize)> =
        [(0x55, 311), (0x07, 319), (0x20, 334), (0x20, 339), (0x03, 349), (0x51, 343), (0x04, 348)].into_iter().map(|(r, s)| (uw(r), s)).collect();
    assert_eq!(clipped, skips, "where CLIP is played");
    // Nothing is fought there: the monsters live (pinned by the walk), FIGHT is never the objective, and the return
    // visits find no kill banked.
    for (room, step, monsters) in rings {
        let ticks = at(t, uw(room), step);
        for m in monsters {
            assert!(ticks.iter().any(|k| k.types.contains(m)), "9:{room:#04x} on step {step}: no {m:#04x}");
        }
        assert!(ticks.iter().all(|k| k.kind != ObjectiveKind::Fight), "9:{room:#04x} on step {step}: fought");
        assert!(ticks.iter().all(|k| k.banked == 0), "9:{room:#04x} on step {step}: banked {}", ticks[0].banked);
    }
    // The fights the clip way keeps.
    for (room, step) in [(0x14, 313), (0x16, 326), (0x05, 328), (0x61, 332), (0x10, 336), (0x30, 346), (0x52, 351)] {
        assert!(at(t, uw(room), step).iter().any(|k| k.kind == ObjectiveKind::Fight), "9:{room:#04x}: fought");
    }
    // The eight bomb walls, 0x20's on its first visit among its five Wizzrobes.
    let walls = [(0x56, 310), (0x26, 316), (0x27, 317), (0x17, 318), (0x06, 327), (0x20, 334), (0x31, 345), (0x04, 348)];
    for (room, step) in walls {
        let bombs = at(t, uw(room), step).iter().filter(|k| k.action.kind == ActionKind::Bomb).count();
        assert_eq!(bombs, 1, "9:{room:#04x}");
    }
    assert_eq!(t.iter().filter(|k| k.room.level == 9 && k.action.kind == ActionKind::Bomb).count(), 8);
    // The ring and the arrow from their cellars on; the end.
    assert!(t.iter().filter(|k| k.step.is_some_and(|s| s > 320)).all(|k| k.ring));
    assert!(t.iter().filter(|k| k.step.is_some_and(|s| s > 337)).all(|k| k.silver));
    assert!(game_ended(w.s.state()) && w.s.state().full_location() == uw(0x32) && w.s.state().triforce_of_power());
    assert_eq!(t.last().unwrap().step, Some(STEPS - 1));
    let end_state = w.s.state();
    assert_eq!((start.1, end_state.max_health()), (14, 15));
    assert_eq!(w.ledger.cheated.keys, 0);
    eprintln!(
        "d9 clip walk: {} actions, {} frames from the D8 triforce; start {:?}; end {} hearts of {}, {} rupees, {} bombs, {} keys; cheated bombs {} rupees {}; {} clips",
        t.len(),
        w.s.frame() - frames_in,
        start,
        end_state.health(),
        end_state.max_health(),
        end_state.rupees(),
        end_state.bombs(),
        end_state.keys(),
        w.ledger.cheated.bombs,
        w.ledger.cheated.rupees,
        t.iter().filter(|k| k.action.kind == ActionKind::Clip).count(),
    );
}

/// Walks from the D8 triforce until Link plays route step `step` (on the plain way with the clips switch off when the
/// step is that way's), and returns the walk. A room the route enters more than once is told apart by what Link holds
/// on the later steps (the ring from step 321, the silver arrow from step 338).
fn walk_to(step: usize) -> Option<Walk> {
    let mut w = Walk::at_the_d8_triforce()?;
    let route = w.s.game_map().routes[ROUTE].clone();
    if !route[step].on_way(true) {
        w.clips_off();
    }
    let room = route[step].location();
    let (ring, silver) = (step >= 321, step >= 338);
    let there = |s: &S| {
        let st = s.state();
        st.full_location() == room && matches!(st.mode(), 5 | 9 | 11) && st.red_ring() == ring && st.silver_arrow() == silver
    };
    w.run(there, 200_000);
    // The run stops on the first playable frame in the room; the objective picks its step on the next tick.
    let state = w.s.state().clone();
    w.gc.objectives(Some(&state), &state);
    assert_eq!(w.gc.step(), Some(step), "{}", describe(&w.s));
    Some(w)
}

fn place(w: &mut Walk, x: u8, y: u8, d: Direction) {
    let ram = [(Var::LinkX, x), (Var::LinkY, y), (Var::LinkDirection, d.value()), (Var::LinkGridOffset, 0), (Var::DoorwayDir, 0)];
    w.s.apply_ram(&ram);
}

/// The objective and the masks for the state as it stands (the walk's room rule first unless `monsters`).
fn look(w: &mut Walk, monsters: bool) -> Objective {
    if !monsters {
        settle_room(&mut w.s, false);
    }
    let state = w.s.state().clone();
    let objective = w.gc.objectives(Some(&state), &state);
    w.masks(&objective);
    objective
}

/// Frames with nothing pressed (Link's hearts kept) until `ready`.
fn frames_until(s: &mut S, limit: usize, ready: impl Fn(&S) -> bool) -> bool {
    for _ in 0..limit {
        if ready(s) {
            return true;
        }
        s.press(0, 1);
    }
    ready(s)
}

/// The monsters of Link's room that count (alive, not dying).
fn monsters(s: &S) -> usize {
    let st = s.state();
    (1..=11u16).filter(|&slot| counting(st.peek(OBJ_TYPE + slot)) && st.peek(OBJ_METASTATE + slot) == 0).count()
}

/// A start in a room the route comes back to (item 3 of round 29's vocabulary): the pre-entry frame of 9:0x20
/// entered from N (step 340, the plain way: after the walk emptied the room on step 335 and went on through 0x10 and
/// its cellar) loaded the three ways a start state is loaded. As the game left it: empty. With the fresh-room write
/// every start state gets (`LoadOptions::fresh_room`): its five Wizzrobes are back, which is what the evaluation did
/// before round 29. With the route's `banked_kills` written after it, as the env and the chain do for this step by
/// the state's name and the clips switch: empty again, `IsRoomAllDead` set, and the block moves. The first visit's
/// start (out of the tunnel) has none, nor has the clip way's return (it left the room full).
#[test]
fn a_start_in_a_room_the_route_comes_back_to_keeps_its_kills() {
    let Some(w) = walk_to(340) else { return };
    let map = w.s.game_map().clone();
    let kills = |room: MapLocation, entry: &str, visit: u32, route: Option<&str>, clips: bool| {
        start_banked_kills(&map, room, Some(entry), Some(visit), route, clips)
    };
    assert_eq!(kills(uw(0x20), "N", 1, Some(ROUTE), false), Some(15));
    assert_eq!(kills(uw(0x20), "N", 1, None, false), Some(15), "by every route that has the step");
    assert_eq!(kills(uw(0x20), "N", 1, Some(ROUTE), true), None, "the clip way left it full");
    assert_eq!(kills(uw(0x20), "cave", 1, Some(ROUTE), false), None, "the first visit: a full room");
    assert_eq!((kills(uw(0x07), "cave", 1, Some(ROUTE), false), kills(uw(0x07), "cave", 1, Some(ROUTE), true)), (Some(5), None));
    for clips in [false, true] {
        assert_eq!(kills(uw(0x10), "cave", 1, Some(ROUTE), clips), Some(5), "0x10 is fought on both ways");
        assert_eq!(kills(uw(0x61), "cave", 1, Some(ROUTE), clips), Some(15));
        assert_eq!(kills(uw(0x61), "E", 1, Some(ROUTE), clips), None);
    }
    assert!(banked_kills_follow_clips(&map, uw(0x07), Some("cave"), Some(1), None));
    assert!(banked_kills_follow_clips(&map, uw(0x20), Some("N"), Some(1), None));
    assert!(!banked_kills_follow_clips(&map, uw(0x10), Some("cave"), Some(1), None));
    assert_eq!(kills(cellar(0x75), "S", 2, Some(ROUTE), false), None, "a cellar banks nothing");
    assert_eq!(kills(uw(0x20), "N", 1, Some("clips-gauntlet"), false), None, "not a step of that route");
    let snapshot = w.entry.clone().expect("the pre-entry frame of 0x20 from N");
    let load = |fresh_room: bool, banked: Option<u8>| {
        let mut s = Session::new(Counted(open_core().unwrap(), 0), map.clone());
        let banked_kills = banked.map(|k| ((9, 0x20), k)).into_iter().collect();
        let opts = LoadOptions { randomize_rng: true, fresh_room, banked_kills, ..LoadOptions::default() };
        s.load(&snapshot.machine, &opts, &mut ChaCha8Rng::seed_from_u64(7)).unwrap();
        assert_eq!(s.state().full_location(), uw(0x20));
        // The room's monsters are out of their clouds within 100 frames.
        frames_until(&mut s, 100, |_| false);
        (monsters(&s), s.state().peek(ROOM_ALL_DEAD) != 0, s)
    };
    let (kept, all_dead, _) = load(false, None);
    assert_eq!((kept, all_dead), (0, true), "as the game left it");
    let (fresh, all_dead, _) = load(true, None);
    assert_eq!((fresh, all_dead), (5, false), "the fresh-room write alone brings the five Wizzrobes back");
    let (banked, all_dead, s) = load(true, Some(15));
    assert_eq!((banked, all_dead), (0, true), "with the route's banked kills");
    // (`IsRoomAllDead` is what the block asks; the walk pushes it on this step in the room the game kept empty.)
    assert!(s.state().push_block_pending(), "{}", describe(&s));
}

/// The left ware of 0x47's cave is the red potion: taking it leaves no heart container for good, and the route is
/// `missed` (the heart waypoint's rule, as 0x2c's cave).
#[test]
fn the_potion_of_the_heart_cave_is_missed() {
    let Some(mut w) = walk_to(298) else { return };
    assert_eq!(w.s.state().full_location(), MapLocation::new(0, 0x47, true));
    // Link's hearts are full, so the two wares are told apart by the containers: walk up to the left ware.
    assert!(frames_until(&mut w.s, 300, |s| s.state().link.status == 0));
    place(&mut w, 0x58, 0xBD, Direction::N);
    for _ in 0..40 {
        w.s.press(direction_buttons(Direction::N), 1);
    }
    assert!(frames_until(&mut w.s, 300, |s| s.state().link.status == 0));
    assert_eq!(w.s.state().max_health(), 14, "the potion, not the container: {}", describe(&w.s));
    let state = w.s.state().clone();
    w.gc.objectives(Some(&state), &state);
    assert!(w.gc.missed().is_some_and(|m| m.contains("took the other ware")), "{:?}", w.gc.missed());
}

/// 0x66 with seven pieces: the old man stands in the way, the shutters stay shut, and the route is `missed` (game.yaml
/// `requires_pieces: 8`); with eight (the walk) it is not.
#[test]
fn the_old_man_of_0x66_asks_for_the_eight_pieces() {
    let Some(mut w) = walk_to(308) else { return };
    // In the entrance 0x76, one piece taken away (written): Link walks N into 0x66.
    w.s.apply_ram(&[(Var::Triforce, 0x7F)]);
    for _ in 0..200 {
        if w.s.state().full_location() == uw(0x66) && w.s.state().mode() == 5 {
            break;
        }
        w.s.act(Action::parse("MOVE N 8").unwrap()).unwrap();
    }
    assert_eq!(w.s.state().full_location(), uw(0x66), "{}", describe(&w.s));
    frames_until(&mut w.s, 60, |_| false);
    assert!((1..=11u16).any(|slot| (0x4B..=0x52).contains(&w.s.state().peek(OBJ_TYPE + slot))), "the old man stands there");
    assert!(w.s.state().is_door_barred(Direction::W) && w.s.state().is_door_barred(Direction::N));
    let state = w.s.state().clone();
    w.gc.objectives(Some(&state), &state);
    assert!(w.gc.missed().is_some_and(|m| m.contains("7 Triforce pieces of 8")), "{:?}", w.gc.missed());
}

/// The key doors off the route in 0x63 (N and S: the route's last key goes into W): at their mouths MOVE into them
/// is not offered and a MOVE played anyway spends no key; by the controller one frame there spends it. The route's
/// own W door is offered.
#[test]
fn the_key_doors_of_0x63_off_the_route_are_masked() {
    for (x, y, d) in [(0x78u8, 0x5Du8, Direction::N), (0x78, 0xBD, Direction::S)] {
        let Some(mut w) = walk_to(330) else { return };
        assert_eq!(w.s.state().keys(), 1);
        place(&mut w, x, y, d);
        look(&mut w, false);
        let into = Action::new(ActionKind::Move, d, 32).unwrap();
        assert!(w.s.state().at_doorway_mouth(d), "{}", describe(&w.s));
        assert!(!w.s.mask().is_allowed(&into), "{}", w.s.mask());
        w.s.act(into).unwrap();
        assert_eq!((w.s.state().keys(), w.s.state().full_location()), (1, uw(0x63)));
        place(&mut w, 0x20, 0x8D, Direction::W);
        look(&mut w, false);
        assert!(w.s.mask().is_allowed(&Action::new(ActionKind::Move, Direction::W, 8).unwrap()), "{}", w.s.mask());
        place(&mut w, x, y, d);
        look(&mut w, false);
        // The game's rule, by the controller: one frame of the direction at the mouth spends the key.
        w.s.press(direction_buttons(d), 1);
        assert_eq!(w.s.state().keys(), 0, "{}", describe(&w.s));
    }
}

/// ARROW in Ganon's room (the family `ganon`, `arrows: true`, rupee reserve 0): never while he is hidden; while he
/// is brown (written: his state byte, as the fourth hit sets it) only on his row or column and only toward him, in
/// all four facings; with no rupee Link is topped up to one while Ganon lives (game.yaml `requires_rupees: 1`,
/// cheated and counted when shot), and the one arrow kills.
#[test]
fn the_silver_arrow_is_offered_only_at_the_brown_ganon() {
    let Some(mut w) = walk_to(352) else { return };
    assert!(frames_until(&mut w.s, 400, |s| matches!(ganon(s.state()), Some((_, GanonPhase::Hidden))) && s.state().link.status == 0));
    assert!(w.s.state().silver_arrow() && w.gc.rupee_reserve() == 0);
    // No rupee: the ledger tops the purse up to 1 while he lives.
    w.s.apply_ram(&[(Var::Rupees, 0)]);
    let from = w.s.state().full_location();
    w.ledger.after_action(&mut w.s, &w.gc, from);
    assert_eq!(w.s.state().rupees(), 1, "topped up");
    let slot = u16::from(ganon(w.s.state()).unwrap().0.index);
    // He is put in the middle of the floor (written), hidden: no ARROW anywhere.
    for (addr, value) in [(OBJ_X + slot, 0x70u8), (OBJ_Y + slot, 0x8D)] {
        zelda::Nes::poke(w.s.nes_mut(), addr, value);
    }
    let arrows = |w: &mut Walk, x: u8, y: u8| {
        place(w, x, y, Direction::N);
        look(w, true);
        offered(&w.s, ActionKind::Arrow)
    };
    for (x, y) in [(0x30u8, 0x8Du8), (0x70, 0xBD), (0xC0, 0x95), (0x70, 0x5D)] {
        assert_eq!(arrows(&mut w, x, y), vec![], "hidden, at ({x:#04x}, {y:#04x})");
    }
    // Brown (his state byte written `$FF`, what the fourth hit sets): on his row facing him, on his column facing
    // him, nowhere else.
    zelda::Nes::poke(w.s.nes_mut(), 0xAC + slot, 0xFF);
    w.s.apply_ram(&[]);
    assert_eq!(ganon(w.s.state()).unwrap().1, GanonPhase::Brown);
    assert_eq!(arrows(&mut w, 0x30, 0x8D), [Direction::E], "W of him on his row");
    assert_eq!(arrows(&mut w, 0xC0, 0x95), [Direction::W], "E of him, 8 pixels under his Y");
    assert_eq!(arrows(&mut w, 0x70, 0xBD), [Direction::N], "under him");
    assert_eq!(arrows(&mut w, 0x78, 0x5D), [Direction::S], "over him, 8 pixels right of his X");
    assert_eq!(arrows(&mut w, 0x30, 0xAD), vec![], "off his row and column");
    assert_eq!(arrows(&mut w, 0x30, 0x85), vec![], "8 pixels over his row");
    let objective = look(&mut w, true);
    assert_eq!(objective.kind, ObjectiveKind::Fight);
    assert!(objective.targets.contains(&Target::Tile((6, 10))) && !objective.targets.contains(&Target::Tile((6, 14))), "{:?}", objective.targets);
    // The one arrow, with the one (granted) rupee: he dies, and the rupee counts as cheated.
    assert_eq!(arrows(&mut w, 0x30, 0x8D), [Direction::E]);
    w.s.act(Action::new(ActionKind::Arrow, Direction::E, 0).unwrap()).unwrap();
    let from = w.s.state().full_location();
    w.ledger.after_action(&mut w.s, &w.gc, from);
    assert!(frames_until(&mut w.s, 60, |s| matches!(ganon(s.state()), Some((_, GanonPhase::Dying)))), "{}", describe(&w.s));
    // (The purse is topped up once more while the arrow flies, he not dying yet: a grant never used is not counted.)
    assert_eq!(w.ledger.cheated.rupees, 1, "{} rupees", w.s.state().rupees());
    assert_eq!(arrows(&mut w, 0x30, 0x8D), vec![], "none while he dies");
}

/// Where the game ends: `zelda_tiles` are the two tiles in front of Zelda, and a MOVE N onto their row from under
/// her ends the game on the action's last frame (game mode `$13`); a room without her has none.
#[test]
fn the_game_ends_in_front_of_zelda() {
    let Some(mut w) = walk_to(353) else { return };
    // On Link's first tick her object may still be on the room's spawn spot, her init not run: the target is the row
    // under where the init puts her all the same.
    assert_eq!(zelda_tiles(w.s.state()), [(14, 11), (15, 11)], "{}", describe(&w.s));
    // Her init has run when she stands at Y $88.
    assert!(frames_until(&mut w.s, 60, |s| (1..=11u16).any(|slot| s.state().peek(OBJ_TYPE + slot) == ZELDA && s.state().peek(OBJ_Y + slot) == 0x88)));
    assert_eq!(zelda_tiles(w.s.state()), [(14, 11), (15, 11)]);
    assert!(w.gc.ends_game(w.s.state()) && !game_ended(w.s.state()));
    let objective = look(&mut w, true);
    assert_eq!((objective.kind, objective.next_rooms.len()), (ObjectiveKind::Move, 0));
    assert_eq!(offered(&w.s, ActionKind::Arrow), vec![], "no ARROW in Zelda's room");
    // Through the fires without a stab (it costs half a heart, an eighth with the ring): the game ends all the same.
    for _ in 0..40 {
        if game_ended(w.s.state()) {
            break;
        }
        w.s.act(Action::parse("MOVE N 8").unwrap()).unwrap();
    }
    assert!(game_ended(w.s.state()), "{}", describe(&w.s));
}
