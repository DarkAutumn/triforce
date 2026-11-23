//! Timed options frame by frame with interrupts, and uncontrollable-frame skipping.

use crate::actions::Action;
use crate::attack;
use crate::blocks::{ProjectileBlock, ShieldWatch};
use crate::clips::{
    ClipAbort, ClipOutcome, Controller, Decision, Input, MAX_SKIP_FRAMES, MAX_STILL_FRAMES, Phase, SkipKind, SkipTarget,
    skip_target,
};
use crate::enums::{ActionKind, Direction, ObjectClass, Tile};
use crate::game::GameView;
use crate::nes::Nes;
use crate::room_entry::is_pre_entry;
use crate::tables::{RAM_SIZE, SRAM_OFFSET, TILE_LAYOUT};
use crate::submenu;
use crate::vars::Var;
use std::fmt;

pub const MODE_DUNGEON_TRANSITION: u8 = 2;
pub const MODE_REVEAL: u8 = 3;
pub const MODE_SCROLL_COMPLETE: u8 = 4;
pub const MODE_GAMEPLAY: u8 = 5;
pub const MODE_SCROLL_START: u8 = 6;
pub const MODE_SCROLL: u8 = 7;
pub const MODE_GAME_OVER: u8 = 8;
pub const MODE_UNDERGROUND: u8 = 9;
pub const MODE_UNDERGROUND_TRANSITION: u8 = 10;
pub const MODE_CAVE: u8 = 11;
pub const MODE_CAVE_TRANSITION: u8 = 16;
pub const MODE_DYING: u8 = 17;
pub const MODE_HOLDING_TRIFORCE: u8 = 18;

pub const SCROLLING_MODES: [u8; 8] = [
    MODE_SCROLL_COMPLETE,
    MODE_SCROLL,
    MODE_SCROLL_START,
    MODE_UNDERGROUND_TRANSITION,
    MODE_CAVE_TRANSITION,
    MODE_REVEAL,
    MODE_DUNGEON_TRANSITION,
    MODE_HOLDING_TRIFORCE,
];

pub fn is_game_over_mode(mode: u8) -> bool {
    matches!(mode, MODE_DYING | MODE_GAME_OVER)
}

pub const STUN_FLAG: u8 = 0x40;
pub const CAVE_COOLDOWN: u32 = 60;
pub const MAX_ANIMATION_WAIT: u32 = 60;
pub const MAX_UNCONTROLLABLE_FRAMES: u32 = 2000;
pub const INTERRUPT_RADIUS: i32 = 24;

const ADDR_LEVEL: usize = 0x10;
const ADDR_MODE: usize = 0x12;
const ADDR_LOCATION: usize = 0xEB;
const ADDR_IS_UPDATING_MODE: usize = 0x11;
const ADDR_LINK_X: usize = 0x70;
const ADDR_LINK_Y: usize = 0x84;
const ADDR_GRID_OFFSET: usize = 0x394;
const ADDR_LINK_STATUS: usize = 0xAC;
/// Link's knockback direction (ObjShoveDir slot 0); the game takes no direction input while it is set.
const ADDR_LINK_SHOVE_DIR: usize = 0xC0;
const ADDR_SWORD_ANIMATION: usize = 0xB9;
const ADDR_TREASURE_FLAG: usize = 0xBF;
const ADDR_TRIGGERED_DOOR_CMD: usize = 0x54;
const ADDR_OPENED_DOORS: usize = 0xEE;
const ADDR_HEARTS: usize = 0x66F;
const ADDR_PARTIAL_HEARTS: usize = 0x670;
/// Turn frames an attack may take beyond its plan (lag frames repeat a decision) before it gives up.
const TURN_SLACK_FRAMES: u32 = 4;
/// Frames an attack waits for Link's item-use pose (a boomerang caught during the turn) to end before its press.
const MAX_POSE_FRAMES: u32 = 32;
/// asm:FluteTimer: frames left of the recorder's freeze ($98 when played; the NMI counts it down).
const ADDR_FLUTE_TIMER: usize = 0x3C;
/// The freeze's length plus slack (a cap, never reached in play).
const MAX_FREEZE_FRAMES: u32 = 0x98 + 8;
/// asm:ObjDir of Link (`$98`).
const ADDR_LINK_DIRECTION: usize = 0x98;
/// asm:InvTriforce.
const ADDR_TRIFORCE: usize = 0x671;
/// Plays the dock warp may make in all (a sequence needs two: r36's search, docs/routes/d3first.md).
const MAX_DOCK_PLAYS: u32 = 3;
/// Frames the dock warp waits after its last play for the whirlwind to pick Link up (it reaches X `$7E` about 70
/// frames after the freeze with its spawn cloud; a cap, never reached in play).
const MAX_DOCK_WAIT_FRAMES: u32 = 200;
const ADDR_LADDER_SLOT: usize = 0x64;
const OBJ_X: usize = 0x70;
const OBJ_Y: usize = 0x84;
const OBJ_ID: usize = 0x34F;
const OBJ_SPAWN_STATE: usize = 0x405;
const INVENTORY: (usize, usize) = (0x657, 0x67E);

/// NES controller bits.
pub mod buttons {
    pub const A: u8 = 0x01;
    pub const B: u8 = 0x02;
    pub const SELECT: u8 = 0x04;
    pub const START: u8 = 0x08;
    pub const UP: u8 = 0x10;
    pub const DOWN: u8 = 0x20;
    pub const LEFT: u8 = 0x40;
    pub const RIGHT: u8 = 0x80;
}

/// The controller byte for a direction (diagonals press both).
pub fn direction_buttons(direction: Direction) -> u8 {
    use buttons::*;
    match direction {
        Direction::N => UP,
        Direction::S => DOWN,
        Direction::W => LEFT,
        Direction::E => RIGHT,
        Direction::NW => UP | LEFT,
        Direction::NE => UP | RIGHT,
        Direction::SW => DOWN | LEFT,
        Direction::SE => DOWN | RIGHT,
        Direction::None => 0,
    }
}

/// Reasons a MOVE or WAIT ended early.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Interrupt {
    Damage,
    Spawn,
    EnemyNear,
    Transition,
    Item,
    Door,
    Wall,
    GameOver,
}

impl Interrupt {
    pub const ALL: [Interrupt; 8] = [
        Interrupt::Damage,
        Interrupt::Spawn,
        Interrupt::EnemyNear,
        Interrupt::Transition,
        Interrupt::Item,
        Interrupt::Door,
        Interrupt::Wall,
        Interrupt::GameOver,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Interrupt::Damage => "damage",
            Interrupt::Spawn => "spawn",
            Interrupt::EnemyNear => "enemy-near",
            Interrupt::Transition => "transition",
            Interrupt::Item => "item",
            Interrupt::Door => "door",
            Interrupt::Wall => "wall",
            Interrupt::GameOver => "game-over",
        }
    }

    pub const fn index(self) -> usize {
        self as usize
    }
}

/// Errors of the session/executor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ZeldaError {
    /// Link stayed uncontrollable for 2000 frames.
    LinkNeverControllable { mode: u8 },
    /// `act` before `load`.
    NotLoaded,
    /// The power-on sequence (`power_on`) can't run or left the screens it expects.
    Boot(String),
}

impl fmt::Display for ZeldaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ZeldaError::LinkNeverControllable { mode } => {
                write!(f, "Link never became controllable (mode={mode})")
            }
            ZeldaError::NotLoaded => f.write_str("call load() first"),
            ZeldaError::Boot(reason) => write!(f, "power on: {reason}"),
        }
    }
}

impl std::error::Error for ZeldaError {}

/// What happened while executing one option.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OptionResult {
    /// Every frame the option played (the item sub-screen's included).
    pub frames: u32,
    pub interrupt: Option<Interrupt>,
    /// Frames of the option itself, before uncontrollable skipping.
    pub control_frames: u32,
    pub projectile_blocks: Vec<ProjectileBlock>,
    /// The skip a CLIP performed (`crate::clips`).
    pub clip: Option<ClipOutcome>,
    /// Frames the game was paused or frozen (`crate::submenu`, the recorder), included in `frames`: the item
    /// sub-screen choosing the B item, and the recorder's tune, during which nothing is updated (no input, no
    /// movement, no hit: asm:Mode5Play_Update; only the NMI's frame timers run). They don't count against the
    /// action (discounting, frame penalties and the observation's history use `frames - menu_frames`).
    pub menu_frames: u32,
}

impl OptionResult {
    pub fn blocks(&self) -> usize {
        self.projectile_blocks.len()
    }
}

/// A per-frame callback (watching).
pub type FrameSink<N> = Box<dyn FnMut(&N) + Send>;

/// The emulator plus the session's frame counter, render flag and frame sink.
pub struct Machine<N: Nes> {
    pub nes: N,
    /// Real frames emulated since the session was created (look-ahead frames excluded).
    pub frame: u64,
    pub render: bool,
    pub sink: Option<FrameSink<N>>,
}

impl<N: Nes> Machine<N> {
    pub fn new(nes: N) -> Machine<N> {
        Machine {
            nes,
            frame: 0,
            render: false,
            sink: None,
        }
    }

    /// Holds `buttons` for `frames` real frames.
    pub fn step(&mut self, buttons: u8, frames: u32) {
        for _ in 0..frames {
            self.nes.run_frame(buttons, self.render);
            self.frame += 1;
            if let Some(sink) = self.sink.as_mut() {
                sink(&self.nes);
            }
        }
    }

    pub fn read(&self, addr: usize) -> u8 {
        self.nes.ram()[addr]
    }

    pub fn ram(&self) -> &[u8; 0x800] {
        self.nes.ram()
    }

    /// RAM followed by PRG-RAM.
    pub fn ram_image(&self) -> Box<[u8; RAM_SIZE]> {
        let mut out = Box::new([0u8; RAM_SIZE]);
        out[..0x800].copy_from_slice(self.nes.ram());
        out[0x800..].copy_from_slice(self.nes.prg_ram());
        out
    }

    /// Writes `ram_image` index `index` (CPU RAM below 0x800, PRG-RAM above).
    pub fn write_image(&mut self, index: usize, value: u8) {
        let addr = if index < 0x800 {
            index
        } else {
            index + SRAM_OFFSET
        };
        self.nes.poke(addr as u16, value);
    }

    pub fn set_var(&mut self, var: Var, value: u8) {
        self.nes.poke(var.addr(), value);
    }
}

/// A spawn or an object near Link that interrupts: anything that can hurt Link ([`ObjectClass::can_hurt`]) but the
/// cave's standing fires, which never move (the walk to a cave's ware passes them).
fn is_threat(id: u8) -> bool {
    ObjectClass::can_hurt(id) && id != crate::enums::object::STANDING_FIRE
}

/// The RAM fields interrupts compare.
struct FrameWatch {
    mode: u8,
    location: u8,
    level: u8,
    health: (u8, u8),
    containers: u8,
    ids: [u8; 11],
    inventory: [u8; INVENTORY.1 - INVENTORY.0],
    treasure_flag: u8,
    doors: (u8, u8),
    link: (u8, u8),
    /// `LadderSlot` ($64): a frame that lays or puts away the ladder doesn't move Link.
    ladder: u8,
    near: u16,
}

impl FrameWatch {
    fn new(ram: &[u8; 0x800]) -> FrameWatch {
        let mut w = FrameWatch {
            mode: ram[ADDR_MODE],
            location: ram[ADDR_LOCATION],
            level: ram[ADDR_LEVEL],
            health: (ram[ADDR_HEARTS] & 0x0F, ram[ADDR_PARTIAL_HEARTS]),
            containers: ram[ADDR_HEARTS] >> 4,
            ids: ram[OBJ_ID + 1..OBJ_ID + 12].try_into().unwrap(),
            inventory: ram[INVENTORY.0..INVENTORY.1].try_into().unwrap(),
            treasure_flag: ram[ADDR_TREASURE_FLAG],
            doors: (ram[ADDR_OPENED_DOORS], ram[ADDR_TRIGGERED_DOOR_CMD]),
            link: (ram[ADDR_LINK_X], ram[ADDR_LINK_Y]),
            ladder: ram[ADDR_LADDER_SLOT],
            near: 0,
        };
        let (lx, ly) = (w.link.0 as i32, w.link.1 as i32);
        for i in 0..11 {
            let id = w.ids[i];
            if !is_threat(id) {
                continue;
            }
            let slot = i + 1;
            if id <= 0x48 && ram[OBJ_SPAWN_STATE + slot] != 0 {
                continue;
            }
            let dx = ram[OBJ_X + slot] as i32 - lx;
            let dy = ram[OBJ_Y + slot] as i32 - ly;
            if dx * dx + dy * dy < INTERRUPT_RADIUS * INTERRUPT_RADIUS {
                w.near |= 1 << slot;
            }
        }
        w
    }
}

/// Compares two consecutive frames (`detect_interrupt`).
fn detect_interrupt(before: &FrameWatch, after: &FrameWatch, start_near: u16) -> Option<Interrupt> {
    if is_game_over_mode(after.mode) {
        return Some(Interrupt::GameOver);
    }
    if after.mode != before.mode || after.location != before.location || after.level != before.level
    {
        return Some(Interrupt::Transition);
    }
    if after.health < before.health || after.containers < before.containers {
        return Some(Interrupt::Damage);
    }
    for i in 0..11 {
        if is_threat(after.ids[i]) && after.ids[i] != before.ids[i] {
            return Some(Interrupt::Spawn);
        }
    }
    if after.inventory != before.inventory
        || after.treasure_flag != before.treasure_flag
        || after.health > before.health
        || after.containers > before.containers
    {
        return Some(Interrupt::Item);
    }
    if after.doors != before.doors {
        return Some(Interrupt::Door);
    }
    if after.near & !before.near & !start_near != 0 {
        return Some(Interrupt::EnemyNear);
    }
    None
}

/// Runs actions until Link is controllable again (`OptionExecutor`).
pub struct Executor<S> {
    pub was_in_cave: bool,
    /// Link just came out of a cave or dungeon mouth and stands on it (kept by `Session`, which sets and clears it).
    pub just_exited_cave: bool,
    /// Keep the last frame before each room entry an action causes.
    pub capture_pre_entry: bool,
    pub last_pre_entry: Option<S>,
    blocks: Option<Vec<ProjectileBlock>>,
    /// Dungeon doorways a MOVE may not walk into ([`Direction::value`] bits; round 27, kept by `Session`: the
    /// route's `keep_out` doorways the objective does not leave through): a MOVE toward one ends at its mouth.
    pub vetoed_doorways: u8,
    /// The level of the dock warp the route plays here (round 36, kept by `Session`: `crate::Session::set_dock_warp`):
    /// RECORDER played in the water on a dock's row plays the whole sequence ([`Executor::run_dock_warp`]).
    pub dock_warp: Option<u8>,
    /// The latched buttons (A, B, Select, Start) the game read as held when this action began
    /// ([`latched_held`]): the action's first frame that presses one of them is preceded by one frame of no buttons.
    release: u8,
}

/// The game's held buttons of controller 1 (`ButtonsDown`, $FA: A $80, B $40, Select $20, Start $10, the read's bit
/// order) as the executor's latched button bits ([`buttons`]). The game takes A, B, Select and Start as new presses
/// only (`ButtonsPressed`: held now, not at the last read; asm:Input_ReadOneController; asm:Link_HandleInput for the
/// sword and the B item, asm:Mode5Play_Update for Select and Start) and the directions as held (`ButtonsDown`,
/// asm:Mode5Play_Update@CheckMenu -> `ObjInputDir`).
fn latched_held<N: Nes>(m: &Machine<N>) -> u8 {
    let down = m.read(0xFA);
    [(0x80, buttons::A), (0x40, buttons::B), (0x20, buttons::SELECT), (0x10, buttons::START)]
        .iter()
        .filter(|(game, _)| down & game != 0)
        .fold(0, |bits, (_, ours)| bits | ours)
}

impl<S> Default for Executor<S> {
    fn default() -> Self {
        Executor {
            was_in_cave: false,
            just_exited_cave: false,
            capture_pre_entry: false,
            last_pre_entry: None,
            blocks: None,
            vetoed_doorways: 0,
            dock_warp: None,
            release: 0,
        }
    }
}

impl<S> Executor<S> {
    /// Clears per-episode state.
    pub fn reset(&mut self) {
        self.was_in_cave = false;
        self.just_exited_cave = false;
        self.last_pre_entry = None;
    }

    /// Skips frames until Link is controllable, then one frame more (a load's first decision is where it always
    /// was; an action closes with no such frame, round 40).  Returns frames elapsed.
    pub fn gain_control<N: Nes>(
        &mut self,
        m: &mut Machine<N>,
        capture: &mut dyn FnMut(&Machine<N>, &Self) -> S,
    ) -> Result<u32, ZeldaError> {
        let start = m.frame;
        self.step(m, 0, 1);
        self.skip_uncontrollable(m, None, capture)?;
        self.step(m, 0, 1);
        Ok((m.frame - start) as u32)
    }

    /// Executes `action` from the controllable `state`, until Link is controllable again; no frame of no buttons
    /// closes it (round 40, decisions 2026-11-23: it stopped Link for a frame between every two MOVEs, and the game
    /// needs a released frame only before a new press of a button it still reads as held, [`latched_held`]).
    /// `capture` builds `last_pre_entry` (with `capture_pre_entry`).
    pub fn execute<N: Nes>(
        &mut self,
        m: &mut Machine<N>,
        action: &Action,
        state: &GameView,
        capture: &mut dyn FnMut(&Machine<N>, &Self) -> S,
    ) -> Result<OptionResult, ZeldaError> {
        self.last_pre_entry = None;
        let start = m.frame;
        let location = (state.level(), state.location(), state.full_location().in_cave);
        self.blocks = Some(Vec::new());
        let result = (|| {
            self.release = latched_held(m);
            let mut clip = None;
            let mut menu_frames = 0;
            let interrupt = if action.kind == ActionKind::Clip {
                // The overworld's one N clip, the dock warp's (round 36), is offered only on a dock-warp step.
                let target = skip_target(state, action.direction).or_else(|| {
                    (action.direction == Direction::N).then(|| crate::clips::dock_clip_target(state)).flatten()
                });
                let (interrupt, outcome) = self.run_skip(m, action.direction, target);
                clip = Some(outcome);
                interrupt
            } else if matches!(action.kind, ActionKind::Move | ActionKind::Wait) {
                self.run_timed(m, action, state)
            } else {
                let (interrupt, frames) = self.run_attack(m, action, state);
                menu_frames = frames;
                interrupt
            };
            let control_frames = (m.frame - start) as u32;
            if interrupt != Some(Interrupt::GameOver) && !is_game_over_mode(m.read(ADDR_MODE)) {
                self.skip_uncontrollable(m, Some(location), capture)?;
            }
            Ok(OptionResult {
                frames: (m.frame - start) as u32,
                interrupt,
                control_frames,
                projectile_blocks: self.blocks.take().unwrap_or_default(),
                clip,
                menu_frames,
            })
        })();
        self.blocks = None;
        result
    }

    /// One or more real frames, watching shield collisions while an option executes. The action's first frame that
    /// presses a latched button the game still reads as held is preceded by one frame of no buttons, so the game sees
    /// a new press (asm:Input_ReadOneController).
    fn step<N: Nes>(&mut self, m: &mut Machine<N>, buttons: u8, frames: u32) {
        if std::mem::take(&mut self.release) & buttons != 0 {
            self.step_frames(m, 0, 1);
        }
        self.step_frames(m, buttons, frames);
    }

    fn step_frames<N: Nes>(&mut self, m: &mut Machine<N>, buttons: u8, frames: u32) {
        let Some(blocks) = self.blocks.as_mut() else {
            m.step(buttons, frames);
            return;
        };
        for _ in 0..frames {
            let before = ShieldWatch::new(m.ram());
            m.step(buttons, 1);
            blocks.extend(before.collisions(m.ram(), m.frame));
        }
    }

    fn run_timed<N: Nes>(
        &mut self,
        m: &mut Machine<N>,
        action: &Action,
        state: &GameView,
    ) -> Option<Interrupt> {
        let is_move = action.kind == ActionKind::Move;
        // The secret gravestone (round 27): pushed from below it slides for 32 frames with Link behind it at half
        // speed, so the press is held as for a push block; the MOVE ends on the stairs square it uncovers.
        let stone = if is_move { state.secret_push() } else { None };
        let pushing_stone = is_move
            && action.direction == Direction::N
            && (state.is_secret_push_direction(Direction::N) || state.secret_push_sliding());
        let pushing = is_move && (state.is_push_block_direction(action.direction) || pushing_stone);
        let mut first = true;
        // Round 36: on a dock-warp step a MOVE along the land below the dock's row ends on the clip's column (the
        // water spot nearest the dock, `crate::clips::dock_spot`), where the session offers the clip.
        let dock_column = match (self.dock_warp, action.direction) {
            (Some(_), Direction::W | Direction::E) if is_move => crate::clips::dock_spot(state.room()),
            _ => None,
        };
        // Where a MOVE ends without another frame of its direction (round 27): on a grid point facing a gravestone
        // square (one frame of pressing releases a flying Ghini, asm:Spawn_CheckTouchedArmosOrGravestone; the
        // route's own stone from below excepted), on the stairs the pushed stone uncovered (walking on leaves them
        // and presses into the moved stone), and at the mouth of a doorway the route keeps out of.
        let stops = |m: &Machine<N>, first: bool, vetoed: u8| -> bool {
            let ram = m.ram();
            if vetoed & action.direction.value() != 0 && crate::game::at_doorway_mouth(ram, action.direction) {
                return true;
            }
            if ram[ADDR_LEVEL] != 0 || ram[ADDR_MODE] != MODE_GAMEPLAY || ram[ADDR_GRID_OFFSET] != 0 {
                return false;
            }
            let (px, py) = (i32::from(ram[ADDR_LINK_X]), i32::from(ram[ADDR_LINK_Y]));
            if let Some((x, row)) = dock_column
                && px == x
                && (row + 8..=row + 24).contains(&py)
            {
                return true;
            }
            let tiles = &m.nes.prg_ram()[TILE_LAYOUT.0 - 0x800..][..TILE_LAYOUT.1];
            let tile = |(c, r): Tile| {
                let inside = (0..crate::room::COLS).contains(&c) && (0..crate::room::ROWS).contains(&r);
                if inside { tiles[(c * crate::room::ROWS + r) as usize] } else { 0 }
            };
            if let Some((c, r)) = stone
                && (!first || action.direction == Direction::N)
                && tile((c, r)) == crate::room::STAIRS_TOP_LEFT_TILE
                && (px, py) == (c * 8, r * 8 + 0x3D)
            {
                return true;
            }
            !(first && pushing_stone)
                && crate::room::Room::hotspot_tiles(px, py, action.direction)
                    .into_iter()
                    .any(|t| crate::game::is_gravestone_tile(tile(t)))
        };
        if is_move && stops(m, first, self.vetoed_doorways) {
            return Some(Interrupt::Wall);
        }
        if is_move && !pushing && !state.can_link_move(action.direction) {
            // A move into a wall costs one frame.
            self.step(m, direction_buttons(action.direction), 1);
            return Some(Interrupt::Wall);
        }
        let buttons = if is_move {
            direction_buttons(action.direction)
        } else {
            0
        };
        let mut before = FrameWatch::new(m.ram());
        let start_near = before.near;
        let mut stalled = 0;
        for _ in 0..action.duration {
            if is_move && !first && stops(m, first, self.vetoed_doorways) {
                return Some(Interrupt::Wall);
            }
            first = false;
            self.step(m, buttons, 1);
            let after = FrameWatch::new(m.ram());
            if let Some(i) = detect_interrupt(&before, &after, start_near) {
                return Some(i);
            }
            if SCROLLING_MODES.contains(&after.mode) {
                return Some(Interrupt::Transition);
            }
            if is_move {
                if after.link != before.link {
                    stalled = 0;
                } else if !pushing && after.ladder == before.ladder && m.read(ADDR_LINK_SHOVE_DIR) == 0 {
                    // Link moves at least every other frame; two still frames means he hit something (a frame that
                    // puts the ladder away or lays it is still too, asm:Ladder_CheckAndDraw@StashLadder). Frames of a
                    // shove don't count: the game reads no direction then (asm:Link_HandleInput returns while
                    // ObjShoveDir is set), and a shove into a wall stands still (round 40: without the closing frame a
                    // MOVE right after a hit can start in it).
                    stalled += 1;
                    if stalled >= 2 {
                        return Some(Interrupt::Wall);
                    }
                }
            }
            before = after;
        }
        None
    }

    /// The grid-point skip toward `target` (`crate::clips`): the decision table every frame until Link stands at the
    /// end point (a wrap exit: then Left until the room change starts). A hit, a room change or game over end it at
    /// once (as a MOVE); spawns and nearby monsters don't. Missed presses back off and retry; the presses, a
    /// stand-still and a frame cap end it cleanly. Link is left where he stands, from where MOVE (and the skip itself,
    /// after its back-off) works again; a wrap exit cut past the bound leaves him at X `$F1`-`$00` on its row, where
    /// MOVE E or W goes on and MOVE W from X `$00` leaves by the W edge (zelda-asm/mechanics.md). A returning boomerang
    /// caught during the skip puts Link in the catch pose (`$20`, `$21`, then `$31` for a frame), in which the game
    /// zeroes his move (asm:Walker_Move@SetMovingDir): those frames are waited out with nothing pressed (at most
    /// `MAX_POSE_FRAMES`), outside the decision table and the stand-still count, as `run_attack` does before its press.
    fn run_skip<N: Nes>(
        &mut self,
        m: &mut Machine<N>,
        direction: Direction,
        target: Option<SkipTarget>,
    ) -> (Option<Interrupt>, ClipOutcome) {
        let Some(target) = target else {
            let outcome =
                ClipOutcome { direction, kind: None, abort: Some(ClipAbort::Unavailable), presses: 0, target: None };
            return (None, outcome);
        };
        let mut controller = Controller::new(target);
        let mut before = FrameWatch::new(m.ram());
        let (mut still, mut posed) = (0, 0);
        let mut abort = Some(ClipAbort::FrameCap);
        let mut interrupt = None;
        let ended = |before: &FrameWatch, after: &FrameWatch| match detect_interrupt(before, after, 0) {
            Some(i @ (Interrupt::GameOver | Interrupt::Transition | Interrupt::Damage)) => Some(i),
            _ if SCROLLING_MODES.contains(&after.mode) => Some(Interrupt::Transition),
            _ => None,
        };
        for _ in 0..MAX_SKIP_FRAMES {
            if m.read(ADDR_LINK_STATUS) != 0 && posed < MAX_POSE_FRAMES {
                posed += 1;
                self.step(m, 0, 1);
                let after = FrameWatch::new(m.ram());
                if let Some(i) = ended(&before, &after) {
                    interrupt = Some(i);
                    abort = Some(ClipAbort::Interrupted(i));
                    break;
                }
                before = after;
                continue;
            }
            let s = target.read(m.ram());
            let input = match controller.decide(s) {
                Decision::Done => {
                    abort = None;
                    break;
                }
                Decision::GiveUp => {
                    abort = Some(ClipAbort::Missed);
                    break;
                }
                Decision::Input(input) => input,
            };
            let buttons = match input {
                Input::Toward => target.walk(true),
                Input::Away => target.walk(false),
                Input::Press => target.press(),
                Input::Release => 0,
            };
            self.step(m, buttons, 1);
            let after = FrameWatch::new(m.ram());
            if let Some(i) = ended(&before, &after) {
                interrupt = Some(i);
                // A block or ladder clip past its press, holding the heading, that runs on into the next room (4:$21's
                // N wall already open) did what it set out to: Link is past the skipped point. A wrap exit ends this
                // way: its Left on the landing frame starts the room change. A wrap never changes rooms on its
                // crossing; one that does walked out the ordinary way.
                let carried_on = i == Interrupt::Transition
                    && match target.kind {
                        SkipKind::WrapExit => controller.phase == Phase::Exit && input == Input::Away,
                        SkipKind::Wrap => false,
                        SkipKind::Block | SkipKind::Ladder => {
                            controller.phase == Phase::Crossing && input == Input::Toward
                        }
                    };
                abort = if carried_on { None } else { Some(ClipAbort::Interrupted(i)) };
                break;
            }
            let now = target.read(m.ram());
            if input == Input::Press {
                controller.pressed(now);
            }
            // A frame that puts Link in the catch pose is no stand-still: the pose's frames are waited out above.
            still = if now.raw == s.raw && m.read(ADDR_LINK_STATUS) == 0 { still + 1 } else { 0 };
            if still >= MAX_STILL_FRAMES {
                abort = Some(ClipAbort::Stalled);
                break;
            }
            before = after;
        }
        let outcome =
            ClipOutcome { direction, kind: Some(target.kind), abort, presses: controller.presses, target: Some(target) };
        (interrupt, outcome)
    }

    /// An attack played with the controller alone ([`crate::attack`]; no RAM is written): the item sub-screen
    /// first when the B item isn't the selected one ([`Self::select_item`]), then the turn frames the game needs and
    /// the press, then the weapon's own frames. RECORDER's press freezes the game for the tune ($98 frames), which
    /// the option waits out: those frozen frames are menu frames too. An attack the buttons can't make from here (the
    /// mask doesn't offer it) presses nothing. Returns the interrupt that cut the turn short (a hit, a room change,
    /// game over) and the paused or frozen frames.
    fn run_attack<N: Nes>(&mut self, m: &mut Machine<N>, action: &Action, state: &GameView) -> (Option<Interrupt>, u32) {
        let (kind, d) = (action.kind, action.direction);
        let Some(turn_frames) = attack::plan(state, kind, d) else {
            return (None, 0);
        };
        let mut menu_frames = 0;
        if let Some(slot) = submenu::slot(kind)
            && m.read(submenu::ADDR_SELECTED_ITEM) != slot
        {
            menu_frames = self.select_item(m, slot);
            if m.read(submenu::ADDR_SELECTED_ITEM) != slot {
                return (None, menu_frames);
            }
        }
        // The turn, closed loop on the real game (a lag frame repeats a decision); a few frames past the plan's
        // count end it. A returning boomerang caught during the turn puts Link in the catch pose ($20 -> $21 -> $31
        // -> 0), and the game reads A and B only once his state is 0 again (asm:Link_HandleInput): those frames are
        // waited out with nothing pressed, so the press that follows is a new one.
        let mut before = FrameWatch::new(m.ram());
        let start_near = before.near;
        let (mut turns, mut posed) = (0, 0);
        let press = loop {
            let link = attack::Link::read(m.ram());
            let input = if link.state != 0 && posed < MAX_POSE_FRAMES {
                posed += 1;
                0
            } else {
                match attack::next(kind, d, &link) {
                    attack::Next::Press(buttons) => break Some(buttons),
                    attack::Next::Turn(t) if turns <= turn_frames + TURN_SLACK_FRAMES => {
                        turns += 1;
                        direction_buttons(t)
                    }
                    attack::Next::Turn(_) | attack::Next::Impossible => break None,
                }
            };
            self.step(m, input, 1);
            let after = FrameWatch::new(m.ram());
            match detect_interrupt(&before, &after, start_near) {
                Some(i @ (Interrupt::GameOver | Interrupt::Transition | Interrupt::Damage)) => {
                    return (Some(i), menu_frames);
                }
                _ if SCROLLING_MODES.contains(&after.mode) => return (Some(Interrupt::Transition), menu_frames),
                _ => {}
            }
            before = after;
        };
        let Some(press) = press else {
            return (None, menu_frames);
        };
        // In a busy room the game skips input on lag frames: a one-frame press can go unread. Hold the buttons one
        // more frame when nothing started (holding is no new press once the game has read it).
        if kind == ActionKind::Recorder {
            // asm:Flute_Wield sets FluteTimer and leaves Link's state at 0. Every frame that ends with the timer
            // still running had no play update (asm:Mode5Play_Update): those are the freeze's frames; the frame
            // that counts it to 0 plays again.
            menu_frames += self.play_recorder(m, press);
            if let Some(level) = self.dock_warp
                && crate::clips::on_dock_spot(state)
            {
                let (interrupt, frozen) = self.run_dock_warp(m, level);
                return (interrupt, menu_frames + frozen);
            }
            return (None, menu_frames);
        }
        self.step(m, press, 1);
        if m.read(ADDR_LINK_STATUS) == 0 && m.read(ADDR_SWORD_ANIMATION) == 0 {
            self.step(m, press, 1);
        }
        for _ in 0..MAX_ANIMATION_WAIT {
            self.step(m, 0, 1);
            if m.read(ADDR_LINK_STATUS) == 0 && m.read(ADDR_SWORD_ANIMATION) == 0 {
                break;
            }
        }
        (None, menu_frames)
    }

    /// One play of the recorder: the press (held a second frame when the first started nothing) and the freeze waited
    /// out. Returns the frozen frames.
    fn play_recorder<N: Nes>(&mut self, m: &mut Machine<N>, press: u8) -> u32 {
        let mut frozen = 0;
        self.step(m, press, 1);
        if m.read(ADDR_FLUTE_TIMER) == 0 {
            self.step(m, press, 1);
        }
        for _ in 0..MAX_FREEZE_FRAMES {
            if m.read(ADDR_FLUTE_TIMER) == 0 {
                break;
            }
            self.step(m, 0, 1);
            if m.read(ADDR_FLUTE_TIMER) != 0 {
                frozen += 1;
            }
        }
        frozen
    }

    /// The dock warp after its first play (round 36, docs/routes/d3first.md "The wrong warps"), Link in the water on
    /// the dock's row: the plays that leave the game's index on `level` with the whirlwind out
    /// ([`crate::whirlwind::plays`], recomputed after each), each facing N for a step up or W for a step down (the
    /// water blocks both: the one-frame turn moves him not; S and E would), on the first free frame after the
    /// freeze; then nothing pressed until the whirlwind picks him up (`WhirlwindTeleportingState` 1), goes by (a hit
    /// left him invincible or off the row), or a hit, room change or game over cuts it short. Returns that interrupt
    /// and the frozen frames.
    fn run_dock_warp<N: Nes>(&mut self, m: &mut Machine<N>, level: u8) -> (Option<Interrupt>, u32) {
        use crate::whirlwind::{FACING_UP, LEVEL_INDEX, OBJ_WHIRLWIND, SUMMONED, TELEPORTING, plays};
        let mut frozen = 0;
        let row = m.read(OBJ_Y);
        for _ in 1..MAX_DOCK_PLAYS {
            let summoned = m.read(SUMMONED as usize) != 0;
            let Some((n, facings)) = plays(m.read(LEVEL_INDEX as usize), m.read(ADDR_TRIFORCE), summoned, level) else {
                break;
            };
            if n == 0 || m.read(OBJ_Y) != row {
                break;
            }
            let face = if facings & FACING_UP != 0 { Direction::N } else { Direction::W };
            if m.read(ADDR_LINK_DIRECTION) != face.value() {
                self.step(m, direction_buttons(face), 1);
            }
            frozen += self.play_recorder(m, buttons::B);
        }
        let mut before = FrameWatch::new(m.ram());
        let start_near = before.near;
        for _ in 0..MAX_DOCK_WAIT_FRAMES {
            let whirlwind = (1..=0x0B).any(|slot| m.read(OBJ_ID + slot) == OBJ_WHIRLWIND);
            if m.read(TELEPORTING as usize) != 0 || !whirlwind {
                break;
            }
            self.step(m, 0, 1);
            let after = FrameWatch::new(m.ram());
            match detect_interrupt(&before, &after, start_near) {
                Some(i @ (Interrupt::GameOver | Interrupt::Transition | Interrupt::Damage)) => return (Some(i), frozen),
                _ if SCROLLING_MODES.contains(&after.mode) => return (Some(Interrupt::Transition), frozen),
                _ => {}
            }
            before = after;
        }
        (None, frozen)
    }

    /// Selects B item `slot` on the item sub-screen ([`crate::submenu`]): START, the scroll in, one step per frame
    /// toward the slot the shorter way (alternating the direction and the direction with Up, so every frame is a new
    /// input), START on the frame that reaches it, the scroll out. Returns the frames played, all of them with the
    /// game paused; the sub-screen is closed (`SubmenuState` 0, back in play) when it returns.
    fn select_item<N: Nes>(&mut self, m: &mut Machine<N>, slot: u8) -> u32 {
        let start = m.frame;
        let active = submenu::active_state(m.read(ADDR_LEVEL));
        self.step(m, buttons::START, 1);
        if m.read(submenu::ADDR_SUBMENU_STATE) != 0 {
            for _ in 0..submenu::MAX_OPEN_FRAMES {
                if m.read(submenu::ADDR_SUBMENU_STATE) == active {
                    break;
                }
                self.step(m, 0, 1);
            }
            let peek = |m: &Machine<N>| {
                let ram = *m.ram();
                move |addr: u16| ram[addr as usize]
            };
            for _ in 0..submenu::MAX_SELECT_FRAMES {
                if m.read(submenu::ADDR_SUBMENU_STATE) != active {
                    break;
                }
                let selected = m.read(submenu::ADDR_SELECTED_ITEM);
                let inventory = peek(m);
                let Some((dir, _)) = (selected != slot).then(|| submenu::way(inventory, selected, slot)).flatten() else {
                    self.step(m, buttons::START, 1);
                    break;
                };
                let input = if m.read(submenu::ADDR_SEARCH_DIR) == dir { dir | submenu::SEARCH_UP } else { dir };
                let mut pad = if dir == submenu::SEARCH_RIGHT { buttons::RIGHT } else { buttons::LEFT };
                if input & submenu::SEARCH_UP != 0 {
                    pad |= buttons::UP;
                }
                if submenu::search(inventory, selected, dir) == slot {
                    pad |= buttons::START;
                }
                self.step(m, pad, 1);
            }
            for _ in 0..submenu::MAX_CLOSE_FRAMES {
                if m.read(submenu::ADDR_SUBMENU_STATE) == 0 {
                    break;
                }
                self.step(m, 0, 1);
            }
        }
        (m.frame - start) as u32
    }

    /// Steps (no buttons) through scrolls, cave entries, stuns and level transitions until Link is controllable.
    fn skip_uncontrollable<N: Nes>(
        &mut self,
        m: &mut Machine<N>,
        start_location: Option<(u8, u8, bool)>,
        capture: &mut dyn FnMut(&Machine<N>, &Self) -> S,
    ) -> Result<(), ZeldaError> {
        let mut controllable = false;
        for _ in 0..MAX_UNCONTROLLABLE_FRAMES {
            let mode = m.read(ADDR_MODE);
            if is_game_over_mode(mode) {
                return Ok(());
            }
            if mode == MODE_CAVE && !self.was_in_cave {
                self.step(m, 0, CAVE_COOLDOWN);
                self.was_in_cave = true;
                continue;
            }
            if SCROLLING_MODES.contains(&mode)
                || is_cellar_entry(m)
                || m.read(ADDR_LINK_STATUS) & STUN_FLAG != 0
                || is_level_transition(m, start_location)
            {
                if self.capture_pre_entry && is_pre_entry(|a| m.read(a as usize)) {
                    self.last_pre_entry = Some(capture(m, self));
                }
                self.step(m, 0, 1);
                continue;
            }
            controllable = true;
            break;
        }
        if !controllable {
            return Err(ZeldaError::LinkNeverControllable {
                mode: m.read(ADDR_MODE),
            });
        }
        self.was_in_cave = m.read(ADDR_MODE) == MODE_CAVE;
        Ok(())
    }
}

/// Mode 9 still entering its cellar: the cellar is laid out over the room above's tiles and Link walks down its
/// ladder (asm:Mode9Cellar_Init_Sub8_EnterCellar / asm:Mode9Cellar_Init_Sub9_WalkIn) until IsUpdatingMode turns on.
fn is_cellar_entry<N: Nes>(m: &Machine<N>) -> bool {
    m.read(ADDR_MODE) == MODE_UNDERGROUND && m.read(ADDR_IS_UPDATING_MODE) == 0
}

fn is_level_transition<N: Nes>(m: &Machine<N>, start_location: Option<(u8, u8, bool)>) -> bool {
    let Some((level, location, in_cave)) = start_location else {
        return false;
    };
    let now_in_cave = m.read(ADDR_MODE) == MODE_CAVE;
    if in_cave || now_in_cave {
        return false;
    }
    level != m.read(ADDR_LEVEL) && location == m.read(ADDR_LOCATION)
}

#[cfg(test)]
pub(crate) mod tests {
    //! The scripted-shield executor tests of tests/test_core.py, on a scripted `Nes` double, plus interrupt
    //! precedence and option edge cases on RAM scripts.
    use super::*;
    use crate::blocks::tests::{parry_frame, shield_ram};
    use crate::enums::enemy;
    use crate::game_map::GameMap;
    use std::sync::{Arc, Mutex};

    /// Deterministic RAM frames: Link stunned before `stunned_until`, a shield parry on frame `collision`.
    struct Scripted {
        ram: [u8; 0x800],
        prg: [u8; 0x2000],
        frame: u64,
        collision: u64,
        stunned_until: u64,
        buttons: Vec<u8>,
    }

    impl Nes for Scripted {
        type State = ([u8; 0x800], u64);

        fn run_frame(&mut self, buttons: u8, _render: bool) {
            self.buttons.push(buttons);
            self.frame += 1;
            self.ram[0xAC] = if self.frame < self.stunned_until {
                0x40
            } else {
                0
            };
            if self.frame == self.collision {
                parry_frame(&mut self.ram, 0x5C);
            }
        }

        fn ram(&self) -> &[u8; 0x800] {
            &self.ram
        }

        fn prg_ram(&self) -> &[u8; 0x2000] {
            &self.prg
        }

        fn poke(&mut self, addr: u16, value: u8) {
            self.ram[addr as usize] = value;
        }

        fn snapshot(&self) -> Self::State {
            (self.ram, self.frame)
        }

        fn restore(&mut self, state: &Self::State) {
            (self.ram, self.frame) = *state;
        }

        fn poke_state(state: &mut Self::State, addr: u16, value: u8) {
            state.0[addr as usize] = value;
        }
    }

    fn machine(collision: u64, stunned_until: u64) -> Machine<Scripted> {
        Machine::new(Scripted {
            ram: shield_ram(0x5C),
            prg: [0; 0x2000],
            frame: 0,
            collision,
            stunned_until,
            buttons: Vec::new(),
        })
    }

    fn run(m: &mut Machine<Scripted>, action: &str) -> OptionResult {
        let view = GameView::new(&m.ram_image(), m.frame, false, Arc::new(GameMap::builtin()));
        let mut executor: Executor<()> = Executor::default();
        executor
            .execute(m, &Action::parse(action).unwrap(), &view, &mut |_, _| ())
            .unwrap()
    }

    #[test]
    fn executor_counts_every_collision_without_interrupting_or_replacing_callback() {
        for duration in [4u32, 32] {
            let mut m = machine(2, 0);
            let observed = Arc::new(Mutex::new(Vec::new()));
            let sink = observed.clone();
            m.sink = Some(Box::new(move |n: &Scripted| {
                sink.lock().unwrap().push(n.frame)
            }));
            let result = run(&mut m, &format!("WAIT {duration}"));
            assert_eq!(
                (result.frames, result.control_frames, result.interrupt),
                (duration, duration, None),
                "no frame closes an action (round 40)"
            );
            assert_eq!(result.blocks(), 1);
            assert_eq!(result.projectile_blocks[0].frame, 2);
            assert_eq!(
                *observed.lock().unwrap(),
                (1..=duration as u64).collect::<Vec<_>>()
            );
            assert!(m.sink.is_some());
            assert_eq!(m.nes.buttons, vec![0; result.frames as usize]);
        }
    }

    #[test]
    fn executor_counts_blocks_in_attack_settling_and_uncontrollable_frames() {
        // The script never starts the sword (Link's state stays 0): the press is held a second frame, as on a lag
        // frame. Link already faces E in this RAM, so A goes alone (no turn to press).
        let mut m = machine(3, 0);
        let result = run(&mut m, "SWORD E");
        assert_eq!(
            (result.control_frames, result.frames, result.blocks()),
            (3, 3, 1)
        );
        assert_eq!(m.nes.buttons, vec![buttons::A, buttons::A, 0]);

        let mut m = machine(6, 6);
        let result = run(&mut m, "WAIT 4");
        assert_eq!(
            (result.control_frames, result.frames, result.blocks()),
            (4, 6, 1)
        );
        assert_eq!(result.projectile_blocks[0].frame, 6);
    }

    // --- interrupt detection and option edge cases ---

    /// A playable dungeon frame: Link at (120, 140), full 3 hearts, nothing else.
    pub(crate) fn playable() -> [u8; 0x800] {
        let mut ram = [0u8; 0x800];
        (ram[ADDR_LEVEL], ram[ADDR_MODE], ram[ADDR_LOCATION]) = (1, MODE_GAMEPLAY, 0x44);
        (ram[ADDR_LINK_X], ram[ADDR_LINK_Y]) = (120, 140);
        (ram[ADDR_HEARTS], ram[ADDR_PARTIAL_HEARTS]) = (0x22, 0xFF);
        ram
    }

    fn interrupt(before: &[u8; 0x800], after: &[u8; 0x800]) -> Option<Interrupt> {
        let b = FrameWatch::new(before);
        detect_interrupt(&b, &FrameWatch::new(after), b.near)
    }

    /// Places object `slot` with `id` at (x, y).
    fn object(ram: &mut [u8; 0x800], slot: usize, id: u8, x: u8, y: u8) {
        (ram[OBJ_ID + slot], ram[OBJ_X + slot], ram[OBJ_Y + slot]) = (id, x, y);
    }

    #[test]
    fn interrupts_follow_detection_precedence() {
        let before = playable();
        // Every change at once, then peel the highest-priority one off each time.
        let mut after = playable();
        after[ADDR_MODE] = MODE_DYING;
        after[ADDR_LOCATION] = 0x45;
        after[ADDR_HEARTS] = 0x21;
        object(&mut after, 3, enemy::STALFOS, 120, 150);
        after[INVENTORY.0 + 1] = 1;
        after[ADDR_OPENED_DOORS] = 1;
        let expected = [
            (Interrupt::GameOver, ADDR_MODE),
            (Interrupt::Transition, ADDR_LOCATION),
            (Interrupt::Damage, ADDR_HEARTS),
            (Interrupt::Spawn, OBJ_ID + 3),
            (Interrupt::Item, INVENTORY.0 + 1),
            (Interrupt::Door, ADDR_OPENED_DOORS),
        ];
        for (want, addr) in expected {
            assert_eq!(interrupt(&before, &after), Some(want));
            after[addr] = before[addr];
        }
        assert_eq!(interrupt(&before, &after), None);
    }

    #[test]
    fn health_and_container_changes() {
        let before = playable();
        let mut after = playable();
        after[ADDR_PARTIAL_HEARTS] = 0x7F;
        assert_eq!(interrupt(&before, &after), Some(Interrupt::Damage));
        let mut after = playable();
        after[ADDR_HEARTS] = 0x12; // one container less, same filled hearts
        assert_eq!(interrupt(&before, &after), Some(Interrupt::Damage));
        let mut hurt = playable();
        hurt[ADDR_HEARTS] = 0x21;
        assert_eq!(interrupt(&hurt, &before), Some(Interrupt::Item), "healed");
        let mut after = playable();
        after[ADDR_HEARTS] = 0x32; // a new heart container
        assert_eq!(interrupt(&before, &after), Some(Interrupt::Item));
        let mut after = playable();
        after[ADDR_TREASURE_FLAG] = 0xFF;
        assert_eq!(
            interrupt(&before, &after),
            Some(Interrupt::Item),
            "treasure taken"
        );
    }

    #[test]
    fn spawns_count_threats_only() {
        let before = playable();
        // Items, tile objects, people (the cave old men), the standing fire and the invisible boulder generator.
        for id in [0x40, 0x60, 0x63, 0x64, 0x68, 0x6A, 0x6B, 0x6C, 0x1F, 0x2F] {
            let mut after = playable();
            object(&mut after, 2, id, 20, 20);
            assert_eq!(interrupt(&before, &after), None, "{id:#x}");
        }
        let mut after = playable();
        object(&mut after, 2, 0x53, 20, 20); // an octorok rock
        assert_eq!(interrupt(&before, &after), Some(Interrupt::Spawn));
        // A slot that empties is not a spawn.
        let mut with = playable();
        object(&mut with, 2, enemy::STALFOS, 20, 20);
        assert_eq!(interrupt(&with, &before), None);
    }

    #[test]
    fn enemy_near_fires_once_for_newly_close_spawned_threats() {
        let mut before = playable();
        object(&mut before, 4, enemy::STALFOS, 200, 200);
        let approach = |x: u8, y: u8, spawn_state: u8| {
            let mut ram = before;
            (ram[OBJ_X + 4], ram[OBJ_Y + 4]) = (x, y);
            ram[OBJ_SPAWN_STATE + 4] = spawn_state;
            ram
        };
        // Strictly inside the 24-pixel radius.
        assert_eq!(
            interrupt(&before, &approach(120 + 23, 140, 0)),
            Some(Interrupt::EnemyNear)
        );
        assert_eq!(interrupt(&before, &approach(120 + 24, 140, 0)), None);
        assert_eq!(
            interrupt(&before, &approach(120 + 16, 140 + 17, 0)),
            Some(Interrupt::EnemyNear)
        );
        assert_eq!(interrupt(&before, &approach(120 + 17, 140 + 17, 0)), None);
        // A monster still spawning doesn't count; a projectile ignores spawn state.
        assert_eq!(interrupt(&before, &approach(120, 140, 1)), None);
        let near = approach(120, 140, 0);
        // Already near at the option start: no interrupt when it stays near.
        let start = FrameWatch::new(&near);
        assert_eq!(
            detect_interrupt(&start, &FrameWatch::new(&near), start.near),
            None
        );
        // Near at the start, away, then back: still the same enemy, not a new approach.
        let away = FrameWatch::new(&before);
        assert_eq!(
            detect_interrupt(&away, &FrameWatch::new(&near), start.near),
            None
        );

        let mut shot = playable();
        object(&mut shot, 5, 0x53, 200, 200);
        let mut close = shot;
        (
            close[OBJ_X + 5],
            close[OBJ_Y + 5],
            close[OBJ_SPAWN_STATE + 5],
        ) = (125, 140, 3);
        assert_eq!(interrupt(&shot, &close), Some(Interrupt::EnemyNear));
    }

    /// `Nes` double whose frames run a script over RAM.
    pub(crate) struct Script {
        pub(crate) ram: [u8; 0x800],
        prg: [u8; 0x2000],
        frame: u64,
        pub(crate) buttons: Vec<u8>,
        on_frame: fn(u64, &mut [u8; 0x800]),
    }

    impl Nes for Script {
        type State = ([u8; 0x800], u64);

        fn run_frame(&mut self, buttons: u8, _render: bool) {
            self.buttons.push(buttons);
            self.frame += 1;
            (self.on_frame)(self.frame, &mut self.ram);
        }

        fn ram(&self) -> &[u8; 0x800] {
            &self.ram
        }

        fn prg_ram(&self) -> &[u8; 0x2000] {
            &self.prg
        }

        fn poke(&mut self, addr: u16, value: u8) {
            self.ram[addr as usize] = value;
        }

        fn snapshot(&self) -> Self::State {
            (self.ram, self.frame)
        }

        fn restore(&mut self, state: &Self::State) {
            (self.ram, self.frame) = *state;
        }

        fn poke_state(state: &mut Self::State, addr: u16, value: u8) {
            state.0[addr as usize] = value;
        }
    }

    pub(crate) fn scripted(
        ram: [u8; 0x800],
        on_frame: fn(u64, &mut [u8; 0x800]),
    ) -> Machine<Script> {
        Machine::new(Script {
            ram,
            prg: [0; 0x2000],
            frame: 0,
            buttons: Vec::new(),
            on_frame,
        })
    }

    fn execute(m: &mut Machine<Script>, action: &str) -> Result<OptionResult, ZeldaError> {
        let view = GameView::new(&m.ram_image(), m.frame, false, Arc::new(GameMap::builtin()));
        Executor::<()>::default().execute(m, &Action::parse(action).unwrap(), &view, &mut |_, _| ())
    }

    #[test]
    fn link_stunned_forever_is_an_error_after_the_frame_cap() {
        let mut m = scripted(playable(), |_, ram| ram[ADDR_LINK_STATUS] = STUN_FLAG);
        let err = Executor::<()>::default()
            .gain_control(&mut m, &mut |_, _| ())
            .unwrap_err();
        assert_eq!(
            err,
            ZeldaError::LinkNeverControllable {
                mode: MODE_GAMEPLAY
            }
        );
        assert_eq!(err.to_string(), "Link never became controllable (mode=5)");
        assert_eq!(m.frame, 1 + MAX_UNCONTROLLABLE_FRAMES as u64);
    }

    #[test]
    fn game_over_ends_the_option_without_skipping() {
        // Link dies on the third frame of a WAIT: the option stops there and never waits for control.
        let mut m = scripted(playable(), |frame, ram| {
            if frame >= 3 {
                ram[ADDR_MODE] = MODE_DYING;
                ram[ADDR_LINK_STATUS] = STUN_FLAG;
            }
        });
        let result = execute(&mut m, "WAIT 16").unwrap();
        assert_eq!(
            (result.interrupt, result.control_frames, result.frames),
            (Some(Interrupt::GameOver), 3, 3)
        );
        // An attack that ends in death returns once the animation wait is over.
        let mut m = scripted(playable(), |_, ram| ram[ADDR_MODE] = MODE_GAME_OVER);
        let result = execute(&mut m, "SWORD N").unwrap();
        assert_eq!(
            (result.interrupt, result.frames),
            (None, result.control_frames)
        );
    }

    #[test]
    fn attacks_press_buttons_and_write_nothing() {
        // (selected B item, facing, action, first frames pressed). The script's game never reacts, so a turn is
        // repeated until the plan's slack runs out and the B item is never pressed after it.
        let turn = vec![buttons::RIGHT; 1 + 1 + TURN_SLACK_FRAMES as usize];
        for (selected, facing, action, pressed) in [
            (0, 0, "BOOMERANG SW", vec![buttons::DOWN | buttons::LEFT | buttons::B]),
            (0, 8, "BOOMERANG NE", vec![buttons::UP | buttons::RIGHT | buttons::B]),
            (2, 2, "ARROW W", vec![buttons::B]),
            (1, 2, "BOMB E", turn),
            (7, 2, "SWORD S", vec![buttons::DOWN | buttons::A]),
            (7, 4, "SWORD S", vec![buttons::A]),
            // Not the selected B item: the sub-screen first (this script never opens it, so nothing follows).
            (7, 2, "BOMB W", vec![buttons::START]),
        ] {
            let mut ram = playable();
            ram[Var::SelectedItem.addr() as usize] = selected;
            ram[Var::LinkDirection.addr() as usize] = facing;
            let mut m = scripted(ram, |_, _| {});
            execute(&mut m, action).unwrap();
            assert_eq!(m.nes.ram, ram, "{action}: no memory written");
            assert_eq!(&m.nes.buttons[..pressed.len()], &pressed[..], "{action} facing {facing}");
        }
    }

    #[test]
    fn level_change_before_the_room_id_changes_is_still_a_transition() {
        // Between levels the level byte switches while $EB still holds the old room: not controllable yet.
        let mut m = scripted(playable(), |_, _| {});
        m.nes.ram[ADDR_LEVEL] = 2;
        assert!(is_level_transition(&m, Some((1, 0x44, false))));
        assert!(!is_level_transition(&m, Some((1, 0x45, false))));
        assert!(!is_level_transition(&m, Some((1, 0x44, true))));
        assert!(!is_level_transition(&m, None));
        m.nes.ram[ADDR_MODE] = MODE_CAVE;
        assert!(!is_level_transition(&m, Some((1, 0x44, false))));
    }
}
