//! Timed options frame by frame with interrupts, and uncontrollable-frame skipping (`triforce/frame_skip.py`).

use crate::actions::Action;
use crate::blocks::{ProjectileBlock, ShieldWatch};
use crate::enums::{ActionKind, Direction, selected};
use crate::game::GameView;
use crate::nes::Nes;
use crate::room_entry::is_pre_entry;
use crate::tables::{RAM_SIZE, SRAM_OFFSET};
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
const ADDR_LINK_STATUS: usize = 0xAC;
const ADDR_SWORD_ANIMATION: usize = 0xB9;
const ADDR_TREASURE_FLAG: usize = 0xBF;
const ADDR_TRIGGERED_DOOR_CMD: usize = 0x54;
const ADDR_OPENED_DOORS: usize = 0xEE;
const ADDR_HEARTS: usize = 0x66F;
const ADDR_PARTIAL_HEARTS: usize = 0x670;
const OBJ_X: usize = 0x70;
const OBJ_Y: usize = 0x84;
const OBJ_ID: usize = 0x34F;
const OBJ_SPAWN_STATE: usize = 0x405;
const INVENTORY: (usize, usize) = (0x657, 0x67E);
const NON_THREAT_IDS: [u8; 6] = [0x40, 0x60, 0x63, 0x64, 0x68, 0x6A];

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
}

impl fmt::Display for ZeldaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ZeldaError::LinkNeverControllable { mode } => {
                write!(f, "Link never became controllable (mode={mode})")
            }
            ZeldaError::NotLoaded => f.write_str("call load() first"),
        }
    }
}

impl std::error::Error for ZeldaError {}

/// What happened while executing one option.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OptionResult {
    pub frames: u32,
    pub interrupt: Option<Interrupt>,
    /// Frames of the option itself, before uncontrollable skipping.
    pub control_frames: u32,
    pub projectile_blocks: Vec<ProjectileBlock>,
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

fn is_threat(id: u8) -> bool {
    id != 0 && !NON_THREAT_IDS.contains(&id)
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
    pub just_exited_cave: bool,
    /// Keep the last frame before each room entry an action causes.
    pub capture_pre_entry: bool,
    pub last_pre_entry: Option<S>,
    blocks: Option<Vec<ProjectileBlock>>,
}

impl<S> Default for Executor<S> {
    fn default() -> Self {
        Executor {
            was_in_cave: false,
            just_exited_cave: false,
            capture_pre_entry: false,
            last_pre_entry: None,
            blocks: None,
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

    /// Skips frames until Link is controllable.  Returns frames elapsed.
    pub fn gain_control<N: Nes>(
        &mut self,
        m: &mut Machine<N>,
        capture: &mut dyn FnMut(&Machine<N>, &Self) -> S,
    ) -> Result<u32, ZeldaError> {
        let start = m.frame;
        self.step(m, 0, 1);
        self.skip_uncontrollable(m, None, capture)?;
        Ok((m.frame - start) as u32)
    }

    /// Executes `action` from the controllable `state`.  `capture` builds `last_pre_entry` (with
    /// `capture_pre_entry`).
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
            let interrupt = if matches!(action.kind, ActionKind::Move | ActionKind::Wait) {
                self.run_timed(m, action, state)
            } else {
                self.run_attack(m, action);
                None
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
            })
        })();
        self.blocks = None;
        result
    }

    /// One or more real frames, watching shield collisions while an option executes.
    fn step<N: Nes>(&mut self, m: &mut Machine<N>, buttons: u8, frames: u32) {
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
        let pushing = is_move && state.is_push_block_direction(action.direction);
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
                    self.just_exited_cave = false;
                    stalled = 0;
                } else if !pushing {
                    // Link moves at least every other frame; two still frames means he hit something.
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

    fn run_attack<N: Nes>(&mut self, m: &mut Machine<N>, action: &Action) {
        let item = match action.kind {
            ActionKind::Bomb => Some(selected::BOMBS),
            ActionKind::Boomerang => Some(selected::BOOMERANG),
            ActionKind::Arrow => Some(selected::ARROWS),
            _ => None,
        };
        if let Some(item) = item {
            m.set_var(Var::SelectedItem, item);
        }
        let facing = match action.direction {
            Direction::NW | Direction::NE => Direction::N,
            Direction::SW | Direction::SE => Direction::S,
            d => d,
        };
        m.set_var(Var::LinkDirection, facing.value());
        let button = if item.is_some() {
            buttons::B
        } else {
            buttons::A
        };
        self.step(m, direction_buttons(action.direction) | button, 1);
        for _ in 0..MAX_ANIMATION_WAIT {
            self.step(m, 0, 1);
            if m.read(ADDR_LINK_STATUS) == 0 && m.read(ADDR_SWORD_ANIMATION) == 0 {
                break;
            }
        }
    }

    fn skip_uncontrollable<N: Nes>(
        &mut self,
        m: &mut Machine<N>,
        start_location: Option<(u8, u8, bool)>,
        capture: &mut dyn FnMut(&Machine<N>, &Self) -> S,
    ) -> Result<(), ZeldaError> {
        let prev_in_cave = self.was_in_cave;
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
        if prev_in_cave && !self.was_in_cave {
            self.just_exited_cave = true;
        }
        self.step(m, 0, 1);
        Ok(())
    }
}

/// Mode 9 still entering its cellar: the cellar is laid out over the room above's tiles and Link walks down its
/// ladder (Z_05 InitMode9_EnterCellar / InitMode9_WalkCellar) until IsUpdatingMode turns on.
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
                (duration + 1, duration, None)
            );
            assert_eq!(result.blocks(), 1);
            assert_eq!(result.projectile_blocks[0].frame, 2);
            assert_eq!(
                *observed.lock().unwrap(),
                (1..=duration as u64 + 1).collect::<Vec<_>>()
            );
            assert!(m.sink.is_some());
            assert_eq!(m.nes.buttons, vec![0; result.frames as usize]);
        }
    }

    #[test]
    fn executor_counts_blocks_in_attack_settling_and_uncontrollable_frames() {
        let mut m = machine(3, 0);
        let result = run(&mut m, "SWORD E");
        assert_eq!(
            (result.control_frames, result.frames, result.blocks()),
            (2, 3, 1)
        );
        assert_eq!(m.nes.buttons, vec![buttons::RIGHT | buttons::A, 0, 0]);

        let mut m = machine(6, 6);
        let result = run(&mut m, "WAIT 4");
        assert_eq!(
            (result.control_frames, result.frames, result.blocks()),
            (4, 7, 1)
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
        for id in NON_THREAT_IDS {
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
    fn b_items_select_the_item_and_face_the_cardinal_part_of_a_diagonal() {
        for (action, item, facing, pressed) in [
            (
                "BOOMERANG SW",
                selected::BOOMERANG,
                Direction::S,
                buttons::DOWN | buttons::LEFT,
            ),
            (
                "BOOMERANG NE",
                selected::BOOMERANG,
                Direction::N,
                buttons::UP | buttons::RIGHT,
            ),
            ("ARROW W", selected::ARROWS, Direction::W, buttons::LEFT),
            ("BOMB E", selected::BOMBS, Direction::E, buttons::RIGHT),
        ] {
            let mut ram = playable();
            ram[Var::SelectedItem.addr() as usize] = 0x7;
            let mut m = scripted(ram, |_, _| {});
            execute(&mut m, action).unwrap();
            assert_eq!(m.read(Var::SelectedItem.addr() as usize), item, "{action}");
            assert_eq!(
                m.read(Var::LinkDirection.addr() as usize),
                facing.value(),
                "{action}"
            );
            assert_eq!(m.nes.buttons[0], pressed | buttons::B, "{action}");
        }
        // The sword keeps the selected B item and presses A.
        let mut ram = playable();
        ram[Var::SelectedItem.addr() as usize] = 0x7;
        let mut m = scripted(ram, |_, _| {});
        execute(&mut m, "SWORD S").unwrap();
        assert_eq!(m.read(Var::SelectedItem.addr() as usize), 0x7);
        assert_eq!(m.nes.buttons[0], buttons::DOWN | buttons::A);
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
