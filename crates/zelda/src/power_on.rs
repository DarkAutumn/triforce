//! From console power on to the first controllable frame, by controller inputs alone: the title screen, the
//! file-select screen, an idle of `delay` frames there, REGISTER YOUR NAME, and the game start in slot 1.
//!
//! **The delay is the run's randomness.**  The game's random number generator (`$18–$24`) is shifted once per
//! NMI in every game mode and depends on nothing else, and its period is 32767 frames.  So the delays
//! `0..=`[`MAX_DELAY`] (32766) give every distinct RNG phase at the first controllable frame, and no memory
//! needs to be written.  The idle sits on the file-select screen, before the name is entered, so one saved
//! machine state per idle length serves every name: `triforce/states/power_on/idle_<frames>.state`, one every
//! [`IDLE_STEP`] frames ([`idle_state_name`], [`nearest_idle`]).  A run loads the nearest one at or below its
//! delay and idles the rest; without the files it idles every frame.
//!
//! The inputs are those of `corpus/scripts/power_on_to_start.txt` (the base movie every start state comes
//! from), with its asserts as [`ZeldaError::Boot`] errors.  Every frame runs unrendered.

use crate::executor::buttons::{A, DOWN, LEFT, RIGHT, SELECT, START, UP};
use crate::{Nes, ZeldaError};

/// The largest idle: one frame short of the RNG's period (module doc).
pub const MAX_DELAY: u32 = 32766;
/// A saved idle state every this many frames...
pub const IDLE_STEP: u32 = 2000;
/// ...this many of them: after 2000, 4000, ..., 30000 idle frames.
pub const IDLE_STATES: u32 = 15;
/// The name runs register by default (the base movie's).
pub const DEFAULT_NAME: &str = "LINK";

/// The inputs from power on to the file-select screen, as `(buttons, frames)`: what ran before an idle state.
pub const TO_FILE_SELECT: [(u8, u32); 3] = [(0, 120), (START, 1), (0, 30)];

const MODE: u16 = 0x12;
const LEVEL: u16 = 0x10;
const ROOM: u16 = 0xEB;
const MODE_TITLE: u8 = 0;
const MODE_FILE_SELECT: u8 = 1;
const MODE_REGISTER: u8 = 0x0E;
const MODE_GAMEPLAY: u8 = 5;
const START_ROOM: u8 = 0x77;
/// The register screen's character board: 4 rows of 11 cells (A-K, L-V, W-Z and symbols, digits and space).
const BOARD_COLUMNS: u8 = 11;
const NAME_LENGTH: usize = 8;
/// Frames the game may take from the last START to the first gameplay frame.
const START_FRAMES: u32 = 600;

fn boot_error<T>(message: impl Into<String>) -> Result<T, ZeldaError> {
    Err(ZeldaError::Boot(message.into()))
}

/// The state store name of the machine `frames` idle frames into the file-select screen
/// (`power_on/idle_02000`).
pub fn idle_state_name(frames: u32) -> String {
    format!("power_on/idle_{frames:05}")
}

/// The idle frames of the saved state a run with `delay` starts from (0: none, it starts at power on).
pub fn nearest_idle(delay: u32) -> u32 {
    (delay / IDLE_STEP).min(IDLE_STATES) * IDLE_STEP
}

/// The register board cells of `name`, upper-cased: 1 to 8 of `A-Z`, `0-9` and space, not all spaces.  A
/// name that starts with `ZELDA` is refused: it starts the second quest.
pub fn name_cells(name: &str) -> Result<Vec<u8>, ZeldaError> {
    let name = name.to_ascii_uppercase();
    if name.is_empty() || name.chars().count() > NAME_LENGTH {
        return boot_error(format!("the name {name:?} must have 1 to {NAME_LENGTH} characters"));
    }
    if name.trim_matches(' ').is_empty() {
        return boot_error("the name is all spaces");
    }
    if name.starts_with("ZELDA") {
        return boot_error("the name ZELDA starts the second quest");
    }
    name.chars()
        .map(|c| match c {
            'A'..='Z' => Ok(c as u8 - b'A'),
            '0'..='9' => Ok(33 + (c as u8 - b'0')),
            ' ' => Ok(43),
            _ => boot_error(format!("the name {name:?} has {c:?}: only A-Z, 0-9 and spaces can be registered")),
        })
        .collect()
}

fn hold<N: Nes>(nes: &mut N, buttons: u8, frames: u32) -> u32 {
    for _ in 0..frames {
        nes.run_frame(buttons, false);
    }
    frames
}

/// One menu press: the button for 2 frames, then 8 frames of nothing.
fn press<N: Nes>(nes: &mut N, buttons: u8) -> u32 {
    hold(nes, buttons, 2) + hold(nes, 0, 8)
}

fn expect<N: Nes>(nes: &N, addr: u16, value: u8, what: &str) -> Result<(), ZeldaError> {
    let have = nes.peek(addr);
    if have == value {
        return Ok(());
    }
    boot_error(format!("{what}: ${addr:02X} is {have:#04x}, not {value:#04x}"))
}

/// From power on to the file-select screen, the cursor on REGISTER YOUR NAME ([`TO_FILE_SELECT`]).  Returns
/// the frames run.
pub fn to_file_select<N: Nes>(nes: &mut N) -> Result<u32, ZeldaError> {
    let [(_, title), (start, pressed), (_, wait)] = TO_FILE_SELECT;
    let mut frames = hold(nes, 0, title);
    expect(nes, MODE, MODE_TITLE, "no title screen after power on")?;
    frames += hold(nes, start, pressed) + hold(nes, 0, wait);
    expect(nes, MODE, MODE_FILE_SELECT, "no file-select screen after START")?;
    Ok(frames)
}

/// `frames` frames with no button (on the file-select screen: the run's delay).
pub fn idle<N: Nes>(nes: &mut N, frames: u32) {
    hold(nes, 0, frames);
}

/// From the file-select screen: registers `name` in slot 1, starts the game and runs to the first gameplay
/// frame (overworld room 0x77).  Returns the frames run.
pub fn register_and_start<N: Nes>(nes: &mut N, name: &str) -> Result<u32, ZeldaError> {
    let cells = name_cells(name)?;
    expect(nes, MODE, MODE_FILE_SELECT, "not on the file-select screen")?;
    let mut frames = hold(nes, START, 2) + hold(nes, 0, 29);
    expect(nes, MODE, MODE_REGISTER, "no register screen after START")?;
    // The board cursor starts on A (cell 0) and stays where the last character was taken.
    let mut at = 0u8;
    for cell in cells {
        let (row, column) = (cell / BOARD_COLUMNS, cell % BOARD_COLUMNS);
        let (from_row, from_column) = (at / BOARD_COLUMNS, at % BOARD_COLUMNS);
        for _ in 0..row.abs_diff(from_row) {
            frames += press(nes, if row > from_row { DOWN } else { UP });
        }
        for _ in 0..column.abs_diff(from_column) {
            frames += press(nes, if column > from_column { RIGHT } else { LEFT });
        }
        frames += press(nes, A);
        at = cell;
    }
    // SELECT walks slot 1, slot 2, slot 3, REGISTER END.
    for _ in 0..3 {
        frames += press(nes, SELECT);
    }
    frames += hold(nes, START, 2) + hold(nes, 0, 38);
    expect(nes, MODE, MODE_FILE_SELECT, "no file-select screen after REGISTER END")?;
    frames += press(nes, START);
    let mut waited = 0;
    while nes.peek(MODE) != MODE_GAMEPLAY {
        if waited == START_FRAMES {
            return boot_error(format!("the game did not start in {START_FRAMES} frames"));
        }
        waited += hold(nes, 0, 1);
    }
    expect(nes, LEVEL, 0, "the game did not start on the overworld")?;
    expect(nes, ROOM, START_ROOM, "the game did not start in room 0x77")?;
    Ok(frames + waited)
}

/// Power on to the first gameplay frame: [`to_file_select`], `delay` idle frames, [`register_and_start`].
/// `nes` must be at power on.  Returns the frames run.
pub fn boot<N: Nes>(nes: &mut N, name: &str, delay: u32) -> Result<u32, ZeldaError> {
    name_cells(name)?;
    if delay > MAX_DELAY {
        return boot_error(format!("the delay {delay} is over {MAX_DELAY} (the RNG's period is 32767 frames)"));
    }
    let frames = to_file_select(nes)?;
    idle(nes, delay);
    Ok(frames + delay + register_and_start(nes, name)?)
}
