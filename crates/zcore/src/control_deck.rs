//! [`ControlDeck`], which loads a ROM into a [`Bus`] and drives it one frame at a time.
//!
//! [`ControlDeck::clock_frame`] runs one NES frame; [`ControlDeck::clock_instr`] steps a single
//! instruction. Each refreshes what [`ControlDeck::frame_buffer`] reports.

use crate::{
    bus::{Bus, State},
    cart::{self, Cart},
    common::ResetKind,
    input::{Joypad, Player},
    mapper,
    video::{Frame, Video, VideoFilter},
};
use bitflags::bitflags;
use serde::{Deserialize, Serialize};
use std::io::Read;

/// Result returned from [`ControlDeck`] methods.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Errors that [`ControlDeck`] can return.
#[derive(Debug)]
#[must_use]
#[non_exhaustive]
pub enum Error {
    /// [`Cart`] error when loading a ROM.
    Cart(cart::Error),
    /// Operational error indicating a ROM must be loaded first.
    RomNotLoaded,
    /// CPU state is corrupted and emulation can't continue. Could be due to a bad ROM image or a
    /// corrupt save state.
    CpuCorrupted,
    /// The ROM's mapper number has no board implementation.
    UnimplementedMapper(u16),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cart(err) => err.fmt(f),
            Self::RomNotLoaded => f.write_str("no rom is loaded"),
            Self::CpuCorrupted => f.write_str("cpu state is corrupted"),
            Self::UnimplementedMapper(n) => write!(f, "unimplemented mapper `{n}`"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Cart(err) => std::error::Error::source(err),
            _ => None,
        }
    }
}

impl From<cart::Error> for Error {
    fn from(err: cart::Error) -> Self {
        Self::Cart(err)
    }
}

bitflags! {
    /// Headless mode flags to skip video processing, reducing CPU usage.
    #[derive(Default, Debug, Copy, Clone, PartialEq, Serialize, Deserialize, )]
    #[must_use]
    pub struct HeadlessMode: u8 {
        /// Disable pixel rendering.
        const NO_VIDEO = 0x02;
    }
}

/// Control deck configuration settings.
#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
#[must_use]
#[non_exhaustive]
pub struct Config {
    /// Video filter.
    pub filter: VideoFilter,
    /// Headless mode.
    pub headless_mode: HeadlessMode,
}

/// Represents a loaded ROM [`Cart`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct LoadedRom {
    /// Name of ROM.
    pub name: String,
    /// Whether the loaded Cart is battery-backed.
    pub battery_backed: bool,
}

/// Represents an NES Control Deck. Encapsulates the entire emulation state.
#[derive(Debug)]
#[must_use]
pub struct ControlDeck {
    /// Whether a ROM is loaded and the emulation is currently running or not.
    running: bool,
    /// Video output and filtering.
    video: Video,
    /// Whether `video.frame` still needs the filter applied, letting `frame_buffer` be cached when
    /// called more than once between clocks.
    video_frame_stale: bool,
    /// The currently loaded ROM [`Cart`], if any.
    loaded_rom: Option<LoadedRom>,
    /// The console: CPU, PPU, APU, input and the cart's board.
    bus: Bus,
}

impl Default for ControlDeck {
    fn default() -> Self {
        Self::new()
    }
}

impl ControlDeck {
    /// Creates a NES `ControlDeck` with the default configuration.
    pub fn new() -> Self {
        Self::with_config(Config::default())
    }

    /// Creates a NES `ControlDeck` with the given [`Config`].
    pub fn with_config(cfg: Config) -> Self {
        let mut bus = Bus::new();
        bus.render = !cfg.headless_mode.contains(HeadlessMode::NO_VIDEO);
        Self {
            running: false,
            video: Video::with_filter(cfg.filter),
            video_frame_stale: true,
            loaded_rom: None,
            bus,
        }
    }

    /// Loads a ROM cartridge from anything implementing [`Read`], hard-resetting the console.
    ///
    /// # Errors
    ///
    /// If the ROM header is malformed, its mapper is unimplemented, or the data can't be read, then
    /// an error is returned.
    pub fn load_rom<S: ToString, F: Read>(&mut self, name: S, rom: &mut F) -> Result<LoadedRom> {
        let name = name.to_string();
        self.unload_rom();
        // `Cart::from_rom` reports an unimplemented mapper as a nested `cart::Error`; unwrap it
        // back to this crate's own variant so callers see `unimplemented mapper \`69\`` rather
        // than a cart error wrapping a mapper error.
        let cart = Cart::from_rom(&name, rom).map_err(|err| match err {
            cart::Error::InvalidMapper(mapper::Error::Unimplemented(num)) => {
                Error::UnimplementedMapper(num)
            }
            err => Error::Cart(err),
        })?;
        let loaded_rom = LoadedRom {
            name,
            battery_backed: cart.battery_backed(),
        };
        self.bus.load_cart(cart);
        self.loaded_rom = Some(loaded_rom.clone());
        self.reset(ResetKind::Hard);
        Ok(loaded_rom)
    }

    /// Unloads the currently loaded ROM.
    pub fn unload_rom(&mut self) {
        self.loaded_rom = None;
        self.bus.unload_cart();
        self.running = false;
    }

    /// Replaces the running console with a previously saved [`State`] (of this cart).
    #[inline]
    pub fn load_bus(&mut self, state: State) {
        self.bus.state = state;
        // The console this deck was showing is gone; anything cached from it describes a machine
        // that is no longer here.
        self.video_frame_stale = true;
    }

    /// Returns the frame as RGBA pixels, with the configured [`VideoFilter`] applied.
    ///
    /// The filter runs on the first call after a clock and the result is cached, so calling this
    /// more than once per frame is cheap.
    #[inline]
    pub fn frame_buffer(&mut self) -> &[u8; Frame::SIZE] {
        if !self.video_frame_stale {
            return self.video.frame.as_array();
        }
        self.video_frame_stale = false;
        self.video
            .apply_filter(self.bus.frame(), self.bus.ppu.frame_number())
    }

    /// Returns the number of frames the console has rendered since power on.
    #[inline(always)]
    #[must_use]
    pub fn frame_number(&self) -> u32 {
        self.bus.ppu.frame_number()
    }

    /// Steps the control deck a single CPU instruction.
    ///
    /// # Errors
    ///
    /// If the CPU encounters an invalid opcode, then an error is returned.
    pub fn clock_instr(&mut self) -> Result<()> {
        self.video_frame_stale = true;
        self.step_instr()
    }

    /// One instruction, with no invalidation at all.
    #[inline(always)]
    fn step_instr(&mut self) -> Result<()> {
        self.bus.clock_instr();
        if self.cpu_corrupted() {
            self.running = false;
            return Err(Error::CpuCorrupted);
        }
        Ok(())
    }

    /// Invalidates the outputs of the previous clock.
    #[inline]
    fn begin_clock(&mut self) -> Result<()> {
        if !self.running {
            return Err(Error::RomNotLoaded);
        }
        self.video_frame_stale = true;
        Ok(())
    }

    /// Clocks exactly one NES frame.
    ///
    /// # Errors
    ///
    /// If no ROM is loaded, or the CPU encounters an invalid opcode, then an error is returned.
    pub fn clock_frame(&mut self) -> Result<()> {
        self.begin_clock()?;
        let frame = self.frame_number();
        while frame == self.frame_number() {
            self.step_instr()?;
        }
        self.bus.clock_sync();
        Ok(())
    }

    /// [`ControlDeck::clock_frame`] that stops between instructions as soon as `stop` accepts the
    /// console an instruction leaves behind, reporting whether it stopped. Calling again resumes
    /// the frame.
    ///
    /// # Errors
    ///
    /// If no ROM is loaded, or the CPU encounters an invalid opcode, then an error is returned.
    pub fn clock_frame_until(&mut self, stop: impl Fn(&Bus) -> bool) -> Result<bool> {
        self.begin_clock()?;
        let frame = self.frame_number();
        while frame == self.frame_number() {
            self.step_instr()?;
            if stop(&self.bus) {
                // The instruction may have been the last of the frame, which still has to be
                // wound up before the caller is told anything.
                if frame != self.frame_number() {
                    self.bus.clock_sync();
                }
                return Ok(true);
            }
        }
        self.bus.clock_sync();
        Ok(false)
    }

    /// Returns whether the CPU is corrupted, which means it encountered an invalid/unhandled
    /// opcode and can't proceed executing the current ROM.
    #[inline]
    #[must_use]
    pub fn cpu_corrupted(&self) -> bool {
        self.bus.cpu.corrupted
    }

    /// Returns the ControlDeck bus state which contains all components.
    #[inline]
    pub const fn bus(&self) -> &Bus {
        &self.bus
    }

    /// Returns a mutable reference to the current [`Joypad`] state for a given controller slot.
    #[inline]
    pub fn joypad_mut(&mut self, slot: Player) -> &mut Joypad {
        self.bus.input.joypad_mut(slot)
    }

    /// Set the video filter for frame buffer output when calling [`ControlDeck::frame_buffer`].
    #[inline]
    pub const fn set_filter(&mut self, filter: VideoFilter) {
        self.video.filter = filter;
        // The cached frame was filtered with the old one.
        self.video_frame_stale = true;
    }

    /// Resets the console.
    pub fn reset(&mut self, kind: ResetKind) {
        self.bus.reset(kind);
        self.video_frame_stale = true;
        if self.loaded_rom.is_some() {
            self.running = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{cpu::Cpu, input::JoypadBtn};
    use std::{
        fs::File,
        hash::{DefaultHasher, Hash, Hasher},
    };

    fn spritecans() -> ControlDeck {
        let mut deck = ControlDeck::new();
        let path = "test_roms/spritecans.nes";
        let mut rom = File::open(path).expect("test rom exists");
        deck.load_rom(path, &mut rom).expect("test rom loads");
        deck
    }

    /// Clock `frames` and return a hash of what ends up on screen.
    fn run(deck: &mut ControlDeck, frames: u32) -> u64 {
        for _ in 0..frames {
            deck.clock_frame().expect("clocks");
        }
        let mut hasher = DefaultHasher::new();
        deck.frame_buffer().hash(&mut hasher);
        hasher.finish()
    }

    /// A restored state has to resume bit-identically, both as a copy and through the serialized
    /// form.
    #[test]
    fn a_restored_state_resumes_identically() {
        let mut deck = spritecans();
        run(&mut deck, 30);
        let state = deck.bus().state;
        let bytes = postcard::to_allocvec(&deck.bus().state).expect("serializes");
        let expected = run(&mut deck, 30);

        let mut restored = spritecans();
        run(&mut restored, 5); // somewhere else entirely
        restored.load_bus(state);
        assert_eq!(run(&mut restored, 30), expected);

        let mut decoded = spritecans();
        run(&mut decoded, 5);
        decoded.load_bus(postcard::from_bytes(&bytes).expect("deserializes"));
        assert_eq!(run(&mut decoded, 30), expected);
    }

    /// `frame_buffer` caches the filtered frame, so every path that moves the console has to say
    /// so, stepping instructions included.
    #[test]
    fn stepping_instructions_refreshes_the_frame() {
        let mut deck = spritecans();
        // Past the ROM's first few frames, which are still identical to each other and so cannot
        // tell a refreshed frame from a cached one.
        run(&mut deck, 10);
        let first = deck.frame_buffer().to_vec();

        let frame = deck.frame_number();
        while frame == deck.frame_number() {
            deck.clock_instr().expect("steps");
        }
        assert_ne!(
            first,
            deck.frame_buffer().to_vec(),
            "a whole frame was stepped, so the cached one is not the current one"
        );
    }

    /// The frame buffer is not emulated state: a restored console keeps showing the last frame
    /// drawn until it draws its next one, which is then the restored console's.
    #[test]
    fn the_frame_after_a_restore_is_the_restored_consoles() {
        let mut deck = spritecans();
        run(&mut deck, 10);
        let early = deck.bus().state;
        deck.clock_frame().expect("clocks");
        let next_frame = deck.frame_buffer().to_vec();

        run(&mut deck, 30);
        let late_frame = deck.frame_buffer().to_vec();
        assert_ne!(next_frame, late_frame, "the game has moved on");

        deck.load_bus(early);
        deck.clock_frame().expect("clocks");
        assert_eq!(
            deck.frame_buffer().to_vec(),
            next_frame,
            "draws the restored console, not the one it replaced"
        );
    }

    /// Which buttons are held is emulated state the game reads, so it travels with the state.
    #[test]
    fn restoring_a_state_keeps_the_buttons_it_recorded() {
        let mut deck = spritecans();
        deck.clock_frame().expect("clocks");
        deck.joypad_mut(Player::One).press(JoypadBtn::A);
        let held = deck.bus().state;

        deck.joypad_mut(Player::One).buttons = 0;
        deck.load_bus(held);
        assert!(
            deck.joypad_mut(Player::One).button(JoypadBtn::A),
            "the state recorded A held, so the restored console has it held"
        );
    }

    /// The program counter a fresh console reaches after `instrs` instructions, which the same
    /// console reaches again from reset however it is clocked there.
    fn pc_after(instrs: usize) -> u16 {
        let mut deck = spritecans();
        for _ in 0..instrs {
            deck.clock_instr().expect("clocks");
        }
        deck.bus().cpu.pc
    }

    #[test]
    fn a_stop_condition_that_never_fires_clocks_the_frame_as_usual() {
        let mut deck = spritecans();
        let frame = deck.frame_number();
        assert!(!deck.clock_frame_until(|_| false).expect("clocks"));
        assert_eq!(deck.frame_number(), frame + 1);
    }

    /// A stop leaves the console *on* the instruction, not past it.
    #[test]
    fn clocking_stops_before_the_instruction_it_was_asked_to_stop_at() {
        let target = pc_after(100);

        let mut deck = spritecans();
        assert!(
            deck.clock_frame_until(|bus| bus.cpu.pc == target)
                .expect("clocks")
        );
        assert_eq!(deck.bus().cpu.pc, target);
        assert_eq!(deck.frame_number(), 0, "the frame was clocked to the end");
    }

    /// Stopping is between instructions and nothing else, so a run broken in the middle ends up
    /// exactly where an unbroken one does.
    #[test]
    fn stopping_and_resuming_leaves_the_same_console_as_running_through() {
        let target = pc_after(100);

        let mut unbroken = spritecans();
        run(&mut unbroken, 3);

        let mut broken = spritecans();
        for _ in 0..3 {
            while broken
                .clock_frame_until(|bus| bus.cpu.pc == target)
                .expect("clocks")
            {}
        }

        let &Cpu {
            pc, acc, x, y, sp, ..
        } = &unbroken.bus().cpu;
        assert_eq!(broken.bus().cpu.pc, pc);
        assert_eq!(
            (broken.bus().cpu.acc, broken.bus().cpu.x, broken.bus().cpu.y),
            (acc, x, y)
        );
        assert_eq!(broken.bus().cpu.sp, sp);
        assert_eq!(broken.bus().cpu.cycle, unbroken.bus().cpu.cycle);
        assert_eq!(broken.frame_number(), unbroken.frame_number());
        assert_eq!(broken.frame_buffer(), unbroken.frame_buffer());
    }
}
