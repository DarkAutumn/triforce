//! [`ControlDeck`], which loads a ROM into a [`Bus`] and drives it one frame at a time.
//!
//! [`ControlDeck::clock_frame`] runs one NES frame; [`ControlDeck::clock_instr`] steps a single
//! instruction. Each refreshes what [`ControlDeck::frame_buffer`] and
//! [`ControlDeck::audio_samples`] report, and discards what the previous call produced - so audio
//! cannot silently accumulate. See [`Config::clear_audio_on_clock`] to opt out of that.

use crate::{
    apu::{self, Apu, Channel},
    bus::{self, Bus},
    cart::{self, Cart},
    common::{NesRegion, ResetKind},
    input::{FourPlayer, Joypad, Player},
    mapper::{self, Bf909Revision, Mapper, MapperRevision, Mmc3Revision},
    memory::RamState,
    ppu,
    video::{Frame, Video, VideoFilter},
};
use bitflags::bitflags;
use serde::{Deserialize, Serialize};
use std::io::Read;
use thiserror::Error;

/// Result returned from [`ControlDeck`] methods.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Errors that [`ControlDeck`] can return.
#[derive(Error, Debug)]
#[must_use]
#[non_exhaustive]
pub enum Error {
    /// [`Cart`] error when loading a ROM.
    #[error(transparent)]
    Cart(#[from] cart::Error),
    /// Operational error indicating a ROM must be loaded first.
    #[error("no rom is loaded")]
    RomNotLoaded,
    /// CPU state is corrupted and emulation can't continue. Could be due to a bad ROM image or a
    /// corrupt save state.
    #[error("cpu state is corrupted")]
    CpuCorrupted,
    /// The ROM's mapper number has no board implementation.
    #[error("unimplemented mapper `{0}`")]
    UnimplementedMapper(u16),
    /// A save state that does not belong to the loaded cart.
    #[error(transparent)]
    StateMismatch(#[from] crate::cpu::StateMismatch),
}

bitflags! {
    /// Headless mode flags to disable audio and video processing, reducing CPU usage.
    #[derive(Default, Debug, Copy, Clone, PartialEq, Serialize, Deserialize, )]
    #[must_use]
    pub struct HeadlessMode: u8 {
        /// Disable audio mixing.
        const NO_AUDIO = 0x01;
        /// Disable pixel rendering.
        const NO_VIDEO = 0x02;
    }
}

/// Set of desired mapper revisions to use when loading a ROM matching the available mapper types.
#[derive(Default, Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[must_use]
#[non_exhaustive]
pub struct MapperRevisionsConfig {
    /// MMC3 mapper revision.
    pub mmc3: Mmc3Revision,
    /// BF909 mapper revision.
    pub bf909: Bf909Revision,
}

impl MapperRevisionsConfig {
    /// Set the desired mapper revision to use when loading a ROM matching the available mapper types.
    pub const fn set(&mut self, rev: MapperRevision) {
        match rev {
            MapperRevision::Mmc3(rev) => self.mmc3 = rev,
            MapperRevision::Bf909(rev) => self.bf909 = rev,
        }
    }
}

/// Control deck configuration settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
#[must_use]
#[non_exhaustive]
pub struct Config {
    /// Video filter.
    pub filter: VideoFilter,
    /// NES region.
    pub region: NesRegion,
    /// RAM initialization state.
    pub ram_state: RamState,
    /// Four player adapter.
    pub four_player: FourPlayer,
    /// Enable zapper gun.
    pub zapper: bool,
    /// Whether to support concurrent D-Pad input which wasn't possible on the original NES.
    pub concurrent_dpad: bool,
    /// Apu channels enabled.
    pub channels_enabled: [bool; Apu::MAX_CHANNEL_COUNT],
    /// Headless mode.
    pub headless_mode: HeadlessMode,
    /// Which mapper revisions to emulate for any ROM loaded that uses this mapper.
    pub mapper_revisions: MapperRevisionsConfig,
    /// Whether to emulate PPU warmup where writes to certain registers are ignored. Can result in
    /// some games not working correctly.
    ///
    /// See: <https://www.nesdev.org/wiki/PPU_power_up_state>
    pub emulate_ppu_warmup: bool,
    /// Whether clocking discards the previous call's audio samples first.
    ///
    /// `true`, the default, means [`ControlDeck::audio_samples`] holds exactly what the most recent
    /// clock produced. Set it to `false` to accumulate samples across several clock calls, in which
    /// case you must call [`ControlDeck::clear_audio_samples`] yourself or the buffer grows without
    /// bound.
    pub clear_audio_on_clock: bool,
    /// Reproduce FCEUmm's approximations of the hardware instead of the hardware itself.
    ///
    /// `false`, the default, emulates the console. `true` is for embedders that need frame-exact
    /// agreement with FCEUmm (libretro `fceumm`, commit e0f3086). Each behaviour it changes is a
    /// numbered quirk, `FQ-NN`, labelled at every place it is implemented; see `FCEUMM_QUIRKS.md`.
    pub fceumm_compat: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            filter: VideoFilter::default(),
            region: NesRegion::Auto,
            ram_state: RamState::Random,
            four_player: FourPlayer::default(),
            zapper: false,
            concurrent_dpad: false,
            channels_enabled: [true; Apu::MAX_CHANNEL_COUNT],
            headless_mode: HeadlessMode::empty(),
            mapper_revisions: MapperRevisionsConfig::default(),
            emulate_ppu_warmup: false,
            clear_audio_on_clock: true,
            fceumm_compat: false,
        }
    }
}

/// Represents a loaded ROM [`Cart`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct LoadedRom {
    /// Name of ROM.
    pub name: String,
    /// Whether the loaded Cart is battery-backed.
    pub battery_backed: bool,
    /// Auto-detected of the loaded Cart.
    pub region: NesRegion,
}

/// Represents an NES Control Deck. Encapsulates the entire emulation state.
#[derive(Debug, Clone)]
#[must_use]
pub struct ControlDeck {
    /// Whether a ROM is loaded and the emulation is currently running or not.
    running: bool,
    /// Video output and filtering.
    video: Video,
    /// Whether `video.frame` still needs the filter applied, letting `frame_buffer` be cached when
    /// called more than once between clocks.
    video_frame_stale: bool,
    /// Whether clocking discards the previous call's audio samples first.
    clear_audio_on_clock: bool,
    /// The currently loaded ROM [`Cart`], if any.
    loaded_rom: Option<LoadedRom>,
    /// Mapper revisions to emulate for any ROM loaded that matches the given mappers.
    mapper_revisions: MapperRevisionsConfig,
    /// Whether to auto-detect the region based on the loaded Cart.
    auto_detect_region: bool,
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
        let mut bus = Bus::new(cfg.region, cfg.ram_state);
        bus.ppu.skip_rendering = cfg.headless_mode.contains(HeadlessMode::NO_VIDEO);
        bus.ppu.emulate_warmup = cfg.emulate_ppu_warmup;
        bus.ppu.fceumm_compat = cfg.fceumm_compat;
        bus.apu.skip_mixing = cfg.headless_mode.contains(HeadlessMode::NO_AUDIO);
        if cfg.region.is_auto() {
            bus.set_region(NesRegion::default());
        } else {
            bus.set_region(cfg.region);
        }
        bus.input.set_concurrent_dpad(cfg.concurrent_dpad);
        bus.input.set_four_player(cfg.four_player);
        bus.input.connect_zapper(cfg.zapper);
        for (i, enabled) in cfg.channels_enabled.iter().enumerate() {
            match Channel::try_from(i) {
                Ok(channel) => bus.apu.set_channel_enabled(channel, *enabled),
                Err(apu::ParseChannelError) => tracing::error!("invalid APU channel: {i}"),
            }
        }
        Self {
            running: false,
            video: Video::with_filter(cfg.filter),
            video_frame_stale: true,
            clear_audio_on_clock: cfg.clear_audio_on_clock,
            loaded_rom: None,
            mapper_revisions: cfg.mapper_revisions,
            auto_detect_region: cfg.region.is_auto(),
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
        let cart = Cart::from_rom(&name, rom, self.bus.ram_state).map_err(|err| match err {
            cart::Error::InvalidMapper(mapper::Error::Unimplemented(num)) => {
                Error::UnimplementedMapper(num)
            }
            err => Error::Cart(err),
        })?;
        let loaded_rom = LoadedRom {
            name,
            battery_backed: cart.battery_backed(),
            region: cart.region,
        };
        if self.auto_detect_region {
            self.bus.set_region(loaded_rom.region);
        }
        self.bus.load_cart(cart);
        self.loaded_rom = Some(loaded_rom.clone());
        self.update_mapper_revisions();
        self.reset(ResetKind::Hard);
        Ok(loaded_rom)
    }

    /// Unloads the currently loaded ROM.
    pub fn unload_rom(&mut self) {
        self.loaded_rom = None;
        self.bus.unload_cart();
        self.running = false;
    }

    /// Replaces the running console with a previously saved [`Bus`] state.
    ///
    /// # Errors
    ///
    /// If the state was not produced by the currently loaded cart, in which case the running
    /// console is left untouched.
    #[inline]
    pub fn load_bus(&mut self, bus: Bus) -> Result<()> {
        self.bus.load_state(bus)?;
        // Page tables are derived state and aren't serialized, so rebuild them from the restored
        // mapper registers.
        self.bus.rebuild_mapper_state();
        // A board revision is the player's choice, not the machine's - the state carries whichever
        // one was selected when it was recorded, and restoring that would change IRQ timing
        // underneath a running game.
        self.update_mapper_revisions();
        // The console this deck was showing is gone; anything cached from it describes a machine
        // that is no longer here.
        self.video_frame_stale = true;
        Ok(())
    }

    /// Set the [`MapperRevision`] to emulate for the any ROM loaded that uses this mapper.
    #[inline]
    pub const fn set_mapper_revision(&mut self, rev: MapperRevision) {
        self.mapper_revisions.set(rev);
        self.update_mapper_revisions();
    }

    /// Internal method to update the loaded ROM mapper revision when `mapper_revisions` is
    /// updated.
    const fn update_mapper_revisions(&mut self) {
        match &mut self.bus.mapper {
            Mapper::Txrom(mapper) => {
                mapper.set_revision(self.mapper_revisions.mmc3);
            }
            Mapper::Bf909x(mapper) => {
                mapper.set_revision(self.mapper_revisions.bf909);
            }
            // Remaining mappers all have more concrete detection via ROM headers
            _ => (),
        }
    }

    /// Set the headless mode which can increase performance when the frame and audio outputs are
    /// not needed.
    #[inline]
    pub const fn set_headless_mode(&mut self, mode: HeadlessMode) {
        self.bus.ppu.skip_rendering = mode.contains(HeadlessMode::NO_VIDEO);
        self.bus.apu.skip_mixing = mode.contains(HeadlessMode::NO_AUDIO);
    }

    /// Returns the loaded ROM, if any.
    #[inline]
    #[must_use]
    pub const fn loaded_rom(&self) -> Option<&LoadedRom> {
        self.loaded_rom.as_ref()
    }

    /// Returns the auto-detected [`NesRegion`] for the loaded ROM. Returns `None` if no ROM is
    /// loaded.
    #[inline]
    #[must_use]
    pub fn cart_region(&self) -> Option<NesRegion> {
        self.loaded_rom.as_ref().map(|rom| rom.region)
    }

    /// Returns whether the loaded ROM is battery-backed. Returns `None` if no ROM is loaded.
    #[inline]
    #[must_use]
    pub fn cart_battery_backed(&self) -> Option<bool> {
        self.loaded_rom.as_ref().map(|rom| rom.battery_backed)
    }

    /// Returns the NES Work RAM.
    #[inline]
    #[must_use]
    pub fn wram(&self) -> &[u8; bus::size::WRAM] {
        self.bus.wram()
    }

    /// Returns the NES Work RAM, for writing.
    #[inline]
    #[must_use]
    pub fn wram_mut(&mut self) -> &mut [u8; bus::size::WRAM] {
        self.bus.wram_mut()
    }

    /// The cart's whole battery as one slice. Empty for a cart without a battery.
    #[must_use]
    pub fn sram(&mut self) -> &[u8] {
        if self.cart_battery_backed() == Some(true) {
            self.bus.sram()
        } else {
            &[]
        }
    }

    /// Replaces battery-backed Save RAM with raw bytes, as [`ControlDeck::sram`] hands them out.
    /// No-op for a cart without a battery.
    pub fn set_sram(&mut self, data: &[u8]) {
        if self.cart_battery_backed() == Some(true) {
            self.bus.set_sram(data);
        }
    }

    /// Returns the frame as raw PPU pixels, one palette index per pixel.
    #[inline]
    #[must_use]
    pub fn frame_buffer_raw(&self) -> &[u16; ppu::size::FRAME] {
        self.bus.ppu.frame_buffer()
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
            .apply_filter(self.bus.ppu.frame_buffer(), self.bus.ppu.frame_number())
    }

    /// Returns the number of frames the console has rendered since power on.
    #[inline(always)]
    #[must_use]
    pub const fn frame_number(&self) -> u32 {
        self.bus.ppu.frame_number()
    }

    /// Returns the audio samples produced by the most recent clock.
    #[inline(always)]
    #[must_use]
    pub fn audio_samples(&self) -> &[f32] {
        self.bus.audio_samples()
    }

    /// Discards the audio samples produced so far.
    #[inline]
    pub fn clear_audio_samples(&mut self) {
        self.bus.clear_audio_samples();
    }

    /// Sets whether clocking discards the previous call's audio samples first.
    #[inline]
    pub const fn set_clear_audio_on_clock(&mut self, enabled: bool) {
        self.clear_audio_on_clock = enabled;
    }

    /// CPU clock rate based on currently configured NES region.
    #[inline]
    #[must_use]
    pub const fn clock_rate(&self) -> f32 {
        self.bus.clock_rate()
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
        if self.clear_audio_on_clock {
            self.bus.clear_audio_samples();
        }
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
    pub const fn cpu_corrupted(&self) -> bool {
        self.bus.cpu.corrupted
    }

    /// Returns the ControlDeck bus state which contains all components.
    #[inline]
    pub const fn bus(&self) -> &Bus {
        &self.bus
    }

    /// Returns a mutable reference to the ControlDeck bus state.
    #[inline]
    pub const fn bus_mut(&mut self) -> &mut Bus {
        &mut self.bus
    }

    /// Enable/Disable Four Score for 4-player controllers.
    #[inline]
    pub fn set_four_player(&mut self, four_player: FourPlayer) {
        self.bus.input.set_four_player(four_player);
    }

    /// Returns the current [`Joypad`] state for a given controller slot.
    #[inline]
    pub const fn joypad(&self, slot: Player) -> &Joypad {
        self.bus.input.joypad(slot)
    }

    /// Returns a mutable reference to the current [`Joypad`] state for a given controller slot.
    #[inline]
    pub const fn joypad_mut(&mut self, slot: Player) -> &mut Joypad {
        self.bus.input.joypad_mut(slot)
    }

    /// Returns whether the [`Zapper`](crate::input::Zapper) gun is connected.
    #[inline]
    pub const fn zapper_connected(&self) -> bool {
        self.bus.input.zapper.connected
    }

    /// Enable [`Zapper`](crate::input::Zapper) gun.
    #[inline]
    pub const fn connect_zapper(&mut self, enabled: bool) {
        self.bus.input.connect_zapper(enabled);
    }

    /// Trigger [`Zapper`](crate::input::Zapper) gun.
    #[inline]
    pub fn trigger_zapper(&mut self) {
        self.bus.input.zapper.trigger();
    }

    /// Aim [`Zapper`](crate::input::Zapper) gun.
    #[inline]
    pub const fn aim_zapper(&mut self, x: u16, y: u16) {
        self.bus.input.zapper.aim(x, y);
    }

    /// Set the video filter for frame buffer output when calling [`ControlDeck::frame_buffer`].
    #[inline]
    pub const fn set_filter(&mut self, filter: VideoFilter) {
        self.video.filter = filter;
        // The cached frame was filtered with the old one.
        self.video_frame_stale = true;
    }

    /// Set the [`Apu`] sample rate.
    #[inline]
    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        self.bus.apu.set_sample_rate(sample_rate);
    }

    /// Returns whether a given [`Apu`] [`Channel`] is enabled.
    #[inline]
    #[must_use]
    pub const fn apu_channel_enabled(&self, channel: Channel) -> bool {
        self.bus.apu.channel_enabled(channel)
    }

    /// Enable or disable a given [`Apu`] [`Channel`].
    #[inline]
    pub const fn set_apu_channel_enabled(&mut self, channel: Channel, enabled: bool) {
        self.bus.apu.set_channel_enabled(channel, enabled);
    }

    /// Returns whether the control deck is currently running.
    #[inline]
    #[must_use]
    pub const fn is_running(&self) -> bool {
        self.running
    }

    /// Get the NES format for the emulation.
    pub const fn region(&self) -> NesRegion {
        self.bus.region
    }

    /// Set the NES format for the emulation.
    pub fn set_region(&mut self, region: NesRegion) {
        self.auto_detect_region = region.is_auto();
        if self.auto_detect_region {
            self.bus.set_region(self.cart_region().unwrap_or_default());
        } else {
            self.bus.set_region(region);
        }
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
        let mut deck = ControlDeck::with_config(Config {
            ram_state: RamState::AllZeros,
            ..Default::default()
        });
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
    /// form. Page tables are derived state, so this only holds if the restore replays the mapper's
    /// registers through `Map::update_banks`.
    #[test]
    fn a_restored_state_resumes_identically() {
        let mut deck = spritecans();
        run(&mut deck, 30);
        let state = deck.bus().clone();
        let bytes = postcard::to_allocvec(deck.bus()).expect("serializes");
        let expected = run(&mut deck, 30);

        let mut restored = spritecans();
        run(&mut restored, 5); // somewhere else entirely
        restored.load_bus(state).expect("loads");
        assert_eq!(run(&mut restored, 30), expected);

        let mut decoded = spritecans();
        run(&mut decoded, 5);
        decoded
            .load_bus(postcard::from_bytes(&bytes).expect("deserializes"))
            .expect("loads");
        assert_eq!(run(&mut decoded, 30), expected);
    }

    /// Clocking drops the previous call's samples by default, so a caller that never reaches for
    /// `clear_audio_samples` still does not grow the buffer without bound.
    #[test]
    fn audio_samples_do_not_accumulate_across_frames() {
        let mut deck = spritecans();
        deck.clock_frame().expect("clocks");
        let one_frame = deck.audio_samples().len();
        assert!(one_frame > 0, "a frame should produce audio");

        let mut max = 0;
        for _ in 0..60 {
            deck.clock_frame().expect("clocks");
            max = max.max(deck.audio_samples().len());
        }
        assert!(
            max < one_frame * 2,
            "samples accumulated across frames without a manual clear: \
             one frame is {one_frame}, peaked at {max}"
        );
    }

    /// Opting out lets the samples accumulate, for callers that clock several times before draining.
    #[test]
    fn audio_samples_accumulate_when_opted_out() {
        let mut deck = spritecans();
        deck.set_clear_audio_on_clock(false);
        deck.clock_frame().expect("clocks");
        let one_frame = deck.audio_samples().len();

        for _ in 0..9 {
            deck.clock_frame().expect("clocks");
        }
        assert!(
            deck.audio_samples().len() > one_frame * 9,
            "10 frames should have accumulated roughly 10 frames of audio"
        );

        deck.clear_audio_samples();
        assert_eq!(deck.audio_samples().len(), 0);
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

    /// A restored state replaces the console outright. Anything cached from the old one describes
    /// a machine that is no longer here, including the frame a caller is about to blit.
    #[test]
    fn loading_a_state_refreshes_the_frame() {
        let mut deck = spritecans();
        deck.clock_frame().expect("clocks");
        let early = deck.bus().clone();
        let early_frame = deck.frame_buffer().to_vec();

        run(&mut deck, 30);
        let late_frame = deck.frame_buffer().to_vec();
        assert_ne!(early_frame, late_frame, "the game has moved on");

        deck.load_bus(early).expect("restores");
        assert_eq!(
            deck.frame_buffer().to_vec(),
            early_frame,
            "reports the restored console, not the one it replaced"
        );
    }

    /// Which buttons are held is emulated state the game reads, so it travels with the state.
    #[test]
    fn restoring_a_state_keeps_the_buttons_it_recorded() {
        let mut deck = spritecans();
        deck.clock_frame().expect("clocks");
        deck.joypad_mut(Player::One).set_button(JoypadBtn::A, true);
        let held = deck.bus().clone();

        deck.joypad_mut(Player::One).set_button(JoypadBtn::A, false);
        deck.load_bus(held).expect("restores");
        assert!(
            deck.joypad_mut(Player::One).button(JoypadBtn::A.into()),
            "the state recorded A held, so the restored console has it held"
        );
    }

    /// A board revision is the player's choice, not the machine's. A state recorded under another
    /// one must not put it back and change IRQ timing underneath a running game.
    #[test]
    fn restoring_a_state_keeps_the_selected_mapper_revision() {
        let mut deck = ControlDeck::with_config(Config {
            ram_state: RamState::AllZeros,
            ..Default::default()
        });
        let path = "test_roms/mapper/m004_txrom/rev_b.nes";
        let mut rom = File::open(path).expect("test rom exists");
        deck.load_rom(path, &mut rom).expect("test rom loads");

        let revision = |deck: &ControlDeck| match &deck.bus().mapper {
            Mapper::Txrom(board) => board.mmc3.revision,
            _ => panic!("expected an MMC3 board"),
        };

        deck.set_mapper_revision(MapperRevision::Mmc3(Mmc3Revision::A));
        let recorded_under_a = deck.bus().clone();
        assert_eq!(revision(&deck), Mmc3Revision::A);

        deck.set_mapper_revision(MapperRevision::Mmc3(Mmc3Revision::BC));
        deck.load_bus(recorded_under_a).expect("restores");
        assert_eq!(
            revision(&deck),
            Mmc3Revision::BC,
            "the state carries A, but BC is what the player has selected now"
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
