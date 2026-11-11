//! The machine-level API the rest of triforce uses (plan step 11). Everything else in this crate is
//! the emulator behind it, being cut down by the subtraction ladder (`UPSTREAM.md`); this module's
//! interface stays put while that happens.

use crate::{
    bus::{Bus, State},
    cart::Cart,
    common::ResetKind,
    cpu::TraceHook,
    input::Player,
    memory::PrgRom,
};
use std::sync::Arc;

/// Frame width in pixels (the whole picture, no overscan crop).
pub const WIDTH: usize = 256;
/// Frame height in pixels (the whole picture, no overscan crop).
pub const HEIGHT: usize = 240;

/// SHA-1 of the 128 KiB PRG payload of *The Legend of Zelda* (USA) as triforce uses it.
pub const ZELDA_PAYLOAD_SHA1: [u8; 20] = [
    0xa1, 0x2d, 0x74, 0xc7, 0x3a, 0x04, 0x81, 0x59, 0x9a, 0x5d, 0x83, 0x23, 0x61, 0xd1, 0x68, 0xf4,
    0x73, 0x7b, 0xbc, 0xf6,
];

/// Why a ROM image was refused.
#[derive(Debug)]
pub enum RomError {
    /// No iNES header.
    NotInes,
    /// An iNES image, but not shaped like Zelda's cartridge.
    WrongCart(String),
    /// Zelda-shaped, but not the expected PRG contents.
    WrongPayload(String),
}

impl std::fmt::Display for RomError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotInes => f.write_str("not an iNES file"),
            Self::WrongCart(why) => write!(
                f,
                "not The Legend of Zelda (mapper 1, 8x16 KiB PRG, CHR-RAM): {why}"
            ),
            Self::WrongPayload(sha) => {
                write!(f, "ROM payload SHA-1 {sha} is not The Legend of Zelda's")
            }
        }
    }
}

impl std::error::Error for RomError {}

/// A validated Zelda ROM image.
pub struct Rom {
    bytes: Vec<u8>,
    /// The PRG-ROM image every console of this ROM shares.
    prg: Arc<PrgRom>,
    payload_sha1: [u8; 20],
}

impl std::fmt::Debug for Rom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Rom")
            .field("payload_sha1", &self.payload_sha1)
            .finish_non_exhaustive()
    }
}

/// The iNES header of the Zelda image every console is built from (that of the common dump).
const CANONICAL_HEADER: [u8; 16] = [b'N', b'E', b'S', 0x1A, 8, 0, 0x12, 0, 0, 0, 0, 0, 0, 0, 0, 0];

impl Rom {
    /// Validate an iNES image (header, cartridge shape, payload SHA-1).
    pub fn from_ines(bytes: &[u8]) -> Result<Arc<Rom>, RomError> {
        if bytes.len() < 16 || &bytes[..4] != b"NES\x1a" {
            return Err(RomError::NotInes);
        }
        let mapper = (bytes[6] >> 4) | (bytes[7] & 0xF0);
        if mapper != 1 || bytes[4] != 8 || bytes[5] != 0 || bytes.len() != 16 + 8 * 0x4000 {
            return Err(RomError::WrongCart(format!(
                "mapper {mapper}, {} PRG banks, {} CHR banks, {} bytes",
                bytes[4],
                bytes[5],
                bytes.len()
            )));
        }
        let payload_sha1 = sha1_smol::Sha1::from(&bytes[16..]).digest().bytes();
        if payload_sha1 != ZELDA_PAYLOAD_SHA1 {
            let hex: String = payload_sha1.iter().map(|b| format!("{b:02x}")).collect();
            return Err(RomError::WrongPayload(hex));
        }
        let prg: Box<PrgRom> = bytes[16..]
            .to_vec()
            .into_boxed_slice()
            .try_into()
            .expect("128 KiB PRG-ROM");
        // The machine is built from a canonical header (mapper 1, 8 PRG banks, CHR-RAM, battery
        // PRG-RAM, iNES 1.0) plus the validated payload, so header bits the cart loader would refuse
        // or interpret (trainer, four-screen, NES 2.0 sizes, battery) can't fail or change it.
        let mut canonical = CANONICAL_HEADER.to_vec();
        canonical.extend_from_slice(&bytes[16..]);
        Ok(Arc::new(Rom {
            bytes: canonical,
            prg: Arc::from(prg),
            payload_sha1,
        }))
    }

    /// SHA-1 of the bytes after the 16-byte iNES header.
    pub const fn payload_sha1(&self) -> [u8; 20] {
        self.payload_sha1
    }
}

/// A complete, restorable machine state: every emulated byte, in fixed arrays, so a snapshot is a
/// plain copy. The PRG-ROM is not part of it (the [`ZCore`] that restores it supplies its own),
/// nor is the frame buffer.
#[derive(Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct MachineState {
    pub(crate) state: State,
    frame: u64,
    cycles: u64,
}

impl MachineState {
    /// Write RAM ($0000-$07FF) or PRG-RAM ($6000-$7FFF) of a saved state, as [`ZCore::poke`] does on a running
    /// machine (a snapshot fixed up before it is written to a file). Panics on any other address.
    pub fn poke(&mut self, addr: u16, value: u8) {
        match addr {
            0x0000..0x0800 => self.state.wram[usize::from(addr)] = value,
            0x6000..0x8000 => self.state.memory.prg_ram_mut()[usize::from(addr) - 0x6000] = value,
            _ => panic!("poke address ${addr:04X} is outside $0000-$07FF/$6000-$7FFF"),
        }
    }

    /// Read RAM or PRG-RAM of a saved state. Panics on any other address.
    pub fn peek(&self, addr: u16) -> u8 {
        match addr {
            0x0000..0x0800 => self.state.wram[usize::from(addr)],
            0x6000..0x8000 => self.state.memory.prg_ram()[usize::from(addr) - 0x6000],
            _ => panic!("peek address ${addr:04X} is outside $0000-$07FF/$6000-$7FFF"),
        }
    }

    /// The checks a decoded state must pass before a [`ZCore`] runs it: the fields that only ever
    /// hold one value, positions in range and indexes that can't leave their tables. A state that
    /// fails them would load a different machine, panic or never finish a frame.
    pub fn validate(&self) -> Result<(), String> {
        let s = &self.state;
        let checks = [
            (s.cpu.start_cycles == 6 && s.cpu.end_cycles == 6, "CPU cycle constants"),
            (s.ppu.clock_divider == 4, "PPU clock divider"),
            (s.ppu.vblank_scanline == 241 && s.ppu.prerender_scanline == 261, "PPU scanline constants"),
            (s.ppu.scanline <= 261 && s.ppu.cycle <= 340, "PPU position"),
            (s.apu.frame_counter.step < s.apu.frame_counter.step_cycles.len(), "APU frame counter step"),
            (s.memory.pages_in_arena(), "memory page outside the arena"),
        ];
        match checks.iter().find(|(ok, _)| !ok) {
            Some((_, what)) => Err(format!("invalid machine state: {what}")),
            None => Ok(()),
        }
    }
}

/// One Zelda console.
pub struct ZCore {
    rom: Arc<Rom>,
    bus: Bus,
    frame: u64,
    /// CPU cycles since power-on; the CPU's own counter is 32 bits and wraps after ~144,000
    /// frames, so each frame's advance is accumulated here.
    cycles: u64,
    rendered: bool,
    pixels: Box<[u8; WIDTH * HEIGHT]>,
}

impl ZCore {
    /// Power on: CPU RAM $00, battery PRG-RAM zeroed, no frame run yet.
    pub fn power_on(rom: Arc<Rom>) -> Self {
        let mut cart =
            Cart::from_rom("zelda.nes", &mut &rom.bytes[..]).expect("a validated Zelda ROM loads");
        // Share the ROM's image rather than keep the copy the cart was parsed into.
        cart.prg_rom = Arc::clone(&rom.prg);
        let mut bus = Bus::new();
        bus.load_cart(cart);
        bus.reset(ResetKind::Hard);
        // Power-on spec (TFMOVIE v1): an empty, zero-filled battery.
        bus.memory.prg_ram_mut().fill(0);
        Self {
            rom,
            cycles: u64::from(bus.cpu.cycle),
            bus,
            frame: 0,
            rendered: false,
            pixels: Box::new([0; WIDTH * HEIGHT]),
        }
    }

    /// The ROM this console runs.
    pub const fn rom(&self) -> &Arc<Rom> {
        &self.rom
    }

    /// Run one frame with the NES controller byte (A=0x01, B=0x02, Select=0x04, Start=0x08, Up, Down,
    /// Left, Right). With `render` false no pixel is produced; emulation is identical either way.
    pub fn run_frame(&mut self, buttons: u8, render: bool) {
        self.bus.render = render;
        self.bus.input.joypad_mut(Player::One).buttons = buttons;
        let start = self.bus.cpu.cycle;
        self.bus.sound.begin_frame(self.bus.apu.cpu_cycle);
        self.bus.clock_frame();
        self.bus.sound.end_frame(self.bus.apu.cpu_cycle);
        assert!(!self.bus.cpu.corrupted, "zcore: the CPU jammed");
        self.cycles += u64::from(self.bus.cpu.cycle.wrapping_sub(start));
        self.frame += 1;
        self.rendered = render;
        if render {
            for (dst, &src) in self.pixels.iter_mut().zip(self.bus.frame().iter()) {
                *dst = (src & 0x3F) as u8;
            }
        }
    }

    /// Starts or stops keeping each frame's sound events ([`ZCore::sound`]). Emulation is identical either
    /// way: sound is computed beside the machine ([`crate::sound`]).
    pub fn set_sound(&mut self, on: bool) {
        self.bus.sound.set_logging(on);
    }

    /// The sound events of the last frame run, for [`Synth::run`](crate::sound::Synth::run); `None` unless
    /// [`ZCore::set_sound`] is on.
    pub fn sound(&self) -> Option<crate::sound::Frame<'_>> {
        self.bus.sound.frame()
    }

    /// The tap and the APU a synthesizer resumes from.
    pub(crate) fn sound_source(&self) -> (&crate::sound::Tap, &crate::apu::Apu) {
        (&self.bus.sound, &self.bus.state.apu)
    }

    /// Palette indices (0..=63) of the last frame, row-major, or `None` if it was not rendered.
    pub fn pixels(&self) -> Option<&[u8; WIDTH * HEIGHT]> {
        self.rendered.then_some(&*self.pixels)
    }

    /// The last frame as RGB (FCEUmm's default palette), or false if it was not rendered. Grayscale
    /// is already applied to the palette indices by the PPU.
    pub fn rgb(&self, out: &mut [u8; WIDTH * HEIGHT * 3]) -> bool {
        if !self.rendered {
            return false;
        }
        for (&px, rgb) in self.pixels.iter().zip(out.chunks_exact_mut(3)) {
            rgb.copy_from_slice(&PALETTE[usize::from(px)]);
        }
        true
    }

    /// CPU RAM $0000-$07FF.
    pub fn ram(&self) -> &[u8; 0x800] {
        &self.bus.wram
    }

    /// Cartridge PRG-RAM $6000-$7FFF.
    pub fn prg_ram(&self) -> &[u8; 0x2000] {
        self.bus.memory.prg_ram()
    }

    /// Write RAM ($0000-$07FF) or PRG-RAM ($6000-$7FFF). Panics on any other address.
    pub fn poke(&mut self, addr: u16, value: u8) {
        match addr {
            0x0000..0x0800 => self.bus.wram[usize::from(addr)] = value,
            0x6000..0x8000 => {
                self.bus.memory.prg_ram_mut()[usize::from(addr) - 0x6000] = value;
            }
            _ => panic!("poke address ${addr:04X} is outside $0000-$07FF/$6000-$7FFF"),
        }
    }

    /// A complete copy of the machine state.
    pub fn snapshot(&self) -> MachineState {
        MachineState {
            state: self.bus.state,
            frame: self.frame,
            cycles: self.cycles,
        }
    }

    /// Return to a [`ZCore::snapshot`] (or a decoded state file). The next frame must be rendered
    /// before `pixels` is valid.
    pub fn restore(&mut self, state: &MachineState) {
        self.bus.state = state.state;
        self.frame = state.frame;
        self.cycles = state.cycles;
        self.rendered = false;
        self.bus.sound.forget();
    }

    /// Return to a snapshot taken during this run and set whether the last frame counts as rendered: for a detour
    /// that rendered no frame since the snapshot (a look-ahead; `run_frame(_, false)` leaves `pixels` alone), pass
    /// the flag from before the detour so the frame before it stays the screen.
    pub fn restore_keeping_frame(&mut self, state: &MachineState, rendered: bool) {
        self.bus.state = state.state;
        self.frame = state.frame;
        self.cycles = state.cycles;
        self.rendered = rendered;
        self.bus.sound.forget();
    }

    /// The last frame was rendered (`pixels` is valid).
    pub const fn rendered(&self) -> bool {
        self.rendered
    }

    /// Frames run since power-on (snapshot/restore carry it).
    pub const fn frame(&self) -> u64 {
        self.frame
    }

    /// CPU cycles since power-on.
    pub const fn cycles(&self) -> u64 {
        self.cycles
    }

    /// Call `hook` before every instruction (at the boundary, before the opcode fetch) until it is
    /// replaced or cleared with `None`. Unset, the cost is one branch per instruction; the hook only
    /// observes, so emulation is identical with or without it.
    pub fn set_trace(&mut self, hook: Option<TraceHook>) {
        self.bus.trace = hook;
    }

    /// The emulator underneath, for tools that read internal state.
    #[doc(hidden)]
    pub const fn bus(&self) -> &Bus {
        &self.bus
    }
}

/// FCEUmm's default NES palette, entries 0-63: libretro-fceumm `src/palettes/palettes.h` `palette[]`
/// at commit e0f3086 (FCEUmm is GPL-2.0-or-later; these 64 colour values are reproduced so that
/// rendered frames look like the stable-retro/FCEUmm frames triforce used before).
pub const PALETTE: [[u8; 3]; 64] = [
    [117, 117, 117],
    [36, 24, 142],
    [0, 0, 170],
    [69, 0, 158],
    [142, 0, 117],
    [170, 0, 16],
    [166, 0, 0],
    [125, 8, 0],
    [65, 44, 0],
    [0, 69, 0],
    [0, 81, 0],
    [0, 60, 20],
    [24, 60, 93],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [190, 190, 190],
    [0, 113, 239],
    [32, 56, 239],
    [130, 0, 243],
    [190, 0, 190],
    [231, 0, 89],
    [219, 40, 0],
    [203, 77, 12],
    [138, 113, 0],
    [0, 150, 0],
    [0, 170, 0],
    [0, 146, 56],
    [0, 130, 138],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [255, 255, 255],
    [60, 190, 255],
    [93, 150, 255],
    [207, 138, 255],
    [247, 121, 255],
    [255, 117, 182],
    [255, 117, 97],
    [255, 154, 56],
    [243, 190, 60],
    [130, 211, 16],
    [77, 223, 73],
    [89, 251, 154],
    [0, 235, 219],
    [121, 121, 121],
    [0, 0, 0],
    [0, 0, 0],
    [255, 255, 255],
    [170, 231, 255],
    [199, 215, 255],
    [215, 203, 255],
    [255, 199, 255],
    [255, 199, 219],
    [255, 190, 178],
    [255, 219, 170],
    [255, 231, 162],
    [227, 255, 162],
    [170, 243, 190],
    [178, 255, 207],
    [158, 255, 243],
    [199, 199, 199],
    [0, 0, 0],
    [0, 0, 0],
];
