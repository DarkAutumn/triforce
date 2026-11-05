//! The machine-level API the rest of triforce uses (plan step 11). Everything else in this crate is
//! the emulator behind it, being cut down by the subtraction ladder (`UPSTREAM.md`); this module's
//! interface stays put while that happens.

use crate::{
    control_deck::{Config, ControlDeck, HeadlessMode},
    input::{JoypadBtnState, Player},
    memory::{RamState, Src},
    video::VideoFilter,
};
use std::sync::Arc;

/// Frame width in pixels (the whole picture, no overscan crop).
pub const WIDTH: usize = 256;
/// Frame height in pixels (the whole picture, no overscan crop).
pub const HEIGHT: usize = 240;

/// SHA-1 of the 128 KiB PRG payload of *The Legend of Zelda* (USA) as triforce uses it.
pub const ZELDA_PAYLOAD_SHA1: [u8; 20] = [
    0xa1, 0x2d, 0x74, 0xc7, 0x3a, 0x04, 0x81, 0x59, 0x9a, 0x5d, 0x83, 0x23, 0x61, 0xd1, 0x68, 0xf4, 0x73, 0x7b,
    0xbc, 0xf6,
];

/// Why a ROM image was refused.
#[derive(Debug, thiserror::Error)]
pub enum RomError {
    /// No iNES header.
    #[error("not an iNES file")]
    NotInes,
    /// An iNES image, but not shaped like Zelda's cartridge.
    #[error("not The Legend of Zelda (mapper 1, 8x16 KiB PRG, CHR-RAM): {0}")]
    WrongCart(String),
    /// Zelda-shaped, but not the expected PRG contents.
    #[error("ROM payload SHA-1 {0} is not The Legend of Zelda's")]
    WrongPayload(String),
}

/// A validated Zelda ROM image.
#[derive(Debug)]
pub struct Rom {
    bytes: Vec<u8>,
    payload_sha1: [u8; 20],
}

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
        Ok(Arc::new(Rom { bytes: bytes.to_vec(), payload_sha1 }))
    }

    /// SHA-1 of the bytes after the 16-byte iNES header.
    pub const fn payload_sha1(&self) -> [u8; 20] {
        self.payload_sha1
    }
}

/// A complete, restorable machine state.
#[derive(Clone)]
pub struct MachineState {
    deck: crate::bus::Bus,
    frame: u64,
}

/// One Zelda console.
pub struct ZCore {
    rom: Arc<Rom>,
    deck: ControlDeck,
    frame: u64,
    rendered: bool,
    pixels: Box<[u8; WIDTH * HEIGHT]>,
}

impl ZCore {
    /// Power on: CPU RAM $FF, battery PRG-RAM zeroed, no frame run yet.
    pub fn power_on(rom: Arc<Rom>) -> Self {
        let mut cfg = Config::default();
        cfg.ram_state = RamState::AllOnes;
        cfg.sram_dir = None;
        cfg.filter = VideoFilter::Pixellate;
        cfg.headless_mode = HeadlessMode::NO_AUDIO;
        cfg.concurrent_dpad = true;
        cfg.fceumm_compat = true;
        let mut deck = ControlDeck::with_config(cfg);
        deck.load_rom("zelda.nes", &mut &rom.bytes[..]).expect("a validated Zelda ROM loads");
        let len = deck.sram().len();
        deck.set_sram(&vec![0u8; len]);
        Self { rom, deck, frame: 0, rendered: false, pixels: Box::new([0; WIDTH * HEIGHT]) }
    }

    /// The ROM this console runs.
    pub fn rom(&self) -> &Arc<Rom> {
        &self.rom
    }

    /// Run one frame with the NES controller byte (A=0x01, B=0x02, Select=0x04, Start=0x08, Up, Down,
    /// Left, Right). With `render` false no pixel is produced; emulation is identical either way.
    pub fn run_frame(&mut self, buttons: u8, render: bool) {
        self.deck.bus_mut().ppu.skip_rendering = !render;
        self.deck.joypad_mut(Player::One).buttons = JoypadBtnState::from_bits_truncate(u16::from(buttons));
        let _ = self.deck.clock_frame().expect("zcore clock_frame");
        self.frame += 1;
        self.rendered = render;
        if render {
            for (dst, &src) in self.pixels.iter_mut().zip(self.deck.frame_buffer_raw().iter()) {
                *dst = (src & 0x3F) as u8;
            }
        }
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
        self.deck.wram()
    }

    /// Cartridge PRG-RAM $6000-$7FFF.
    pub fn prg_ram(&self) -> &[u8; 0x2000] {
        self.deck.bus().memory.region_ref(Src::PrgRam)[..0x2000].try_into().expect("8 KiB PRG-RAM")
    }

    /// Write RAM ($0000-$07FF) or PRG-RAM ($6000-$7FFF). Panics on any other address.
    pub fn poke(&mut self, addr: u16, value: u8) {
        match addr {
            0x0000..0x0800 => self.deck.wram_mut()[usize::from(addr)] = value,
            0x6000..0x8000 => {
                self.deck.bus_mut().memory.region_mut(Src::PrgRam)[usize::from(addr) - 0x6000] = value;
            }
            _ => panic!("poke address ${addr:04X} is outside $0000-$07FF/$6000-$7FFF"),
        }
    }

    /// A complete copy of the machine state.
    pub fn snapshot(&self) -> MachineState {
        MachineState { deck: self.deck.bus().clone(), frame: self.frame }
    }

    /// Return to a [`ZCore::snapshot`]. The next frame must be rendered before `pixels` is valid.
    pub fn restore(&mut self, state: &MachineState) {
        *self.deck.bus_mut() = state.deck.clone();
        self.frame = state.frame;
        self.rendered = false;
    }

    /// Frames run since power-on (snapshot/restore carry it).
    pub const fn frame(&self) -> u64 {
        self.frame
    }

    /// CPU cycles since power-on.
    pub fn cycles(&self) -> u64 {
        u64::from(self.deck.bus().cpu.cycle)
    }

    /// The emulator underneath, for tools that read internal state.
    pub const fn deck(&self) -> &ControlDeck {
        &self.deck
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
