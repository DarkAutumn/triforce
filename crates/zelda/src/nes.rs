//! The emulator surface the game model needs.
//!
//! Implemented by `zcore::ZCore` (shipping) and, in tests only, by a wrapper over the dev-only FCEUmm
//! oracle (`oracle` crate, a dev-dependency). Everything above this trait — the game view, executor,
//! session — is emulator-agnostic.

pub trait Nes {
    /// A complete, restorable machine snapshot (cheap `Copy` state for zcore).
    type State: Clone;

    /// Runs one frame (to the frame boundary at scanline 240) with the NES controller byte
    /// (A=0x01, B=0x02, Select=0x04, Start=0x08, Up=0x10, Down=0x20, Left=0x40, Right=0x80).
    /// `render` false skips pixel output for this frame only; emulation is identical either way.
    fn run_frame(&mut self, buttons: u8, render: bool);

    /// CPU RAM $0000-$07FF.
    fn ram(&self) -> &[u8; 0x800];

    /// Cartridge PRG-RAM $6000-$7FFF.
    fn prg_ram(&self) -> &[u8; 0x2000];

    /// Direct write to RAM ($0000-$07FF) or PRG-RAM ($6000-$7FFF). Panics on any other address.
    fn poke(&mut self, addr: u16, value: u8);

    fn snapshot(&self) -> Self::State;

    fn restore(&mut self, state: &Self::State);

    /// Reads RAM or PRG-RAM by CPU address.
    fn peek(&self, addr: u16) -> u8 {
        match addr {
            0x0000..0x0800 => self.ram()[addr as usize],
            0x6000..0x8000 => self.prg_ram()[addr as usize - 0x6000],
            _ => panic!("peek address ${addr:04X} is outside $0000-$07FF/$6000-$7FFF"),
        }
    }
}
