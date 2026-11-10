//! The emulator surface the game model needs.
//!
//! Implemented by `zcore::ZCore` (shipping) and, in tests, by counting wrappers and scripted doubles.
//! Everything above this trait — the game view, executor, session — is emulator-agnostic.

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

    /// Whether the last frame was rendered.
    fn rendered(&self) -> bool {
        false
    }

    /// `restore` after a detour that rendered no frame (a look-ahead), with the `rendered` flag from before it, so
    /// the frame before the detour stays the screen.
    fn restore_keeping_frame(&mut self, state: &Self::State, _rendered: bool) {
        self.restore(state);
    }

    /// Reads RAM or PRG-RAM by CPU address.
    fn peek(&self, addr: u16) -> u8 {
        match addr {
            0x0000..0x0800 => self.ram()[addr as usize],
            0x6000..0x8000 => self.prg_ram()[addr as usize - 0x6000],
            _ => panic!("peek address ${addr:04X} is outside $0000-$07FF/$6000-$7FFF"),
        }
    }
}

#[cfg(feature = "zcore")]
impl Nes for zcore::ZCore {
    type State = zcore::MachineState;

    fn run_frame(&mut self, buttons: u8, render: bool) {
        zcore::ZCore::run_frame(self, buttons, render);
    }

    fn ram(&self) -> &[u8; 0x800] {
        zcore::ZCore::ram(self)
    }

    fn prg_ram(&self) -> &[u8; 0x2000] {
        zcore::ZCore::prg_ram(self)
    }

    fn poke(&mut self, addr: u16, value: u8) {
        zcore::ZCore::poke(self, addr, value);
    }

    fn snapshot(&self) -> Self::State {
        zcore::ZCore::snapshot(self)
    }

    fn restore(&mut self, state: &Self::State) {
        zcore::ZCore::restore(self, state);
    }

    fn rendered(&self) -> bool {
        zcore::ZCore::rendered(self)
    }

    fn restore_keeping_frame(&mut self, state: &Self::State, rendered: bool) {
        zcore::ZCore::restore_keeping_frame(self, state, rendered);
    }
}
