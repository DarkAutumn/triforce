//! The cartridge board: SxROM/MMC1, which is what *The Legend of Zelda* runs on.
//!
//! <https://wiki.nesdev.org/w/index.php/Mapper>
//!
//! A board is register state plus [`Map::update_banks`], which publishes its banking into
//! [`Memory`]'s page tables; reads never reach the board. NROM exists only under `#[cfg(test)]`, for
//! the CPU/PPU/APU test ROMs.

use crate::{bus::State, cart::Cart, common::ResetKind, memory::Memory, ppu::Mirroring};
use serde::{Deserialize, Serialize};

pub mod m001_sxrom;
pub mod mmc1;

#[cfg(test)]
pub mod m000_nrom;

#[cfg(test)]
pub use m000_nrom::Nrom;
pub use m001_sxrom::Sxrom;
pub use mmc1::Mmc1;

/// Errors that mappers can return while loading.
#[derive(Debug)]
#[must_use]
#[non_exhaustive]
pub enum Error {
    /// No board serves this mapper number.
    Unimplemented(u16),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unimplemented(n) => write!(f, "unimplemented mapper: {n}"),
        }
    }
}

impl std::error::Error for Error {}

/// The cartridge board.
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
#[must_use]
pub enum Mapper {
    /// No board, i.e. no cart loaded.
    None(()),
    /// `SxROM`/`MMC1` (Mapper 001).
    Sxrom(Sxrom),
    /// `NROM` (Mapper 000), for the test ROMs.
    #[cfg(test)]
    Nrom(Nrom),
}

impl From<Sxrom> for Mapper {
    fn from(board: Sxrom) -> Self {
        Self::Sxrom(board)
    }
}

#[cfg(test)]
impl From<Nrom> for Mapper {
    fn from(board: Nrom) -> Self {
        Self::Nrom(board)
    }
}

/// One `match` over every `Mapper` variant, running the same call against each.
macro_rules! dispatch {
    ($self:expr, $board:ident => $call:expr) => {
        match $self {
            Mapper::None($board) => $call,
            Mapper::Sxrom($board) => $call,
            #[cfg(test)]
            Mapper::Nrom($board) => $call,
        }
    };
}

impl Mapper {
    /// An empty Mapper.
    pub const fn none() -> Self {
        Self::None(())
    }

    /// Whether mapper is `None`.
    pub const fn is_none(&self) -> bool {
        matches!(self, Self::None(_))
    }

    /// Pick and load the board a cart's mapper number calls for.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unimplemented`] if no board serves this mapper number.
    pub fn from_cart(cart: &mut Cart) -> Result<Self, Error> {
        match cart.mapper_num() {
            1 => Sxrom::load(cart),
            #[cfg(test)]
            0 => Nrom::load(cart),
            num => Err(Error::Unimplemented(num)),
        }
    }

    /// Returns the current [`Mirroring`] mode.
    #[inline(always)]
    pub fn mirroring(&self) -> Mirroring {
        dispatch!(self, m => m.mirroring())
    }

    /// Handle a CPU-space write, re-banking as needed.
    #[inline(always)]
    pub fn write_register(&mut self, memory: &mut Memory, addr: u16, val: u8) {
        dispatch!(self, m => m.write_register(memory, addr, val))
    }

    /// Rebuild the page tables from this board's register state.
    pub fn update_banks(&mut self, memory: &mut Memory) {
        dispatch!(self, m => m.update_banks(memory))
    }

    /// Reset the component given the [`ResetKind`].
    pub fn reset(&mut self, kind: ResetKind) {
        dispatch!(self, m => m.reset(kind))
    }

    /// Clock the board once per CPU cycle.
    #[inline(always)]
    pub fn clock(&mut self) {
        if let Self::Sxrom(m) = self {
            m.clock();
        }
    }

    /// Whether [`Mapper::clock`] has anything to count down; while not, a cycle may skip it.
    #[inline(always)]
    #[must_use]
    pub const fn clocking(&self) -> bool {
        matches!(self, Self::Sxrom(m) if m.mmc1.write_just_occurred > 0)
    }
}

impl Default for Mapper {
    fn default() -> Self {
        Self::none()
    }
}

/// Keeping the state derived from the board in step.
impl State {
    /// Rebuild the page tables from the mapper's register state (after a reset).
    pub fn rebuild_mapper_state(&mut self) {
        let Self { mapper, memory, .. } = self;
        mapper.update_banks(memory);
    }
}

/// Trait implemented by every board a [`Mapper`] can hold.
///
/// Boards are pure register state: every read is served from [`Memory`]'s page tables, which a
/// board rewrites from [`Map::write_register`] and [`Map::update_banks`].
pub trait Map {
    /// Returns the current [`Mirroring`] mode.
    fn mirroring(&self) -> Mirroring;

    /// Handle a CPU-space write, re-banking as needed.
    ///
    /// Called for every write in `$4100..=$FFFF`; the plain data store into PRG-RAM has already
    /// happened, so this only needs to handle registers.
    fn write_register(&mut self, _memory: &mut Memory, _addr: u16, _val: u8) {}

    /// Rebuild the page tables from this board's register state.
    ///
    /// Page tables are derived state and are not serialized, so they must be reconstructed after
    /// loading a save state. Also called after reset, where `Reset` has no access to [`Memory`].
    fn update_banks(&mut self, _memory: &mut Memory) {}

    /// Reset the board given the [`ResetKind`].
    fn reset(&mut self, _kind: ResetKind) {}
}

impl Map for () {
    fn mirroring(&self) -> Mirroring {
        Mirroring::default()
    }
}

#[cfg(test)]
pub(crate) mod test_utils {
    //! Shared scaffolding for the per-board banking tests.
    //!
    //! A board's job is to put the right physical bank behind the right CPU/PPU address, so these
    //! build a cart whose every 1K page is filled with its own index. A single `peek` then names
    //! the bank it was served from.

    use super::*;
    use crate::{
        cart::Cart,
        memory::{PAGE_SIZE, PRG_ROM_SIZE, PrgRom, Src},
    };
    use std::sync::Arc;

    /// A `Cart` with page-indexed contents: PRG-ROM page `n` is filled with `n` (mirrored up to the
    /// image size), CHR page `n` with `0x80 | n`, so PRG and CHR reads can never be mistaken for
    /// each other.
    ///
    /// Sizes are in bytes. A `chr` of 0 gives 8K of CHR-RAM instead of 8K of CHR-ROM.
    pub fn page_indexed_cart(prg_rom: usize, chr: usize) -> Cart {
        let mut cart = Cart::empty_sized(prg_rom, chr);
        let mut image: Box<PrgRom> = vec![0; PRG_ROM_SIZE]
            .into_boxed_slice()
            .try_into()
            .expect("PRG_ROM_SIZE bytes");
        for (page, data) in image.chunks_mut(PAGE_SIZE).enumerate() {
            data.fill((page % (prg_rom / PAGE_SIZE)) as u8);
        }
        cart.prg_rom = Arc::from(image);
        for (page, data) in cart
            .memory
            .region_mut(Src::Chr)
            .chunks_mut(PAGE_SIZE)
            .enumerate()
        {
            data.fill(0x80 | page as u8);
        }
        // Distinguishable from both, so "did PRG-RAM land at $6000" needs no separate setup.
        cart.memory.region_mut(Src::PrgRam).fill(0x5A);
        cart
    }

    /// Mirrors `Bus::write`: the data store happens first, then the board acts on the register.
    pub fn write(mapper: &mut Mapper, cart: &mut Cart, addr: u16, val: u8) {
        assert!(
            (0x4100..=0xFFFF).contains(&addr),
            "${addr:04X} does not reach the board on the CPU bus"
        );
        cart.memory.prg_write(addr, val);
        mapper.write_register(&mut cart.memory, addr, val);
    }

    /// What the CPU reads at `addr`.
    pub fn prg_peek(_mapper: &Mapper, cart: &Cart, addr: u16) -> u8 {
        if addr >= 0x8000 {
            cart.prg_rom[cart.memory.rom_index(addr)]
        } else {
            cart.memory.prg_peek(addr)
        }
    }

    /// What the PPU reads at `addr`.
    pub fn chr_peek(_mapper: &Mapper, cart: &Cart, addr: u16) -> u8 {
        cart.memory.chr_peek(addr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Src;

    #[test]
    fn an_unimplemented_mapper_number_is_an_error() {
        let mut cart = Cart::empty();
        cart.header.mapper_num = 4;
        let err = Mapper::from_cart(&mut cart).expect_err("MMC3 is not a zcore board");
        assert!(
            matches!(err, Error::Unimplemented(4)),
            "expected Unimplemented(4), got {err:?}"
        );
    }

    /// The page tables are part of the `Copy` state and travel with it through serde, so a
    /// decoded state reads through the same banks as the console it was taken from.
    #[test]
    fn page_tables_survive_a_state_round_trip() {
        let mut cart = test_utils::page_indexed_cart(128 * 1024, 0);
        let mut mapper = Sxrom::load(&mut cart).expect("valid mapper");
        // Select PRG bank 3 at $8000: five serial writes to $E000, clocked apart.
        for bit in 0..5 {
            test_utils::write(&mut mapper, &mut cart, 0xE000, (3 >> bit) & 1);
            mapper.clock();
            mapper.clock();
        }
        let peek = |memory: &Memory| cart.prg_rom[memory.rom_index(0x8000)];
        assert_eq!(peek(&cart.memory), 3 * 16, "bank switched");

        let bytes = postcard::to_allocvec(&cart.memory).expect("memory serializes");
        let restored: Memory = postcard::from_bytes(&bytes).expect("memory deserializes");
        assert_eq!(peek(&restored), 3 * 16, "the mapping came back");
        assert_eq!(
            restored.region_ref(Src::PrgRam),
            cart.memory.region_ref(Src::PrgRam)
        );
    }
}
