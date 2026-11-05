//! `SxROM`/`MMC1` (Mapper 001).
//!
//! <https://wiki.nesdev.org/w/index.php/MMC1>

// Board register state, whose meaning is the mapper hardware's rather than this crate's. See the
// module docs on `mapper` for what a board is.
#![allow(missing_docs)]

use crate::{
    cart::Cart,
    common::ResetKind,
    mapper::{self, Map, Mapper, Mmc1},
    memory::{Memory, Src},
    ppu::Mirroring,
};
use serde::{Deserialize, Serialize};

/// `SxROM`/`MMC1` (Mapper 001), MMC1B/C, up to 256K of PRG-ROM.
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
#[must_use]
pub struct Sxrom {
    pub mmc1: Mmc1,
}

impl Sxrom {
    const PRG_RAM_WINDOW: usize = 8 * 1024;
    const PRG_WINDOW: usize = 16 * 1024;
    const CHR_WINDOW: usize = 4 * 1024;
    const PRG_BANK_MASK: u8 = 0x0F;

    // PPU $0000..=$0FFF 4K CHR Bank Switchable, or 8K across both in 8K mode
    // PPU $1000..=$1FFF 4K CHR Bank Switchable
    // CPU $6000..=$7FFF 8K PRG-RAM, write-protectable
    // CPU $8000..=$BFFF 16K PRG-ROM, switchable or fixed to first depending on mode
    // CPU $C000..=$FFFF 16K PRG-ROM, fixed to last or switchable depending on mode
    pub fn load(cart: &mut Cart) -> Result<Mapper, mapper::Error> {
        let mut sxrom = Self { mmc1: Mmc1::new() };
        sxrom.update_banks(&mut cart.memory);
        Ok(sxrom.into())
    }

    /// Clock the board once per CPU cycle: the busy-cycle counter that ignores rapid consecutive
    /// writes. MMC1 has no IRQ.
    #[inline(always)]
    pub const fn clock(&mut self) {
        self.mmc1.clock();
    }
}

impl Map for Sxrom {
    fn mirroring(&self) -> Mirroring {
        self.mmc1.mirroring
    }

    fn write_register(&mut self, memory: &mut Memory, addr: u16, val: u8) {
        // $6000..=$7FFF is PRG-RAM, already stored by the caller when the window is writable.
        if addr >= 0x8000 && self.mmc1.process_shift_register_write(addr, val) {
            self.update_banks(memory);
        }
    }

    fn update_banks(&mut self, memory: &mut Memory) {
        let mmc1 = &self.mmc1;
        if mmc1.prg_mode {
            if mmc1.prg_bank_select {
                // $8000 switchable, $C000 fixed to the last bank.
                memory.map_prg(0x8000, Self::PRG_WINDOW, i32::from(mmc1.prg), Src::PrgRom);
                memory.map_prg(
                    0xC000,
                    Self::PRG_WINDOW,
                    i32::from(Self::PRG_BANK_MASK),
                    Src::PrgRom,
                );
            } else {
                // $8000 fixed to the first bank, $C000 switchable.
                memory.map_prg(0x8000, Self::PRG_WINDOW, 0, Src::PrgRom);
                memory.map_prg(0xC000, Self::PRG_WINDOW, i32::from(mmc1.prg), Src::PrgRom);
            }
        } else {
            // 32K mode ignores the low bank bit.
            let bank = i32::from(mmc1.prg & 0xFE);
            memory.map_prg(0x8000, Self::PRG_WINDOW, bank, Src::PrgRom);
            memory.map_prg(0xC000, Self::PRG_WINDOW, bank + 1, Src::PrgRom);
        }

        if mmc1.chr_mode {
            memory.map_chr(0x0000, Self::CHR_WINDOW, i32::from(mmc1.chr0), Src::Chr);
            memory.map_chr(0x1000, Self::CHR_WINDOW, i32::from(mmc1.chr1), Src::Chr);
        } else {
            // 8K mode ignores the low bank bit.
            let bank = i32::from(mmc1.chr0 & 0x1E);
            memory.map_chr(0x0000, Self::CHR_WINDOW, bank, Src::Chr);
            memory.map_chr(0x1000, Self::CHR_WINDOW, bank + 1, Src::Chr);
        }

        // Disabled PRG-RAM reads as open bus and ignores writes, which unmapping gives for free.
        if mmc1.prg_ram_enabled() {
            memory.map_prg(0x6000, Self::PRG_RAM_WINDOW, 0, Src::PrgRam);
        } else {
            memory.unmap_prg(0x6000, Self::PRG_RAM_WINDOW);
        }

        memory.set_mirroring(mmc1.mirroring);
    }

    fn reset(&mut self, kind: ResetKind) {
        self.mmc1.reset(kind);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mapper::test_utils::{chr_peek, page_indexed_cart, prg_peek, write};

    /// 128K PRG-ROM (8 16K banks), 8K PRG-RAM, 8K CHR-ROM (2 4K banks).
    fn mmc1() -> (Mapper, Cart) {
        let mut cart = page_indexed_cart(128 * 1024, 8 * 1024);
        let mapper = Sxrom::load(&mut cart).expect("valid mapper");
        (mapper, cart)
    }

    /// Feeds a register through the serial port: five writes of one bit each, LSB first.
    ///
    /// MMC1 ignores a write that lands within two CPU cycles of the previous one - it is how the
    /// real chip rejects the second half of a read-modify-write - so the clocks between bits are
    /// load-bearing, not padding.
    fn serial_write(mapper: &mut Mapper, cart: &mut Cart, addr: u16, val: u8) {
        for bit in 0..5 {
            write(mapper, cart, addr, (val >> bit) & 0x01);
            mapper.clock();
            mapper.clock();
        }
    }

    /// Sets the control register: mirroring in bits 0-1, PRG slot select in 2, PRG mode in 3, CHR
    /// mode in 4.
    fn control(mapper: &mut Mapper, cart: &mut Cart, val: u8) {
        serial_write(mapper, cart, 0x8000, val);
    }

    /// MMC1 powers on in 16K mode with $C000 fixed to the last bank, which is what puts a reset
    /// vector under the CPU before the game has written a single register.
    #[test]
    fn powers_on_in_16k_mode_with_the_last_bank_at_c000() {
        let (mapper, cart) = mmc1();
        assert_eq!(prg_peek(&mapper, &cart, 0x8000), 0, "bank 0 at $8000");
        assert_eq!(prg_peek(&mapper, &cart, 0xC000), 7 * 16, "last bank");
        assert_eq!(prg_peek(&mapper, &cart, 0xFFFF), 127);
    }

    /// A write within two cycles of the previous one is dropped, so a four-bit-then-ignored
    /// sequence must not latch a register.
    #[test]
    fn consecutive_writes_within_two_cycles_are_ignored() {
        let (mut mapper, mut cart) = mmc1();

        // Five bits with no clocks between them: only the first is accepted, so the shift register
        // never reaches its fifth bit and nothing is latched.
        for _ in 0..5 {
            write(&mut mapper, &mut cart, 0xE000, 0x01);
        }
        assert_eq!(
            prg_peek(&mapper, &cart, 0x8000),
            0,
            "no register was latched"
        );

        // The same value fed properly does latch.
        serial_write(&mut mapper, &mut cart, 0xE000, 3);
        assert_eq!(prg_peek(&mapper, &cart, 0x8000), 3 * 16);
    }

    /// The three PRG modes: 32K, 16K with $8000 fixed, and 16K with $C000 fixed.
    #[test]
    fn prg_modes_select_32k_or_either_fixed_16k_half() {
        let (mut mapper, mut cart) = mmc1();

        // Mode 3 (bits 3 and 2 set): $8000 switchable, $C000 fixed last.
        control(&mut mapper, &mut cart, 0b01100);
        serial_write(&mut mapper, &mut cart, 0xE000, 3);
        assert_eq!(prg_peek(&mapper, &cart, 0x8000), 3 * 16, "switchable");
        assert_eq!(prg_peek(&mapper, &cart, 0xC000), 7 * 16, "fixed last");

        // Mode 2 (bit 3 set, bit 2 clear): $8000 fixed first, $C000 switchable.
        control(&mut mapper, &mut cart, 0b01000);
        serial_write(&mut mapper, &mut cart, 0xE000, 3);
        assert_eq!(prg_peek(&mapper, &cart, 0x8000), 0, "fixed first");
        assert_eq!(prg_peek(&mapper, &cart, 0xC000), 3 * 16, "switchable");

        // Mode 0/1 (bit 3 clear): one 32K bank, ignoring the low bit of the bank number.
        control(&mut mapper, &mut cart, 0b00000);
        serial_write(&mut mapper, &mut cart, 0xE000, 4);
        assert_eq!(prg_peek(&mapper, &cart, 0x8000), 4 * 16);
        assert_eq!(
            prg_peek(&mapper, &cart, 0xC000),
            5 * 16,
            "the 32K is one bank"
        );
        serial_write(&mut mapper, &mut cart, 0xE000, 5);
        assert_eq!(
            prg_peek(&mapper, &cart, 0x8000),
            4 * 16,
            "the low bank bit is ignored in 32K mode"
        );
    }

    /// CHR is either one 8K bank or two independent 4K banks.
    #[test]
    fn chr_modes_select_one_8k_bank_or_two_4k_banks() {
        let (mut mapper, mut cart) = mmc1();

        // Bit 4 clear: 8K mode, low bank bit ignored.
        control(&mut mapper, &mut cart, 0b00000);
        serial_write(&mut mapper, &mut cart, 0xA000, 1);
        assert_eq!(chr_peek(&mapper, &cart, 0x0000), 0x80, "low bit ignored");
        assert_eq!(chr_peek(&mapper, &cart, 0x1000), 0x80 | 4, "same 8K");

        // Bit 4 set: two independent 4K banks.
        control(&mut mapper, &mut cart, 0b10000);
        serial_write(&mut mapper, &mut cart, 0xA000, 1);
        serial_write(&mut mapper, &mut cart, 0xC000, 0);
        assert_eq!(chr_peek(&mapper, &cart, 0x0000), 0x80 | 4);
        assert_eq!(chr_peek(&mapper, &cart, 0x1000), 0x80);
        // Bank numbers wrap within the 8K.
        serial_write(&mut mapper, &mut cart, 0xC000, 3);
        assert_eq!(chr_peek(&mapper, &cart, 0x1000), 0x80 | 4);
    }

    /// Bit 7 of any load-register write resets the shift register and forces the control register's
    /// PRG bits back to mode 3 - which is how a game recovers a known state at boot.
    #[test]
    fn bit_7_resets_the_shift_register_and_restores_mode_3() {
        let (mut mapper, mut cart) = mmc1();

        control(&mut mapper, &mut cart, 0b00000); // 32K mode
        serial_write(&mut mapper, &mut cart, 0xE000, 4);
        assert_eq!(prg_peek(&mapper, &cart, 0xC000), 5 * 16, "32K mode");

        write(&mut mapper, &mut cart, 0x8000, 0x80);
        assert_eq!(
            prg_peek(&mapper, &cart, 0xC000),
            7 * 16,
            "back to mode 3, last bank fixed at $C000"
        );
        // The reset write arms the two-cycle lockout like any other, so the next bit needs to wait.
        mapper.clock();
        mapper.clock();

        // The shift register is also empty, so a fresh five-bit sequence latches cleanly.
        serial_write(&mut mapper, &mut cart, 0xE000, 2);
        assert_eq!(prg_peek(&mapper, &cart, 0x8000), 2 * 16);
    }

    /// Mirroring comes from the low two bits of the control register, and MMC1 is the one board
    /// whose one-screen modes come before the usual vertical/horizontal pair.
    #[test]
    fn control_bits_0_and_1_set_mirroring() {
        let (mut mapper, mut cart) = mmc1();

        control(&mut mapper, &mut cart, 0b00010); // vertical
        cart.memory.chr_write(0x2000, 0x11);
        assert_eq!(chr_peek(&mapper, &cart, 0x2800), 0x11, "vertical");

        control(&mut mapper, &mut cart, 0b00011); // horizontal
        cart.memory.chr_write(0x2000, 0x22);
        assert_eq!(chr_peek(&mapper, &cart, 0x2400), 0x22, "horizontal");

        control(&mut mapper, &mut cart, 0b00000); // one-screen, lower
        cart.memory.chr_write(0x2000, 0x33);
        for nt in [0x2400, 0x2800, 0x2C00] {
            assert_eq!(chr_peek(&mapper, &cart, nt), 0x33, "one-screen A");
        }
    }

    /// Bit 4 of the PRG register disables PRG-RAM, which must read as open bus rather than as
    /// stale contents.
    #[test]
    fn prg_ram_disable_is_honoured() {
        let (mut mapper, mut cart) = mmc1();
        assert_eq!(
            prg_peek(&mapper, &cart, 0x6000),
            0x5A,
            "enabled at power-on"
        );

        serial_write(&mut mapper, &mut cart, 0xE000, 0x10);
        assert_eq!(prg_peek(&mapper, &cart, 0x6000), 0, "disabled reads as 0");

        serial_write(&mut mapper, &mut cart, 0xE000, 0x00);
        assert_eq!(prg_peek(&mapper, &cart, 0x6000), 0x5A, "re-enabled");
    }

    /// `update_banks` must rebuild every window from the registers alone, which is what
    /// [`State::rebuild_mapper_state`](crate::bus::State::rebuild_mapper_state) relies on after a
    /// reset.
    #[test]
    fn update_banks_rebuilds_every_window_from_register_state() {
        let (mut mapper, mut cart) = mmc1();
        control(&mut mapper, &mut cart, 0b11000);
        serial_write(&mut mapper, &mut cart, 0xA000, 1);
        serial_write(&mut mapper, &mut cart, 0xC000, 0);
        serial_write(&mut mapper, &mut cart, 0xE000, 5);

        let sample = |mapper: &Mapper, cart: &Cart| -> Vec<u8> {
            [0x6000, 0x8000, 0xC000]
                .into_iter()
                .map(|addr| prg_peek(mapper, cart, addr))
                .chain(
                    [0x0000, 0x1000]
                        .into_iter()
                        .map(|addr| chr_peek(mapper, cart, addr)),
                )
                .collect()
        };
        let before = sample(&mapper, &cart);

        cart.memory.unmap_prg(0x6000, 0x2000);
        cart.memory.map_prg(0x8000, 0x8000, 3, Src::PrgRom);
        cart.memory.unmap_chr(0x0000, 0x4000);
        mapper.update_banks(&mut cart.memory);

        assert_eq!(before, sample(&mapper, &cart));
    }

    /// The lockout lasts exactly two CPU cycles: one clock after a write is still too soon, two
    /// is enough.
    #[test]
    fn the_write_lockout_lasts_exactly_two_cycles() {
        let (mut mapper, mut cart) = mmc1();
        // With one clock after each write every other write is dropped: writes 0, 2, 4, 6, 8 give
        // bits 1, 0, 0, 0, 0 (bank 1). Were all accepted, writes 0-4 would latch 3 and 5-9 latch 0.
        for val in [1, 1, 0, 0, 0, 0, 0, 0, 0, 0] {
            write(&mut mapper, &mut cart, 0xE000, val);
            mapper.clock();
        }
        assert_eq!(prg_peek(&mapper, &cart, 0x8000), 16, "bank 1");
    }

    /// A bit-7 write mid-sequence discards the bits shifted so far, so the next five writes latch
    /// a clean value.
    #[test]
    fn bit_7_discards_a_partial_sequence() {
        let (mut mapper, mut cart) = mmc1();
        for _ in 0..3 {
            write(&mut mapper, &mut cart, 0xE000, 1);
            mapper.clock();
            mapper.clock();
        }
        write(&mut mapper, &mut cart, 0xE000, 0x80);
        mapper.clock();
        mapper.clock();
        serial_write(&mut mapper, &mut cart, 0xE000, 2);
        assert_eq!(
            prg_peek(&mapper, &cart, 0x8000),
            2 * 16,
            "not 0b1_0111 shifted in"
        );
    }

    /// Only the fifth write's address selects the register; the first four may go anywhere in
    /// $8000-$FFFF.
    #[test]
    fn the_fifth_writes_address_picks_the_register() {
        let (mut mapper, mut cart) = mmc1();
        for (bit, addr) in [0x8000u16, 0xA000, 0xC000, 0x9FFF, 0xFFFF]
            .into_iter()
            .enumerate()
        {
            write(&mut mapper, &mut cart, addr, (6 >> bit) & 1);
            mapper.clock();
            mapper.clock();
        }
        assert_eq!(
            prg_peek(&mapper, &cart, 0x8000),
            6 * 16,
            "latched into PRG ($E000-$FFFF)"
        );
        assert_eq!(
            chr_peek(&mapper, &cart, 0x0000),
            0x80,
            "CHR bank 0 untouched"
        );

        serial_write(&mut mapper, &mut cart, 0xBFFF, 0b10001); // CHR bank 0 = 17 (wraps to 1)
        serial_write(&mut mapper, &mut cart, 0x9FFF, 0b10000); // control: 4K CHR, 32K PRG
        assert_eq!(
            chr_peek(&mapper, &cart, 0x0000),
            0x80 | 4,
            "$BFFF is CHR bank 0"
        );
        assert_eq!(prg_peek(&mapper, &cart, 0x8000), 6 * 16, "32K mode, bank 6");
        assert_eq!(prg_peek(&mapper, &cart, 0xC000), 7 * 16);
    }

    /// Control mirroring value 1 is one-screen from the upper CIRAM bank: all four nametables
    /// read the same bank, distinct from the lower one.
    #[test]
    fn control_mirroring_1_is_one_screen_upper() {
        let (mut mapper, mut cart) = mmc1();
        control(&mut mapper, &mut cart, 0b00000);
        cart.memory.chr_write(0x2000, 0x33);
        control(&mut mapper, &mut cart, 0b00001);
        cart.memory.chr_write(0x2C00, 0x44);
        for nt in [0x2000, 0x2400, 0x2800, 0x2C00] {
            assert_eq!(
                chr_peek(&mapper, &cart, nt),
                0x44,
                "one-screen B at ${nt:04X}"
            );
        }
        control(&mut mapper, &mut cart, 0b00000);
        assert_eq!(
            chr_peek(&mapper, &cart, 0x2400),
            0x33,
            "the lower bank kept its byte"
        );
    }

    /// Both resets return to PRG mode 3 and empty the shift register; only a hard reset (power
    /// cycle) also re-enables PRG-RAM and clears the write lockout.
    #[test]
    fn soft_reset_keeps_prg_ram_disabled_hard_reset_does_not() {
        let (mut mapper, mut cart) = mmc1();
        control(&mut mapper, &mut cart, 0b00000);
        serial_write(&mut mapper, &mut cart, 0xE000, 0x10);
        write(&mut mapper, &mut cart, 0xE000, 1); // a pending bit and an armed lockout
        let Mapper::Sxrom(sxrom) = &mut mapper else {
            panic!("an SxROM board")
        };

        sxrom.mmc1.reset(ResetKind::Soft);
        assert_eq!((sxrom.mmc1.shift_count, sxrom.mmc1.write_buffer), (0, 0));
        assert!(sxrom.mmc1.prg_mode && sxrom.mmc1.prg_bank_select, "mode 3");
        assert!(
            !sxrom.mmc1.prg_ram_enabled(),
            "soft reset keeps PRG-RAM off"
        );
        assert_ne!(
            sxrom.mmc1.write_just_occurred, 0,
            "soft reset keeps the lockout"
        );

        sxrom.mmc1.reset(ResetKind::Hard);
        assert!(
            sxrom.mmc1.prg_ram_enabled(),
            "power cycle re-enables PRG-RAM"
        );
        assert_eq!(sxrom.mmc1.write_just_occurred, 0);
        mapper.update_banks(&mut cart.memory);
        assert_eq!(prg_peek(&mapper, &cart, 0x6000), 0x5A);
        assert_eq!(prg_peek(&mapper, &cart, 0xC000), 7 * 16, "last bank fixed");
    }
}
