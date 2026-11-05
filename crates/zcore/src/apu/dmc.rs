//! APU DMC (Delta Modulation Channel): the sample reader the console can observe.
//!
//! See: <https://www.nesdev.org/wiki/APU_DMC>

// The APU's internal wiring, whose meaning is the hardware's rather than this crate's. Public for
// tools that digest machine state, not a stable surface - see the module docs on `apu`.
#![allow(missing_docs)]

use crate::common::ResetKind;
use serde::{Deserialize, Serialize};

/// The DMC's timer and sample reader: when it asks the CPU for a sample byte (DMA), the bytes
/// left (`$4015` bit 4) and its IRQ.
///
/// The output unit went with audio synthesis (ladder stage Z4): what the reader fetches is not
/// kept, only when it fetches. The timer runs whether or not a sample plays; each expiry shifts
/// out one bit, and every eighth starts a new output cycle, which empties the sample buffer into
/// the shifter and asks for the next byte (nesdev "APU DMC", memory reader).
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
#[must_use]
pub struct Dmc {
    /// APU cycle (within the APU's cycle block) the timer has been run to.
    pub cycle: u32,
    /// Timer counter: cycles left before the next expiry, less one.
    pub counter: u16,
    /// Timer period: the `$4010` rate's CPU cycles, less one.
    pub period: u16,
    /// $4010 bit 7.
    pub irq_enabled: bool,
    /// DMC IRQ asserted (`$4015` bit 7).
    pub irq_pending: bool,
    /// A sample byte fetch the CPU has yet to pick up.
    pub dma_pending: bool,
    /// $4010 bit 6: restart the sample when it ends.
    pub loops: bool,
    /// Address of the next sample byte.
    pub addr: u16,
    /// $4012 sample start address.
    pub sample_addr: u16,
    /// Sample bytes still to fetch (`$4015` bit 4).
    pub bytes_remaining: u16,
    /// $4013 sample length in bytes.
    pub sample_length: u16,
    /// Whether the sample buffer is empty.
    pub buffer_empty: bool,
    /// Cycles until the fetch a `$4015` enable starts (0: none pending).
    pub init: u8,
    /// Bits left in the current output cycle.
    pub bits_remaining: u8,
}

impl Default for Dmc {
    fn default() -> Self {
        Self::new()
    }
}

impl Dmc {
    const PERIOD_TABLE_NTSC: [u16; 16] = [
        428, 380, 340, 320, 286, 254, 226, 214, 190, 160, 142, 128, 106, 84, 72, 54,
    ];

    pub const fn new() -> Self {
        Self {
            cycle: 0,
            counter: Self::period(0),
            period: Self::period(0),
            irq_enabled: false,
            irq_pending: false,
            dma_pending: false,
            loops: false,
            addr: 0xC000,
            sample_addr: 0x0000,
            bytes_remaining: 0x0000,
            sample_length: 0x0001,
            buffer_empty: true,
            init: 0,
            bits_remaining: 0x08,
        }
    }

    /// The timer period for a `$4010` rate index.
    pub const fn period(val: u8) -> u16 {
        Self::PERIOD_TABLE_NTSC[(val & 0x0F) as usize] - 1
    }

    /// Where the next sample byte is fetched from.
    #[must_use]
    pub const fn dma_addr(&self) -> u16 {
        self.addr
    }

    const fn init_sample(&mut self) {
        self.addr = self.sample_addr;
        self.bytes_remaining = self.sample_length;
    }

    /// A fetched sample byte reaches the buffer.
    pub const fn load_buffer(&mut self) {
        if self.bytes_remaining > 0 {
            self.buffer_empty = false;
            self.addr = if self.addr == 0xFFFF {
                0x8000
            } else {
                self.addr + 1
            };
            self.bytes_remaining -= 1;
            if self.bytes_remaining == 0 {
                if self.loops {
                    self.init_sample();
                } else if self.irq_enabled {
                    self.irq_pending = true;
                }
            }
        }
    }

    /// $4010 flags and rate.
    pub const fn write_timer(&mut self, val: u8) {
        self.irq_enabled = val & 0x80 == 0x80;
        self.loops = val & 0x40 == 0x40;
        self.period = Self::period(val);
        if !self.irq_enabled {
            self.irq_pending = false;
        }
    }

    /// $4012 sample address.
    pub const fn write_addr(&mut self, val: u8) {
        self.sample_addr = 0xC000 | ((val as u16) << 6);
    }

    /// $4013 sample length.
    pub const fn write_length(&mut self, val: u8) {
        self.sample_length = ((val as u16) << 4) | 1;
    }

    /// $4015 bit 4. `cycle` is the CPU cycle of the write: an enable that has to fetch does so
    /// 2 or 3 cycles later, by its parity.
    pub const fn set_enabled(&mut self, enabled: bool, cycle: u32) {
        if !enabled {
            self.bytes_remaining = 0;
        } else if self.bytes_remaining == 0 {
            self.init_sample();
            self.init = if cycle & 0x01 == 0x00 { 2 } else { 3 };
        }
    }

    /// One cycle of an enable's fetch countdown (`init` above zero).
    pub const fn clock_init(&mut self) {
        self.init -= 1;
        if self.init == 0 && self.buffer_empty && self.bytes_remaining > 0 {
            self.dma_pending = true;
        }
    }

    /// The APU cycle from which the DMC has something to do: every cycle of an enable's
    /// countdown, else the expiry ending the output cycle when it will ask for a byte (a full
    /// buffer to empty and bytes left), else never (`u32::MAX`). Expiries in between only shift.
    #[must_use]
    pub const fn next_event(&self) -> u32 {
        if self.init > 0 {
            return 0;
        }
        if self.buffer_empty || self.bytes_remaining == 0 {
            return u32::MAX;
        }
        let per = self.period as u32 + 1;
        self.cycle + self.counter as u32 + 1 + (self.bits_remaining as u32 - 1) * per
    }

    /// Runs the timer to APU cycle `target`, shifting a bit per expiry.
    pub const fn clock_to(&mut self, target: u32) {
        // The timer can sit ahead of the cycle asked for (`Dmc::reset` parks it a cycle forward);
        // running to a cycle already passed does nothing.
        while target > self.cycle {
            let remaining = target - self.cycle;
            if remaining <= self.counter as u32 {
                self.counter -= remaining as u16;
                self.cycle = target;
                return;
            }
            // `counter + 1` cycles to the expiry: the counter counts down on every cycle but the
            // one it reads zero on, which reloads it.
            self.cycle += self.counter as u32 + 1;
            self.counter = self.period;
            self.step_output();
        }
    }

    const fn step_output(&mut self) {
        if self.bits_remaining > 0 {
            self.bits_remaining -= 1;
        }
        if self.bits_remaining == 0 {
            self.bits_remaining = 8;
            if !self.buffer_empty {
                self.buffer_empty = true;
                if self.bytes_remaining > 0 {
                    self.dma_pending = true;
                }
            }
        }
    }

    pub const fn reset(&mut self, kind: ResetKind) {
        self.period = Self::period(0);
        self.counter = self.period;
        // Parked a cycle ahead: TetaNES' start-up alignment, which its DMA test ROMs need.
        self.cycle = 1;
        if let ResetKind::Hard = kind {
            self.sample_addr = 0xC000;
            self.sample_length = 1;
        }
        self.irq_enabled = false;
        self.irq_pending = false;
        self.dma_pending = false;
        self.loops = false;
        self.addr = 0x0000;
        self.bytes_remaining = 0;
        self.buffer_empty = true;
        self.init = 0;
        self.bits_remaining = 0x08;
    }
}
