//! APU DMC (Delta Modulation Channel): the registers and sample fetcher the console can observe.
//!
//! See: <https://www.nesdev.org/wiki/APU_DMC>

// The APU's internal wiring, whose meaning is the hardware's rather than this crate's. Public for
// tools that digest machine state, not a stable surface - see the module docs on `apu`.
#![allow(missing_docs)]

use crate::common::ResetKind;
use serde::{Deserialize, Serialize};

/// The DMC as `$4015` reports it, plus FCEUmm's sample fetcher (FQ-12).
///
/// The output unit and the hardware sample reader went with audio synthesis (ladder stage Z4):
/// FQ-12 already dropped every fetch the hardware reader asked for, so it never loaded a byte, never
/// counted `bytes_remaining` down and never raised the DMC IRQ.
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
#[must_use]
pub struct Dmc {
    /// $4013 sample length in bytes.
    pub sample_length: u16,
    /// What `$4015` bit 4 reports: the sample length once enabled through `$4015`, zero once
    /// disabled. Nothing counts it down (see the type docs; FCEUmm reports [`FceuDmc::size`]).
    pub bytes_remaining: u16,
    /// FQ-12: FCEUmm's sample fetcher, which times the DMA cycles the CPU spends.
    pub fceu: FceuDmc,
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
            sample_length: 0x0001,
            bytes_remaining: 0x0000,
            fceu: FceuDmc::new(),
        }
    }

    /// $4013 DMC length
    pub const fn write_length(&mut self, val: u8) {
        self.sample_length = ((val as u16) << 4) | 1;
    }

    /// $4015 WRITE
    pub const fn set_enabled(&mut self, enabled: bool) {
        if !enabled {
            self.bytes_remaining = 0;
        } else if self.bytes_remaining == 0 {
            self.bytes_remaining = self.sample_length;
        }
    }

    pub const fn reset(&mut self, kind: ResetKind) {
        if let ResetKind::Hard = kind {
            self.sample_length = 1;
        }
        self.bytes_remaining = 0;
        // FQ-12: `FCEUSND_Reset` on every reset.
        self.fceu = FceuDmc::new();
    }
}

/// FQ-12: FCEUmm's DMC sample fetcher.
///
/// FCEUmm runs the DMC from a hook called once per instruction (`FCEU_SoundCPUHook`, libretro-fceumm
/// `src/sound.c:503-555`): when the sample buffer is empty it fetches the next byte right away (four
/// CPU reads, FQ-11), then counts the instruction's cycles down a timer that shifts a bit per period
/// and empties the buffer every eight bits. Only the fetch timing matters to the console (it steals
/// CPU cycles).
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct FceuDmc {
    pub acc: i32,
    pub period: i32,
    pub bit_count: u8,
    pub format: u8,
    pub address_latch: u8,
    pub size_latch: u8,
    /// Sample address, 15 bits (read at $8000 + address).
    pub address: u16,
    pub size: i32,
    pub have_dma: bool,
    pub have_sample: bool,
    /// Cumulative CPU cycles already counted down the timer.
    pub counted: u32,
}

impl Default for FceuDmc {
    fn default() -> Self {
        Self::new()
    }
}

impl FceuDmc {
    /// Power-on/reset state (`FCEUSND_Reset`, `src/sound.c:1170-1187`).
    pub const fn new() -> Self {
        Self {
            acc: 1,
            period: Dmc::PERIOD_TABLE_NTSC[0] as i32,
            bit_count: 0,
            format: 0,
            address_latch: 0,
            size_latch: 0,
            address: 0,
            size: 0,
            have_dma: false,
            have_sample: false,
            counted: 0,
        }
    }

    /// A DMC register write ($4010, $4012, $4013, $4015).
    pub fn write(&mut self, addr: u16, val: u8) {
        match addr {
            0x4010 => {
                self.format = val;
                self.period = i32::from(Dmc::PERIOD_TABLE_NTSC[usize::from(val & 0x0F)]);
            }
            0x4012 => self.address_latch = val,
            0x4013 => self.size_latch = val,
            0x4015 => {
                if val & 0x10 == 0 {
                    self.size = 0;
                } else if self.size == 0 {
                    self.prep();
                }
            }
            _ => (),
        }
    }

    fn prep(&mut self) {
        self.address = 0x4000 + (u16::from(self.address_latch) << 6);
        self.size = (i32::from(self.size_latch) << 4) + 1;
    }

    /// The per-instruction hook. `now` is the CPU cycle count FCEUmm has reached (the instruction's
    /// start plus its base cycles). Returns the address to fetch from when a sample byte is fetched
    /// now; the caller spends the four read cycles.
    pub fn hook(&mut self, now: u32) -> Option<u16> {
        let cycles = now.wrapping_sub(self.counted) as i32;
        self.counted = now;
        let fetch = if self.size != 0 && !self.have_dma {
            let addr = 0x8000 + self.address;
            self.have_dma = true;
            self.address = (self.address + 1) & 0x7FFF;
            self.size -= 1;
            if self.size == 0 && self.format & 0x40 != 0 {
                self.prep();
            }
            Some(addr)
        } else {
            None
        };
        self.acc -= cycles;
        while self.acc <= 0 {
            self.acc += self.period;
            self.bit_count = (self.bit_count + 1) & 7;
            if self.bit_count == 0 {
                self.have_sample = self.have_dma;
                self.have_dma = false;
            }
        }
        fetch
    }
}
