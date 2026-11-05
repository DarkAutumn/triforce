//! APU DMC (Delta Modulation Channel) implementation.
//!
//! See: <https://www.nesdev.org/wiki/APU_DMC>

// The APU's internal wiring: channel state, dividers and filter taps, whose meaning is the
// hardware's rather than this crate's. Public for embedders and debuggers, not a stable surface -
// see the module docs on `apu`.
#![allow(missing_docs)]

use crate::{apu::timer::Timer, common::ResetKind};
use serde::{Deserialize, Serialize};
use tracing::trace;

/// APU DMC (Delta Modulation Channel) provides sample playback.
///
/// See: <https://www.nesdev.org/wiki/APU_DMC>
#[derive(Debug, Clone, Serialize, Deserialize)]
#[must_use]
pub struct Dmc {
    pub timer: Timer,
    pub force_silent: bool,
    pub irq_enabled: bool,
    pub irq_pending: bool,
    pub dma_pending: bool,
    pub loops: bool,
    pub addr: u16,
    pub sample_addr: u16,
    pub bytes_remaining: u16,
    pub sample_length: u16,
    pub sample_buffer: u8,
    pub buffer_empty: bool,
    pub init: u8,
    pub output_level: u8,
    pub bits_remaining: u8,
    pub shift: u8,
    pub silence: bool,
    pub should_clock: bool,
    /// FQ-12: FCEUmm's sample fetcher, used instead of the hardware one with `fceumm_compat`.
    #[serde(default)]
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
            timer: Timer::preload(Self::period(0)),
            force_silent: false,
            irq_enabled: false,
            irq_pending: false,
            dma_pending: false,
            loops: false,
            addr: 0xC000,
            sample_addr: 0x0000,
            bytes_remaining: 0x0000,
            sample_length: 0x0001,
            sample_buffer: 0x00,
            buffer_empty: true,
            init: 0,
            output_level: 0x00,
            bits_remaining: 0x08,
            shift: 0x00,
            silence: true,
            should_clock: false,
            fceu: FceuDmc::new(),
        }
    }

    #[must_use]
    pub const fn silent(&self) -> bool {
        self.force_silent
    }

    pub const fn set_silent(&mut self, silent: bool) {
        self.force_silent = silent;
    }

    #[cold]
    #[must_use]
    pub fn irq_pending_in(&self, cycles_to_run: u32) -> bool {
        if self.irq_enabled && self.bytes_remaining > 0 {
            let cycles_to_empty = (u16::from(self.bits_remaining) + (self.bytes_remaining - 1) * 8)
                * self.timer.period;
            cycles_to_run >= u32::from(cycles_to_empty)
        } else {
            false
        }
    }

    #[must_use]
    pub const fn dma_addr(&self) -> u16 {
        self.addr
    }

    fn init_sample(&mut self) {
        self.addr = self.sample_addr;
        self.bytes_remaining = self.sample_length;
        trace!(
            "APU DMC sample started. bytes remaining: {}",
            self.bytes_remaining
        );
        self.should_clock = self.bytes_remaining > 0;
    }

    /// Load a sample into the DMC buffer - returns `true` if an IRQ is triggered.
    pub fn load_buffer(&mut self, val: u8) {
        if self.bytes_remaining > 0 {
            self.sample_buffer = val;
            self.buffer_empty = false;
            if self.addr == 0xFFFF {
                self.addr = 0x8000;
            } else {
                self.addr += 1;
            }
            self.bytes_remaining -= 1;
            trace!("APU DMC bytes remaining: {}", self.bytes_remaining);
            if self.bytes_remaining == 0 {
                self.should_clock = false;
                if self.loops {
                    self.init_sample();
                } else if self.irq_enabled {
                    self.irq_pending = true;
                }
            }
        }
    }

    const fn period(val: u8) -> u16 {
        Self::PERIOD_TABLE_NTSC[(val & 0x0F) as usize] - 1
    }

    /// $4010 DMC timer
    pub const fn write_timer(&mut self, val: u8) {
        self.irq_enabled = val & 0x80 == 0x80;
        self.loops = val & 0x40 == 0x40;
        self.timer.period = Self::period(val);
        if !self.irq_enabled {
            self.irq_pending = false;
        }
    }

    /// $4011 DMC output
    pub const fn write_output(&mut self, val: u8) {
        self.output_level = val & 0x7F;
    }

    /// $4012 DMC addr load
    pub fn write_addr(&mut self, val: u8) {
        self.sample_addr = 0xC000 | (u16::from(val) << 6);
    }

    /// $4013 DMC length
    pub fn write_length(&mut self, val: u8) {
        self.sample_length = (u16::from(val) << 4) | 1;
    }

    /// $4015 WRITE
    pub fn set_enabled(&mut self, enabled: bool, cycle: u32) {
        if !enabled {
            self.bytes_remaining = 0;
            self.should_clock = false;
        } else if self.bytes_remaining == 0 {
            self.init_sample();
            // Delay a number of cycles based on even/odd cycle
            self.init = if cycle & 0x01 == 0x00 { 2 } else { 3 };
        }
    }

    #[inline(always)]
    pub fn should_clock(&mut self) -> bool {
        if self.init > 0 {
            self.init -= 1;
            if self.init == 0 && self.buffer_empty && self.bytes_remaining > 0 {
                trace!("APU DMC DMA pending");
                self.dma_pending = true;
            }
        }
        self.should_clock
    }
    /// Raw channel level before conversion to a float sample.
    ///
    /// Callers that only need to index a mixer lookup table should use this rather than casting
    /// `output()` back to an integer, which costs a saturating float-to-int conversion.
    #[inline]
    pub const fn level(&self) -> u8 {
        if self.silent() { 0 } else { self.output_level }
    }
    pub fn output(&self) -> f32 {
        f32::from(self.level())
    }
    pub const fn cycle(&self) -> u32 {
        self.timer.cycle
    }
    /// The next cycle this channel's output could change on.
    pub const fn next_change(&self) -> u32 {
        self.timer.next_expiry()
    }
    //                          Timer
    //                            |
    //                            v
    // Reader ---> Buffer ---> Shifter ---> Output level ---> (to the mixer)
    pub fn clock(&mut self) {
        if self.timer.tick() {
            self.step_output();
        }
    }

    /// Clock forward to `cycle`, which for the DMC is a shift of the sample buffer per expiry, and
    /// is where a sample fetch or its IRQ becomes pending.
    pub fn clock_to(&mut self, cycle: u32) {
        while self.timer.run_to(cycle) {
            self.step_output();
        }
    }

    fn step_output(&mut self) {
        if !self.silence {
            // Update output level but clamp to 0..=127 range
            if self.shift & 0x01 == 0x01 {
                if self.output_level <= 125 {
                    self.output_level += 2;
                }
            } else if self.output_level >= 2 {
                self.output_level -= 2;
            }
            self.shift >>= 1;
        }

        if self.bits_remaining > 0 {
            self.bits_remaining -= 1;
        }
        trace!("APU DMC bits remaining: {}", self.bits_remaining);

        if self.bits_remaining == 0 {
            self.bits_remaining = 8;
            self.silence = self.buffer_empty;
            if !self.buffer_empty {
                self.shift = self.sample_buffer;
                self.buffer_empty = true;
                if self.bytes_remaining > 0 {
                    trace!("APU DMC DMA pending");
                    self.dma_pending = true;
                }
            }
        }
    }

    pub const fn reset(&mut self, kind: ResetKind) {
        self.timer.reset(kind);
        self.timer.period = Self::period(0);
        self.timer.reload();
        self.timer.cycle += 1; // FIXME: Startup timing is slightly wrong, DMA tests fail with the
        // default
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
        self.sample_buffer = 0x00;
        self.buffer_empty = true;
        self.output_level = 0x00;
        self.bits_remaining = 0x08;
        self.shift = 0x00;
        self.silence = true;
        self.should_clock = false;
        // FQ-12: `FCEUSND_Reset` on every reset.
        self.fceu = FceuDmc::new();
    }
}

/// FQ-12 (`fceumm_compat`): FCEUmm's DMC sample fetcher.
///
/// FCEUmm runs the DMC from a hook called once per instruction (`FCEU_SoundCPUHook`, libretro-fceumm
/// `src/sound.c:503-555`): when the sample buffer is empty it fetches the next byte right away (four
/// CPU reads, FQ-11), then counts the instruction's cycles down a timer that shifts a bit per period
/// and empties the buffer every eight bits. Only the fetch timing matters to the console (it steals
/// CPU cycles); the output level stays with [`Dmc`].
#[derive(Debug, Clone, Serialize, Deserialize)]
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
