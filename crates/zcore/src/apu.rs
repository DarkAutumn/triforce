//! NES APU (Audio Processing Unit), minus the audio: what the console can observe of it.
//!
//! See: <https://www.nesdev.org/wiki/APU>
//!
//! zcore produces no sound (ladder stage Z4). What stays is every APU effect the CPU can see: the
//! register writes' timing side effects, the four length counters `$4015` reports, the frame
//! counter and its IRQ, and the DMC's sample reader, whose fetches steal CPU cycles (DMA), and its
//! IRQ.
//!
//! # Stability
//!
//! [`Apu`]'s fields, and the types in this module's submodules, are the emulation's internal
//! wiring. They are public so that tools can digest machine state, but they track the
//! implementation, not a stable surface.

use crate::{
    apu::{
        dmc::Dmc,
        frame_counter::{FrameCounter, FrameType},
        length_counter::LengthCounter,
    },
    common::ResetKind,
};
use serde::{Deserialize, Serialize};

pub mod dmc;
pub mod frame_counter;
pub mod length_counter;

/// NES APU (Audio Processing Unit), without sound synthesis.
///
/// See: <https://wiki.nesdev.org/w/index.php/APU>
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
#[must_use]
pub struct Apu {
    /// Cycles into the current [`Apu::CYCLE_SIZE`] block.
    pub master_clock: u32,
    /// Total CPU cycles clocked.
    pub cpu_cycle: u32,
    /// APU cycles clocked.
    pub clock: u32,
    /// The frame sequencer that clocks the length counters.
    pub frame_counter: FrameCounter,
    /// The first square-wave channel's length counter.
    pub pulse1: LengthCounter,
    /// The second square-wave channel's length counter.
    pub pulse2: LengthCounter,
    /// The triangle channel's length counter.
    pub triangle: LengthCounter,
    /// The noise channel's length counter.
    pub noise: LengthCounter,
    /// The sample channel.
    pub dmc: Dmc,
    /// Whether a register write left a length counter update for the next cycle to apply.
    pub should_clock: bool,
}

impl Default for Apu {
    fn default() -> Self {
        Self::new()
    }
}

impl Apu {
    /// How many CPU cycles a block spans before every cycle counter rolls back to zero.
    pub const CYCLE_SIZE: u32 = 10_000;

    /// Create a new APU instance.
    pub const fn new() -> Self {
        Self {
            master_clock: 0,
            cpu_cycle: 0,
            clock: 0,
            frame_counter: FrameCounter::new(),
            pulse1: LengthCounter::new(),
            pulse2: LengthCounter::new(),
            triangle: LengthCounter::new(),
            noise: LengthCounter::new(),
            dmc: Dmc::new(),
            should_clock: false,
        }
    }

    /// Counts one CPU cycle. That is all a cycle costs until `master_clock` reaches
    /// [`Apu::next_clock`], when [`Apu::clock_due`] has work to do: the caller keeps that
    /// deadline, and anything that changes the APU from outside (a register access, a sync, a
    /// reset) has to bring it forward.
    #[inline(always)]
    pub const fn tick(&mut self) {
        self.cpu_cycle = self.cpu_cycle.wrapping_add(1);
        self.master_clock += 1;
    }

    /// The work of a cycle on which the frame counter, a length counter or the DMC may move.
    #[inline(always)]
    pub fn clock_due(&mut self) {
        if self.master_clock == Self::CYCLE_SIZE - 1 {
            self.clock_sync();
        } else if self.should_clock() {
            self.clock_to(self.master_clock);
        }
    }

    /// The `master_clock` from which [`Apu::clock_due`] has work to do: the next cycle while a
    /// register write or a frame counter block is pending, else the cycle before the frame
    /// counter's next step (`FrameCounter::should_clock`), the DMC's next fetch request
    /// ([`Dmc::next_event`]) or the end of the cycle block.
    #[must_use]
    pub fn next_clock(&self) -> u32 {
        let fc = &self.frame_counter;
        let next = self.master_clock + 1;
        if self.should_clock || fc.block_counter > 0 || fc.write_buffer.is_some() {
            return next;
        }
        let step = fc.step_cycles[fc.step] - 1;
        let due = (step.saturating_sub(fc.cycle) + self.clock).min(self.dmc.next_event());
        due.max(next).min(Self::CYCLE_SIZE - 1)
    }

    /// Runs the frame counter up to the master clock and starts the next [`Apu::CYCLE_SIZE`]
    /// block, putting the cycle counters back to zero.
    #[cold]
    #[inline(never)]
    pub fn clock_sync(&mut self) {
        self.clock_to(self.master_clock);
        self.master_clock = 0;
        self.clock = 0;
        self.dmc.cycle = 0;
    }

    #[inline(always)]
    fn should_clock(&mut self) -> bool {
        if self.dmc.init > 0 {
            self.dmc.clock_init();
        }
        // A length counter write is applied on the next cycle.
        if self.should_clock {
            self.should_clock = false;
            return true;
        }
        let cycles = self.master_clock - self.clock;
        self.frame_counter.should_clock(cycles) || self.master_clock >= self.dmc.next_event()
    }

    fn clock_to(&mut self, cycle: u32) {
        self.master_clock = cycle;
        while self.master_clock - self.clock > 0 {
            self.clock += self
                .frame_counter
                .clock_with(self.master_clock - self.clock, |ty| {
                    if ty == FrameType::Half {
                        self.pulse1.clock();
                        self.pulse2.clock();
                        self.triangle.clock();
                        self.noise.clock();
                    }
                });

            self.pulse1.reload();
            self.pulse2.reload();
            self.triangle.reload();
            self.noise.reload();
        }
        self.dmc.clock_to(self.clock);
    }

    /// A write to `$4000-$4013`, `$4015` or `$4017`. Every mapped register first brings the
    /// frame counter up to date; `$4009` and `$400D` are unmapped and do nothing.
    pub fn write(&mut self, addr: u16, val: u8) {
        if matches!(addr, 0x4009 | 0x400D) {
            return;
        }
        self.clock_to(self.master_clock);
        match addr {
            // Pulse/noise control: the halt flag is D5.
            0x4000 => self.write_halt(Channel::Pulse1, val & 0x20 != 0),
            0x4004 => self.write_halt(Channel::Pulse2, val & 0x20 != 0),
            0x400C => self.write_halt(Channel::Noise, val & 0x20 != 0),
            // Triangle linear counter control: the halt flag is D7.
            0x4008 => self.write_halt(Channel::Triangle, val & 0x80 != 0),
            // Length loads.
            0x4003 => self.write_length(Channel::Pulse1, val),
            0x4007 => self.write_length(Channel::Pulse2, val),
            0x400B => self.write_length(Channel::Triangle, val),
            0x400F => self.write_length(Channel::Noise, val),
            0x4010 => self.dmc.write_timer(val),
            0x4012 => self.dmc.write_addr(val),
            0x4013 => self.dmc.write_length(val),
            0x4015 => {
                self.pulse1.set_enabled(val & 0x01 == 0x01);
                self.pulse2.set_enabled(val & 0x02 == 0x02);
                self.triangle.set_enabled(val & 0x04 == 0x04);
                self.noise.set_enabled(val & 0x08 == 0x08);
                self.dmc.set_enabled(val & 0x10 == 0x10, self.cpu_cycle);
                self.dmc.irq_pending = false;
            }
            0x4017 => self.frame_counter.write(val, self.cpu_cycle),
            // Sweep, timer periods, envelope volume, DMC output: sound only.
            _ => (),
        }
    }

    const fn length(&mut self, channel: Channel) -> &mut LengthCounter {
        match channel {
            Channel::Pulse1 => &mut self.pulse1,
            Channel::Pulse2 => &mut self.pulse2,
            Channel::Triangle => &mut self.triangle,
            Channel::Noise => &mut self.noise,
        }
    }

    const fn write_halt(&mut self, channel: Channel, halt: bool) {
        self.length(channel).write_ctrl(halt);
        self.should_clock = true;
    }

    const fn write_length(&mut self, channel: Channel, val: u8) {
        let length = self.length(channel);
        length.write(val >> 3);
        self.should_clock = length.enabled;
    }

    /// Read APU Status.
    ///
    /// $4015   if-d nt21   DMC IRQ, frame IRQ, length counter statuses
    pub fn read_status(&mut self) -> u8 {
        self.clock_to(self.master_clock);
        let val = self.peek_status();
        self.frame_counter.irq_pending = false;
        val
    }

    /// Read APU Status without side-effects.
    ///
    /// Non-mutating version of `read_status`.
    pub const fn peek_status(&self) -> u8 {
        let mut status = 0x00;
        if self.pulse1.counter > 0 {
            status |= 0x01;
        }
        if self.pulse2.counter > 0 {
            status |= 0x02;
        }
        if self.triangle.counter > 0 {
            status |= 0x04;
        }
        if self.noise.counter > 0 {
            status |= 0x08;
        }
        if self.dmc.bytes_remaining > 0 {
            status |= 0x10;
        }
        if self.frame_counter.irq_pending {
            status |= 0x40;
        }
        if self.dmc.irq_pending {
            status |= 0x80;
        }
        status
    }

    /// Whether the frame counter or the DMC is asserting IRQ.
    #[inline(always)]
    pub const fn irq_pending(&self) -> bool {
        self.frame_counter.irq_pending | self.dmc.irq_pending
    }

    /// Resets the APU.
    pub fn reset(&mut self, kind: ResetKind) {
        self.cpu_cycle = 0;
        self.master_clock = 0;
        self.clock = 0;
        self.should_clock = false;
        self.frame_counter.reset(kind);
        self.pulse1.reset(true);
        self.pulse2.reset(true);
        // The triangle's length counter survives a soft reset.
        self.triangle.reset(kind == ResetKind::Hard);
        self.noise.reset(true);
        self.dmc.reset(kind);
    }
}

/// The four channels with a length counter.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum Channel {
    Pulse1,
    Pulse2,
    Triangle,
    Noise,
}
