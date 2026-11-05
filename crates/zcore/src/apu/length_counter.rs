//! APU Length Counter implementation.
//!
//! See: <https://www.nesdev.org/wiki/APU_Length_Counter>

// The APU's internal wiring, whose meaning is the hardware's rather than this crate's. Public for
// tools that digest machine state, not a stable surface - see the module docs on `apu`.
#![allow(missing_docs)]

use serde::{Deserialize, Serialize};

/// APU Length Counter: how long a channel keeps playing, and what `$4015` reports of it.
///
/// See: <https://www.nesdev.org/wiki/APU_Length_Counter>
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
#[must_use]
pub struct LengthCounter {
    pub enabled: bool,
    pub halt: bool,
    pub new_halt: bool,
    pub counter: u8, // Entry into LENGTH_TABLE
    pub previous_counter: u8,
    pub reload: u8,
}

impl Default for LengthCounter {
    fn default() -> Self {
        Self::new()
    }
}

impl LengthCounter {
    const LENGTH_TABLE: [u8; 32] = [
        10, 254, 20, 2, 40, 4, 80, 6, 160, 8, 60, 10, 14, 12, 26, 14, 12, 16, 24, 18, 48, 20, 96,
        22, 192, 24, 72, 26, 16, 28, 32, 30,
    ];

    pub const fn new() -> Self {
        Self {
            enabled: false,
            halt: false,
            new_halt: false,
            counter: 0,
            previous_counter: 0,
            reload: 0,
        }
    }

    /// Length load (`$4003`/`$4007`/`$400B`/`$400F` bits 3-7, already shifted down).
    #[inline]
    pub const fn write(&mut self, val: u8) {
        if self.enabled {
            self.reload = Self::LENGTH_TABLE[val as usize]; // D7..D3
            self.previous_counter = self.counter;
        }
    }

    #[inline]
    pub const fn set_enabled(&mut self, enabled: bool) {
        if !enabled {
            self.counter = 0;
        }
        self.enabled = enabled;
    }

    #[inline]
    pub const fn reload(&mut self) {
        if self.reload > 0 {
            if self.counter == self.previous_counter {
                self.counter = self.reload;
            }
            self.reload = 0;
        }
        self.halt = self.new_halt;
    }

    /// The halt flag (`$4000`/`$4004`/`$400C` bit 5, `$4008` bit 7), applied at the next reload.
    #[inline]
    pub const fn write_ctrl(&mut self, halt: bool) {
        self.new_halt = halt;
    }

    pub const fn clock(&mut self) {
        if self.counter > 0 && !self.halt {
            self.counter -= 1;
        }
    }

    /// Reset: always disables; `clear` also zeroes the counter and its pending writes (every
    /// channel on power-on, all but the triangle on a soft reset).
    pub const fn reset(&mut self, clear: bool) {
        self.enabled = false;
        if clear {
            self.halt = false;
            self.new_halt = false;
            self.counter = 0;
            self.reload = 0;
            self.previous_counter = 0;
        }
    }
}
