//! The two standard controller ports.
//!
//! <https://wiki.nesdev.org/w/index.php/Standard_controller>
//!
//! A controller's buttons are the raw NES button byte (A=0x01, B=0x02, Select=0x04, Start=0x08,
//! Up=0x10, Down=0x20, Left=0x40, Right=0x80), set by the caller as it is; opposing D-Pad
//! directions are not filtered.

use crate::bus::Bus;
#[cfg(test)]
use crate::common::ResetKind;
use serde::{Deserialize, Serialize};

/// A controller port.
#[derive(Default, Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[must_use]
pub enum Player {
    /// Port 1 ($4016).
    #[default]
    One,
    /// Port 2 ($4017).
    Two,
}

/// A single controller button.
#[cfg(test)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum JoypadBtn {
    /// Left D-Pad.
    Left,
    /// Right D-Pad.
    Right,
    /// Up D-Pad.
    Up,
    /// Down D-Pad.
    Down,
    /// A Button.
    A,
    /// B Button.
    B,
    /// Select Button.
    Select,
    /// Start Button.
    Start,
}

#[cfg(test)]
impl JoypadBtn {
    /// The button's bit in the NES button byte.
    #[must_use]
    pub const fn bit(self) -> u8 {
        match self {
            Self::A => 0x01,
            Self::B => 0x02,
            Self::Select => 0x04,
            Self::Start => 0x08,
            Self::Up => 0x10,
            Self::Down => 0x20,
            Self::Left => 0x40,
            Self::Right => 0x80,
        }
    }
}

/// The console's input hardware: the two controller ports.
#[derive(Default, Debug, Copy, Clone, Serialize, Deserialize)]
#[must_use]
pub struct Input {
    /// The two controller ports.
    pub joypads: [Joypad; 2],
}

impl Input {
    /// Creates input state with nothing pressed.
    pub const fn new() -> Self {
        Self {
            joypads: [Joypad::new(); 2],
        }
    }

    /// Mutably borrows one port's controller, which is how buttons are set.
    pub const fn joypad_mut(&mut self, player: Player) -> &mut Joypad {
        &mut self.joypads[player as usize]
    }

    /// Reads one controller port, shifting its register on.
    pub const fn read_port(&mut self, player: Player) -> u8 {
        self.joypads[player as usize].read()
    }

    /// Reads one controller port without shifting.
    #[must_use]
    pub const fn peek_port(&self, player: Player) -> u8 {
        self.joypads[player as usize].peek()
    }

    /// Writes $4016, whose bit 0 is the strobe that reloads every controller register.
    pub const fn write(&mut self, val: u8) {
        self.joypads[0].write(val);
        self.joypads[1].write(val);
    }

    /// Resets input state.
    #[cfg(test)]
    pub const fn reset(&mut self, kind: ResetKind) {
        self.joypads[0].reset(kind);
        self.joypads[1].reset(kind);
    }
}

/// The console's view of the input ports.
impl Bus {
    /// Reads $4016/$4017 for one port, shifting the controller's register on.
    pub(crate) const fn input_read(&mut self, player: Player) -> u8 {
        self.state.input.read_port(player) | 0x40
    }

    /// Reads $4016/$4017 without shifting.
    #[must_use]
    pub(crate) const fn input_peek(&self, player: Player) -> u8 {
        self.state.input.peek_port(player) | 0x40
    }
}

/// A standard NES controller: eight buttons behind a shift register.
#[derive(Default, Debug, Copy, Clone, Serialize, Deserialize)]
#[must_use]
pub struct Joypad {
    /// Which buttons are currently held, as the NES button byte.
    pub buttons: u8,
    /// How far the shift register has been read.
    pub index: u8,
    /// Whether $4016 bit 0 is holding the register in reload.
    pub strobe: bool,
}

impl Joypad {
    /// Creates a controller with nothing pressed.
    pub const fn new() -> Self {
        Self {
            buttons: 0,
            index: 0,
            strobe: false,
        }
    }

    /// Whether a button is currently held.
    #[cfg(test)]
    #[must_use]
    pub const fn button(&self, button: JoypadBtn) -> bool {
        self.buttons & button.bit() != 0
    }

    /// Presses a button.
    #[cfg(test)]
    pub const fn press(&mut self, button: JoypadBtn) {
        self.buttons |= button.bit();
    }

    /// Shifts the next button out of the register.
    #[must_use]
    pub const fn read(&mut self) -> u8 {
        let val = self.peek();
        if !self.strobe && self.index < 8 {
            self.index += 1;
        }
        val
    }

    /// Returns the next button without shifting.
    #[must_use]
    pub const fn peek(&self) -> u8 {
        if self.index < 8 {
            (self.buttons & (1 << self.index)) >> self.index
        } else {
            0x01
        }
    }

    /// Writes the strobe line; while it is high the register reloads continuously.
    pub const fn write(&mut self, val: u8) {
        let prev_strobe = self.strobe;
        self.strobe = val & 0x01 == 0x01;
        if prev_strobe && !self.strobe {
            self.index = 0;
        }
    }

    /// Resets the shift register and releases every button.
    #[cfg(test)]
    pub const fn reset(&mut self, _kind: ResetKind) {
        self.buttons = 0;
        self.index = 0;
        self.strobe = false;
    }
}
