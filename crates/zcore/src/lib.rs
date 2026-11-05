#![doc = include_str!("../README.md")]
#![warn(missing_docs)]

// The emulator behind `ZCore`. Public only so `tf` can digest internal state for the subtraction
// ladder; it tracks the implementation, not a stable interface.
#[doc(hidden)]
pub mod apu;
#[doc(hidden)]
pub mod bus;
#[doc(hidden)]
pub mod cart;
#[doc(hidden)]
#[macro_use]
pub mod common;
#[doc(hidden)]
pub mod cpu;
#[doc(hidden)]
pub mod input;
#[doc(hidden)]
pub mod mapper;
#[doc(hidden)]
pub mod memory;
#[doc(hidden)]
pub mod ppu;

// The TetaNES frame-at-a-time driver and RGBA video filters, kept for the CPU/PPU/APU test-ROM
// harness (`common::tests`) only.
#[cfg(test)]
mod control_deck;
#[cfg(test)]
mod video;

pub mod machine;
pub mod state_file;
pub use cpu::{TraceHook, TraceRow};
pub use machine::{MachineState, Rom, RomError, ZCore};
pub use state_file::StateFormatError;

#[cfg(test)]
mod fceumm_compat_tests;
