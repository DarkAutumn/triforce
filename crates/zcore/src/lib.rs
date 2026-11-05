#![doc = include_str!("../README.md")]
#![doc(
    html_favicon_url = "https://github.com/lukexor/tetanes/blob/main/assets/linux/icon.png?raw=true",
    html_logo_url = "https://github.com/lukexor/tetanes/blob/main/assets/linux/icon.png?raw=true"
)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]

pub mod apu;
pub mod bus;
pub mod cart;
#[macro_use]
pub mod common;
pub mod control_deck;
pub mod cpu;
pub mod input;
pub mod mapper;
pub mod memory;
pub mod ppu;
pub mod video;

pub mod machine;
pub mod state_file;
pub use machine::{MachineState, Rom, RomError, ZCore};
pub use state_file::StateFormatError;

pub mod prelude {
    //! The prelude re-exports all the common structs/enums used for basic NES emulation.

    pub use crate::{
        apu::{Apu, Channel},
        cart::Cart,
        common::ResetKind,
        control_deck::{Config, ControlDeck, HeadlessMode},
        cpu::Cpu,
        input::{Input, Player},
        mapper::{Map, Mapper},
        ppu::{Mirroring, Ppu},
        video::Frame,
    };
}

#[cfg(test)]
mod fceumm_compat_tests;
