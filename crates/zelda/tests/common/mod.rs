//! Shared test helpers: zcore over the ROM (and a frame-counting wrapper) and the named-state loader
//! (`TFZSTATE` files in `triforce/states`).
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock};
use zcore::{MachineState, Rom, ZCore};
use zelda::Nes;

/// zcore with `.1` counting every frame run (look-ahead frames included; restores don't rewind it).
pub struct Counted(pub ZCore, pub u64);

impl Nes for Counted {
    type State = MachineState;

    fn run_frame(&mut self, buttons: u8, render: bool) {
        self.0.run_frame(buttons, render);
        self.1 += 1;
    }

    fn ram(&self) -> &[u8; 0x800] {
        self.0.ram()
    }

    fn prg_ram(&self) -> &[u8; 0x2000] {
        self.0.prg_ram()
    }

    fn poke(&mut self, addr: u16, value: u8) {
        self.0.poke(addr, value);
    }

    fn snapshot(&self) -> MachineState {
        self.0.snapshot()
    }

    fn restore(&mut self, state: &MachineState) {
        self.0.restore(state);
    }
}

pub fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The ROM (`$TRIFORCE_ROM`, else `roms/zelda.nes`). Missing: panics (`test_rom`), or None with
/// `TRIFORCE_ALLOW_NO_ROM=1`.
pub fn rom() -> Option<Arc<Rom>> {
    test_rom::path()?; // before the LazyLock, so every test fails with the ROM message, not a poisoned lock
    static ROM: LazyLock<Option<Arc<Rom>>> =
        LazyLock::new(|| test_rom::bytes().map(|bytes| Rom::from_ines(&bytes).expect("valid Zelda ROM")));
    ROM.clone()
}

/// A powered-on zcore, or None when the ROM is missing.
pub fn open_core() -> Option<ZCore> {
    rom().map(ZCore::power_on)
}

/// THE state loader of the zelda tests: every test that starts from a named state decodes
/// `triforce/states/<name>.state` here.
pub fn load_named_state(name: &str) -> MachineState {
    let path = repo().join(format!("triforce/states/{name}.state"));
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    zcore::state_file::decode(&rom().expect("ROM"), &bytes)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .machine
}
