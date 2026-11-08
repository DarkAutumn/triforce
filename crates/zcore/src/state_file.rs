//! `TFZSTATE`, the on-disk machine state (start states, curriculum tracks, saved games):
//!
//! | Field | Encoding |
//! |---|---|
//! | magic | `b"TFZSTATE"` |
//! | version | `u16` LE, [`VERSION`] |
//! | ROM payload SHA-1 | 20 bytes |
//! | executor flags | `u8`: bit 0 `was_in_cave`, bit 1 `just_exited_cave` |
//! | zstd length | `u32` LE |
//! | body | zstd (level 3) of the `postcard` encoding of [`MachineState`] |
//!
//! Every layout change of [`MachineState`] bumps the version; older versions are refused (regenerate
//! states with `python -m triforce.entry_states`).

use crate::machine::{MachineState, Rom};

/// File magic.
pub const MAGIC: &[u8; 8] = b"TFZSTATE";
/// Current format version (2: the PPU I/O latch decay, FQ-13; 3: the hardware DMC reader,
/// FQ-11/FQ-12; 4: no instruction-start PPUSTATUS snapshot or FCEUmm NMI request, FQ-03..FQ-05;
/// 5: no dead frames after reset, FQ-01; 6: power-on RAM $00, so every earlier state is stale).
pub const VERSION: u16 = 6;
const HEADER: usize = 8 + 2 + 20 + 1 + 4;

/// A state file that can't be loaded for this ROM and build.
#[derive(Debug)]
pub struct StateFormatError(pub String);

impl std::fmt::Display for StateFormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for StateFormatError {}

/// A decoded state file.
#[derive(Clone)]
pub struct StateFile {
    /// The machine.
    pub machine: MachineState,
    /// Executor flags (bit 0 `was_in_cave`, bit 1 `just_exited_cave`).
    pub flags: u8,
}

/// Encode `machine` for `rom`.
pub fn encode(rom: &Rom, machine: &MachineState, flags: u8) -> Vec<u8> {
    let body = postcard::to_allocvec(machine).expect("MachineState serializes");
    let z = zstd::bulk::compress(&body, 3).expect("zstd compress");
    let mut out = Vec::with_capacity(HEADER + z.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&rom.payload_sha1());
    out.push(flags);
    out.extend_from_slice(
        &u32::try_from(z.len())
            .expect("state under 4 GiB")
            .to_le_bytes(),
    );
    out.extend_from_slice(&z);
    out
}

/// Decode a state file written for `rom`.
pub fn decode(rom: &Rom, bytes: &[u8]) -> Result<StateFile, StateFormatError> {
    let err = |m: &str| Err(StateFormatError(m.to_owned()));
    if bytes.starts_with(&[0x1f, 0x8b]) {
        return err(
            "legacy stable-retro state; regenerate it with python -m triforce.entry_states",
        );
    }
    if bytes.len() < HEADER || &bytes[..8] != MAGIC {
        return err("not a TFZSTATE file");
    }
    let version = u16::from_le_bytes([bytes[8], bytes[9]]);
    if version != VERSION {
        return Err(StateFormatError(format!(
            "TFZSTATE version {version}, expected {VERSION}"
        )));
    }
    if bytes[10..30] != rom.payload_sha1() {
        return err("state was saved with a different ROM");
    }
    let flags = bytes[30];
    let len = u32::from_le_bytes(bytes[31..35].try_into().expect("4 bytes")) as usize;
    let Some(z) = bytes.get(HEADER..HEADER + len) else {
        return err("truncated TFZSTATE file");
    };
    let body =
        zstd::stream::decode_all(z).map_err(|e| StateFormatError(format!("bad zstd body: {e}")))?;
    let machine = postcard::from_bytes(&body)
        .map_err(|e| StateFormatError(format!("bad state body: {e}")))?;
    Ok(StateFile { machine, flags })
}
