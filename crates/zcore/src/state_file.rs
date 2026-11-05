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
//! Every layout change of [`MachineState`] bumps the version. Plan step 18a restores hardware
//! behaviour one FCEUmm quirk at a time while the start states are still the v1 files of step 17,
//! so v1 bodies are decoded into the current state (module `v1`); other old versions are refused.
//! Step 18b regenerates the states and deletes the v1 path.

use crate::machine::{MachineState, Rom};

/// File magic.
pub const MAGIC: &[u8; 8] = b"TFZSTATE";
/// Current format version (2: the PPU I/O latch decay, FQ-13 restored).
pub const VERSION: u16 = 2;
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
    if version != VERSION && version != 1 {
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
    let bad = |e: postcard::Error| StateFormatError(format!("bad state body: {e}"));
    let machine = if version == 1 {
        postcard::from_bytes::<v1::MachineState>(&body)
            .map_err(bad)?
            .into_current()
    } else {
        postcard::from_bytes(&body).map_err(bad)?
    };
    Ok(StateFile { machine, flags })
}

/// `TFZSTATE` v1 bodies: the layout of the FCEUmm-faithful build that wrote the step-17 states,
/// as frozen copies of every type that has changed since, each converted into the current type.
mod v1 {
    use crate::{
        apu::Apu,
        bus::{self, size},
        cpu::Cpu,
        input::Input,
        mapper::Mapper,
        memory::{ConstArray, Memory},
        ppu::{self, PaletteRam, mask, scroll::Scroll, sprite::Sprite},
    };
    use serde::Deserialize;

    /// A v1 struct: `kept` fields carry over to the current type unchanged, `dropped` ones (after
    /// them in v1's order) no longer exist.
    macro_rules! v1_struct {
        ($name:ident => $cur:ty { $($f:ident: $t:ty,)* } dropped { $($d:ident: $dt:ty,)* }) => {
            #[derive(Deserialize)]
            #[allow(dead_code)]
            pub(super) struct $name {
                $($f: $t,)*
                $($d: $dt,)*
            }

            impl $name {
                /// Copies the kept fields over `cur`.
                fn apply(self, cur: &mut $cur) {
                    $(cur.$f = self.$f;)*
                }
            }
        };
    }

    v1_struct!(Ppu => ppu::Ppu {
        master_clock: u32,
        cycle: u16,
        scanline: u16,
        mask_min_draw_bg_cycle: u16,
        mask_min_draw_spr_cycle: u16,
        mask_rendering_enabled: bool,
        mask_prev_rendering_enabled: bool,
        mask_pending_rendering_update: bool,
        ctrl_bg_select: u16,
        ctrl_spr_select: u16,
        ctrl_spr_height: u16,
        status_spr_zero_hit: bool,
        status_spr_overflow: bool,
        status_in_vblank: bool,
        ctrl_vram_increment: bool,
        ctrl_master_slave: u8,
        ctrl_nmi_enabled: bool,
        scroll: Scroll,
        vblank_scanline: u16,
        prerender_scanline: u16,
        tile_shift_lo: u16,
        tile_shift_hi: u16,
        tile_addr: u16,
        tile_lo: u8,
        tile_hi: u8,
        clock_divider: u8,
        open_bus: u8,
        curr_palette: u8,
        prev_palette: u8,
        next_palette: u8,
        is_visible_scanline: bool,
        is_prerender_scanline: bool,
        is_render_scanline: bool,
        mask_emphasis: u16,
        mask_grayscale: u8,
        mask_bits: mask::Bits,
        spr_in_range: bool,
        spr_zero_in_range: bool,
        secondary_oamaddr: u8,
        oam_eval_done: bool,
        oamaddr_lo: u8,
        oamaddr_hi: u8,
        oam_fetch: u8,
        oamaddr: u8,
        spr_zero_visible: bool,
        spr_count: u8,
        overflow_count: u8,
        frame_count: u32,
        color_bits_applied: usize,
        palette: PaletteRam,
        secondary_oamdata: ConstArray<u8, 32>,
        sprites: [Sprite; 8],
        spr_cover: ConstArray<u8, 256>,
        oamdata: ConstArray<u8, 256>,
        nmi_pending: bool,
        vram_buffer: u8,
        prevent_vbl: bool,
        dead_frames: u8,
        status_at_instr: u8,
        fceu_nmi: u8,
    } dropped {});

    impl Ppu {
        fn into_current(self) -> ppu::Ppu {
            let mut ppu = ppu::Ppu::new();
            self.apply(&mut ppu);
            // FCEUmm's latch never decayed: every bit counts as refreshed at the save.
            ppu.open_bus_refreshed = [ppu.frame_count; 8];
            ppu
        }
    }

    /// [`bus::State`] v1.
    #[derive(Deserialize)]
    pub(super) struct State {
        cpu: Cpu,
        ppu: Ppu,
        mapper: Mapper,
        memory: Memory,
        apu: Apu,
        input: Input,
        wram: ConstArray<u8, { size::WRAM }>,
        open_bus: u8,
    }

    /// [`crate::MachineState`] v1.
    #[derive(Deserialize)]
    pub(super) struct MachineState {
        state: State,
        frame: u64,
        cycles: u64,
    }

    impl MachineState {
        pub(super) fn into_current(self) -> crate::MachineState {
            let s = self.state;
            let state = bus::State {
                cpu: s.cpu,
                ppu: s.ppu.into_current(),
                mapper: s.mapper,
                memory: s.memory,
                apu: s.apu,
                input: s.input,
                wram: s.wram,
                open_bus: s.open_bus,
            };
            crate::MachineState::from_parts(state, self.frame, self.cycles)
        }
    }
}
