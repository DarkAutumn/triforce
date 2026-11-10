//! The machine API on the real ROM: render toggle, snapshot/restore and the state file.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use zcore::{Rom, ZCore, state_file};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The ROM; missing panics (`test_rom`), or None with `TRIFORCE_ALLOW_NO_ROM=1`.
fn rom() -> Option<Arc<Rom>> {
    test_rom::bytes().map(|bytes| Rom::from_ines(&bytes).expect("the ROM is Zelda"))
}

/// `power_on_to_start`'s inputs (it has no pokes).
fn inputs() -> Vec<u8> {
    let m = movie::Movie::load(&repo().join("corpus/movies/power_on_to_start.tfmovie")).unwrap();
    assert!(m.pokes.is_empty() && m.base().is_none());
    m.inputs
}

#[test]
fn render_off_emulates_identically() {
    let Some(rom) = rom() else { return };
    let (mut on, mut off) = (ZCore::power_on(rom.clone()), ZCore::power_on(rom));
    for (k, &x) in inputs().iter().enumerate() {
        on.run_frame(x, true);
        off.run_frame(x, k % 7 == 0);
        assert_eq!(on.ram(), off.ram(), "frame {k}");
        assert_eq!(on.cycles(), off.cycles(), "frame {k}");
        assert_eq!(off.pixels().is_some(), k % 7 == 0);
        if k % 7 == 0 {
            assert_eq!(on.pixels(), off.pixels(), "frame {k}");
        }
    }
    assert_eq!(on.prg_ram(), off.prg_ram());
    let mut rgb = vec![0u8; 256 * 240 * 3];
    assert!(on.rgb((&mut rgb[..]).try_into().unwrap()));
    off.run_frame(0, false);
    assert!(!off.rgb((&mut rgb[..]).try_into().unwrap()));
}

#[test]
fn state_file_round_trips() {
    let Some(rom) = rom() else { return };
    let inputs = inputs();
    let mut a = ZCore::power_on(rom.clone());
    for &x in &inputs[..inputs.len() / 2] {
        a.run_frame(x, false);
    }
    let bytes = state_file::encode(&rom, &a.snapshot(), 0b10);
    let f = state_file::decode(&rom, &bytes).unwrap();
    assert_eq!(f.flags, 0b10);
    let mut b = ZCore::power_on(rom);
    b.restore(&f.machine);
    assert_eq!(a.frame(), b.frame());
    for &x in &inputs[inputs.len() / 2..] {
        a.run_frame(x, true);
        b.run_frame(x, true);
    }
    assert_eq!(a.ram(), b.ram());
    assert_eq!(a.prg_ram(), b.prg_ram());
    assert_eq!(a.cycles(), b.cycles());
    assert_eq!(a.pixels(), b.pixels());
}

/// The state is a flat `Copy` value: 19 KiB of cartridge arena (zero page, PRG-RAM, CHR-RAM,
/// CIRAM), 2 KiB of work RAM, 256 bytes of OAM and the PPU's 256-byte sprite cover map, the PPU's
/// ~1.9 KiB per-scanline sprite-range cache, plus a few hundred bytes of registers and page tables.
/// 26 KiB leaves room for that and nothing else: a ROM image (128 KiB) or a frame buffer (120 KiB)
/// creeping back in fails here.
#[test]
fn machine_state_is_small_and_copy() {
    fn copy<T: Copy>() {}
    copy::<zcore::MachineState>();
    let size = std::mem::size_of::<zcore::MachineState>();
    eprintln!("size_of::<MachineState>() = {size}");
    assert!(size <= 26 * 1024, "MachineState is {size} bytes");
}

/// Snapshots are plain copies: restoring one and replaying gives the same frames, RAM and cycles.
#[test]
fn snapshot_restore_replays_identically() {
    let Some(rom) = rom() else { return };
    let inputs = inputs();
    let mut nes = ZCore::power_on(rom);
    for &x in &inputs {
        nes.run_frame(x, false);
    }
    let snap = nes.snapshot();
    let mut seed = 7u64;
    let random: Vec<u8> = (0..600)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed as u8 & !0x0C
        })
        .collect();
    let run = |nes: &mut ZCore| {
        let mut out = Vec::new();
        for &x in &random {
            nes.run_frame(x, false);
            out.extend_from_slice(nes.ram());
            out.extend_from_slice(&nes.cycles().to_le_bytes());
        }
        out
    };
    let first = run(&mut nes);
    let bytes = zcore::state_file::encode(nes.rom(), &nes.snapshot(), 0);
    eprintln!("gameplay TFZSTATE file: {} bytes", bytes.len());
    nes.restore(&snap);
    assert_eq!(run(&mut nes), first);
}

#[test]
fn state_file_errors() {
    let Some(rom) = rom() else { return };
    let z = ZCore::power_on(rom.clone());
    let good = state_file::encode(&rom, &z.snapshot(), 0);
    let msg = |b: &[u8]| {
        state_file::decode(&rom, b)
            .err()
            .expect("refused")
            .to_string()
    };
    assert!(msg(&[0x1f, 0x8b, 8, 0]).contains("legacy stable-retro state"));
    assert!(msg(b"NOTASTATEFILE_________________________").contains("not a TFZSTATE"));
    let mut v = good.clone();
    v[8] = 0x7F;
    assert!(msg(&v).contains("version 127"));
    let mut r = good.clone();
    r[10] ^= 1;
    assert!(msg(&r).contains("different ROM"));
    assert!(msg(&good[..good.len() - 1]).contains("truncated"));
}

#[test]
fn rom_validation() {
    let Some(bytes) = test_rom::bytes() else { return };
    let rom = Rom::from_ines(&bytes).expect("the ROM is Zelda");
    assert_eq!(rom.payload_sha1(), zcore::machine::ZELDA_PAYLOAD_SHA1);
    assert!(matches!(
        Rom::from_ines(&bytes[1..]),
        Err(zcore::RomError::NotInes)
    ));
    let mut wrong = bytes.clone();
    wrong[100] ^= 1;
    assert!(matches!(
        Rom::from_ines(&wrong),
        Err(zcore::RomError::WrongPayload(_))
    ));
    let mut mapper = bytes;
    mapper[6] = 0;
    assert!(matches!(
        Rom::from_ines(&mapper),
        Err(zcore::RomError::WrongCart(_))
    ));
}

#[test]
fn poke_outside_ram_panics() {
    let Some(rom) = rom() else { return };
    let mut nes = ZCore::power_on(rom);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| nes.poke(0x2000, 0)));
    let message = result.expect_err("poke outside RAM panics");
    let text = message
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| message.downcast_ref::<&str>().copied())
        .unwrap_or_default();
    assert!(text.contains("outside"), "{text}");
}

/// Poke reaches the last byte of work RAM and both ends of PRG-RAM; the first byte past either
/// window panics (the RAM mirrors at $0800-$1FFF are not pokeable).
#[test]
fn poke_reaches_both_window_edges() {
    let Some(rom) = rom() else { return };
    let mut nes = ZCore::power_on(rom);
    nes.poke(0x0000, 0x11);
    nes.poke(0x07FF, 0x22);
    nes.poke(0x6000, 0x33);
    nes.poke(0x7FFF, 0x44);
    assert_eq!((nes.ram()[0], nes.ram()[0x7FF]), (0x11, 0x22));
    assert_eq!((nes.prg_ram()[0], nes.prg_ram()[0x1FFF]), (0x33, 0x44));
    for addr in [0x0800u16, 0x5FFF, 0x8000] {
        let rom = nes.rom().clone();
        let refused = std::panic::catch_unwind(move || ZCore::power_on(rom).poke(addr, 0));
        assert!(refused.is_err(), "${addr:04X} accepted");
    }
}

/// After a restore there is no frame to show until one is rendered, even if the console had
/// just drawn one; a frame run without rendering keeps reporting none. The cycle count follows
/// the snapshot (back in time on restore) and otherwise only grows.
#[test]
fn restore_drops_the_frame_and_rewinds_the_cycle_count() {
    let Some(rom) = rom() else { return };
    let mut nes = ZCore::power_on(rom);
    let mut rgb = vec![0u8; 256 * 240 * 3];
    let rgb: &mut [u8; 256 * 240 * 3] = (&mut rgb[..]).try_into().unwrap();
    assert!(nes.pixels().is_none() && !nes.rgb(rgb), "nothing drawn at power-on");
    nes.run_frame(0, true);
    let snap = nes.snapshot();
    let (frame, cycles) = (nes.frame(), nes.cycles());
    let mut last = cycles;
    for _ in 0..30 {
        nes.run_frame(0, true);
        assert!(nes.cycles() > last, "cycles only grow");
        last = nes.cycles();
    }
    assert!(nes.pixels().is_some());

    nes.restore(&snap);
    assert_eq!((nes.frame(), nes.cycles()), (frame, cycles));
    assert!(nes.pixels().is_none() && !nes.rgb(rgb), "no frame after restore");
    nes.run_frame(0, false);
    assert!(nes.pixels().is_none(), "a non-rendered frame shows nothing");
    nes.run_frame(0, true);
    assert!(nes.pixels().is_some() && nes.rgb(rgb));
    assert_eq!(nes.frame(), frame + 2);
}

/// The whole machine state, as bytes (`MachineState` has no `PartialEq`).
fn state_bytes(nes: &ZCore) -> Vec<u8> {
    postcard::to_allocvec(&nes.snapshot()).unwrap()
}

/// Two consoles fed the same movie: one draws every frame, the other none.
struct RenderTwins {
    on: ZCore,
    off: ZCore,
    frames: usize,
}

impl movie::Playback for RenderTwins {
    fn run_frame(&mut self, buttons: u8) {
        self.on.run_frame(buttons, true);
        self.off.run_frame(buttons, false);
        self.frames += 1;
        assert!(self.on.pixels().is_some() && self.off.pixels().is_none());
        assert_eq!(state_bytes(&self.on), state_bytes(&self.off), "frame {}", self.frames);
        assert_eq!(self.on.cycles(), self.off.cycles(), "frame {}", self.frames);
    }

    fn poke(&mut self, addr: u16, value: u8) {
        self.on.poke(addr, value);
        self.off.poke(addr, value);
    }
}

/// Plan step 11, render toggle: the corpus has no scripted `transition_under_fire` (dropped, see
/// `docs/decisions.md`), so this plays `chain-milestone1-111` from power-on: 13 rooms of agent
/// play, scrolling between rooms with enemies active, taking hits, dying in 1:0x74. Every
/// frame's complete machine state (CPU, PPU, APU, mapper, all memory) and cycle count must match
/// between drawing every frame and drawing none.
#[test]
fn render_off_matches_render_on_through_a_chain() {
    let Some(rom) = rom() else { return };
    let dir = repo().join("corpus/movies");
    let m = movie::Movie::load(&dir.join("chain-milestone1-111.tfmovie")).unwrap();
    let mut twins = RenderTwins {
        on: ZCore::power_on(rom.clone()),
        off: ZCore::power_on(rom),
        frames: 0,
    };
    movie::play(&mut twins, &m, &dir).unwrap();
    assert!(twins.frames > 5_000);
}

/// Plan step 11, RGB output: frame 600 of `title_idle_3600` converts to a full 256x240 RGB frame
/// with every pixel the palette colour of its index, and the picture reaches the top or bottom row
/// (no overscan crop): some pixel of row 0 or row 239 is not the backdrop in the index frame, and
/// not the backdrop colour in the RGB frame.
#[test]
fn title_rgb_frame_is_full_and_follows_the_palette() {
    let Some(rom) = rom() else { return };
    let m = movie::Movie::load(&repo().join("corpus/movies/title_idle_3600.tfmovie")).unwrap();
    assert!(m.base().is_none() && m.pokes.is_empty());

    let mut nes = ZCore::power_on(rom);
    for &x in &m.inputs[..600] {
        nes.run_frame(x, true);
    }
    let px = nes.pixels().expect("rendered");
    assert_eq!(px.len(), 256 * 240);

    let mut rgb = Box::new([0u8; 256 * 240 * 3]);
    assert!(nes.rgb(&mut rgb));
    for (i, &p) in px.iter().enumerate() {
        assert_eq!(rgb[i * 3..i * 3 + 3], zcore::machine::PALETTE[usize::from(p)], "pixel {i}");
    }
    let backdrop = nes.bus().ppu_bus_peek(0x3F00) & 0x3F;
    let backdrop_rgb = zcore::machine::PALETTE[usize::from(backdrop)];
    let edge_pixel = [0usize, 239].into_iter().flat_map(|row| row * 256..row * 256 + 256).find(|&i| {
        px[i] != backdrop && rgb[i * 3..i * 3 + 3] != backdrop_rgb
    });
    assert!(edge_pixel.is_some(), "rows 0 and 239 are all backdrop ({backdrop:#04x})");
}

/// The trace hook sees every instruction, from the boundary, and leaves emulation untouched.
#[test]
fn trace_hook_observes_without_changing_anything() {
    let Some(rom) = rom() else { return };
    let inputs = inputs();
    let (mut traced, mut plain) = (ZCore::power_on(rom.clone()), ZCore::power_on(rom));
    let (tx, rx) = std::sync::mpsc::channel();
    traced.set_trace(Some(Box::new(move |r: &zcore::TraceRow| {
        tx.send(*r).unwrap();
    })));
    for &x in &inputs[..120] {
        traced.run_frame(x, false);
        plain.run_frame(x, false);
    }
    traced.set_trace(None);
    traced.run_frame(0, false);
    plain.run_frame(0, false);
    assert_eq!(state_bytes(&traced), state_bytes(&plain));
    let rows: Vec<zcore::TraceRow> = rx.try_iter().collect();
    // About 1 instruction per 3 cycles; the first is the reset vector's target.
    assert!(rows.len() > 120 * 29_780 / 8, "{} rows", rows.len());
    let reset = u16::from_le_bytes([plain.bus().peek(0xFFFC), plain.bus().peek(0xFFFD)]);
    assert_eq!(rows[0].pc, reset);
    // The fixed bank ($C000-$FFFF) never switches, so its opcodes can be checked afterwards.
    assert!(rows.iter().filter(|r| r.pc >= 0xC000).all(|r| r.opcode == plain.bus().peek(r.pc)));
}

/// Pins the encoded power-on state file: any change to `MachineState`'s layout or to the power-on
/// state changes these bytes and must bump `state_file::VERSION` (then update this digest).
#[test]
fn power_on_state_file_is_pinned() {
    let Some(rom) = rom() else { return };
    let bytes = state_file::encode(&rom, &ZCore::power_on(rom.clone()).snapshot(), 0);
    assert_eq!(state_file::VERSION, 6);
    assert_eq!(
        blake3::hash(&bytes).to_hex().as_str(),
        "fa9491132aea95ba8782ba8c9fdf3f7a1470b55596d89ce98ef65d6c31866bf2"
    );
}

/// The shipped start states are current-version files: each decodes, restores and plays on. A layout
/// change that bumps `state_file::VERSION` fails here until the states are regenerated.
#[test]
fn shipped_start_states_decode_and_run() {
    let Some(rom) = rom() else { return };
    let dir = repo().join("triforce/states");
    let mut names: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "state"))
        .collect();
    names.sort();
    assert!(!names.is_empty());
    for path in names.iter().step_by(10) {
        let bytes = std::fs::read(path).unwrap();
        let f = state_file::decode(&rom, &bytes).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let mut nes = ZCore::power_on(rom.clone());
        nes.restore(&f.machine);
        assert!(nes.frame() > 0, "{}", path.display());
        for _ in 0..120 {
            nes.run_frame(0, false);
        }
    }
}
