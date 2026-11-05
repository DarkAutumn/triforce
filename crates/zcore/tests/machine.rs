//! The machine API on the real ROM: render toggle, snapshot/restore and the state file.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use zcore::{Rom, ZCore, state_file};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn rom() -> Option<Arc<Rom>> {
    let path = repo().join("roms/zelda.nes");
    match std::fs::read(&path) {
        Ok(bytes) => Some(Rom::from_ines(&bytes).expect("roms/zelda.nes is Zelda")),
        Err(_) => {
            eprintln!("{} missing: skipping", path.display());
            None
        }
    }
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
/// CIRAM), 2 KiB of work RAM, 256 bytes of OAM and the PPU's 256-byte sprite cover map, plus a
/// few hundred bytes of registers and page tables. 24 KiB leaves room for that and nothing else:
/// a ROM image (128 KiB) or a frame buffer (120 KiB) creeping back in fails here.
#[test]
fn machine_state_is_small_and_copy() {
    fn copy<T: Copy>() {}
    copy::<zcore::MachineState>();
    let size = std::mem::size_of::<zcore::MachineState>();
    eprintln!("size_of::<MachineState>() = {size}");
    assert!(size <= 24 * 1024, "MachineState is {size} bytes");
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
    let Some(rom) = rom() else { return };
    assert_eq!(rom.payload_sha1(), zcore::machine::ZELDA_PAYLOAD_SHA1);
    let bytes = std::fs::read(repo().join("roms/zelda.nes")).unwrap();
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
#[should_panic(expected = "outside")]
fn poke_outside_ram_panics() {
    let Some(rom) = rom() else {
        panic!("outside (no ROM)")
    };
    ZCore::power_on(rom).poke(0x2000, 0);
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

/// Plan step 11, RGB output: frame 600 of `title_idle_3600` matches the FCEUmm oracle's palette
/// index frame (its committed golden digest, which like `tf diff` blanks the rows
/// `corpus/pixel-allowlist.toml` lists for every movie, 63 and 64), and its RGB conversion shows
/// the picture reaching the top or bottom row (no overscan crop): some pixel of row 0 or row 239
/// is not the backdrop colour, and every pixel is the FCEUmm palette colour of its index.
#[test]
#[ignore = "step 18: FCEUmm-recorded movie desyncs on hardware timing (FQ-01, FQ-06..FQ-08)"]
fn title_rgb_frame_is_full_and_matches_the_oracle() {
    let Some(rom) = rom() else { return };
    let movie_path = repo().join("corpus/movies/title_idle_3600.tfmovie");
    let m = movie::Movie::load(&movie_path).unwrap();
    assert!(m.base().is_none() && m.pokes.is_empty());
    let golden =
        movie::Golden::load(&repo().join("corpus/golden/title_idle_3600.tfgolden")).unwrap();
    let pixel_slot = golden
        .domains()
        .iter()
        .position(|&d| d == movie::Domain::Pixels)
        .expect("golden has pixel digests");

    let mut nes = ZCore::power_on(rom);
    for &x in &m.inputs[..600] {
        nes.run_frame(x, true);
    }
    let px = nes.pixels().expect("rendered");
    let mut blanked = *px;
    blanked[63 * 256..65 * 256].fill(0);
    assert_eq!(
        movie::digest8(&blanked),
        golden.records[600][pixel_slot],
        "frame 600 differs from the oracle's index frame"
    );

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
    assert_eq!(state_file::VERSION, 5);
    assert_eq!(
        blake3::hash(&bytes).to_hex().as_str(),
        "7fdbf9210403227ca21e3cfe211c86af0b1e8b6ecb7ab1b698bc822b1a97f1bd"
    );
}

/// Plan step 18a: the step-17 start states are `TFZSTATE` v1 files, decoded into the current
/// layout. Each loads, keeps its RAM, and plays on.
#[test]
fn v1_start_states_decode_and_run() {
    let Some(rom) = rom() else { return };
    let dir = repo().join("triforce/states");
    let mut names: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "state"))
        .collect();
    names.sort();
    assert!(!names.is_empty());
    for path in names.iter().step_by(20) {
        let bytes = std::fs::read(path).unwrap();
        if u16::from_le_bytes([bytes[8], bytes[9]]) != 1 {
            continue;
        }
        let f = state_file::decode(&rom, &bytes).unwrap();
        let mut nes = ZCore::power_on(rom.clone());
        nes.restore(&f.machine);
        let mode = nes.ram()[0x12];
        assert!(mode != 0xFF && nes.frame() > 0, "{}", path.display());
        for _ in 0..120 {
            nes.run_frame(0, false);
        }
    }
}
