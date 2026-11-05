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
    v[8] = 2;
    assert!(msg(&v).contains("version 2"));
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
