//! Tests for the FCEUmm quirks zcore still reproduces (plan step 18a restores them to hardware
//! one at a time). Each asserts the FCEUmm behaviour; the hardware behaviour each replaces is in
//! `FCEUMM_QUIRKS.md`.

use crate::{bus::Bus, cart::Cart, common::ResetKind, mapper::Nrom};

/// An NROM console spinning on `JMP $0000` in work RAM.
fn console() -> Bus {
    let mut bus = Bus::default();
    let mut cart = Cart::empty();
    cart.mapper = Nrom::load(&mut cart).unwrap();
    bus.load_cart(cart);
    bus.reset(ResetKind::Hard);
    for (addr, val) in [(0x0000u16, 0x4Cu8), (0x0001, 0x00), (0x0002, 0x00)] {
        bus.cpu_bus_write(addr, val);
    }
    bus.cpu.pc = 0x0000;
    bus
}

/// Clock the PPU alone until the frame counter changes; returns the dots it took.
fn ppu_dots_to_next_frame(bus: &mut Bus) -> u32 {
    let start = bus.ppu.frame_number();
    let mut dots = 0;
    while bus.ppu.frame_number() == start {
        bus.ppu_clock();
        dots += 1;
    }
    dots
}

#[test]
fn fq06_reset_spends_no_cycles() {
    assert_eq!(console().cpu.cycle, 0);
}

#[test]
fn fq07_ppu_starts_on_the_post_render_line() {
    let bus = console();
    assert_eq!((bus.ppu.scanline, bus.ppu.cycle), (240, 2));
}

#[test]
fn fq08_frame_ends_one_dot_into_line_240() {
    let mut bus = console();
    ppu_dots_to_next_frame(&mut bus);
    assert_eq!((bus.ppu.scanline, bus.ppu.cycle), (240, 1));
}

#[test]
fn fq01_no_vblank_in_the_two_frames_after_reset() {
    let mut bus = console();
    let mut seen = [false; 3];
    for seen in &mut seen {
        let start = bus.ppu.frame_number();
        while bus.ppu.frame_number() == start {
            bus.ppu_clock();
            *seen |= bus.ppu.status_in_vblank;
        }
    }
    assert_eq!(seen, [false, false, true]);
}

#[test]
fn fq05_nmi_requested_12_dots_after_vblank() {
    let mut bus = console();
    for _ in 0..2 {
        ppu_dots_to_next_frame(&mut bus);
    }
    bus.write_ctrl(0x80);
    while bus.ppu.fceu_nmi == 0 {
        bus.ppu_clock();
    }
    assert_eq!((bus.ppu.scanline, bus.ppu.cycle), (241, 13));
    assert!(!bus.ppu.nmi_pending, "the hardware NMI line is not used");

    // Taken at the end of the instruction during which it was requested.
    let mut bus = console();
    for _ in 0..2 {
        let start = bus.ppu.frame_number();
        while bus.ppu.frame_number() == start {
            bus.clock_instr();
        }
    }
    bus.write_ctrl(0x80);
    let sp = bus.cpu.sp;
    let mut steps = 0;
    while bus.cpu.sp == sp {
        let requested_before = bus.ppu.fceu_nmi != 0;
        bus.clock_instr();
        steps += 1;
        assert!(steps < 50_000, "NMI never taken");
        if bus.cpu.sp != sp {
            assert!(!requested_before, "taken in the same step it was requested");
        }
    }
    assert_eq!(bus.cpu.sp, sp.wrapping_sub(3));
}

#[test]
fn fq03_status_read_sees_flags_as_of_the_instruction_start() {
    let mut bus = console();
    // Raised during the reading instruction: the read misses it and leaves it set.
    bus.ppu.status_in_vblank = true;
    bus.ppu.status_at_instr = 0;
    assert_eq!(bus.ppu.read_status() & 0x80, 0);
    assert!(bus.ppu.status_in_vblank);
    // Up when the instruction started: seen and cleared.
    bus.ppu.status_at_instr = 0x80;
    assert_eq!(bus.ppu.read_status() & 0x80, 0x80);
    assert!(!bus.ppu.status_in_vblank);
    // Sprite 0 hit cleared at the end of VBlank during the reading instruction: still seen.
    bus.ppu.status_spr_zero_hit = false;
    bus.ppu.status_at_instr = 0x40;
    assert_eq!(bus.ppu.read_status() & 0x40, 0x40);
    // Raised during the reading instruction: seen at once.
    bus.ppu.status_spr_zero_hit = true;
    bus.ppu.status_at_instr = 0;
    assert_eq!(bus.ppu.read_status() & 0x40, 0x40);
}

#[test]
fn fq02_fq04_status_read_neither_suppresses_vblank_nor_cancels_nmi() {
    let mut bus = console();
    bus.ppu.scanline = bus.ppu.vblank_scanline;
    bus.ppu.cycle = 0;
    bus.ppu.fceu_nmi = 1;
    bus.ppu.read_status();
    assert!(!bus.ppu.prevent_vbl, "FQ-02: no suppression race");
    assert_eq!(bus.ppu.fceu_nmi, 1, "FQ-04: a read never cancels an NMI");
}

#[test]
fn fq04_enabling_nmi_during_vblank_delays_one_instruction() {
    let mut bus = console();
    bus.ppu.status_at_instr = 0x80;
    bus.write_ctrl(0x80);
    assert_eq!(bus.ppu.fceu_nmi, 2);
    let sp = bus.cpu.sp;
    bus.clock_instr();
    assert_eq!(bus.cpu.sp, sp, "first instruction after the write runs");
    bus.clock_instr();
    assert_eq!(bus.cpu.sp, sp.wrapping_sub(3), "then the NMI is taken");

    // Disabling never cancels a requested NMI.
    let mut bus = console();
    bus.ppu.fceu_nmi = 1;
    bus.write_ctrl(0x00);
    assert_eq!(bus.ppu.fceu_nmi, 1);
}
